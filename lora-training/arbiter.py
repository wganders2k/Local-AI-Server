"""
The trainer's whole relationship with the GPU: two calls, and a name.

    acquire()           take the card, or be refused
    release()           done with it
    wait_until_wanted() block until something outranking us wants it back

Deliberately a copy of the two calls in `proxy/arbiter.py` rather than a shared
library. The previous design vendored a client into three repos and every change
to the protocol became a three-repo change; the protocol is now small enough that
duplicating forty lines is cheaper than coupling two images' build contexts.

The name is the only thing the trainer says about itself. It does not claim a
priority, state how much VRAM it needs, or name what should be stopped for it —
the arbiter reads all of that out of jobs.yaml, keyed on the name. A job that
could assert its own importance would be grading its own homework, and being the
*lowest*-ranked tenant is no more self-assessed than being the highest.

The first two are what every GPU tenant on this box says. The third exists
because the arbiter cannot tear this job down: killing the container would take
the supervisor with it, and Docker suppresses a container's restart policy for
any API-initiated stop or kill, so the run would simply be over. So it is *told*,
and supervisor.py kills the training process itself — which is the real cgroup
teardown, just performed from inside.

Being refused is normal and is not an error. The card belongs to whatever
outranks this, most of the day; the answer is to wait and ask again, holding no
VRAM meanwhile.
"""

import logging
import os

import httpx

logger = logging.getLogger("arbiter-client")

# Long enough to cover the arbiter's own acquire timeout plus a cooperative stop
# running to its grace period. If this fires first we lose the arbiter's reason
# for refusing, which is the thing worth reading in the log.
_TIMEOUT = 240.0

# The blocking notice call returns on its own well before this; the margin is for
# a slow response, not for the wait itself.
_NOTICE_TIMEOUT = 120.0


class ArbiterClient:
    def __init__(self, base_url: str | None = None, job_name: str | None = None):
        self.base_url = (base_url or os.environ.get("ARBITER_URL", "http://arbiter:11438")).rstrip("/")
        self.job_name = job_name or os.environ.get("ARBITER_JOB_NAME", "lora-trainer")
        self._client = httpx.Client(timeout=httpx.Timeout(_TIMEOUT, connect=5.0))

    def acquire(self) -> tuple[bool, str]:
        """
        Ask for the GPU. Returns (granted, detail).

        An unreachable arbiter is a refusal, not a licence to proceed. Loading a
        model without knowing what else is on the card is the one option that can
        OOM another tenant, and this is the tenant that matters least.
        """
        try:
            resp = self._client.post(f"{self.base_url}/gpu/acquire", json={"name": self.job_name})
        except httpx.HTTPError as exc:
            return False, f"arbiter unreachable: {exc}"

        if resp.status_code == 503:
            try:
                return False, resp.json().get("reason", "refused")
            except ValueError:
                return False, "refused"
        if resp.status_code != 200:
            return False, f"arbiter returned HTTP {resp.status_code}"
        return True, resp.json().get("detail", "")

    def release(self) -> None:
        """
        Tell the arbiter we are done.

        Best-effort. A failure here only delays the next tenant, which the
        arbiter's reaper corrects once this process is gone — it can never cause
        an OOM, so it must not be allowed to fail a run that has already finished.
        """
        try:
            self._client.post(f"{self.base_url}/gpu/release", json={"name": self.job_name})
        except httpx.HTTPError as exc:
            logger.warning(f"Could not tell the arbiter we are done ({exc}) — it will catch up")

    def wait_until_wanted(self) -> bool:
        """
        Block until something outranking us wants the card. True if it does.

        False means the call came back with nothing wanted — the arbiter's own
        timeout, so the connection is renewed periodically, or a transport error.
        Either way, ask again: a dead arbiter and a quiet one look the same from
        here, and reading False as "we may keep the card" is how a preemption
        goes unheard until the arbiter's timeout expires.
        """
        try:
            resp = self._client.get(
                f"{self.base_url}/gpu/reclaim-notice",
                params={"name": self.job_name},
                timeout=httpx.Timeout(_NOTICE_TIMEOUT, connect=5.0),
            )
            resp.raise_for_status()
            return bool(resp.json().get("reclaim"))
        except (httpx.HTTPError, ValueError) as exc:
            logger.debug(f"Reclaim notice failed ({exc}) — will ask again")
            return False

    def close(self) -> None:
        self._client.close()
