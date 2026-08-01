"""
The proxy's whole relationship with the GPU: two calls, and a name.

    acquire()   take the card, or refuse
    release()   done with it

The name is the only thing the proxy says about itself. It does not claim a
priority, state how much VRAM it needs, or name what should be stopped for it —
the arbiter reads all of that out of its own config, keyed on the name. A caller
that could assert its own importance would be assessing itself, and the LLM is
not exempt from that just because it happens to rank highest.

Nor does the proxy know what a job is, how many exist, how one is stopped, or how
much VRAM is free. It used to know all of it, and the knowledge was duplicated
into every client that competed with it. Now one service holds it.

``acquire`` failing is not an error to route around — it is the arbiter saying it
could not guarantee the card, and the only safe response is to refuse the request.
Loading a model over a job that may still hold several GB OOMs both workloads.
"""

import logging

import httpx

logger = logging.getLogger(__name__)

# Long enough to cover the arbiter's own acquire timeout plus a stop that runs to
# its grace period. If this fires first, we lose the arbiter's reason for failing,
# which is the thing worth reading in the log.
_TIMEOUT = 240.0


class ArbiterClient:
    def __init__(self, base_url: str, job_name: str):
        self.base_url = base_url.rstrip("/")
        self.job_name = job_name
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(_TIMEOUT, connect=5.0))

    async def acquire(self) -> tuple[bool, str]:
        """
        Ask for the GPU. Returns (granted, detail).

        An unreachable arbiter is treated as a refusal. Proceeding blind is the
        one option that risks an OOM: it would mean loading a model without
        knowing whether anything else is on the card.
        """
        try:
            resp = await self._client.post(
                f"{self.base_url}/gpu/acquire", json={"name": self.job_name}
            )
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

    async def release(self) -> None:
        """
        Tell the arbiter we are done.

        Best-effort by design: a failure here delays jobs restarting, which the
        arbiter recovers from on its next tick. It can never cause an OOM, so it
        must not be allowed to fail a request that has already been served.
        """
        try:
            await self._client.post(
                f"{self.base_url}/gpu/release", json={"name": self.job_name}
            )
        except httpx.HTTPError as exc:
            logger.warning(f"Could not tell the arbiter we are idle ({exc}) — it will catch up")

    async def status(self) -> dict:
        try:
            resp = await self._client.get(f"{self.base_url}/gpu/status", timeout=10.0)
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            return {"error": str(exc)}

    async def aclose(self) -> None:
        await self._client.aclose()
