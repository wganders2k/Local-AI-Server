"""
The Docker API surface the arbiter actually uses: three calls.

Talks to `docker-socket-proxy` over TCP rather than mounting the real socket,
which is root-equivalent on the host. That proxy already restricts the API to
`CONTAINERS` plus POST, which is exactly this module's surface and nothing more —
the arbiter cannot build images, read secrets, or touch the host filesystem even
if something got in through its three endpoints.

Deliberately hand-rolled rather than the `docker` SDK: it is a large dependency
for `inspect`, `start` and `stop`, and keeping the capability to thirty lines is
also the security argument.

Blocking on purpose — callers wrap it in asyncio.to_thread. An async HTTP client
would be another dependency for no benefit at this call volume (a handful of
requests per handover).
"""

import json
import logging
import socket
import urllib.parse

logger = logging.getLogger(__name__)

DEFAULT_HOST = "tcp://docker-socket-proxy:2375"


class DockerError(RuntimeError):
    pass


class DockerClient:
    def __init__(self, host: str = DEFAULT_HOST, timeout: float = 120.0):
        """
        ``host`` is either ``tcp://host:port`` or a path to a unix socket. TCP is
        the deployed shape; the unix path exists so this can be pointed at a real
        socket for local debugging without a second code path in the caller.
        """
        self.host = host
        self.timeout = timeout
        self._tcp: tuple[str, int] | None = None
        if host.startswith("tcp://"):
            hostname, _, port = host[len("tcp://"):].partition(":")
            self._tcp = (hostname, int(port or 2375))

    # -- the three operations --

    def state(self, container: str) -> dict | None:
        """
        Container state, or None if it does not exist.

        A missing container is not an error: a job may be defined in config
        before its image is built, and the scheduler should skip it rather than
        crash the arbiter.
        """
        status, body = self._request("GET", f"/containers/{urllib.parse.quote(container)}/json")
        if status == 404:
            return None
        if status != 200:
            raise DockerError(f"inspect {container}: HTTP {status} {body[:200]}")
        return json.loads(body).get("State", {})

    def kill(self, container: str, signal: str = "SIGKILL") -> None:
        """
        Signal a container. This is how a job is preempted, and `stop` is not.

        The distinction is load-bearing and cost an evening to find. Docker
        deliberately suppresses a container's restart policy when it is stopped
        through the API — a manual stop is taken to mean "and stay down" — so a
        preempted trainer with `restart: on-failure` never came back and its run
        was simply over. A signalled container dies the way it would from any
        other cause, the policy applies, and it restarts and asks for the card
        again. Which is the entire preempt-and-resume loop.
        """
        path = f"/containers/{urllib.parse.quote(container)}/kill?signal={signal}"
        status, body = self._request("POST", path)
        # 409 is "not running" — already the state we wanted.
        if status not in (204, 409):
            raise DockerError(f"kill {container}: HTTP {status} {body[:200]}")

    def ping(self) -> bool:
        try:
            status, _ = self._request("GET", "/_ping")
            return status == 200
        except OSError as exc:
            logger.warning(f"Docker socket not answering: {exc}")
            return False

    # -- transport --

    def _connect(self) -> socket.socket:
        if self._tcp:
            return socket.create_connection(self._tcp, timeout=self.timeout)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self.host)
        return sock

    def _request(self, method: str, path: str) -> tuple[int, str]:
        with self._connect() as sock:
            sock.settimeout(self.timeout)
            request = (
                f"{method} {path} HTTP/1.1\r\n"
                f"Host: docker\r\n"
                f"Accept: application/json\r\n"
                f"Connection: close\r\n\r\n"
            )
            sock.sendall(request.encode())

            chunks = []
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)

        raw = b"".join(chunks)
        head, _, body = raw.partition(b"\r\n\r\n")
        try:
            status = int(head.split(b"\r\n", 1)[0].split()[1])
        except (IndexError, ValueError) as exc:
            raise DockerError(f"unparseable response from docker: {raw[:200]!r}") from exc

        # Header names are case-insensitive per RFC 9110, and this is not
        # pedantry: the daemon sends "Transfer-Encoding", docker-socket-proxy is
        # haproxy and normalises it to "transfer-encoding". Matching on the
        # daemon's spelling left the chunk-size prefixes in the body, and every
        # inspect died in json.loads with "Extra data" — but only once a
        # container actually existed, since a 404 body is empty either way.
        if b"transfer-encoding: chunked" in head.lower():
            body = _dechunk(body)
        return status, body.decode("utf-8", errors="replace")


def _dechunk(body: bytes) -> bytes:
    """
    Reassemble a chunked response body. Docker uses it for inspect payloads.

    Works on bytes rather than decoded text on purpose — a chunk size is a count
    of bytes, and container names and image labels are not guaranteed ASCII. On
    a decoded string one multi-byte character silently shifts every subsequent
    chunk boundary.
    """
    out = []
    while body:
        size_line, sep, rest = body.partition(b"\r\n")
        if not sep:
            break
        try:
            size = int(size_line.split(b";")[0], 16)
        except ValueError:
            break
        if size == 0:
            break
        out.append(rest[:size])
        body = rest[size:]
        if body.startswith(b"\r\n"):
            body = body[2:]
    return b"".join(out)
