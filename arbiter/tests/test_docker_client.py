"""
The transport, against a socket that answers the way the real one does.

These exist because the hand-rolled client shipped with a bug that no amount of
scheduler testing could have caught: the daemon sends `Transfer-Encoding`,
docker-socket-proxy (haproxy) rewrites it to `transfer-encoding`, and a
case-sensitive match left the chunk-size prefixes in the body. Every inspect
failed in json.loads — but only once a container existed, because until then the
response was a 404 with an empty body and nothing to dechunk.

So: assert against bytes on a real socket, with the header spelled the way the
deployed proxy spells it.

Run:  .venv-test/bin/python -m pytest tests -q
"""

import socket
import threading

import pytest

from docker_client import DockerClient, DockerError, _dechunk


def _serve(response: bytes) -> tuple[str, int, threading.Thread]:
    """A one-shot HTTP server that replies with exactly ``response``."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    host, port = srv.getsockname()

    def run():
        conn, _ = srv.accept()
        with conn:
            conn.recv(65536)
            conn.sendall(response)
        srv.close()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return host, port, t


def _chunked(body: bytes, header: bytes = b"transfer-encoding: chunked") -> bytes:
    """A chunked response, header spelled as the caller wants it."""
    return (
        b"HTTP/1.1 200 OK\r\n"
        b"content-type: application/json\r\n" + header + b"\r\n"
        b"connection: close\r\n\r\n"
        + hex(len(body))[2:].encode() + b"\r\n" + body + b"\r\n0\r\n\r\n"
    )


def _client(host, port):
    return DockerClient(f"tcp://{host}:{port}", timeout=5.0)


# -- the bug that reached hardware --

@pytest.mark.parametrize(
    "header",
    [b"transfer-encoding: chunked", b"Transfer-Encoding: chunked"],
    ids=["lowercase-as-haproxy-sends-it", "titlecase-as-the-daemon-sends-it"],
)
def test_chunked_inspect_is_reassembled_whatever_the_header_case(header):
    payload = b'{"Id":"abc","State":{"Running":true,"ExitCode":0,"Status":"running"}}'
    host, port, t = _serve(_chunked(payload, header))

    state = _client(host, port).state("lora-trainer")

    t.join(timeout=5)
    assert state == {"Running": True, "ExitCode": 0, "Status": "running"}


def test_a_multibyte_body_survives_dechunking():
    """
    Chunk sizes count bytes. Dechunking a decoded string shifts every boundary
    after the first non-ASCII character, which container labels can carry.
    """
    payload = '{"Id":"abc","State":{"Status":"exité"}}'.encode()
    host, port, t = _serve(_chunked(payload))

    state = _client(host, port).state("x")

    t.join(timeout=5)
    assert state["Status"] == "exité"


def test_a_body_split_across_several_chunks_is_rejoined():
    body = b'{"State":{"Running":false}}'
    frames = b"".join(
        hex(len(p))[2:].encode() + b"\r\n" + p + b"\r\n"
        for p in [body[:10], body[10:20], body[20:]]
    )
    resp = (
        b"HTTP/1.1 200 OK\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n"
        + frames + b"0\r\n\r\n"
    )
    host, port, t = _serve(resp)

    state = _client(host, port).state("x")

    t.join(timeout=5)
    assert state == {"Running": False}


# -- the rest of the surface --

def test_a_missing_container_is_none_not_an_error():
    """A job may be configured before its image is built; that must not crash."""
    host, port, t = _serve(b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\n\r\n")

    assert _client(host, port).state("never-built") is None
    t.join(timeout=5)


def test_a_server_error_raises():
    host, port, t = _serve(b"HTTP/1.1 500 Server Error\r\ncontent-length: 4\r\n\r\nboom")

    with pytest.raises(DockerError):
        _client(host, port).state("x")
    t.join(timeout=5)


def test_killing_something_already_dead_is_success():
    """409 is Docker saying it is not running, which is the state we wanted."""
    host, port, t = _serve(b"HTTP/1.1 409 Conflict\r\ncontent-length: 0\r\n\r\n")

    _client(host, port).kill("lora-trainer")  # must not raise
    t.join(timeout=5)


def test_kill_names_the_signal_and_does_not_use_stop():
    """
    `stop` would leave a preempted job down forever: Docker suppresses the
    restart policy for a container stopped through its API. The signal has to
    reach the daemon on the kill endpoint, and the path must not be /stop.
    """
    seen = []
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    host, port = srv.getsockname()

    def run():
        conn, _ = srv.accept()
        with conn:
            seen.append(conn.recv(65536).decode())
            conn.sendall(b"HTTP/1.1 204 No Content\r\ncontent-length: 0\r\n\r\n")
        srv.close()

    t = threading.Thread(target=run, daemon=True)
    t.start()

    _client(host, port).kill("lora-trainer", "SIGKILL")

    t.join(timeout=5)
    assert "/containers/lora-trainer/kill?signal=SIGKILL" in seen[0]
    assert "/stop" not in seen[0]


def test_dechunk_stops_at_the_terminator():
    assert _dechunk(b"4\r\nabcd\r\n0\r\n\r\n") == b"abcd"


def test_dechunk_of_a_truncated_body_does_not_hang():
    """A short read must end, not spin — this runs inside the scheduler loop."""
    assert _dechunk(b"ff\r\nshort") == b"short"
