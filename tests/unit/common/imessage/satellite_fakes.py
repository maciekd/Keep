"""A SatelliteService on an ephemeral localhost port, served by the real hypercorn: both ends of
the wire are tested over it — the real HTTP, a fake MessagesHost behind it — and nothing here
leaves this machine. Also the stand-ins around it: a follower that only records, and peers that
are not satellites at all."""
import asyncio
import socket
import threading
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from hypercorn.asyncio import serve

from common import Version
from common.imessage import Wire
from common.imessage.MessagesHost import SatelliteStatus
from common.imessage.RemoteMessagesHost import RemoteMessagesHost
from common.imessage.SatelliteService import SatelliteService, hypercorn_config
from tests.unit.integrations.imessage.host_fakes import FakeHost

TOKEN = "kT7vQ2mX9pL4wZ8nB3cR6yH1sD5fG0jA"       # fictional, the length install.sh writes
SERVER_BRANCH, SERVER_COMMIT = "release", "5e" * 20
LOCALHOST = "127.0.0.1"


class FakeFollower:
    """Records what each request said the server runs and answers with a status the test sets."""

    def __init__(self):
        self.notices: list[tuple[str, str]] = []
        self.answer = SatelliteStatus(Wire.WIRE_VERSION, "v2026.11.7 (0a1b2c3d, release)", "release", "0a" * 20)

    def notice(self, branch: str, commit: str) -> None:
        self.notices.append((branch, commit))

    def status(self) -> SatelliteStatus:
        return self.answer


def listening_socket() -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((LOCALHOST, 0))
    sock.listen(16)
    return sock


class RunningSatellite:
    """One service behind hypercorn on its own thread and loop, until `stop`."""

    def __init__(self):
        self.service = SatelliteService(FakeHost(), TOKEN, FakeFollower())
        sock = listening_socket()       # listening before hypercorn is up: a request waits, no test sleeps
        self.address = f"http://{LOCALHOST}:{sock.getsockname()[1]}"
        config = hypercorn_config(f"fd://{sock.detach()}")      # hypercorn as the satellite configures it
        config.graceful_timeout = 0.05
        config.errorlog = None          # no "Running on …" line per module in the suite's output
        self._loop = asyncio.new_event_loop()
        self._shutdown = asyncio.Event()
        self._thread = threading.Thread(target=self._loop.run_until_complete, name="test-satellite", daemon=True,
                                        args=(serve(self.service.app, config, shutdown_trigger=self._shutdown.wait),))
        self._thread.start()

    def stop(self):
        self._loop.call_soon_threadsafe(self._shutdown.set)
        self._thread.join(timeout=2)
        if not self._thread.is_alive():
            self._loop.close()


@pytest.fixture(scope="module")
def running_satellite():
    satellite = RunningSatellite()
    yield satellite
    satellite.stop()


@pytest.fixture
def satellite(running_satellite):
    """The running service with a fresh fake host and follower behind it."""
    service = running_satellite.service
    service.host, service.follower, service.wire, service._caller_wire = FakeHost(), FakeFollower(), Wire.WIRE_VERSION, None
    return running_satellite


@pytest.fixture
def fake(satellite) -> FakeHost:
    return satellite.service.host


@pytest.fixture
def follower(satellite) -> FakeFollower:
    return satellite.service.follower


@contextmanager
def server_identity(branch=SERVER_BRANCH, commit=SERVER_COMMIT):
    """What the process building a RemoteMessagesHost runs, as `common.Version` would tell it."""
    with patch.object(Version, "branch", return_value=branch), patch.object(Version, "commit", return_value=commit):
        yield


@pytest.fixture
def remote(satellite):
    with server_identity():
        host = RemoteMessagesHost(satellite.address, TOKEN)
    yield host
    host._session.close()


def closed_port() -> str:
    """An address nothing listens on."""
    sock = listening_socket()
    port = sock.getsockname()[1]
    sock.close()
    return f"http://{LOCALHOST}:{port}"


@contextmanager
def silent_peer():
    """Takes the connection and never answers."""
    sock = listening_socket()
    try:
        yield f"http://{LOCALHOST}:{sock.getsockname()[1]}"
    finally:
        sock.close()


def whole_request(connection: socket.socket) -> bytes:
    """Headers and body: a peer that closes on a half-read request resets the connection instead of answering."""
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = connection.recv(65536)
        if not chunk:
            return data
        data += chunk
    head, _, body = data.partition(b"\r\n\r\n")
    length = next((int(line.split(b":")[1]) for line in head.split(b"\r\n") if line.lower().startswith(b"content-length:")), 0)
    while len(body) < length:
        chunk = connection.recv(65536)
        if not chunk:
            break
        body += chunk
    return head + b"\r\n\r\n" + body


@contextmanager
def canned_peer(response: bytes):
    """Answers every connection with the same bytes and keeps what it was sent."""
    sock = listening_socket()
    received: list[bytes] = []

    def answer():
        while True:
            try:
                connection, _ = sock.accept()
            except OSError:
                return
            with connection:
                received.append(whole_request(connection))
                connection.sendall(response)

    threading.Thread(target=answer, name="canned-peer", daemon=True).start()
    try:
        yield f"http://{LOCALHOST}:{sock.getsockname()[1]}", received
    finally:
        sock.close()


@contextmanager
def trickling_peer(at_once: bytes, drip: bytes, pause: float = 0.01):
    """Sends `at_once`, then `drip` a byte at a time with a pause between: no read of the caller
    ever waits long, and the answer is never through. Keeps how many connections it took."""
    sock = listening_socket()
    connections: list[socket.socket] = []
    done = threading.Event()

    def serve(connection: socket.socket):
        try:
            whole_request(connection)
            connection.sendall(at_once)
            for byte in drip:
                if done.wait(pause):
                    return
                connection.sendall(bytes([byte]))
        except OSError:
            pass
        finally:
            connection.close()

    def accept():
        while True:
            try:
                connection, _ = sock.accept()
            except OSError:
                return
            connections.append(connection)
            threading.Thread(target=serve, args=(connection,), name="trickling-peer", daemon=True).start()

    threading.Thread(target=accept, name="trickling-peer-accept", daemon=True).start()
    try:
        yield f"http://{LOCALHOST}:{sock.getsockname()[1]}", connections
    finally:
        done.set()
        sock.close()


def http_response(status: str, body: bytes = b"", **headers: str) -> bytes:
    lines = [f"HTTP/1.1 {status}", f"Content-Length: {len(body)}", "Connection: close"]
    lines += [f"{name.replace('_', '-')}: {value}" for name, value in headers.items()]
    return "\r\n".join(lines).encode() + b"\r\n\r\n" + body
