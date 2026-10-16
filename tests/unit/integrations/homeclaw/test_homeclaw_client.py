"""HomeClawClient against a socket server of its own: the request line, the envelope, the
app's error, and the two ways the app can be absent."""
import json
import os
import socket
import tempfile
import threading
from pathlib import Path

import pytest

from integrations.homeclaw.HomeClawClient import HomeClawClient, HomeClawError, HomeClawUnavailable


class FakeApp:
    """Answers one request per connection with `reply`, keeping what it was asked."""

    def __init__(self, path: Path, reply):
        self.path, self.reply, self.requests = path, reply, []
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(path))
        self._server.listen(2)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while True:
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            with conn:
                line = b""
                while not line.endswith(b"\n"):
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    line += chunk
                self.requests.append(json.loads(line.decode()))
                conn.sendall((json.dumps(self.reply) + "\n").encode())

    def close(self):
        self._server.close()


@pytest.fixture
def sock_dir():
    # macOS caps a socket path at 104 bytes; pytest's tmp_path is longer than that.
    folder = Path(tempfile.mkdtemp(prefix="hc", dir="/tmp"))
    yield folder
    for child in folder.iterdir():
        child.unlink()
    folder.rmdir()


@pytest.mark.unit
class TestCall:
    def test_sends_the_command_with_its_args_and_returns_the_data(self, sock_dir):
        app = FakeApp(sock_dir / "s.sock", {"success": True, "data": {"ready": True, "homes": 1}})
        try:
            data = HomeClawClient(str(sock_dir / "s.sock")).call("status", home_id=None, refresh=False)
        finally:
            app.close()
        assert data == {"ready": True, "homes": 1}
        assert app.requests == [{"command": "status", "args": {"refresh": False}}]

    def test_the_apps_error_is_raised_with_its_text(self, sock_dir):
        app = FakeApp(sock_dir / "s.sock", {"success": False, "error": "Accessory not found: Lamp"})
        try:
            with pytest.raises(HomeClawError, match="Accessory not found: Lamp"):
                HomeClawClient(str(sock_dir / "s.sock")).call("get_accessory", id="Lamp")
        finally:
            app.close()

    def test_a_missing_socket_is_the_app_not_running(self, sock_dir):
        with pytest.raises(HomeClawUnavailable, match="not running"):
            HomeClawClient(str(sock_dir / "none.sock")).call("status")

    def test_a_socket_nobody_listens_on_is_the_app_not_running(self, sock_dir):
        app = FakeApp(sock_dir / "s.sock", {"success": True, "data": None})
        app.close()
        assert os.path.exists(sock_dir / "s.sock")
        with pytest.raises(HomeClawUnavailable, match="not running"):
            HomeClawClient(str(sock_dir / "s.sock")).call("status")

    def test_the_default_path_is_the_apps_group_container_in_the_home_folder(self):
        client = HomeClawClient()
        assert client.path.is_absolute() and "group.com.shahine.homeclaw" in str(client.path)
