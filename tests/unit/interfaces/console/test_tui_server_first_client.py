"""The first client after a process start: the curtains play before the setup that clears the
screen, the client is registered only afterwards, and what it typed meanwhile is dropped
except for a resize. Later clients get the setup straight away."""
import select
import socket
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from interfaces.console.framework import Curtain
from interfaces.console.framework.TUIServer import TUIServer, FRAME_INTERVAL, machine_name
from interfaces.console.framework.TUIScreen import TUIScreen

ADDR = ("test", 0)
SETUP_START = (TUIScreen.BEGIN_SYNC + TUIScreen.ENTER_ALT_SCREEN).encode()


@pytest.fixture(autouse=True)
def accept_loop_notices_the_stop_at_once(monkeypatch):
    """Closing the listening socket does not wake a blocked accept() on macOS, so every teardown
    here would otherwise wait out the production second."""
    monkeypatch.setattr("interfaces.console.framework.TUIServer.ACCEPT_TIMEOUT", 0.02)


class TestDropTypedAhead:
    @pytest.fixture
    def pair(self):
        server_end, client_end = socket.socketpair()
        yield server_end, client_end
        server_end.close()
        client_end.close()

    def test_keys_typed_during_the_curtains_are_dropped(self, pair):
        server_end, client_end = pair
        client_end.sendall(b"q\x1b[A")
        assert TUIServer._drop_typed_ahead(server_end, ADDR, 80, 24) == (80, 24)
        server_end.setblocking(False)
        with pytest.raises(BlockingIOError):
            server_end.recv(1)

    def test_a_resize_is_kept_and_the_last_one_wins(self, pair):
        server_end, client_end = pair
        client_end.sendall(b"x\nRESIZE 30 100\nRESIZE 40 120\ny")
        assert TUIServer._drop_typed_ahead(server_end, ADDR, 80, 24) == (120, 40)

class TestFirstClient:
    @pytest.fixture
    def server(self):
        srv = TUIServer(draw_screen_callback=lambda screen, frame: None, host="127.0.0.1", port=0)
        srv.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.server_socket.bind(("127.0.0.1", 0))
        srv.server_socket.listen(5)
        srv.running = True
        thread = threading.Thread(target=srv._accept_clients, daemon=True)
        # autospec: a call that does not match the real signature must fail here too
        with patch.object(TUIServer, "_next_frame", autospec=True,
                          side_effect=lambda self, due, interval: due + interval):
            thread.start()
            yield srv
            srv.running = False
            srv.server_socket.close()
            thread.join(timeout=3)

    @staticmethod
    def connect(srv, dims=b"24 80\n"):
        client = socket.create_connection(srv.server_socket.getsockname(), timeout=3)
        client.sendall(dims)
        return client

    @staticmethod
    def wait_for_clients(srv, count):
        deadline = time.time() + 3
        while time.time() < deadline:
            with srv.clients_lock:
                if len(srv.clients) == count:
                    return
            time.sleep(0.01)
        raise AssertionError(f"{count} client(s) never registered")

    @staticmethod
    def drain(client) -> bytes:
        client.settimeout(0.05)   # loopback, and the server has already sent everything
        data = b""
        try:
            while chunk := client.recv(65536):
                data += chunk
        except socket.timeout:
            pass
        return data

    def test_the_first_client_gets_the_curtains_and_then_the_setup(self, server):
        client = self.connect(server)
        self.wait_for_clients(server, 1)
        data = self.drain(client)
        curtains = b"".join(Curtain.frames(80, 24, round(1 / FRAME_INTERVAL)))
        assert data.startswith(curtains)
        assert data[len(curtains):].startswith(SETUP_START)
        client.close()

    def test_the_first_client_is_registered_only_after_the_curtains(self, server):
        drawn = threading.Event()
        registered_meanwhile = []
        real = TUIServer._draw_curtains

        def draw(self_, *args):
            real(self_, *args)
            with self_.clients_lock:
                registered_meanwhile.append(len(self_.clients))
            drawn.set()

        with patch.object(TUIServer, "_draw_curtains", draw):
            client = self.connect(server)
            assert drawn.wait(3)
            self.wait_for_clients(server, 1)
        assert registered_meanwhile == [0]
        client.close()

    def test_a_key_typed_during_the_curtains_is_dropped_but_a_resize_is_honoured(self, server):
        started = threading.Event()
        real = TUIServer._draw_curtains

        def draw(self_, client_socket, *args):
            started.set()
            real(self_, client_socket, *args)
            select.select([client_socket], [], [], 3)   # the curtains last until the type-ahead has arrived

        with patch.object(TUIServer, "_draw_curtains", draw):
            client = self.connect(server)
            # Only once the dimensions are read: sent earlier, the type-ahead would land in
            # that same recv and the server would fall back to 80x24.
            assert started.wait(3)
            client.sendall(b"q\nRESIZE 30 100\n")
            self.wait_for_clients(server, 1)
        registered = next(iter(server.clients))
        assert server.client_dimensions[registered] == (100, 30)
        client.close()

    def test_a_later_client_gets_the_setup_straight_away(self, server):
        first = self.connect(server)
        self.wait_for_clients(server, 1)
        second = self.connect(server, b"40 120\n")
        self.wait_for_clients(server, 2)
        data = self.drain(second)
        assert data.startswith(SETUP_START)
        assert not server.first_client_pending
        first.close()
        second.close()

    def test_the_setup_titles_the_tab_with_the_servers_machine_not_the_address_dialled(self, server):
        with patch("interfaces.console.framework.TUIServer.machine_name", return_value="Marcus’s Mac\x07"):
            client = self.connect(server)
            self.wait_for_clients(server, 1)
            data = self.drain(client).decode()
        assert "\x1b]0;Keep - Marcus’s Mac\x07" in data   # the bell inside the name cannot end the title early
        client.close()


class TestTerminalProgram:
    """tui.py appends its TERM_PROGRAM to the dimension line; Terminal.app cannot draw the
    Crystal's half-block pixels, so the screen it drives gets full cells."""

    def test_a_third_token_names_the_terminal_and_terminal_app_gets_full_blocks(self):
        srv = TUIServer(draw_screen_callback=lambda frame, screen: None, host="127.0.0.1", port=0)
        srv.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.server_socket.bind(("127.0.0.1", 0))
        srv.server_socket.listen(5)
        srv.running = True
        srv.first_client_pending = False
        thread = threading.Thread(target=srv._accept_clients, daemon=True)
        thread.start()
        try:
            client = socket.create_connection(srv.server_socket.getsockname(), timeout=3)
            client.sendall(b"24 80 Apple_Terminal\n")
            deadline = time.time() + 3
            while time.time() < deadline and not srv.clients:
                time.sleep(0.01)
            registered = next(iter(srv.clients))
            assert srv.client_dimensions[registered] == (80, 24)
            assert srv.client_terminals[registered] == "Apple_Terminal"
            assert not srv.half_blocks_for("Apple_Terminal")
            assert srv.half_blocks_for("iTerm.app") and srv.half_blocks_for("")
            assert not srv.graphics_for("Apple_Terminal") and not srv.graphics_for("") and not srv.graphics_for("tmux")
            assert srv.graphics_for("ghostty") and srv.graphics_for("iTerm.app") and srv.graphics_for("kitty")
            client.close()
        finally:
            srv.running = False
            srv.server_socket.close()
            thread.join(timeout=3)


class TestMachineName:
    """The name a tab is titled with: Sharing's ComputerName, the short host name where scutil
    is missing (a Linux box) or answers nothing."""

    @pytest.fixture(autouse=True)
    def uncached(self):
        machine_name.cache_clear()
        yield
        machine_name.cache_clear()

    def test_without_scutil_the_short_host_name_stands_in(self):
        with patch("interfaces.console.framework.TUIServer.subprocess.run", side_effect=FileNotFoundError("scutil")), \
                patch("interfaces.console.framework.TUIServer.socket.gethostname", return_value="keep-box.local"):
            assert machine_name() == "keep-box"

    def test_an_empty_answer_from_scutil_is_no_name(self):
        with patch("interfaces.console.framework.TUIServer.subprocess.run",
                   return_value=SimpleNamespace(stdout="\n")) as run, \
                patch("interfaces.console.framework.TUIServer.socket.gethostname", return_value="keep-box.local"):
            assert machine_name() == "keep-box"
            assert machine_name() == "keep-box"
        assert run.call_count == 1, "read once per process"
