import pytest

from interfaces.console.framework import TUIServer as tui_server
from interfaces.console.framework.TUIServer import TUIServer, FRAME_INTERVAL

ADDR = ("test", 0)


@pytest.fixture
def server_keys():
    keys = []
    srv = TUIServer(
        draw_screen_callback=lambda screen, frame: None,
        input_callback=lambda key, addr: keys.append(key),
    )
    return srv, keys


class TestSingleChars:
    def test_multiple_chars_in_one_buffer(self, server_keys):
        srv, keys = server_keys
        leftover = srv._process_input_buffer("abc", ADDR)
        assert keys == ["a", "b", "c"]
        assert leftover == ""


class TestEscapeSequences:
    def test_arrow_keys(self, server_keys):
        srv, keys = server_keys
        srv._process_input_buffer("\x1b[A\x1b[B\x1b[C\x1b[D", ADDR)
        assert keys == ["\x1b[A", "\x1b[B", "\x1b[C", "\x1b[D"]

    def test_shift_arrow_keys(self, server_keys):
        srv, keys = server_keys
        leftover = srv._process_input_buffer("\x1b[1;2A", ADDR)
        assert keys == ["\x1b[1;2A"]
        assert leftover == ""

    def test_page_up_down(self, server_keys):
        srv, keys = server_keys
        srv._process_input_buffer("\x1b[5~\x1b[6~", ADDR)
        assert keys == ["\x1b[5~", "\x1b[6~"]

    def test_bare_escape_then_char(self, server_keys):
        srv, keys = server_keys
        leftover = srv._process_input_buffer("\x1bz", ADDR)
        assert keys == ["\x1b", "z"]
        assert leftover == ""


class TestIncompleteSequences:
    def test_lone_escape_buffered(self, server_keys):
        srv, keys = server_keys
        leftover = srv._process_input_buffer("\x1b", ADDR)
        assert keys == []
        assert leftover == "\x1b"

    def test_escape_bracket_buffered(self, server_keys):
        srv, keys = server_keys
        leftover = srv._process_input_buffer("\x1b[", ADDR)
        assert keys == []
        assert leftover == "\x1b["

    def test_complete_then_incomplete(self, server_keys):
        srv, keys = server_keys
        leftover = srv._process_input_buffer("a\x1b", ADDR)
        assert keys == ["a"]
        assert leftover == "\x1b"

    def test_arrow_then_incomplete(self, server_keys):
        srv, keys = server_keys
        leftover = srv._process_input_buffer("\x1b[A\x1b[", ADDR)
        assert keys == ["\x1b[A"]
        assert leftover == "\x1b["


class TestMixedInput:
    def test_char_arrow_char(self, server_keys):
        srv, keys = server_keys
        srv._process_input_buffer("x\x1b[Ay", ADDR)
        assert keys == ["x", "\x1b[A", "y"]

class TestCallbackErrors:
    def test_raising_callback_does_not_halt_processing(self):
        def boom(key, addr):
            raise RuntimeError("callback failure")

        srv = TUIServer(draw_screen_callback=lambda s, f: None, input_callback=boom)
        # Exception is caught internally; buffer is still fully consumed.
        leftover = srv._process_input_buffer("ab", ADDR)
        assert leftover == ""

    def test_no_callback_still_consumes_buffer(self):
        srv = TUIServer(draw_screen_callback=lambda s, f: None, input_callback=None)
        leftover = srv._process_input_buffer("abc", ADDR)
        assert leftover == ""


class TestGenericEscapeSequences:
    """The parser accepts any CSI / SS3 / Alt chord, so the composer gets its editing keys."""

    def test_home_end_delete_variants(self, server_keys):
        srv, keys = server_keys
        srv._process_input_buffer("\x1b[H\x1b[F\x1b[1~\x1b[4~\x1b[3~\x1bOH\x1bOF", ADDR)
        assert keys == ["\x1b[H", "\x1b[F", "\x1b[1~", "\x1b[4~", "\x1b[3~", "\x1bOH", "\x1bOF"]

    def test_readline_word_chords(self, server_keys):
        srv, keys = server_keys
        srv._process_input_buffer("\x1bb\x1bf\x1b\x7fz", ADDR)
        assert keys == ["\x1bb", "\x1bf", "\x1b\x7f", "z"]

    def test_alt_enter_is_one_chord_and_a_kitty_key_report_arrives_whole(self, server_keys):
        srv, keys = server_keys
        srv._process_input_buffer("\x1b\r\x1b[13;2ua", ADDR)
        assert keys == ["\x1b\r", "\x1b[13;2u", "a"]

    def test_bracketed_paste_markers_are_separate_keys(self, server_keys):
        srv, keys = server_keys
        srv._process_input_buffer("\x1b[200~hi\x1b[201~", ADDR)
        assert keys == ["\x1b[200~", "h", "i", "\x1b[201~"]

    def test_incomplete_csi_is_kept(self, server_keys):
        srv, keys = server_keys
        assert srv._process_input_buffer("\x1b[1;", ADDR) == "\x1b[1;"
        assert srv._process_input_buffer("\x1bO", ADDR) == "\x1bO"
        assert keys == []

    def test_a_graphics_reply_is_dropped_and_what_follows_is_a_key(self, server_keys):
        srv, keys = server_keys
        leftover = srv._process_input_buffer("\x1b_Gi=1;OK\x1b\\a", ADDR)
        assert keys == ["a"]
        assert leftover == ""

    def test_an_unfinished_graphics_reply_is_kept_for_the_next_read(self, server_keys):
        srv, keys = server_keys
        assert srv._process_input_buffer("\x1b_Gi=1;O", ADDR) == "\x1b_Gi=1;O"
        assert keys == []

    def test_an_unfinished_graphics_reply_past_the_cap_is_dropped_not_kept_forever(self, server_keys):
        """A terminal that never closes its APC string would otherwise grow the buffer on every
        read and swallow every key after it; past the cap the buffer is thrown away."""
        srv, keys = server_keys
        under = "\x1b_G" + "x" * (srv._APC_MAX - 3)
        assert srv._process_input_buffer(under, ADDR) == under
        over = "\x1b_G" + "x" * srv._APC_MAX
        assert srv._process_input_buffer(over, ADDR) == ""
        assert keys == []
        assert srv._process_input_buffer("q", ADDR) == "" and keys == ["q"]   # the next read is a key again

    def test_malformed_csi_is_dropped(self, server_keys):
        srv, keys = server_keys
        srv._process_input_buffer("\x1b[\x01q", ADDR)
        assert keys == ["q"]


class TestKeyNames:
    @pytest.mark.parametrize("seq, name", [
        ("\x1b[A", "up"), ("\x1b[1;2D", "shift_left"), ("\x1b[5~", "pageup"),
        ("\x1b[H", "home"), ("\x1b[1~", "home"), ("\x1bOH", "home"),
        ("\x1b[F", "end"), ("\x1b[4~", "end"), ("\x1bOF", "end"),
        ("\x1b[3~", "delete"),
        ("\x1b[1;3D", "alt_left"), ("\x1bb", "alt_left"), ("\x1b[1;3C", "alt_right"), ("\x1bf", "alt_right"),
        ("\x1b[1;5D", "ctrl_left"), ("\x1b[1;5C", "ctrl_right"), ("\x1b\x7f", "alt_backspace"),
        ("\x1b", "escape"), ("a", "a"), ("/", "/"), ("\x17", "\x17"),
        ("\t", "tab"), ("\x1b[Z", "shift_tab"),
        ("\x1b[200~", ""), ("\x1b[201~", ""), ("", ""),
        # Shift+Enter: the kitty keyboard protocol, xterm's modifyOtherKeys, and Alt+Enter
        # from a legacy terminal, all one newline
        ("\x1b[13;2u", "shift_enter"), ("\x1b[13;3u", "shift_enter"), ("\x1b[13;5u", "shift_enter"),
        ("\x1b[27;2;13~", "shift_enter"), ("\x1b\r", "shift_enter"), ("\x1b\n", "shift_enter"),
        # the other keys the protocol's disambiguate flag re-encodes keep their legacy names
        ("\x1b[13u", "\r"), ("\x1b[57414u", "\r"), ("\x1b[27u", "escape"), ("\x1b[9;2u", "shift_tab"),
        ("\x1b[127;3u", "alt_backspace"), ("\x1b[120;5u", "\x18"), ("\x1b[114;5u", "\x12"),
        ("\x1b[119;5u", "\x17"), ("\x1b[98;3u", "alt_left"), ("\x1b[102;3u", "alt_right"),
        ("\x1b[27;5;120~", "\x18"), ("\x1b[99;3u", ""),
    ])
    def test_parse_key(self, seq, name):
        from interfaces.console.TUIFloxyConsole import TUIFloxyConsole
        assert TUIFloxyConsole._parse_key(seq) == name


class TestLoneEscape:
    """A bare Escape only becomes a key once the select timeout passes with nothing after it;
    an arrow sequence split across two reads must still arrive as one chord."""

    @staticmethod
    def _run(srv, chunks):
        reads = list(chunks)

        class Sock:
            def recv(self, n):
                return reads.pop(0)

        sock = Sock()
        srv.running = True
        srv.clients.add(sock)
        quiet_ticks = iter([True, False])

        def fake_select(rlist, wlist, xlist, timeout):
            # Readable once per chunk, then one quiet timeout, then the server is stopped.
            if reads:
                return rlist, [], []
            if not next(quiet_ticks):
                srv.running = False
            return [], [], []

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("interfaces.console.framework.TUIServer.select.select", fake_select)
            srv._handle_client_input(sock, ADDR)

    def test_lone_escape_is_emitted_after_the_select_timeout(self, server_keys):
        srv, keys = server_keys
        self._run(srv, [b"\x1b"])
        assert keys == ["\x1b"]

    def test_arrow_split_across_reads_is_one_key(self, server_keys):
        srv, keys = server_keys
        self._run(srv, [b"\x1b", b"[A"])
        assert keys == ["\x1b[A"]


class FakeSock:
    def __init__(self, fail=False):
        self.fail, self.closed = fail, False

    def sendall(self, data):
        if self.fail:
            raise BrokenPipeError

    def close(self):
        self.closed = True
        if self.fail:
            raise OSError("already closed")


class TestDropClient:
    def test_one_dead_client_does_not_take_the_others_down(self):
        srv = TUIServer(draw_screen_callback=lambda screen, frame: None)
        good, bad = FakeSock(), FakeSock(fail=True)
        srv.clients = {good, bad}
        srv.client_dimensions = {good: (80, 24), bad: (100, 30)}
        srv.last_resized_client = bad
        srv._broadcast(b"x")
        assert srv.clients == {good} and bad.closed and not good.closed
        assert srv.client_dimensions == {good: (80, 24)}
        assert srv.last_resized_client is None
        with srv.clients_lock:
            srv._drop_client(bad, "again")         # idempotent
        assert srv.clients == {good}


class TestFramePacing:
    """The animation loop holds its rate: a frame sleeps only what is left of its slot, and the
    sleep is the wake event's wait so input can end it early."""

    def _server(self, monkeypatch, now):
        slept = []
        monkeypatch.setattr(tui_server.time, "perf_counter", lambda: now)
        srv = TUIServer(draw_screen_callback=lambda frame, screen: None)
        monkeypatch.setattr(srv._wake, "wait", lambda timeout: slept.append(timeout) and False)
        return srv, slept

    def test_a_quick_frame_sleeps_the_rest_of_its_slot(self, monkeypatch):
        srv, slept = self._server(monkeypatch, now=100.010)          # drawing took 10 ms of the slot
        due = 100.0 + FRAME_INTERVAL
        assert srv._next_frame(due, FRAME_INTERVAL) == pytest.approx(due + FRAME_INTERVAL)
        assert slept == [pytest.approx(FRAME_INTERVAL - 0.010)]

    def test_a_slow_frame_does_not_sleep_and_starts_a_fresh_slot(self, monkeypatch):
        srv, slept = self._server(monkeypatch, now=100.050)          # drawing took 50 ms, past the slot
        assert srv._next_frame(100.0 + FRAME_INTERVAL, FRAME_INTERVAL) == pytest.approx(100.050 + FRAME_INTERVAL)
        assert slept == []

    def test_a_frame_scheduled_an_idle_slot_ahead_sleeps_only_a_full_slot(self, monkeypatch):
        srv, slept = self._server(monkeypatch, now=100.0)
        assert srv._next_frame(100.0 + 0.2, FRAME_INTERVAL) == pytest.approx(100.0 + 2 * FRAME_INTERVAL)
        assert slept == [pytest.approx(FRAME_INTERVAL)]

    def test_a_wake_ends_the_sleep_and_the_next_slot_starts_now(self, monkeypatch):
        srv, slept = self._server(monkeypatch, now=100.0)
        monkeypatch.setattr(srv._wake, "wait", lambda timeout: True)
        assert srv._next_frame(100.0 + 0.2, FRAME_INTERVAL) == pytest.approx(100.0 + FRAME_INTERVAL)
