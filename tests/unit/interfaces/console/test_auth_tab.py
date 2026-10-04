"""AuthTab: the shared login screen and code entry of the iCloud, Arlo, Ring and Telegram
tabs. The keys arrive as TUIFloxyConsole._parse_key names them: Enter is '\\r', Backspace
'\\x7f', Escape 'escape'."""
import time
from unittest.mock import Mock

import pytest

from assistant.integration.SingletonMixin import SingletonMixin
from interfaces.console.framework.TUIScreen import TUIScreen
from interfaces.console.tabs import AuthTab as auth_tab_module
from interfaces.console.Theme import COLOR_GREEN, COLOR_RED, COLOR_YELLOW
from interfaces.console.tabs.AuthTab import AuthTab, code_display
from interfaces.console.widgets.Board import COLOR_PROMPT, SIDE_BY_SIDE_MIN, Table
from interfaces.console.tabs.ArloTab import ArloTab
from interfaces.console.tabs.iCloudTab import iCloudTab
from interfaces.console.tabs.RingTab import RingTab
from interfaces.console.tabs.TelegramTab import INPUT_CODE, INPUT_PASSWORD, INPUT_PHONE, TelegramTab


class FakeTab(AuthTab):
    TITLE = "Fake Service"

    def __init__(self, api=None):
        super().__init__()
        self.verified = []
        self.refreshed = 0
        self._api = api

    def _connect(self):
        raise RuntimeError("not configured")

    def _verify_code(self, api, code):
        self.verified.append(code)

    def _refresh(self):
        self.refreshed += 1


def make_tab(status="authenticated", message="ok"):
    api = Mock()
    api.get_auth_status.return_value = {"status": status, "message": message}
    tab = FakeTab(api)
    tab._auth_status = {"status": status, "message": message}
    tab._run = lambda fn: fn()   # actions run inline instead of on a worker thread
    return tab


def rows(screen):
    return ["".join(c.char for c in line).rstrip() for line in screen.next_frame]


def draw(tab, width=100, height=30):
    screen = TUIScreen(width, height)
    tab._last_check_time = time.time() + 1e6   # no background check during the test
    tab.draw(screen, 0)
    return screen


@pytest.mark.unit
class TestCodeEntry:
    def test_digits_fill_the_code_and_the_sixth_submits_it(self):
        tab = make_tab()
        tab.handle_input('\x05')
        assert tab._input_mode is True
        for ch in "12345":
            tab.handle_input(ch)
        assert tab.verified == [] and tab._input_text == "12345"
        tab.handle_input('6')
        assert tab.verified == ["123456"]
        assert tab._input_mode is False and tab._input_text == ""

    def test_backspace_as_the_terminal_sends_it_edits_the_code(self):
        tab = make_tab()
        tab._begin_entry()
        for ch in "123":
            tab.handle_input(ch)
        assert tab.handle_input('\x7f')
        assert tab._input_text == "12"
        tab.handle_input('backspace')
        assert tab._input_text == "1"

    def test_enter_as_the_terminal_sends_it_submits_a_complete_code(self):
        tab = make_tab()
        tab._begin_entry()
        tab._input_text = "12345"
        tab.handle_input('\r')
        assert tab.verified == []          # incomplete: nothing sent
        tab._input_text = "123456"
        tab.handle_input('\r')
        assert tab.verified == ["123456"]

    def test_escape_cancels_the_entry_and_clears_the_outcome(self):
        tab = make_tab()
        tab._status_message = "SMS code sent"
        tab._begin_entry()
        tab.handle_input('1')
        assert tab.handle_input('escape')
        assert tab._input_mode is False and tab._input_text == "" and tab._status_message == ""

    def test_every_key_is_consumed_while_entering(self):
        tab = make_tab()
        tab._begin_entry()
        assert tab.handle_input('x') and tab.handle_input('up') and tab.handle_input('/')
        assert tab._input_text == ""

    def test_outside_entry_only_the_shortcuts_are_taken(self):
        tab = make_tab()
        assert tab.handle_input('\x12') and tab.refreshed == 1
        assert tab.handle_input('1') is False

    def test_leaving_the_tab_drops_a_half_typed_code(self):
        tab = make_tab()
        tab._begin_entry()
        tab.handle_input('4')
        tab.on_deactivate()
        assert tab._input_mode is False and tab._input_text == ""


@pytest.mark.unit
class TestActions:
    def test_submitting_without_an_integration_reports_it(self):
        tab = FakeTab()
        tab._run = lambda fn: fn()
        tab._begin_entry()
        tab._input_text = "123456"
        tab.handle_input('\r')
        assert tab._status_message == "Error: API not initialized"
        assert tab._input_mode is False
        assert tab._auth_status["status"] == "error" and "not configured" in tab._auth_status["message"]

    def test_a_failing_action_becomes_the_outcome(self):
        tab = make_tab()
        tab._verify_code = Mock(side_effect=RuntimeError("bad code"))
        tab._begin_entry()
        for ch in "123456":
            tab.handle_input(ch)
        assert tab._status_message == "Error: bad code"

    def test_status_check_records_the_time_and_asks_the_tab_for_its_lists(self):
        tab = make_tab(status="needs_2fa")
        tab._on_status = Mock()
        tab._check_auth_status()
        assert tab._last_check_time > 0
        tab._on_status.assert_called_once_with(tab._api)

    def test_an_unavailable_integration_reads_as_error_and_is_logged_once(self, monkeypatch):
        # _get_api is asked every frame by some tabs: the same failure must not flood the log
        lines = []
        monkeypatch.setattr(auth_tab_module, "trace", lines.append)
        tab = FakeTab()
        for _ in range(3):
            assert tab._get_api() is None
        assert tab._auth_status == {"status": "error", "message": "not configured"}
        assert len(lines) == 1 and "not configured" in lines[0]

    def test_only_one_status_check_runs_at_a_time(self):
        tab = make_tab()
        started = []
        tab._run = started.append   # the worker is started but never finishes
        tab._check_auth_status_async()
        assert tab._checking is True
        tab._check_auth_status_async()
        assert len(started) == 1
        started[0]()
        assert tab._checking is False


@pytest.mark.unit
class RebuildableService(SingletonMixin):
    def get_auth_status(self):
        return {"status": "authenticated", "message": "Connected"}


class SingletonTab(FakeTab):
    def _connect(self):
        return RebuildableService.get_instance()


class TestRebuiltIntegration:
    """A tab built before config_request created the section must follow the monitor's
    restart to the new singleton, not keep the unconfigured instance it cached."""

    @pytest.fixture(autouse=True)
    def reset(self):
        RebuildableService.reset_instance()
        yield
        RebuildableService.reset_instance()

    def test_the_cached_instance_is_kept_while_it_is_the_singleton(self):
        tab = SingletonTab()
        first = tab._get_api()
        assert tab._get_api() is first

    def test_a_restart_replaces_the_cached_instance(self):
        tab = SingletonTab()
        first = tab._get_api()
        RebuildableService.reset_instance()
        assert tab._get_api() is not first
        assert tab._get_api() is RebuildableService.get_instance()

    def test_a_plain_object_is_never_treated_as_stale(self):
        tab = FakeTab(api=Mock())
        assert tab._get_api() is tab._api


class TestOutcomeColor:
    @pytest.mark.parametrize("message, color", [
        ("2FA verification successful!", COLOR_GREEN),
        ("Connection refreshed", COLOR_GREEN),
        ("Verifying 2FA code...", COLOR_YELLOW),
        ("Reconnecting...", COLOR_YELLOW),
        ("Error: boom", COLOR_RED),
        ("Auth error: nope", COLOR_RED),
    ])
    def test_done_busy_and_failed(self, message, color):
        assert AuthTab._outcome_color(message) == color


def row_with(lines, needle):
    matches = [l for l in lines if needle in l]
    assert len(matches) == 1, (needle, lines)
    return matches[0]


def cell_style(screen, lines, needle):
    """The style of the first cell of `needle` on the one row that carries it."""
    y = lines.index(row_with(lines, needle))
    return screen.next_frame[y][lines[y].index(needle)].style


class TableTab(FakeTab):
    def __init__(self, names=()):
        super().__init__(api=Mock())
        self._auth_status = {"status": "authenticated", "message": "ok"}
        self.names = list(names)

    def _table(self):
        return Table("Gadgets", [("Gadget", 20), ("Room", None)],
                     [(COLOR_GREEN, [(name, COLOR_GREEN), ("hall", COLOR_GREEN)]) for name in self.names],
                     "nothing to list")


@pytest.mark.unit
class TestDraw:
    def test_the_board_has_connection_message_entry_outcome_and_hints(self):
        tab = make_tab(status="authenticated", message="Session valid")
        tab._begin_entry()
        tab._input_text = "12"
        tab._status_message = "SMS code sent to 05"
        screen = draw(tab)
        lines = rows(screen)
        assert "Connected" in row_with(lines, "Connection")
        assert "Session valid" in row_with(lines, "Status")
        entry = row_with(lines, "1 2 _ _ _ _")
        assert lines.index(row_with(lines, "Status")) < lines.index(entry) < lines.index(
            row_with(lines, "SMS code sent to 05"))
        # the field the entry waits on is labelled in the prompt colour
        y = lines.index(entry)
        label_x = len(entry) - len(entry.lstrip())
        assert screen.next_frame[y][label_x].style.fg == COLOR_PROMPT
        assert "Esc" in lines[-2] and "Ctrl-R" not in lines[-2]

    def test_the_hints_name_the_shortcuts_outside_an_entry(self):
        hints = rows(draw(make_tab()))[-2]
        assert "Ctrl-R" in hints and "Ctrl-E" in hints

    @pytest.mark.parametrize("status, color", [
        ("authenticated", COLOR_GREEN), ("error", COLOR_RED),
    ])
    def test_the_connection_tile_carries_the_status_in_its_colour(self, status, color):
        tab = make_tab(status=status)
        screen = draw(tab)
        label = AuthTab.STATUS_LABELS[status][0]
        assert cell_style(screen, rows(screen), label).fg == color

    def test_an_unknown_status_reads_as_not_initialized(self):
        assert "Not initialized" in row_with(rows(draw(make_tab(status="whatever"))), "Connection")

    def test_checking_wins_over_the_status(self):
        tab = make_tab()
        tab._checking = True
        line = row_with(rows(draw(tab)), "Connection")
        assert "Checking" in line and "Connected" not in line

    def test_long_lines_stay_inside_the_tile(self):
        tab = make_tab(message="m" * 300)
        tab._status_message = "o" * 300
        width = 60
        screen = draw(tab, width=width)
        lines = rows(screen)
        assert all(len(l) <= width for l in lines)
        message_rows = [l for l in lines if "mmmm" in l]
        assert 1 <= len(message_rows) <= 2          # wrapped, then cut
        assert len([l for l in lines if "oooo" in l]) == 1

    def test_an_error_message_is_drawn_in_red(self):
        tab = make_tab(status="error", message="login refused")
        screen = draw(tab)
        assert cell_style(screen, rows(screen), "login refused").fg == COLOR_RED


@pytest.mark.unit
class TestTable:
    def test_every_row_is_listed_under_the_column_titles_with_the_count(self):
        lines = rows(draw(TableTab(["lamp", "fan"]), width=100, height=40))
        title = row_with(lines, "Gadgets")
        assert title.rstrip().endswith("2")
        header = lines.index(row_with(lines, "Room"))
        assert "lamp" in lines[header + 1] and "fan" in lines[header + 2]
        assert lines[header + 1].lstrip().startswith("●")

    def test_an_empty_table_says_so(self):
        lines = rows(draw(TableTab(), width=100, height=40))
        assert any("nothing to list" in l for l in lines)
        assert not any("Room" in l for l in lines)

    def test_the_sign_in_tile_is_beside_the_table_on_a_wide_screen_and_above_it_on_a_narrow_one(self):
        wide = rows(draw(TableTab(["lamp"]), width=SIDE_BY_SIDE_MIN, height=40))
        line = row_with(wide, "Gadgets")
        assert line.index("Gadgets") < line.index("Sign-in")
        narrow = rows(draw(TableTab(["lamp"]), width=SIDE_BY_SIDE_MIN - 1, height=40))
        assert narrow.index(row_with(narrow, "Sign-in")) < narrow.index(row_with(narrow, "Gadgets"))

    def test_down_scrolls_a_list_longer_than_its_tile(self):
        tab = TableTab([f"gadget-{i:02d}" for i in range(40)])
        assert any("gadget-00" in l for l in rows(draw(tab, width=100, height=30)))
        assert tab.handle_input('down')
        lines = rows(draw(tab, width=100, height=30))
        assert not any("gadget-00" in l for l in lines) and any("gadget-01" in l for l in lines)
        for _ in range(100):
            tab.handle_input('down')
        assert any("gadget-39" in l for l in rows(draw(tab, width=100, height=30)))   # clamped at the end


@pytest.mark.unit
class TestICloudBoard:
    def make(self, status="authenticated"):
        tab = iCloudTab()
        tab._api = Mock(auto_2fa_message=None)
        tab._auth_status = {"status": status, "message": ""}
        tab._icloud_devices = [
            {"id": "id-phone-0001", "name": "Marcus's iPhone", "model": "iPhone 15", "configured_as": "Marcus"},
            {"id": "id-pad-0002", "name": "Kitchen iPad", "model": "iPad Air", "configured_as": None},
        ]
        tab._devices_list = "Marcus (iPhone)"
        return tab

    def test_a_configured_device_shows_its_name_and_a_green_dot_an_unconfigured_one_neither(self):
        screen = draw(self.make(), width=120, height=40)
        lines = rows(screen)
        phone, pad = row_with(lines, "id-phone-0001"), row_with(lines, "id-pad-0002")
        assert "Marcus's iPhone" in phone and "iPhone 15" in phone
        assert phone.count("Marcus") == 2 and "Marcus" not in pad
        dot = lambda line: screen.next_frame[lines.index(line)][line.index("●")].style.fg
        assert dot(phone) == COLOR_GREEN and dot(pad) != COLOR_GREEN

    def test_the_tiles_count_the_devices_seen_and_configured(self):
        lines = rows(draw(self.make(), width=120, height=40))
        assert row_with(lines, "Devices").split("Devices")[1].split()[0] == "2"
        assert row_with(lines, "Configured  ").rstrip().endswith("1")
        assert any("Marcus (iPhone)" in l for l in lines)

    def test_an_unverified_session_lists_no_devices(self):
        lines = rows(draw(self.make(status="needs_2fa"), width=120, height=40))
        assert not any("id-phone-0001" in l for l in lines)


@pytest.mark.unit
class TestArloBoard:
    def make(self):
        tab = ArloTab()
        tab._api = Mock()
        tab._auth_status = {"status": "authenticated", "message": ""}
        tab._camera_names = ["Driveway", "Garden"]
        return tab

    def test_the_cameras_are_listed_and_counted(self):
        lines = rows(draw(self.make(), width=120, height=40))
        assert any(l.strip() == "● Driveway" for l in lines) and any(l.strip() == "● Garden" for l in lines)
        assert not any("Rate limited" in l for l in lines)

    def test_a_cooldown_gets_a_tile_counting_it_down(self):
        tab = self.make()
        tab._rate_limit_until = time.time() + 125.5
        line = row_with([l for l in rows(draw(tab, width=120, height=40)) if "Cameras" in l], "Rate limited")
        assert "02:05" in line


@pytest.mark.unit
class TestCodeDisplay:
    def test_slots_for_the_digits_still_to_come(self):
        assert code_display("", 6) == "_ _ _ _ _ _ _"
        assert code_display("12", 6) == "1 2 _ _ _ _"
        assert code_display("123456", 6) == "1 2 3 4 5 6"


@pytest.mark.unit
class TestTelegramEntry:
    def make(self):
        tab = TelegramTab()
        tab._api = Mock()
        tab._run = lambda fn: fn()
        return tab

    def test_ctrl_p_starts_the_phone_entry_with_the_plus(self):
        tab = self.make()
        assert tab.handle_input('\x10')
        assert tab._input_mode == INPUT_PHONE and tab._input_text == "+"
        for ch in "46123":
            tab.handle_input(ch)
        tab.handle_input('x')
        assert tab._input_text == "+46123"

    def test_backspace_keeps_the_plus_and_escape_cancels(self):
        tab = self.make()
        tab.handle_input('\x10')
        tab.handle_input('1')
        tab.handle_input('\x7f')
        tab.handle_input('\x7f')
        assert tab._input_text == "+"
        assert tab.handle_input('escape')
        assert tab._input_mode is False and tab._input_text == ""

    def test_enter_sends_the_code_request_and_moves_on_to_the_code(self):
        tab = self.make()
        tab.handle_input('\x10')
        for ch in "46123456789":
            tab.handle_input(ch)
        tab.handle_input('\r')
        tab._api.send_code_request.assert_called_once_with("+46123456789")
        assert tab._input_mode == INPUT_CODE and tab._phone_number == "+46123456789"
        for ch in "12345":
            tab.handle_input(ch)
        tab._api.sign_in.assert_called_once_with("+46123456789", "12345")

    def test_a_password_challenge_switches_to_the_password_entry(self):
        tab = self.make()
        tab._api.sign_in.side_effect = RuntimeError("Two-step verification password needed")
        tab._phone_number = "+1"
        tab._input_mode = INPUT_CODE
        tab._input_text = "12345"
        tab.handle_input('\r')
        assert tab._status_message == "2FA password required"
        assert tab._input_mode == "password"
        for ch in "s3cret":
            tab.handle_input(ch)
        tab.handle_input('enter')
        tab._api.sign_in_2fa.assert_called_once_with("s3cret")

    def test_the_password_prompt_never_shows_the_password(self):
        # Every connected TUI client keeps the frame in its scrollback.
        tab = self.make()
        tab._auth_status = {"status": "needs_auth", "message": ""}
        tab._phone_number = "+46701740605"
        tab._input_mode, tab._input_text = INPUT_PASSWORD, "hunter2"
        lines = rows(draw(tab, width=90))
        password = row_with(lines, "Password")
        assert password.count("•") == len("hunter2")
        assert not any("hunter2" in l for l in lines)
        assert "+46701740605" in row_with(lines, "Phone")     # the answered step stays on show

    def test_the_step_being_typed_is_the_one_in_the_prompt_colour(self):
        tab = self.make()
        tab._auth_status = {"status": "needs_auth", "message": ""}
        tab._phone_number = "+46701740605"
        tab._input_mode, tab._input_text = INPUT_CODE, "12"
        screen = draw(tab, width=90)
        lines = rows(screen)
        assert "1 2 _ _ _" in row_with(lines, "Code")
        assert cell_style(screen, lines, "Code").fg == COLOR_PROMPT
        assert cell_style(screen, lines, "Phone").fg != COLOR_PROMPT
        assert cell_style(screen, lines, "Password").fg != COLOR_PROMPT

    def test_the_password_takes_printable_keys_only(self):
        tab = self.make()
        tab._input_mode = INPUT_PASSWORD
        for key in ("h", "ü", "up", "\x01", "\t", " "):
            assert tab.handle_input(key)
        assert tab._input_text == "hü "

    def test_enter_on_a_short_phone_or_code_is_ignored(self):
        tab = self.make()
        tab.handle_input('\x10')
        for ch in "461234":
            tab.handle_input(ch)
        assert tab.handle_input('\r')
        assert tab._input_mode == INPUT_PHONE and tab._input_text == "+461234"
        tab._api.send_code_request.assert_not_called()
        tab._input_mode, tab._input_text = INPUT_CODE, "1234"
        assert tab.handle_input('\r')
        assert tab._input_mode == INPUT_CODE
        tab._api.sign_in.assert_not_called()

    def test_the_phone_stops_growing_at_its_maximum(self):
        tab = self.make()
        tab.handle_input('\x10')
        for ch in "1" * 30:
            tab.handle_input(ch)
        assert len(tab._input_text) == 16


@pytest.mark.unit
class TestBackgroundCheck:
    """The tab learns it is connected from the periodic check and from being opened; every
    draw test above disables the former, so pin both here."""

    def test_a_stale_status_is_rechecked_on_the_next_frame_and_on_activation(self):
        tab = make_tab()
        started = []
        tab._run = started.append
        tab._last_check_time = time.time() - AuthTab.CHECK_INTERVAL - 1
        tab.draw(TUIScreen(100, 30), 0)
        assert len(started) == 1 and tab._checking is True
        tab.draw(TUIScreen(100, 30), 1)          # one check at a time
        assert len(started) == 1
        started[0]()
        assert tab._checking is False
        tab.draw(TUIScreen(100, 30), 2)          # fresh now
        assert len(started) == 1
        tab.on_activate()
        assert len(started) == 2

    def test_a_check_that_raises_releases_the_lock(self):
        tab = make_tab()
        tab._api.get_auth_status.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            tab._check_auth_status_async()
        assert tab._checking is False


@pytest.mark.unit
class TestSubclassStatusLabels:
    @pytest.mark.parametrize("cls, status, label", [
        (iCloudTab, "partial_auth", iCloudTab.STATUS_LABELS["partial_auth"][0]),
        (iCloudTab, "needs_2fa", iCloudTab.STATUS_LABELS["needs_2fa"][0]),
        (iCloudTab, "authenticated", AuthTab.STATUS_LABELS["authenticated"][0]),
        (TelegramTab, "needs_auth", TelegramTab.STATUS_LABELS["needs_auth"][0]),
        (TelegramTab, "error", AuthTab.STATUS_LABELS["error"][0]),
    ])
    def test_own_and_inherited_labels_reach_the_connection_tile(self, cls, status, label):
        tab = cls()
        tab._api = Mock(auto_2fa_message=None)
        tab._auth_status = {"status": status, "message": ""}
        assert label in row_with(rows(draw(tab, width=120)), "Connection")


@pytest.mark.unit
class TestAutoTwoFactorHook:
    @pytest.mark.parametrize("cls", [iCloudTab, RingTab])
    def test_the_integration_verifying_a_code_itself_ends_manual_entry(self, cls):
        api = Mock()
        api.auto_2fa_message = "2FA auto-verified from iMessage"
        api.get_auth_status.return_value = {"status": "authenticated", "message": ""}
        tab = cls()
        tab._api = api
        tab._run = lambda fn: fn()
        tab._auth_status = {"status": "needs_2fa", "message": ""}
        tab._input_mode, tab._input_text = True, "123"
        tab._before_draw()
        assert tab._status_message == "2FA auto-verified from iMessage"
        assert tab._input_mode is False and tab._input_text == ""
        assert tab.status == "authenticated"


@pytest.mark.unit
class TestTelegramLoginLifecycle:
    def make(self):
        tab = TelegramTab()
        tab._api = Mock()
        tab._run = lambda fn: fn()
        return tab

    def test_escape_during_the_code_entry_aborts_the_login(self):
        tab = self.make()
        tab._input_mode, tab._input_text = INPUT_CODE, "12"
        assert tab.handle_input('escape')
        assert tab._input_mode is False
        tab._api.abort_login.assert_called_once_with()

    def test_the_code_request_reports_where_the_code_went(self):
        tab = self.make()
        tab._api.code_destination.return_value = "by SMS"
        tab.handle_input('\x10')
        for ch in "46123456789":
            tab.handle_input(ch)
        tab.handle_input('\r')
        tab._api.code_destination.assert_called_once_with(tab._api.send_code_request.return_value)
        assert tab._status_message == "Code sent by SMS — waiting for it."
        assert tab._outcome_color(tab._status_message) == COLOR_YELLOW

    def test_a_sign_in_restarts_the_integration(self):
        tab = self.make()
        tab._phone_number, tab._input_mode, tab._input_text = "+1", INPUT_CODE, "1234"
        tab.handle_input('5')
        tab._api.sign_in.assert_called_once_with("+1", "12345")
        tab._api.restart_feed.assert_called_once_with()
        assert tab._status_message == "Authentication successful!"
