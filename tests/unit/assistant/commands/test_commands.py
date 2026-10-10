import threading
import time
from collections import deque
from unittest.mock import Mock, call, patch, MagicMock

import pytest

from assistant.ChannelProfile import profile_for
from assistant.commands.CommandParser import CommandParser
from assistant.commands.Commands import Commands
from assistant.tools.WebAccess import WebAccess


@pytest.fixture(autouse=True)
def mock_email_provider():
    with patch.object(Commands, '_mail_access', return_value=Mock()) as mock:
        yield mock


@pytest.fixture
def commands_instance():
    mock_cp = Mock()
    mock_cp.background_command_start = Mock()
    mock_cp.background_command_done = Mock()
    mock_cp.set_command_reply = Mock()
    # Match a fresh CommandProcessor — a bare Mock's truthy attributes would make
    # _publish_bg_results treat the reply as a pending error and fold it in.
    mock_cp.command_reply = None
    mock_cp.command_reply_contains_error = False
    return Commands(mock_cp)


@pytest.fixture(autouse=True)
def mock_home():
    # The house commands run against HomeAutomation.active(), resolved per call.
    backend = Mock(name="home_backend")
    for method in ("turn_on", "turn_off", "dim", "execute"):
        getattr(backend, method).return_value = ""
    with patch("assistant.commands.Commands.home.active", return_value=backend):
        yield backend


@pytest.fixture(autouse=True)
def mock_web_access():
    with patch.object(WebAccess, "get_instance", return_value=Mock()) as get_instance:
        yield get_instance.return_value


@pytest.fixture(autouse=True)
def reset_false_positive_timestamps():
    Commands.false_positive_timestamps = deque(maxlen=10)


# ---------------------------------------------------------------------------
# TestRun
# ---------------------------------------------------------------------------

class TestRun:
    def test_dispatches_to_cmd_method(self, commands_instance):
        commands_instance.cmd_turn_on = Mock(return_value="ok")
        result = commands_instance.run("turn_on", ["kitchen"])
        commands_instance.cmd_turn_on.assert_called_once_with("kitchen")
        assert result == "ok"

    def test_dim_zero_converts_to_turn_off(self, commands_instance):
        commands_instance.cmd_turn_off = Mock(return_value="")
        commands_instance.run("dim", ["kitchen", "0"])
        commands_instance.cmd_turn_off.assert_called_once_with("kitchen")

    def test_dim_nonzero_stays_dim(self, commands_instance):
        commands_instance.cmd_dim = Mock(return_value="")
        commands_instance.run("dim", ["light", "50"])
        commands_instance.cmd_dim.assert_called_once_with("light", "50")

    def test_catches_exception(self, commands_instance):
        commands_instance.cmd_turn_on = Mock(side_effect=RuntimeError("boom"))
        result = commands_instance.run("turn_on", ["kitchen"])
        assert "boom" in result

    def test_exception_without_a_message_is_still_a_failure(self, commands_instance):
        # "" means the command ran — a bare TimeoutError() must not read as success.
        commands_instance.cmd_turn_on = Mock(side_effect=TimeoutError())
        result = commands_instance.run("turn_on", ["kitchen"])
        assert result and "TimeoutError" in result


# ---------------------------------------------------------------------------
# TestCmdFalsePositive
# ---------------------------------------------------------------------------

class TestCmdFalsePositive:
    def test_single_no_flood(self, commands_instance):
        with patch.object(Commands, "report_fp_flood") as mock_flood:
            commands_instance.cmd_false_positive("")
            mock_flood.assert_not_called()

    def test_two_within_window_triggers_flood(self, commands_instance):
        with patch.object(Commands, "report_fp_flood") as mock_flood:
            commands_instance.cmd_false_positive("")
            commands_instance.cmd_false_positive("")
            mock_flood.assert_called_once()


# ---------------------------------------------------------------------------
# TestListCommands
# ---------------------------------------------------------------------------

class TestListCommands:
    def test_returns_all_commands(self, commands_instance):
        cmds = commands_instance.list_commands()
        assert isinstance(cmds, list)
        assert "turn_on" in cmds
        assert "turn_off" in cmds
        assert "dim" in cmds
        assert "false_positive" in cmds
        assert "browse_screenshot" in cmds
        assert "browse_snapshot" in cmds
        assert "browse_click_at" in cmds
        for c in cmds:
            assert not c.startswith("cmd_")


class TestBrowseSnapshotAndClickAt:
    def test_click_at_rejoins_positional_coordinates(self, commands_instance):
        commands_instance.web_access.browse_click_at = Mock(return_value="Clicked")
        commands_instance.cmd_browse_click_at("412", " 583")
        commands_instance.wait_for_background(timeout=5)
        commands_instance.web_access.browse_click_at.assert_called_once_with("412,583")
        assert "browse_click_at" in Commands.MULTI_PARAM_COMMANDS
        assert commands_instance.describe("browse_click_at", "412, 583") == "Clicking at 412, 583"


class TestBrowseWaitAndNetwork:
    def test_wait_rejoins_positional_args_so_a_comma_in_the_pattern_survives(self, commands_instance):
        commands_instance.web_access.browse_wait = Mock(return_value="MATCHED")
        commands_instance.cmd_browse_wait("pending", " complete", " 45")
        commands_instance.wait_for_background(timeout=5)
        spec = commands_instance.web_access.browse_wait.call_args.args[0]
        assert WebAccess._parse_wait_spec(spec) == ("pending, complete", 45.0)
        assert "browse_wait" in Commands.MULTI_PARAM_COMMANDS

    def test_wait_description_names_the_pattern_and_the_timeout(self, commands_instance):
        text = commands_instance.describe("browse_wait", ["callback|thanks", "45"])
        assert "callback|thanks" in text and "45" in text
        assert commands_instance.describe("browse_wait", "") is not None

    def test_wait_and_network_run_on_the_browser_lane(self, commands_instance):
        with patch.object(commands_instance, "background_command") as bg:
            commands_instance.cmd_browse_wait("done", "10")
            commands_instance.cmd_browse_network("pay\\.example")
        targets = [(call.args[0], call.kwargs.get("lane")) for call in bg.call_args_list]
        assert targets == [(commands_instance.web_access.browse_wait, Commands.BROWSER_LANE),
                           (commands_instance.web_access.browse_network, Commands.BROWSER_LANE)]


class TestSendScreenshot:
    def test_voice_mode_refuses(self, commands_instance):
        commands_instance._current_profile = profile_for("VoiceToVoice")
        result = commands_instance._run_send_screenshot("look at this")
        assert "voice mode" in result

    def test_no_screenshot_cached(self, commands_instance):
        commands_instance.web_access.get_last_screenshot = Mock(return_value=(None, None))
        result = commands_instance._run_send_screenshot("")
        assert "no screenshot available" in result

    def test_no_notifier_reports_missing_channel(self, commands_instance):
        # No chat interface installed: the NullNotifier drops the photo and the LLM is told.
        commands_instance.web_access.get_last_screenshot = Mock(return_value=(b"png", "/tmp/x.png"))
        with patch('assistant.tools.ImageAnalyzer.resize_to_jpeg', return_value=b"jpeg"):
            result = commands_instance._run_send_screenshot("")
        assert "no notification channel configured" in result

    def test_notifier_send_failure_reports_missing_channel(self, commands_instance):
        commands_instance.web_access.get_last_screenshot = Mock(return_value=(b"png", "/tmp/x.png"))
        bot = Mock()
        bot.send_photo.return_value = False
        with patch('assistant.Notifier._notifier', bot), \
             patch('assistant.tools.ImageAnalyzer.resize_to_jpeg', return_value=b"jpeg"):
            result = commands_instance._run_send_screenshot("")
        assert "no notification channel configured" in result

    def test_happy_path_sends_jpeg_with_caption(self, commands_instance):
        commands_instance.web_access.get_last_screenshot = Mock(return_value=(b"png_bytes", "/tmp/x.png"))
        mock_telegram = Mock()
        with patch('assistant.Notifier._notifier', mock_telegram), \
             patch('assistant.tools.ImageAnalyzer.resize_to_jpeg', return_value=b"jpeg_bytes") as mock_resize:
            result = commands_instance._run_send_screenshot("  here you go  ")
        mock_resize.assert_called_once_with(b"png_bytes", max_dim=2048)
        mock_telegram.send_photo.assert_called_once_with(b"jpeg_bytes", caption="here you go")
        assert result == ""

    def test_the_photo_answers_the_chat_the_request_came_from(self, commands_instance):
        """The route is captured at dispatch and travels with the background command, so
        the photo lands in the requester's chat and not on the proactive notification route."""
        commands_instance.web_access.get_last_screenshot = Mock(return_value=(b"png_bytes", "/tmp/x.png"))
        commands_instance._current_context = Mock(source="TextChat", conversation=222)
        route = commands_instance._request_route()
        assert route == {"source": "TextChat", "conversation": 222}
        mock_telegram = Mock()
        with patch('assistant.Notifier._notifier', mock_telegram), \
             patch('assistant.tools.ImageAnalyzer.resize_to_jpeg', return_value=b"jpeg_bytes"):
            assert commands_instance._run_send_screenshot("cap", route=route) == ""
        mock_telegram.send_photo.assert_called_once_with(b"jpeg_bytes", caption="cap", source="TextChat", conversation=222)

    def test_no_request_means_no_route(self, commands_instance):
        commands_instance._current_context = None
        assert commands_instance._request_route() is None

    def test_resize_failure_falls_back_to_png(self, commands_instance):
        commands_instance.web_access.get_last_screenshot = Mock(return_value=(b"png_bytes", "/tmp/x.png"))
        mock_telegram = Mock()
        with patch('assistant.Notifier._notifier', mock_telegram), \
             patch('assistant.tools.ImageAnalyzer.resize_to_jpeg', return_value=None):
            commands_instance._run_send_screenshot("")
        mock_telegram.send_photo.assert_called_once_with(b"png_bytes", caption="")

# ---------------------------------------------------------------------------
# TestBackgroundCommand
# ---------------------------------------------------------------------------

class TestBackgroundCommand:
    def test_queues_and_executes(self, commands_instance):
        fn = Mock(return_value="result_data")
        commands_instance.background_command(fn, "params")
        commands_instance.wait_for_background(timeout=5)
        fn.assert_called_once_with("params")
        commands_instance.command_processor.set_command_reply.assert_called_once_with("result_data", contains_error=False)
        commands_instance.command_processor.background_command_done.assert_called_once()

    def test_worker_exception_publishes_with_error_flag(self, commands_instance):
        fn = Mock(side_effect=RuntimeError("boom"))
        commands_instance.background_command(fn, "params")
        commands_instance.wait_for_background(timeout=5)
        commands_instance.command_processor.set_command_reply.assert_called_once_with(
            "Command failed: boom", contains_error=True)

    def test_publish_leaves_the_pending_sync_reply_to_the_processor(self, commands_instance):
        """A sync command error appended earlier this round is the processor's own part of
        the reply — the publish hands over the batch alone, or the processor's composition
        would carry the error (and, on a later publish, the earlier results) twice."""
        cp = commands_instance.command_processor
        cp.command_reply = "Command 'send_email' was denied by user."
        cp.command_reply_contains_error = True

        fn = Mock(return_value="weather data")
        commands_instance.background_command(fn, "params")
        commands_instance.wait_for_background(timeout=5)

        cp.set_command_reply.assert_called_once_with("weather data", contains_error=False)

    def test_forget_batch_results_starts_the_next_publish_from_nothing(self, commands_instance):
        cp = commands_instance.command_processor
        commands_instance.background_command(Mock(side_effect=RuntimeError("boom")), None)
        assert commands_instance.wait_for_background(timeout=5)
        commands_instance.forget_batch_results()

        commands_instance.background_command(Mock(return_value="two"), None)
        assert commands_instance.wait_for_background(timeout=5)
        cp.set_command_reply.assert_called_with("two", contains_error=False)


# ---------------------------------------------------------------------------
# TestBackgroundWorkerReplies
# ---------------------------------------------------------------------------

class TestBackgroundWorkerReplies:
    def test_bg_worker_skips_reply_on_all_empty(self, commands_instance):
        fn = Mock(return_value="")
        commands_instance.background_command(fn, "params")
        commands_instance.wait_for_background(timeout=5)

        commands_instance.command_processor.set_command_reply.assert_not_called()
        commands_instance.command_processor.background_command_done.assert_called_once()

    def test_bg_worker_filters_empty_results(self, commands_instance):
        fn_empty = Mock(return_value="")
        fn_data = Mock(return_value="weather data")

        commands_instance.background_command(fn_empty, None)
        commands_instance.background_command(fn_data, None)
        assert commands_instance.wait_for_background(timeout=5)

        commands_instance.command_processor.set_command_reply.assert_called_once_with("weather data", contains_error=False)


# ---------------------------------------------------------------------------
# TestBackgroundLanes — commands of one reply run concurrently, results keep
# the order the LLM emitted them in, and the browser lane stays serial.
# ---------------------------------------------------------------------------

class _Gate:
    """A command that blocks until released and records when it started/finished."""

    def __init__(self, name: str, log: list, lock: threading.Lock):
        self.name = name
        self.log = log
        self.lock = lock
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, _params):
        with self.lock:
            self.log.append(("start", self.name))
        self.started.set()
        assert self.release.wait(timeout=5), f"{self.name} was never released"
        with self.lock:
            self.log.append(("end", self.name))
        return f"{self.name} result"


class TestBackgroundLanes:

    @pytest.fixture
    def journal(self):
        return [], threading.Lock()

    def test_io_lane_commands_overlap(self, commands_instance, journal):
        log, lock = journal
        first, second = _Gate("first", log, lock), _Gate("second", log, lock)
        commands_instance.background_command(first, None)
        commands_instance.background_command(second, None)

        # Both are running before either has finished — that is the whole point.
        assert first.started.wait(timeout=2)
        assert second.started.wait(timeout=2)
        assert commands_instance.background_outstanding() == 2
        commands_instance.command_processor.background_command_done.assert_not_called()

        first.release.set()
        second.release.set()
        assert commands_instance.wait_for_background(timeout=5)
        assert log[:2] == [("start", "first"), ("start", "second")]

    def test_results_publish_in_emission_order_not_completion_order(self, commands_instance, journal):
        log, lock = journal
        slow, fast = _Gate("slow", log, lock), _Gate("fast", log, lock)
        commands_instance.background_command(slow, None)
        commands_instance.background_command(fast, None)
        assert slow.started.wait(timeout=2) and fast.started.wait(timeout=2)

        fast.release.set()
        # Let the fast one finish first while the slow one is still held.
        deadline = threading.Event()
        for _ in range(50):
            with lock:
                if ("end", "fast") in log:
                    break
            deadline.wait(0.02)
        slow.release.set()
        assert commands_instance.wait_for_background(timeout=5)

        commands_instance.command_processor.set_command_reply.assert_called_once_with(
            "slow result\n\nfast result", contains_error=False)

    def test_lifecycle_fires_once_around_the_whole_batch(self, commands_instance, journal):
        log, lock = journal
        gates = [_Gate(f"g{i}", log, lock) for i in range(3)]
        for g in gates:
            commands_instance.background_command(g, None)
        for g in gates:
            assert g.started.wait(timeout=2)
        cp = commands_instance.command_processor
        cp.background_command_start.assert_called_once()

        gates[1].release.set()
        gates[2].release.set()
        for _ in range(50):
            if commands_instance.background_outstanding() == 1:
                break
            threading.Event().wait(0.02)
        cp.background_command_done.assert_not_called()
        cp.set_command_reply.assert_not_called()

        gates[0].release.set()
        assert commands_instance.wait_for_background(timeout=5)
        cp.background_command_done.assert_called_once()
        cp.set_command_reply.assert_called_once()

    def test_one_failure_keeps_the_others_results(self, commands_instance):
        # Held until all three are in: an instant command would finish (and publish a batch
        # of its own) before the next one is submitted.
        submitted = threading.Event()

        def fine(_):
            submitted.wait(timeout=2)
            return "fine"

        def boom(_):
            submitted.wait(timeout=2)
            raise RuntimeError("boom")

        commands_instance.background_command(fine, None)
        commands_instance.background_command(boom, None)
        commands_instance.background_command(fine, None)
        submitted.set()
        assert commands_instance.wait_for_background(timeout=5)

        commands_instance.command_processor.set_command_reply.assert_called_once_with(
            "fine\n\nCommand failed: boom\n\nfine", contains_error=True)

    def test_browser_lane_runs_strictly_in_order(self, commands_instance, journal):
        log, lock = journal
        first, second = _Gate("first", log, lock), _Gate("second", log, lock)
        commands_instance.background_command(first, None, lane=Commands.BROWSER_LANE)
        commands_instance.background_command(second, None, lane=Commands.BROWSER_LANE)

        assert first.started.wait(timeout=2)
        # The second one must not start while the first still holds the tab.
        assert not second.started.wait(timeout=0.05)
        first.release.set()
        assert second.started.wait(timeout=2)
        second.release.set()
        assert commands_instance.wait_for_background(timeout=5)
        assert log == [("start", "first"), ("end", "first"), ("start", "second"), ("end", "second")]

    def test_browser_lane_does_not_hold_up_the_io_lane(self, commands_instance, journal):
        log, lock = journal
        page, search = _Gate("page", log, lock), _Gate("search", log, lock)
        commands_instance.background_command(page, None, lane=Commands.BROWSER_LANE)
        commands_instance.background_command(search, None)

        assert page.started.wait(timeout=2)
        assert search.started.wait(timeout=2)
        search.release.set()
        page.release.set()
        assert commands_instance.wait_for_background(timeout=5)
        commands_instance.command_processor.set_command_reply.assert_called_once_with(
            "page result\n\nsearch result", contains_error=False)

    def test_config_lane_runs_one_at_a_time_in_order(self, commands_instance, journal):
        log, lock = journal
        gates = [_Gate(f"set{i}", log, lock) for i in range(3)]
        for gate in gates:
            commands_instance.background_command(gate, None, lane=Commands.CONFIG_LANE)
        assert gates[0].started.wait(timeout=2)
        assert not gates[1].started.wait(timeout=0.05)
        for gate in gates:
            gate.release.set()
            assert gate.started.wait(timeout=2)
        assert commands_instance.wait_for_background(timeout=5)
        assert log == [(e, f"set{i}") for i in range(3) for e in ("start", "end")]

    def test_config_lane_does_not_wait_for_the_browser_lane(self, commands_instance, journal):
        log, lock = journal
        page, setting = _Gate("page", log, lock), _Gate("setting", log, lock)
        commands_instance.background_command(page, None, lane=Commands.BROWSER_LANE)
        commands_instance.background_command(setting, None, lane=Commands.CONFIG_LANE)
        assert page.started.wait(timeout=2)
        assert setting.started.wait(timeout=2)
        setting.release.set()
        page.release.set()
        assert commands_instance.wait_for_background(timeout=5)

    def test_config_commands_use_the_config_lane(self, commands_instance):
        commands_instance._current_context = Mock(source="Console", conversation="tui")
        with patch.object(commands_instance, "background_command") as bg, \
             patch.object(ConfigStore, "check_key"):
            commands_instance.cmd_config_set("owner_name", "Anna")
            commands_instance.cmd_config_request("email.password")
            commands_instance.cmd_restart_integration("Gmail")
        assert len(bg.call_args_list) == 3
        for call in bg.call_args_list:
            assert call.kwargs.get("lane") == Commands.CONFIG_LANE, call

    def test_browser_commands_use_the_browser_lane(self, commands_instance):
        with patch.object(commands_instance, "background_command") as bg:
            commands_instance.cmd_open_url("https://x")
            commands_instance.cmd_browse_open("https://x")
            commands_instance.cmd_browse_act("3")
            commands_instance.cmd_browse_extract("")
            commands_instance.cmd_browse_screenshot("")
            commands_instance.cmd_send_screenshot("caption")
        for call in bg.call_args_list:
            assert call.kwargs.get("lane") == Commands.BROWSER_LANE, call

    def test_each_command_completes_its_own_echo(self, commands_instance, journal):
        log, lock = journal
        first, second = _Gate("first", log, lock), _Gate("second", log, lock)
        echoes = []
        with patch("assistant.commands.Commands.notifier") as mock_notifier:
            mock_notifier.return_value.complete_command_notification.side_effect = \
                lambda handle, detail: echoes.append((handle, detail))
            commands_instance._current_echo = "echo-1"
            commands_instance.background_command(first, None)
            commands_instance._current_echo = "echo-2"
            commands_instance.background_command(second, None)
            second.release.set()
            first.release.set()
            assert commands_instance.wait_for_background(timeout=5)
        assert sorted(h for h, _ in echoes) == ["echo-1", "echo-2"]
        assert all("done" in d for _, d in echoes)

    def test_submit_after_publish_starts_a_new_batch_keeping_the_pending_reply(self, commands_instance):
        cp = commands_instance.command_processor
        commands_instance.background_command(Mock(return_value="one"), None)
        assert commands_instance.wait_for_background(timeout=5)
        cp.set_command_reply.assert_called_once_with("one", contains_error=False)
        # The reply is still unconsumed: a real CommandProcessor would hold it.
        cp.command_reply = "one"

        commands_instance.background_command(Mock(return_value="two"), None)
        assert commands_instance.wait_for_background(timeout=5)
        assert cp.background_command_start.call_count == 2
        assert cp.background_command_done.call_count == 2
        cp.set_command_reply.assert_called_with("one\n\ntwo", contains_error=False)



# ---------------------------------------------------------------------------
# TestCmdHomeAutomationLogs
# ---------------------------------------------------------------------------

class TestCmdHomeAutomationLogs:
    def test_fetch_returns_the_backends_error_log_under_heading(self, mock_home):
        mock_home.error_logs.return_value = "indigo err"
        result = Commands._fetch_home_automation_logs(None)
        assert result.startswith("# Home Automation Error Logs")
        assert "indigo err" in result
        assert "Server Logs" not in result


# ---------------------------------------------------------------------------
# TestCmdServerLogs
# ---------------------------------------------------------------------------

class TestCmdServerLogs:
    def test_fetch_returns_log_tail(self, tmp_path, mock_home):
        log_file = tmp_path / "floxy.out"
        log_file.write_text("line1\nline2\n")
        with patch("assistant.commands.Commands.SERVER_LOG_PATH", str(log_file)):
            result = Commands._fetch_server_logs(None)
        assert result.startswith("# Keep AI Server Logs")
        assert "line1" in result and "line2" in result
        mock_home.error_logs.assert_not_called()

    def test_tail_is_limited_to_400_lines(self, tmp_path):
        from assistant.commands.Commands import SERVER_LOG_TAIL_LINES
        assert SERVER_LOG_TAIL_LINES == 400
        log_file = tmp_path / "floxy.out"
        log_file.write_text("".join(f"line{i}\n" for i in range(500)))
        with patch("assistant.commands.Commands.SERVER_LOG_PATH", str(log_file)):
            result = Commands._fetch_server_logs(None)
        assert "line99\n" not in result
        assert "line100\n" in result
        assert "line499\n" in result

    def test_missing_log_file(self, tmp_path):
        with patch("assistant.commands.Commands.SERVER_LOG_PATH", str(tmp_path / "nope.out")):
            result = Commands._fetch_server_logs(None)
        assert "Log file not found" in result

    def test_a_stray_non_utf8_byte_does_not_break_the_command(self, tmp_path):
        # floxy.out is redirected process output; a subprocess can write anything into it.
        log_file = tmp_path / "floxy.out"
        log_file.write_bytes(b"line1\n\xff\xfe garbage\nline3\n")
        with patch("assistant.commands.Commands.SERVER_LOG_PATH", str(log_file)):
            result = Commands._fetch_server_logs(None)
        assert "line1\n" in result and "line3\n" in result

    def test_ansi_codes_stripped(self, tmp_path):
        log_file = tmp_path / "floxy.out"
        log_file.write_text("\x1b[31mred text\x1b[0m\n")
        with patch("assistant.commands.Commands.SERVER_LOG_PATH", str(log_file)):
            result = Commands._fetch_server_logs(None)
        assert "\x1b[" not in result
        assert "red text" in result

    @staticmethod
    def _stamped_log(tmp_path, count=600):
        # One line per minute from 08:00; line i (0-based) is stamped 08:00 + i min.
        log_file = tmp_path / "floxy.out"
        log_file.write_text("".join(
            f"\x1b[2mI│2026-03-01 {8 + i // 60:02d}:{i % 60:02d}:30.000 +0.000 │ main │ line{i}\n" for i in range(count)))
        return patch("assistant.commands.Commands.SERVER_LOG_PATH", str(log_file))

    def test_from_time_returns_the_first_400_lines_at_or_after_it(self, tmp_path):
        with self._stamped_log(tmp_path):
            result = Commands._fetch_server_logs("2026-03-01 09:40")
        assert "line99\n" not in result
        assert "line100\n" in result
        assert "line499\n" in result
        assert "line500\n" not in result

    def test_from_time_with_seconds_and_t_separator(self, tmp_path):
        # 09:40:30 is line 100's own stamp; 09:40:31 is past it
        with self._stamped_log(tmp_path):
            at = Commands._fetch_server_logs("2026-03-01T09:40:30")
            after = Commands._fetch_server_logs("2026-03-01 09:40:31")
        assert "line100\n" in at
        assert "line100\n" not in after and "line101\n" in after

    def test_from_date_alone_means_midnight(self, tmp_path):
        with self._stamped_log(tmp_path):
            result = Commands._fetch_server_logs("2026-03-01")
        assert "line0\n" in result

    def test_from_clock_time_means_today(self, tmp_path):
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = tmp_path / "floxy.out"
        log_file.write_text(f"I│{today} 00:00:01.000 │ early\nI│{today} 23:59:58.000 │ late\n")
        with patch("assistant.commands.Commands.SERVER_LOG_PATH", str(log_file)):
            result = Commands._fetch_server_logs("23:59")
        assert "late" in result and "early" not in result

    def test_continuation_lines_without_a_stamp_are_skipped_when_seeking_but_kept_in_the_window(self, tmp_path):
        log_file = tmp_path / "floxy.out"
        log_file.write_text("I│2026-03-01 08:00:00.000 │ before\nTraceback (most recent call last):\n"
                            "I│2026-03-01 09:00:00.000 │ after\n  File x, line 1\n")
        with patch("assistant.commands.Commands.SERVER_LOG_PATH", str(log_file)):
            result = Commands._fetch_server_logs("2026-03-01 08:30")
        assert "before" not in result and "Traceback" not in result
        assert "after\n  File x, line 1\n" in result

    def test_from_time_past_the_end_says_nothing_logged(self, tmp_path):
        with self._stamped_log(tmp_path, count=5):
            result = Commands._fetch_server_logs("2026-03-02 00:00")
        assert result.startswith("# Keep AI Server Logs")
        assert "line0" not in result
        assert "2026-03-02 00:00:00" in result

    def test_unreadable_time_fails_without_touching_the_log(self, tmp_path):
        with patch("assistant.commands.Commands.SERVER_LOG_PATH", str(tmp_path / "nope.out")):
            result = Commands._fetch_server_logs("yesterday-ish")
        assert result.startswith("server_logs failed")
        assert "yesterday-ish" in result

    def test_command_hands_the_first_argument_or_none_to_the_fetch(self, commands_instance):
        with patch.object(commands_instance, 'background_command') as mock_bg:
            commands_instance.cmd_server_logs("")
            mock_bg.assert_called_once_with(commands_instance._fetch_server_logs, None)
            mock_bg.reset_mock()
            commands_instance.cmd_server_logs('"2026-03-01 09:40"')
            mock_bg.assert_called_once_with(commands_instance._fetch_server_logs, "2026-03-01 09:40")


# ---------------------------------------------------------------------------
# TestCmdEmailCommands
# ---------------------------------------------------------------------------

class TestCmdEmailCommands:
    def test_latest_email(self, commands_instance):
        with patch.object(commands_instance, 'background_command') as mock_bg:
            commands_instance.cmd_latest_email("@SaneLater")
            mock_bg.assert_called_once_with(
                commands_instance._email_provider.fetch_latest, ["@SaneLater", False])

    def test_search_email(self, commands_instance):
        with patch.object(commands_instance, 'background_command') as mock_bg:
            commands_instance.cmd_search_email("subject:hello")
            mock_bg.assert_called_once_with(commands_instance._email_provider.search, "subject:hello")

    def test_read_email(self, commands_instance):
        with patch.object(commands_instance, 'background_command') as mock_bg:
            commands_instance.cmd_read_email("123")
            mock_bg.assert_called_once_with(commands_instance._email_provider.read, "123")

    def test_latest_email_unread_flag(self, commands_instance):
        with patch.object(commands_instance, 'background_command') as mock_bg:
            commands_instance.cmd_latest_email("@SaneLater", "unread")
            assert mock_bg.call_args[0][1] == ["@SaneLater", True]

    def test_latest_email_unread_flag_truthy_variants(self, commands_instance):
        for token in ("unread", "true", "1", "yes", '"unread"'):
            with patch.object(commands_instance, 'background_command') as mock_bg:
                commands_instance.cmd_latest_email("@SaneLater", token)
                assert mock_bg.call_args[0][1] == ["@SaneLater", True], token

    def test_latest_email_no_args_defaults(self, commands_instance):
        with patch.object(commands_instance, 'background_command') as mock_bg:
            commands_instance.cmd_latest_email()
            assert mock_bg.call_args[0][1] == ["", False]

    def test_describe_read_email_shows_subject(self, commands_instance):
        commands_instance._email_provider.subject_for.return_value = "Order confirmation"
        assert commands_instance.describe_read_email("GM-AA") == "Reading email 'Order confirmation'"

    def test_describe_read_email_truncates_long_subject(self, commands_instance):
        commands_instance._email_provider.subject_for.return_value = "x" * 80
        result = commands_instance.describe_read_email("GM-AA")
        assert result == "Reading email '" + "x" * 50 + "…'"

    def test_describe_read_email_falls_back_on_cache_miss(self, commands_instance):
        commands_instance._email_provider.subject_for.return_value = None
        assert commands_instance.describe_read_email("GM-AA") == "Reading email…"

    def test_describe_read_email_never_leaks_id(self, commands_instance):
        commands_instance._email_provider.subject_for.return_value = None
        assert "GM-AA" not in commands_instance.describe_read_email("GM-AA")

    def test_describe_read_email_ignores_no_subject_placeholder(self, commands_instance):
        commands_instance._email_provider.subject_for.return_value = "(no subject)"
        assert commands_instance.describe_read_email("GM-AA") == "Reading email…"

    def test_describe_read_email_survives_lookup_error(self, commands_instance):
        commands_instance._email_provider.subject_for.side_effect = Exception("boom")
        assert commands_instance.describe_read_email("GM-AA") == "Reading email…"

    def test_describe_move_email_shows_subject_and_folder(self, commands_instance):
        commands_instance._email_provider.subject_for.return_value = "Order confirmation"
        assert commands_instance.describe_move_email("GM-AA, Archive") == "Moving email 'Order confirmation' to Archive"
        commands_instance._email_provider.subject_for.assert_called_once_with("GM-AA")

    def test_describe_move_email_falls_back_on_cache_miss(self, commands_instance):
        commands_instance._email_provider.subject_for.return_value = None
        assert commands_instance.describe_move_email("GM-AA, Archive") == "Moving email to Archive"

    def test_describe_move_email_no_id_skips_lookup(self, commands_instance):
        assert commands_instance.describe_move_email("Archive") == "Moving email to Archive"
        commands_instance._email_provider.subject_for.assert_not_called()

# ---------------------------------------------------------------------------
# TestCmdWebAccess
# ---------------------------------------------------------------------------

class TestCmdWebAccess:
    def test_open_url(self, commands_instance):
        with patch.object(commands_instance, 'background_command') as mock_bg:
            commands_instance.cmd_open_url("https://example.com")
            mock_bg.assert_called_once_with(commands_instance.web_access.fetch_webpage, "https://example.com", lane=Commands.BROWSER_LANE)

    def test_google(self, commands_instance):
        with patch.object(commands_instance, 'background_command') as mock_bg:
            commands_instance.cmd_google("best pizza")
            mock_bg.assert_called_once_with(commands_instance.web_access.search, "best pizza")

    def test_browse_eval_dispatches_to_web_access(self, commands_instance):
        with patch.object(commands_instance, 'background_command') as mock_bg:
            js = "return document.title"
            commands_instance.cmd_browse_eval(js)
            mock_bg.assert_called_once_with(commands_instance.web_access.browse_eval, js, lane=Commands.BROWSER_LANE)

    def test_browse_key_dispatches_to_web_access(self, commands_instance):
        # Dispatch is name-based, so the registered-callable test alone can't catch
        # cmd_browse_key forwarding to the wrong WebAccess method.
        with patch.object(commands_instance, 'background_command') as mock_bg:
            commands_instance.cmd_browse_key("3,Enter")
            mock_bg.assert_called_once_with(commands_instance.web_access.browse_key, "3,Enter", lane=Commands.BROWSER_LANE)


# ---------------------------------------------------------------------------
# TestCmdRemember
# ---------------------------------------------------------------------------

class TestCmdRemember:
    def test_strips_whitespace(self, tmp_path, monkeypatch):
        mem_file = tmp_path / "memory" / "bank.md"
        monkeypatch.chdir(tmp_path)
        mock_cp = Mock()
        cmd = Commands(mock_cp)
        cmd.cmd_remember("  padded  ")
        content = mem_file.read_text()
        assert "padded" in content
        assert "  padded  " not in content

    def test_appends_to_existing(self, tmp_path, monkeypatch):
        mem_file = tmp_path / "memory" / "bank.md"
        mem_file.parent.mkdir()
        mem_file.write_text("existing\n")
        monkeypatch.chdir(tmp_path)
        mock_cp = Mock()
        cmd = Commands(mock_cp)
        cmd.cmd_remember("new note")
        content = mem_file.read_text()
        assert "existing" in content
        assert "new note" in content


# ---------------------------------------------------------------------------
# TestCmdScheduleAndCaffeinate
# ---------------------------------------------------------------------------

class TestCmdScheduleAndCaffeinate:
    def test_schedule_wakeup(self, commands_instance):
        with patch("assistant.commands.Commands.AutonomousActionManager") as mock_aam:
            mock_instance = Mock()
            mock_aam.get_instance.return_value = mock_instance
            commands_instance.cmd_schedule_wakeup("5.5")
            mock_instance.schedule.assert_called_once_with(5.6, "")

    def test_schedule_wakeup_carries_the_reason(self, commands_instance):
        with patch("assistant.commands.Commands.AutonomousActionManager") as mock_aam:
            mock_instance = Mock()
            mock_aam.get_instance.return_value = mock_instance
            commands_instance.cmd_schedule_wakeup("30", "turn off the TV")
            mock_instance.schedule.assert_called_once_with(30.1, "turn off the TV")

    def test_schedule_wakeup_rejoins_an_unquoted_reason_split_on_its_commas(self, commands_instance):
        with patch("assistant.commands.Commands.AutonomousActionManager") as mock_aam:
            mock_instance = Mock()
            mock_aam.get_instance.return_value = mock_instance
            commands_instance.cmd_schedule_wakeup("10", "check the T-0", "then message Marcus")
            mock_instance.schedule.assert_called_once_with(10.1, "check the T-0, then message Marcus")

    def test_describe_schedule_wakeup_shows_the_reason(self, commands_instance):
        text = commands_instance.describe("schedule_wakeup", ["20", "turn off the TV"])
        assert "20" in text and "turn off the TV" in text

# ---------------------------------------------------------------------------
# TestCmdSelfAnalysis
# ---------------------------------------------------------------------------

class TestCmdSelfAnalysis:
    def test_disabled_integration_answers_at_once(self, commands_instance):
        with patch("assistant.commands.Commands.CodeQualityMaintainer") as mock_cqm, \
             patch.object(commands_instance, "fire_and_forget") as fire:
            mock_cqm.unavailable_reason.return_value = "disabled"
            assert commands_instance.cmd_self_analysis("why?") == "disabled"
            assert commands_instance.cmd_self_improvement("fix") == "disabled"
        fire.assert_not_called()
        mock_cqm.analyze_code.assert_not_called()

    def test_parsed_request_reaches_the_agent_whole(self, commands_instance):
        """The request the LLM wrote, commas and all, is what Claude Code is asked to do:
        prod's run of 2026-10-08 got the words before the first comma and nothing else."""
        request = ("Marcus asked at 10:46 on Oct 8, 2026: the approval prompt shows no handle, "
                   "so he could not tell which Bob it was about to text. Fix: show it, with a test.")
        parser = CommandParser(["self_improvement"])
        (name, params), _ = parser.extract_command_and_params(f"self_improvement({request})")
        with patch("assistant.commands.Commands.CodeQualityMaintainer") as mock_cqm, \
             patch.object(commands_instance, "fire_and_forget") as fire:
            mock_cqm.unavailable_reason.return_value = ""
            commands_instance.run(name, params)
        fire.assert_called_once()
        assert fire.call_args.args[1] == request

    """The self-analysis command is fire-and-forget (background thread). We
    run the inner callable synchronously by intercepting fire_and_forget
    and immediately invoking the function it was handed, so the assertions
    don't depend on thread scheduling."""

    def _run_sync(self, commands_instance, params):
        captured = {}
        def sync(fn, p):
            captured['fn'] = fn
            fn(p)
        with patch.object(commands_instance, 'fire_and_forget', side_effect=sync):
            commands_instance.cmd_self_analysis(params)
        return captured

    def test_full_report_pushed_to_aam_history(self, commands_instance):
        # Pins a60239ef: the LLM history must receive the FULL analysis report,
        # not the short "[Self-analysis complete — report sent to user via Telegram]"
        # status note that 73d7ef2e introduced. In practice the AI follows up
        # better when it can see the findings, and a silent re-revert to the
        # short note hides everything behind a Telegram-only delivery.
        report = "## Findings\n- Bug X at file.py:42\n- Bug Y in foo()"
        with patch("assistant.commands.Commands.CodeQualityMaintainer") as mock_cqm, \
             patch("assistant.commands.Commands.AutonomousActionManager") as mock_aam:
            mock_cqm.unavailable_reason.return_value = None
            mock_cqm.analyze_code.return_value = report
            mock_aam_instance = Mock()
            mock_aam_instance.prompt_manager = Mock()
            mock_aam.get_instance.return_value = mock_aam_instance

            self._run_sync(commands_instance, "params describing the bug")

            mock_cqm.analyze_code.assert_called_once_with("params describing the bug")
            update_calls = mock_aam_instance.prompt_manager.update_history.call_args_list
            assert len(update_calls) == 1
            args, _ = update_calls[0]
            history_content = args[2]
            # The full report body must land in history — not just a status note.
            assert report in history_content, \
                f"AAM history must include full report, got: {history_content!r}"
            assert history_content.startswith("[Self-analysis complete]"), \
                "prefix tag is what the LLM keys off to recognize the entry"

    def test_telegram_delivery_still_happens_alongside_history(self, commands_instance):
        # The fix kept the Telegram delivery — both paths must fire. If a future
        # change drops Telegram by mistake, the user stops getting the message.
        mock_tb_instance = Mock()
        mock_tb_instance.markdown_to_html.return_value = "<p>Report body</p>"
        with patch("assistant.commands.Commands.CodeQualityMaintainer") as mock_cqm, \
             patch("assistant.commands.Commands.AutonomousActionManager") as mock_aam, \
             patch("assistant.Notifier._notifier", mock_tb_instance):
            mock_cqm.unavailable_reason.return_value = None
            mock_cqm.analyze_code.return_value = "Report body"
            mock_aam.get_instance.return_value = Mock(prompt_manager=Mock())

            self._run_sync(commands_instance, "params")

            mock_tb_instance.send_message.assert_called_once()
            sent = mock_tb_instance.send_message.call_args[0][0]
            assert "Self-analysis report" in sent
            assert "<p>Report body</p>" in sent

    def test_history_row_holds_the_report_in_the_chat_dialect(self, commands_instance):
        # The Home tab draws the history row, so it carries the converted report, not the
        # markdown, and the push is marked as a history row so the console gets no second copy.
        chat = Mock()
        chat.markdown_to_html.side_effect = lambda md: f"<converted>{md}</converted>"
        with patch("assistant.commands.Commands.CodeQualityMaintainer") as mock_cqm, \
             patch("assistant.commands.Commands.AutonomousActionManager") as mock_aam, \
             patch("assistant.Notifier._notifier", chat):
            mock_cqm.unavailable_reason.return_value = None
            mock_cqm.analyze_code.return_value = "## Findings"
            prompt_manager = Mock()
            mock_aam.get_instance.return_value = Mock(prompt_manager=prompt_manager)

            self._run_sync(commands_instance, "params")

            assert chat.send_message.call_args.kwargs == {"in_history": True}
            args, _ = prompt_manager.update_history.call_args
            assert args[2] == "[Self-analysis complete] <converted>## Findings</converted>"

# ---------------------------------------------------------------------------
# TestCmdFindPhone
# ---------------------------------------------------------------------------

class TestCmdFindPhone:
    def test_fire_and_forget_exception_logged(self, commands_instance):
        mock_fn = Mock(side_effect=RuntimeError("boom"))
        logged = threading.Event()
        with patch("assistant.commands.Commands.error", side_effect=lambda *a, **k: logged.set()) as mock_error:
            commands_instance.fire_and_forget(mock_fn, "params")
            assert logged.wait(timeout=2), "error() was not called in background thread"
            mock_error.assert_called_once()
            assert "boom" in str(mock_error.call_args)




class TestCmdSendEmail:
    @staticmethod
    def _accounts(mock_email_provider, *names):
        mock_email_provider.return_value.named.side_effect = (
            lambda name: Mock() if name.strip().strip("'\"").lower() in {n.lower() for n in names} else None)

    def test_returns_send_result_synchronously(self, commands_instance, mock_email_provider):
        self._accounts(mock_email_provider, "Personal")
        mock_email_provider.return_value.send.return_value = ""
        result = commands_instance.cmd_send_email("alice@example.com", "Hi", "Hello there")
        assert result == ""
        mock_email_provider.return_value.send.assert_called_once_with("alice@example.com", "Hi", "Hello there", "")

    def test_a_fourth_argument_naming_an_account_sends_from_it(self, commands_instance, mock_email_provider):
        self._accounts(mock_email_provider, "Personal", "Work")
        commands_instance.cmd_send_email("alice@example.com", "Hi", "Hello there", '"Work"')
        mock_email_provider.return_value.send.assert_called_once_with("alice@example.com", "Hi", "Hello there", "Work")

    def test_a_trailing_token_that_is_no_account_stays_part_of_the_body(self, commands_instance, mock_email_provider):
        self._accounts(mock_email_provider, "Personal", "Work")
        commands_instance.cmd_send_email("alice@example.com", "Hi", "Bring Bob", "Carol and Dave")
        mock_email_provider.return_value.send.assert_called_once_with("alice@example.com", "Hi", "Bring Bob, Carol and Dave", "")

    def test_email_body_carries_no_ai_footer(self, commands_instance, mock_email_provider):
        self._accounts(mock_email_provider)
        commands_instance.cmd_send_email("alice@example.com", "Hi", "Hello there")
        assert commands_instance.AI_MESSAGE_SUFFIX not in mock_email_provider.return_value.send.call_args[0][2]

    def test_missing_parts_returns_error_without_calling_send(self, commands_instance, mock_email_provider):
        result = commands_instance.cmd_send_email("alice@example.com, Subject")
        mock_email_provider.return_value.send.assert_not_called()
        assert "send_email failed" in result

    def test_empty_recipient_returns_error(self, commands_instance, mock_email_provider):
        self._accounts(mock_email_provider)
        result = commands_instance.cmd_send_email("", "Subject", "body")
        mock_email_provider.return_value.send.assert_not_called()
        assert "send_email failed" in result

    def test_send_email_is_sensitive(self, commands_instance):
        assert commands_instance.is_sensitive("send_email")

# ---------------------------------------------------------------------------
# TestCmdExecute
# ---------------------------------------------------------------------------

class TestCmdExecute:
    def test_delegates_to_the_backend_with_the_raw_name(self, commands_instance, mock_home):
        mock_home.execute.return_value = "ok"
        assert commands_instance.cmd_execute(" AC On [Gästhus] ") == "ok"
        mock_home.execute.assert_called_once_with("AC On (Gästhus)")

# ---------------------------------------------------------------------------
# TestReportFPFlood
# ---------------------------------------------------------------------------

class TestReportFPFlood:
    """The flood itself (wakeword broadcast + SpeechAnalyzer cooldown) lives in
    interfaces.voice.microphones.FalsePositiveFlood; Commands only reaches it through the
    hook the launcher installs. See test_false_positive_flood.py for the broadcast tests."""

    @pytest.fixture(autouse=True)
    def _no_reporter(self):
        saved = Commands.false_positive_reporter
        Commands.false_positive_reporter = None
        yield
        Commands.false_positive_reporter = saved

    def test_two_false_positives_within_window_report_a_flood(self, commands_instance):
        reporter = MagicMock()
        Commands.false_positive_reporter = reporter
        Commands.false_positive_timestamps.clear()
        commands_instance.cmd_false_positive("")
        reporter.assert_not_called()
        commands_instance.cmd_false_positive("")
        reporter.assert_called_once_with()


# ---------------------------------------------------------------------------
# TestCmdTurnOnOff
# ---------------------------------------------------------------------------

class TestCmdTurnOnOff:
    def test_turn_on(self, commands_instance, mock_home):
        assert commands_instance.cmd_turn_on("Desk lamp [Office]") == ""
        mock_home.turn_on.assert_called_once_with("Desk lamp (Office)")

    def test_dim(self, commands_instance, mock_home):
        commands_instance.cmd_dim("light", " 50% ")
        mock_home.dim.assert_called_once_with("light", 50)

    def test_stop_reaches_the_backend_with_the_raw_name(self, commands_instance, mock_home):
        mock_home.stop.return_value = ""
        assert commands_instance.cmd_stop("Markis [Uterum]") == ""
        mock_home.stop.assert_called_once_with("Markis (Uterum)")

    def test_dim_rejects_a_non_level(self, commands_instance, mock_home):
        assert commands_instance.cmd_dim("light", "bright") == "dim failed: 'bright' is not a level"
        mock_home.dim.assert_not_called()

    def test_the_not_configured_hint_comes_back_as_the_result(self, commands_instance):
        from assistant.integration.HomeAutomation import NotConfiguredHome
        with patch("assistant.commands.Commands.home.active", return_value=NotConfiguredHome("Enable Indigo.")):
            assert commands_instance.cmd_turn_on("Lamp") == "Enable Indigo."


# ---------------------------------------------------------------------------
# TestCmdFetchCameraImages
# ---------------------------------------------------------------------------

class TestCmdFetchCameraImages:
    """The command fans out over CameraProvider.active(); a provider is anything with the
    contract's two methods, the folders are never named."""

    @staticmethod
    def _provider(name, blocks=None, cameras=(), fail=None):
        provider = Mock(name=name)
        provider.name = name
        provider.camera_names.return_value = list(cameras)
        if fail:
            provider.fetch_snapshots.side_effect = fail
        else:
            provider.fetch_snapshots.return_value = list(blocks or [])
        return provider

    @staticmethod
    def _image(camera):
        return [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": camera.lower()}},
                {"type": "text", "text": f"Camera: {camera}"}]

    def test_combines_every_providers_results_under_one_header(self):
        arlo, ring = self._provider("Arlo", self._image("Backyard")), self._provider("Ring", self._image("Front Door"))
        with patch("assistant.commands.Commands.cameras.active", return_value=[arlo, ring]):
            result = Commands._fetch_all_camera_images(None)

        assert result[0]["type"] == "text" and "camera images" in result[0]["text"]
        texts = [b["text"] for b in result if b.get("type") == "text"]
        assert any("Backyard" in t for t in texts) and any("Front Door" in t for t in texts)
        assert sum(1 for b in result if b.get("type") == "image") == 2
        arlo.fetch_snapshots.assert_called_once_with(None)

    def test_a_failing_provider_does_not_hide_the_others(self):
        broken = self._provider("Arlo", fail=RuntimeError("Not connected to Arlo"))
        ring = self._provider("Ring", self._image("Front Door"))
        with patch("assistant.commands.Commands.cameras.active", return_value=[broken, ring]):
            result = Commands._fetch_all_camera_images(None)
        assert any("Front Door" in b.get("text", "") for b in result)

    def test_no_images_from_anyone_is_a_plain_answer(self):
        with patch("assistant.commands.Commands.cameras.active", return_value=[self._provider("Arlo", [])]):
            assert Commands._fetch_all_camera_images(None) == "No camera images available."

    def test_no_provider_ready_answers_with_the_setup_hint(self):
        with patch("assistant.commands.Commands.cameras.active", return_value=[]), \
             patch("assistant.commands.Commands.cameras.no_provider_hint", return_value="Arlo is disabled: enable it."):
            assert Commands._fetch_all_camera_images(None) == "Arlo is disabled: enable it."

    def test_the_named_cameras_reach_every_provider(self):
        arlo, ring = self._provider("Arlo", self._image("Front Garden")), self._provider("Ring", [])
        with patch("assistant.commands.Commands.cameras.active", return_value=[arlo, ring]):
            result = Commands._fetch_all_camera_images(["front"])
        arlo.fetch_snapshots.assert_called_once_with(["front"])
        ring.fetch_snapshots.assert_called_once_with(["front"])
        assert any("Front Garden" in b.get("text", "") for b in result)

    def test_a_name_nothing_matches_lists_the_cameras_there_are(self):
        arlo = self._provider("Arlo", [], cameras=["Backyard", "Garage"])
        with patch("assistant.commands.Commands.cameras.active", return_value=[arlo]):
            result = Commands._fetch_all_camera_images(["pool"])
        assert "pool" in result and "Backyard" in result and "Garage" in result

    def test_a_name_nothing_matches_with_no_camera_reachable_says_so_instead_of_an_empty_list(self):
        arlo, ring = self._provider("Arlo", [], cameras=[]), self._provider("Ring", [], cameras=[])
        with patch("assistant.commands.Commands.cameras.active", return_value=[arlo, ring]):
            result = Commands._fetch_all_camera_images(["pool"])
        assert "pool" in result and "reachable" in result
        assert "The cameras:" not in result

    def test_the_command_hands_the_trimmed_names_to_the_background_fetch(self, commands_instance):
        """background_command splats a list/tuple parameter (`function(*params)`), so the
        names must arrive at the fetch as one list — not as one name's characters, nor as two
        positional names."""
        with patch.object(commands_instance, "background_command") as bg, \
             patch.object(commands_instance, "_fetch_all_camera_images") as fetch:
            commands_instance.cmd_fetch_camera_images(" Front ", "", "Back")
            commands_instance.cmd_fetch_camera_images("Front")
            commands_instance.cmd_fetch_camera_images()
            for call in bg.call_args_list:
                function, params = call.args[:2]
                function(*params)
        assert [c.args for c in fetch.call_args_list] == [(["Front", "Back"],), (["Front"],), (None,)]


# ---------------------------------------------------------------------------
# TestCmdShowHistory
# ---------------------------------------------------------------------------

class TestCmdShowHistory:
    def test_valid_range_returns_formatted_entries(self):
        results = [
            {"content": "Hello world", "timestamp": 1714300000.0,
             "role": "user", "source": "VoiceToVoice", "location": "Kitchen"},
            {"content": "Hi there!", "timestamp": 1714300060.0,
             "role": "assistant", "source": None, "location": None},
        ]
        with patch('assistant.commands.Commands.FloxyDatabase') as mock_db_cls:
            mock_db_cls.get_instance.return_value.get_history_range.return_value = results
            result = Commands._run_show_history("2024-04-28 12:00", "2024-04-28 13:00")

        assert "user (VoiceToVoice)" in result
        assert "[Kitchen]" in result
        assert "Hello world" in result
        assert "assistant:" in result
        assert "Hi there!" in result

    def test_wrong_param_count(self):
        result = Commands._run_show_history("2024-04-28 12:00")
        assert "expected two parameters" in result

    def test_invalid_datetime_format(self):
        result = Commands._run_show_history("not-a-date", "also-not-a-date")
        assert "YYYY-MM-DD HH:MM" in result

    def test_no_results(self):
        with patch('assistant.commands.Commands.FloxyDatabase') as mock_db_cls:
            mock_db_cls.get_instance.return_value.get_history_range.return_value = []
            result = Commands._run_show_history("2024-01-01 00:00", "2024-01-01 01:00")

        assert "No history entries found" in result

    def test_deduplicates_entries(self):
        results = [
            {"content": "same content here", "timestamp": 1714300000.0,
             "role": "user", "source": None, "location": None},
            {"content": "same content here", "timestamp": 1714300060.0,
             "role": "user", "source": None, "location": None},
        ]
        with patch('assistant.commands.Commands.FloxyDatabase') as mock_db_cls:
            mock_db_cls.get_instance.return_value.get_history_range.return_value = results
            result = Commands._run_show_history("2024-04-28 12:00", "2024-04-28 13:00")

        assert result.count("same content here") == 1

    def test_truncates_long_content(self):
        long_text = "A" * 700
        results = [
            {"content": long_text, "timestamp": 1714300000.0,
             "role": "user", "source": None, "location": None},
        ]
        with patch('assistant.commands.Commands.FloxyDatabase') as mock_db_cls:
            mock_db_cls.get_instance.return_value.get_history_range.return_value = results
            result = Commands._run_show_history("2024-04-28 12:00", "2024-04-28 13:00")

        assert "..." in result
        # Content should be truncated to 600 + "..."
        lines = result.split("\n")
        assert all(len(line) < 700 for line in lines)

    def test_no_source_or_location_omits_tags(self):
        results = [
            {"content": "plain entry", "timestamp": 1714300000.0,
             "role": "user", "source": None, "location": None},
        ]
        with patch('assistant.commands.Commands.FloxyDatabase') as mock_db_cls:
            mock_db_cls.get_instance.return_value.get_history_range.return_value = results
            result = Commands._run_show_history("2024-04-28 12:00", "2024-04-28 13:00")

        assert "user:" in result
        assert "(None)" not in result
        assert "[None]" not in result


@pytest.mark.unit
class TestShortUrl:
    def test_short_url_under_limit_returns_unchanged(self):
        assert Commands._short_url("https://x.com/abc") == "https://x.com/abc"

    def test_short_url_strips_scheme_when_too_long(self):
        u = "https://www.example.com/some/very/long/path/that/keeps/going"
        out = Commands._short_url(u, max_len=30)
        assert "…" not in out
        assert "https://" not in out
        assert "www.example.com" in out
        assert len(out) <= 30

    def test_short_url_handles_garbage(self):
        out = Commands._short_url("not a url at all but still very long " * 3, max_len=20)
        assert "…" not in out
        assert len(out) <= 20


@pytest.mark.unit
class TestDescribeDispatcher:
    def test_describe_dim_with_brightness(self, commands_instance):
        assert commands_instance.describe("dim", "kitchen, 50") == "Dimming kitchen to 50%"

    def test_describe_open_url_truncates(self, commands_instance):
        long = "https://www.example.com/" + "x" * 200
        out = commands_instance.describe("open_url", long)
        assert out is not None
        assert out.startswith("Reading ")
        assert not out.endswith("…")
        assert len(out) < 80

    def test_describe_unknown_command_returns_none(self, commands_instance):
        assert commands_instance.describe("does_not_exist", "x") is None

    def test_describe_browse_act_uses_label(self, commands_instance):
        commands_instance.web_access.get_snapshot_label = Mock(return_value="Boka bord")
        assert commands_instance.describe("browse_act", "0") == "Clicking on Boka bord"

    def test_describe_browse_act_falls_back_to_index(self, commands_instance):
        commands_instance.web_access.get_snapshot_label = Mock(return_value=None)
        assert commands_instance.describe("browse_act", "3") == "Clicking on element 3"

    def test_describe_browse_act_with_value_says_filling(self, commands_instance):
        commands_instance.web_access.get_snapshot_label = Mock(return_value="Email")
        assert commands_instance.describe("browse_act", "2=test@example.com") == \
            "Filling Email with 'test@example.com'"

    def test_describe_browse_act_with_enter_marker(self, commands_instance):
        commands_instance.web_access.get_snapshot_label = Mock(return_value="Search")
        assert commands_instance.describe("browse_act", "2=shampoo\\n") == \
            "Filling Search with 'shampoo' and pressing Enter"

    def test_describe_browse_act_batch_describes_each_op(self, commands_instance):
        commands_instance.web_access.get_snapshot_label = Mock(return_value=None)
        assert commands_instance.describe("browse_act", "2=foo; 3") == \
            "Filling element 2 with 'foo' · Clicking on element 3"

    def test_describe_browse_key_with_index_uses_label(self, commands_instance):
        commands_instance.web_access.get_snapshot_label = Mock(return_value="Search box")
        assert commands_instance.describe("browse_key", "3,ArrowDown") == \
            "Pressing ArrowDown on Search box"

    def test_describe_browse_key_index_falls_back_to_element(self, commands_instance):
        commands_instance.web_access.get_snapshot_label = Mock(return_value=None)
        assert commands_instance.describe("browse_key", "3,Escape") == \
            "Pressing Escape on element 3"

    def test_describe_swallows_exceptions(self, commands_instance):
        commands_instance.describe_turn_on = Mock(side_effect=RuntimeError("boom"))
        assert commands_instance.describe("turn_on", "kitchen") is None

    def test_describe_send_imessage_uses_full_name_when_available(self, commands_instance):
        from integrations.imessage.IMessageIntegration import IMessageIntegration
        m = IMessageIntegration.get_instance()
        m._name_to_handle = {"alice": "+12345678901"}
        m._handle_to_full_name = {"+12345678901": "Alice Smith"}
        m._contact_cache_time = 1e18
        try:
            assert commands_instance.describe("send_imessage", "Alice, hi") == \
                "Sending iMessage to Alice Smith (+12345678901)"
        finally:
            IMessageIntegration.reset_instance()

    def test_describe_send_imessage_falls_back_to_recipient_when_no_full_name(self, commands_instance):
        from integrations.imessage.IMessageIntegration import IMessageIntegration
        m = IMessageIntegration.get_instance()
        m._name_to_handle = {"alice": "+12345678901"}
        m._handle_to_full_name = {}
        m._contact_cache_time = 1e18
        try:
            assert commands_instance.describe("send_imessage", "Alice, hi") == \
                "Sending iMessage to Alice (+12345678901)"
        finally:
            IMessageIntegration.reset_instance()


# ---------------------------------------------------------------------------
# TestCmdVaultAdd
# ---------------------------------------------------------------------------

class TestCmdVaultAdd:
    @pytest.fixture
    def vault(self, tmp_path):
        from common.vault.Vault import Vault
        vault = Vault(tmp_path / "vault.bin", "pw")
        with patch.object(Vault, "get_instance", return_value=vault):
            yield vault

    def test_adds_secret_and_returns_no_error(self, commands_instance, vault):
        assert commands_instance.cmd_vault_add("phone", "Anna's mobile", "070-123 45 67") == ""
        secret = vault.get("anna-s-mobile")
        assert secret.kind == "phone" and secret.value == "070-123 45 67"

    def test_runs_through_dispatcher_as_multi_param(self, commands_instance, vault):
        assert commands_instance.run("vault_add", ["card", "Visa", "4111 1111 1111 1111"]) == ""
        assert "visa" in vault

    def test_unquoted_value_split_on_commas_is_rejoined(self, commands_instance, vault):
        assert commands_instance.cmd_vault_add("text", "Safe code", "alpha", "bravo") == ""
        assert vault.get("safe-code").value == "alpha, bravo"

    def test_unknown_kind_is_an_error(self, commands_instance, vault):
        result = commands_instance.cmd_vault_add("ssn", "Personnummer", "19900101-1234")
        assert result.startswith("vault_add failed: unknown kind 'ssn'")
        assert len(vault) == 0

    def test_duplicate_value_is_not_a_failure(self, commands_instance, vault):
        # The value is in the vault either way; the id is in the prompt's vault section.
        vault.add("phone", "Marcus mobile", "+46701234567")
        assert commands_instance.cmd_vault_add("phone", "My phone", "070-123 45 67") == ""
        assert len(vault) == 1

    def test_invalid_value_reports_vault_error(self, commands_instance, vault):
        result = commands_instance.cmd_vault_add("card", "Visa", "1234")
        assert result == "vault_add failed: card number must be 13-19 digits"

    def test_disabled_vault_refuses(self, commands_instance, tmp_path):
        from common.vault.Vault import Vault
        disabled = Vault(tmp_path / "vault.bin", None)
        with patch.object(Vault, "get_instance", return_value=disabled):
            result = commands_instance.cmd_vault_add("phone", "Anna", "070-123 45 67")
        assert result == "vault_add failed: the vault is disabled (vault.key not set in config.yaml)"

    def test_describe_never_shows_the_value(self, commands_instance):
        text = commands_instance.describe("vault_add", ["card", "Visa", "4111 1111 1111 1111"])
        assert text == "Adding Visa to the vault (card)"
        assert "4111" not in text
        assert commands_instance.describe("vault_add", []) == "Adding a secret to the vault"


# ---------------------------------------------------------------------------
# TestCmdVaultRequest
# ---------------------------------------------------------------------------

class TestCmdVaultRequest:
    """The value comes from a masked prompt through the notifier; the LLM gets the placeholder."""

    @pytest.fixture
    def vault(self, tmp_path):
        from common.vault.Vault import Vault
        vault = Vault(tmp_path / "vault.bin", "pw")
        with patch.object(Vault, "get_instance", return_value=vault):
            yield vault

    @staticmethod
    def _notifier(*answers, decision="accepted"):
        """A notifier whose request_secret feeds `answers` to accept in turn (retrying on
        SecretRejected like the hub) and returns `decision`."""
        from assistant.Notifier import SecretRejected
        fake = Mock()
        fake.calls = []

        def request_secret(title, details, accept, timeout=None, source=None, conversation=None):
            fake.calls.append((title, details))
            fake.origin = (source, conversation)
            for value in answers:
                try:
                    fake.outcome = accept(value)
                    return "accepted"
                except SecretRejected as e:
                    fake.reason = str(e)
            return decision

        fake.request_secret = request_secret
        return fake

    def test_runs_in_the_background_after_validation(self, commands_instance, vault):
        with patch.object(commands_instance, "background_command") as bg:
            assert commands_instance.run("vault_request", ["card", "Marcus's Amex"]) == ""
        bg.assert_called_once_with(commands_instance._run_vault_request, ("card", "Marcus's Amex", None, None))

    def test_prompt_is_addressed_to_the_asking_requests_chat(self, commands_instance, vault):
        # Captured when the command runs, so a later request replacing the context
        # (or none at all, for the bg worker) cannot redirect the prompt.
        commands_instance._current_context = Mock(source="Console", conversation="tui")
        with patch.object(commands_instance, "background_command") as bg:
            commands_instance.cmd_vault_request("card", "Amex")
        bg.assert_called_once_with(commands_instance._run_vault_request, ("card", "Amex", "Console", "tui"))
        fake = self._notifier("4111 1111 1111 1111")
        with patch("assistant.commands.Commands.notifier", return_value=fake):
            commands_instance._run_vault_request("card", "Amex", "Console", "tui")
        assert fake.origin == ("Console", "tui")

    def test_unquoted_description_split_on_commas_is_rejoined(self, commands_instance, vault):
        with patch.object(commands_instance, "background_command") as bg:
            commands_instance.cmd_vault_request("text", "Safe", "hallway")
        bg.assert_called_once_with(commands_instance._run_vault_request, ("text", "Safe, hallway", None, None))

    def test_validation_errors_never_open_a_prompt(self, commands_instance, vault):
        with patch.object(commands_instance, "background_command") as bg:
            assert "expected" in commands_instance.cmd_vault_request("card")
            assert commands_instance.cmd_vault_request("ssn", "Personnummer").startswith(
                "vault_request failed: unknown kind 'ssn'")
            assert commands_instance.cmd_vault_request("card", "  ") == "vault_request failed: description must not be empty"
        bg.assert_not_called()

    def test_disabled_vault_refuses(self, commands_instance, tmp_path):
        from common.vault.Vault import Vault
        disabled = Vault(tmp_path / "vault.bin", None)
        with patch.object(Vault, "get_instance", return_value=disabled), \
             patch.object(commands_instance, "background_command") as bg:
            result = commands_instance.cmd_vault_request("card", "Amex")
        assert result == "vault_request failed: the vault is disabled (vault.key not set in config.yaml)"
        bg.assert_not_called()

    def test_accepted_value_is_stored_and_the_reply_carries_only_the_placeholder(self, commands_instance, vault):
        fake = self._notifier("4111 1111 1111 1111")
        with patch("assistant.commands.Commands.notifier", return_value=fake):
            reply = commands_instance._run_vault_request("card", "Marcus's Amex")
        assert fake.calls == [("Marcus's Amex (card)", "")]
        assert fake.outcome == "Stored as marcus-s-amex (•••• 1111)"
        assert vault.get("marcus-s-amex").value == "4111 1111 1111 1111"
        assert reply.startswith("Stored as {{secret:marcus-s-amex}} (card: Marcus's Amex).")
        assert "4111" not in reply

    def test_rejected_value_is_asked_again_then_stored(self, commands_instance, vault):
        fake = self._notifier("1234", "4111 1111 1111 1111")
        with patch("assistant.commands.Commands.notifier", return_value=fake):
            reply = commands_instance._run_vault_request("card", "Visa")
        assert fake.reason == "card number must be 13-19 digits"
        assert "{{secret:visa}}" in reply and "visa" in vault

    def test_duplicate_value_names_the_existing_placeholder(self, commands_instance, vault):
        vault.add("phone", "Marcus mobile", "+46701234567")
        fake = self._notifier("070-123 45 67")
        with patch("assistant.commands.Commands.notifier", return_value=fake):
            reply = commands_instance._run_vault_request("phone", "My phone")
        assert fake.outcome == "Already in the vault as marcus-mobile"
        assert reply == "That value is already in the vault as {{secret:marcus-mobile}} — nothing new was stored."
        assert len(vault) == 1

    @pytest.mark.parametrize("decision, expected", [
        ("cancel", "The user cancelled the prompt — nothing was stored."),
        ("timeout", "The user did not answer the prompt in time — nothing was stored."),
        ("unavailable", "No chat interface can show the prompt right now — ask the user to add the value in the "
                        "Vault tab of the console instead."),
        ("rejected", "The value was refused (it could not be stored) — nothing was stored."),
    ])
    def test_other_outcomes_store_nothing(self, commands_instance, vault, decision, expected):
        fake = self._notifier(decision=decision)
        with patch("assistant.commands.Commands.notifier", return_value=fake):
            assert commands_instance._run_vault_request("card", "Amex") == expected
        assert len(vault) == 0

    def test_rejected_after_every_attempt_names_the_last_reason(self, commands_instance, vault):
        fake = self._notifier("12", "34", "56", decision="rejected")
        with patch("assistant.commands.Commands.notifier", return_value=fake):
            reply = commands_instance._run_vault_request("card", "Amex")
        assert reply == "The value was refused (card number must be 13-19 digits) — nothing was stored."

# ---------------------------------------------------------------------------
# Self-setup: list_integrations / config_help / config_request / restart_integration
# (plans/SELF_SETUP_PLAN.md §3.4)
# ---------------------------------------------------------------------------

from common import ConfigSchema, ConfigStore
from common.ConfigStore import ConfigStoreError


def _secret_notifier(*answers, decision="accepted"):
    """Same shape as TestCmdVaultRequest._notifier: feeds answers to accept, retrying on
    SecretRejected the way the hub does."""
    from assistant.Notifier import SecretRejected
    fake = Mock()
    fake.calls = []

    def request_secret(title, details, accept, timeout=None, source=None, conversation=None):
        fake.calls.append((title, details))
        fake.origin = (source, conversation)
        for value in answers:
            try:
                fake.outcome = accept(value)
                return "accepted"
            except SecretRejected as e:
                fake.reason = str(e)
        return decision

    fake.request_secret = request_secret
    return fake


class TestCmdListIntegrations:
    @staticmethod
    def _status(name, state, keys=(), last_error=None, restartable=True, checked=None, accounts=()):
        return {"name": name, "state": state, "last_error": last_error, "last_check_time": checked,
                "consecutive_failures": 0, "config_keys": [], "restartable": restartable,
                "keys": [{"key": k, "set": v} for k, v in keys], "accounts": list(accounts)}

    def test_an_account_holding_integration_lists_each_accounts_verdict_and_a_lists_summary(self, commands_instance):
        monitor = Mock()
        status = self._status("IMAP Mail", "warning", last_error="Work: auth failed",
                              accounts=[{"name": "Personal", "state": "ok", "last_error": None},
                                        {"name": "Work", "state": "error", "last_error": "auth failed"}])
        status["keys"] = [{"key": "imap.accounts", "set": True, "summary": "2 accounts, 1 without password"}]
        monitor.describe_all.return_value = [status]
        with patch.object(Commands, "_integration_monitor", return_value=monitor):
            line, = commands_instance._run_list_integrations().splitlines()
        assert "accounts: Personal ok, Work: auth failed" in line
        assert "[imap.accounts: 2 accounts, 1 without password]" in line

    def test_one_line_per_integration_sorted_with_state_keys_and_restartability(self, commands_instance):
        monitor = Mock()
        monitor.describe_all.return_value = [
            self._status("Tesla", "error", [("tesla.tessie_access_token", True)], last_error="401 Unauthorized",
                         checked=time.time() - 130),
            self._status("Gmail", "not_configured", [("email.address", False), ("email.password", False)]),
            self._status("Fish Audio", "ok", restartable=False),
            self._status("iCloud", "needs_2fa", [("icloud.apple_id", True)], last_error="2FA required"),
            self._status("Indigo", "ok", [("indigo.host", True)]),
        ]
        with patch.object(Commands, "_integration_monitor", return_value=monitor):
            text = commands_instance._run_list_integrations()
        assert text.splitlines() == [
            "Fish Audio — ok; needs a server restart to change",
            "Gmail — not configured [email.address: not set, email.password: not set]; restartable",
            "iCloud — needs 2FA: 2FA required [icloud.apple_id: set]; restartable",
            "Indigo — ok [indigo.host: set]; restartable",
            "Tesla — error: 401 Unauthorized (checked 2 min ago) [tesla.tessie_access_token: set]; restartable",
        ]

    def test_a_disabled_integration_shows_its_enable_line(self, commands_instance):
        monitor = Mock()
        monitor.describe_all.return_value = [
            self._status("Ring", "disabled", [("ring.username", False), ("ring.password", False)]),
            self._status("Web Access", "ok"),
            self._status("Indigo", "disabled", [("indigo.host", False)]),
        ]
        with patch.object(Commands, "_integration_monitor", return_value=monitor):
            text = commands_instance._run_list_integrations()
        assert text.splitlines() == [
            'Indigo — disabled; enable: enable_integration("Indigo") [indigo.host: not set]; restartable',
            'Ring — disabled; enable: enable_integration("Ring") [ring.username: not set, ring.password: not set]; restartable',
            "Web Access — ok; restartable",
        ]

    def test_enable_integration_writes_the_name_and_restarts(self, commands_instance):
        monitor = Mock()
        monitor.get_all_statuses.return_value = [{"name": "Ring"}, {"name": "Sonos & Spotify"}]
        monitor.restart.return_value = {"restartable": True, "status": {
            "name": "Ring", "state": "not_configured", "last_error": None, "config_keys": ["ring.username"]}}
        with patch.object(Commands, "_integration_monitor", return_value=monitor), \
             patch("assistant.commands.Commands.ConfigStore.enable_integration", return_value="Ring enabled") as enable, \
             patch("assistant.commands.Commands.ConfigStore.describe_keys",
                   return_value=[{"key": "ring.username", "set": False}]):
            text = commands_instance._run_enable_integration("ring")
        enable.assert_called_once_with("Ring")   # the name as list_integrations shows it, whatever the LLM typed
        monitor.restart.assert_called_once_with("Ring")
        assert text == "Ring enabled; Ring restarted: not configured — not set: ring.username"

    def test_enable_integration_unknown_name(self, commands_instance):
        monitor = Mock()
        monitor.get_all_statuses.return_value = [{"name": "Ring"}]
        with patch.object(Commands, "_integration_monitor", return_value=monitor), \
             patch("assistant.commands.Commands.ConfigStore.enable_integration") as enable:
            text = commands_instance._run_enable_integration("Rong")
        enable.assert_not_called()
        assert text == "No integration named 'Rong'. Known: Ring"

    def test_disable_integration_removes_the_name_and_restarts_to_a_placeholder(self, commands_instance):
        monitor = Mock()
        monitor.get_all_statuses.return_value = [{"name": "Ring"}]
        monitor.restart.return_value = {"restartable": True, "status": {
            "name": "Ring", "state": "disabled", "last_error": None, "config_keys": []}}
        with patch.object(Commands, "_integration_monitor", return_value=monitor), \
             patch("assistant.commands.Commands.ConfigStore.disable_integration", return_value="Ring disabled") as disable:
            text = commands_instance._run_disable_integration("Ring")
        disable.assert_called_once_with("Ring")
        assert text == 'Ring disabled; Ring restarted: disabled — not in enabled_integrations; enable_integration("Ring") adds it'

    @pytest.mark.parametrize("listed, enable, reported", [
        (False, True, [call("Ring", True, "ok")]),
        (True, False, [call("Ring", False, "ok")]),
        (True, True, []),      # already on: the list did not change
        (False, False, []),
    ])
    def test_telemetry_hears_of_a_switch_only_when_the_list_changed(self, commands_instance, listed, enable, reported):
        from assistant.Telemetry import Telemetry
        monitor = Mock()
        monitor.get_all_statuses.return_value = [{"name": "Ring"}]
        monitor.restart.return_value = {"restartable": True, "status": {
            "name": "Ring", "state": "ok", "last_error": None, "config_keys": []}}
        with patch.object(Commands, "_integration_monitor", return_value=monitor), \
             patch("assistant.commands.Commands.ConfigStore.enable_integration", return_value="Ring enabled"), \
             patch("assistant.commands.Commands.ConfigStore.disable_integration", return_value="Ring disabled"), \
             patch("assistant.commands.Commands.ConfigManager.get_instance") as config, \
             patch.object(Telemetry, "get_instance") as telemetry:
            config.return_value.integration_enabled.return_value = listed
            commands_instance._switch_integration("ring", enable)
        assert telemetry.return_value.integration_switch.call_args_list == reported

    def test_enable_and_disable_run_in_the_config_lane(self, commands_instance):
        with patch.object(commands_instance, "background_command") as bg:
            assert commands_instance.run("enable_integration", ['"Ring"']) == ""
            assert commands_instance.run("disable_integration", ['"Ring"']) == ""
        assert bg.call_args_list == [
            call(commands_instance._run_enable_integration, "Ring", lane=Commands.CONFIG_LANE),
            call(commands_instance._run_disable_integration, "Ring", lane=Commands.CONFIG_LANE),
        ]
        assert commands_instance.run("enable_integration", []).startswith("enable_integration failed")

    def test_nothing_registered(self, commands_instance):
        monitor = Mock(describe_all=Mock(return_value=[]))
        with patch.object(Commands, "_integration_monitor", return_value=monitor):
            assert commands_instance._run_list_integrations() == "No integrations are registered."

    def test_a_feed_integration_shows_its_feed_state(self, commands_instance):
        catching_up = {"state": "catching_up", "progress": 0.43, "remaining": 1203, "last_error": None}
        stopped = {"state": "stopped", "progress": None, "remaining": None,
                   "last_error": "no session file — authenticate in the Telegram tab"}
        monitor = Mock()
        monitor.describe_all.return_value = [
            dict(self._status("iMessage", "ok"), feed=catching_up),
            dict(self._status("Telegram Feed", "needs_2fa", last_error="2FA required"), feed=stopped),
            dict(self._status("Photos", "ok"), feed=None),
            dict(self._status("Indigo", "ok", [("indigo.host", True)]),
                 feed={"state": "live", "paused": False, "progress": None, "remaining": None, "last_error": None}),
        ]
        with patch.object(Commands, "_integration_monitor", return_value=monitor):
            text = commands_instance._run_list_integrations()
        assert text.splitlines() == [
            "iMessage — ok; feed: catching up 43%, 1203 left; restartable",
            "Indigo — ok [indigo.host: set]; feed: live; restartable",
            "Photos — ok; feed: not started; restartable",
            "Telegram Feed — needs 2FA: 2FA required; feed: stopped: no session file — authenticate in the "
            "Telegram tab; restartable",
        ]

    def test_a_disabled_feed_is_still_called_a_feed(self, commands_instance):
        # A disabled feed has no runner to describe (feed=None), but the line must still carry
        # the word "feed" — that is how "set up the telegram feed" finds Telegram Feed.
        monitor = Mock()
        monitor.describe_all.return_value = [
            dict(self._status("Telegram Feed", "disabled"), feed=None),
            dict(self._status("Photos", "ok"), feed=None),
        ]
        with patch.object(Commands, "_integration_monitor", return_value=monitor):
            text = commands_instance._run_list_integrations()
        assert text.splitlines() == [
            "Photos — ok; feed: not started; restartable",
            'Telegram Feed — disabled; enable: enable_integration("Telegram Feed"); feed; restartable',
        ]

    def test_a_private_data_integration_carries_the_warning_and_the_others_do_not(self, commands_instance):
        """Enabling a sensitive integration ends the beta's transcript telemetry for good, so the
        assistant must be able to warn before it does — the line has to say so, and only there."""
        monitor = Mock()
        monitor.describe_all.return_value = [
            dict(self._status("Gmail", "disabled", [("email.address", False)]), sensitive=True),
            dict(self._status("Tesla", "ok", [("tesla.tessie_access_token", True)]), sensitive=False),
            self._status("Weather", "ok"),
        ]
        with patch.object(Commands, "_integration_monitor", return_value=monitor):
            lines = {line.split(" — ")[0]: line for line in commands_instance._run_list_integrations().splitlines()}
        assert "private data" in lines["Gmail"] and "transcript" in lines["Gmail"]
        assert 'enable_integration("Gmail")' in lines["Gmail"]   # the warning does not displace the enable line
        assert "private data" not in lines["Tesla"] and "private data" not in lines["Weather"]

class TestCmdConfigHelp:
    @pytest.fixture(autouse=True)
    def example(self, tmp_path, monkeypatch):
        path = tmp_path / "config.example.yaml"
        path.write_text("email:\n  address: your_gmail_address@gmail.com\n"
                        "  password: YOUR_GMAIL_APP_PASSWORD  # Generate at https://myaccount.google.com/apppasswords\n\n"
                        "owner_name: YOUR_NAME\n")
        monkeypatch.setattr(ConfigSchema, "EXAMPLE_PATH", path)
        ConfigSchema.reset()
        yield
        ConfigSchema.reset()

    def test_runs_in_the_background_with_the_section(self, commands_instance):
        with patch.object(commands_instance, "background_command") as bg:
            commands_instance.run("config_help", ['"email"'])
        bg.assert_called_once_with(commands_instance._run_config_help, "email")

    def test_a_setup_command_notes_setup_activity(self, commands_instance):
        with patch("assistant.commands.Commands.FirstRun.note_setup_activity") as noted, \
                patch.object(ConfigStore, "summary", return_value=""):
            commands_instance._run_config_help("email")
        noted.assert_called_once()

    def test_section_text_with_comments_and_the_set_flags(self, commands_instance):
        with patch.object(ConfigStore, "summary", return_value="email.address: not set, email.password: not set"):
            text = commands_instance._run_config_help("email")
        assert text == ("email:\n  address: your_gmail_address@gmail.com\n"
                        "  password: YOUR_GMAIL_APP_PASSWORD  # Generate at https://myaccount.google.com/apppasswords"
                        "\n\nCurrently: email.address: not set, email.password: not set")

    def test_no_argument_lists_sections_with_counts(self, commands_instance):
        with patch.object(ConfigStore, "section_summary", side_effect=lambda s: {"email": "0 of 2 set", "owner_name": "1 of 1 set"}[s]):
            text = commands_instance._run_config_help("")
        assert "email (0 of 2 set)\nowner_name (1 of 1 set)" in text

    def test_unknown_section_names_the_known_ones(self, commands_instance):
        assert commands_instance._run_config_help("gmail") == \
            "No section 'gmail' in config.example.yaml. Sections: email, owner_name"

    def test_a_documentation_only_section_comes_back_as_is(self, commands_instance, tmp_path, monkeypatch):
        """iMessage has no keys of its own: the section is its comments, and the "Currently:"
        summary only makes sense for a section with values."""
        path = tmp_path / "doc-only.example.yaml"
        path.write_text("# Reads Messages.app's database on this Mac, so nothing to ask the user for.\n"
                        "#imessage:\n\n"
                        "owner_name: YOUR_NAME\n")
        monkeypatch.setattr(ConfigSchema, "EXAMPLE_PATH", path)
        ConfigSchema.reset()
        with patch.object(ConfigStore, "summary") as summary:
            text = commands_instance._run_config_help("imessage")
        summary.assert_not_called()
        assert text == "# Reads Messages.app's database on this Mac, so nothing to ask the user for.\nimessage:"


class TestCmdConfigRequest:
    def test_validates_synchronously_then_waits_in_the_background(self, commands_instance):
        commands_instance._current_context = Mock(source="Console", conversation="tui")
        with patch.object(ConfigStore, "check_key") as check, patch.object(commands_instance, "background_command") as bg:
            assert commands_instance.run("config_request", ['"email.password"']) == ""
        check.assert_called_once_with("email.password")
        bg.assert_called_once_with(commands_instance._run_config_request, ("email.password", "Console", "tui"), lane=Commands.CONFIG_LANE)

    def test_a_refused_key_never_opens_a_prompt(self, commands_instance):
        with patch.object(ConfigStore, "check_key", side_effect=ConfigStoreError("'vault.key' cannot be changed from here")), \
             patch.object(commands_instance, "background_command") as bg:
            assert commands_instance.cmd_config_request("vault.key") == \
                "config_request failed: 'vault.key' cannot be changed from here"
            assert commands_instance.cmd_config_request("") == \
                "config_request failed: expected a setting key such as imap.accounts[0].password"
        bg.assert_not_called()

    def test_accepted_value_is_written_and_the_reply_never_carries_it(self, commands_instance, capsys):
        fake = _secret_notifier("abcd efgh ijkl mnop")
        with patch("assistant.commands.Commands.notifier", return_value=fake), \
             patch.object(ConfigStore, "is_set", return_value=False), \
             patch.object(ConfigStore, "set_value", return_value="email.password saved") as set_value:
            reply = commands_instance._run_config_request("email.password", "Console", "tui")
        set_value.assert_called_once_with("email.password", "abcd efgh ijkl mnop")
        assert fake.calls == [("Setting email.password", "")]
        assert fake.origin == ("Console", "tui")
        assert fake.outcome == "Saved email.password"
        assert reply == ("email.password saved. Restart the integration that uses it (restart_integration) for the "
                         "change to take effect.")
        assert "abcd" not in reply and "abcd" not in capsys.readouterr().out

    def test_replacing_an_existing_value_says_so_in_the_prompt(self, commands_instance):
        fake = _secret_notifier("x")
        with patch("assistant.commands.Commands.notifier", return_value=fake), \
             patch.object(ConfigStore, "is_set", return_value=True), patch.object(ConfigStore, "set_value"):
            commands_instance._run_config_request("owner_name")
        assert fake.calls == [("Setting owner_name", "This replaces the current value.")]

    def test_a_value_of_the_wrong_shape_is_asked_again(self, commands_instance):
        fake = _secret_notifier("abc", "12345")
        calls = []

        def set_value(key, raw):
            calls.append(raw)
            if raw == "abc":
                raise ConfigStoreError("'telegram_feed.api_id' expects a int")

        with patch("assistant.commands.Commands.notifier", return_value=fake), \
             patch.object(ConfigStore, "is_set", return_value=False), patch.object(ConfigStore, "set_value", set_value):
            reply = commands_instance._run_config_request("telegram_feed.api_id")
        assert fake.reason == "'telegram_feed.api_id' expects a int"
        assert calls == ["abc", "12345"]
        assert reply.startswith("telegram_feed.api_id saved.")

    @pytest.mark.parametrize("decision, expected", [
        ("cancel", "The user cancelled the prompt — nothing was saved."),
        ("timeout", "The user did not answer the prompt in time — nothing was saved."),
        ("unavailable", "No chat interface can show the prompt right now — ask the user to open the console or "
                        "Telegram and try again."),
        ("rejected", "The value was refused (it could not be stored) — nothing was saved."),
    ])
    def test_other_outcomes_save_nothing(self, commands_instance, decision, expected):
        fake = _secret_notifier(decision=decision)
        with patch("assistant.commands.Commands.notifier", return_value=fake), \
             patch.object(ConfigStore, "is_set", return_value=False), patch.object(ConfigStore, "set_value") as set_value:
            assert commands_instance._run_config_request("email.password") == expected
        set_value.assert_not_called()

    def test_rejected_after_every_attempt_names_the_last_reason(self, commands_instance):
        fake = _secret_notifier("a", "b", "c", decision="rejected")
        with patch("assistant.commands.Commands.notifier", return_value=fake), \
             patch.object(ConfigStore, "is_set", return_value=False), \
             patch.object(ConfigStore, "set_value", side_effect=ConfigStoreError("the value is empty")):
            reply = commands_instance._run_config_request("owner_name")
        assert reply == "The value was refused (the value is empty) — nothing was saved."

class TestCmdConfigSet:
    def test_validates_synchronously_then_writes_in_the_background(self, commands_instance):
        with patch.object(ConfigStore, "check_key") as check, patch.object(commands_instance, "background_command") as bg:
            assert commands_instance.run("config_set", ['"icloud.devices[1].id"', '"uuid-anna"']) == ""
        check.assert_called_once_with("icloud.devices[1].id")
        bg.assert_called_once_with(commands_instance._run_config_set, ["icloud.devices[1].id", "uuid-anna"], lane=Commands.CONFIG_LANE)

    def test_a_parens_index_is_normalized_before_the_store_sees_it(self, commands_instance):
        """What the 2026-09-08 log showed: the parser had turned [0] into (0) and the store refused it."""
        with patch.object(ConfigStore, "check_key") as check, patch.object(commands_instance, "background_command") as bg:
            assert commands_instance.cmd_config_set("icloud.devices(0).id", "uuid-dad") is None
        check.assert_called_once_with("icloud.devices[0].id")
        bg.assert_called_once_with(commands_instance._run_config_set, ["icloud.devices[0].id", "uuid-dad"], lane=Commands.CONFIG_LANE)
        with patch.object(ConfigStore, "check_key") as check, patch.object(commands_instance, "background_command"):
            commands_instance._current_context = Mock(source="Console", conversation="tui")
            commands_instance.cmd_config_request("icloud.devices(1).id")
        check.assert_called_once_with("icloud.devices[1].id")

    def test_a_denied_key_is_refused_before_anything_runs(self, commands_instance):
        with patch.object(ConfigStore, "check_key",
                          side_effect=ConfigStoreError("'vault.key' cannot be changed from here — edit config.yaml by hand")), \
             patch.object(commands_instance, "background_command") as bg:
            assert commands_instance.cmd_config_set("vault.key", "x") == \
                "config_set failed: 'vault.key' cannot be changed from here — edit config.yaml by hand"
        bg.assert_not_called()

    def test_run_writes_and_points_at_the_restart(self, commands_instance):
        with patch.object(ConfigStore, "set_value", return_value="owner_name saved") as set_value:
            reply = commands_instance._run_config_set("owner_name", "Anna")
        set_value.assert_called_once_with("owner_name", "Anna")
        assert reply == "owner_name saved. Restart the integration that uses it (restart_integration) for the change to take effect."

    def test_a_flag_already_in_effect_needs_no_restart(self, commands_instance):
        """The FTUE assistant once wrote interfaces.telegram.enabled=true (the default) and then
        offered a server restart on the hint — the bot had been running for ten seconds."""
        with patch.object(ConfigStore, "set_value", return_value="interfaces.telegram.enabled saved"), \
             patch.object(ConfigStore, "interface_enabled", return_value=True):
            reply = commands_instance._run_config_set("interfaces.telegram.enabled", "true")
        assert reply == ("interfaces.telegram.enabled saved, but the telegram interface was already on (its default) "
                         "— nothing changes and no restart is needed.")

    def test_a_flag_that_flips_points_at_the_server_restart(self, commands_instance):
        with patch.object(ConfigStore, "set_value", return_value="interfaces.voice.enabled saved"), \
             patch.object(ConfigStore, "interface_enabled", side_effect=[False, True]):
            reply = commands_instance._run_config_set("interfaces.voice.enabled", "true")
        assert reply.endswith("offer restart_server (developer toolkit) for the change to take effect.")

    def test_a_model_provider_key_needs_no_restart_for_setup(self, commands_instance):
        """The FTUE assistant once saved api_keys.gemini, saw no integration owning it and offered
        a server restart — Photos and video descriptions build their LLM on first use."""
        with patch.object(ConfigStore, "set_value", return_value="api_keys.gemini saved"):
            reply = commands_instance._run_config_set("api_keys.gemini", "AIza-x")
        assert reply.startswith("api_keys.gemini saved. Model-provider key. Photo indexing, video descriptions")
        assert "setup needs no restart" in reply

    def test_a_service_key_keeps_the_integration_restart_hint(self, commands_instance):
        with patch.object(ConfigStore, "set_value", return_value="api_keys.serper saved"):
            reply = commands_instance._run_config_set("api_keys.serper", "x")
        assert reply == "api_keys.serper saved. " + Commands.RESTART_HINT

    def test_developer_release_points_at_the_server_restart_that_switches_branch(self, commands_instance):
        with patch.object(ConfigStore, "set_value", return_value="developer_release saved"):
            reply = commands_instance._run_config_set("developer_release", "true")
        assert reply == "developer_release saved. " + Commands.DEVELOPER_RELEASE_HINT
        assert ConfigStore.coerce("developer_release", "true") is True

    def test_the_ai_footer_flag_is_read_live_so_no_restart_is_offered(self, commands_instance):
        """Contract with the config.yaml key ai_message_footer: a bool the sends read per message."""
        with patch.object(ConfigStore, "set_value", return_value="ai_message_footer saved"):
            reply = commands_instance._run_config_set("ai_message_footer", "false")
        assert reply.startswith("ai_message_footer saved. ")
        assert "no restart needed" in reply
        assert "restart_integration" not in reply and "restart_server" not in reply
        assert ConfigStore.coerce("ai_message_footer", "false") is False

    @pytest.mark.parametrize("key", ["telemetry.transcripts", "telemetry.enabled"])
    def test_a_telemetry_key_gets_the_telemetry_hint_not_the_server_restart(self, commands_instance, key):
        """The telemetry reads its keys per event, so turning it off needs no restart; the
        assistant must not be told to offer one as it is for every other section."""
        with patch.object(ConfigStore, "set_value", return_value=f"{key} saved"):
            reply = commands_instance._run_config_set(key, "false")
        assert reply == f"{key} saved. " + Commands.TELEMETRY_HINT
        assert Commands.TELEMETRY_HINT not in (Commands.RESTART_HINT, Commands.SERVER_RESTART_HINT)

    def test_a_telegram_key_points_at_the_bot_restart(self, commands_instance):
        with patch.object(ConfigStore, "set_value", return_value="telegram_bot.link_previews saved"):
            reply = commands_instance._run_config_set("telegram_bot.link_previews", "true")
        assert reply == ('telegram_bot.link_previews saved. Run restart_integration("Telegram") for the bot to pick it up '
                         '— no server restart.')

    def test_a_telegram_chat_id_needs_no_restart(self, commands_instance):
        """The bot reads the chat-id lists live; a restart from the Telegram chat that wrote
        one used to kill that conversation."""
        with patch.object(ConfigStore, "set_value", return_value="telegram_bot.allowed_chat_ids[0] saved"):
            reply = commands_instance._run_config_set("telegram_bot.allowed_chat_ids[0]", "50017001")
        assert reply == "telegram_bot.allowed_chat_ids[0] saved. " + Commands.TELEGRAM_LIVE_HINT

    def test_run_reports_a_store_error_calmly(self, commands_instance):
        with patch.object(ConfigStore, "set_value", side_effect=ConfigStoreError("icloud.devices has 1 item(s) — [1] adds the next one")):
            assert commands_instance._run_config_set("icloud.devices[3].id", "x") == \
                "config_set failed: icloud.devices has 1 item(s) — [1] adds the next one"

class TestCmdRestartIntegration:
    @staticmethod
    def _monitor(result, names=("Gmail", "Tesla")):
        monitor = Mock()
        monitor.restart.return_value = result
        monitor.get_all_statuses.return_value = [{"name": n} for n in names]
        return monitor

    @staticmethod
    def _status(name, state, last_error=None, config_keys=()):
        return {"name": name, "state": state, "last_error": last_error, "config_keys": list(config_keys),
                "last_check_time": None, "consecutive_failures": 0, "restartable": True}

    def test_runs_in_the_background_with_the_name(self, commands_instance):
        with patch.object(commands_instance, "background_command") as bg:
            assert commands_instance.run("restart_integration", ['"Gmail"']) == ""
            assert commands_instance.cmd_restart_integration("") == \
                "restart_integration failed: expected an integration name from list_integrations"
        bg.assert_called_once_with(commands_instance._run_restart_integration, "Gmail", lane=Commands.CONFIG_LANE)

    def test_ok(self, commands_instance):
        monitor = self._monitor({"restartable": True, "status": self._status("Gmail", "ok")})
        with patch.object(Commands, "_integration_monitor", return_value=monitor):
            assert commands_instance._run_restart_integration("gmail") == "Gmail restarted: ok"
        monitor.restart.assert_called_once_with("gmail")

    def test_failure_carries_the_error(self, commands_instance):
        monitor = self._monitor({"restartable": True,
                                 "status": self._status("Gmail", "warning", "[AUTHENTICATIONFAILED] Invalid credentials")})
        with patch.object(Commands, "_integration_monitor", return_value=monitor):
            assert commands_instance._run_restart_integration("Gmail") == \
                "Gmail restarted: warning — [AUTHENTICATIONFAILED] Invalid credentials"

    def test_not_configured_names_the_missing_keys(self, commands_instance):
        monitor = self._monitor({"restartable": True, "status": self._status("Gmail", "not_configured", config_keys=["email"])})
        with patch.object(Commands, "_integration_monitor", return_value=monitor), \
             patch.object(ConfigStore, "describe_keys", return_value=[{"key": "email.address", "set": True},
                                                                       {"key": "email.password", "set": False}]):
            assert commands_instance._run_restart_integration("Gmail") == \
                "Gmail restarted: not configured — not set: email.password"

    def test_needs_2fa_points_at_the_console(self, commands_instance):
        monitor = self._monitor({"restartable": True, "status": self._status("iCloud", "needs_2fa")})
        with patch.object(Commands, "_integration_monitor", return_value=monitor):
            assert commands_instance._run_restart_integration("iCloud").startswith(
                "iCloud restarted: needs 2FA — the login must be completed in the console's tab")

    def test_not_restartable(self, commands_instance):
        monitor = self._monitor({"restartable": False, "status": self._status("Fish Audio", "ok")})
        with patch.object(Commands, "_integration_monitor", return_value=monitor):
            assert commands_instance._run_restart_integration("Fish Audio") == \
                "Fish Audio cannot be restarted from here — restart the server for a config change to apply."

    def test_unknown_name_lists_the_known_ones(self, commands_instance):
        with patch.object(Commands, "_integration_monitor", return_value=self._monitor(None)):
            assert commands_instance._run_restart_integration("gmial") == \
                "No integration named 'gmial'. Known: Gmail, Tesla"

class TestRestartNamesTheLogin:
    """The monitor's status carries the integration's `login_hint`; without one the console's tab is named."""

    def test_restart_needs_2fa_names_the_command(self, commands_instance):
        status = {"name": "Telegram Feed", "state": "needs_2fa", "last_error": None, "config_keys": [],
                  "login_hint": "sign the account in with telegram_login(phone) from the console chat"}
        assert commands_instance._describe_restart(status, True) == \
            "Telegram Feed restarted: needs 2FA — sign the account in with telegram_login(phone) from the console chat"

    def test_restart_needs_2fa_without_a_hint_names_the_tab(self, commands_instance):
        status = {"name": "iCloud", "state": "needs_2fa", "last_error": None, "config_keys": [], "login_hint": ""}
        assert "console's tab" in commands_instance._describe_restart(status, True)

class TestLoadToolkitEcho:
    """4065d62f: the Home tab's progress line names the toolkit(s) being loaded."""

    def test_names_one_or_many_toolkits(self, commands_instance):
        assert commands_instance.describe("load_toolkit", ["web"]) == "Loading the web toolkit"
        assert commands_instance.describe("load_toolkit", ['"home"', "'music'"]) == \
            "Loading the home and music toolkits"
        assert commands_instance.describe("load_toolkit", "web, home, tesla") == \
            "Loading the web, home and tesla toolkits"

    def test_without_a_name_the_echo_is_generic(self, commands_instance):
        assert commands_instance.describe("load_toolkit", []) == "Loading a toolkit"
        assert commands_instance.describe("load_toolkit", " , ") == "Loading a toolkit"


class TestRestartServerNote:
    """b4308ef7: the restart's outcome is a system note (a history row the assistant sees on
    the next turn), not a bare chat push."""

    def _run(self, commands_instance):
        with patch.object(commands_instance, 'fire_and_forget', side_effect=lambda fn, p: fn(p)), \
             patch("assistant.commands.Commands.CodeQualityMaintainer") as cqm, \
             patch("assistant.commands.Commands.system_note") as note:
            cqm.pull_and_restart.return_value = "Already up to date. Restarting."
            commands_instance.cmd_restart_server("")
        return cqm, note

    def test_outcome_lands_as_a_server_system_note(self, commands_instance):
        cqm, note = self._run(commands_instance)
        cqm.pull_and_restart.assert_called_once_with()
        note.assert_called_once_with("server", "Server restart: Already up to date. Restarting.")

    def test_a_failed_restart_leaves_no_note(self, commands_instance):
        with patch.object(commands_instance, 'fire_and_forget', side_effect=lambda fn, p: fn(p)), \
             patch("assistant.commands.Commands.CodeQualityMaintainer") as cqm, \
             patch("assistant.commands.Commands.system_note") as note:
            cqm.pull_and_restart.side_effect = RuntimeError("git down")
            commands_instance.cmd_restart_server("")
        note.assert_not_called()


class TestWithAiFooter:
    @staticmethod
    def _footer(value):
        import common.ConfigManager as config_module
        real_get = config_module.config.get

        def get(key, *args, **kwargs):
            return value if key == "ai_message_footer" else real_get(key, *args, **kwargs)
        return patch.object(config_module.config, "get", get)

    def test_the_footer_is_appended_when_the_example_config_says_nothing_else(self, commands_instance):
        sent = commands_instance.with_ai_footer("On my way")
        assert sent.startswith("On my way\n\n")
        assert "help of AI" in sent

    def test_false_sends_the_body_as_written(self, commands_instance):
        with self._footer(False):
            assert commands_instance.with_ai_footer("On my way") == "On my way"

    def test_an_absent_key_keeps_the_footer(self, commands_instance):
        """An install whose config.yaml predates the setting must not change behaviour."""
        import common.ConfigManager as config_module
        with patch.object(config_module.config, "get", side_effect=lambda key, default=None: default):
            assert commands_instance.with_ai_footer("On my way").endswith(commands_instance.AI_MESSAGE_SUFFIX)
