import json
import queue
import threading
import time
import pytest
from unittest.mock import Mock, patch

from assistant import WatchList
from assistant.llm.AbstractLLM import LLMRefusalError

from assistant.agents.AutonomousActionManager import (
    AutonomousActionManager,
    AUTONOMOUS_ACTION_PROMPT,
    PERIODIC_CHECK_INTERVAL_MINUTES,
    PERIODIC_CHECK_INTERVAL_SECONDS,
    POSTPONED_CHECK_RETRY_SECONDS,
    THINKING_TAG_INSTRUCTION,
)

# What a model without thinking of its own receives (a bare Mock LLM counts as one).
FORMATTED_AUTONOMOUS_ACTION_PROMPT = AUTONOMOUS_ACTION_PROMPT.format(
    interval_minutes=PERIODIC_CHECK_INTERVAL_MINUTES,
    watch_list=WatchList.NOTHING_DUE,
    thinking_tag_instruction=THINKING_TAG_INSTRUCTION,
)


@pytest.fixture(autouse=True)
def empty_watch_list(tmp_path, monkeypatch):
    monkeypatch.setattr(WatchList, "WATCHLIST_DIR", tmp_path / "watchlist")
    monkeypatch.setattr(WatchList, "LOG_DIR", tmp_path / "log" / "watchlist")


@pytest.fixture(autouse=True)
def reset_singleton():
    AutonomousActionManager.reset_instance()
    yield
    AutonomousActionManager.reset_instance()


def make_mock_prompt_manager(prompt="test prompt", night=False):
    pm = Mock()
    pm.prompt = prompt
    pm.save_prompt_to_file = Mock()
    pm.update_history = Mock()
    pm.message_history.get_history_messages.return_value = []
    # Night state is now read from live Indigo data via is_night_mode_active(),
    # not by scanning pm.prompt. Default False so periodic/scheduled checks run;
    # the night-skip tests flip it to True.
    pm.is_night_mode_active.return_value = night
    # Real (prompt=None) runs inject the clock the same way get_messages() does.
    # Return a fixed string so append_user_turn's concatenation works under Mock.
    pm.current_time_note.return_value = "(Current time: 12:00, Monday, 2026-05-25)"
    return pm


@patch('assistant.agents.AutonomousActionManager.create_llm')
def make_manager(mock_claude_cls, prompt_manager=None):
    mock_llm = Mock()
    mock_claude_cls.return_value = mock_llm
    if prompt_manager is None:
        prompt_manager = make_mock_prompt_manager()
    mgr = AutonomousActionManager.get_instance(prompt_manager, test_mode=True)
    return mgr, mock_llm


class TestAutonomousActionManager:
    @patch('assistant.agents.AutonomousActionManager.create_llm')
    def test_singleton_pattern(self, mock_claude):
        mock_claude.return_value = Mock()
        pm = make_mock_prompt_manager()
        mgr1 = AutonomousActionManager.get_instance(pm, test_mode=True)
        mgr2 = AutonomousActionManager.get_instance(pm, test_mode=True)
        assert mgr1 is mgr2

    @patch('assistant.agents.AutonomousActionManager.create_llm')
    def test_test_mode_does_not_start_thread(self, mock_claude):
        mock_claude.return_value = Mock()
        pm = make_mock_prompt_manager()
        mgr = AutonomousActionManager.get_instance(pm, test_mode=True)
        assert mgr.test_mode is True
        assert not mgr.thread.is_alive()

    @patch('assistant.agents.AutonomousActionManager.create_llm')
    def test_activity_check_falls_back_when_the_llm_refuses(self, mock_claude):
        """Fable refuses the periodic check in bursts; the turn goes to the tier's
        refusal_fallback() once, as DialogManager does for a user turn."""
        fallback = Mock()
        fallback.name.return_value = "Claude (claude-opus-4-6)"
        fallback.think_with_messages.return_value = "<thinking>check</thinking>\nNO"
        mock_llm = Mock()
        mock_llm.name.return_value = "Claude (claude-fable-5-1)"
        mock_llm.think_with_messages.side_effect = LLMRefusalError("refused")
        mock_llm.refusal_fallback.return_value = fallback
        mock_claude.return_value = mock_llm
        pm = make_mock_prompt_manager()
        mgr = AutonomousActionManager.get_instance(pm, test_mode=True)

        with patch('assistant.commands.CommandProcessor.CommandProcessor') as mock_cp:
            mock_cp_inst = Mock()
            mock_cp_inst.command_queue = queue.Queue()
            mock_cp_inst.command_being_processed = False
            mock_cp_inst.command_running_in_background = False
            mock_cp_inst.command_reply = None
            mock_cp_inst.flush_buffer.return_value = None
            mock_cp_inst.process.return_value = None
            mock_cp.return_value = mock_cp_inst

            result = mgr.activity_check(prompt="test prompt")

        assert "NO" in result
        mock_llm.think_with_messages.assert_called_once()
        fallback.initialize.assert_called_once()
        fallback.think_with_messages.assert_called_once_with(mock_llm.think_with_messages.call_args[0][0])

    @patch('assistant.agents.AutonomousActionManager.create_llm')
    def test_activity_check_skips_when_the_fallback_refuses_too(self, mock_claude):
        """The fallback is asked once; its own refusal ends the check quietly instead of
        escaping the interval thread."""
        fallback = Mock()
        fallback.name.return_value = "Claude (claude-opus-5)"
        fallback.think_with_messages.side_effect = LLMRefusalError("refused again")
        mock_llm = Mock()
        mock_llm.name.return_value = "Claude (claude-fable-5-1)"
        mock_llm.think_with_messages.side_effect = LLMRefusalError("refused")
        mock_llm.refusal_fallback.return_value = fallback
        mock_claude.return_value = mock_llm
        pm = make_mock_prompt_manager()
        mgr = AutonomousActionManager.get_instance(pm, test_mode=True)

        with patch('assistant.commands.CommandProcessor.CommandProcessor') as mock_cp:
            mock_cp_inst = Mock()
            mock_cp_inst.command_queue = queue.Queue()
            mock_cp_inst.command_being_processed = False
            mock_cp_inst.command_running_in_background = False
            mock_cp_inst.command_reply = None
            mock_cp_inst.flush_buffer.return_value = None
            mock_cp_inst.process.return_value = None
            mock_cp.return_value = mock_cp_inst

            result = mgr.activity_check(prompt="test prompt")

        assert not result.strip()
        fallback.think_with_messages.assert_called_once()

    @patch('assistant.agents.AutonomousActionManager.create_llm')
    def test_activity_check_without_a_fallback_yields_an_empty_result(self, mock_claude):
        mock_llm = Mock()
        mock_llm.name.return_value = "Claude (claude-opus-4-6)"
        mock_llm.think_with_messages.side_effect = LLMRefusalError("refused")
        mock_llm.refusal_fallback.return_value = None
        mock_claude.return_value = mock_llm
        pm = make_mock_prompt_manager()
        mgr = AutonomousActionManager.get_instance(pm, test_mode=True)

        with patch('assistant.commands.CommandProcessor.CommandProcessor') as mock_cp:
            mock_cp_inst = Mock()
            mock_cp_inst.command_queue = queue.Queue()
            mock_cp_inst.command_being_processed = False
            mock_cp_inst.command_running_in_background = False
            mock_cp_inst.command_reply = None
            mock_cp_inst.flush_buffer.return_value = None
            mock_cp_inst.process.return_value = None
            mock_cp.return_value = mock_cp_inst

            result = mgr.activity_check(prompt="test prompt")

        assert not result.strip()
        mock_llm.think_with_messages.assert_called_once()

    @patch('assistant.agents.AutonomousActionManager.create_llm')
    def test_activity_check_sends_telegram_when_action(self, mock_claude):
        mock_llm = Mock()
        mock_llm.think_with_messages.return_value = "<thinking>check</thinking>\nYour car is low on battery"
        mock_claude.return_value = mock_llm
        pm = make_mock_prompt_manager()
        mgr = AutonomousActionManager.get_instance(pm, test_mode=True)

        with patch('assistant.commands.CommandProcessor.CommandProcessor') as mock_cp, \
             patch('assistant.Notifier._notifier', Mock()) as mock_tg_inst:
            mock_cp_inst = Mock()
            mock_cp_inst.command_queue = queue.Queue()
            mock_cp_inst.command_being_processed = False
            mock_cp_inst.command_running_in_background = False
            mock_cp_inst.command_reply = None
            mock_cp_inst.flush_buffer.return_value = None
            mock_cp_inst.process.return_value = "Your car is low on battery "
            mock_cp.return_value = mock_cp_inst
            mgr.test_mode = False

            result = mgr.activity_check(prompt="test prompt")

            mock_tg_inst.send_message.assert_called_once()
            mgr.test_mode = True  # restore

    @patch('assistant.agents.AutonomousActionManager.create_llm')
    def test_activity_check_preserves_newlines_and_telegram_html(self, mock_claude):
        """Regression: the reply used to be .split() into words and re-joined with
        spaces, collapsing every newline, and all HTML tags were stripped — Telegram
        messages arrived as one unformatted line. Newlines and the Telegram HTML the
        prompt asks for (<b>, <i>) must survive to send_message."""
        reply = (
            "<thinking>check</thinking>\n"
            "<b>📋 Läget:</b> Kollade listan —\n"
            "\n"
            "• <b>Punkt ett: klar!</b> ✅ gjord 1 jan\n"
            "• <b>Punkt två:</b> två väntar nu."
        )
        mock_llm = Mock()
        mock_llm.think_with_messages.return_value = reply
        mock_claude.return_value = mock_llm
        pm = make_mock_prompt_manager()
        mgr = AutonomousActionManager.get_instance(pm, test_mode=True)

        with patch('assistant.commands.CommandProcessor.CommandProcessor') as mock_cp, \
             patch('assistant.Notifier._notifier', Mock()) as mock_tg_inst:
            mock_cp_inst = Mock()
            mock_cp_inst.command_queue = queue.Queue()
            mock_cp_inst.command_being_processed = False
            mock_cp_inst.command_running_in_background = False
            mock_cp_inst.command_reply = None
            mock_cp_inst.flush_buffer.return_value = None
            mock_cp_inst.process.side_effect = lambda content, dm: content
            mock_cp.return_value = mock_cp_inst
            mgr.test_mode = False

            mgr.activity_check(prompt="test prompt")

            # The whole reply is fed as ONE whitespace-preserving chunk, not word-split
            fed = mock_cp_inst.process.call_args[0][0]
            assert "\n" in fed

            sent_text = mock_tg_inst.send_message.call_args[0][0]
            assert "\n\n" in sent_text
            assert "• <b>Punkt ett: klar!</b> ✅ gjord 1 jan\n" in sent_text
            assert "<thinking>" not in sent_text
            mgr.test_mode = True

    @patch('assistant.agents.AutonomousActionManager.create_llm')
    def test_activity_check_skips_telegram_when_filtered_empty(self, mock_claude):
        mock_llm = Mock()
        mock_llm.think_with_messages.return_value = "<thinking>check</thinking>\n"
        mock_claude.return_value = mock_llm
        pm = make_mock_prompt_manager()
        mgr = AutonomousActionManager.get_instance(pm, test_mode=True)

        with patch('assistant.commands.CommandProcessor.CommandProcessor') as mock_cp, \
             patch('assistant.Notifier._notifier', Mock()) as mock_tg_inst:
            mock_cp_inst = Mock()
            mock_cp_inst.command_queue = queue.Queue()
            mock_cp_inst.command_being_processed = False
            mock_cp_inst.command_running_in_background = False
            mock_cp_inst.command_reply = None
            mock_cp_inst.flush_buffer.return_value = None
            mock_cp_inst.process.return_value = ""
            mock_cp.return_value = mock_cp_inst
            mgr.test_mode = False

            mgr.activity_check(prompt="test prompt")

            mock_tg_inst.send_message.assert_not_called()
            mgr.test_mode = True

    @patch('assistant.agents.AutonomousActionManager.create_llm')
    def test_activity_check_uses_prompt_manager_when_none(self, mock_claude):
        mock_llm = Mock()
        mock_llm.think_with_messages.return_value = "NO"
        mock_claude.return_value = mock_llm
        pm = make_mock_prompt_manager(prompt="manager prompt")
        mgr = AutonomousActionManager.get_instance(pm, test_mode=True)

        with patch('assistant.commands.CommandProcessor.CommandProcessor') as mock_cp:
            mock_cp_inst = Mock()
            mock_cp_inst.command_queue = queue.Queue()
            mock_cp_inst.command_being_processed = False
            mock_cp_inst.command_running_in_background = False
            mock_cp_inst.command_reply = None
            mock_cp_inst.flush_buffer.return_value = None
            mock_cp_inst.process.return_value = None
            mock_cp.return_value = mock_cp_inst

            mgr.activity_check(prompt=None)

        messages = mock_llm.think_with_messages.call_args[0][0]
        assert messages[0] == {"role": "system", "content": "manager prompt"}

    @patch('assistant.agents.AutonomousActionManager.create_llm')
    def test_activity_check_includes_conversation_history(self, mock_claude):
        # Regression: history moved out of the prompt text into message turns, so
        # the autonomous check must append it explicitly — otherwise it evaluates
        # "the conversation history above" with no history and the saved prompt
        # file does not reflect what was sent.
        mock_llm = Mock()
        pm = make_mock_prompt_manager(prompt="manager prompt")
        pm.message_history.get_history_messages.return_value = [
            {"role": "user", "content": "Marcus (14:40): remind me in 10 min"},
            {"role": "assistant", "content": "Will do."},
        ]

        # Snapshot what is sent/saved — the list is mutated afterwards (the reply
        # is appended), so capture copies at call time rather than by reference.
        sent, saved = {}, {}
        mock_llm.think_with_messages.side_effect = lambda msgs, **kw: sent.update(m=list(msgs)) or "NO"
        pm.save_prompt_to_file.side_effect = lambda msgs: saved.update(m=list(msgs))
        mock_claude.return_value = mock_llm
        mgr = AutonomousActionManager.get_instance(pm, test_mode=True)

        with patch('assistant.commands.CommandProcessor.CommandProcessor') as mock_cp:
            mock_cp_inst = Mock()
            mock_cp_inst.command_queue = queue.Queue()
            mock_cp_inst.command_being_processed = False
            mock_cp_inst.command_running_in_background = False
            mock_cp_inst.command_reply = None
            mock_cp_inst.flush_buffer.return_value = None
            mock_cp_inst.process.return_value = None
            mock_cp.return_value = mock_cp_inst

            mgr.activity_check(prompt=None)

        # system + 2 history turns + autonomous instruction
        assert sent["m"][0] == {"role": "system", "content": "manager prompt"}
        assert {"role": "user", "content": "Marcus (14:40): remind me in 10 min"} in sent["m"]
        assert {"role": "assistant", "content": "Will do."} in sent["m"]
        assert FORMATTED_AUTONOMOUS_ACTION_PROMPT in sent["m"][-1]["content"]
        # The saved log file reflects exactly what was sent.
        assert saved["m"] == sent["m"]

    @patch('assistant.agents.AutonomousActionManager.create_llm')
    def test_activity_check_real_run_injects_current_time(self, mock_claude):
        # Regression: the autonomous check builds its own message list (it does not
        # call get_messages), so it must inject the clock itself. The clock moved out
        # of the system prompt onto the trailing user turn to keep the prompt bytes
        # cache-stable — but AUTONOMOUS_ACTION_PROMPT still asks the LLM to compare
        # elapsed time against the current time. Without this, time-based reminders
        # ("remind me in 10m") have no clock to measure against.
        mock_llm = Mock()
        pm = make_mock_prompt_manager(prompt="manager prompt")
        pm.current_time_note.return_value = "(Current time: 14:50, Friday, 2026-05-29)"
        pm.message_history.get_history_messages.return_value = [
            {"role": "user", "content": "Marcus (14:40): remind me in 10 min"},
            {"role": "assistant", "content": "Will do."},
        ]
        sent = {}
        mock_llm.think_with_messages.side_effect = lambda msgs, **kw: sent.update(m=list(msgs)) or "NO"
        mock_claude.return_value = mock_llm
        mgr = AutonomousActionManager.get_instance(pm, test_mode=True)

        with patch('assistant.commands.CommandProcessor.CommandProcessor') as mock_cp:
            mock_cp_inst = Mock()
            mock_cp_inst.command_queue = queue.Queue()
            mock_cp_inst.command_being_processed = False
            mock_cp_inst.command_running_in_background = False
            mock_cp_inst.command_reply = None
            mock_cp_inst.flush_buffer.return_value = None
            mock_cp_inst.process.return_value = None
            mock_cp.return_value = mock_cp_inst

            mgr.activity_check(prompt=None)

        pm.current_time_note.assert_called_once()
        # The clock rides on the final user turn, alongside the autonomous instruction.
        assert "(Current time: 14:50, Friday, 2026-05-29)" in sent["m"][-1]["content"]
        assert FORMATTED_AUTONOMOUS_ACTION_PROMPT in sent["m"][-1]["content"]

    @patch('assistant.agents.AutonomousActionManager.create_llm')
    def test_activity_check_provided_prompt_omits_current_time(self, mock_claude):
        # Provided-prompt (test/replay) runs must stay free of wall-clock state so
        # they remain deterministic — the clock is injected on real runs only.
        mock_llm = Mock()
        pm = make_mock_prompt_manager()
        sent = {}
        mock_llm.think_with_messages.side_effect = lambda msgs, **kw: sent.update(m=list(msgs)) or "NO"
        mock_claude.return_value = mock_llm
        mgr = AutonomousActionManager.get_instance(pm, test_mode=True)

        with patch('assistant.commands.CommandProcessor.CommandProcessor') as mock_cp:
            mock_cp_inst = Mock()
            mock_cp_inst.command_queue = queue.Queue()
            mock_cp_inst.command_being_processed = False
            mock_cp_inst.command_running_in_background = False
            mock_cp_inst.command_reply = None
            mock_cp_inst.flush_buffer.return_value = None
            mock_cp_inst.process.return_value = None
            mock_cp.return_value = mock_cp_inst

            mgr.activity_check(prompt="custom prompt")

        pm.current_time_note.assert_not_called()
        joined = "\n".join(m["content"] for m in sent["m"] if isinstance(m["content"], str))
        assert "Current time:" not in joined

    def test_result_cleaning_removes_thinking_tags(self):
        mgr_cls = AutonomousActionManager
        with patch('assistant.agents.AutonomousActionManager.create_llm') as mock_claude:
            mock_claude.return_value = Mock()
            pm = make_mock_prompt_manager()
            mgr = mgr_cls.get_instance(pm, test_mode=True)

            result = mgr.clean_result_and_potentially_add_to_history("<thinking>deep thought</thinking>Action taken")
            assert "<thinking>" not in result
            assert "Action taken" in result

    def test_result_cleaning_removes_xml_tags(self):
        with patch('assistant.agents.AutonomousActionManager.create_llm') as mock_claude:
            mock_claude.return_value = Mock()
            pm = make_mock_prompt_manager()
            mgr = AutonomousActionManager.get_instance(pm, test_mode=True)

            result = mgr.clean_result_and_potentially_add_to_history("<tag>content</tag>Hello")
            assert "<tag>" not in result
            assert "Hello" in result

    def test_result_adds_to_history_when_not_no(self):
        with patch('assistant.agents.AutonomousActionManager.create_llm') as mock_claude:
            mock_claude.return_value = Mock()
            pm = make_mock_prompt_manager()
            mgr = AutonomousActionManager.get_instance(pm, test_mode=True)

            mgr.clean_result_and_potentially_add_to_history("Important update")
            pm.update_history.assert_called_once()

    def test_result_does_not_add_to_history_when_no(self):
        with patch('assistant.agents.AutonomousActionManager.create_llm') as mock_claude:
            mock_claude.return_value = Mock()
            pm = make_mock_prompt_manager()
            mgr = AutonomousActionManager.get_instance(pm, test_mode=True)

            mgr.clean_result_and_potentially_add_to_history("NO")
            pm.update_history.assert_not_called()

    @patch('assistant.agents.AutonomousActionManager.time.sleep')
    @patch('assistant.agents.AutonomousActionManager.create_llm')
    def test_schedule_calls_activity_check_after_delay(self, mock_claude, mock_sleep):
        mock_llm = Mock()
        mock_llm.think.return_value = "NO"
        mock_claude.return_value = mock_llm
        pm = make_mock_prompt_manager()
        mgr = AutonomousActionManager.get_instance(pm, test_mode=True)

        import threading
        original_activity_check = mgr.activity_check
        check_called = threading.Event()

        def patched_activity_check(**kwargs):
            check_called.set()
            return "NO"

        with patch.object(mgr, 'activity_check', side_effect=patched_activity_check):
            mock_sleep.side_effect = lambda _: None  # Don't actually sleep
            mgr.schedule(5)
            check_called.wait(timeout=2)

        sleep_args = [call.args[0] for call in mock_sleep.call_args_list if call.args]
        assert any(abs(arg - 300) < 1 for arg in sleep_args), f"Expected sleep(~300), got {sleep_args}"
        assert check_called.is_set()

    @patch('assistant.agents.AutonomousActionManager.time.sleep')
    @patch('assistant.agents.AutonomousActionManager.create_llm')
    def test_caffeinate_calls_activity_check_until_the_deadline(self, mock_claude, mock_sleep):
        mock_llm = Mock()
        mock_llm.think.return_value = "NO"
        mock_claude.return_value = mock_llm
        pm = make_mock_prompt_manager()
        mgr = AutonomousActionManager.get_instance(pm, test_mode=True)

        calls = []
        with patch.object(mgr, 'activity_check', side_effect=lambda **kw: calls.append(1) or "NO"):
            # Each sleep "passes" its seconds by pulling the deadline closer instead of
            # touching the wall clock: 2.5 min at 1-min ticks gives wakeups at 1.5 and
            # 0.5 min left, then the third tick finds the deadline passed.
            mock_sleep.side_effect = lambda seconds: setattr(mgr, '_caffeine_until', mgr._caffeine_until - seconds)
            mgr.caffeinate(duration_minutes=2.5, interval_minutes=1)
            mgr._caffeine_thread.join(timeout=2)

        assert len(calls) == 2

    @patch('assistant.agents.AutonomousActionManager.time.sleep')
    @patch('assistant.agents.AutonomousActionManager.create_llm')
    def test_caffeinate_respects_running_false(self, mock_claude, mock_sleep):
        mock_llm = Mock()
        mock_claude.return_value = mock_llm
        pm = make_mock_prompt_manager()
        mgr = AutonomousActionManager.get_instance(pm, test_mode=True)

        calls = []

        def sleep_and_stop(seconds):
            mgr.running = False

        with patch.object(mgr, 'activity_check', side_effect=lambda **kw: calls.append(1) or "NO"):
            mock_sleep.side_effect = sleep_and_stop
            mgr.caffeinate(duration_minutes=10, interval_minutes=1)
            mgr._caffeine_thread.join(timeout=2.0)

        assert calls == []
        mgr.running = True  # restore

    @patch('assistant.agents.AutonomousActionManager.create_llm')
    def test_caffeinate_again_moves_the_deadline_instead_of_starting_a_second_loop(self, mock_claude):
        # Regression for 2026-09-23: every autonomous wakeup re-issued caffeinate() and
        # each call spawned its own loop — 67 of them ran side by side, one check every
        # 3 seconds for two hours. A repeat call must only extend the single loop.
        mock_claude.return_value = Mock()
        mgr = AutonomousActionManager.get_instance(make_mock_prompt_manager(), test_mode=True)
        threads = []

        def fake_thread(target, daemon):
            t = Mock()
            t.is_alive.return_value = True
            threads.append(t)
            return t

        with patch('assistant.agents.AutonomousActionManager.threading.Thread', side_effect=fake_thread):
            mgr.caffeinate(duration_minutes=10)
            first_deadline = mgr._caffeine_until
            mgr.caffeinate(duration_minutes=60)
            mgr.caffeinate()

        assert len(threads) == 1
        threads[0].start.assert_called_once()
        assert mgr._caffeine_until > first_deadline

    @patch('assistant.agents.AutonomousActionManager.create_llm')
    def test_run_periodic_check_skips_at_night(self, mock_claude):
        mock_llm = Mock()
        mock_claude.return_value = mock_llm
        pm = make_mock_prompt_manager(night=True)
        mgr = AutonomousActionManager.get_instance(pm, test_mode=True)

        call_count = [0]
        iteration = [0]

        def patched_sleep(seconds):
            iteration[0] += 1
            if iteration[0] >= 2:
                mgr.running = False

        def patched_activity_check(**kwargs):
            call_count[0] += 1
            return "NO"

        with patch('assistant.agents.AutonomousActionManager.time.sleep', side_effect=patched_sleep), \
             patch.object(mgr, 'activity_check', side_effect=patched_activity_check):
            mgr._run_periodic_check()

        assert call_count[0] == 0
        mgr.running = True  # restore

    @staticmethod
    def _run_loop(mgr, iterations: int, check=None, interaction_active: bool = False, check_seconds: float = 0.0):
        """Run the periodic loop for `iterations` sleeps on a fake clock that advances by
        every sleep and by `check_seconds` per check; returns (sleep lengths, the check mock)."""
        from assistant.agents import AutonomousActionManager as module
        clock = [1_000_000.0]
        sleeps = []

        def patched_sleep(seconds):
            sleeps.append(seconds)
            clock[0] += seconds
            if len(sleeps) >= iterations:
                mgr.running = False

        def patched_check(**kwargs):
            clock[0] += check_seconds
            if check is not None:
                return check()
            return "NO"

        with patch.object(module.time, 'time', side_effect=lambda: clock[0]), \
             patch.object(module.time, 'sleep', side_effect=patched_sleep), \
             patch.object(mgr, 'is_user_interaction_active', return_value=interaction_active), \
             patch.object(mgr, 'activity_check', side_effect=patched_check) as activity:
            mgr._run_periodic_check()
        mgr.running = True  # restore
        return sleeps, activity

    @patch('assistant.agents.AutonomousActionManager.create_llm')
    def test_a_checks_own_duration_does_not_stretch_the_cadence(self, mock_claude):
        """Start to start stays at the interval: a 10-minute check is followed by a 45-minute
        sleep, so two checks never sit more than the interval (under the 1h cache TTL) apart."""
        mock_claude.return_value = Mock()
        mgr = AutonomousActionManager.get_instance(make_mock_prompt_manager(), test_mode=True)

        sleeps, activity = self._run_loop(mgr, iterations=3, check_seconds=10 * 60)

        assert activity.call_count == 3
        assert sleeps[0] == PERIODIC_CHECK_INTERVAL_SECONDS
        assert sleeps[1] == sleeps[2] == PERIODIC_CHECK_INTERVAL_SECONDS - 10 * 60

    @patch('assistant.agents.AutonomousActionManager.create_llm')
    def test_a_check_postponed_for_user_activity_is_retried_soon_not_an_interval_later(self, mock_claude):
        mock_claude.return_value = Mock()
        mgr = AutonomousActionManager.get_instance(make_mock_prompt_manager(), test_mode=True)

        sleeps, activity = self._run_loop(mgr, iterations=3, interaction_active=True)

        activity.assert_not_called()
        assert sleeps[1] == sleeps[2] == POSTPONED_CHECK_RETRY_SECONDS
        assert POSTPONED_CHECK_RETRY_SECONDS < PERIODIC_CHECK_INTERVAL_SECONDS / 5

    @patch('assistant.agents.AutonomousActionManager.create_llm')
    def test_a_failing_check_does_not_end_the_loop(self, mock_claude):
        mock_claude.return_value = Mock()
        mgr = AutonomousActionManager.get_instance(make_mock_prompt_manager(), test_mode=True)
        outcomes = iter([RuntimeError("credit balance too low"), "NO", "NO"])

        def check():
            outcome = next(outcomes)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        sleeps, activity = self._run_loop(mgr, iterations=3, check=check)

        assert activity.call_count == 3, "the checks after the failed one still ran"
        assert sleeps[1] == PERIODIC_CHECK_INTERVAL_SECONDS

    @staticmethod
    def _run_two_iterations_at_four(mgr, curation_pending: bool):
        """Two ticks of the periodic loop at 04:05 the same day, night mode on so nothing but
        the curation slot runs; returns how often the nightly curation ran and how often the
        first-run stage was consulted."""
        from datetime import datetime as real_datetime
        four_am = real_datetime(2026, 9, 15, 4, 5)

        class FrozenDatetime:
            @staticmethod
            def now():
                return four_am

        iteration = [0]

        def patched_sleep(seconds):
            iteration[0] += 1
            if iteration[0] >= 2:
                mgr.running = False

        with patch('datetime.datetime', FrozenDatetime), \
             patch('assistant.FirstRun.curation_pending', return_value=curation_pending) as pending, \
             patch('assistant.agents.AutonomousActionManager.time.sleep', side_effect=patched_sleep), \
             patch.object(mgr, '_run_memory_curation') as curation:
            mgr._run_periodic_check()
            for thread in threading.enumerate():
                if thread.name == "MemoryCuration":
                    thread.join()
        mgr.running = True  # restore
        return curation.call_count, pending.call_count

    @patch('assistant.agents.AutonomousActionManager.create_llm')
    def test_nightly_curation_runs_once_at_four(self, mock_claude):
        mock_claude.return_value = Mock()
        mgr = AutonomousActionManager.get_instance(make_mock_prompt_manager(night=True), test_mode=True)
        assert self._run_two_iterations_at_four(mgr, curation_pending=False) == (1, 1)

    @patch('assistant.agents.AutonomousActionManager.create_llm')
    def test_nightly_curation_waits_for_the_first_run_pipeline(self, mock_claude):
        """While the first-run pipeline still has to build the first bank, the nightly run
        would curate the stub instead; the slot is skipped, and marked as used for the day
        so the skip is not re-evaluated every tick."""
        mock_claude.return_value = Mock()
        mgr = AutonomousActionManager.get_instance(make_mock_prompt_manager(night=True), test_mode=True)
        assert self._run_two_iterations_at_four(mgr, curation_pending=True) == (0, 1)

    @patch('assistant.agents.AutonomousActionManager.create_llm')
    def test_scheduled_wakeup_skips_at_night(self, mock_claude):
        mock_llm = Mock()
        mock_claude.return_value = mock_llm
        pm = make_mock_prompt_manager(night=True)
        mgr = AutonomousActionManager.get_instance(pm, test_mode=True)

        import threading
        check_called = threading.Event()

        def patched_activity_check(**kwargs):
            check_called.set()
            return "NO"

        created = []
        real_thread_cls = threading.Thread

        def capture_thread(*args, **kwargs):
            t = real_thread_cls(*args, **kwargs)
            created.append(t)
            return t

        with patch.object(mgr, 'activity_check', side_effect=patched_activity_check), \
             patch.object(mgr, '_remove_wakeup'), \
             patch('assistant.agents.AutonomousActionManager.time.sleep'), \
             patch('assistant.agents.AutonomousActionManager.threading.Thread', side_effect=capture_thread):
            mgr._start_wakeup_thread({"at": time.time() + 0.01, "scheduled_at": time.time(), "reason": ""})
            created[0].join(timeout=2.0)

        assert not check_called.is_set(), "activity_check should not be called during Night mode"

    @patch('assistant.agents.AutonomousActionManager.create_llm')
    def test_periodic_check_runs_when_live_state_not_night_despite_stale_prompt(self, mock_claude):
        # The night guard must consult LIVE state (is_night_mode_active), not the
        # frozen prompt bytes. Here the cached prompt still literally contains
        # '"Night": true' (a stale delta-frozen snapshot from before Night turned
        # off) but live state reports day — the periodic check MUST run. Reading the
        # prompt text, as the old code did, would suppress it until the 1h rebuild.
        mock_claude.return_value = Mock()
        pm = make_mock_prompt_manager(prompt='stale snapshot "Night": true', night=False)
        mgr = AutonomousActionManager.get_instance(pm, test_mode=True)

        call_count = [0]

        def patched_sleep(seconds):
            mgr.running = False  # stop after the first iteration's sleep

        def patched_activity_check(**kwargs):
            call_count[0] += 1
            return "NO"

        with patch('assistant.agents.AutonomousActionManager.time.sleep', side_effect=patched_sleep), \
             patch.object(mgr, 'activity_check', side_effect=patched_activity_check):
            mgr._run_periodic_check()

        assert call_count[0] == 1, "check must run when live state is not Night, even if prompt text is stale"
        mgr.running = True  # restore

    @patch('assistant.agents.AutonomousActionManager.create_llm')
    def test_scheduled_wakeup_runs_when_live_state_not_night_despite_stale_prompt(self, mock_claude):
        # Same invariant for the scheduled-wakeup path (line 392): a stale prompt
        # snapshot saying Night must not block a wakeup once live state is day.
        mock_claude.return_value = Mock()
        pm = make_mock_prompt_manager(prompt='stale "Night": true snapshot', night=False)
        mgr = AutonomousActionManager.get_instance(pm, test_mode=True)

        import threading
        check_called = threading.Event()

        def patched_activity_check(**kwargs):
            check_called.set()
            return "NO"

        created = []
        real_thread_cls = threading.Thread

        def capture_thread(*args, **kwargs):
            t = real_thread_cls(*args, **kwargs)
            created.append(t)
            return t

        with patch.object(mgr, 'activity_check', side_effect=patched_activity_check), \
             patch.object(mgr, '_remove_wakeup'), \
             patch('assistant.agents.AutonomousActionManager.time.sleep'), \
             patch('assistant.agents.AutonomousActionManager.threading.Thread', side_effect=capture_thread):
            mgr._start_wakeup_thread({"at": time.time() + 0.01, "scheduled_at": time.time(), "reason": ""})
            created[0].join(timeout=2.0)

        assert check_called.is_set(), "wakeup must run when live state is not Night, even with stale prompt text"

    @patch('assistant.agents.AutonomousActionManager.create_llm')
    def test_activity_check_strips_thinking_before_command_processing(self, mock_claude):
        """Commands inside <thinking> tags must not be executed (commit b6ded170).
        E.g. if the LLM thinks about 'turn_on(kitchen)' inside thinking tags,
        it should not actually trigger that command."""
        mock_llm = Mock()
        mock_llm.think_with_messages.return_value = (
            "<thinking>I could turn_on(kitchen) but I won't</thinking>\n"
            "Everything looks good, no action needed"
        )
        mock_claude.return_value = mock_llm
        pm = make_mock_prompt_manager()
        mgr = AutonomousActionManager.get_instance(pm, test_mode=True)

        with patch('assistant.commands.CommandProcessor.CommandProcessor') as mock_cp:
            mock_cp_inst = Mock()
            mock_cp_inst.command_queue = queue.Queue()
            mock_cp_inst.command_being_processed = False
            mock_cp_inst.command_running_in_background = False
            mock_cp_inst.command_reply = None
            mock_cp_inst.flush_buffer.return_value = None
            processed_tokens = []

            def track_process(token, dm):
                processed_tokens.append(token)
                return token
            mock_cp_inst.process.side_effect = track_process
            mock_cp.return_value = mock_cp_inst

            mgr.activity_check(prompt="test prompt")

        # The tokens fed to process() must NOT contain the thinking content
        all_text = "".join(processed_tokens)
        assert "turn_on(kitchen)" not in all_text
        assert "I could" not in all_text
        # But the actual response text should be there
        assert "looks" in all_text or "good" in all_text

    @patch('assistant.agents.AutonomousActionManager.create_llm')
    def test_activity_check_returns_no_when_lock_held(self, mock_claude):
        """activity_check returns 'NO' when the lock can't be acquired."""
        mock_claude.return_value = Mock()
        pm = make_mock_prompt_manager()
        mgr = AutonomousActionManager.get_instance(pm, test_mode=True)

        mock_lock = Mock()
        mock_lock.acquire.return_value = False
        mgr.activity_check_lock = mock_lock
        result = mgr.activity_check(prompt="test")
        assert result == "NO"

    def test_join_queue_with_timeout_returns_on_undrained_queue(self):
        q = queue.Queue()
        q.put("task")
        # queue has unfinished_tasks=1, never calling task_done -> should timeout, not hang
        AutonomousActionManager._join_queue_with_timeout(q, timeout=0.02)
        assert q.unfinished_tasks == 1

    @patch('assistant.agents.AutonomousActionManager.create_llm')
    def test_run_memory_curation_error_handling(self, mock_claude):
        mock_claude.return_value = Mock()
        pm = make_mock_prompt_manager()
        manager = AutonomousActionManager.get_instance(pm, test_mode=True)

        with patch('assistant.agents.AutonomousActionManager.LongTermMemoryCurator',
                   side_effect=RuntimeError("curation failed")):
            # Should not raise
            manager._run_memory_curation()

    @pytest.mark.parametrize("ran, reported", [({'interaction_count': 3}, [(3, 16, 2, 1)]), (None, [])])
    @patch('assistant.agents.AutonomousActionManager.create_llm')
    def test_a_curation_that_ran_reports_the_memory_size_and_a_skipped_one_does_not(self, mock_claude, ran, reported):
        mock_claude.return_value = Mock()
        manager = AutonomousActionManager.get_instance(make_mock_prompt_manager(), test_mode=True)
        module = 'assistant.agents.AutonomousActionManager'
        with patch(f'{module}.LongTermMemoryCurator.run', return_value=ran), \
             patch(f'{module}.MemoryFile.read', return_value="Marcus likes\ntea"), \
             patch(f'{module}.WatchList.items', return_value=[Mock(), Mock()]), \
             patch(f'{module}.Procedures.names', return_value=["descale-the-kettle"]), \
             patch(f'{module}.Telemetry') as telemetry:
            manager._run_memory_curation()
        assert [c.args for c in telemetry.get_instance().memory_size.call_args_list] == reported


@pytest.mark.unit
class TestVisionTierRouting:
    """18fac7aa/dbcc7c7e: a command reply that returns image blocks (e.g. a camera snapshot)
    must reach the vision tier. The autonomous loop runs both its initial check and its
    follow-up call through _llm_for, so an image-bearing message routes to the vision LLM and
    a text-only one stays on the cheaper autonomous LLM. Guards against the regression where
    the follow-up was hardcoded to self.llm and silently skipped the vision model."""

    IMAGE_MESSAGE = {"role": "user", "content": [
        {"type": "text", "text": "here's the snapshot"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}},
    ]}
    TEXT_MESSAGE = {"role": "user", "content": "is anything happening?"}

    @patch('assistant.agents.AutonomousActionManager.create_llm')
    def test_image_bearing_messages_route_to_vision_tier(self, mock_create):
        # First create_llm builds the autonomous tier in __init__; the vision tier is created
        # lazily on the first image-bearing check, so the patch must stay live for both.
        autonomous_llm, vision_llm = Mock(name="autonomous"), Mock(name="vision")
        mock_create.side_effect = [autonomous_llm, vision_llm]
        mgr = AutonomousActionManager.get_instance(make_mock_prompt_manager(), test_mode=True)
        assert mgr._llm_for([self.TEXT_MESSAGE, self.IMAGE_MESSAGE]) is vision_llm

    @patch('assistant.agents.AutonomousActionManager.create_llm')
    def test_text_only_messages_stay_on_autonomous_tier(self, mock_create):
        autonomous_llm = Mock(name="autonomous")
        mock_create.side_effect = [autonomous_llm, Mock(name="vision")]
        mgr = AutonomousActionManager.get_instance(make_mock_prompt_manager(), test_mode=True)
        assert mgr._llm_for([self.TEXT_MESSAGE]) is autonomous_llm

    @patch('assistant.agents.AutonomousActionManager.create_llm')
    def test_vision_llm_is_built_once_and_reused(self, mock_create):
        vision_llm = Mock(name="vision")
        mock_create.side_effect = [Mock(name="autonomous"), vision_llm]
        mgr = AutonomousActionManager.get_instance(make_mock_prompt_manager(), test_mode=True)
        assert mgr._llm_for([self.IMAGE_MESSAGE]) is vision_llm
        assert mgr._llm_for([self.IMAGE_MESSAGE]) is vision_llm  # cached, not rebuilt
        vision_llm.initialize.assert_called_once()


class TestWakeupPersistence:
    """Scheduled wake-ups survive a restart, each with the reason it was scheduled for."""

    @pytest.fixture(autouse=True)
    def use_tmp_schedule_file(self, tmp_path, monkeypatch):
        self.schedule_file = str(tmp_path / "scheduled_wakeups.json")
        monkeypatch.setattr('assistant.agents.AutonomousActionManager.SCHEDULE_FILE', self.schedule_file)

    @staticmethod
    def wakeup(at: float, reason: str = "") -> dict:
        return {"at": at, "scheduled_at": at - 600, "reason": reason}

    def read(self) -> list:
        with open(self.schedule_file) as f:
            return json.load(f)

    def test_save_wakeup_appends_to_existing(self):
        mgr, _ = make_manager()
        mgr._save_wakeup(self.wakeup(1000.0, "turn off the TV"))
        mgr._save_wakeup(self.wakeup(2000.0))
        assert [(w["at"], w["reason"]) for w in self.read()] == [(1000.0, "turn off the TV"), (2000.0, "")]

    def test_load_wakeups_returns_empty_on_missing_file(self):
        assert AutonomousActionManager._load_wakeups() == []

    def test_load_wakeups_returns_empty_on_corrupt_json(self):
        with open(self.schedule_file, "w") as f:
            f.write("{corrupt json!!")
        assert AutonomousActionManager._load_wakeups() == []

    def test_load_wakeups_reads_the_bare_timestamps_of_the_old_file_shape(self):
        """A file written before wake-ups carried a reason holds plain timestamps; those
        still restore, as reason-less wake-ups."""
        with open(self.schedule_file, "w") as f:
            json.dump([6000.0, {"at": 7000.0, "scheduled_at": 6500.0, "reason": "ping Marcus"}], f)
        wakeups = AutonomousActionManager._load_wakeups()
        assert [(w["at"], w["reason"]) for w in wakeups] == [(6000.0, ""), (7000.0, "ping Marcus")]

    def test_remove_wakeup_deletes_specific_entry(self):
        mgr, _ = make_manager()
        mgr._save_wakeup(self.wakeup(1000.0))
        mgr._save_wakeup(self.wakeup(2000.0))
        mgr._remove_wakeup(1000.0)
        assert [w["at"] for w in self.read()] == [2000.0]

    def test_remove_wakeup_noop_when_not_found(self):
        mgr, _ = make_manager()
        mgr._save_wakeup(self.wakeup(1000.0))
        mgr._remove_wakeup(9999.0)
        assert [w["at"] for w in self.read()] == [1000.0]

    @patch('assistant.agents.AutonomousActionManager.threading.Thread')
    @patch('assistant.agents.AutonomousActionManager.time.time')
    def test_restore_fires_the_wakeups_that_fell_due_while_down_oldest_first_and_keeps_them_on_disk(self, mock_time, mock_thread):
        """Regression for 2026-10-11: a 09:00 reminder was discarded on the evening restart.
        A missed wake-up now fires after startup; it leaves the file only when it has."""
        mock_time.return_value = 5000.0
        with open(self.schedule_file, "w") as f:
            json.dump([self.wakeup(3000.0, "late one"), self.wakeup(1000.0, "early one"), self.wakeup(2000.0)], f)

        mgr, _ = make_manager()
        with patch.object(mgr, '_start_wakeup_thread') as mock_start:
            mgr._restore_scheduled_wakeups()
            mock_start.assert_not_called()
        (missed,) = mock_thread.call_args.kwargs["args"]
        assert [(w["at"], w["reason"]) for w in missed] == [(1000.0, "early one"), (2000.0, ""), (3000.0, "late one")]
        mock_thread.return_value.start.assert_called_once()
        assert sorted(w["at"] for w in self.read()) == [1000.0, 2000.0, 3000.0]

    @patch('assistant.agents.AutonomousActionManager.threading.Thread')
    @patch('assistant.agents.AutonomousActionManager.time.time')
    def test_restore_tells_a_pending_wakeup_from_a_missed_one(self, mock_time, mock_thread):
        mock_time.return_value = 5000.0
        with open(self.schedule_file, "w") as f:
            json.dump([self.wakeup(4000.0, "missed"), self.wakeup(6000.0, "pending")], f)

        mgr, _ = make_manager()
        with patch.object(mgr, '_start_wakeup_thread') as mock_start:
            mgr._restore_scheduled_wakeups()
            assert [c.args[0]["reason"] for c in mock_start.call_args_list] == ["pending"]
        (missed,) = mock_thread.call_args.kwargs["args"]
        assert [w["reason"] for w in missed] == ["missed"]

    @patch('assistant.agents.AutonomousActionManager.MISSED_WAKEUP_STARTUP_GRACE_SECONDS', 0)
    def test_missed_wakeups_run_one_after_the_other_each_leaving_the_file_as_it_fires(self):
        now = time.time()
        missed = [self.wakeup(now - 7200, "first"), self.wakeup(now - 600, "second")]
        for w in missed:
            self._save(w)
        mgr, _ = make_manager()
        mgr.running = True
        mgr.prompt_manager.is_night_mode_active.return_value = False
        on_disk_when_fired = []

        def check(wakeup=None):
            on_disk_when_fired.append((wakeup["reason"], wakeup["missed"], [w["reason"] for w in self.read()]))
            return ""

        with patch.object(mgr, 'is_user_interaction_active', return_value=False), \
             patch.object(mgr, 'activity_check', side_effect=check):
            mgr._fire_missed_wakeups(missed)
        assert on_disk_when_fired == [("first", True, ["second"]), ("second", True, [])]

    @patch('assistant.agents.AutonomousActionManager.MISSED_WAKEUP_STARTUP_GRACE_SECONDS', 0)
    def test_a_missed_wakeup_is_dropped_at_night_like_one_on_time(self):
        now = time.time()
        wakeup = self.wakeup(now - 600, "nightly")
        self._save(wakeup)
        mgr, _ = make_manager()
        mgr.running = True
        mgr.prompt_manager.is_night_mode_active.return_value = True
        with patch.object(mgr, 'activity_check') as check:
            mgr._fire_missed_wakeups([wakeup])
            check.assert_not_called()
        assert self.read() == []

    @patch('assistant.agents.AutonomousActionManager.MISSED_WAKEUP_STARTUP_GRACE_SECONDS', 0)
    def test_one_failing_missed_check_does_not_stop_the_next(self):
        now = time.time()
        missed = [self.wakeup(now - 1200, "breaks"), self.wakeup(now - 600, "still runs")]
        mgr, _ = make_manager()
        mgr.running = True
        mgr.prompt_manager.is_night_mode_active.return_value = False
        with patch.object(mgr, 'is_user_interaction_active', return_value=False), \
             patch.object(mgr, 'activity_check', side_effect=[RuntimeError("API down"), ""]) as check:
            mgr._fire_missed_wakeups(missed)
        assert [c.kwargs["wakeup"]["reason"] for c in check.call_args_list] == ["breaks", "still runs"]

    def _save(self, wakeup: dict):
        AutonomousActionManager._write_wakeups(AutonomousActionManager._load_wakeups() + [wakeup])

    @patch('assistant.agents.AutonomousActionManager.threading.Thread')
    @patch('assistant.agents.AutonomousActionManager.time.time')
    def test_restore_reschedules_active_wakeups_with_their_reason(self, mock_time, _thread):
        mock_time.return_value = 5000.0
        with open(self.schedule_file, "w") as f:
            json.dump([self.wakeup(1000.0), self.wakeup(6000.0, "defrost the car"), self.wakeup(7000.0)], f)

        mgr, _ = make_manager()
        with patch.object(mgr, '_start_wakeup_thread') as mock_start:
            mgr._restore_scheduled_wakeups()
            restored = [(c.args[0]["at"], c.args[0]["reason"]) for c in mock_start.call_args_list]
            assert restored == [(6000.0, "defrost the car"), (7000.0, "")]
        assert [w["at"] for w in self.read()] == [1000.0, 6000.0, 7000.0]

    def test_schedule_persists_the_reason_and_hands_it_to_the_wake(self):
        mgr, _ = make_manager()
        with patch.object(mgr, '_start_wakeup_thread') as mock_start:
            mgr.schedule(5, "message Marcus 30 minutes before the launch")
        (wakeup,) = mock_start.call_args.args
        assert wakeup["reason"] == "message Marcus 30 minutes before the launch"
        assert 290 < wakeup["at"] - wakeup["scheduled_at"] < 310
        assert self.read()[0]["reason"] == wakeup["reason"]


class TestWakeupLateness:
    """A wake-up that fires long after its time tells the LLM so, and why."""

    def test_an_on_time_wakeup_carries_no_lateness(self):
        now = time.time()
        note = AutonomousActionManager._wakeup_note({"at": now - 5, "scheduled_at": now - 605, "reason": "turn off the TV"})
        assert "turn off the TV" in note and "late" not in note

    def test_a_missed_wakeup_says_the_server_was_down_and_how_long_ago_it_was_due(self):
        now = time.time()
        due = now - 11 * 3600 - 20 * 60
        note = AutonomousActionManager._wakeup_note({"at": due, "scheduled_at": due - 3600, "reason": "passport!",
                                                     "missed": True})
        assert "11 h 20 min late" in note and "server was down" in note
        assert time.strftime("%Y-%m-%d %H:%M", time.localtime(due)) in note

    def test_a_wakeup_held_back_by_the_user_says_so_instead(self):
        now = time.time()
        note = AutonomousActionManager._wakeup_note({"at": now - 900, "scheduled_at": now - 1800, "reason": ""})
        assert "15 min late" in note and "server was down" not in note

    @pytest.mark.parametrize("seconds, text", [(90, "1 min"), (3 * 3600 + 5 * 60, "3 h 5 min"), (26 * 3600 + 59, "1 d 2 h")])
    def test_lateness_reads_in_the_largest_fitting_unit(self, seconds, text):
        from assistant.agents.AutonomousActionManager import _describe_duration
        assert _describe_duration(seconds) == text


class TestScheduledWakeupReason:
    """The check a scheduled wake-up triggers is told what it was scheduled for: the exchange
    that scheduled it may have been trimmed from the history window by the time it fires."""

    @patch('assistant.agents.AutonomousActionManager.create_llm')
    def test_wake_thread_passes_its_entry_to_the_check(self, mock_claude):
        mock_claude.return_value = Mock()
        mgr = AutonomousActionManager.get_instance(make_mock_prompt_manager(night=False), test_mode=True)
        wakeup = {"at": time.time() + 0.01, "scheduled_at": time.time(), "reason": "turn off the TV"}
        seen = {}

        def capture(*args, **kwargs):
            seen.update(kwargs)
            return "NO"

        created = []
        real_thread_cls = threading.Thread

        def capture_thread(*args, **kwargs):
            t = real_thread_cls(*args, **kwargs)
            created.append(t)
            return t

        with patch.object(mgr, 'activity_check', side_effect=capture), \
             patch.object(mgr, '_remove_wakeup'), \
             patch('assistant.agents.AutonomousActionManager.time.sleep'), \
             patch('assistant.agents.AutonomousActionManager.threading.Thread', side_effect=capture_thread):
            mgr._start_wakeup_thread(wakeup)
            created[0].join(timeout=2.0)
        assert seen["wakeup"] is wakeup

    @staticmethod
    def manager_recording_its_prompt():
        """A manager whose LLM answers NO and whose saved prompt (what was sent) is captured."""
        pm = make_mock_prompt_manager()
        saved = {}
        pm.save_prompt_to_file.side_effect = lambda msgs: saved.update(m=list(msgs))
        mgr, mock_llm = make_manager(prompt_manager=pm)
        mock_llm.handles_thinking_internally.return_value = True
        mock_llm.think_with_messages.return_value = "NO"
        return mgr, saved

    @pytest.fixture(autouse=True)
    def idle_command_processor(self):
        with patch('assistant.commands.CommandProcessor.CommandProcessor') as mock_cp:
            inst = Mock()
            inst.command_queue = queue.Queue()
            inst.command_being_processed = False
            inst.command_running_in_background = False
            inst.command_reply = None
            inst.flush_buffer.return_value = None
            inst.process.return_value = None
            mock_cp.return_value = inst
            yield

    def test_check_prompt_ends_with_the_reason(self):
        mgr, saved = self.manager_recording_its_prompt()

        mgr.activity_check(wakeup={"at": time.time(), "scheduled_at": time.time() - 3600,
                                   "reason": "message Marcus 30 minutes before the launch"})

        last = saved["m"][-1]
        assert last["role"] == "user"
        assert "message Marcus 30 minutes before the launch" in last["content"]
        assert last["content"].index("AUTONOMOUS ACTION CHECK") < last["content"].index("SCHEDULED WAKE-UP")

    def test_check_without_a_wakeup_carries_no_wakeup_note(self):
        mgr, saved = self.manager_recording_its_prompt()

        mgr.activity_check()

        assert "SCHEDULED WAKE-UP" not in saved["m"][-1]["content"]

    def test_reasonless_wakeup_still_says_it_was_scheduled(self):
        """A wake-up from the old file shape, or from a model that gave no reason, must
        still tell the check it is a scheduled one so it looks for the pending timer."""
        mgr, saved = self.manager_recording_its_prompt()

        mgr.activity_check(wakeup={"at": time.time(), "scheduled_at": time.time() - 60, "reason": ""})

        assert "SCHEDULED WAKE-UP" in saved["m"][-1]["content"]


class TestUserInteractionGating:
    """Tests for the user-interaction gate that suppresses double replies
    (periodic/caffeinate/scheduled checks must defer while a user request is
    in flight or just finished)."""

    def test_started_marks_active(self):
        mgr, _ = make_manager()
        mgr.notify_user_interaction_started()
        assert mgr.is_user_interaction_active() is True
        assert mgr._user_requests_in_flight == 1
        assert mgr.last_user_interaction_at > 0.0

    def test_ended_decrements_counter_but_stays_active_during_quiet_period(self):
        mgr, _ = make_manager()
        mgr.notify_user_interaction_started()
        mgr.notify_user_interaction_ended()
        assert mgr._user_requests_in_flight == 0
        # Still within the quiet period — must read as active.
        assert mgr.is_user_interaction_active() is True

    def test_inactive_after_quiet_period_elapses(self):
        mgr, _ = make_manager()
        mgr.notify_user_interaction_started()
        mgr.notify_user_interaction_ended()
        # Backdate the last-interaction timestamp past the quiet window.
        mgr.last_user_interaction_at = time.time() - 999.0
        assert mgr.is_user_interaction_active() is False

    def test_overlapping_requests_keep_active_until_all_end(self):
        mgr, _ = make_manager()
        mgr.notify_user_interaction_started()
        mgr.notify_user_interaction_started()
        mgr.notify_user_interaction_ended()
        # Still has one request in flight — even with quiet_seconds=0 should be active.
        assert mgr.is_user_interaction_active(quiet_seconds=0) is True
        mgr.notify_user_interaction_ended()
        assert mgr.is_user_interaction_active(quiet_seconds=0) is False

    def test_ended_without_started_does_not_underflow(self):
        mgr, _ = make_manager()
        mgr.notify_user_interaction_ended()
        assert mgr._user_requests_in_flight == 0

    def test_stale_counter_self_heals(self):
        """A leaked in-flight counter (start without end) should auto-reset
        once the timestamp is older than USER_INTERACTION_STALE_COUNTER_SECONDS."""
        mgr, _ = make_manager()
        mgr.notify_user_interaction_started()
        assert mgr._user_requests_in_flight == 1
        # Backdate to simulate a leaked pairing that's been stale for 15 minutes
        mgr.last_user_interaction_at = time.time() - 900.0
        assert mgr.is_user_interaction_active() is False
        assert mgr._user_requests_in_flight == 0

    def test_stale_counter_does_not_reset_when_recent(self):
        """A non-zero in-flight counter within the stale window should remain active."""
        mgr, _ = make_manager()
        mgr.notify_user_interaction_started()
        # last_user_interaction_at is fresh (just set by started)
        assert mgr.is_user_interaction_active() is True
        assert mgr._user_requests_in_flight == 1

    def test_caffeinate_skips_iteration_when_user_active(self):
        """The caffeinate loop must skip activity_check while a user request is
        in flight or recently completed (regression test for the double-reply
        bug where a 3-min caffeinate wakeup could fire on top of a user turn)."""
        mgr, _ = make_manager()
        mgr.notify_user_interaction_started()  # active throughout

        def pass_time(seconds):
            mgr._caffeine_until -= seconds

        with patch.object(mgr, 'activity_check') as mock_check, \
                patch('assistant.agents.AutonomousActionManager.time.sleep', side_effect=pass_time), \
                patch('assistant.agents.AutonomousActionManager.threading.Thread') as mock_thread:
            captured = {}

            def capture(target, daemon):
                captured['target'] = target
                t = Mock()
                t.start = Mock()
                return t

            mock_thread.side_effect = capture
            mgr.caffeinate(duration_minutes=1.5, interval_minutes=1)
            captured['target']()  # run the loop body synchronously

        mock_check.assert_not_called()

    def test_caffeinate_runs_check_when_user_inactive(self):
        mgr, _ = make_manager()
        # No notify_user_interaction_started -> inactive

        def pass_time(seconds):
            mgr._caffeine_until -= seconds

        with patch.object(mgr, 'activity_check') as mock_check, \
                patch('assistant.agents.AutonomousActionManager.time.sleep', side_effect=pass_time), \
                patch('assistant.agents.AutonomousActionManager.threading.Thread') as mock_thread:
            captured = {}

            def capture(target, daemon):
                captured['target'] = target
                t = Mock()
                t.start = Mock()
                return t

            mock_thread.side_effect = capture
            mgr.caffeinate(duration_minutes=1.5, interval_minutes=1)
            captured['target']()

        mock_check.assert_called_once()


class TestAutonomousActionPromptInvariant:
    """Regression guard for e0b50665: the autonomous-action prompt must keep
    the thinking-tag containment rule that d45d4718 introduced for the
    request-path prompt.

    The bug: on 2026-05-17 a caffeinate iteration closed </thinking> after one
    paragraph, then continued reasoning in plain text before the final
    Telegram message. The cleanup regex only strips paired
    <thinking>...</thinking> blocks, so untagged middle prose was pushed
    verbatim to Telegram. The PromptManager test already pins this invariant
    for the user-facing path; this mirrors it for the autonomous path so a
    well-meaning edit to AUTONOMOUS_ACTION_PROMPT can't silently re-open the
    leak. The paragraph lives in THINKING_TAG_INSTRUCTION and reaches a model that
    does not think on its own (FORMATTED_AUTONOMOUS_ACTION_PROMPT); a model that
    handles thinking internally never emits the tag, so the rule is moot there."""

    def test_prompt_requires_all_reasoning_inside_thinking_tag(self):
        low = FORMATTED_AUTONOMOUS_ACTION_PROMPT.lower()
        # Must demand that ANY reasoning lives inside the tag — otherwise
        # untagged preambles slip past _remove_thinking() and reach Telegram.
        assert "all reasoning" in low, (
            "AUTONOMOUS_ACTION_PROMPT must require ALL reasoning to stay "
            "inside <thinking>; without this, untagged prose leaks to Telegram"
        )

    def test_prompt_constrains_post_thinking_content(self):
        # The fix also adds an explicit post-tag constraint: once </thinking>
        # is closed, only commands / final Telegram message / "NO" are
        # allowed. Pin the three allowed outputs by name.
        text = FORMATTED_AUTONOMOUS_ACTION_PROMPT
        low = text.lower()
        assert "</thinking>" in text, "prompt must reference the closing tag"
        assert "commands" in low, "post-tag rule must permit commands"
        # The literal "NO" sentinel is what _save_autonomous_result_to_history
        # checks for — quoting it in the prompt keeps that contract aligned.
        assert '"NO"' in text or "'NO'" in text, (
            "prompt must explicitly allow the 'NO' sentinel after </thinking>"
        )


class TestSaveAutonomousResultToHistory:
    """Tests for _save_autonomous_result_to_history (commit bbb6bc2)."""

    def test_saves_thinking_and_visible_text(self):
        mgr, _ = make_manager()
        mgr._save_autonomous_result_to_history(
            "<thinking>Humidity was 41%, below threshold</thinking>\nAll good."
        )
        call_args = mgr.prompt_manager.update_history.call_args
        saved = call_args[0][2]
        assert "<thinking>Humidity was 41%, below threshold</thinking>" in saved
        assert "All good." in saved

    def test_saves_thinking_only_when_visible_is_no(self):
        mgr, _ = make_manager()
        mgr._save_autonomous_result_to_history(
            "<thinking>Checked sensors, everything fine</thinking>\nNO"
        )
        call_args = mgr.prompt_manager.update_history.call_args
        saved = call_args[0][2]
        assert "<thinking>Checked sensors, everything fine</thinking>" in saved
        assert "NO" not in saved

    def test_no_history_when_no_thinking_and_result_is_no(self):
        mgr, _ = make_manager()
        mgr._save_autonomous_result_to_history("NO")
        mgr.prompt_manager.update_history.assert_not_called()

    def test_saves_visible_text_without_thinking(self):
        mgr, _ = make_manager()
        mgr._save_autonomous_result_to_history("Turned on the porch lights.")
        call_args = mgr.prompt_manager.update_history.call_args
        saved = call_args[0][2]
        assert "Turned on the porch lights." in saved

    def test_strips_xml_tags_from_visible(self):
        mgr, _ = make_manager()
        mgr._save_autonomous_result_to_history("<result>Action taken</result>")
        call_args = mgr.prompt_manager.update_history.call_args
        saved = call_args[0][2]
        assert "<result>" not in saved
        assert "Action taken" in saved


class TestThinkingTagInstruction:
    """The <thinking>-tag paragraph is for a model without thinking of its own; a model that
    handles thinking internally (Claude) gets the prompt without it — asking would have the
    check refused as reasoning extraction."""

    def test_dropped_for_a_model_that_thinks_on_its_own(self):
        llm = Mock()
        llm.handles_thinking_internally.return_value = True
        prompt = AutonomousActionManager._action_prompt(llm)
        assert "<thinking>" not in prompt and "thinking" not in prompt.lower().split("## response format")[1]
        assert "## Response Format\n\nIf you should execute commands:" in prompt
        assert str(PERIODIC_CHECK_INTERVAL_MINUTES) in prompt or f"{PERIODIC_CHECK_INTERVAL_MINUTES:.0f}" in prompt

    def test_a_bare_mock_counts_as_not_internal(self):
        """Mock().handles_thinking_internally() is a truthy Mock, not True: the paragraph stays."""
        assert "<thinking>" in AutonomousActionManager._action_prompt(Mock())


    @pytest.mark.parametrize("internal", [True, False])
    def test_activity_check_sends_the_prompt_built_for_its_llm(self, internal):
        """The refusals came from the check itself, not the helper: what reaches a model that
        thinks on its own must carry no <thinking> tag anywhere."""
        mgr, llm = make_manager()
        llm.handles_thinking_internally.return_value = internal
        sent = []
        # Snapshot at call time: the check appends the reply to the same list afterwards.
        llm.think_with_messages.side_effect = lambda messages: sent.extend(messages) or "NO"

        mgr.activity_check(prompt="test prompt")

        assert sent[-1] == {"role": "user", "content": AutonomousActionManager._action_prompt(llm)}
        assert any("<thinking>" in m["content"] for m in sent) is not internal

    def test_image_follow_up_on_a_self_thinking_vision_tier_loses_the_paragraph(self):
        """The prompt is built for the first round's model. When a later round carries camera
        images, _think hands it to the vision tier: if that one thinks on its own, the
        paragraph must be gone from what it receives — and nothing else."""
        mgr, llm = make_manager()
        llm.handles_thinking_internally.return_value = False
        vision = Mock()
        vision.handles_thinking_internally.return_value = True
        vision.think_with_messages.side_effect = lambda messages: [m.copy() for m in messages]
        mgr.llm_vision = vision
        messages = [{"role": "system", "content": "test prompt"},
                    {"role": "user", "content": "(Current time: 14:50)\n\n" + AutonomousActionManager._action_prompt(llm)},
                    {"role": "assistant", "content": "camera_snapshot(front door)"},
                    {"role": "user", "content": [{"type": "image", "source": {"type": "base64", "data": "abc"}}]}]

        sent = mgr._think(messages)

        vision.handles_thinking_internally.return_value = True
        assert sent[1]["content"] == "(Current time: 14:50)\n\n" + AutonomousActionManager._action_prompt(vision)
        assert sent[3] == messages[3]

    def test_text_round_on_a_tag_model_keeps_the_paragraph(self):
        mgr, llm = make_manager()
        llm.handles_thinking_internally.return_value = False
        llm.think_with_messages.side_effect = lambda messages: [m.copy() for m in messages]
        prompt = AutonomousActionManager._action_prompt(llm)

        sent = mgr._think([{"role": "system", "content": "test prompt"}, {"role": "user", "content": prompt}])

        assert sent[1]["content"] == prompt and "<thinking>" in prompt


@pytest.mark.unit
class TestWatchListInTheInstruction:
    def test_only_an_item_whose_poll_is_due_comes_whole_with_its_cadence_and_stamp(self):
        note = "order 4711 with the courier, " + "tracking page shows no movement yet, " * 4 + "door code needed"
        WatchList.save("Parcel from shop.example", note, "2026-03-12", every="6h")
        assert note not in AutonomousActionManager._action_prompt(Mock()), "just checked — not due yet"
        path = WatchList.WATCHLIST_DIR / "parcel-from-shop-example.md"
        path.write_text(path.read_text().replace(
            f"Checked: {WatchList.get('parcel-from-shop-example').checked}", "Checked: 2026-03-10 08:00"))
        prompt = AutonomousActionManager._action_prompt(Mock())
        assert note in prompt and "6h" in prompt and "2026-03-10 08:00" in prompt

    def test_an_item_without_a_cadence_is_left_to_load_watch(self):
        WatchList.save("Dentist for Charlotte", "12:30, bring the referral", "2026-03-12")
        prompt = AutonomousActionManager._action_prompt(Mock())
        assert "bring the referral" not in prompt and WatchList.NOTHING_DUE in prompt