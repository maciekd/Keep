"""The post-call hold of the command processor: on a backend whose model may resume reasoning
in the content stream after a call (`AbstractLLM.resumes_thinking_after_calls`, ds4), the
text after a dispatched call is held back until a bare </think> shows it was reasoning —
dropped, and never parsed for calls — or the stream ends and it was the reply.

Prod 2026-11-08 23:10: a Telegram reply began with the model's reasoning ("Keep reply short,
in English … Yes, briefly — it's money.</think>Good — so the Frame goes back tomorrow"), and
at 23:12 the reasoning repeated the call it followed, so the command ran twice."""
from unittest.mock import Mock

import pytest

from assistant.commands.CommandProcessor import CommandProcessor
from assistant.commands.formats.KeepFormat import KeepFormat
from assistant.commands.formats.RescueFormat import RescueFormat


class ResumingLLM:
    def resumes_thinking_after_calls(self):
        return True


class PlainLLM:
    def resumes_thinking_after_calls(self):
        return False


def dialog(ai):
    dm = Mock()
    dm.ai = ai
    dm.is_killed = False
    dm.web_request_handler.wait_so_that_response_timing_is_natural = Mock(return_value=True)
    return dm


@pytest.fixture
def processor():
    processor = CommandProcessor(test_suite_mode=True)
    processor.format = RescueFormat.wrap(KeepFormat(processor.possible_commands))
    processor.calls = []
    dispatch = processor.invoke_command
    processor.invoke_command = lambda name, params, dm: (processor.calls.append((name, params)), dispatch(name, params, dm))
    yield processor
    processor.kill()


def feed(processor, dm, *chunks) -> list[str | None]:
    shown = [processor.process(chunk, dm) for chunk in chunks]
    shown.append(processor.flush_buffer(dm))
    processor.command_queue.join()
    return shown


CALL = "remember(Marcus sends the frame back tomorrow)\n"
REASONING = "He's sorted it. Keep reply short, in English.\n\nShould I mention the freight? Yes, briefly."
REPLY = "Good — so the frame goes back tomorrow."


@pytest.mark.unit
class TestTheHoldOnAResumingBackend:
    def test_reasoning_closed_with_a_bare_tag_after_a_call_never_reaches_the_reply(self, processor):
        dm = dialog(ResumingLLM())
        shown = feed(processor, dm, CALL, REASONING[:20], REASONING[20:] + "</th", "ink>" + REPLY[:10], REPLY[10:])
        text = "".join(s for s in shown if s)
        assert text.strip() == REPLY
        assert "Keep reply short" not in text and "</think>" not in text
        assert [name for name, _ in processor.calls] == ["remember"]

    def test_the_hold_yields_nothing_until_the_tag_is_seen(self, processor):
        dm = dialog(ResumingLLM())
        shown = feed(processor, dm, CALL, REASONING, "</think>", REPLY)
        # Nothing shown for the call, nothing for the reasoning, nothing for the tag itself.
        assert not any(shown[:3])
        assert shown[3].strip() == REPLY

    def test_a_call_repeated_inside_the_dropped_reasoning_does_not_run(self, processor):
        """Prod 2026-11-08 23:12: `self_analysis` ran twice because the reasoning repeated it."""
        dm = dialog(ResumingLLM())
        feed(processor, dm, "volume_up(kitchen)\n", "Short line while it runs. volume_up(kitchen)\n", "</think>", "Running.")
        assert processor.calls == [("volume_up", ["kitchen"])]
        assert len(processor.dropped_thinking) == 1

    def test_text_after_a_call_with_no_tag_is_the_reply_and_is_released_at_the_end(self, processor):
        dm = dialog(ResumingLLM())
        shown = feed(processor, dm, CALL, "Noted, ", "I'll remind you.")
        assert not any(shown[:-1])          # held while the stream runs
        assert shown[-1].strip() == "Noted, I'll remind you."
        assert processor.dropped_thinking == []

    def test_a_reply_that_ends_inside_the_second_thinking_block_shows_none_of_it(self, processor):
        """Prod 2026-11-05 15:57: the trailing text of an autonomous check was reasoning and </think>."""
        dm = dialog(ResumingLLM())
        shown = feed(processor, dm, CALL, "Nothing to add here.", "</think>")
        assert not any(shown)
        assert len(processor.dropped_thinking) == 1

    def test_a_second_call_after_the_reasoning_runs_as_usual(self, processor):
        dm = dialog(ResumingLLM())
        feed(processor, dm, "volume_up(kitchen)\n", "Louder, then the lights.</think>", "volume_down(kitchen)\n", "Done.")
        assert [name for name, _ in processor.calls] == ["volume_up", "volume_down"]

    def test_the_stored_turn_loses_the_dropped_block_and_keeps_the_call(self, processor):
        dm = dialog(ResumingLLM())
        raw = CALL + REASONING + "</think>" + REPLY
        feed(processor, dm, raw)
        assert processor.without_dropped_thinking(raw) == CALL + REPLY   # the newline after the call stays

    def test_a_block_the_raw_text_does_not_hold_verbatim_is_left_alone(self, processor):
        dm = dialog(ResumingLLM())
        feed(processor, dm, CALL + "thought</think>" + REPLY)
        raw = CALL + "`thought`</think>" + REPLY     # the dialog stripped backticks before the processor saw it
        assert processor.without_dropped_thinking(raw) == raw

    def test_text_before_any_call_streams_live(self, processor):
        dm = dialog(ResumingLLM())
        shown = feed(processor, dm, "Sure, one moment.\n", CALL, "Done.")
        assert shown[0] == "Sure, one moment.\n"
        assert shown[-1].strip() == "Done."


@pytest.mark.unit
class TestOtherBackendsAreUntouched:
    def test_text_after_a_call_streams_live_on_a_plain_backend(self, processor):
        dm = dialog(PlainLLM())
        shown = feed(processor, dm, CALL, "Noted, ", "I'll remind you.")
        assert shown[1].strip() == "Noted," and shown[2] == "I'll remind you."
        assert processor.dropped_thinking == []

    def test_a_bare_tag_on_a_plain_backend_passes_through_as_today(self, processor):
        """The hold is off, so the tag is ordinary text — the flag, not the tag, decides."""
        dm = dialog(PlainLLM())
        text = "".join(s for s in feed(processor, dm, CALL, "text</think>more") if s)
        assert text.strip() == "text</think>more"

    def test_a_dialog_without_an_llm_handle_does_not_hold(self, processor):
        shown = feed(processor, None, CALL, "Done.")
        assert shown[1].strip() == "Done."
