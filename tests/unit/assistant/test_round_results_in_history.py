"""A round's results from the commands that produced them to the history row: the commands
publish the results one by one beside the reply, the processor and the request frame the reply,
and the history shares its cap between the results it finds in the framed text. Each step is
tested alone against a mock of the next; these tests run them together, since the cap is only
shared while every framing leaves the results standing as they were published."""
from unittest.mock import Mock, patch

import pytest

from assistant.RequestProcessor import (_build_error_round_reply, _build_repeated_round_reply,
                                        _command_result_for_history)
from assistant.commands.CommandProcessor import CommandProcessor
from assistant.prompt.MessageHistory import MessageHistory
from tests.unit.waiting import FAILURE_BOUND

CAP = MessageHistory.COMMAND_RESULT_CHARS
PAGE = "P" * (CAP * 3)
BOARD = "B" * (CAP * 2)
IMAGE = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "xx"}}


@pytest.fixture
def processor():
    processor = CommandProcessor(test_suite_mode=True)
    with patch.object(processor, "_log_command_reply"):
        yield processor


def run_round(processor, *results):
    """The round's commands, each returning one of *results* (an exception is raised)."""
    for result in results:
        command = Mock(side_effect=result) if isinstance(result, Exception) else Mock(return_value=result)
        processor.commands.background_command(command, None)
    assert processor.commands.wait_for_background(timeout=FAILURE_BOUND)


def stored(result: str, parts: list[str]) -> str:
    with patch.object(MessageHistory, "_load_history"):
        history = MessageHistory()
    history.history = []
    with patch.object(history, "_save_history_entry") as save, patch.object(history, "_notify_changed"):
        history.add_command_result(result, parts=parts)
    return save.call_args.args[1]


@pytest.mark.unit
class TestARoundsResultsShareTheHistoryCap:
    def test_two_pages_and_a_short_result_reach_the_history_with_the_short_one_whole(self, processor):
        run_round(processor, PAGE, BOARD, "Kimi 46")
        row = stored(_command_result_for_history(f"result: \n{processor.command_reply}"), processor.result_parts)
        assert row.startswith("result: \n")
        assert "Kimi 46" in row
        assert row.count("P") + row.count("B") == CAP - len("Kimi 46")
        assert abs(row.count("P") - row.count("B")) <= 1

    def test_a_snapshot_beside_a_page_keeps_its_caption_and_the_page_its_share(self, processor):
        """A block result (a camera snapshot) makes the reply a block list: its caption is a
        result of the round like the page, and the image is only noted."""
        run_round(processor, [{"type": "text", "text": "Front door camera:"}, IMAGE], PAGE)
        processor.add_note("The reply was cut at the first command.")
        assert isinstance(processor.command_reply, list)
        row = stored(_command_result_for_history(processor.command_reply), processor.result_parts)
        assert "The reply was cut at the first command." in row
        assert "Front door camera:" in row
        assert row.count("P") == CAP - len("Front door camera:")
        assert f"{len(PAGE):,}" in row, "the cut page says how much the turn read"
        assert "xx" not in row

    def test_a_failed_command_beside_two_pages_leaves_the_cap_shared(self, processor):
        """The error round frames the reply with what failed and what ran: the pages still
        stand in it as published, so neither is pushed out by the other."""
        run_round(processor, PAGE, RuntimeError("the shop timed out"), BOARD)
        assert processor.command_reply_contains_error
        framed = _build_error_round_reply(processor.command_reply, ["open_url(a)", "open_url(b)"])
        row = stored(_command_result_for_history(framed), processor.result_parts)
        assert "the shop timed out" in row and "open_url(b)" in row
        assert min(row.count("P"), row.count("B")) >= (CAP - 200) // 2
        assert row.count("P") + row.count("B") <= CAP

    def test_a_repeated_round_keeps_its_closing_note_after_the_cut_page(self, processor):
        run_round(processor, PAGE)
        framed = _build_repeated_round_reply(processor.command_reply)
        row = stored(_command_result_for_history(framed), processor.result_parts)
        assert row.count("P") == CAP
        assert row.endswith(framed[-60:]), "what the round was told after the result is not cut away with it"

    def test_the_next_reply_shares_the_cap_between_its_own_results_only(self, processor):
        """A consumed reply takes its results with it: the round after it would otherwise cap
        its page against results that are no longer in its text."""
        run_round(processor, BOARD)
        processor.clear_command_reply()
        run_round(processor, PAGE)
        assert processor.result_parts == [PAGE]
        row = stored(_command_result_for_history(f"result: \n{processor.command_reply}"), processor.result_parts)
        assert row.count("P") == CAP and "B" not in row.replace("result", "")
