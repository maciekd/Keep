"""A call runs where it starts a line or follows another call; inside a sentence it is the reply's text.

The cases are the shapes the production server's replies held between June and November 2026
(with the cast's names): the calls that were meant, and the mentions that ran.
"""
from unittest.mock import Mock

import pytest

from assistant.commands.CommandParser import CommandParser
from assistant.commands.CommandProcessor import CommandProcessor
from assistant.commands.formats.InvokeFormat import InvokeFormat, render_call
from assistant.commands.formats.KeepFormat import KeepFormat
from assistant.commands.formats.RescueFormat import RescueFormat
from assistant.commands.StoredReplyReader import StoredReplyReader

NAMES = ["turn_on", "turn_off", "dim", "play", "sonos_enqueue_and_play", "stop", "remember", "restart_server",
         "hue_pair", "try_pull_request", "pr_feedback", "load_toolkit", "find_phone", "save_watch"]

LAMP = 'turn_on("Kitchen lamp")'
RUNS = [("turn_on", ["Kitchen lamp"])]

MEANT = {
    "first thing in the reply": LAMP,
    "on a line of its own after prose": f"On it.\n{LAMP}\nDone.",
    "indented": f"On it.\n   {LAMP}",
    "wrapped in backticks": f"`{LAMP}`",
    "wrapped in a code tag": f"On it.\n<code>{LAMP}</code>",
    "with the answer after it on the line": f"{LAMP} The lamp is on.",
    "after a thinking block": f"<thinking>The lamp.\nYes.</thinking>{LAMP}",
}

MENTIONED = {
    "after words on its line": f"Want me to run {LAMP} now?",
    "in a code span inside a sentence": f"I wrote `{LAMP}` in my last message.",
    "in a code tag inside a sentence": f"I can <code>{LAMP}</code> for you.",
    "behind a dash": f"- {LAMP} — fails",
    "behind a bullet": f"• `{LAMP}` worked",
    "behind a number": f"1. {LAMP} finds the lamp",
    "in a table row": f"| 20:50 | {LAMP} | ok |",
    "behind a quote mark": f"> {LAMP}",
}


@pytest.fixture
def parser():
    return CommandParser(list(NAMES))


class TestWhereACallStands:
    @pytest.mark.parametrize("reply", MEANT.values(), ids=MEANT.keys())
    def test_a_call_that_starts_its_line_is_read(self, parser, reply):
        assert parser.extract_all_commands(reply) == RUNS
        assert parser.calls_in_prose(reply) == []

    @pytest.mark.parametrize("reply", MENTIONED.values(), ids=MENTIONED.keys())
    def test_a_call_inside_a_sentence_is_not_read(self, parser, reply):
        assert parser.extract_all_commands(reply) == []
        assert parser.calls_in_prose(reply) == RUNS

    def test_calls_that_follow_one_another_on_a_line_are_all_read(self, parser):
        both = [("turn_on", ["Hall"]), ("turn_off", ["Porch"])]
        assert parser.extract_all_commands('turn_on("Hall") turn_off("Porch")') == both
        assert parser.extract_all_commands('`turn_on("Hall")``turn_off("Porch")`The hall is lit.') == both

    def test_words_between_two_calls_on_a_line_end_the_run(self, parser):
        reply = 'turn_on("Hall") and then turn_off("Porch")'
        assert parser.extract_all_commands(reply) == [("turn_on", ["Hall"])]
        assert parser.calls_in_prose(reply) == [("turn_off", ["Porch"])]

    def test_every_call_of_a_sentence_is_a_mention_and_the_next_line_starts_fresh(self, parser):
        reply = 'You can run try_pull_request(72) and post with pr_feedback(72, "works").\nhue_pair()'
        assert parser.extract_all_commands(reply) == [("hue_pair", [])]
        assert parser.calls_in_prose(reply) == [("try_pull_request", ["72"]), ("pr_feedback", ["72", "works"])]

    def test_a_call_inside_the_value_of_a_call_that_runs_is_its_value(self, parser):
        reply = 'save_watch("lamps", """Each evening:\nturn_on("Hall")\nthen run hue_pair() once""")'
        calls = parser.extract_all_commands(reply)
        assert [name for name, _ in calls] == ["save_watch"]
        assert parser.calls_in_prose(reply) == []

    def test_a_call_never_closed_inside_a_sentence_takes_no_later_call_with_it(self, parser):
        reply = 'The outer hue_pair( was never closed.\nturn_on("Hall")\nDone :)'
        assert parser.extract_all_commands(reply) == [("turn_on", ["Hall"])]
        assert parser.calls_in_prose(reply) == []

    def test_a_name_inside_a_longer_word_is_no_call_and_no_mention(self, parser):
        reply = 'display("x")\nwindow.stop() and self.driver.stop()\nphone = location_find_phone("Charlotte")'
        assert parser.extract_all_commands(reply) == []
        assert parser.calls_in_prose(reply) == []

    def test_a_stray_character_glued_to_a_line_start_call_is_not_run(self, parser):
        """The one call of the five months that was meant and did not start its line."""
        assert parser.extract_all_commands('_load_toolkit("web")') == []
        assert parser.calls_in_prose('_load_toolkit("web")') == [("load_toolkit", ["web"])]

    def test_the_longer_of_two_names_that_end_alike_is_the_call(self, parser):
        calls = parser.extract_all_commands('sonos_enqueue_and_play("Kitchen", "F-2")')
        assert [name for name, _ in calls] == ["sonos_enqueue_and_play"]

    def test_a_name_added_after_the_parser_was_built_is_read(self, parser):
        assert parser.extract_all_commands('volume_up("Kitchen")') == []
        parser.possible_commands.append("volume_up")
        assert parser.extract_all_commands('volume_up("Kitchen")') == [("volume_up", ["Kitchen"])]


def dialog():
    dm = Mock()
    dm.is_killed = False
    return dm


@pytest.fixture
def processor():
    processor = CommandProcessor(test_suite_mode=True)
    processor.format = RescueFormat(KeepFormat(list(NAMES)), [InvokeFormat(list(NAMES), table={"dim": ["light_name", "dim_percentage"]})])
    processor.calls = []

    def dispatch(name, params, dm):
        processor.calls.append((name, params))
        processor.commands_dispatched += 1
    processor.invoke_command = dispatch
    yield processor
    processor.kill()


def stream(processor, reply: str, size: int, dm="a dialog") -> str:
    dm = dialog() if dm == "a dialog" else dm
    shown = [processor.process(reply[at:at + size], dm) for at in range(0, len(reply), size)]
    shown.append(processor.flush_buffer(dm))
    return "".join(text for text in shown if text)


def words(text: str) -> str:
    """The text without its spacing: how much whitespace around a call is shown depends on the chunks."""
    return "".join(text.split())


EVERY_CHUNKING = [1, 2, 3, 5, 8, 13, 1000]


@pytest.mark.parametrize("size", EVERY_CHUNKING)
class TestTheStream:
    @pytest.mark.parametrize("reply", [r for r in MEANT.values() if "<thinking>" not in r and "<code>" not in r])
    def test_a_call_that_starts_its_line_runs_and_is_not_shown(self, processor, size, reply):
        shown = stream(processor, reply, size)
        assert processor.calls == RUNS
        assert "turn_on" not in shown

    @pytest.mark.parametrize("reply", MENTIONED.values(), ids=MENTIONED.keys())
    def test_a_call_inside_a_sentence_is_shown_and_does_not_run(self, processor, size, reply):
        shown = stream(processor, reply, size)
        assert processor.calls == []
        assert words(shown) == words(reply)

    def test_a_sentence_of_mentions_then_a_real_call(self, processor, size):
        reply = 'I could run try_pull_request(72), then pr_feedback(72, "ok").\n\nturn_on("Hall")\nThe hall is lit.'
        shown = stream(processor, reply, size)
        assert processor.calls == [("turn_on", ["Hall"])]
        assert words(shown) == words(reply.replace('turn_on("Hall")', ""))

    def test_calls_that_follow_one_another_on_a_line_all_run(self, processor, size):
        shown = stream(processor, 'On it.\n  dim("Hall", 40) dim("Porch", 10)\nBoth dimmed.', size)
        assert processor.calls == [("dim", ["Hall", "40"]), ("dim", ["Porch", "10"])]
        assert words(shown) == "Onit.Bothdimmed."

    def test_a_call_after_the_answer_on_the_line_of_a_call_does_not_run(self, processor, size):
        stream(processor, 'turn_on("Hall") Done. Say the word and I run restart_server() too.', size)
        assert processor.calls == [("turn_on", ["Hall"])]

    def test_a_word_that_begins_like_a_command_is_shown_whole(self, processor, size):
        reply = "I will stop here, play it safe and remember the next step."
        assert stream(processor, reply, size) == reply and processor.calls == []

    def test_a_block_of_another_syntax_still_runs_inside_a_sentence(self, processor, size):
        block = render_call("dim", [("light_name", "Hall"), ("dim_percentage", "20")])
        stream(processor, f"Dimming it: {block}", size)
        assert processor.calls == [("dim", ["Hall", "20"])]


class TestNobodyIsCalledBack:
    """A reply that mentioned a command is over when it ends: no round goes back to the model for it."""

    def test_a_mention_leaves_nothing_for_a_follow_up_round(self, processor):
        stream(processor, "Want me to run restart_server() after you have edited the file?", 7)
        assert processor.unterminated_command_report is None and processor.command_reply is None

    def test_a_call_the_reply_never_closed_is_still_reported_alone(self, processor):
        stream(processor, 'I ran hue_pair() before.\nturn_on("Hall', 1000)
        report = processor.take_unterminated_command_report()
        assert "turn_on" in report and "hue_pair" not in report

    def test_the_next_reply_through_the_processor_starts_on_a_clear_line(self, processor):
        stream(processor, "Want me to run restart_server() now?", 1000)
        processor.start_new_reply()
        stream(processor, 'turn_on("Hall")', 1000)
        assert processor.calls == [("turn_on", ["Hall"])]


class TestTheStoredTurn:
    """Whatever reads a stored reply back reads the calls the stream ran, and no others."""

    def test_the_cut_counts_the_calls_that_ran_and_not_a_mention_before_them(self):
        fmt = KeepFormat(list(NAMES))
        reply = 'I ran hue_pair() before.\nturn_on("Hall")\nturn_on("Porch")\nturn_on("Attic")\nAll lit.'
        kept = fmt.cut(reply, 2)
        assert fmt.extract_all(kept) == [("turn_on", ["Hall"]), ("turn_on", ["Porch"])]
        assert kept.startswith("I ran hue_pair() before.") and "Attic" not in kept

    def test_a_rescued_block_inside_a_sentence_is_stored_as_a_call_that_starts_its_line(self):
        names = list(NAMES)
        fmt = RescueFormat(KeepFormat(names), [InvokeFormat(names, table={"dim": ["light_name", "dim_percentage"]})])
        stored = fmt.rewritten("Dimming it: " + render_call("dim", [("light_name", "Hall"), ("dim_percentage", "20")]))
        assert "<invoke" not in stored
        assert KeepFormat(names).extract_all(stored) == [("dim", ["Hall", "20"])]

    def test_a_chat_redrawn_from_the_history_keeps_the_mention_in_the_prose(self):
        read = StoredReplyReader().read('Say the word and I run restart_server() for you.\nturn_on("Kitchen lamp")')
        assert "restart_server()" in read.prose and "turn_on" not in read.prose
        assert len(read.before) + len(read.after) == 1

