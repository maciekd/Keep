"""StoredReplyReader: a stored assistant reply read back as its stream showed it — the
prose without the command calls, an echo per call."""
import pytest

from assistant.commands.StoredReplyReader import StoredReplyReader


@pytest.fixture(scope="module")
def reader():
    return StoredReplyReader()


def test_a_reply_without_commands_is_all_prose(reader):
    reply = reader.read("Parked at <b>Kyrkogatan</b>, spot <code>B4</code>.")
    assert (reply.prose, reply.before, reply.after) == ("Parked at <b>Kyrkogatan</b>, spot <code>B4</code>.", [], [])


def test_a_block_quoted_argument_is_taken_out_whole(reader):
    """The text of a save_watch — parentheses, quotes and line breaks in it — is the LLM's, not the user's."""
    reply = reader.read('save_watch("ferry-strike", """Strike (day 2), "no boats"\nuntil further notice.""", '
                        '"2026-10-07", "6h")\n\nOn the watch list.')
    assert reply.prose == "On the watch list."
    assert len(reply.before) == 1 and "ferry-strike" in reply.before[0] and "no boats" not in reply.before[0]


def test_echoes_keep_their_place_around_the_prose(reader):
    reply = reader.read('google("ferry strike")\nBack on Friday.\n\nremember(Marcus follows the ferry strike)')
    assert reply.prose == "Back on Friday."
    assert len(reply.before) == 1 and "ferry strike" in reply.before[0]
    assert len(reply.after) == 1


@pytest.mark.parametrize("wrapped", ['`google("ferry strike")`', '<code>google("ferry strike")</code>',
                                     '```\ngoogle("ferry strike")\n```'])
def test_the_wrapping_of_a_call_goes_with_it(reader, wrapped):
    reply = reader.read(f"Looking.\n{wrapped}\nDone.")
    assert reply.prose == "Looking.\n\nDone." and len(reply.after) == 1


def test_code_tags_in_the_prose_after_a_call_survive(reader):
    assert reader.read('google("ferry strike")\nSpot <code>B4</code>.').prose == "Spot <code>B4</code>."


def test_a_call_that_never_closes_stays_prose(reader):
    reply = reader.read("I could run google( but will not")
    assert reply.prose == "I could run google( but will not" and not reply.before and not reply.after



class TestTheInvokeFormat:
    """A reply stored under Claude's own `<function_calls>` syntax: the reader parses it with the
    active format, and a block holding several invokes — the format's native shape — is read
    call by call."""

    @pytest.fixture
    def invoke_reader(self):
        from assistant.commands.formats.CommandFormats import CommandFormats
        from assistant.commands.formats.InvokeFormat import InvokeFormat
        CommandFormats.set_active(InvokeFormat, "a test")
        try:
            yield StoredReplyReader()
        finally:
            CommandFormats.reset()

    @staticmethod
    def block(*invokes: tuple[str, str]) -> str:
        body = "".join(f'<invoke name="{name}">\n<parameter name="device">{value}</parameter>\n</invoke>\n'
                       for name, value in invokes)
        return f"<function_calls>\n{body}</function_calls>"

    @staticmethod
    def guarded(reader, text, rounds=20):
        """`read` with a bound on its rounds: the reader loops forever on some blocks (the bug
        below), and a hung test would hold its worker. A count of the format's `find_start`
        calls, not an alarm: a signal lands wherever the loop happens to be, and
        `Commands.describe`, called every round, swallows any exception raised inside it — a
        one-shot alarm caught there left the loop running for good."""
        find_start = reader._format.find_start
        left = iter(range(rounds))

        def bounded(remaining):
            if next(left, None) is None:
                raise AssertionError(f"StoredReplyReader.read did not return in {rounds} rounds")
            return find_start(remaining)

        reader._format.find_start = bounded
        return reader.read(text)

    def test_a_block_with_one_invoke_reads_like_a_keep_call(self, invoke_reader):
        reply = self.guarded(invoke_reader, f"On it.\n{self.block(('turn_on', 'Kitchen lamp'))}\nDone.")
        assert reply.prose == "On it.\n\nDone." and reply.before == []
        assert len(reply.after) == 1 and "Kitchen lamp" in reply.after[0]

    @pytest.mark.xfail(strict=True, reason="the reader finds the end of a call by the length of the format's "
                                           "remainder, but a block's further invokes come back re-rendered, so "
                                           "the cut lands inside the block: raw XML in the prose, one echo short")
    def test_a_block_with_two_invokes_gives_two_echoes_and_no_xml(self, invoke_reader):
        text = f"On it.\n{self.block(('turn_on', 'Kitchen lamp'), ('turn_off', 'Hall lamp'))}\nDone."
        reply = self.guarded(invoke_reader, text)
        assert reply.prose == "On it.\n\nDone."
        assert len(reply.after) == 2 and "Kitchen lamp" in reply.after[0] and "Hall lamp" in reply.after[1]

    @pytest.mark.xfail(strict=True, reason="a block whose re-rendered remainder is longer than the text makes the "
                                           "cut negative, the remainder never shrinks and read() loops forever")
    def test_a_block_of_short_invokes_does_not_hang_the_reader(self, invoke_reader):
        text = ('<function_calls><invoke name="turn_on"></invoke><invoke name="turn_off"></invoke>'
                '<invoke name="turn_on"></invoke></function_calls>')
        reply = self.guarded(invoke_reader, text)
        assert reply.prose == "" and len(reply.before) + len(reply.after) == 3
