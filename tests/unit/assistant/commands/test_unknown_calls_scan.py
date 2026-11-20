"""The scan for a line that calls what is no command (`CommandParser.unknown_calls`) reads a
triple-quoted block as one value: where a block ends, no other one begins.

The scan runs over every reply at its end, on the request's thread, so a reply it cannot get
through holds the turn — and, the regex engine keeping the interpreter to itself, the server."""
import pytest

from assistant.commands.CommandParser import CommandParser

NAMES = ["save_procedure", "remember", "turn_on"]


@pytest.mark.xfail(strict=True, reason="unknown_calls: _CALL_LINE_RE lets a block open at the quotes that close "
                                       "the one before, so a line with text behind its call is matched together "
                                       "with the next line: the headless call there is not found, and the "
                                       "first line is reported with it. The same reading of a closing quote "
                                       "as an opening one makes the scan try every pairing of the triple "
                                       "quotes behind a line that is no call: the time doubles with each "
                                       "block (16 such lines: 0.2 s, 40: never), with the GIL held — no "
                                       "test holds that, since a pin of a scan that does not come back "
                                       "can only wait out its bound")
def test_a_call_line_behind_a_line_that_goes_on_after_its_block_is_found():
    reply = '_remember("""The sauna is at 80 degrees.""") Noted.\n_load_toolkit("""web""")'
    assert CommandParser(list(NAMES)).unknown_calls(reply) == [("_load_toolkit", '_load_toolkit("""web""")')]


def test_a_block_that_runs_over_lines_is_still_one_value():
    """What the block alternative is for: a call of a command whose value holds a line that
    looks like a call of something else."""
    reply = 'save_procedure("lamps", """Each evening:\n_act("35")\nthen wait.""")'
    assert CommandParser(list(NAMES)).unknown_calls(reply) == []
