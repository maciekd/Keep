"""The scan for a line that calls what is no command (`CommandParser.unknown_calls`) reads a
triple-quoted block as one value: where a block ends, no other one begins.

The scan runs over every reply at its end, on the request's thread, so a reply it cannot get
through holds the turn — and, the regex engine keeping the interpreter to itself, the server."""
import subprocess
import sys

import pytest

from assistant.commands import CommandParser as parser_module
from assistant.commands.CommandParser import CommandParser
from tests.unit.waiting import FAILURE_BOUND

NAMES = ["save_procedure", "remember", "turn_on"]

SCAN = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("CommandParser", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(module.CommandParser(sys.argv[2].split(",")).unknown_calls(sys.stdin.read()))
"""


@pytest.mark.xfail(strict=True, reason="unknown_calls: _CALL_LINE_RE lets a block open at the quotes that close "
                                       "the one before, so a line with text behind its call is matched together "
                                       "with the next line: the headless call there is not found, and the "
                                       "first line is reported with it")
def test_a_call_line_behind_a_line_that_goes_on_after_its_block_is_found():
    reply = '_remember("""The sauna is at 80 degrees.""") Noted.\n_load_toolkit("""web""")'
    assert CommandParser(list(NAMES)).unknown_calls(reply) == [("_load_toolkit", '_load_toolkit("""web""")')]


@pytest.mark.xfail(strict=True, reason="unknown_calls: _CALL_LINE_RE backtracks through every pairing of the "
                                       "triple quotes behind a line that is no call — the time doubles with each "
                                       "block (20 such lines: seconds, 40: never), with the GIL held")
def test_a_reply_of_many_blocks_with_text_behind_each_is_scanned_at_all():
    """Forty saves, each with a word behind it on its line, and one headless call at the end:
    the scan is run in a process of its own, since a regex that does not come back cannot be
    stopped from inside."""
    reply = "\n".join(f'save_procedure("step {n}", """Open the hall door.""") Saved.' for n in range(40))
    reply += '\n_act("35")'
    done = subprocess.run([sys.executable, "-c", SCAN, parser_module.__file__, ",".join(NAMES)], input=reply,
                          capture_output=True, text=True, timeout=FAILURE_BOUND)
    assert done.stdout.strip() == repr([("_act", '_act("35")')]), done.stderr


def test_a_block_that_runs_over_lines_is_still_one_value():
    """What the block alternative is for: a call of a command whose value holds a line that
    looks like a call of something else."""
    reply = 'save_procedure("lamps", """Each evening:\n_act("35")\nthen wait.""")'
    assert CommandParser(list(NAMES)).unknown_calls(reply) == []
