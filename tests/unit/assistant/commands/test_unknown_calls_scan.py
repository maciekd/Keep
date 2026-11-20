"""The scan for a line that calls what is no command (`CommandParser.unknown_calls`) reads a
triple-quoted block as one value: where a block ends, no other one begins.

The scan runs over every reply at its end, on the request's thread, so a reply it cannot get
through holds the turn — and, the regex engine keeping the interpreter to itself, the server."""
from assistant.commands.CommandParser import CommandParser

NAMES = ["save_procedure", "remember", "turn_on"]


def test_a_call_line_behind_a_line_that_goes_on_after_its_block_is_found():
    reply = '_remember("""The sauna is at 80 degrees.""") Noted.\n_load_toolkit("""web""")'
    assert CommandParser(list(NAMES)).unknown_calls(reply) == [("_load_toolkit", '_load_toolkit("""web""")')]


def test_a_block_that_runs_over_lines_is_still_one_value():
    """What the block alternative is for: a call of a command whose value holds a line that
    looks like a call of something else."""
    reply = 'save_procedure("lamps", """Each evening:\n_act("35")\nthen wait.""")'
    assert CommandParser(list(NAMES)).unknown_calls(reply) == []
