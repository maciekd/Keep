"""A closed call block that names no function, in the formats the Qwen tests do not walk: the
rule sits in CommandProcessor, so DSML and invoke replies are reported the same way."""
from unittest.mock import Mock

import pytest

from assistant.commands.CommandProcessor import CommandProcessor
from assistant.commands.formats import DsmlFormat, InvokeFormat, QwenFormat

TABLE = {"turn_on": ["light_name"], "dim": ["light_name", "dim_percentage"]}

# The parameters of a call, in each format's own block, with the tag that names the function left out.
NAMELESS = {
    "qwen": (QwenFormat.QwenFormat,
             f"{QwenFormat.OPEN}\n<parameter=light_name>\n%s\n</parameter>\n</function>\n{QwenFormat.CLOSE}"),
    "dsml": (DsmlFormat.DsmlFormat,
             f'{DsmlFormat.OPEN}\n{DsmlFormat.PARAMETER_OPEN} name="light_name" string="true">%s'
             f"{DsmlFormat.PARAMETER_CLOSE}\n{DsmlFormat.CLOSE}"),
    "invoke": (InvokeFormat.InvokeFormat,
               f'{InvokeFormat.OPEN}\n{InvokeFormat.PARAMETER_OPEN} name="light_name">%s'
               f"{InvokeFormat.PARAMETER_CLOSE}\n{InvokeFormat.CLOSE}"),
}


@pytest.fixture(params=sorted(NAMELESS))
def nameless(request):
    """(a processor reading the format, the block around a lamp's name)."""
    format_class, block = NAMELESS[request.param]
    processor = CommandProcessor(test_suite_mode=True)
    processor.format = format_class(processor.possible_commands, table=TABLE, variadic=set())
    processor.calls = []

    def ran(name, params, dialog_manager):
        processor.calls.append((name, params))
        processor.commands_dispatched += 1
    processor.invoke_command = ran
    yield processor, block
    processor.kill()


def stream(processor, chunks) -> str:
    dialog = Mock(is_killed=False)
    out = ""
    for chunk in chunks:
        out += processor.process(chunk, dialog) or ""
    return out + (processor.flush_buffer(dialog) or "")


@pytest.mark.parametrize("pieces", [list, lambda reply: [reply]], ids=["by the character", "in one chunk"])
def test_the_block_is_reported_in_every_format_and_the_prose_around_it_is_shown(nameless, pieces):
    processor, block = nameless
    out = stream(processor, pieces(f"On it.\n{block % 'Kitchen lamp'}\nAnything else?"))
    report = processor.take_unterminated_command_report()
    assert processor.calls == []
    assert "On it." in out and "Anything else?" in out and "Kitchen lamp" not in out
    assert "Kitchen lamp" in report
    assert "On it." not in report and "Anything else?" not in report


def test_two_such_blocks_are_reported_in_one_round(nameless):
    processor, block = nameless
    stream(processor, [f"{block % 'Kitchen lamp'}\nand\n{block % 'Hall lamp'}"])
    report = processor.take_unterminated_command_report()
    assert "Kitchen lamp" in report and "Hall lamp" in report
    assert processor.take_unterminated_command_report() is None

