"""A call that goes on after the closing quote of a value is not run (prod 2026-11-17: a mail
body written as a Python expression — a triple-quoted block, `.replace(…) if False else` and a
second block — was sent as one body holding both blocks and the code between them)."""
from unittest.mock import patch

import pytest

from assistant.commands.CommandParser import CommandParser
from assistant.commands.CommandProcessor import CommandProcessor
from assistant.commands.formats.KeepFormat import KeepFormat
from assistant.commands.formats.RescueFormat import RescueFormat

BLOCK = '"""'
EXPRESSION = (f'send_email("alice@example.com", "Radering", {BLOCK}Hej,\nradera mina uppgifter.{BLOCK}'
              f'.replace("Marcus", "Marcus") if False else {BLOCK}Hej, radera mina uppgifter.{BLOCK}, "Gmail")')


@pytest.fixture
def parser():
    return CommandParser(["send_email", "remember", "save_procedure", "browse_eval", "report_issue", "latest_email",
                          "move_email", "turn_on"])


@pytest.mark.unit
class TestTextAfterQuote:
    def test_code_after_a_block_is_found(self, parser):
        assert parser.text_after_quote(EXPRESSION, blocks_only=True).startswith(".replace(")

    def test_a_block_followed_by_the_next_argument_or_the_end_is_clean(self, parser):
        clean = f'save_procedure("pool", {BLOCK}Step 1, open "the" valve (slowly){BLOCK} ,\n "v2")'
        assert parser.text_after_quote(clean) is None
        assert parser.text_after_quote(f'remember({BLOCK}a, b{BLOCK}\n)') is None

    def test_text_after_a_plain_string_counts_only_when_asked_for(self, parser):
        call = 'remember("Bob said "hi" to me")'
        assert parser.text_after_quote(call, blocks_only=True) is None
        assert parser.text_after_quote(call) is not None

    def test_a_plain_call_is_clean(self, parser):
        assert parser.text_after_quote('send_email("alice@example.com", "Hi, there", "A (b), c \\"d\\"", "Gmail")') is None
        assert parser.text_after_quote("turn_on(kitchen lights)") is None

    def test_a_command_whose_argument_is_verbatim_text_is_never_looked_into(self, parser):
        assert parser.text_after_quote('browse_eval("a" + "b", c)') is None
        assert parser.text_after_quote(f'report_issue({BLOCK}x{BLOCK} and more, text)') is None

    def test_the_arguments_are_read_as_before(self, parser):
        (_, params), _ = parser.extract_command_and_params('remember("Bob said "hi" to me")')
        assert params == ['Bob said hi" to me"']   # glued on after the closing quote, nothing lost


@pytest.mark.unit
class TestMailFolderNames:
    def test_a_gmail_folder_keeps_its_square_brackets(self, parser):
        (_, latest), _ = parser.extract_command_and_params('latest_email("[Gmail]/Sent Mail")')
        (_, move), _ = parser.extract_command_and_params('move_email("GM-AB", [Gmail]/Trash)')
        assert latest == ["[Gmail]/Sent Mail"] and move == ["GM-AB", "[Gmail]/Trash"]

    def test_a_room_suffix_still_becomes_parentheses(self, parser):
        (_, params), _ = parser.extract_command_and_params('turn_on("Ceiling Spots [Garage]")')
        assert params == ["Ceiling Spots (Garage)"]


@pytest.fixture
def processor():
    processor = CommandProcessor(test_suite_mode=True)
    processor.format = RescueFormat.wrap(KeepFormat(processor.possible_commands))
    processor.test_mode_dont_actually_execute_commands = False
    with patch.object(processor.commands, "run", return_value="") as run, \
            patch.object(processor, "_gate_sensitive_command", return_value=False) as gate:
        processor.run, processor.gate = run, gate
        yield processor
    processor.kill()


def feed(processor, text: str) -> str:
    shown = (processor.process(text, None) or "") + (processor.flush_buffer(None) or "")
    processor.command_queue.join()
    return shown


@pytest.mark.unit
class TestTheProcessorRefuses:
    def test_the_mail_written_as_an_expression_is_neither_approved_nor_sent(self, processor):
        feed(processor, EXPRESSION + "\n")
        processor.gate.assert_not_called()
        processor.run.assert_not_called()
        assert "send_email" in processor.command_reply and ".replace(" in processor.command_reply
        assert processor.command_reply_contains_error

    def test_the_calls_around_it_run_and_the_reply_goes_on(self, processor):
        shown = feed(processor, 'remember("first")\n' + EXPRESSION + '\nremember("second")\nDone.')
        assert [call.args for call in processor.run.call_args_list] == [("remember", ["first"]), ("remember", ["second"])]
        assert "Done." in shown and processor.commands_dispatched == 3

    def test_a_send_with_unescaped_quotes_in_a_plain_string_is_refused(self, processor):
        feed(processor, 'send_email("alice@example.com", "Hi", "He said "hello, there" to me")\n')
        processor.run.assert_not_called()
        assert "send_email" in processor.command_reply

    def test_a_command_without_the_approval_gate_keeps_its_tolerance_for_a_plain_string(self, processor):
        feed(processor, 'remember("Bob said "hi" to me")\n')
        processor.run.assert_called_once_with("remember", ['Bob said hi" to me"'])
        assert not processor.command_reply

    def test_code_after_a_block_is_refused_for_any_command(self, processor):
        feed(processor, f'remember({BLOCK}Bob{BLOCK}.upper())\n')
        processor.run.assert_not_called()
        assert "remember" in processor.command_reply

    def test_a_proper_send_goes_to_the_gate_and_runs(self, processor):
        feed(processor, f'send_email("alice@example.com", "Hi", {BLOCK}Hej, "Alice" (du){BLOCK}, "Gmail")\n')
        processor.gate.assert_called_once()
        processor.run.assert_called_once_with("send_email", ["alice@example.com", "Hi", 'Hej, "Alice" (du)', "Gmail"])
