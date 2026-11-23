"""The repeat and degenerate-value guards of the command processor (issue #123): a reply that
writes the same call over and over, or rescued `<invoke>` blocks holding punctuation alone,
is cut where the runaway shows instead of at the 15-command cap."""
import pytest

from assistant.commands.CommandProcessor import CommandProcessor
from assistant.commands.formats.KeepFormat import KeepFormat
from assistant.commands.formats.RescueFormat import RescueFormat


def invoke(name: str, **values: str) -> str:
    lines = [f'<invoke name="{name}">'] + [f'<parameter name="{k}">{v}</parameter>' for k, v in values.items()] + ["</invoke>"]
    return "\n".join(lines) + "\n\n"


@pytest.fixture
def processor():
    processor = CommandProcessor(test_suite_mode=True)
    processor.format = RescueFormat.wrap(KeepFormat(processor.possible_commands))
    processor.calls = []
    dispatch = processor.invoke_command
    processor.invoke_command = lambda name, params, dm: (processor.calls.append((name, params)), dispatch(name, params, dm))
    yield processor
    processor.kill()


def feed(processor, *chunks) -> str:
    shown = [processor.process(chunk, None) for chunk in chunks]
    shown.append(processor.flush_buffer(None))
    processor.command_queue.join()
    return "".join(s for s in shown if s)


@pytest.mark.unit
class TestRepeatGuard:
    def test_a_stop_and_wait_call_written_twice_is_cut_at_the_second(self, processor):
        """Prod 2026-11-04: `load_watch` 15 times in a row to an "ok"."""
        raw = "load_watch(radon)\nload_watch(radon)\nload_watch(radon)\nDone.\n"
        shown = feed(processor, raw)
        assert shown == "" and processor.reply_cut
        assert processor.calls == [("load_watch", ["radon"])]
        assert processor.kept_of(raw).strip() == "load_watch(radon)"
        assert "load_watch" in processor.command_reply and "2 times" in processor.command_reply
        assert not processor.command_reply_contains_error   # nothing failed: a note, not an error report

    def test_the_15_empty_server_logs_of_10_26_stop_at_the_first_repeat(self, processor):
        raw = invoke("server_logs", from_time="12:00", pattern="x") + invoke("server_logs") * 14
        feed(processor, raw)
        assert processor.reply_cut
        assert [name for name, _ in processor.calls] == ["server_logs", "server_logs"]   # with args, then one bare
        assert len(processor.format.extract_all(processor.kept_of(raw))) == 2

    def test_an_accumulating_call_may_repeat_up_to_the_allowance(self, processor):
        allowance = CommandProcessor.MAX_IDENTICAL_CALLS
        shown = feed(processor, "volume_up(kitchen)\n" * allowance + "Louder now.")
        assert not processor.reply_cut and len(processor.calls) == allowance
        assert "Louder now." in shown

    def test_one_past_the_allowance_cuts_the_reply(self, processor):
        allowance = CommandProcessor.MAX_IDENTICAL_CALLS
        raw = "volume_up(kitchen)\n" * (allowance + 5) + "Louder now."
        shown = feed(processor, raw)
        assert shown == "" and processor.reply_cut
        assert len(processor.calls) == allowance
        assert processor.kept_of(raw).count("volume_up") == allowance
        assert f"{allowance + 1} times" in processor.command_reply

    def test_the_same_command_with_other_arguments_is_no_repeat(self, processor):
        feed(processor, "\n".join(f"turn_on(lamp {i})" for i in range(8)) + "\n")
        assert not processor.reply_cut and len(processor.calls) == 8

    def test_a_call_between_two_copies_breaks_the_run(self, processor):
        feed(processor, "volume_up(kitchen)\nvolume_up(hall)\n" * 4)
        assert not processor.reply_cut and len(processor.calls) == 8

    def test_the_run_is_the_replys_not_the_processors(self, processor):
        feed(processor, "load_watch(radon)\n")
        processor.clear_command_reply()
        processor.start_new_reply()
        feed(processor, "load_watch(radon)\n")
        assert not processor.reply_cut and len(processor.calls) == 2


@pytest.mark.unit
class TestDegenerateValueGuard:
    JUNK = '"\n}\n]'

    def test_rescued_browse_eval_blocks_of_punctuation_cut_the_reply_after_the_real_call(self, processor):
        """Prod 2026-10-29 16:34: one real `browse_eval`, then 313 blocks of `"`, `}` and `]`."""
        raw = invoke("browse_eval", js="return document.title") + invoke("browse_eval", js=self.JUNK) * 20
        feed(processor, raw)
        assert processor.reply_cut
        assert processor.calls == [("browse_eval", ["return document.title"])]
        assert len(processor.format.extract_all(processor.kept_of(raw))) == 1
        assert "browse_eval" in processor.command_reply and not processor.command_reply_contains_error

    def test_a_rescued_call_with_real_values_runs(self, processor):
        feed(processor, invoke("turn_on", light_name="lamp"))
        assert not processor.reply_cut and processor.calls == [("turn_on", ["lamp"])]

    def test_a_rescued_call_with_no_values_is_left_alone(self, processor):
        feed(processor, invoke("load_watch"))
        assert not processor.reply_cut and [name for name, _ in processor.calls] == ["load_watch"]

    def test_a_call_in_the_taught_syntax_is_not_checked(self, processor):
        feed(processor, 'remember("}")\n')
        assert not processor.reply_cut and processor.calls == [("remember", ["}"])]

    def test_the_rescue_counter_starts_over_with_the_reply(self, processor):
        feed(processor, invoke("turn_on", light_name="lamp"))
        assert processor.format.rescued == 1
        processor.start_new_reply()
        assert processor.format.rescued == 0


@pytest.mark.unit
class TestInventedResultGuard:
    """Prod 2026-11-22: after each real `<invoke>` the model wrote a block of the same name
    holding the result it expected in a `result` parameter, and the value ran as the argument."""

    def test_a_block_holding_a_result_parameter_does_not_run(self, processor):
        raw = (invoke("browse_open", url="https://shop.example/cart")
               + invoke("browse_open", result="URL: https://shop.example/cart\n[12] input card number")
               + invoke("browse_act", spec="12=4000 0000 0000 0000; 13"))
        shown = feed(processor, raw)
        assert shown == "" and processor.reply_cut
        assert processor.calls == [("browse_open", ["https://shop.example/cart"])]
        assert processor.format.extract_all(processor.kept_of(raw)) == processor.calls
        assert "result" in processor.command_reply and "browse_open" in processor.command_reply
        assert not processor.command_reply_contains_error

    def test_an_invented_result_before_any_call_still_gets_its_note(self, processor):
        feed(processor, invoke("load_watch", result="radon: 80 Bq/m3"))
        assert processor.reply_cut and processor.calls == []
        assert "result" in processor.command_reply

    def test_a_result_beside_the_declared_parameter_is_cut_too(self, processor):
        feed(processor, invoke("browse_open", url="https://shop.example/", result="URL: https://shop.example/"))
        assert processor.reply_cut and processor.calls == []

    def test_an_honest_call_with_a_misnamed_parameter_still_runs(self, processor):
        """`query` is not what `search_email` declares; the rescue has always taken it in the model's order."""
        feed(processor, invoke("search_email", query="invoice"))
        assert not processor.reply_cut
        assert processor.calls == [("search_email", ["invoice"])]

    def test_the_guard_is_the_replys_not_the_processors(self, processor):
        feed(processor, invoke("load_watch", result="radon: 80 Bq/m3"))
        processor.clear_command_reply()
        processor.start_new_reply()
        feed(processor, invoke("load_watch", item="radon"))
        assert not processor.reply_cut and len(processor.calls) == 1


@pytest.mark.unit
class TestPageGuard:
    """A call that acts on the page while a browse result is pending rests on a snapshot the
    model has not read (prod 2026-11-22: a card number typed into an unknown element)."""

    @pytest.mark.parametrize("acting", ['browse_act("12=Marcus; 13")', 'browse_key("3,Enter")', "browse_click_at(120, 340)"])
    def test_an_action_on_a_page_just_opened_is_cut(self, processor, acting):
        raw = f'browse_open("https://shop.example/checkout")\n{acting}\n'
        feed(processor, raw)
        assert processor.reply_cut
        assert [name for name, _ in processor.calls] == ["browse_open"]
        assert processor.kept_of(raw).strip() == 'browse_open("https://shop.example/checkout")'
        assert "browse_open" in processor.command_reply and acting.split("(")[0] in processor.command_reply

    def test_a_second_action_after_an_action_is_cut(self, processor):
        feed(processor, 'browse_act("3=2")\nbrowse_act("7")\n')
        assert processor.reply_cut and processor.calls == [("browse_act", ["3=2"])]

    def test_reads_after_a_browse_call_still_chain(self, processor):
        feed(processor, 'browse_act("7")\nbrowse_network("cart")\nbrowse_screenshot()\n')
        assert not processor.reply_cut
        assert [name for name, _ in processor.calls] == ["browse_act", "browse_network", "browse_screenshot"]

    def test_the_page_is_named_as_what_was_pending_when_a_read_stands_between(self, processor):
        feed(processor, 'browse_open("https://shop.example/")\nload_toolkit("email")\nbrowse_act("3")\n')
        assert processor.reply_cut and len(processor.calls) == 2
        assert "browse_open" in processor.command_reply

    def test_an_action_after_a_fetch_that_is_no_browse_call_runs(self, processor):
        """The snapshot it acts on is an earlier round's."""
        feed(processor, 'load_toolkit("email")\nbrowse_act("3")\n')
        assert not processor.reply_cut and len(processor.calls) == 2
