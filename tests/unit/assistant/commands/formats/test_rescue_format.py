"""The rescue: a call the model writes as an `<invoke>` or a DSML block under another format still runs,
and the stored turn carries it in the active syntax."""
import pytest

from assistant.commands.CommandProcessor import CommandProcessor
from assistant.commands.formats.CommandFormats import CommandFormats
from assistant.commands.formats import DsmlFormat as dsml
from assistant.commands.formats.DsmlFormat import DsmlFormat
from assistant.commands.formats.InvokeFormat import InvokeFormat, OPEN, CLOSE, INVOKE_OPEN, INVOKE_CLOSE, render_call
from assistant.commands.formats.KeepFormat import KeepFormat
from assistant.commands.formats.RescueFormat import RescueFormat

NAMES = ["turn_on", "dim", "load_toolkit", "restart_server", "remember"]
TABLE = {"turn_on": ["light_name"], "dim": ["light_name", "dim_percentage"], "load_toolkit": ["name"],
         "restart_server": [], "remember": ["text"]}
VARIADIC = {"load_toolkit"}


def block(name: str, *pairs: tuple[str, str]) -> str:
    return render_call(name, list(pairs))


def bare(name: str, *pairs: tuple[str, str]) -> str:
    """The shape Opus wrote in production: the invoke with no `<function_calls>` wrapper."""
    lines = [f'{INVOKE_OPEN} name="{name}">'] + [f'<parameter name="{k}">{v}</parameter>' for k, v in pairs] + [INVOKE_CLOSE]
    return "\n".join(lines)


def dsml_block(name: str, *pairs: tuple[str, str]) -> str:
    return dsml.render_call(name, [(k, v, True) for k, v in pairs])


# The shapes V4.1 Flash wrote on the ds4 host on 2026-11-07 (the argument as a second attribute,
# a self-closed invoke, no closer; then a closer and a stray parameter closer in the body).
DSML_SLIP_UNCLOSED = (f'{dsml.OPEN}\n{dsml.INVOKE_OPEN} name="load_toolkit" name="developer" />\n\n'
                      f'{dsml.OPEN}\n{dsml.INVOKE_OPEN} name="load_toolkit" name="history" />')
DSML_SLIP_CLOSED = (f'{dsml.OPEN}\n{dsml.INVOKE_OPEN} name="load_toolkit" name="developer">\n\n'
                    f'{dsml.PARAMETER_CLOSE}\n{dsml.INVOKE_CLOSE}\n{dsml.CLOSE}')


@pytest.fixture
def fmt():
    return RescueFormat(KeepFormat(NAMES), [InvokeFormat(NAMES, table=TABLE, variadic=VARIADIC),
                                            DsmlFormat(NAMES, table=TABLE, variadic=VARIADIC)])


@pytest.fixture
def processor(fmt):
    processor = CommandProcessor(test_suite_mode=True)
    processor.format = fmt
    processor.calls = []
    processor.invoke_command = lambda name, params, dm: processor.calls.append((name, params))
    yield processor
    processor.kill()


def stream(processor, chunks):
    out = ""
    for chunk in chunks:
        if (text := processor.process(chunk, None)) is not None:
            out += text
    if (text := processor.flush_buffer(None)) is not None:
        out += text
    return out


class TestWrapping:
    def test_the_processor_reads_through_the_rescue_under_every_format_with_the_rescues_it_is_not(self):
        for primary, rescues in ((KeepFormat, {InvokeFormat, DsmlFormat}), (DsmlFormat, {InvokeFormat}),
                                 (InvokeFormat, {DsmlFormat})):
            CommandFormats.set_active(primary)
            try:
                processor = CommandProcessor(test_suite_mode=True)
                assert isinstance(processor.format, RescueFormat) and isinstance(processor.format.primary, primary)
                assert {type(r) for r in processor.format.rescues} == rescues
                processor.kill()
            finally:
                CommandFormats.reset()

    def test_the_wrapper_shares_the_command_list_so_a_name_added_later_is_known_to_both(self):
        names = ["turn_on"]
        fmt = RescueFormat.wrap(KeepFormat(names))
        names.append("dim")
        assert all(r.possible_commands is names for r in fmt.rescues) and fmt.primary.possible_commands is names


class TestStreaming:
    def test_the_production_reply_a_bare_invoke_alone_runs_and_shows_no_xml(self, processor):
        # Prod 2026-10-18 16:07: the whole reply was this block, Telegram got the raw XML.
        out = stream(processor, list(bare("dim", ("light_name", "Kitchen lamp"), ("dim_percentage", "40"))))
        assert processor.calls == [("dim", ["Kitchen lamp", "40"])]
        assert out.strip() == ""
        assert processor.format.rescued == 1

    def test_the_fused_production_reply_runs_the_call_and_shows_no_xml(self, processor):
        # Prod 2026-10-22 12:34: `<invoke_load_toolkit>` under Keep's syntax, nothing in the prompt to prime it.
        reply = "\n<invoke_load_toolkit>\n<parameter name=\"name\">developer</parameter>\n</invoke_load_toolkit>"
        out = stream(processor, list(reply))
        assert processor.calls == [("load_toolkit", ["developer"])]
        assert out.strip() == "" and processor.format.rescued == 1

    def test_a_wrapped_block_of_two_invokes_after_prose_runs_both_and_keeps_the_prose_before_them(self, processor):
        reply = "On it.\n\n" + block("turn_on", ("light_name", "Lamp")) + "\n" + block("load_toolkit", ("name", '["email", "web"]')) + "\nDone."
        out = stream(processor, [reply[:4], reply[4:30], reply[30:]])
        assert processor.calls == [("turn_on", ["Lamp"]), ("load_toolkit", ["email", "web"])]
        assert out.strip().startswith("On it.") and "Done." not in out and "<" not in out
        assert processor.reply_cut  # the prose after load_toolkit, a stop-and-wait command, is dropped

    def test_keep_calls_in_the_same_reply_still_run_in_order_with_the_rescued_one(self, processor):
        reply = 'turn_on("Hall") then ' + bare("dim", ("light_name", "Hall"), ("dim_percentage", "20")) + ' and restart_server()'
        out = stream(processor, list(reply))
        assert processor.calls == [("turn_on", ["Hall"]), ("dim", ["Hall", "20"]), ("restart_server", [])]
        assert "<" not in out and processor.format.rescued == 1

    def test_a_keep_reply_is_untouched_by_the_rescue(self, processor):
        out = stream(processor, ["Sure. ", "turn_on(", '"Lamp")', " Done — see <b>bold</b> text."])
        assert processor.calls == [("turn_on", ["Lamp"])]
        assert out.strip() == "Sure.  Done — see <b>bold</b> text." and processor.format.rescued == 0

    def test_a_closer_nothing_opened_is_dropped_not_shown(self, processor):
        out = stream(processor, [bare("restart_server") + "\n" + CLOSE + "\nRestarting."])
        assert processor.calls == [("restart_server", [])]
        assert out.strip() == "Restarting."

    def test_an_invoke_the_stream_ended_inside_is_reported_as_unterminated(self, processor):
        processor.format.primary.parser  # keep is the primary: its unterminated() knows no <invoke
        dm = type("DM", (), {"is_killed": False})()
        for chunk in [f'{INVOKE_OPEN} name="dim">\n<parameter name="light_name">Lamp</parameter>']:
            processor.process(chunk, dm)
        processor.flush_buffer(dm)
        assert processor.calls == []
        assert processor.take_unterminated_command_report().startswith("dim was not run")

    def test_the_dsml_production_reply_with_no_closer_runs_both_loads_once_the_stream_ends(self, processor):
        out = stream(processor, list(DSML_SLIP_UNCLOSED))
        assert processor.calls == [("load_toolkit", ["developer"]), ("load_toolkit", ["history"])]
        assert "DSML" not in out and processor.format.rescued == 2

    def test_the_dsml_production_reply_with_the_argument_in_the_tag_runs_the_call(self, processor):
        out = stream(processor, ["Loading. ", DSML_SLIP_CLOSED])
        assert processor.calls == [("load_toolkit", ["developer"])]
        assert out.strip() == "Loading." and processor.format.rescued == 1

    def test_a_well_formed_dsml_block_under_keep_runs_and_shows_nothing(self, processor):
        out = stream(processor, list("Sure.\n" + dsml_block("dim", ("dim_percentage", "40"), ("light_name", "Lamp"))))
        assert processor.calls == [("dim", ["Lamp", "40"])]
        assert out.strip() == "Sure." and processor.format.rescued == 1

    def test_a_dsml_block_the_stream_ended_inside_is_reported_as_unterminated(self, processor):
        dm = type("DM", (), {"is_killed": False})()
        processor.process(f'{dsml.OPEN}\n{dsml.INVOKE_OPEN} name="dim">\n{dsml.PARAMETER_OPEN} name="light_name" string="true">La', dm)
        processor.flush_buffer(dm)
        assert processor.calls == []
        assert processor.take_unterminated_command_report().startswith("dim was not run")


class TestRewritten:
    def test_the_bare_block_becomes_keep_text_that_keep_reads_to_the_same_call(self, fmt):
        raw = "Dimming.\n\n" + bare("dim", ("dim_percentage", "40"), ("light_name", "Kitchen lamp"))
        stored = fmt.rewritten(raw)
        assert "<" not in stored and stored.startswith("Dimming.\n\n")
        assert KeepFormat(NAMES).extract_all(stored) == [("dim", ["Kitchen lamp", "40"])]

    def test_the_fused_block_is_stored_in_keep_syntax_too(self, fmt):
        stored = fmt.rewritten('Loading.\n<invoke_load_toolkit>\n<parameter name="name">developer</parameter>\n</invoke_load_toolkit>')
        assert "<" not in stored and stored.startswith("Loading.\n")
        assert KeepFormat(NAMES).extract_all(stored) == [("load_toolkit", ["developer"])]

    def test_a_block_of_several_invokes_becomes_one_call_per_line(self, fmt):
        raw = block("turn_on", ("light_name", "Lamp")) + "\n" + block("load_toolkit", ("name", '["email", "web"]')) + "\nDone."
        stored = fmt.rewritten(raw)
        assert KeepFormat(NAMES).extract_all(stored) == [("turn_on", ["Lamp"]), ("load_toolkit", ["email", "web"])]
        assert stored.endswith("\nDone.") and OPEN not in stored

    def test_a_multi_line_value_survives_as_a_block_value(self, fmt):
        text = "Charlotte likes tea.\nFrances does not."
        stored = fmt.rewritten(bare("remember", ("text", text)))
        assert KeepFormat(NAMES).extract_all(stored) == [("remember", [text])]

    def test_a_stray_closer_is_dropped_from_the_stored_turn(self, fmt):
        stored = fmt.rewritten(bare("restart_server") + "\n" + CLOSE + "\nRestarting.")
        assert CLOSE not in stored and stored.endswith("\nRestarting.")
        assert KeepFormat(NAMES).extract_all(stored) == [("restart_server", [])]

    def test_a_keep_reply_comes_back_as_it_is(self, fmt):
        raw = 'Sure. turn_on("Lamp") — and <b>that</b> is all.'
        assert fmt.rewritten(raw) is raw

    def test_an_invoke_quoted_inside_a_keep_argument_is_not_a_block(self, fmt):
        raw = 'remember("""the model once wrote ' + bare("dim", ("light_name", "x")) + '""")'
        assert fmt.rewritten(raw) is raw
        assert fmt.extract_all(raw) == [("remember", ["the model once wrote " + bare("dim", ("light_name", "x"))])]

    def test_the_dsml_production_replies_are_stored_as_keep_calls(self, fmt):
        stored = fmt.rewritten("Checking.\n" + DSML_SLIP_UNCLOSED)
        assert "DSML" not in stored and stored.startswith("Checking.\n")
        assert KeepFormat(NAMES).extract_all(stored) == [("load_toolkit", ["developer"]), ("load_toolkit", ["history"])]
        stored = fmt.rewritten(DSML_SLIP_CLOSED + "\nFair cop.")
        assert "DSML" not in stored and stored.endswith("\nFair cop.")
        assert KeepFormat(NAMES).extract_all(stored) == [("load_toolkit", ["developer"])]

    def test_a_dsml_json_list_value_becomes_keep_variadic_values(self, fmt):
        stored = fmt.rewritten(dsml.render_call("load_toolkit", [("name", '["email", "web"]', False)]))
        assert KeepFormat(NAMES).extract_all(stored) == [("load_toolkit", ["email", "web"])]

    def test_an_invoke_block_and_a_dsml_block_in_one_reply_are_both_stored_as_keep_calls(self, fmt):
        raw = ("First. " + bare("turn_on", ("light_name", "Hall")) + "\nThen. "
               + dsml_block("dim", ("light_name", "Hall"), ("dim_percentage", "20")) + " restart_server() done")
        stored = fmt.rewritten(raw)
        assert "<invoke" not in stored and "DSML" not in stored
        assert stored.startswith("First. ") and "\nThen. " in stored and stored.endswith(" restart_server() done")
        assert KeepFormat(NAMES).extract_all(stored) == [("turn_on", ["Hall"]), ("dim", ["Hall", "20"]), ("restart_server", [])]

    def test_the_invoke_format_itself_stores_the_turn_as_written(self):
        raw = block("dim", ("light_name", "Lamp"), ("dim_percentage", "40"))
        assert InvokeFormat(NAMES, table=TABLE, variadic=VARIADIC).rewritten(raw) is raw


class TestWholeText:
    def test_extract_all_reads_both_syntaxes_in_text_order(self, fmt):
        text = 'turn_on("Hall") ' + bare("dim", ("light_name", "Hall"), ("dim_percentage", "20")) + ' restart_server()'
        assert fmt.extract_all(text) == [("turn_on", ["Hall"]), ("dim", ["Hall", "20"]), ("restart_server", [])]

    def test_extract_all_reads_all_three_syntaxes_in_text_order(self, fmt):
        text = 'turn_on("Hall") ' + dsml_block("dim", ("light_name", "Hall"), ("dim_percentage", "20")) + ' ' + bare("restart_server")
        assert fmt.extract_all(text) == [("turn_on", ["Hall"]), ("dim", ["Hall", "20"]), ("restart_server", [])]

    def test_cut_keeps_the_first_calls_across_both_syntaxes(self, fmt):
        text = 'turn_on("A") ' + bare("dim", ("light_name", "B"), ("dim_percentage", "20")) + ' turn_on("C") tail'
        cut = fmt.cut(text, 2)
        assert fmt.extract_all(cut) == [("turn_on", ["A"]), ("dim", ["B", "20"])]
        assert 'turn_on("C")' not in cut


class TestInvokeBlocks:
    def test_blocks_give_the_named_arguments_as_written_and_the_span_of_each_block(self):
        fmt = InvokeFormat(NAMES, table=TABLE, variadic=VARIADIC)
        text = "x " + bare("dim", ("dim_percentage", "40"), ("light_name", "Lamp")) + " y " + CLOSE
        blocks = fmt.blocks(text)
        assert [(s, text[s:e].startswith(INVOKE_OPEN) or text[s:e] == CLOSE) for s, e, _ in blocks] == [(2, True), (len(text) - len(CLOSE), True)]
        assert blocks[0][2] == [("dim", [("dim_percentage", "40"), ("light_name", "Lamp")])]
        assert blocks[1][2] == []
