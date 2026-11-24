"""The Qwen `<tool_call>` format: rendering the texts, reading the model's calls, streaming."""
from unittest.mock import Mock

import pytest

from assistant.commands.CommandProcessor import CommandProcessor
from assistant.commands.formats import Signatures
from assistant.commands.formats.QwenFormat import QwenFormat, OPEN, CLOSE, render_call, argument_text
from assistant.commands.formats.Signatures import Declaration, Parameter, Value, calls, read_declaration, read_values

NAMES = ["turn_on", "dim", "load_toolkit", "homekit_cover", "restart_server", "browse_eval", "improve_integration",
         "report_issue"]
TABLE = {"turn_on": ["light_name"], "dim": ["light_name", "dim_percentage"], "load_toolkit": ["name"],
         "homekit_cover": ["ref", "position", "dry_run"], "restart_server": [], "browse_eval": ["js"],
         "improve_integration": ["integration", "request"], "report_issue": ["report"]}
VARIADIC = {"load_toolkit"}


@pytest.fixture
def fmt():
    return QwenFormat(NAMES, table=TABLE, variadic=VARIADIC)


def declaration(text: str) -> Declaration:
    return read_declaration(calls(text)[0])


def block(name: str, *pairs: tuple[str, str]) -> str:
    return render_call(name, list(pairs))


# --- rendering -----------------------------------------------------------------------------

class TestRendering:
    def test_declaration_becomes_a_function_schema_with_required_and_typed_parameters(self, fmt):
        import json
        rendered = fmt.render_declaration(declaration("`dim(light_name: str, dim_percentage: int)`"))
        schema = json.loads(rendered.strip("`"))
        assert schema["type"] == "function"
        parameters = schema["function"]["parameters"]
        assert schema["function"]["name"] == "dim"
        assert parameters["required"] == ["light_name", "dim_percentage"]
        assert parameters["properties"]["dim_percentage"] == {"type": "integer"}

    def test_a_default_makes_the_parameter_optional_and_an_unknown_type_is_a_described_string(self, fmt):
        import json
        rendered = fmt.render_declaration(declaration("`homekit_cover(ref: str, position: open|close|0-100, dry_run: str = \"\")`"))
        parameters = json.loads(rendered.strip("`"))["function"]["parameters"]
        assert parameters["required"] == ["ref", "position"]
        assert parameters["properties"]["position"] == {"type": "string", "description": "open|close|0-100"}
        assert parameters["properties"]["dry_run"]["default"] == ""

    def test_a_variadic_declaration_is_one_array_parameter(self, fmt):
        import json
        parameters = json.loads(fmt.render_declaration(declaration("`load_toolkit(name: str, ...)`")).strip("`"))["function"]["parameters"]
        assert parameters["properties"]["name"] == {"type": "array", "items": {"type": "string"}}

    def test_a_parameterless_declaration_has_no_required_list(self, fmt):
        import json
        parameters = json.loads(fmt.render_declaration(Declaration("restart_server", ())).strip("`"))["function"]["parameters"]
        assert "required" not in parameters and parameters["properties"] == {}

    def test_example_pairs_values_with_the_declared_names_in_order(self, fmt):
        call = calls('`dim("Kitchen lamp", 40)`')[0]
        rendered = fmt.render_example("dim", read_values(call), declaration("`dim(light_name: str, dim_percentage: int)`"))
        assert rendered == block("dim", ("light_name", "Kitchen lamp"), ("dim_percentage", "40"))

    def test_example_resolves_an_escaped_quote_and_renders_a_block_value_whole(self, fmt):
        decl = declaration("`turn_on(light_name: str)`")
        assert fmt.render_example("turn_on", [Value('Say \\"hi\\"', "string")], decl) == block("turn_on", ("light_name", 'Say "hi"'))
        rendered = fmt.render_example("turn_on", [Value("line one\nline two", "block")], decl)
        assert "<parameter=light_name>\nline one\nline two\n</parameter>" in rendered

    def test_variadic_example_puts_every_value_in_one_json_list(self, fmt):
        call = calls('`load_toolkit("email", "web")`')[0]
        rendered = fmt.render_example("load_toolkit", read_values(call), declaration("`load_toolkit(name: str, ...)`"))
        assert rendered == block("load_toolkit", ("name", '["email", "web"]'))

    def test_instructions_teach_the_block_and_name_no_real_command_call(self, fmt):
        """The rewriter runs over the filled-in instructions: a backticked call of a real command
        in them would be rendered, so the text must carry none."""
        text = fmt.instructions()
        assert OPEN in text and CLOSE in text and "<function=" in text and "<parameter=" in text
        assert not [c for c in calls(text) if c.name in Signatures.table()]

    def test_real_texts_render_without_a_keep_declaration_left(self):
        """Every typed declaration of the real texts is rendered as a schema, none stays
        `name(param: type)` — the identity test's counterpart for a format that changes the texts."""
        from assistant.commands.formats import Rewriter
        from assistant.commands.Commands import Commands
        fmt = QwenFormat(Commands.command_names())
        table = Signatures.declarations()
        for label, text in Signatures.texts().items():
            rendered = Rewriter.rewrite(text, fmt, table)
            left = [c for c in calls(rendered) if read_declaration(c) is not None and c.name in table]
            assert not left, f"{label}: {[c.name for c in left]}"


# --- parsing -------------------------------------------------------------------------------

class TestParsing:
    def test_round_trip_declaration_to_model_call_to_positional_params(self, fmt):
        decl = declaration("`dim(light_name: str, dim_percentage: int)`")
        rendered = fmt.render_example("dim", [Value("Kitchen lamp", "string"), Value("40", "bare")], decl)
        (name, params, display), rest = fmt.extract(rendered)
        assert (name, params) == ("dim", ["Kitchen lamp", "40"])
        assert display == "Kitchen lamp, 40" and rest == ""

    def test_parameters_in_the_models_order_are_put_in_the_declared_order(self, fmt):
        text = block("dim", ("dim_percentage", "1"), ("light_name", "Hall lamp"))
        assert fmt.extract(text)[0][1] == ["Hall lamp", "1"]

    def test_an_optional_parameter_left_out_in_the_middle_is_an_empty_string_and_trailing_ones_are_dropped(self, fmt):
        assert fmt.extract(block("homekit_cover", ("ref", "A1B2"), ("dry_run", "dry_run")))[0][1] == ["A1B2", "", "dry_run"]
        assert fmt.extract(block("homekit_cover", ("ref", "A1B2"), ("position", "40")))[0][1] == ["A1B2", "40"]

    def test_an_undeclared_command_or_parameter_keeps_the_models_order(self, fmt):
        assert fmt.extract(block("mystery", ("b", "2"), ("a", "1")))[0][1] == ["2", "1"]
        assert fmt.extract(block("dim", ("brightness", "5"), ("light_name", "Lamp")))[0][1] == ["5", "Lamp"]

    @pytest.mark.parametrize("value", ['["email", "web"]', "email, web", "email,web"])
    def test_a_variadic_parameter_is_split_into_positional_values(self, fmt, value):
        assert fmt.extract(block("load_toolkit", ("name", value)))[0][1] == ["email", "web"]

    def test_a_free_text_commands_values_join_into_the_one_payload_its_handler_splits(self, fmt):
        # cmd_improve_integration takes "name, request" as one string (the Keep parser hands a
        # free-text command's payload over verbatim) and Commands.run passes a single-parameter
        # command params[0] alone: two values would lose the request.
        text = block("improve_integration", ("request", "add a charging schedule"), ("integration", "tesla"))
        assert fmt.extract(text)[0][1] == ["tesla, add a charging schedule"]
        assert fmt.extract_all(text) == [("improve_integration", ["tesla, add a charging schedule"])]
        assert fmt.extract(block("report_issue", ("report", "the porch, lit at noon")))[0][1] == ["the porch, lit at noon"]

    def test_one_leading_and_one_trailing_newline_are_stripped_and_inner_newlines_kept(self, fmt):
        text = f"{OPEN}\n<function=browse_eval>\n<parameter=js>\n\nreturn 1;\n\n</parameter>\n</function>\n{CLOSE}"
        assert fmt.extract(text)[0][1] == ["\nreturn 1;\n"]

    def test_several_invokes_in_one_block_come_back_in_order_through_rest(self, fmt):
        text = (f"{OPEN}\n<function=turn_on>\n<parameter=light_name>\nA\n</parameter>\n</function>\n"
                f"<function=turn_on>\n<parameter=light_name>\nB\n</parameter>\n</function>\n{CLOSE} tail")
        first, rest = fmt.extract(text)
        assert first[:2] == ("turn_on", ["A"])
        second, rest2 = fmt.extract(rest)
        assert second[:2] == ("turn_on", ["B"]) and rest2 == " tail"
        assert fmt.extract_all(text) == [("turn_on", ["A"]), ("turn_on", ["B"])]

    def test_text_before_and_after_the_block_is_located_and_kept(self, fmt):
        text = "Sure. " + block("turn_on", ("light_name", "Lamp")) + " Done."
        assert not fmt.is_possibly_command(text)
        assert fmt.find_start(text) == len("Sure. ")
        assert fmt.extract(text)[1] == " Done."

    def test_an_incomplete_block_is_possibly_a_command_but_not_complete(self, fmt):
        partial = f"{OPEN}\n<function=turn_on>\n<parameter=light_name>\nKit"
        assert fmt.is_possibly_command(partial) and not fmt.complete(partial, streaming=True)
        assert fmt.extract(partial) == (None, None)
        assert fmt.unterminated(partial) == "turn_on"

    @pytest.mark.parametrize("buffer, possible", [("", True), ("  ", True), ("<", True), ("<tool_", True),
                                                  ("<tool_call", True), ("<b>", False), ("Hello", False)])
    def test_a_prefix_of_the_opener_may_still_become_a_call(self, fmt, buffer, possible):
        assert fmt.is_possibly_command(buffer) is possible

    def test_partial_tail_is_the_opener_prefix_at_the_end_of_text(self, fmt):
        assert fmt.partial_tail("Turning it on <tool_") == len("Turning it on ")
        assert fmt.partial_tail("<tool_") == -1
        assert fmt.partial_tail("Turning it on <b") == -1
        assert fmt.partial_tail("Turning it on") == -1

    def test_a_block_without_an_invoke_is_no_call_and_hands_back_what_follows_it(self, fmt):
        assert fmt.extract(f"{OPEN}\nnothing\n{CLOSE}\nDone.") == (None, "\nDone.")
        assert fmt.call_end(f"{OPEN}\nnothing\n{CLOSE}\nDone.") == len(f"{OPEN}\nnothing\n{CLOSE}")
        assert fmt.unterminated(f"{OPEN}\nnoth") is None

    def test_a_bare_name_or_plain_text_is_never_unterminated(self, fmt):
        assert fmt.unterminated("turn_on") is None
        assert fmt.unterminated("just text") is None

    def test_argument_text_spells_values_the_way_the_model_writes_them(self):
        assert argument_text("Lamp") == "Lamp"
        assert argument_text(1) == "1" and argument_text(True) == "true"
        assert argument_text(["a", "b"]) == '["a", "b"]'


# --- streaming through the real processor --------------------------------------------------

class TestStreaming:
    REPLY = ("Sure thing.\n" + block("dim", ("dim_percentage", "40"), ("light_name", "Kitchen lamp"))
             + "\n" + block("load_toolkit", ("name", '["email", "web"]')) + "\nDone.")

    @pytest.fixture
    def processor(self):
        processor = CommandProcessor(test_suite_mode=True)
        processor.format = QwenFormat(processor.possible_commands, table=TABLE, variadic=VARIADIC)
        processor.calls = []
        processor.invoke_command = lambda name, params, dm: processor.calls.append((name, params))
        yield processor
        processor.kill()

    @staticmethod
    def _stream(processor, chunks, dialog_manager=None):
        out = ""
        for chunk in chunks:
            if (text := processor.process(chunk, dialog_manager)) is not None:
                out += text
        if (text := processor.flush_buffer(dialog_manager)) is not None:
            out += text
        return out

    def test_split_at_every_character_extracts_both_calls_and_cuts_the_text_after_the_load(self, processor):
        # load_toolkit is stop-and-wait: the prose after it is dropped (the after-the-fetch guard).
        out = self._stream(processor, list(self.REPLY))
        assert processor.calls == [("dim", ["Kitchen lamp", "40"]), ("load_toolkit", ["email", "web"])]
        assert OPEN not in out and "<function" not in out
        assert out.startswith("Sure thing.") and "Done." not in out and processor.reply_cut

    def test_whole_reply_in_one_chunk_gives_the_same_result(self, processor):
        out = self._stream(processor, [self.REPLY])
        assert processor.calls == [("dim", ["Kitchen lamp", "40"]), ("load_toolkit", ["email", "web"])]
        assert OPEN not in out and "Done." not in out and processor.reply_cut

    def test_an_empty_block_is_dropped_but_what_follows_it_is_kept(self, processor):
        reply = (f"{OPEN}\n{CLOSE}\nSure, done.\n"
                 + block("turn_on", ("light_name", "Kitchen lamp")) + "\nAnd that.")
        out = self._stream(processor, [reply])
        assert processor.calls == [("turn_on", ["Kitchen lamp"])]
        assert "Sure, done." in out and "And that." in out and OPEN not in out

    def test_a_stream_cut_inside_the_block_reports_the_unterminated_command(self, processor):
        cut = self.REPLY[:self.REPLY.index("Kitchen")]
        out = self._stream(processor, list(cut), Mock(is_killed=False))  # the report needs a dialog
        assert processor.calls == []
        assert OPEN not in (out or "")
        assert "dim" in (processor.take_unterminated_command_report() or "")

    # A call opened with the wrong tag: closed as a function, every parameter in place.
    MISNAMED = (f"{OPEN}\n<parameter=dim>\n<parameter=light_name>\nKitchen lamp\n</parameter>\n"
                f"<parameter=dim_percentage>\n40\n</parameter>\n</function>\n{CLOSE}")

    @pytest.mark.parametrize("pieces", [list, lambda reply: [reply]])
    def test_a_block_that_names_no_function_is_reported_and_never_run(self, processor, pieces):
        out = self._stream(processor, pieces(self.MISNAMED), Mock(is_killed=False))
        assert processor.calls == [] and not out.strip()
        report = processor.take_unterminated_command_report()
        assert "Kitchen lamp" in report and "Nothing was run" in report

    def test_the_report_quotes_the_block_alone_and_the_prose_around_it_is_shown(self, processor):
        out = self._stream(processor, [f"On it.\n{self.MISNAMED}\nAnything else?"], Mock(is_killed=False))
        report = processor.take_unterminated_command_report()
        assert "On it." in out and "Anything else?" in out and OPEN not in out
        assert "Kitchen lamp" in report and "Anything else?" not in report

    @pytest.mark.parametrize("reply", [
        f"{OPEN}\n{CLOSE}\nDone.",
        f"{OPEN}\n<function>\n</function>\n{CLOSE}",
        MISNAMED + "\n" + block("dim", ("light_name", "Kitchen lamp"), ("dim_percentage", "40")),
        f"A call looks like this:\n```\n{MISNAMED}\n```",
    ], ids=["empty", "empty function", "written again, right", "shown as code"])
    def test_no_report_where_a_round_to_tell_the_model_would_be_for_nothing(self, processor, reply):
        processor.invoke_command = lambda name, params, dm: setattr(processor, "commands_dispatched", processor.commands_dispatched + 1)
        self._stream(processor, [reply], Mock(is_killed=False))
        assert processor.take_unterminated_command_report() is None

    def test_a_dropped_block_is_not_reported_with_the_next_reply(self, processor):
        self._stream(processor, [self.MISNAMED], Mock(is_killed=False))
        processor.start_new_reply()
        self._stream(processor, ["All good."], Mock(is_killed=False))
        assert processor.take_unterminated_command_report() is None

    def test_an_autonomous_run_has_no_dialog_to_report_to(self, processor):
        self._stream(processor, [self.MISNAMED])
        assert processor.take_unterminated_command_report() is None


class TestCut:
    def test_a_block_of_several_functions_is_cut_inside_and_closed_again(self, fmt):
        functions = "\n".join(f"<function=turn_on>\n<parameter=light_name>lamp {i}</parameter>\n</function>" for i in range(3))
        kept = fmt.cut(f"{OPEN}\n{functions}\n{CLOSE}\nDone.", 2)
        assert [p for _, p in fmt.extract_all(kept)] == [["lamp 0"], ["lamp 1"]]
        assert kept.endswith(CLOSE) and "Done." not in kept

    def test_one_block_per_call_is_cut_between_blocks(self, fmt):
        blocks = "\n".join(render_call("turn_on", [("light_name", f"lamp {i}")]) for i in range(3))
        kept = fmt.cut(blocks, 2)
        assert [p for _, p in fmt.extract_all(kept)] == [["lamp 0"], ["lamp 1"]]
        assert kept.endswith(CLOSE)
