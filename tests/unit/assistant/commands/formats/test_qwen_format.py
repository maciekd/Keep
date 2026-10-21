"""The Qwen `<tool_call>` format: rendering the texts, reading the model's calls, streaming."""
import pytest

from assistant.commands.CommandProcessor import CommandProcessor
from assistant.commands.formats import Signatures
from assistant.commands.formats.QwenFormat import QwenFormat, OPEN, CLOSE, render_call, argument_text
from assistant.commands.formats.Signatures import Declaration, Parameter, Value, calls, read_declaration, read_values

NAMES = ["turn_on", "dim", "load_toolkit", "homekit_cover", "restart_server", "browse_eval"]
TABLE = {"turn_on": ["light_name"], "dim": ["light_name", "dim_percentage"], "load_toolkit": ["name"],
         "homekit_cover": ["ref", "position", "dry_run"], "restart_server": [], "browse_eval": ["js"]}
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

    def test_a_block_without_an_invoke_is_no_call(self, fmt):
        assert fmt.extract(f"{OPEN}\nnothing\n{CLOSE}") == (None, None)
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

    def test_split_at_every_character_extracts_both_calls_and_releases_the_text_only(self, processor):
        out = self._stream(processor, list(self.REPLY))
        assert processor.calls == [("dim", ["Kitchen lamp", "40"]), ("load_toolkit", ["email", "web"])]
        assert OPEN not in out and "<function" not in out
        assert out.startswith("Sure thing.") and out.rstrip().endswith("Done.")

    def test_whole_reply_in_one_chunk_gives_the_same_result(self, processor):
        out = self._stream(processor, [self.REPLY])
        assert processor.calls == [("dim", ["Kitchen lamp", "40"]), ("load_toolkit", ["email", "web"])]
        assert OPEN not in out and "Done." in out

    @pytest.mark.xfail(strict=True, reason="a complete <tool_call> block without a <function=…> extracts to no "
                                           "call, and process_one_command then truncates the reply: the prose and "
                                           "the calls after the empty block are lost")
    def test_an_empty_block_is_dropped_but_what_follows_it_is_kept(self, processor):
        reply = (f"{OPEN}\n{CLOSE}\nSure, done.\n"
                 + block("turn_on", ("light_name", "Kitchen lamp")) + "\nAnd that.")
        out = self._stream(processor, [reply])
        assert processor.calls == [("turn_on", ["Kitchen lamp"])]
        assert "Sure, done." in out and "And that." in out and OPEN not in out

    def test_a_stream_cut_inside_the_block_reports_the_unterminated_command(self, processor):
        from unittest.mock import Mock
        cut = self.REPLY[:self.REPLY.index("Kitchen")]
        out = self._stream(processor, list(cut), Mock(is_killed=False))  # the report needs a dialog
        assert processor.calls == []
        assert OPEN not in (out or "")
        assert "dim" in (processor.take_unterminated_command_report() or "")
