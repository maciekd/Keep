"""KeepFormat.render_tool_call — a call a server parsed (Ollama's or ds4's `tool_calls`) written
back as Keep's positional call text, so an Ollama or ds4 fast tier under a Claude smart tier
hands the Keep parser a call it reads."""
import json

import pytest

from assistant.commands.formats.DsmlFormat import DsmlFormat
from assistant.commands.formats.KeepFormat import KeepFormat
from assistant.commands.formats.QwenFormat import QwenFormat

NAMES = ["dim", "turn_on", "homekit_cover", "load_toolkit", "browse_eval", "report_issue", "restart_server"]


@pytest.fixture
def fmt():
    return KeepFormat(NAMES)


def _read(fmt, text):
    return fmt.extract_all(text)


class TestRenderToolCall:
    def test_named_arguments_come_out_in_the_declared_order_and_read_back_positionally(self, fmt):
        # The model named dim_percentage first; the texts declare dim(light_name, dim_percentage).
        text = KeepFormat.render_tool_call("dim", {"dim_percentage": 40, "light_name": "Kitchen lamp"})
        assert _read(fmt, text) == [("dim", ["Kitchen lamp", "40"])]

    def test_the_servers_json_text_arguments_are_decoded(self, fmt):
        text = KeepFormat.render_tool_call("turn_on", json.dumps({"light_name": "Hall lamp"}))
        assert _read(fmt, text) == [("turn_on", ["Hall lamp"])]

    def test_a_quote_and_a_backslash_in_a_value_survive_the_round_trip(self, fmt):
        value = 'the "reading" lamp \\ hall'
        text = KeepFormat.render_tool_call("turn_on", {"light_name": value})
        assert _read(fmt, text) == [("turn_on", [value])]

    def test_a_multi_line_value_is_a_block_and_reads_back_whole(self, fmt):
        value = "line one\nline two, with a comma (and parens)"
        text = KeepFormat.render_tool_call("turn_on", {"light_name": value})
        assert text.startswith('turn_on("""') and _read(fmt, text) == [("turn_on", [value])]

    def test_an_optional_parameter_left_out_in_the_middle_keeps_the_positions(self, fmt):
        # homekit_cover(ref, position, dry_run): dry_run given, position not → position is "".
        text = KeepFormat.render_tool_call("homekit_cover", {"ref": "A1B2", "dry_run": "dry_run"})
        assert _read(fmt, text) == [("homekit_cover", ["A1B2", "", "dry_run"])]

    def test_a_variadic_list_is_spread_into_values(self, fmt):
        text = KeepFormat.render_tool_call("load_toolkit", {"name": ["email", "web"]})
        assert _read(fmt, text) == [("load_toolkit", ["email", "web"])]

    def test_an_opaque_commands_payload_is_passed_verbatim(self, fmt):
        js = 'document.querySelector("a").click(); return 1 + 1'
        text = KeepFormat.render_tool_call("browse_eval", {"js": js})
        assert text == f"browse_eval({js})"
        assert _read(fmt, text) == [("browse_eval", [js])]

    def test_an_undeclared_argument_follows_the_declared_ones(self, fmt):
        text = KeepFormat.render_tool_call("turn_on", {"brightness": 5, "light_name": "Lamp"})
        assert _read(fmt, text) == [("turn_on", ["Lamp", "5"])]

    def test_a_parameterless_call(self, fmt):
        assert _read(fmt, KeepFormat.render_tool_call("restart_server", {})) == [("restart_server", [])]


class TestEveryFormatRendersAParsedCallItReads:
    @pytest.mark.parametrize("cls", [KeepFormat, QwenFormat, DsmlFormat])
    def test_the_rendering_of_a_servers_call_is_read_back_by_the_same_format(self, cls):
        parser = cls(NAMES)
        text = cls.render_tool_call("dim", '{"light_name": "Reading lamp", "dim_percentage": 1}')
        assert parser.extract_all(text) == [("dim", ["Reading lamp", "1"])]

    @pytest.mark.parametrize("cls", [
        pytest.param(KeepFormat, marks=pytest.mark.xfail(
            strict=True, reason="KeepFormat.render_tool_call spreads a variadic value only when it is a list: a "
                                "server's string value ('email, web', or the JSON text of a list) renders as one "
                                "argument, so load_toolkit answers that no such toolkit exists")),
        QwenFormat, DsmlFormat])
    @pytest.mark.parametrize("value", ["email, web", '["email", "web"]'])
    def test_a_variadic_parameter_given_as_text_is_split_into_its_values(self, cls, value):
        # A DSML `string="true"` parameter, or a server that did not decode the list, hands the
        # format the names as one string; every format must read the same two toolkits.
        parser = cls(NAMES)
        assert parser.extract_all(cls.render_tool_call("load_toolkit", {"name": value})) == [("load_toolkit", ["email", "web"])]

    @pytest.mark.parametrize("cls", [KeepFormat, QwenFormat, DsmlFormat])
    def test_arguments_that_are_not_a_json_object_become_one_value(self, cls):
        parser = cls(NAMES)
        assert parser.extract_all(cls.render_tool_call("turn_on", "not json"))[0][1] == ["not json"]
