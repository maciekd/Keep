import pytest

from common import BrowseFillMask


@pytest.mark.unit
class TestSpecLabel:
    def test_a_click_is_its_index_and_a_fill_its_length(self):
        label = BrowseFillMask.spec_label("55=4111 1111 1111 1111; 56=12/29; 57; 58=737")
        assert "4111" not in label and "12/29" not in label and "737" not in label
        assert [part.split("=")[0] for part in label.split("; ")] == ["55", "56", "57", "58"]
        assert "19 chars" in label and "5 chars" in label and "3 chars" in label

    def test_a_semicolon_inside_a_value_does_not_let_the_rest_of_it_out(self):
        label = BrowseFillMask.spec_label("3=first; second half; 4")
        assert "second" not in label and label.endswith("; 4")

    def test_a_label_masked_once_stays_as_it_is(self):
        once = BrowseFillMask.spec_label("3=secret")
        assert BrowseFillMask.spec_label(once) == once


@pytest.mark.unit
class TestMaskFills:
    @pytest.mark.parametrize("line", [
        'Running command: browse_act(56=12/29; 57=737)',
        'browse_act("56=12/29; 57=737")',
        "browse_act('57=737'): page unchanged after the action",
        "Exception running command browse_act(57=737): boom (TimeoutError)",
        "browse_act(57=737",
    ], ids=["the command line", "the reply as written", "the tool's label", "an error after the call", "cut off"])
    def test_no_typed_value_is_left_wherever_the_call_stands(self, line):
        masked = BrowseFillMask.mask_fills(line)
        assert "737" not in masked and "12/29" not in masked
        assert "57=" in masked

    def test_a_value_holding_a_parenthesis_is_masked_whole(self):
        assert "tail" not in BrowseFillMask.mask_fills("Running command: browse_act(4=pw)tail; 5)")

    def test_only_the_lines_with_a_call_change(self):
        text = "first = 1\nbrowse_act(9=hello)\nlast (x=2)"
        first, call, last = BrowseFillMask.mask_fills(text).split("\n")
        assert (first, last) == ("first = 1", "last (x=2)")
        assert "hello" not in call

    def test_a_click_is_shown_as_written(self):
        assert BrowseFillMask.mask_fills("Running command: browse_act(12)") == "Running command: browse_act(12)"

    @pytest.mark.parametrize("name", ["qwen", "dsml", "invoke"])
    def test_a_call_written_as_markup_leaves_no_typed_value_either(self, name):
        from assistant.commands.formats.CommandFormats import CommandFormats
        written = CommandFormats.BY_NAME[name].render_tool_call("browse_act", {"spec": "56=12/29; 57=737"})
        assert "737" in written, "the format writes the value into the reply"
        masked = BrowseFillMask.mask_fills(f"Reply: {written}")
        assert "737" not in masked and "12/29" not in masked

    @pytest.mark.parametrize("name", ["qwen", "dsml", "invoke"])
    def test_markup_of_another_command_and_the_text_around_the_call_stay_as_written(self, name):
        from assistant.commands.formats.CommandFormats import CommandFormats
        fmt = CommandFormats.BY_NAME[name]
        other = fmt.render_tool_call("web_search", {"query": "ferry 7=9 timetable"})
        text = f"before 1=2\n{other}\n{fmt.render_tool_call('browse_act', {'spec': '57=737'})}\nafter 3=4"
        masked = BrowseFillMask.mask_fills(text)
        assert other in masked and masked.startswith("before 1=2\n") and masked.endswith("\nafter 3=4")
        assert "737" not in masked

    def test_a_markup_value_over_several_lines_is_masked_to_its_last_line(self):
        text = '<invoke name="browse_act">\n<parameter name="spec">4=first line\nsecret second</parameter>\n</invoke>'
        assert "secret" not in BrowseFillMask.mask_fills(text)

    def test_a_markup_call_cut_off_inside_its_value_is_masked_to_the_end(self):
        assert "737" not in BrowseFillMask.mask_fills('<function=browse_act>\n<parameter=spec>\n57=737')
