import threading

import pytest
from html import unescape

from interfaces.chat import RichText
from interfaces.chat.RichText import markdown_to_html, safe_prefix_end, sanitize, split, strip_tags


@pytest.mark.unit
class TestSanitize:
    def test_keeps_allowed_tags(self):
        assert sanitize("<b>bold</b>") == "<b>bold</b>"
        assert sanitize("<pre>code</pre>") == "<pre>code</pre>"
        assert sanitize('<a href="x">link</a>') == '<a href="x">link</a>'

    def test_strips_disallowed_tags(self):
        assert sanitize("<div>text</div>") == "text"
        assert sanitize("<span>hi</span>") == "hi"

    def test_mixed_tags(self):
        assert sanitize("<div><b>bold</b> in <span>span</span></div>") == "<b>bold</b> in span"

@pytest.mark.unit
class TestStripTags:
    def test_strips_all_tags(self):
        assert strip_tags("<b>bold</b> and <i>italic</i>") == "bold and italic"

    def test_strips_tags_with_attributes(self):
        assert strip_tags('<a href="url">link</a>') == "link"

@pytest.mark.unit
class TestSafePrefixEnd:
    def test_balanced_tags_returns_full_length(self):
        text = "Hello <b>bold</b> world"
        assert safe_prefix_end(text) == len(text)

    def test_unclosed_tag_cuts_before_it(self):
        assert safe_prefix_end("Hello <pre>code block") == 6

    def test_incomplete_tag_at_end(self):
        assert safe_prefix_end("Hello <b>bold</b> and <pr") == len("Hello <b>bold</b> and ")

    def test_nested_balanced_tags(self):
        text = "<b>bold <i>italic</i></b>"
        assert safe_prefix_end(text) == len(text)

    def test_unclosed_after_balanced(self):
        assert safe_prefix_end("<b>bold</b> then <pre>unclosed") == len("<b>bold</b> then ")

    def test_disallowed_tags_ignored(self):
        assert safe_prefix_end("Hello <div>content</div> <pre>unclosed") == len("Hello <div>content</div> ")

@pytest.mark.unit
class TestSplit:
    def test_long_splits_on_newline(self):
        text = "line one\nline two\nline three"
        chunks = split(text, max_length=20)
        assert len(chunks) > 1
        assert "".join(c.replace("\n", "") for c in chunks).replace(" ", "") == text.replace("\n", "").replace(" ", "")
        assert all(len(c) <= 20 for c in chunks)

    def test_no_newline_splits_on_space_then_hard(self):
        chunks = split("a" * 25, max_length=10)
        assert chunks == ["a" * 10, "a" * 10, "a" * 5]
        assert split("aaaa bbbb cccc", max_length=10) == ["aaaa bbbb", " cccc"]

    @pytest.mark.parametrize("text", [" " + "x" * 25, "\n" + "y" * 15])
    def test_break_point_at_index_zero_does_not_stall_or_emit_empty_chunks(self, text):
        # The remainder after a space-split starts with that space; an unbreakable run after it
        # used to make split_at == 0, appending "" forever. Run on a thread so a regression
        # fails instead of hanging the suite.
        result = []
        worker = threading.Thread(target=lambda: result.append(split(text, max_length=10)), daemon=True)
        worker.start()
        worker.join(timeout=3)
        assert result, "split() did not terminate"
        chunks = result[0]
        assert "".join(chunks) == text
        assert all(0 < len(c) <= 10 for c in chunks)


@pytest.mark.unit
class TestMarkdownToHtml:
    def test_headings(self):
        assert markdown_to_html("# Title") == "<b>Title</b>"
        assert markdown_to_html("## Section") == "<b>Section</b>"
        assert markdown_to_html("### Sub") == "<b>Sub</b>"

    def test_bold_and_inline_code(self):
        assert "<b>bold</b>" in markdown_to_html("This is **bold** text")
        assert "<code>foo()</code>" in markdown_to_html("Use `foo()` here")

    def test_code_block(self):
        result = markdown_to_html("```\nprint('hello')\n```")
        assert "<pre><code>" in result and "</code></pre>" in result
        assert "print(&#x27;hello&#x27;)" in result

    def test_unclosed_code_block(self):
        assert "</code></pre>" in markdown_to_html("```\ncode")

    def test_lists(self):
        result = markdown_to_html("- item one\n- item two")
        assert "• item one" in result and "• item two" in result and "<ul>" not in result
        assert "• item" in markdown_to_html("* item")
        result = markdown_to_html("1. first\n2. second")
        assert "1. first" in result and "2. second" in result

    def test_horizontal_rule(self):
        result = markdown_to_html("above\n---\nbelow")
        assert "———" in result and "<hr" not in result

    def test_table_rows_stay_pipe_rows_for_the_console_pane(self):
        result = markdown_to_html("| Time | **Event** |\n|------|-------|\n| 13:00 | `Start` |")
        assert result == "| Time | <b>Event</b> |\n|------|-------|\n| 13:00 | <code>Start</code> |"

    def test_html_entities_escaped(self):
        result = markdown_to_html("a < b & c > d")
        assert "&lt;" in result and "&amp;" in result and "&gt;" in result
        assert "&lt;" in markdown_to_html("```\nif a < b:\n```")

@pytest.mark.unit
class TestTablesAsPre:
    TABLE = "| Purpose | Calls | Share |\n|---|---|---|\n| Chat (27 requests) | 74 | 41% |\n| Memory | 1 | 13% |"

    def test_a_table_becomes_one_pre_block_with_aligned_columns_and_no_separator(self):
        result = RichText.tables_as_pre(f"Where it goes\n\n{self.TABLE}\n\nUnpriced: 121 calls")
        lines = result.split("\n")
        assert lines[0] == "Where it goes" and lines[-1] == "Unpriced: 121 calls"
        assert lines[2].startswith("<pre>") and lines[4].endswith("</pre>")
        assert "---" not in result
        block = [l.removeprefix("<pre>").removesuffix("</pre>") for l in lines[2:5]]
        assert block[1].index("41%") == block[2].index("13%") == len(block[0]) - 3

    def test_numbers_right_align_and_text_left_aligns(self):
        block = RichText.tables_as_pre("| Name | $ |\n|---|---|\n| Chat | 4.57 |\n| Photos | 10.1 |")
        rows = block.removeprefix("<pre>").removesuffix("</pre>").split("\n")
        assert rows[1].startswith("Chat  ") and rows[2].startswith("Photos")
        assert rows[1].endswith(" 4.57") and rows[2].endswith("10.1")
        assert len(rows[1]) == len(rows[2])

    def test_tags_in_cells_are_stripped_since_telegram_allows_none_inside_pre(self):
        block = RichText.tables_as_pre("| <b>Model</b> | <code>x</code> |\n| a | b |")
        assert "<b>" not in block and "<code>" not in block and "Model" in block

    def test_entities_count_as_one_character_for_the_padding(self):
        block = RichText.tables_as_pre("| a &amp; b | 1 |\n| ccccc | 2 |")
        rows = block.removeprefix("<pre>").removesuffix("</pre>").split("\n")
        assert unescape(rows[0]).index("1") == unescape(rows[1]).index("2")

    def test_pipes_inside_a_pre_block_and_mid_line_are_not_a_table(self):
        text = "a | b in prose\n<pre>| raw |\n| code |</pre>"
        assert RichText.tables_as_pre(text) == text

    def test_an_escaped_pipe_stays_in_its_cell(self):
        block = RichText.tables_as_pre("| a \\| b | c |")
        assert "a | b" in block

    def test_a_table_the_padding_pushes_past_the_limit_keeps_its_rows(self):
        """One wide cell pads every row of its column: a text that fit the message limit
        before the rendering must still fit after it, so such a table is left as written."""
        wide = "| Title | Note |\n|---|---|\n| " + "x" * 120 + " | 1 |\n" + "\n".join(f"| r{i} | {i} |" for i in range(20))
        assert len(wide) < 600 < len(RichText.tables_as_pre(wide))
        assert RichText.tables_as_pre(wide, 600) == wide
        assert RichText.tables_as_pre(self.TABLE, 600).startswith("<pre>")


class TestCommandLine:
    def test_escapes_and_appends_the_detail_in_its_own_italics(self):
        assert RichText.command_line("Reading <x> & y") == " • <i>Reading &lt;x&gt; &amp; y</i>"
        assert RichText.command_line("Reading x", "done after 4s, received 345 chars") == \
            " • <i>Reading x</i> <i>(done after 4s, received 345 chars)</i>"
        assert RichText.command_line("Reading x", "") == " • <i>Reading x</i>"


@pytest.mark.unit
class TestRepairMarkdown:
    def test_bold_pair_becomes_the_dialect_tag(self):
        assert RichText.repair_markdown("First: **who are you?** Tell me") == "First: <b>who are you?</b> Tell me"

    def test_backtick_span_becomes_code(self):
        assert RichText.repair_markdown("run `ls -la` now") == "run <code>ls -la</code> now"

    def test_existing_html_is_left_untouched(self):
        text = "a <b>bold</b> &lt;tag&gt; and <code>x</code>"
        assert RichText.repair_markdown(text) == text

    def test_unclosed_marker_while_streaming_is_left_alone(self):
        assert RichText.repair_markdown("First: **who am") == "First: **who am"
        assert RichText.repair_markdown("see `ls") == "see `ls"

    def test_pair_converts_once_it_closes_and_stays_converted(self):
        first = RichText.repair_markdown("say **hi")
        second = RichText.repair_markdown(first + "** there")
        assert second == "say <b>hi</b> there"
        assert RichText.repair_markdown(second) == second

    def test_backticks_around_a_tag_are_not_touched(self):
        text = "type `<b>` here"
        assert RichText.repair_markdown(text) == text
