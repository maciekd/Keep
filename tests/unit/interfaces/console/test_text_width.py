"""TextWidth: cells, not characters — what every console view measures and cuts with."""
from interfaces.console.framework.TextWidth import (cells, char_at, char_width, chunks, clip, column,
                                                    display_width, drawable, fit, ljust, normalize, rjust, wrap)


class TestNormalize:
    def test_presentation_selectors_are_dropped(self):
        assert normalize("☀️ rain ✔︎") == "☀ rain ✔"

    def test_skin_tone_modifiers_are_dropped(self):
        assert normalize("\U0001F44D\U0001F3FD") == "\U0001F44D"

    def test_a_zwj_sequence_collapses_to_its_first_pictograph(self):
        family = "\U0001F468‍\U0001F469‍\U0001F467"
        assert normalize(family) == "\U0001F468"

    def test_a_flag_with_selector_and_zwj_keeps_only_the_base(self):
        assert normalize("\U0001F3F3️‍\U0001F308") == "\U0001F3F3"

    def test_plain_text_and_regional_indicator_flags_pass_through(self):
        assert normalize("hello \U0001F1F8\U0001F1EA") == "hello \U0001F1F8\U0001F1EA"

    def test_normalize_keeps_control_characters(self):
        assert normalize("a\tb\x1b[0m") == "a\tb\x1b[0m"

    def test_drawable_turns_control_characters_into_spaces_but_keeps_a_newline(self):
        assert drawable("a\tb\r\x1b[0m\x00\x7f\x85c\nd") == "a b  [0m   c\nd"
        assert drawable("☀️\tx") == "☀ x"


class TestWidths:
    def test_pictographs_and_cjk_are_two_cells(self):
        assert char_width("\U0001F600") == 2 and char_width("漢") == 2

    def test_combining_marks_are_zero(self):
        assert char_width("́") == 0

    def test_control_characters_keep_a_cell(self):
        assert char_width("\x1b") == 1

    def test_display_width_counts_cells_after_normalizing(self):
        assert display_width("a\U0001F600b") == 4
        assert display_width("☀️") == 1
        assert display_width("é") == 1


class TestCells:
    def test_a_wide_glyph_leaves_an_empty_continuation(self):
        assert cells("a\U0001F600b") == ["a", "\U0001F600", "", "b"]

    def test_a_combining_mark_joins_the_glyph_before_it(self):
        assert cells("é\U0001F600́") == ["é", "\U0001F600́", ""]

    def test_char_at_addresses_cells(self):
        assert char_at("a\U0001F600b", 1) == "\U0001F600"
        assert char_at("a\U0001F600b", 2) == ""
        assert char_at("a\U0001F600b", 3) == "b"
        assert char_at("a\U0001F600b", 4) == ""


class TestCutting:
    def test_clip_never_splits_a_glyph(self):
        assert clip("a\U0001F600b", 2) == "a"
        assert clip("a\U0001F600b", 3) == "a\U0001F600"
        assert clip("abc", 0) == ""

    def test_fit_ends_in_an_ellipsis_when_cut(self):
        assert fit("a\U0001F600bcdef", 5) == "a..."
        assert fit("abc", 5) == "abc"

    def test_ljust_and_rjust_pad_by_cells(self):
        assert ljust("\U0001F600", 4) == "\U0001F600  "
        assert rjust("\U0001F600", 4) == "  \U0001F600"

    def test_chunks_break_on_cell_boundaries(self):
        assert chunks("ab\U0001F600cd", 3) == ["ab", "\U0001F600c", "d"]
        assert chunks("", 3) == [""]

    def test_column_is_the_cell_offset_of_a_character_index(self):
        assert column("a\U0001F600b", 2) == 3


class TestWrap:
    def test_lines_break_at_spaces_and_a_long_word_is_broken_at_the_width(self):
        assert wrap("hello world foo bar", 12) == ["hello world", "foo bar"]
        assert wrap("abcdefgh ij", 3) == ["abc", "def", "gh", "ij"]

    def test_a_wide_glyph_counts_two_cells(self):
        assert wrap("\U0001F600\U0001F600 ab", 4) == ["\U0001F600\U0001F600", "ab"]

    def test_the_overflow_past_the_last_allowed_row_is_joined_into_it_and_fitted(self):
        lines = wrap("one two three four five six", 9, rows=2)
        assert len(lines) == 2 and lines[0] == "one two"
        assert lines[1].startswith("three") and lines[1].endswith("...") and display_width(lines[1]) == 9

    def test_text_that_fits_the_rows_is_left_alone(self):
        assert wrap("one two three", 9, rows=2) == ["one two", "three"]
        assert wrap("short", 9, rows=1) == ["short"]
