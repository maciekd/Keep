import pytest

from common.DraftPlaceholder import SPINNER_FRAMES_BRAILLE, SPINNER_FRAMES_C64, is_draft_placeholder, progress_bar


@pytest.mark.unit
class TestIsDraftPlaceholder:
    """What a draft bubble reads as before its text arrives, in the HTML the transport sends and
    in the Markdown Telethon renders the same message as (a `<code>` span comes back in backticks)."""

    @pytest.mark.parametrize("text", [
        "<code>█</code>", "<code>░</code>", "`█`", "`░`", "█", " ░ \n",
        "Thinking…\n <code>█</code>", "Thinking…\n `░`",
        "Thinking (↓ 1.2k tokens · 45 t/s)…\n `█`",
        "Prefilling (43%)…\n <code>█</code>", "Prefilling (43% · 1.5k t/s)…\n `░`",
        "Running command…\n `█`", "Running commands…\n <code>░</code>",
        "Waiting in queue…\n `█`",
        "`⠋`", "Thinking…\n ⠹",
        "Thinking (↓ 1.2k tokens · 45 t/s)…", "Prefilling (43%)…", "Prefilling (43% · 1.5k t/s)…", "Thinking…",
    ])
    def test_a_frame_with_or_without_its_phase_line_or_a_phase_line_alone_is_a_placeholder(self, text):
        assert is_draft_placeholder(text) is True

    @pytest.mark.parametrize("text", [
        "", "   ", "\n", None,
        "Hi there", "█ nope", "`█` and then some", "Thinking", "Thinking (43%)… and more",
        "Any line over it…\n `█`", "Thinking…\n `█`\n Done", "Thinking (unclosed…\n `█`",
        "code: `x`",
    ])
    def test_anything_with_words_in_it_is_a_message(self, text):
        assert is_draft_placeholder(text) is False

    def test_every_frame_of_both_spinner_styles_is_known(self):
        for frame in SPINNER_FRAMES_C64 + SPINNER_FRAMES_BRAILLE:
            assert is_draft_placeholder(f"<code>{frame}</code>"), frame


@pytest.mark.unit
class TestPrefillBarLineIsAPlaceholder:
    """The prefill line carries a bar in a code span since 2026-11-09 — in the HTML the transport
    sends and in the Markdown Telethon reads it back as — and the feed must still tell it from
    a message."""

    @pytest.mark.parametrize("text", [
        "Prefilling (<code>▓▓▓▓░░░░░░</code>)…", "Prefilling (`▓▓▓▓░░░░░░`)…",
        "Prefilling (<code>▓▓▓▓░░░░░░</code> · 1.5k t/s)…", "Prefilling (`▓▓▓▓▓▓▓▓▓▓` · 1.5k t/s)…",
        "Prefilling (<code>░░░░░░░░░░</code>)…\n <code>█</code>",
    ])
    def test_the_bar_line_alone_or_over_a_frame_is_a_placeholder(self, text):
        assert is_draft_placeholder(text) is True


@pytest.mark.unit
class TestProgressBar:
    """`progress_bar` is the one bar the console, the Dashboard and Telegram draw: a fixed
    number of cells however far along, fine eighth-blocks for the terminal, whole cells for
    Telegram."""

    @pytest.mark.parametrize("fine", [True, False])
    @pytest.mark.parametrize("fraction", [-0.5, 0.0, 0.03, 0.426, 0.5, 0.96, 0.999, 1.0, 7.0])
    def test_the_bar_is_always_exactly_width_cells(self, fraction, fine):
        filled, empty = progress_bar(fraction, 10, fine=fine)
        assert len(filled) + len(empty) == 10
        assert set(empty) <= {"░"}

    def test_fractions_beyond_the_range_are_clamped(self):
        assert progress_bar(-0.5, 10) == progress_bar(0.0, 10)
        assert progress_bar(7.0, 10) == progress_bar(1.0, 10)
        assert progress_bar(1.0, 10)[1] == "", "a full bar has no empty part"

    def test_the_fine_bar_adds_an_eighth_block_for_the_remainder(self):
        filled, _ = progress_bar(0.45, 10)
        assert filled == "████▌", "4.5 cells: four full blocks and a half block"
        filled, _ = progress_bar(0.99, 10)
        assert filled[-1] == "▉" and len(filled) == 10, "at 9.9 cells the last cell is the fullest partial, never an eleventh"

    def test_the_coarse_bar_lights_a_cell_only_once_it_is_whole(self):
        filled, empty = progress_bar(0.426, 10, fine=False, full="▓")
        assert (filled, empty) == ("▓▓▓▓", "░░░░░░"), "4.26 cells: four whole ones"
        filled, _ = progress_bar(0.45, 10, fine=False, full="▓")
        assert filled == "▓▓▓▓", "4.5 cells is still four — the half cell is not a partial glyph and not a fifth cell"
        filled, _ = progress_bar(0.5, 10, fine=False, full="▓")
        assert filled == "▓▓▓▓▓"
        for fraction in (0.03, 0.5, 0.96):
            filled, _ = progress_bar(fraction, 10, fine=False, full="▓")
            assert set(filled) <= {"▓"}, fraction

    def test_the_glyphs_are_the_callers(self):
        filled, empty = progress_bar(0.5, 4, fine=False, full="#", empty="-")
        assert (filled, empty) == ("##", "--")
