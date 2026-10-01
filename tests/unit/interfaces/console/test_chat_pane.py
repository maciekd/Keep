"""ChatPane: the transcript model, the Codex-style rendering and the key handling."""
import pytest

from interfaces.chat.ChatTypes import PHASE_COMMANDS, PHASE_PREFILL, PHASE_THINKING, PHASE_WAITING, Progress
from interfaces.console.framework.TUIScreen import TUIScreen
from interfaces.console.widgets.ChatPane import (BULLET, KIND_ASSISTANT, KIND_DIFF, KIND_SECRET, KIND_SYSTEM,
                                                  KIND_USER, PROMPT, ChatPane, shimmer)
from interfaces.console.widgets.DiffText import (COLOR_ADDED, COLOR_ADDED_HIGHLIGHT_BG, COLOR_ADDED_LINE_BG,
                                                 COLOR_GUTTER_DEL, COLOR_REMOVED, COLOR_REMOVED_LINE_BG)

WIDTH, HEIGHT = 60, 16


def row(screen: TUIScreen, y: int) -> str:
    return "".join(c.char for c in screen.next_frame[y - 1]).rstrip()


def rows(screen: TUIScreen) -> list[str]:
    return [row(screen, y) for y in range(1, screen.height + 1)]


def reveal_all(pane: ChatPane):
    for entry in pane.entries:
        entry.body.reveal_all()


@pytest.fixture
def pane():
    return ChatPane()


@pytest.fixture
def screen():
    return TUIScreen(WIDTH, HEIGHT)


def draw(pane: ChatPane, screen: TUIScreen, frame: int = 0):
    screen.clear_buffer()
    reveal_all(pane)
    pane.draw(screen, frame, 1, 1, WIDTH, HEIGHT)
    return rows(screen)


class TestComposer:
    def test_empty_composer_shows_rules_prompt_and_footer(self, pane, screen):
        pane.add_notice("x " * 600)   # more transcript than fits: it must stop a row short of the rule
        lines = draw(pane, screen)
        assert lines[HEIGHT - 5] == ""
        assert lines[HEIGHT - 6] != ""
        assert ChatPane.FOOTER.strip()[:6] in lines[HEIGHT - 1]
        assert set(lines[HEIGHT - 2]) == {"─"}
        assert lines[HEIGHT - 3] == PROMPT.rstrip()
        cursor_cell = screen.next_frame[HEIGHT - 3][len(PROMPT)]
        assert cursor_cell.style.bg == ChatPane.COLOR_INPUT
        assert set(lines[HEIGHT - 4]) == {"─"}

    def test_typed_text_carries_the_cursor(self, pane, screen):
        for ch in "hello":
            pane.handle_input(ch)
        pane.handle_input('left')
        lines = draw(pane, screen)
        assert lines[HEIGHT - 3] == PROMPT + "hello"
        cursor_cell = screen.next_frame[HEIGHT - 3][len(PROMPT) + 4]
        assert cursor_cell.style.bg == ChatPane.COLOR_INPUT

    def test_long_input_wraps_onto_more_rows(self, pane, screen):
        text_w = WIDTH - len(PROMPT) - 1
        for ch in "x" * (text_w + 5):
            pane.handle_input(ch)
        lines = draw(pane, screen)
        assert lines[HEIGHT - 4] == PROMPT + "x" * text_w
        assert lines[HEIGHT - 3] == " " * len(PROMPT) + "xxxxx"
        assert set(lines[HEIGHT - 5]) == {"─"}

    def test_input_taller_than_the_composer_scrolls_to_keep_the_cursor_visible(self, pane, screen):
        text_w = WIDTH - len(PROMPT) - 1
        for ch in "a" * text_w + "b" * text_w + "c" * text_w + "d" * text_w + "eeeee":
            pane.handle_input(ch)
        lines = draw(pane, screen)
        top, bottom = HEIGHT - 5, HEIGHT - 3  # the three composer rows, 0-based
        assert lines[top] == PROMPT + "c" * text_w
        assert lines[top + 1] == " " * len(PROMPT) + "d" * text_w
        assert lines[bottom] == " " * len(PROMPT) + "eeeee"
        assert screen.next_frame[bottom][len(PROMPT) + 5].style.bg == ChatPane.COLOR_INPUT

        pane.handle_input('home')
        lines = draw(pane, screen)
        assert lines[top] == PROMPT + "a" * text_w
        assert lines[bottom] == " " * len(PROMPT) + "c" * text_w
        assert screen.next_frame[top][len(PROMPT)].style.bg == ChatPane.COLOR_INPUT

    def test_cursor_at_the_end_of_a_full_row_sits_on_a_fresh_row(self, pane, screen):
        text_w = WIDTH - len(PROMPT) - 1
        for ch in "a" * text_w + "b" * text_w + "c" * text_w:
            pane.handle_input(ch)
        lines = draw(pane, screen)
        top, bottom = HEIGHT - 5, HEIGHT - 3
        assert lines[top] == PROMPT + "b" * text_w
        assert lines[bottom] == ""
        assert screen.next_frame[bottom][len(PROMPT)].style.bg == ChatPane.COLOR_INPUT

    def test_footer_notice_is_right_aligned(self, pane, screen):
        pane.footer_notice = "WAITING FOR USER"
        lines = draw(pane, screen)
        assert lines[HEIGHT - 1].endswith("WAITING FOR USER")

    def test_enter_submits_and_clears(self, pane):
        sent = []
        pane.bind(on_submit=sent.append)
        for ch in "hi":
            pane.handle_input(ch)
        assert pane.handle_input('\r')
        assert sent == ["hi"]
        assert pane.editor.empty

    def test_shift_enter_breaks_a_line_and_enter_sends_both(self, pane, screen):
        sent = []
        pane.bind(on_submit=sent.append)
        for key in list("one") + ['shift_enter'] + list("two"):
            pane.handle_input(key)
        lines = draw(pane, screen)
        assert lines[HEIGHT - 4] == PROMPT + "one"
        assert lines[HEIGHT - 3] == " " * len(PROMPT) + "two"
        assert screen.next_frame[HEIGHT - 3][len(PROMPT) + 3].style.bg == ChatPane.COLOR_INPUT
        assert pane.handle_input('\r')
        assert sent == ["one\ntwo"]
        assert pane.editor.empty

    def test_the_cursor_walks_back_across_a_line_break(self, pane, screen):
        for key in list("ab") + ['shift_enter'] + list("cd") + ['left', 'left', 'left']:
            pane.handle_input(key)
        draw(pane, screen)
        assert screen.next_frame[HEIGHT - 4][len(PROMPT) + 2].style.bg == ChatPane.COLOR_INPUT

    def test_a_full_row_before_a_line_break_takes_no_extra_row(self, pane, screen):
        text_w = WIDTH - len(PROMPT) - 1
        for key in list("a" * text_w) + ['shift_enter', 'b']:
            pane.handle_input(key)
        assert pane._input_rows(text_w) == ["a" * text_w, "b"]
        lines = draw(pane, screen)
        assert lines[HEIGHT - 4] == PROMPT + "a" * text_w
        assert lines[HEIGHT - 3] == " " * len(PROMPT) + "b"

    def test_enter_on_empty_sends_nothing(self, pane):
        sent = []
        pane.bind(on_submit=sent.append)
        pane.handle_input('enter')
        assert sent == []

    def test_escape_clears_the_composer_when_idle(self, pane):
        interrupted = []
        pane.bind(on_interrupt=lambda: interrupted.append(True))
        for ch in "abc":
            pane.handle_input(ch)
        pane.handle_input('escape')
        assert pane.editor.empty
        assert interrupted == []

    def test_escape_interrupts_while_a_reply_runs(self, pane):
        interrupted = []
        pane.bind(on_interrupt=lambda: interrupted.append(True))
        pane.handle_input('a')
        pane.set_working(Progress(phase=PHASE_WAITING))
        pane.handle_input('escape')
        assert interrupted == [True]
        assert pane.editor.text == "a"

    def test_escape_interrupts_while_an_assistant_entry_streams(self, pane):
        interrupted = []
        pane.bind(on_interrupt=lambda: interrupted.append(True))
        pane.add_assistant("partial")
        pane.handle_input('escape')
        assert interrupted == [True]


class TestTranscript:
    def test_user_turn_and_reply(self, pane, screen):
        pane.add_user("turn on the lights")
        entry = pane.add_assistant("On <b>it</b>.")
        pane.update_entry(entry, "On <b>it</b>. Done.", final=True)
        lines = draw(pane, screen)
        assert lines[0] == PROMPT + "turn on the lights"
        assert lines[1] == ""
        assert lines[2] == BULLET + "On it. Done."
        bold_cell = screen.next_frame[2][len(BULLET) + 3]
        assert bold_cell.style.bold and bold_cell.style.fg == ChatPane.COLOR_EMPHASIS
        assert screen.next_frame[2][len(BULLET)].style.fg == ChatPane.COLOR_ASSISTANT

    def test_only_what_the_assistant_emits_reveals_with_the_sweep(self, pane):
        pane.add_user("hi")
        pane.add_notice(" • <i>Reading the calendar</i>")
        pane.add_media("photo", "", 1024)
        thinking = pane.set_thinking(None, "hmm")
        reply = pane.add_assistant("Hello there.")
        assert [e.body.animated for e in pane.entries] == [False, False, False, True, True]
        assert all(e.body.done for e in pane.entries if e not in (thinking, reply))
        assert not thinking.body.done and not reply.body.done

    def test_hub_command_notification_keeps_one_bullet(self, pane, screen):
        pane.add_notice(" • <i>Turning on kitchen</i>")
        lines = draw(pane, screen)
        assert lines[0] == BULLET + "Turning on kitchen"
        assert screen.next_frame[0][len(BULLET)].style.italic

    def test_plain_notice_gets_a_bullet(self, pane, screen):
        pane.add_notice("Sorry, I'm busy right now.")
        assert draw(pane, screen)[0] == BULLET + "Sorry, I'm busy right now."

    def test_media_and_thinking_are_dim_italic(self, pane, screen):
        pane.show_debug = True
        pane.add_media("photo", "the garden", 2048)
        pane.set_thinking(None, "let me see")
        lines = draw(pane, screen)
        assert lines[0] == BULLET + "[photo, 2 kB] the garden"
        assert lines[1] == BULLET + "let me see"
        assert screen.next_frame[1][len(BULLET)].style.fg == ChatPane.COLOR_THINKING
        assert screen.next_frame[1][len(BULLET)].style.italic

    def test_thinking_is_hidden_unless_show_debug(self, pane, screen):
        pane.set_thinking(None, "let me see", done=True)
        pane.add_assistant("Sure.", final=True)
        lines = draw(pane, screen)
        assert lines[0] == BULLET + "Sure." and lines[1] == ""
        pane.show_debug = True
        lines = draw(pane, screen)
        assert lines[0] == BULLET + "let me see" and lines[2] == BULLET + "Sure."

    def test_fallback_notice_is_hidden_unless_show_debug(self, pane, screen):
        pane.add_assistant("Sure.\n\n(Fable refused this request, had to use Opus fallback)", final=True)
        lines = draw(pane, screen)
        assert lines[0] == BULLET + "Sure." and lines[1] == ""
        pane.show_debug = True
        lines = draw(pane, screen)
        assert lines[2].strip() == "(Fable refused this request, had to use Opus fallback)"

    def test_a_stripped_command_leaves_no_blank_rows(self, pane, screen):
        """The stream drops the command but leaves its line (" ") between two paragraph
        breaks, and the fallback notice follows; the bubble ends at the prose."""
        pane.add_assistant("Both saved. Let me fire it up:\n\n \n\n"
                           "(Fable refused this request, had to use Opus fallback)", final=True)
        lines = draw(pane, screen)
        assert lines[0] == BULLET + "Both saved. Let me fire it up:" and lines[1] == "" and lines[2] == ""

    def test_blank_line_runs_inside_a_reply_are_one_paragraph_break(self, pane, screen):
        pane.add_assistant("Saved.\n\n \n\nNow the password:", final=True)
        lines = draw(pane, screen)
        assert lines[:3] == [BULLET + "Saved.", "", "  Now the password:"]

    def test_fallback_notice_never_shows_while_the_reply_reveals(self, pane, screen):
        entry = pane.add_assistant("Sure.\n\n(Fable refused this request, had to use Opus fallback)", final=True)
        entry.body._revealed = float(len("Sure.\n\n(Fable refused"))   # mid-notice
        screen.clear_buffer()
        pane.draw(screen, 0, 1, 1, WIDTH, HEIGHT)
        lines = rows(screen)
        assert lines[0] == BULLET + "Sure." and lines[1] == ""

    def test_footer_progress_bar_sits_centred_between_the_hints_and_the_strip(self, pane):
        from interfaces.console.widgets.FeedStrip import FeedChip
        from interfaces.console.Theme import COLOR_YELLOW_FG
        width = 140
        pane.footer_chips = [FeedChip("●", ChatPane.COLOR_DIM, "Photos")]
        pane.footer_progress, pane.footer_eta = 0.5, 4 * 3600
        screen = TUIScreen(width, HEIGHT)
        screen.clear_buffer()
        pane.draw(screen, 0, 1, 1, width, HEIGHT)
        line = row(screen, HEIGHT)
        assert line.endswith("Feeds  ● Photos") and line.startswith(ChatPane.FOOTER)
        label_at = line.index(ChatPane.PROGRESS_LABEL)
        bar_at = label_at + len(ChatPane.PROGRESS_LABEL)
        assert line[bar_at:bar_at + 10] == "█" * 10, "half of a 20-cell bar is filled"
        assert screen.next_frame[HEIGHT - 1][bar_at].style.fg == COLOR_YELLOW_FG
        eta_at = bar_at + ChatPane.PROGRESS_BAR_WIDTH
        assert line[eta_at:eta_at + 8] == " ETA: 4h"
        hints_end = len(f"{ChatPane.FOOTER}   {ChatPane.DEBUG_HINT} off")
        strip_start = line.index("Feeds  ")
        gap_left, gap_right = label_at - hints_end, strip_start - (eta_at + 8)
        assert abs(gap_left - gap_right) <= 1

    def test_footer_progress_bar_drops_its_label_then_the_eta_then_itself_when_squeezed(self, pane):
        pane.footer_progress, pane.footer_eta = 1.0, 34
        for width, expect_label, expect_eta, expect_bar in ((108, False, True, True), (100, False, False, True),
                                                            (88, False, False, False)):
            screen = TUIScreen(width, HEIGHT)
            screen.clear_buffer()
            pane.draw(screen, 0, 1, 1, width, HEIGHT)
            line = row(screen, HEIGHT)
            assert (ChatPane.PROGRESS_LABEL in line) is expect_label, width
            assert ("ETA: 34s" in line) is expect_eta, width
            assert ("█" * ChatPane.PROGRESS_BAR_WIDTH in line) is expect_bar, width

    def test_no_footer_progress_bar_without_a_curation(self, pane):
        screen = TUIScreen(140, HEIGHT)
        screen.clear_buffer()
        pane.draw(screen, 0, 1, 1, 140, HEIGHT)
        assert "█" not in row(screen, HEIGHT) and ChatPane.PROGRESS_LABEL not in row(screen, HEIGHT)

    def test_footer_shows_the_debug_setting(self, pane):
        screen = TUIScreen(100, HEIGHT)   # wide enough for the whole footer
        screen.clear_buffer()
        pane.draw(screen, 0, 1, 1, 100, HEIGHT)
        assert row(screen, HEIGHT).endswith("^d debug off")
        pane.show_debug = True
        screen.clear_buffer()
        pane.draw(screen, 0, 1, 1, 100, HEIGHT)
        assert row(screen, HEIGHT).endswith("^d debug on")

    def test_thinking_updates_in_place(self, pane):
        first = pane.set_thinking(None, "a")
        second = pane.set_thinking(first, "ab", done=True)
        assert first is second and first.body.text == "ab" and first.final
        assert len(pane.entries) == 1

    def test_working_line_counts_seconds_once_it_takes_a_while(self, pane, screen):
        pane.add_user("hi")
        pane.set_working(Progress(phase=PHASE_WAITING, elapsed=1.2, tick=1))
        lines = draw(pane, screen)
        assert lines[2] == BULLET + "Thinking…"
        pane.set_working(Progress(phase=PHASE_WAITING, elapsed=12.3, tick=2))
        assert draw(pane, screen)[2] == BULLET + "Thinking… (12s)"
        assert screen.next_frame[2][len(BULLET)].style.fg == shimmer(ChatPane.COLOR_WORKING, 0, 0)
        assert screen.next_frame[2][len(BULLET)].style.bold
        assert not screen.next_frame[2][len(BULLET) + len("Thinking… ")].style.bold
        assert screen.next_frame[2][len(BULLET) + len("Thinking… ")].style.fg == ChatPane.COLOR_DIM

    def test_working_line_draws_the_prefill_bar_and_percentage(self, pane, screen):
        pane.add_user("hi")
        pane.set_working(Progress(phase=PHASE_PREFILL, prefill=50.0, elapsed=1.0))
        line = draw(pane, screen)[2]
        assert line.startswith(BULLET + "Prefilling… ")
        width = ChatPane.PREFILL_BAR_WIDTH
        bar = line[len(BULLET + "Prefilling… "):][:width]
        assert bar.count("█") == width // 2 and bar.count("░") == width - width // 2
        assert line.endswith(" (50%)")
        bar_x = len(BULLET + "Prefilling… ")
        assert screen.next_frame[2][bar_x].style.fg == ChatPane.COLOR_WORKING
        assert screen.next_frame[2][bar_x + width - 1].style.fg == ChatPane.COLOR_DIM
        pane.set_working(Progress(phase=PHASE_PREFILL, prefill=99.0, elapsed=12.0))
        assert draw(pane, screen)[2].endswith("▉ (99% · 12s)")

    def test_working_line_names_the_running_commands(self, pane, screen):
        pane.add_user("hi")
        pane.set_working(Progress(phase=PHASE_COMMANDS, commands=1, elapsed=1.0))
        assert draw(pane, screen)[2] == BULLET + "Running command…"
        pane.set_working(Progress(phase=PHASE_COMMANDS, commands=2, elapsed=4.2))
        assert draw(pane, screen)[2] == BULLET + "Running commands… (4s)"
        pane.set_working(Progress(phase=PHASE_COMMANDS, commands=1, elapsed=72.3))
        assert draw(pane, screen)[2] == BULLET + "Running command… (1m 12s)"

    def test_command_echo_detail_is_dim_and_leaves_the_body_alone(self, pane, screen):
        entry = pane.add_notice(" • <i>Reading x</i>")
        assert draw(pane, screen)[0] == BULLET + "Reading x"
        pane.set_detail(entry, "done after 4s, received 345 chars")
        assert draw(pane, screen)[0] == BULLET + "Reading x (done after 4s, received 345 chars)"
        assert screen.next_frame[0][len(BULLET) + len("Reading x ")].style.fg == ChatPane.COLOR_DIM
        assert entry.body.text == " • <i>Reading x</i>"
        pane.set_detail(entry, None)
        assert draw(pane, screen)[0] == BULLET + "Reading x"

    def test_system_row_detail_trails_the_body_dim(self, pane, screen):
        entry = pane.insert(KIND_SYSTEM, "Incoming voice request: hi", 1000.0, header="System (VoiceToVoice)")
        pane.set_detail(entry, "rejected: lost the vote")
        rows = draw(pane, screen)
        row = next(i for i, r in enumerate(rows) if "Incoming voice request" in r)
        assert rows[row].strip() == "Incoming voice request: hi (rejected: lost the vote)"
        assert screen.next_frame[row][rows[row].index("(rejected")].style.fg == ChatPane.COLOR_DIM

    def test_working_label_shimmers_frame_by_frame(self, pane, screen):
        pane.add_user("hi")
        pane.set_working(Progress(phase=PHASE_WAITING))
        draw(pane, screen, frame=0)
        first = [screen.next_frame[2][len(BULLET) + i].style.fg for i in range(len("Working…"))]
        draw(pane, screen, frame=7)
        later = [screen.next_frame[2][len(BULLET) + i].style.fg for i in range(len("Working…"))]
        assert first != later
        assert len(set(first)) > 1                      # a sweep, not a uniform pulse
        base = ChatPane.COLOR_WORKING
        for c in first + later:                          # the hue is kept, only brightness moves
            ratios = [c.r / base.r, c.g / base.g, c.b / base.b]
            assert max(ratios) - min(ratios) < 0.03
            assert 0.4 <= min(ratios) and max(ratios) <= 1.05
        pane.set_working(Progress(phase=PHASE_THINKING, elapsed=3, output_tokens=1500, tick=3))
        assert draw(pane, screen)[2] == BULLET + "Thinking… (↓ 1.5k tokens · 3s)"
        pane.set_working(Progress(phase=PHASE_THINKING, elapsed=2, output_tokens=40, tick=4))
        assert draw(pane, screen)[2] == BULLET + "Thinking… (↓ 40 tokens)"
        pane.set_working(None)
        assert draw(pane, screen)[2] == ""

    def test_status_row_sits_above_the_composer(self, pane, screen):
        pane.set_status(" • <i>Checking the weather</i>")
        lines = draw(pane, screen)
        assert lines[HEIGHT - 6] == "• Checking the weather"
        assert lines[HEIGHT - 5] == ""
        assert set(lines[HEIGHT - 4]) == {"─"}
        pane.set_status(None)
        assert pane.status is None

    def test_long_reply_wraps_with_hanging_indent(self, pane, screen):
        pane.add_assistant("word " * 30, final=True)
        lines = draw(pane, screen)
        assert lines[0].startswith(BULLET + "word")
        assert lines[1].startswith("  word")

    def test_remove_entry(self, pane):
        entry = pane.add_assistant("x")
        pane.remove_entry(entry)
        pane.remove_entry(entry)   # a second removal is harmless
        assert pane.entries == []

    def test_entries_are_capped(self, pane):
        for i in range(ChatPane.MAX_ENTRIES + 10):
            pane.add_notice(f"n{i}")
        entries = pane.entries
        assert len(entries) == ChatPane.MAX_ENTRIES
        assert entries[-1].body.text == f"n{ChatPane.MAX_ENTRIES + 9}"

    def test_arrows_scroll_a_line_at_a_time(self, pane, screen):
        for i in range(40):
            pane.add_user(f"line {i}")
        draw(pane, screen)
        bottom = draw(pane, screen)
        assert any("line 39" in line for line in bottom)
        for _ in range(3):
            pane.handle_input('up')
        assert pane.scroll_offset == 3
        scrolled = draw(pane, screen)
        assert not any("line 39" in line for line in scrolled)
        pane.handle_input('down')
        assert pane.scroll_offset == 2
        for _ in range(5):
            pane.handle_input('down')
        assert pane.scroll_offset == 0

    def test_page_keys_browse_the_composer_history(self, pane, screen):
        pane.handle_input('a'); pane.handle_input('enter')
        pane.handle_input('b'); pane.handle_input('enter')
        assert pane.editor.empty
        pane.handle_input('pageup')
        assert pane.editor.text == "b"
        pane.handle_input('pageup')
        assert pane.editor.text == "a"
        pane.handle_input('pagedown')
        assert pane.editor.text == "b"
        pane.handle_input('pagedown')
        assert pane.editor.empty
        assert pane.scroll_offset == 0

    def test_new_user_turn_scrolls_to_the_bottom(self, pane, screen):
        for i in range(40):
            pane.add_notice(f"n{i}")
        draw(pane, screen)
        pane.handle_input('up')
        assert pane.scroll_offset > 0
        pane.add_user("hello")
        assert pane.scroll_offset == 0

    def test_stream_cursor_follows_a_streaming_reply(self, pane, screen):
        pane.add_assistant("Hel")
        lines = draw(pane, screen)
        assert lines[0] == BULLET + "Hel"
        head = screen.next_frame[0][len(BULLET) + 3]
        assert head.style.bg == ChatPane.COLOR_ASSISTANT

    def test_trail_repaints_continuation_lines_in_place(self, pane, screen):
        """The sweep's trail re-draws the characters under it; on wrapped lines (no bullet)
        they must not shift left by the bullet width."""
        entry = pane.add_assistant("first line\nEn")
        draw(pane, screen, frame=0)
        pane.update_entry(entry, "first line\nEnergy report")
        lines = draw(pane, screen, frame=1)
        assert lines[0] == BULLET + "first line"
        assert lines[1] == "  Energy report"

    def test_trail_is_dropped_when_the_transcript_moves(self, pane, screen):
        """The trail remembers screen cells; once the lines under them scroll away it must
        not keep fading over whatever text moved into those cells."""
        for i in range(HEIGHT):
            pane.add_user(f"line {i}")
        entry = pane.add_assistant("Hel")
        draw(pane, screen, frame=0)
        pane.update_entry(entry, "Hello there")
        draw(pane, screen, frame=1)
        assert len(entry.body._cursor._trail) > 0
        pane.update_entry(entry, "Hello there\nand more")   # wraps to a new row: the transcript scrolls
        draw(pane, screen, frame=2)
        assert len(entry.body._cursor._trail) == 0
        assert entry.body._cursor._last_pos == (1 + len(BULLET) + len("and more"), pane._last_transcript_rows)

    def test_trail_survives_a_frame_where_nothing_moved(self, pane, screen):
        entry = pane.add_assistant("Hel")
        draw(pane, screen, frame=0)
        pane.update_entry(entry, "Hello there")
        draw(pane, screen, frame=1)
        draw(pane, screen, frame=2)
        assert len(entry.body._cursor._trail) > 0

    def test_scrollbar_only_when_overflowing(self, pane, screen):
        pane.add_user("one")
        lines = draw(pane, screen)
        assert "┃" not in "".join(lines) and "│" not in "".join(lines)
        for i in range(40):
            pane.add_user(f"line {i}")
        lines = draw(pane, screen)
        assert "┃" in "".join(lines)


class TestHistoryEntries:
    def test_system_entry_shows_a_dim_header_over_grey_text(self, pane, screen):
        pane.insert(KIND_SYSTEM, "Prompt section `TeslaInfo` updated:\n-Charlottes Model 3", 1000.0,
                    header="System (startup)")
        lines = draw(pane, screen)
        assert lines[0] == BULLET + "System (startup)"
        assert screen.next_frame[0][len(BULLET) + len("System ")].style.fg == ChatPane.COLOR_HEADER_DETAIL
        assert lines[1] == "  Prompt section `TeslaInfo` updated:"
        assert lines[2] == "  -Charlottes Model 3"
        assert lines[3] == ""
        assert screen.next_frame[0][len(BULLET)].style.fg == ChatPane.COLOR_DIM
        assert screen.next_frame[1][2].style.fg == ChatPane.COLOR_SYSTEM

    def test_media_message_shows_text_short_and_a_dim_full_description(self, pane, screen):
        pane.insert(KIND_SYSTEM, '<b>Marcus → Charlie</b> nice\n[A shape]\n<span class="description">A cube.</span>',
                    1000.0, header="System (Telegram)")
        lines = draw(pane, screen)
        assert lines[1:4] == ["  Marcus → Charlie nice", "  [A shape]", "  A cube."]
        assert screen.next_frame[1][2 + len("Marcus → Charlie ")].style.fg == ChatPane.COLOR_SYSTEM
        assert screen.next_frame[2][2].style.fg == ChatPane.COLOR_SYSTEM
        assert screen.next_frame[3][2].style.fg == ChatPane.COLOR_DIM

    def test_system_message_row_shows_the_parties_in_white(self, pane, screen):
        pane.insert(KIND_SYSTEM, "<b>Charlotte → Marcus</b> hej &amp; hå", 1000.0, header="System (iMessage)")
        lines = draw(pane, screen)
        assert lines[1] == "  Charlotte → Marcus hej & hå"
        parties = screen.next_frame[1][2]
        assert parties.style.bold and parties.style.fg == ChatPane.COLOR_ASSISTANT
        assert screen.next_frame[1][2 + len("Charlotte → Marcus ")].style.fg == ChatPane.COLOR_SYSTEM

    def test_diff_entry_draws_its_lines_red_and_green(self, pane, screen):
        pane.insert(KIND_DIFF, 'Prompt section `IndigoData` updated:\n-"Temp": 22.3 °C\n+"Temp": 22.2 °C', 1000.0,
                    header="System (prompt_delta)")
        lines = draw(pane, screen)
        assert lines[1] == "  Prompt section `IndigoData` updated:"
        assert lines[2] == '  - "Temp": 22.3 °C'
        assert lines[3] == '  + "Temp": 22.2 °C'
        assert screen.next_frame[1][2].style.fg == ChatPane.COLOR_SYSTEM
        assert screen.next_frame[2][2].style.fg == COLOR_GUTTER_DEL
        assert screen.next_frame[2][4].style.fg == COLOR_REMOVED
        changed = screen.next_frame[3][2 + len('+ "Temp": 22.')]
        assert changed.style.fg == COLOR_ADDED and changed.style.bg == COLOR_ADDED_HIGHLIGHT_BG
        assert screen.next_frame[3][4].style.bg == COLOR_ADDED_LINE_BG
        assert screen.next_frame[2][WIDTH - 2].style.bg == COLOR_REMOVED_LINE_BG   # the tint spans the row
        assert screen.next_frame[2][WIDTH - 1].style.bg is None                    # but not the scrollbar
        assert screen.next_frame[1][4].style.bg is None

    def test_other_channels_user_turn_puts_the_prompt_on_the_words_under_a_plain_header(self, pane, screen):
        pane.insert(KIND_USER, "Where's my Model X?", 1000.0, header="Marcus (15:14, VoiceToVoice in Kitchen)")
        pane.insert(KIND_ASSISTANT, "Parked at <b>Kyrkogatan</b>.", 1001.0, header="Assistant (15:14)")
        lines = draw(pane, screen)
        assert lines[0] == "  Marcus (15:14, VoiceToVoice in Kitchen)"
        assert lines[1] == PROMPT + "Where's my Model X?"
        assert screen.next_frame[0][2].style.fg == ChatPane.COLOR_DIM
        assert screen.next_frame[1][0].style.fg == ChatPane.COLOR_USER
        assert lines[3] == BULLET + "Assistant (15:14)"     # the bullet stays on the header line
        assert lines[4] == "  Parked at Kyrkogatan."
        assert screen.next_frame[4][2 + len("Parked at ")].style.bold

    def test_insert_places_by_timestamp(self, pane):
        pane.add_user("typed now")                       # timestamp = now
        pane.insert(KIND_SYSTEM, "earlier event", 1000.0)
        pane.insert(KIND_SYSTEM, "even earlier", 500.0)
        pane.insert(KIND_SYSTEM, "between", 750.0)
        assert [e.body.text for e in pane.entries] == ["even earlier", "between", "earlier event", "typed now"]

    def test_inserted_entries_are_instant_and_capped(self, pane):
        for i in range(ChatPane.MAX_ENTRIES + 5):
            pane.insert(KIND_SYSTEM, f"event {i}", float(i))
        assert len(pane.entries) == ChatPane.MAX_ENTRIES
        assert pane.entries[0].body.text == "event 5"
        assert not pane.entries[0].body.animated and pane.entries[0].body.done

    def test_layout_is_cached_per_entry_until_text_or_width_changes(self, pane):
        entry = pane.insert(KIND_ASSISTANT, "one <b>two</b> three", 1000.0)
        first = pane._entry_lines(entry, 40)
        assert pane._entry_lines(entry, 40) is first
        assert pane._entry_lines(entry, 5) is not first
        pane.update_entry(entry, "one two three four")
        assert pane._entry_lines(entry, 5) is not first
        assert "".join(s.text for s in pane._entry_lines(entry, 40)[0].spans) == "one two three four"


class TestApproval:
    def test_prompt_renders_and_y_n_answer_it(self, pane, screen):
        answers = []
        pane.bind(on_approval=lambda token, decision: answers.append((token, decision)))
        entry = pane.add_approval("tok123", "Unlock the front door", "requested by Guest")
        lines = draw(pane, screen)
        assert lines[0] == BULLET + "Unlock the front door"
        assert lines[1] == "  requested by Guest"
        assert lines[2] == "  " + ChatPane.APPROVAL_KEYS
        assert lines[HEIGHT - 3] == PROMPT + ChatPane.APPROVAL_HINT
        assert pane.handle_input('x')          # modal: swallowed
        assert pane.editor.empty
        pane.handle_input('y')
        assert answers == [("tok123", "approve")]
        pane.close_approval(entry, "✅ Approved by console")
        assert pane.pending_approval is None
        lines = draw(pane, screen)
        assert lines[0] == BULLET + "Unlock the front door — Approved by console"
        assert lines[1] == ""
        assert lines[2] == ""

    def test_n_and_escape_deny(self, pane):
        answers = []
        pane.bind(on_approval=lambda token, decision: answers.append(decision))
        pane.add_approval("t1", "Do it", "")
        pane.handle_input('n')
        pane.add_approval("t2", "Do it", "")
        pane.handle_input('escape')
        assert answers == ["deny", "deny"]

    def test_closing_the_newest_of_two_prompts_hands_the_keys_to_the_older_one(self, pane):
        answers = []
        pane.bind(on_approval=lambda token, decision: answers.append((token, decision)))
        first = pane.add_approval("t1", "Unlock the door", "")
        second = pane.add_approval("t2", "Send the email", "")
        pane.handle_input('y')
        assert answers == [("t2", "approve")]  # the newest prompt is the one on screen
        pane.close_approval(second, "✅ Approved")
        assert pane.pending_approval is first
        pane.handle_input('n')
        assert answers[-1] == ("t1", "deny")
        pane.close_approval(first, "❌ Denied")
        assert pane.pending_approval is None
        assert pane.handle_input('x') is True and not pane.editor.empty  # composer is back

    def test_arrows_still_scroll_while_modal(self, pane, screen):
        for i in range(40):
            pane.add_notice(f"n{i}")
        pane.add_approval("t", "Q", "")
        draw(pane, screen)
        assert pane.handle_input('up')
        assert pane.scroll_offset > 0


class TestSecretPrompt:
    """The masked field for vault_request: the value is typed as dots, handed on once, and
    never becomes part of the transcript, the composer history or tui_state.json."""

    def test_prompt_renders_masked_and_enter_hands_the_value_on(self, pane, screen):
        answers = []
        pane.bind(on_secret=lambda token, value: answers.append((token, value)))
        entry = pane.add_secret_prompt("tok123", "Amex (card)", "")
        lines = draw(pane, screen)
        assert lines[0] == BULLET + "Amex (card)"
        assert lines[1] == ""                          # no key hints under the title: the footer has them
        assert lines[HEIGHT - 1] == ChatPane.SECRET_FOOTER
        for ch in "4111":
            assert pane.handle_input(ch)
        lines = draw(pane, screen)
        assert lines[HEIGHT - 3] == PROMPT + "Amex (card): ••••"
        assert "4111" not in "\n".join(lines)
        assert pane.editor.empty                       # the chat composer never saw a key

        pane.handle_input('\t')
        assert pane.secret_revealed
        assert draw(pane, screen)[HEIGHT - 3] == PROMPT + "Amex (card): 4111"
        pane.handle_input('\t')
        pane.handle_input('enter')
        assert answers == [("tok123", "4111")]
        assert pane.pending_secret is entry            # until the hub closes it
        assert draw(pane, screen)[HEIGHT - 3] == PROMPT + "Amex (card):"   # field emptied at once

        pane.close_secret(entry, "✅ Stored as amex (•••• 1111)")
        assert pane.pending_secret is None
        lines = draw(pane, screen)
        assert lines[0] == BULLET + "Amex (card) — Stored as amex (•••• 1111)"
        assert lines[1] == ""
        assert lines[HEIGHT - 1].startswith(ChatPane.FOOTER)
        assert "4111" not in "\n".join(lines)
        assert pane.handle_input('x') is True and not pane.editor.empty    # composer is back

    def test_the_value_never_enters_the_transcript_history_or_tui_state(self, pane, monkeypatch):
        from interfaces.console import TUIState
        monkeypatch.setattr(TUIState, "save", lambda *a, **k: pytest.fail("TUIState.save called with a secret"))
        answers = []
        pane.bind(on_secret=lambda token, value: answers.append(value))
        entry = pane.add_secret_prompt("t", "Password", "")
        for ch in "hunter2":
            pane.handle_input(ch)
        pane.handle_input('enter')
        pane.close_secret(entry, "✅ Stored")
        assert answers == ["hunter2"]
        assert all("hunter2" not in e.body.text for e in pane.entries)
        assert pane.editor.history == [] and pane.editor.empty

    def test_escape_cancels_and_an_empty_enter_is_ignored(self, pane):
        answers = []
        pane.bind(on_secret=lambda token, value: answers.append((token, value)))
        pane.add_secret_prompt("t1", "Password", "")
        pane.handle_input('enter')
        assert answers == []
        pane.handle_input('a')
        pane.handle_input('escape')
        assert answers == [("t1", None)]

    def test_details_are_shown_under_the_title(self, pane, screen):
        pane.add_secret_prompt("t", "Visa (card)", "⚠ card number must be 13-19 digits — try again (2/3)")
        lines = draw(pane, screen)
        assert lines[0] == BULLET + "Visa (card)"
        assert lines[1] == "  card number must be 13-19 digits — try again (2/3)"
        assert lines[2] == ""
        assert draw(pane, screen)[HEIGHT - 3] == PROMPT + "Visa (card):"

    def test_local_only_note_sits_at_the_right_of_the_field_when_it_fits(self, pane):
        pane.add_secret_prompt("t", "Setting telegram_bot.bot_token", "")
        for ch in "abc":
            pane.handle_input(ch)
        wide = TUIScreen(120, HEIGHT)
        wide.clear_buffer()
        pane.draw(wide, 0, 1, 1, 120, HEIGHT)
        line = row(wide, HEIGHT - 2)
        assert line.startswith(PROMPT + "Setting telegram_bot.bot_token: •••")
        assert line.endswith(ChatPane.SECRET_NOTE)
        assert "abc" not in line

    def test_local_only_note_gives_way_to_the_field_in_a_narrow_pane(self, pane, screen):
        pane.add_secret_prompt("t", "Setting telegram_bot.bot_token", "")
        line = draw(pane, screen)[HEIGHT - 3]
        assert line == PROMPT + "Setting telegram_bot.bot_token:"

    def test_re_asking_with_the_same_token_updates_the_open_prompt_in_place(self, pane, screen):
        answers = []
        pane.bind(on_secret=lambda token, value: answers.append(value))
        entry = pane.add_secret_prompt("t", "Visa (card)", "")
        for ch in "1234":
            pane.handle_input(ch)
        pane.handle_input('enter')
        again = pane.add_secret_prompt("t", "Visa (card)", "⚠ card number must be 13-19 digits — try again (2/3)")
        assert again is entry and pane.pending_secret is entry
        assert [e.kind for e in pane.entries].count(KIND_SECRET) == 1
        lines = draw(pane, screen)
        assert lines[0] == BULLET + "Visa (card)"
        assert lines[1] == "  card number must be 13-19 digits — try again (2/3)"
        assert lines[HEIGHT - 3] == PROMPT + "Visa (card):"    # nothing left over from the first try
        for ch in "4111 1111 1111 1111":
            pane.handle_input(ch)
        pane.handle_input('enter')
        assert answers == ["1234", "4111 1111 1111 1111"]

    def test_typing_goes_to_the_secret_field_even_with_an_approval_pending(self, pane):
        approvals, secrets = [], []
        pane.bind(on_approval=lambda t, d: approvals.append(d), on_secret=lambda t, v: secrets.append(v))
        pane.add_approval("a", "Send it", "")
        pane.add_secret_prompt("s", "Password", "")
        pane.handle_input('y')
        pane.handle_input('n')
        pane.handle_input('enter')
        assert approvals == [] and secrets == ["yn"]

    def test_closing_the_focused_prompt_moves_typing_to_the_other_open_one(self, pane):
        secrets = []
        pane.bind(on_secret=lambda t, v: secrets.append((t, v)))
        first = pane.add_secret_prompt("t1", "A", "")
        second = pane.add_secret_prompt("t2", "B", "")
        assert pane.pending_secret is first
        pane.handle_input('x')
        pane.handle_input('enter')
        assert secrets == [("t1", "x")]
        pane.close_secret(first, "✅")
        assert pane.pending_secret is second
        pane.handle_input('y')
        pane.handle_input('enter')
        assert secrets[-1] == ("t2", "y")
        pane.close_secret(second, "✅")
        assert pane.pending_secret is None

    def test_a_long_value_scrolls_so_the_cursor_stays_in_view(self, pane, screen):
        pane.add_secret_prompt("t", "Key", "")
        for _ in range(80):
            pane.handle_input('a')
        row = draw(pane, screen)[HEIGHT - 3]
        assert row.startswith(PROMPT + "Key: •") and len(row) <= WIDTH and "a" not in row

    def test_arrows_still_scroll_while_typing_a_secret(self, pane, screen):
        for i in range(40):
            pane.add_notice(f"n{i}")
        pane.add_secret_prompt("t", "Q", "")
        draw(pane, screen)
        assert pane.handle_input('up')
        assert pane.scroll_offset > 0

    @pytest.mark.parametrize("outcome, shown", [
        ("✅ Stored as amex (•••• 1111)", "Stored as amex (•••• 1111)"),
        ("⚠ too short to vault safely (min 6) — giving up", "too short to vault safely (min 6) — giving up"),
        ("✖ Cancelled by console", "Cancelled by console"),
        ("⏱ Timed out", "Timed out"),
        ("no longer pending", "no longer pending"),
    ])
    def test_outcome_markers_stay_out_of_the_console(self, pane, outcome, shown):
        entry = pane.add_secret_prompt("t", "Q", "")
        pane.close_secret(entry, outcome)
        assert entry.body.text == f"Q — {shown}"

    def test_clear_leaves_secret_mode(self, pane):
        pane.add_secret_prompt("t", "Q", "")
        pane.clear()
        assert pane.pending_secret is None
        assert pane.handle_input('x') is True and not pane.editor.empty


class TestSecretMasking:
    """Vaulted values are masked the way the Vault tab lists them, in every entry kind, and
    the masks are marked orange — the Memory tab's treatment, reused."""
    PHONE = "+46701740605"

    @pytest.fixture
    def stocked(self, tmp_path):
        from common.vault.Vault import Vault
        vault = Vault(tmp_path / "vault.bin", "test")
        vault.add("phone", "Marcus's mobile", self.PHONE, id="marcus-mobile")
        return vault

    @pytest.fixture
    def pane(self, stocked):
        from common.vault.Redactor import Redactor
        return ChatPane(redactor=Redactor(stocked))

    def test_user_and_assistant_text_is_masked_and_marked(self, pane, screen):
        from interfaces.console.Theme import COLOR_SECRET
        pane.add_user("call 070-174 06 05 please")
        pane.add_assistant("Calling <b>+46 70 174 06 05</b> now", final=True)
        lines = draw(pane, screen)
        assert lines[0] == PROMPT + "call •••• 0605 please"
        assert lines[2] == BULLET + "Calling •••• 0605 now"
        assert "0605" in lines[0] and "174" not in "\n".join(lines)
        cell = screen.next_frame[2][len(BULLET) + len("Calling ")]
        assert cell.char == "•" and cell.style.fg == COLOR_SECRET and cell.style.bold

    def test_history_and_streaming_entries_are_masked_too(self, pane, screen):
        pane.insert(KIND_SYSTEM, "Charlotte → Marcus: ring me on 0701740605", 1000.0, header="iMessage")
        entry = pane.add_assistant("Your number is +467017")
        pane.update_entry(entry, "Your number is +46701740605, right?")
        lines = draw(pane, screen)
        assert "ring me on •••• 0605" in lines[1]
        assert any(line.endswith("Your number is •••• 0605, right?") for line in lines)

    def test_a_secret_added_while_the_pane_is_open_is_masked_on_the_next_frame(self, pane, screen, stocked):
        pane.add_user("mail m@mlab.test")
        assert draw(pane, screen)[0] == PROMPT + "mail m@mlab.test"
        stocked.add("email", "Work e-mail", "m@mlab.test")
        assert draw(pane, screen)[0] == PROMPT + "mail m•••@mlab.test"

    def test_a_frame_with_nothing_new_masks_nothing_again(self, pane, screen, monkeypatch):
        """The vault's pattern over a whole transcript costs more than a frame's slot; run
        every frame it keeps the animation loop from ever sleeping and starves the server.
        Only the entry that changed is masked again."""
        pane.add_user("call 070-174 06 05 please")
        pane.insert(KIND_SYSTEM, "Charlotte → Marcus: ring me on 0701740605", 1000.0, header="iMessage")
        entry = pane.add_assistant("Your number is +467017")
        first = draw(pane, screen)
        masked = []
        real_mask = pane.redactor.mask
        monkeypatch.setattr(pane.redactor, "mask", lambda text: masked.append(text) or real_mask(text))

        assert draw(pane, screen) == first
        assert masked == []

        pane.update_entry(entry, "Your number is +46701740605, right?")
        assert any(line.endswith("Your number is •••• 0605, right?") for line in draw(pane, screen))
        assert masked == ["Your number is +46701740605, right?"]

    def test_an_empty_vault_masks_nothing(self, tmp_path, screen):
        from common.vault.Redactor import Redactor
        from common.vault.Vault import Vault
        pane = ChatPane(redactor=Redactor(Vault(tmp_path / "vault.bin", "test")))
        pane.add_user("call 070-174 06 05")
        assert draw(pane, screen)[0] == PROMPT + "call 070-174 06 05"

    def test_constructing_the_pane_does_not_resolve_the_vault(self, monkeypatch):
        from common.vault.Vault import Vault
        monkeypatch.setattr(Vault, "get_instance", lambda: pytest.fail("vault resolved at construction"))
        ChatPane().add_user("hi")


class TestWideGlyphs:
    """Emoji take two cells: the transcript wraps by cells, the cursors sit after the glyph."""

    def test_a_reply_with_emoji_wraps_by_cells(self, pane, screen):
        pane.add_assistant("\U0001F600" * 30, final=True)   # 60 cells of glyphs on a 56-cell line
        lines = draw(pane, screen)
        glyph_rows = [l for l in lines if "\U0001F600" in l]
        assert len(glyph_rows) == 2
        assert glyph_rows[0].count("\U0001F600") == 28
        assert glyph_rows[1].count("\U0001F600") == 2

    def test_a_reply_with_emoji_never_spills_onto_the_next_row(self, pane, screen):
        pane.add_assistant("x " * 20 + "\U0001F600 tail", final=True)
        lines = draw(pane, screen)
        assert not any(l.startswith("tail") for l in lines)

    def test_a_selector_sequence_is_normalized_in_the_transcript(self, pane, screen):
        pane.add_assistant("☀️ sun", final=True)
        assert any("☀ sun" in l for l in draw(pane, screen))

    def test_the_composer_cursor_sits_after_a_typed_emoji(self, pane, screen):
        pane.editor.insert("a\U0001F600")
        draw(pane, screen)
        cursor_cell = screen.next_frame[HEIGHT - 3][len(PROMPT) + 3]
        assert cursor_cell.style.bg == ChatPane.COLOR_INPUT
        assert screen.next_frame[HEIGHT - 3][len(PROMPT) + 1].char == "\U0001F600"

    def test_the_composer_wraps_a_row_that_a_glyph_would_overflow(self, pane, screen):
        text_w = WIDTH - len(PROMPT) - 1
        pane.editor.insert("x" * (text_w - 1) + "\U0001F600")
        rows_ = pane._input_rows(text_w)
        assert rows_ == ["x" * (text_w - 1), "\U0001F600"]

    def test_typing_a_selector_sequence_stores_the_normalized_text(self, pane):
        pane.editor.insert("☀️")
        assert pane.editor.text == "☀" and pane.editor.cursor == 1


class TestTables:
    TABLE = "Totals:\n\n| Scope | Cost |\n|---|---|\n| 3 years | ~$30 |\n| 5 years | ~$50 |\n\nQuite cheap."

    def test_a_pipe_table_in_a_reply_is_drawn_as_aligned_columns(self, pane, screen):
        pane.add_assistant(self.TABLE, final=True)
        draw(pane, screen)
        shown = rows(screen)
        assert shown[0] == BULLET + "Totals:"
        assert shown[2] == "  Scope   │ Cost"
        assert shown[3] == "  ────────┼─────"
        assert shown[4] == "  3 years │ ~$30"
        assert shown[5] == "  5 years │ ~$50"
        assert shown[7] == "  Quite cheap."

    def test_a_reply_that_opens_with_a_table_carries_the_bullet_on_its_first_row(self, pane, screen):
        pane.add_assistant("| a | b |\n|---|---|\n| 1 | 2 |", final=True)
        draw(pane, screen)
        shown = rows(screen)
        assert shown[0] == BULLET + "a │ b"
        assert shown[2] == "  1 │ 2"

    def test_a_history_entry_keeps_its_header_above_the_table(self, pane, screen):
        pane.insert(KIND_SYSTEM, "| a | b |\n| 1 | 2 |", 1000.0, header="System (Telegram)")
        draw(pane, screen)
        shown = rows(screen)
        assert shown[0].startswith(BULLET + "System (Telegram)")
        assert shown[1] == "  a │ b"
        assert shown[2] == "  1 │ 2"

    def test_a_table_wider_than_the_pane_wraps_inside_its_columns(self, pane, screen):
        cell = "a note long enough that it cannot possibly fit on one line of a sixty column pane"
        pane.add_assistant(f"| Scope | Note |\n|---|---|\n| 3 years | {cell} |", final=True)
        draw(pane, screen)
        shown = rows(screen)
        assert all(len(r) <= WIDTH - 1 for r in shown[:4])   # up to the scrollbar column
        assert shown[2].startswith("  3 years │ a note")
        assert shown[3].startswith("          │ ")

    def test_a_streaming_row_without_its_closing_pipe_is_already_a_table_row(self, pane, screen):
        entry = pane.add_assistant("| a | b |\n|---|---|\n| 1 | 2")
        entry.body.reveal_all()
        draw(pane, screen)
        assert rows(screen)[2] == "  1 │ 2"


class TestHeaderRows:
    """A header (the Home logo) takes rows off the top of the transcript and scrolls away
    with it once the lines fill the rows under it; scrolling back brings it down."""

    def _pane(self, header_rows):
        pane = ChatPane()
        pane.header_rows = header_rows
        return pane

    def _draw(self, pane):
        screen = TUIScreen(80, 30)
        pane.draw(screen, 0, 1, 1, 78, 30)
        return screen

    def test_the_header_lifts_a_row_per_line_past_the_rows_under_it_and_no_further(self):
        pane = self._pane(5)
        self._draw(pane)
        under = pane._min_transcript_rows              # transcript rows with the header fully shown
        assert pane.header_scroll == 0 and under > 5
        for i in range(under // 2):                    # a line and a blank each, the last blank dropped
            pane.add_notice(f"n{i}")
        screen = self._draw(pane)
        assert pane._total_lines == under - 1 and pane.header_scroll == 0
        assert "n0" in row(screen, 1 + 5), "the first line sits right under the header"
        pane.add_notice("more")
        screen = self._draw(pane)
        assert pane._total_lines == under + 1 and pane.header_scroll == 1
        assert "n0" in row(screen, 1 + 4), "the header gave up a row and the first line moved up with it"
        for i in range(10):
            pane.add_notice(f"x{i}")
        screen = self._draw(pane)
        assert pane.header_scroll == 5, "gone; from here the transcript scrolls on its own"
        assert pane.scroll_offset == 0 and "x9" in row(screen, 1 + pane._last_transcript_rows - 1)

    def test_scrolling_back_through_the_history_brings_the_header_down_again(self):
        pane = self._pane(5)
        self._draw(pane)
        for i in range(40):
            pane.add_notice(f"n{i}")
        self._draw(pane)
        assert pane.header_scroll == 5
        pane.scroll_offset = 3
        self._draw(pane)
        assert pane.header_scroll == 5, "three lines back: the header is still off the top"
        pane.scroll_offset = pane._total_lines - pane._min_transcript_rows - 2
        screen = self._draw(pane)
        assert pane.header_scroll == 2 and "n0" in row(screen, 1 + 3), "two lines short of the top: two header rows still lifted"
        for _ in range(10):
            pane._scroll("up")                         # the keys stop where the header is fully back
        screen = self._draw(pane)
        assert pane.header_scroll == 0, "at the top of the history the header is fully back"
        assert "n0" in row(screen, 1 + 5)
        assert pane.scroll_offset == pane._total_lines - pane._min_transcript_rows
        pane.scroll_offset = 10 ** 6
        self._draw(pane)
        assert pane.scroll_offset == pane._total_lines - pane._min_transcript_rows and pane.header_scroll == 0

    def test_a_pane_too_short_for_the_header_keeps_the_minimum_transcript(self):
        pane = self._pane(50)
        pane.add_notice("hello")
        screen = TUIScreen(80, 12)
        pane.draw(screen, 0, 1, 1, 78, 12)
        assert pane._last_transcript_rows >= ChatPane.MIN_TRANSCRIPT_ROWS
        assert pane.header_scroll == 0
