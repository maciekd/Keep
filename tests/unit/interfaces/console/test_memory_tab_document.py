"""The Bank tab as a document in tiles: summary tiles (the last curation, the notes taken since,
size, sections), an outline beside the text on a wide screen, scrolling and section jumps that
stay inside the document, and a snapshot that opens on its first change."""
from datetime import datetime

import pytest

from assistant import MemoryFile
from common.vault.Redactor import Redactor
from common.vault.Vault import Vault
from interfaces.console.Theme import COLOR_SELECTED_BG
from interfaces.console.framework.TUIScreen import TUIScreen
from interfaces.console.tabs.BankTab import BankTab
from interfaces.console.tabs.TabView import SUB_CONTENT_TOP

WIDE, NARROW, HEIGHT = 140, 100, 40
BANK = "# The Family\nMarcus likes tea.\n\n# Rules\n- Lights off at midnight\n"


@pytest.fixture
def tab(tmp_path):
    t = BankTab(Redactor(Vault(tmp_path / "vault.bin", "test")))
    t.memory_file_path = str(tmp_path / "memory" / "bank.md")
    t.log_dir = tmp_path / "log" / "memory"
    t.log_dir.mkdir(parents=True)
    MemoryFile.write(BANK, t.memory_file_path)
    return t


def draw(tab: BankTab, width: int = WIDE, height: int = HEIGHT) -> TUIScreen:
    screen = TUIScreen(width, height)
    tab.draw(screen, 0)
    return screen


def rows(screen: TUIScreen) -> list[str]:
    """The content rows as they read, without the scrollbar at a tile's right edge."""
    return [screen.row_text(1, y, screen.width).rstrip(" │┃") for y in range(SUB_CONTENT_TOP, screen.height)]


def text_of(screen: TUIScreen) -> str:
    return "\n".join(rows(screen))


def row_with(screen: TUIScreen, *needles: str) -> str:
    return next(row for row in rows(screen) if all(needle in row for needle in needles))


def snapshot(tab: BankTab, stamp: str, text: str):
    (tab.log_dir / f"{stamp}_memory.md").write_text("## Updated Memory\n" + text)


def numbered(count: int, changed: dict = None) -> str:
    return "".join((changed or {}).get(i, f"fact number {i}") + "\n" for i in range(1, count + 1))


@pytest.mark.unit
class TestSummaryTiles:
    def test_the_last_curation_is_read_from_the_newest_snapshots_name(self, tab, monkeypatch):
        snapshot(tab, "20260101_031000", BANK)
        snapshot(tab, "20260102_031500", BANK)
        monkeypatch.setattr(tab, "now", lambda: datetime(2026, 1, 2, 9, 15))
        screen = draw(tab, width=200)
        assert "6h ago" in row_with(screen, "Last curation")
        assert "03:15" in text_of(screen) and "2 nights" in text_of(screen)

    def test_a_bank_never_curated_says_so(self, tab):
        assert "never" in row_with(draw(tab), "Last curation")

    def test_the_size_tile_measures_the_text_on_show(self, tab):
        MemoryFile.write("word " * 1024, tab.memory_file_path)      # 5 KiB of five-character words
        screen = draw(tab)
        assert "5.0 KB" in row_with(screen, "Size")
        assert "1024 words" in text_of(screen)

    def test_the_sections_tile_counts_the_sections_not_every_heading(self, tab):
        MemoryFile.write("# A\n## A1\n- one\n- two\n# B\n1. three\n", tab.memory_file_path)
        screen = draw(tab)
        assert row_with(screen, "Sections").split("Sections")[1].split()[0] == "2"
        assert "3 headings" in text_of(screen) and "3 entries" in text_of(screen)

    def test_a_narrow_screen_keeps_the_first_tiles_only(self, tab):
        text = text_of(draw(tab, width=70))
        assert "Last curation" in text and "New notes" in text
        assert "Size" not in text and "Sections" not in text
        assert "Marcus likes tea." in text

    def test_a_short_screen_gives_the_tiles_rows_to_the_document(self, tab):
        assert "Last curation" in text_of(draw(tab, height=24))
        text = text_of(draw(tab, height=18))
        assert "Last curation" not in text and "Size" not in text
        assert "Marcus likes tea." in text and "Lights off at midnight" in text


@pytest.mark.unit
class TestNotesSinceTheCuration:
    def test_notes_appended_after_the_curation_sit_under_a_band_of_their_own(self, tab):
        MemoryFile.append("Frances is allergic to cats.", tab.memory_file_path)
        MemoryFile.append("Dave borrows the trailer on Fridays.", tab.memory_file_path)
        screen = draw(tab)
        lines = rows(screen)
        band = next(k for k, row in enumerate(lines) if "2 notes since" in row)
        curated = next(k for k, row in enumerate(lines) if "Lights off at midnight" in row)
        note = next(k for k, row in enumerate(lines) if "Frances is allergic to cats." in row)
        assert curated < band < note
        assert "2 notes" in row_with(screen, "Last curation")       # the tile row carries the count too

    def test_a_note_of_several_lines_is_one_note(self, tab):
        MemoryFile.append("Frances is allergic to cats.\nShe told us in March.", tab.memory_file_path)
        text = text_of(draw(tab))
        assert "1 note since" in text and "She told us in March." in text

    def test_a_freshly_curated_bank_has_no_band(self, tab):
        text = text_of(draw(tab))
        assert "since the last curation" not in text
        assert "none" in row_with(draw(tab), "New notes")

    def test_the_band_goes_once_the_curation_absorbed_the_notes(self, tab):
        MemoryFile.append("Frances is allergic to cats.", tab.memory_file_path)
        assert "1 note since" in text_of(draw(tab))
        MemoryFile.write(BANK + "- Frances is allergic to cats\n", tab.memory_file_path)
        text = text_of(draw(tab))
        assert "since the last curation" not in text and "Frances is allergic to cats" in text

    def test_a_bank_without_a_mark_is_shown_whole_with_no_band(self, tab, tmp_path):
        path = tmp_path / "memory" / "by_hand.md"
        path.write_text(BANK + "\nCarol waters the plants.\n")
        tab.memory_file_path = str(path)
        text = text_of(draw(tab))
        assert "Carol waters the plants." in text and "since the last curation" not in text

    def test_a_vaulted_value_in_a_note_is_masked_too(self, tab):
        tab.redactor.vault.add("phone", "Marcus's mobile", "+46701740605", id="marcus-mobile")
        MemoryFile.append("Ring Marcus on 070-174 06 05.", tab.memory_file_path)
        text = text_of(draw(tab))
        assert "•••• 0605" in text and "174 06 05" not in text


@pytest.mark.unit
class TestOutline:
    def test_a_wide_screen_puts_the_sections_beside_the_text(self, tab):
        screen = draw(tab, width=WIDE)
        assert "Outline" in text_of(screen)
        assert row_with(screen, "● The Family") and row_with(screen, "● Rules")

    def test_a_narrower_screen_has_the_text_alone(self, tab):
        text = text_of(draw(tab, width=NARROW))
        assert "Outline" not in text and "●" not in text
        assert "Marcus likes tea." in text

    def test_a_document_with_one_heading_needs_no_outline(self, tab):
        MemoryFile.write("# The Family\nMarcus likes tea.\n", tab.memory_file_path)
        text = text_of(draw(tab, width=WIDE))
        assert "Outline" not in text and "Marcus likes tea." in text

    def test_the_section_being_read_is_the_highlighted_one(self, tab):
        MemoryFile.write("# First\n" + numbered(80) + "# Second\n" + numbered(80), tab.memory_file_path)

        def highlighted(screen):
            return [screen.row_text(1, y, 30).strip() for y in range(SUB_CONTENT_TOP, HEIGHT)
                    if screen.next_frame[y - 1][5].style.bg == COLOR_SELECTED_BG]

        assert [row[:7] for row in highlighted(draw(tab))] == ["● First"]
        tab.handle_input("n")
        assert [row[:8] for row in highlighted(draw(tab))] == ["● Second"]

    def test_a_snapshot_counts_its_changes_per_section(self, tab):
        snapshot(tab, "20260101_031000", BANK)
        snapshot(tab, "20260102_031000", BANK + "- No announcements while Ingeborg sleeps\n")
        tab.handle_input("left")
        screen = draw(tab)
        assert "+1" in row_with(screen, "● Rules")
        family = row_with(screen, "● The Family").split("The Family")[1][:20]
        assert "+" not in family and "−" not in family
        assert "+1 −0" in row_with(screen, "Changes")


@pytest.mark.unit
class TestScrolling:
    COUNT = 120

    @pytest.fixture(autouse=True)
    def long_bank(self, tab):
        MemoryFile.write("# First\n" + numbered(self.COUNT) + "# Second\nthe last line\n", tab.memory_file_path)

    def facts(self, tab) -> list[int]:
        """The numbers of the facts on screen."""
        return [int(row.split("fact number ")[1].split()[0]) for row in rows(draw(tab, width=NARROW))
                if "fact number " in row]

    def test_the_document_opens_at_its_top(self, tab):
        assert self.facts(tab)[0] == 1

    def test_scrolling_up_from_the_top_stays_at_the_top(self, tab):
        for key in ("up", "pageup", "p", "home"):
            tab.handle_input(key)
        assert self.facts(tab)[0] == 1

    def test_down_moves_a_row_and_page_down_a_screenful(self, tab):
        shown = len(self.facts(tab))
        for _ in range(3):               # the heading and its rule scroll off first
            tab.handle_input("down")
        first = self.facts(tab)[0]
        assert first == 2
        tab.handle_input("pagedown")
        assert shown <= self.facts(tab)[0] - first <= shown + 2
        tab.handle_input("pageup")
        assert self.facts(tab)[0] == first

    def test_end_shows_the_last_line_and_going_further_changes_nothing(self, tab):
        draw(tab, width=NARROW)
        tab.handle_input("end")
        screen = draw(tab, width=NARROW)
        assert "the last line" in text_of(screen) and len(self.facts(tab)) > 10    # a full screen, not the tail alone
        at_end = text_of(screen)
        for key in ("down", "pagedown", "n", "end"):
            tab.handle_input(key)
        assert text_of(draw(tab, width=NARROW)) == at_end
        tab.handle_input("home")
        assert self.facts(tab)[0] == 1

    def test_end_before_the_first_draw_lands_on_the_last_page(self, tab):
        """End used to take the row count of the last drawn frame; before any frame there was none
        and the key did nothing."""
        tab.handle_input("end")
        assert "the last line" in text_of(draw(tab, width=NARROW))

    def test_end_reaches_the_end_of_a_document_that_grew_since_the_last_frame(self, tab):
        draw(tab, width=NARROW)
        MemoryFile.write("# First\n" + numbered(3 * self.COUNT) + "# Second\nthe last line\n", tab.memory_file_path)
        tab.handle_input("end")
        assert "the last line" in text_of(draw(tab, width=NARROW))

    def test_one_step_back_from_far_past_the_end_is_one_step(self, tab):
        draw(tab, width=NARROW)
        tab.handle_input("end")
        tab.handle_input("pagedown")
        first = self.facts(tab)[0]
        tab.handle_input("up")
        assert self.facts(tab)[0] == first - 1

    def test_n_and_p_jump_between_the_sections(self, tab):
        MemoryFile.write("# First\n" + numbered(60) + "# Second\n" + numbered(60) + "# Third\n" + numbered(60),
                         tab.memory_file_path)
        draw(tab, width=NARROW)
        tab.handle_input("n")
        screen = draw(tab, width=NARROW)
        document = [row.strip() for row in rows(screen) if row.strip() and "Memory bank" not in row]
        heading = document.index("Second")
        assert document[heading + 2] == "fact number 1"
        tab.handle_input("p")
        assert self.facts(tab)[0] == 1 and "First" in text_of(draw(tab, width=NARROW))

    def test_the_document_tile_names_the_section_being_read(self, tab):
        assert "First" in row_with(draw(tab, width=NARROW), "Memory bank")
        tab.handle_input("end")
        draw(tab, width=NARROW)
        assert "Second" in row_with(draw(tab, width=NARROW), "Memory bank")

    def test_a_shrunk_document_pulls_the_view_back_inside(self, tab):
        draw(tab, width=NARROW)
        tab.handle_input("end")
        draw(tab, width=NARROW)
        MemoryFile.write("# First\nonly this\n", tab.memory_file_path)
        assert "only this" in text_of(draw(tab, width=NARROW))


@pytest.mark.unit
class TestSnapshots:
    def test_a_snapshot_opens_on_its_first_change(self, tab):
        snapshot(tab, "20260101_031000", numbered(150))
        snapshot(tab, "20260102_031000", numbered(150, {100: "Charlotte plays the cello"}))
        tab.handle_input("left")
        screen = draw(tab)
        text = text_of(screen)
        assert "+ Charlotte plays the cello" in text and "- fact number 100" in text
        assert "fact number 1\n" not in text + "\n"
        assert "fact number 99" in text                   # a little context above the change

    def test_the_document_tile_carries_the_snapshots_date(self, tab):
        snapshot(tab, "20260102_031000", BANK)
        tab.handle_input("left")
        text = text_of(draw(tab))
        assert "January 2, 2026" in text and "1 of 1" in text

    def test_the_oldest_snapshot_has_nothing_to_compare_with(self, tab):
        snapshot(tab, "20260102_031000", BANK)
        tab.handle_input("left")
        screen = draw(tab)
        assert "+" not in row_with(screen, "Changes")
        assert "Lights off at midnight" in text_of(screen)

    def test_a_removed_heading_is_shown_without_its_hashes(self, tab):
        snapshot(tab, "20260101_031000", "# Old plans\nMarcus likes tea.\n")
        snapshot(tab, "20260102_031000", "Marcus likes tea.\n")
        tab.handle_input("left")
        text = text_of(draw(tab))
        assert "- Old plans" in text and "#" not in text

    def test_back_to_live_starts_at_the_top_again(self, tab):
        MemoryFile.write(numbered(150), tab.memory_file_path)
        snapshot(tab, "20260101_031000", numbered(150))
        snapshot(tab, "20260102_031000", numbered(150, {100: "Charlotte plays the cello"}))
        tab.handle_input("left")
        draw(tab)
        tab.handle_input("right")
        text = text_of(draw(tab))
        assert "fact number 1\n" in text and "Charlotte" not in text
