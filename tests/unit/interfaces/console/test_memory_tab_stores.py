"""MemoryTab's live view is the whole memory: the bank, then the watch list and the procedures."""
import pytest

from assistant import Procedures, WatchList
from common.vault.Redactor import Redactor
from common.vault.Vault import Vault
from interfaces.console.framework.TUIScreen import TUIScreen
from interfaces.console.tabs.MemoryTab import MemoryTab
from interfaces.console.tabs.TabView import CONTENT_TOP

WIDTH, HEIGHT = 120, 40
PHONE = "+46701740605"


@pytest.fixture(autouse=True)
def stores(tmp_path, monkeypatch):
    monkeypatch.setattr(WatchList, "WATCHLIST_DIR", tmp_path / "memory" / "watchlist")
    monkeypatch.setattr(WatchList, "LOG_DIR", tmp_path / "log" / "memory" / "watchlist")
    monkeypatch.setattr(Procedures, "PROCEDURES_DIR", tmp_path / "memory" / "procedures")
    monkeypatch.setattr(Procedures, "LOG_DIR", tmp_path / "log" / "memory" / "procedures")


@pytest.fixture
def tab(tmp_path):
    vault = Vault(tmp_path / "vault.bin", "test")
    vault.add("phone", "Marcus's mobile", PHONE, id="marcus-mobile")
    t = MemoryTab(Redactor(vault))
    (tmp_path / "memory").mkdir()
    t.memory_file_path = str(tmp_path / "memory" / "bank.md")
    t.log_dir = tmp_path / "log" / "memory"
    with open(t.memory_file_path, "w", encoding="utf-8") as f:
        f.write("# The Family\nMarcus likes tea.\n")
    return t


def rows(tab: MemoryTab) -> list[str]:
    screen = TUIScreen(WIDTH, HEIGHT)
    tab.draw(screen, 0)
    return [screen.row_text(1, y, WIDTH).rstrip() for y in range(CONTENT_TOP, HEIGHT)]


def index_of(lines: list[str], needle: str) -> int:
    return next(i for i, line in enumerate(lines) if needle in line)


@pytest.mark.unit
class TestLiveViewShowsAllThreeParts:
    def test_watch_items_and_procedures_follow_the_bank(self, tab):
        WatchList.save("Parcel from shop.example", "order 4711", "2026-03-12")
        Procedures.save("shop-checkout", "paying at shop.example", "open the cart\nclick Pay")
        lines = rows(tab)
        bank, item, step = (index_of(lines, n) for n in ("Marcus likes tea.", "parcel-from-shop-example", "click Pay"))
        assert bank < item < step
        assert index_of(lines, "order 4711") > item
        assert index_of(lines, "paying at shop.example") < step

    def test_empty_stores_are_shown_as_empty_not_left_out(self, tab):
        text = "\n".join(rows(tab))
        assert WatchList.EMPTY in text and Procedures.EMPTY in text

    def test_a_new_watch_item_appears_on_the_next_draw(self, tab):
        assert not any("dentist" in line for line in rows(tab))
        WatchList.save("Dentist for Charlotte", "12:30")
        assert any("dentist-for-charlotte" in line for line in rows(tab))

    def test_a_vaulted_value_in_a_watch_note_or_a_procedure_is_masked_too(self, tab):
        WatchList.save("Call the courier", f"ring {PHONE} before noon")
        Procedures.save("courier", "rebooking a delivery", f"call {PHONE} and give the order number")
        text = "\n".join(rows(tab))
        assert PHONE not in text and "0701740605" not in text
        assert text.count("•••• 0605") == 2

    def test_a_missing_bank_still_shows_the_other_parts(self, tab, tmp_path):
        tab.memory_file_path = str(tmp_path / "memory" / "absent.md")
        WatchList.save("Parcel", "ordered")
        assert any("parcel" in line for line in rows(tab))

    def test_a_procedures_own_headings_sit_below_its_name(self, tab):
        """The curator's "## Notes" must not read as a procedure of its own called Notes."""
        Procedures.save("shop-checkout", "paying at shop.example",
                        "open the cart\n\n## Card declined\nuse the other card\n\n### 3-D Secure\napprove in the app")
        Procedures.note("shop-checkout", "the Pay button moved")
        Procedures.save("courier", "rebooking a delivery", "call the courier")
        text = MemoryTab._watchlist_and_procedures()
        headings = [line for line in text.splitlines() if line.startswith("#")]
        assert headings[headings.index("# Procedures") + 1:] == [
            "## courier", "## shop-checkout", "### Card declined", "#### 3-D Secure", "### Notes"]
        assert "the Pay button moved" in text and "use the other card" in text
