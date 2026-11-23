"""The Memory group's three sub-tabs each show one part of the memory: the bank, the procedures,
the watch list. Every one masks a vaulted value; only the bank has a history to browse."""
import pytest

from assistant import WatchList

from assistant.Memos import Memos
from assistant.Procedures import Procedures
from common.vault.Redactor import Redactor
from common.vault.Vault import Vault
from interfaces.console.framework.TUIScreen import TUIScreen
from interfaces.console.tabs.BankTab import BankTab
from interfaces.console.tabs.MemoryTab import MemoryTab
from interfaces.console.tabs.MemosTab import MemosTab
from interfaces.console.tabs.ProceduresTab import ProceduresTab
from interfaces.console.tabs.TabView import SUB_CONTENT_TOP
from interfaces.console.tabs.WatchListTab import WatchListTab

WIDTH, HEIGHT = 120, 40
PHONE = "+46701740605"


@pytest.fixture(autouse=True)
def stores(tmp_path, monkeypatch):
    monkeypatch.setattr(WatchList, "WATCHLIST_DIR", tmp_path / "memory" / "watchlist")
    monkeypatch.setattr(WatchList, "LOG_DIR", tmp_path / "log" / "memory" / "watchlist")
    monkeypatch.setattr(Procedures, "DIR", tmp_path / "memory" / "procedures")
    monkeypatch.setattr(Procedures, "LOG_DIR", tmp_path / "log" / "memory" / "procedures")
    monkeypatch.setattr(Memos, "DIR", tmp_path / "memory" / "memos")
    monkeypatch.setattr(Memos, "LOG_DIR", tmp_path / "log" / "memory" / "memos")


@pytest.fixture
def redactor(tmp_path):
    vault = Vault(tmp_path / "vault.bin", "test")
    vault.add("phone", "Marcus's mobile", PHONE, id="marcus-mobile")
    return Redactor(vault)


def make(cls, redactor, tmp_path) -> MemoryTab:
    t = cls(redactor)
    t.log_dir = tmp_path / "log" / "memory"
    return t


@pytest.fixture
def bank(tmp_path, redactor):
    t = make(BankTab, redactor, tmp_path)
    (tmp_path / "memory").mkdir(exist_ok=True)
    t.memory_file_path = str(tmp_path / "memory" / "bank.md")
    with open(t.memory_file_path, "w", encoding="utf-8") as f:
        f.write("# The Family\nMarcus likes tea.\n")
    return t


@pytest.fixture
def procedures(tmp_path, redactor):
    return make(ProceduresTab, redactor, tmp_path)


@pytest.fixture
def watchlist(tmp_path, redactor):
    return make(WatchListTab, redactor, tmp_path)


def rows(tab: MemoryTab) -> list[str]:
    screen = TUIScreen(WIDTH, HEIGHT)
    tab.draw(screen, 0)
    return [screen.row_text(1, y, WIDTH).rstrip() for y in range(SUB_CONTENT_TOP, HEIGHT)]


def text_of(tab: MemoryTab) -> str:
    return "\n".join(rows(tab))


@pytest.mark.unit
class TestEachTabShowsItsOwnPart:
    def test_the_bank_tab_shows_the_bank_alone(self, bank):
        WatchList.save("Parcel from shop.example", "order 4711", "2026-03-12")
        Procedures.save("shop-checkout", "paying at shop.example", "open the cart\nclick Pay")
        text = text_of(bank)
        assert "Marcus likes tea." in text
        assert "order 4711" not in text and "click Pay" not in text

    def test_the_watch_list_tab_shows_the_items_alone(self, watchlist):
        WatchList.save("Parcel from shop.example", "order 4711", "2026-03-12")
        Procedures.save("shop-checkout", "paying at shop.example", "open the cart\nclick Pay")
        text = text_of(watchlist)
        assert "parcel-from-shop-example" in text and "order 4711" in text
        assert "click Pay" not in text

    def test_the_procedures_tab_shows_the_procedures_alone(self, procedures):
        WatchList.save("Parcel from shop.example", "order 4711", "2026-03-12")
        Procedures.save("shop-checkout", "paying at shop.example", "open the cart\nclick Pay")
        text = text_of(procedures)
        assert "paying at shop.example" in text and "click Pay" in text
        assert "order 4711" not in text

    def test_a_missing_bank_says_so_instead_of_failing(self, bank, tmp_path):
        bank.memory_file_path = str(tmp_path / "memory" / "absent.md")
        assert "absent.md" in bank._live_text()
        assert "Error" not in text_of(bank) and "Marcus likes tea." not in text_of(bank)

    def test_the_memos_tab_shows_the_memos_alone(self, tmp_path, redactor):
        Procedures.save("shop-checkout", "paying at shop.example", "open the cart\nclick Pay")
        memos = make(MemosTab, redactor, tmp_path)
        assert Memos.EMPTY in text_of(memos)
        Memos.save("wine-list", "choosing wine", "## Reds\nBarolo")
        text = memos._live_text()
        assert "## wine-list" in text and "### Reds" in text and "Barolo" in text
        assert "shop-checkout" not in text

    def test_empty_stores_are_shown_as_empty_not_blank(self, watchlist, procedures):
        assert WatchList.EMPTY in text_of(watchlist)
        assert Procedures.EMPTY in text_of(procedures)

    def test_a_new_watch_item_appears_on_the_next_draw(self, watchlist):
        assert "dentist" not in text_of(watchlist)
        WatchList.save("Dentist for Charlotte", "12:30")
        assert "dentist-for-charlotte" in text_of(watchlist)

    def test_a_new_procedure_appears_on_the_next_draw(self, procedures):
        assert "courier" not in text_of(procedures)
        Procedures.save("courier", "rebooking a delivery", "call the courier")
        assert "rebooking a delivery" in text_of(procedures)

    def test_a_vaulted_value_in_a_watch_note_is_masked(self, watchlist):
        WatchList.save("Call the courier", f"ring {PHONE} before noon")
        text = text_of(watchlist)
        assert PHONE not in text and "0701740605" not in text
        assert "•••• 0605" in text

    def test_a_vaulted_value_in_a_procedure_is_masked(self, procedures):
        Procedures.save("courier", "rebooking a delivery", f"call {PHONE} and give the order number")
        text = text_of(procedures)
        assert PHONE not in text and "0701740605" not in text
        assert "•••• 0605" in text

    def test_a_procedures_own_headings_sit_below_its_name(self, procedures):
        """The curator's "## Notes" must not read as a procedure of its own called Notes."""
        Procedures.save("shop-checkout", "paying at shop.example",
                        "open the cart\n\n## Card declined\nuse the other card\n\n### 3-D Secure\napprove in the app")
        Procedures.note("shop-checkout", "the Pay button moved")
        Procedures.save("courier", "rebooking a delivery", "call the courier")
        text = procedures._live_text()
        headings = [line for line in text.splitlines() if line.startswith("#")]
        assert headings == ["## courier", "## shop-checkout", "### Card declined", "#### 3-D Secure", "### notes"]
        assert "the Pay button moved" in text and "use the other card" in text


@pytest.mark.unit
class TestOnlyTheBankHasHistory:
    @pytest.fixture(autouse=True)
    def snapshot(self, tmp_path):
        log_dir = tmp_path / "log" / "memory"
        log_dir.mkdir(parents=True)
        (log_dir / "20260102_120000_memory.md").write_text("## Updated Memory\nMarcus liked coffee.\n")

    def test_left_opens_the_snapshot_on_the_bank(self, bank):
        assert bank.handle_input("left")
        assert "Marcus liked coffee." in text_of(bank)

    @pytest.mark.parametrize("cls", [ProceduresTab, WatchListTab])
    def test_left_and_right_are_left_alone_on_the_other_two(self, cls, redactor, tmp_path):
        tab = make(cls, redactor, tmp_path)
        before = text_of(tab)
        assert not tab.handle_input("left") and not tab.handle_input("right")
        assert "Marcus liked coffee." not in text_of(tab) and "history" not in text_of(tab)
        assert text_of(tab) == before
