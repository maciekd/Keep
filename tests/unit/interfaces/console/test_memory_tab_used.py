"""The Procedures and Memos tabs say when the LLM last loaded each file: a Used column in the
outline, and the Last used and Unused tiles."""
import json
import re
from datetime import datetime, timedelta

import pytest

from assistant.Memos import Memos
from assistant.Procedures import Procedures
from common.vault.Redactor import Redactor
from common.vault.Vault import Vault
from interfaces.console.framework.TUIScreen import TUIScreen
from interfaces.console.tabs.MemosTab import MemosTab
from interfaces.console.tabs.ProceduresTab import IDLE_DAYS, ProceduresTab, age
from interfaces.console.tabs.TabView import SUB_CONTENT_TOP

WIDTH, HEIGHT = 160, 30
NOW = datetime(2026, 3, 12, 14, 0)


@pytest.fixture(autouse=True)
def stores(tmp_path, monkeypatch):
    for cls, folder in ((Procedures, "procedures"), (Memos, "memos")):
        monkeypatch.setattr(cls, "DIR", tmp_path / folder)
        monkeypatch.setattr(cls, "LOG_DIR", tmp_path / "log" / folder)
        monkeypatch.setattr(cls, "USED_FILE", tmp_path / "cache" / f"{folder}_used.json")


def make(cls, tmp_path):
    tab = cls(Redactor(Vault(tmp_path / "vault.bin", "test")))
    tab.now = lambda: NOW
    return tab


def used(cls, **ago: timedelta):
    cls.USED_FILE.parent.mkdir(parents=True, exist_ok=True)
    cls.USED_FILE.write_text(json.dumps({name.replace("_", "-"): (NOW - delta).isoformat() for name, delta in ago.items()}))


def rows(tab) -> list[str]:
    screen = TUIScreen(WIDTH, HEIGHT)
    tab.draw(screen, 0)
    return [screen.row_text(0, y, WIDTH).rstrip() for y in range(SUB_CONTENT_TOP, HEIGHT)]


def used_cell(tab, name: str) -> str:
    """The Used cell of `name`'s outline row: what follows its size."""
    row = next(row for row in rows(tab) if f"● {name} " in row)
    return re.search(rf"● {re.escape(name)} .*?\d+ +(\S+)", row).group(1)


def tile(tab, label: str) -> tuple:
    return next(t for t in tab._live_tiles() if t[0] == label)


@pytest.mark.unit
class TestTheUsedColumn:
    def test_each_procedure_shows_how_long_ago_it_was_loaded_or_never(self, tmp_path):
        for name in ("courier", "printer", "shop"):
            Procedures.save(name, "a when", "step one\nstep two")
        used(Procedures, courier=timedelta(days=3, hours=2), shop=timedelta(minutes=5))
        tab = make(ProceduresTab, tmp_path)
        assert used_cell(tab, "courier") == "3d"
        assert used_cell(tab, "shop") == "5m"
        assert used_cell(tab, "printer") == "never"

    def test_the_outline_carries_column_titles(self, tmp_path):
        Procedures.save("courier", "a when", "step one")
        Procedures.save("shop", "a when", "step one")
        header = next(row for row in rows(make(ProceduresTab, tmp_path)) if "Name" in row)
        assert header.index("Name") < header.index("Size") < header.index("Used")

    def test_the_memos_tab_reads_the_memos_record(self, tmp_path):
        Memos.save("wine-list", "choosing wine", "Barolo")
        Memos.save("trip", "the trip", "flight at 07:10")
        used(Memos, trip=timedelta(hours=5))
        used(Procedures, wine_list=timedelta(hours=1))
        tab = make(MemosTab, tmp_path)
        assert used_cell(tab, "trip") == "5h"
        assert used_cell(tab, "wine-list") == "never"


@pytest.mark.unit
class TestTheTiles:
    def test_last_used_names_the_latest_load(self, tmp_path):
        for name in ("courier", "shop"):
            Procedures.save(name, "a when", "step one")
        used(Procedures, courier=timedelta(days=2), shop=timedelta(hours=3))
        tab = make(ProceduresTab, tmp_path)
        tab.draw(TUIScreen(WIDTH, HEIGHT), 0)
        _, value, _, segments = tile(tab, "Last used")
        assert value.startswith("3h") and segments[0][0] == "shop"

    def test_a_record_of_a_procedure_that_is_gone_is_not_counted(self, tmp_path):
        Procedures.save("shop", "a when", "step one")
        used(Procedures, removed_by_hand=timedelta(minutes=1))
        tab = make(ProceduresTab, tmp_path)
        tab.draw(TUIScreen(WIDTH, HEIGHT), 0)
        assert tile(tab, "Last used")[1] == "–"

    def test_unused_counts_the_never_loaded_and_the_idle(self, tmp_path):
        for name in ("courier", "printer", "shop"):
            Procedures.save(name, "a when", "step one")
        used(Procedures, courier=timedelta(days=IDLE_DAYS + 1), shop=timedelta(days=1))
        tab = make(ProceduresTab, tmp_path)
        tab.draw(TUIScreen(WIDTH, HEIGHT), 0)
        _, value, _, segments = tile(tab, "Unused")
        assert value == "2 of 3"
        assert any(text.startswith("1 never") for text, _ in segments)

    def test_unused_is_none_when_every_procedure_was_loaded_lately(self, tmp_path):
        Procedures.save("shop", "a when", "step one")
        used(Procedures, shop=timedelta(days=IDLE_DAYS - 1))
        tab = make(ProceduresTab, tmp_path)
        tab.draw(TUIScreen(WIDTH, HEIGHT), 0)
        assert tile(tab, "Unused")[1] == "none"


@pytest.mark.unit
@pytest.mark.parametrize("delta, shown", [
    (timedelta(seconds=20), "now"),
    (timedelta(minutes=59), "59m"),
    (timedelta(hours=23, minutes=59), "23h"),
    (timedelta(days=13), "13d"),
    (timedelta(days=14), "2w"),
    (timedelta(days=90), "3mo"),
    (timedelta(days=800), "2y"),
])
def test_an_age_fits_the_column_in_its_largest_unit(delta, shown):
    assert age(delta) == shown and len(shown) <= ProceduresTab.COLUMN_W
