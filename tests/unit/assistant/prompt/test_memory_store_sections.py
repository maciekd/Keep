"""The watch-list and procedures prompt sections render straight from the stores and re-render
when a store's version moves."""
from unittest.mock import patch

import pytest

from assistant import WatchList

from assistant.Procedures import Procedures
from assistant.prompt.sections.ProceduresSectionGenerator import ProceduresSectionGenerator
from assistant.prompt.sections.WatchListSectionGenerator import WatchListSectionGenerator


@pytest.fixture(autouse=True)
def stores(tmp_path, monkeypatch):
    monkeypatch.setattr(WatchList, "WATCHLIST_DIR", tmp_path / "watchlist")
    monkeypatch.setattr(WatchList, "LOG_DIR", tmp_path / "log" / "watchlist")
    monkeypatch.setattr(Procedures, "DIR", tmp_path / "procedures")
    monkeypatch.setattr(Procedures, "LOG_DIR", tmp_path / "log" / "procedures")


def one_tick(section):
    """One pass of the monitoring loop."""
    with patch.object(section, "_wait", side_effect=StopIteration):
        with pytest.raises(StopIteration):
            section._start_monitoring()


@pytest.mark.unit
class TestWatchListSection:
    def test_open_items_are_in_the_section_and_deleted_ones_are_not(self):
        section = WatchListSectionGenerator(None)
        assert section._fetch_content() == WatchList.EMPTY
        WatchList.save("Parcel from shop.example", "order 4711", "2026-03-12")
        WatchList.save("Dentist for Charlotte", "12:30")
        WatchList.delete("dentist-for-charlotte", "done")
        content = section._fetch_content()
        assert "parcel-from-shop-example" in content and "order 4711" in content
        assert "dentist" not in content

    def test_the_section_is_the_index_not_the_whole_notes(self):
        WatchList.save("Parcel", "order 4711\nleave it with Alice next door", "2026-03-12", every="6h")
        content = WatchListSectionGenerator(None)._fetch_content()
        assert "order 4711" in content and "Alice" not in content and "6h" not in content

    def test_the_loop_refreshes_only_when_the_list_changed(self):
        section = WatchListSectionGenerator(None)
        one_tick(section)
        assert section.content == WatchList.EMPTY
        with patch.object(section, "_update_content") as update:
            one_tick(section)
            update.assert_not_called()
        WatchList.save("Parcel", "ordered")
        one_tick(section)
        assert "parcel" in section.content and "ordered" in section.content


@pytest.mark.unit
class TestProceduresSection:
    def test_the_section_is_the_index_never_the_bodies(self):
        section = ProceduresSectionGenerator(None)
        assert section._fetch_content() == Procedures.EMPTY
        Procedures.save("shop-checkout", "paying at shop.example", "STEP BODY")
        content = section._fetch_content()
        assert "shop-checkout" in content and "paying at shop.example" in content
        assert "STEP BODY" not in content

    def test_the_loop_refreshes_only_when_a_procedure_changed(self):
        section = ProceduresSectionGenerator(None)
        one_tick(section)
        with patch.object(section, "_update_content") as update:
            one_tick(section)
            update.assert_not_called()
        Procedures.save("shop-checkout", "paying", "steps")
        one_tick(section)
        assert "shop-checkout" in section.content


@pytest.mark.unit
def test_the_prompt_template_has_a_place_for_both_sections():
    """Contract with res/prompt.md: PromptManager fills a section where its placeholder stands."""
    from common.Paths import RES_DIR
    template = (RES_DIR / "prompt.md").read_text()
    assert "{WATCHLIST}" in template and "{PROCEDURES}" in template
