"""When the LLM last loaded a procedure or a memo: its load command stamps it, the curator's reading does not."""
from datetime import datetime, timedelta

import pytest

from assistant.Memos import Memos
from assistant.Procedures import Procedures

BODY = "open the cart\nclick Pay"


@pytest.fixture(autouse=True)
def stores(tmp_path, monkeypatch):
    for cls, folder in ((Procedures, "procedures"), (Memos, "memos")):
        monkeypatch.setattr(cls, "DIR", tmp_path / folder)
        monkeypatch.setattr(cls, "LOG_DIR", tmp_path / "log" / folder)
        monkeypatch.setattr(cls, "USED_FILE", tmp_path / "cache" / f"{folder}_used.json")


@pytest.mark.unit
class TestLastUsed:
    def test_nothing_loaded_yet_is_no_record(self):
        Procedures.save("shop", "paying", BODY)
        assert Procedures.last_used() == {}

    def test_the_load_command_stamps_each_file_it_found(self):
        Procedures.save("shop", "paying", BODY)
        Procedures.save("courier", "rebooking", BODY)
        before = datetime.now().replace(microsecond=0)
        Procedures.load_all(["Shop", "no-such-thing"])
        used = Procedures.last_used()
        assert set(used) == {"shop"}
        assert before <= used["shop"] <= datetime.now()

    def test_a_later_load_moves_the_stamp_and_keeps_the_others(self):
        Memos.save("wine-list", "choosing wine", "Barolo")
        Memos.save("trip", "the trip", "flight at 07:10")
        Memos.load_all(["wine-list", "trip"])
        Memos.USED_FILE.write_text('{"wine-list": "2026-01-02T03:04:05", "trip": "2026-01-02T03:04:05"}')
        Memos.load_all(["trip"])
        used = Memos.last_used()
        assert used["wine-list"] == datetime(2026, 1, 2, 3, 4, 5)
        assert used["trip"] > datetime.now() - timedelta(minutes=1)

    def test_the_curators_reading_is_not_a_use(self):
        Procedures.save("shop", "paying", BODY)
        Procedures.load("shop")
        assert Procedures.last_used() == {}

    def test_procedures_and_memos_keep_separate_records(self):
        Procedures.save("shop", "paying", BODY)
        Memos.save("shop", "the shop's hours", "9–17")
        Memos.load_all(["shop"])
        assert Procedures.last_used() == {} and set(Memos.last_used()) == {"shop"}

    def test_a_deleted_file_takes_its_record_with_it(self):
        Procedures.save("shop", "paying", BODY)
        Procedures.load_all(["shop"])
        Procedures.delete("shop")
        assert Procedures.last_used() == {}

    def test_an_unreadable_record_is_read_as_none_and_rewritten_on_the_next_load(self):
        Procedures.save("shop", "paying", BODY)
        Procedures.USED_FILE.parent.mkdir(parents=True)
        Procedures.USED_FILE.write_text("{not json")
        assert Procedures.last_used() == {}
        Procedures.load_all(["shop"])
        assert set(Procedures.last_used()) == {"shop"}
