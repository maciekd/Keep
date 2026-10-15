"""WatchList: one file per tracked item under a directory, answering to its name; the note is
the current state, deleting removes the file, and every replaced or deleted version lands in
the log directory. An item may carry a check cadence, and `checked` records the last look
separately from `updated`."""
import os
from datetime import date, datetime, timedelta

import pytest

from assistant import WatchList

TODAY = date(2026, 3, 10)


@pytest.fixture
def wl(tmp_path):
    return tmp_path / "watchlist"


@pytest.fixture
def log(tmp_path):
    return tmp_path / "log"


def log_texts(log) -> list[str]:
    return [p.read_text() for p in sorted(log.glob("*.md"))] if log.is_dir() else []


def backdate(directory, name: str, field: str, day: date):
    """Rewrite one header date of an item's file, as if it had been written on `day`."""
    path = directory / f"{name}.md"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(f"{field}: {day.isoformat()}" if l.startswith(f"{field}:") else l for l in lines) + "\n")


@pytest.mark.unit
class TestLifecycle:
    def test_saved_items_are_read_back_from_their_files(self, wl, log):
        WatchList.save("Parcel from shop.example", "order 4711, courier, evening", "2026-03-12", wl, log)
        WatchList.save("Dentist for Charlotte", "12:30 at the clinic", None, wl, log)
        items = {i.name: (i.note, i.due) for i in WatchList.items(wl)}
        assert items == {
            "parcel-from-shop-example": ("order 4711, courier, evening", "2026-03-12"),
            "dentist-for-charlotte": ("12:30 at the clinic", ""),
        }
        assert len(list(wl.glob("*.md"))) == 2

    def test_items_come_oldest_first(self, wl, log):
        WatchList.save("alpha", "new", None, wl, log)
        WatchList.save("zulu", "old", None, wl, log)
        backdate(wl, "zulu", "Created", date.today() - timedelta(days=3))
        assert [i.name for i in WatchList.items(wl)] == ["zulu", "alpha"]

    def test_a_multi_line_note_survives_the_file(self, wl, log):
        WatchList.save("Invoice", "amount 1,200\n\nbankgiro 123-4567", None, wl, log)
        assert WatchList.items(wl)[0].note == "amount 1,200\n\nbankgiro 123-4567"

    def test_no_directory_means_no_items(self, wl):
        assert WatchList.items(wl) == []

    def test_saving_again_replaces_the_note_and_keeps_the_due_date(self, wl, log):
        WatchList.save("Parcel", "ordered", "2026-03-12", wl, log)
        updated = WatchList.save("Parcel", "shipped, tracking AB12", None, wl, log)
        assert updated.note == "shipped, tracking AB12"
        only, = WatchList.items(wl)
        assert (only.note, only.due) == ("shipped, tracking AB12", "2026-03-12")

    @pytest.mark.parametrize("spelling", ["tv delivery", "TV  Delivery", '"tv-delivery"', "`TV-delivery`", " Tv Delivery "])
    def test_another_spelling_of_the_name_updates_the_same_item(self, wl, log, spelling):
        WatchList.save("TV delivery", "ordered", "2026-03-12", wl, log)
        WatchList.save(spelling, "out for delivery", None, wl, log)
        only, = WatchList.items(wl)
        assert (only.name, only.note, only.due) == ("tv-delivery", "out for delivery", "2026-03-12")

    def test_saving_with_a_due_date_changes_it_and_a_dash_clears_it(self, wl, log):
        WatchList.save("Parcel", "ordered", "2026-03-12", wl, log)
        WatchList.save("Parcel", "delayed", "2026-03-15", wl, log)
        assert WatchList.get("Parcel", wl).due == "2026-03-15"
        WatchList.save("Parcel", "no date any more", "-", wl, log)
        assert WatchList.get("Parcel", wl).due == ""

    def test_an_update_keeps_the_day_the_item_was_created(self, wl, log):
        WatchList.save("Parcel", "ordered", None, wl, log)
        created = date.today() - timedelta(days=6)
        backdate(wl, "parcel", "Created", created)
        WatchList.save("Parcel", "shipped", None, wl, log)
        assert WatchList.get("Parcel", wl).created == created.isoformat()

    def test_an_update_leaves_the_previous_state_in_the_log(self, wl, log):
        WatchList.save("Parcel", "ordered at shop.example", None, wl, log)
        assert log_texts(log) == []
        WatchList.save("Parcel", "shipped", None, wl, log)
        texts = log_texts(log)
        assert len(texts) == 1
        assert "ordered at shop.example" in texts[0] and "shipped" not in texts[0]

    def test_delete_removes_the_file_and_logs_the_outcome(self, wl, log):
        WatchList.save("Parcel", "shipped", None, wl, log)
        deleted = WatchList.delete("parcel", "delivered to the door", wl, log)
        assert deleted.name == "parcel"
        assert WatchList.items(wl) == []
        assert list(wl.glob("*.md")) == []
        texts = log_texts(log)
        assert len(texts) == 1
        assert "shipped" in texts[0] and "delivered to the door" in texts[0]

    def test_a_vault_value_in_the_outcome_reaches_the_log_as_its_placeholder(self, wl, log, monkeypatch):
        monkeypatch.setattr(WatchList, "redact_for_storage",
                            lambda text: text.replace("4111 1111 1111 1111", "{{secret:marcus-visa}}"))
        WatchList.save("Invoice", "due Friday", None, wl, log)
        WatchList.delete("invoice", "paid with 4111 1111 1111 1111", wl, log)
        [text] = log_texts(log)
        assert "4111" not in text and "paid with {{secret:marcus-visa}}" in text

    def test_deleting_an_unknown_item_does_nothing(self, wl, log):
        WatchList.save("Parcel", "shipped", None, wl, log)
        assert WatchList.delete("dentist", "x", wl, log) is None
        assert len(WatchList.items(wl)) == 1
        assert log_texts(log) == []

    def test_a_name_without_letters_or_digits_is_refused(self, wl, log):
        assert WatchList.save(" -- ", "note", None, wl, log) is None
        assert WatchList.get("", wl) is None
        assert WatchList.items(wl) == []

    def test_a_deleted_name_can_be_watched_afresh(self, wl, log):
        WatchList.save("Parcel", "first order", "2026-03-12", wl, log)
        WatchList.delete("Parcel", "delivered", wl, log)
        again = WatchList.save("Parcel", "second order", None, wl, log)
        assert (again.note, again.due) == ("second order", "")


@pytest.mark.unit
class TestChecks:
    """`updated` moves when the state changes, `checked` whenever someone looks."""

    def test_a_cadence_is_kept_across_updates_and_a_dash_clears_it(self, wl, log):
        WatchList.save("Tickets", "not on sale yet", None, wl, log, every="1d")
        WatchList.save("Tickets", "presale announced", None, wl, log)
        assert WatchList.get("Tickets", wl).every == "1d"
        WatchList.save("Tickets", "on sale, bought", None, wl, log, every="-")
        assert WatchList.get("Tickets", wl).every == ""

    def test_an_item_without_a_cadence_has_none(self, wl, log):
        WatchList.save("Dentist", "12:30", None, wl, log)
        assert WatchList.get("Dentist", wl).every == ""

    def test_saving_stamps_the_check_with_the_time_of_day(self, wl, log):
        before = datetime.now().replace(second=0, microsecond=0)
        item = WatchList.save("Tickets", "not yet", None, wl, log, every="6h")
        stamped = datetime.strptime(WatchList.get("Tickets", wl).checked, "%Y-%m-%d %H:%M")
        assert before <= stamped <= datetime.now()
        assert item.checked == WatchList.get("Tickets", wl).checked

    def test_a_check_moves_checked_but_not_the_note_or_updated(self, wl, log):
        WatchList.save("Tickets", "not on sale yet", "2026-08-01", wl, log, every="1d")
        backdate(wl, "tickets", "Updated", date.today() - timedelta(days=5))
        backdate(wl, "tickets", "Checked", date.today() - timedelta(days=5))
        checked = WatchList.check("tickets", wl)
        item = WatchList.get("Tickets", wl)
        assert checked.name == "tickets"
        assert (item.note, item.due, item.every) == ("not on sale yet", "2026-08-01", "1d")
        assert item.updated == (date.today() - timedelta(days=5)).isoformat()
        assert item.checked.startswith(date.today().isoformat())

    def test_a_check_leaves_no_copy_in_the_log(self, wl, log):
        WatchList.save("Tickets", "not yet", None, wl, log, every="1d")
        WatchList.check("Tickets", wl)
        assert log_texts(log) == []

    def test_checking_an_unknown_item_does_nothing(self, wl, log):
        assert WatchList.check("plumber", wl) is None

    def test_an_old_file_without_the_new_headers_still_reads(self, wl):
        wl.mkdir()
        (wl / "parcel.md").write_text("# parcel\nDue: -\nCreated: 2026-03-01\nUpdated: 2026-03-02\n\nordered\n")
        item, = WatchList.items(wl)
        assert (item.note, item.every, item.checked) == ("ordered", "", "")


@pytest.mark.unit
class TestStaleness:
    def item(self, due="", updated=TODAY):
        return WatchList.Item("parcel", "note", due, TODAY.isoformat(), updated.isoformat())

    def test_a_fresh_item_is_not_stale(self):
        assert WatchList.staleness(self.item(due=TODAY.isoformat()), TODAY) == ""

    def test_due_yesterday_is_not_yet_stale_two_days_is(self):
        assert WatchList.staleness(self.item(due=(TODAY - timedelta(days=1)).isoformat()), TODAY) == ""
        assert "STALE" in WatchList.staleness(self.item(due=(TODAY - timedelta(days=2)).isoformat()), TODAY)

    def test_a_due_date_with_a_time_counts_by_its_date(self):
        due = f"{(TODAY - timedelta(days=5)).isoformat()} 17:45"
        assert "5 days" in WatchList.staleness(self.item(due=due), TODAY)

    def test_a_free_text_due_is_never_overdue(self):
        assert WatchList.staleness(self.item(due="early next month"), TODAY) == ""

    def test_untouched_for_two_weeks_is_stale(self):
        assert WatchList.staleness(self.item(updated=TODAY - timedelta(days=13)), TODAY) == ""
        assert "STALE" in WatchList.staleness(self.item(updated=TODAY - timedelta(days=14)), TODAY)


@pytest.mark.unit
class TestRender:
    def test_empty_list_says_so(self, wl):
        assert WatchList.render(wl) == WatchList.EMPTY

    def test_each_open_item_carries_its_name_due_and_note(self, wl, log):
        WatchList.save("Parcel from shop.example", "order 4711", "2026-03-12", wl, log)
        WatchList.save("Dentist for Charlotte", "12:30", None, wl, log)
        blocks = {b.splitlines()[0].split("`")[1]: b for b in WatchList.render(wl, date.today()).split("\n\n")}
        parcel, dentist = blocks["parcel-from-shop-example"], blocks["dentist-for-charlotte"]
        assert "2026-03-12" in parcel and "order 4711" in parcel
        assert "12:30" in dentist
        assert "due" not in dentist.splitlines()[0]

    def test_a_stale_item_is_marked_and_a_fresh_one_is_not(self, wl, log):
        WatchList.save("Old parcel", "x", (date.today() - timedelta(days=9)).isoformat(), wl, log)
        WatchList.save("New parcel", "x", (date.today() + timedelta(days=2)).isoformat(), wl, log)
        blocks = WatchList.render(wl).split("\n\n")
        old = next(b for b in blocks if "old-parcel" in b)
        new = next(b for b in blocks if "new-parcel" in b)
        assert "STALE" in old and "STALE" not in new

    def test_an_item_nobody_touched_is_marked(self, wl, log):
        WatchList.save("Forgotten", "x", None, wl, log)
        backdate(wl, "forgotten", "Updated", date.today() - timedelta(days=30))
        assert "STALE" in WatchList.render(wl)

    def test_a_deleted_item_is_gone(self, wl, log):
        WatchList.save("Parcel", "x", None, wl, log)
        WatchList.delete("Parcel", "done", wl, log)
        assert WatchList.render(wl) == WatchList.EMPTY

    def test_the_cadence_and_the_last_check_stand_side_by_side(self, wl, log):
        WatchList.save("Tickets", "not yet", None, wl, log, every="1d")
        WatchList.save("Dentist", "12:30", None, wl, log)
        blocks = {b.splitlines()[0].split("`")[1]: b.splitlines()[0] for b in WatchList.render(wl).split("\n\n")}
        assert "1d" in blocks["tickets"] and WatchList.get("Tickets", wl).checked in blocks["tickets"]
        assert "every" not in blocks["dentist"]


@pytest.mark.unit
class TestIndex:
    def test_empty_list_says_so(self, wl):
        assert WatchList.render_index(wl) == WatchList.EMPTY

    def test_one_line_per_item_with_its_name_and_the_start_of_its_note(self, wl, log):
        long_note = "order 4711 with the courier, " + "tracking page shows no movement yet, " * 4 + "door code needed"
        WatchList.save("Parcel from shop.example", long_note, "2026-03-12", wl, log)
        WatchList.save("Dentist for Charlotte", "12:30\nbring the referral", None, wl, log)
        lines = {line.split("`")[1]: line for line in WatchList.render_index(wl, TODAY).splitlines()}
        parcel, dentist = lines["parcel-from-shop-example"], lines["dentist-for-charlotte"]
        assert "order 4711" in parcel and "door code needed" not in parcel and len(parcel) < len(long_note)
        assert "12:30" in dentist and "referral" not in dentist and "due" not in dentist

    def test_a_date_still_ahead_is_not_shown_a_reached_one_is(self, wl, log):
        WatchList.save("Sofa", "x", (TODAY + timedelta(days=2)).isoformat(), wl, log)
        WatchList.save("Plumber", "x", f"{TODAY.isoformat()} 17:30", wl, log)
        WatchList.save("Invoice", "x", (TODAY - timedelta(days=3)).isoformat(), wl, log)
        lines = {line.split("`")[1]: line for line in WatchList.render_index(wl, TODAY).splitlines()}
        assert "due" not in lines["sofa"] and "2026" not in lines["sofa"]
        assert "due today 17:30" in lines["plumber"]
        assert "overdue 3 d" in lines["invoice"] and "STALE" not in lines["invoice"]

    def test_an_item_nobody_touched_is_marked_stale(self, wl, log):
        WatchList.save("Forgotten", "x", None, wl, log)
        backdate(wl, "forgotten", "Updated", TODAY - timedelta(days=30))
        assert "STALE" in WatchList.render_index(wl, TODAY)

    def test_a_poll_that_only_stamps_leaves_the_index_as_it_was(self, wl, log):
        WatchList.save("Tickets", "not yet", None, wl, log, every="1d")
        before = WatchList.render_index(wl)
        backdate(wl, "tickets", "Checked", date.today() - timedelta(days=1))
        WatchList.check("Tickets", wl)
        assert WatchList.render_index(wl) == before
        assert "1d" not in before


@pytest.mark.unit
class TestLoad:
    def test_an_item_loads_whole_under_its_name(self, wl, log):
        long_note = "order 4711 with the courier, " + "tracking page shows no movement yet, " * 4 + "door code needed"
        WatchList.save("Parcel from shop.example", long_note, "2026-03-12", wl, log, every="6h")
        loaded = WatchList.load("Parcel from shop.example", wl)
        assert "parcel-from-shop-example" in loaded.splitlines()[0]
        assert long_note in loaded and "2026-03-12" in loaded and "6h" in loaded
        assert WatchList.get("parcel-from-shop-example", wl).checked in loaded

    def test_an_unknown_name_answers_with_the_open_ones(self, wl, log):
        WatchList.save("Parcel", "order 4711", None, wl, log)
        answer = WatchList.load("dentist", wl)
        assert "dentist" in answer and "parcel" in answer and "order 4711" not in answer

    def test_loading_is_not_a_look(self, wl, log):
        WatchList.save("Tickets", "not yet", None, wl, log, every="1d")
        backdate(wl, "tickets", "Checked", date.today() - timedelta(days=1))
        stamp = WatchList.get("Tickets", wl).checked
        WatchList.load("Tickets", wl)
        assert WatchList.get("Tickets", wl).checked == stamp


@pytest.mark.unit
class TestVersion:
    def test_version_moves_on_save_update_and_delete(self, wl, log):
        empty = WatchList.version(wl)
        WatchList.save("Parcel", "ordered", None, wl, log)
        added = WatchList.version(wl)
        assert added != empty
        # A same-day rewrite differs only by mtime: pin the old one so a coarse clock cannot hide it.
        os.utime(wl / "parcel.md", ns=(1, 1))
        before_update = WatchList.version(wl)
        WatchList.save("Parcel", "shipped", None, wl, log)
        updated = WatchList.version(wl)
        assert updated != before_update
        WatchList.delete("Parcel", "", wl, log)
        assert WatchList.version(wl) != updated

    def test_version_is_stable_when_nothing_changes(self, wl, log):
        WatchList.save("Parcel", "x", None, wl, log)
        assert WatchList.version(wl) == WatchList.version(wl)


@pytest.mark.unit
class TestRun:
    def test_save_watch_takes_name_note_and_due(self, wl, log):
        assert WatchList.run("save_watch", ["Parcel", "order 4711", "2026-03-12"], wl, log) == ""
        item = WatchList.items(wl)[0]
        assert (item.name, item.note, item.due) == ("parcel", "order 4711", "2026-03-12")

    def test_save_watch_without_a_due_date_keeps_the_one_it_has(self, wl, log):
        WatchList.run("save_watch", ["Parcel", "ordered", "2026-03-12"], wl, log)
        assert WatchList.run("save_watch", ["parcel", "shipped"], wl, log) == ""
        assert WatchList.run("save_watch", ["parcel", "at the depot", ""], wl, log) == ""
        item, = WatchList.items(wl)
        assert (item.note, item.due) == ("at the depot", "2026-03-12")

    def test_save_watch_takes_the_cadence_as_its_fourth_argument(self, wl, log):
        assert WatchList.run("save_watch", ["Tickets", "not yet", "", "1d"], wl, log) == ""
        item, = WatchList.items(wl)
        assert (item.due, item.every) == ("", "1d")
        assert WatchList.run("save_watch", ["Tickets", "presale"], wl, log) == ""
        assert WatchList.get("Tickets", wl).every == "1d"

    def test_checked_watch_stamps_the_item_and_names_the_open_ones_when_unknown(self, wl, log):
        WatchList.save("Tickets", "not yet", None, wl, log, every="1d")
        backdate(wl, "tickets", "Checked", date.today() - timedelta(days=3))
        assert WatchList.run("checked_watch", ["tickets"], wl, log) == ""
        assert WatchList.get("Tickets", wl).checked.startswith(date.today().isoformat())
        failure = WatchList.run("checked_watch", ["plumber"], wl, log)
        assert "plumber" in failure and "tickets" in failure
        assert WatchList.run("checked_watch", [], wl, log)

    def test_save_watch_without_a_name_or_a_note_is_refused(self, wl, log):
        assert WatchList.run("save_watch", [], wl, log)
        assert WatchList.run("save_watch", ["Parcel"], wl, log)
        assert WatchList.run("save_watch", ["Parcel", "  "], wl, log)
        assert WatchList.run("save_watch", ["  ", "a note"], wl, log)
        assert WatchList.items(wl) == []

    def test_an_unknown_name_is_an_error_that_names_the_open_ones(self, wl, log):
        WatchList.save("Parcel", "x", None, wl, log)
        WatchList.save("Dentist", "x", None, wl, log)
        failure = WatchList.run("delete_watch", ["plumber", "done"], wl, log)
        assert "plumber" in failure and "parcel" in failure and "dentist" in failure
        assert len(WatchList.items(wl)) == 2

    def test_delete_without_a_name_is_an_error(self, wl, log):
        assert WatchList.run("delete_watch", [], wl, log)
        assert WatchList.run("delete_watch", [" "], wl, log)

    def test_delete_joins_a_comma_split_outcome(self, wl, log):
        WatchList.save("Parcel", "x", None, wl, log)
        assert WatchList.run("delete_watch", ["Parcel", "delivered", "left at the door"], wl, log) == ""
        assert "delivered, left at the door" in log_texts(log)[0]

    def test_a_foreign_command_is_refused(self, wl, log):
        assert WatchList.run("save_procedure", ["a", "b", "c"], wl, log)
        assert WatchList.run("watch_close", ["parcel"], wl, log)


@pytest.mark.unit
class TestCadence:
    @pytest.mark.parametrize("every,hours", [
        ("6h", 6), ("1d", 24), ("30m", 0.5), ("weekly", 168), ("every 2 days", 48), ("1d 12h", 36),
        ("every day", 24), ("hourly", 1), ("2w", 336), ("1.5 hours", 1.5),
    ])
    def test_a_cadence_is_read_as_the_time_between_checks(self, every, hours):
        assert WatchList.interval(every).total_seconds() == hours * 3600

    @pytest.mark.parametrize("every", ["", "when the courier answers", "0h"])
    def test_a_cadence_that_names_no_length_has_no_interval(self, every):
        assert WatchList.interval(every) is None

    def test_a_stamp_is_read_with_or_without_its_time(self):
        assert WatchList.stamp("2026-03-12 07:45").hour == 7
        assert WatchList.stamp("2026-03-12").date().isoformat() == "2026-03-12"
        assert WatchList.stamp("next week") is None and WatchList.stamp("2026-13-40") is None


@pytest.mark.unit
class TestDue:
    """What the autonomous check is handed: the items whose poll is due, nothing else."""

    def test_empty_list_says_nothing_is_due(self, wl):
        assert WatchList.render_due(wl) == WatchList.NOTHING_DUE

    def test_an_item_without_a_cadence_is_never_polled(self, wl, log):
        WatchList.save("Dentist", "12:30", "2026-03-12", wl, log)
        backdate(wl, "dentist", "Checked", date.today() - timedelta(days=30))
        assert WatchList.render_due(wl) == WatchList.NOTHING_DUE

    def test_an_item_just_checked_is_not_due_and_one_past_its_cadence_is(self, wl, log):
        WatchList.save("Tickets", "not yet", None, wl, log, every="1d")
        WatchList.save("Parcel", "in transit", None, wl, log, every="6h")
        now = datetime.now()
        assert WatchList.render_due(wl, now) == WatchList.NOTHING_DUE
        due = WatchList.render_due(wl, now + timedelta(hours=7))
        assert "parcel" in due and "tickets" not in due
        both = WatchList.render_due(wl, now + timedelta(hours=25))
        assert "parcel" in both and "tickets" in both

    def test_a_check_every_55_minutes_polls_an_hourly_item_every_time(self, wl, log):
        """The stamp is written a few minutes into a check, so at the next one the item is short
        of its hour: without the check's tolerance it is polled every other time."""
        WatchList.save("Parcel", "in transit", None, wl, log, every="1h")
        next_check = WatchList.stamp(WatchList.get("parcel", wl).checked) + timedelta(minutes=53)
        assert WatchList.render_due(wl, next_check) == WatchList.NOTHING_DUE
        assert "parcel" in WatchList.render_due(wl, next_check, tolerance=timedelta(minutes=27.5))

    def test_a_daily_item_is_polled_by_the_check_nearest_its_cadence(self, wl, log):
        WatchList.save("Tickets", "not yet", None, wl, log, every="1d")
        checked = WatchList.stamp(WatchList.get("tickets", wl).checked)
        tolerance = timedelta(minutes=27.5)
        assert WatchList.render_due(wl, checked + timedelta(hours=23), tolerance=tolerance) == WatchList.NOTHING_DUE
        assert "tickets" in WatchList.render_due(wl, checked + timedelta(hours=23, minutes=40), tolerance=tolerance)

    def test_the_tolerance_never_eats_more_than_half_a_short_cadence(self, wl, log):
        WatchList.save("Oven", "preheating", None, wl, log, every="20m")
        checked = WatchList.stamp(WatchList.get("oven", wl).checked)
        tolerance = timedelta(minutes=27.5)
        assert WatchList.render_due(wl, checked + timedelta(minutes=9), tolerance=tolerance) == WatchList.NOTHING_DUE
        assert "oven" in WatchList.render_due(wl, checked + timedelta(minutes=10), tolerance=tolerance)

    def test_a_due_item_comes_whole(self, wl, log):
        WatchList.save("Parcel", "order 4711, left the depot", "2026-03-12", wl, log, every="1d")
        text = WatchList.render_due(wl, datetime.now() + timedelta(days=2))
        assert "order 4711, left the depot" in text and "2026-03-12" in text and "1d" in text

    def test_a_cadence_the_code_cannot_read_leaves_the_call_to_the_llm(self, wl, log):
        WatchList.save("Tickets", "x", None, wl, log, every="when the tour is announced")
        assert "tickets" in WatchList.render_due(wl)

    def test_an_item_never_checked_is_due(self, wl, log):
        WatchList.save("Tickets", "x", None, wl, log, every="1d")
        path = wl / "tickets.md"
        path.write_text("\n".join(l for l in path.read_text().splitlines() if not l.startswith("Checked:")) + "\n")
        assert "tickets" in WatchList.render_due(wl)


@pytest.mark.unit
class TestDueTime:
    """A due *time* is a moment the assistant acts at: the item is due for a poll from then
    until it is looked at again, so a reminder a check failed to deliver comes back rather than
    being lost; a bare date is a fact in the index and fires nothing."""

    def test_a_due_time_is_read_and_a_bare_date_or_free_text_is_not(self):
        assert WatchList.due_moment(WatchList.Item("x", due="2026-10-11 17:00")) == datetime(2026, 10, 11, 17, 0)
        assert WatchList.due_moment(WatchList.Item("x", due="2026-10-11")) is None
        assert WatchList.due_moment(WatchList.Item("x", due="when the tour is announced")) is None

    def test_an_item_is_due_from_its_moment_until_it_is_looked_at(self, wl, log):
        WatchList.save("Water meter", "read it", "2026-10-11 17:00", wl, log)
        backdate(wl, "water-meter", "Checked", date(2026, 10, 11))
        assert WatchList.render_due(wl, datetime(2026, 10, 11, 16, 59)) == WatchList.NOTHING_DUE
        assert "water-meter" in WatchList.render_due(wl, datetime(2026, 10, 11, 17, 0))
        assert "water-meter" in WatchList.render_due(wl, datetime(2026, 10, 12, 9, 0)), "left alone, it comes back"
        WatchList.check("water-meter", wl)
        assert WatchList.render_due(wl, datetime.now() + timedelta(hours=1)) == WatchList.NOTHING_DUE

    def test_re_dating_the_item_ahead_settles_the_reached_moment(self, wl, log):
        WatchList.save("Water meter", "read it", "2026-10-11 17:00", wl, log)
        backdate(wl, "water-meter", "Checked", date(2026, 10, 11))
        WatchList.save("Water meter", "snoozed", "2026-10-11 19:00", wl, log)
        assert WatchList.render_due(wl, datetime(2026, 10, 11, 18, 0)) == WatchList.NOTHING_DUE

    def test_a_bare_due_date_never_makes_an_item_due_for_a_poll(self, wl, log):
        WatchList.save("Sofa", "on its way", "2026-10-11", wl, log)
        backdate(wl, "sofa", "Checked", date(2026, 10, 1))
        assert WatchList.render_due(wl, datetime(2026, 10, 13, 9, 0)) == WatchList.NOTHING_DUE

    def test_the_due_item_says_how_long_ago_its_moment_was_reached(self, wl, log):
        WatchList.save("Water meter", "read it", "2026-10-11 17:00", wl, log)
        backdate(wl, "water-meter", "Checked", date(2026, 10, 11))
        text = WatchList.render_due(wl, datetime(2026, 10, 11, 19, 10))
        assert "reached 2 h 10 min ago" in text and "not looked at since" in text

    @pytest.mark.parametrize("delta, text", [(timedelta(seconds=20), "just now"), (timedelta(minutes=3), "3 min"),
                                             (timedelta(hours=2, minutes=15), "2 h 15 min"), (timedelta(days=1, hours=4), "1 d 4 h")])
    def test_a_duration_reads_in_its_largest_fitting_unit(self, delta, text):
        assert WatchList.describe_duration(delta) == text


@pytest.mark.unit
class TestPending:
    """The moments the due watcher wakes for."""

    def test_due_times_ahead_and_reached_are_listed_soonest_first(self, wl, log):
        WatchList.save("Later", "x", "2026-10-11 19:00", wl, log)
        WatchList.save("Sooner", "x", "2026-10-11 17:00", wl, log)
        WatchList.save("Dated", "x", "2026-10-11", wl, log)
        for name in ("later", "sooner", "dated"):
            backdate(wl, name, "Checked", date(2026, 10, 11))
        pending = WatchList.pending(datetime(2026, 10, 11, 18, 0), directory=wl)
        assert [(item.name, moment) for item, moment in pending] == [
            ("sooner", datetime(2026, 10, 11, 17, 0)), ("later", datetime(2026, 10, 11, 19, 0))]

    def test_a_due_time_looked_at_since_is_not_pending(self, wl, log):
        WatchList.save("Done", "x", "2026-10-11 17:00", wl, log)   # the save is the look, after the moment
        assert WatchList.pending(datetime(2026, 10, 11, 18, 0), directory=wl) == []

    def test_a_cadence_shorter_than_the_window_polls_at_its_next_check(self, wl, log):
        WatchList.save("Car", "x", None, wl, log, every="3m")
        WatchList.save("Tickets", "x", None, wl, log, every="1d")
        checked = WatchList.stamp(WatchList.get("car", wl).checked)
        pending = WatchList.pending(checked, within=timedelta(minutes=55), directory=wl)
        assert [(item.name, moment) for item, moment in pending] == [("car", checked + timedelta(minutes=3))]

    def test_without_a_window_no_cadence_is_pending(self, wl, log):
        WatchList.save("Car", "x", None, wl, log, every="3m")
        assert WatchList.pending(datetime.now(), directory=wl) == []

    def test_a_cadence_item_never_checked_polls_at_its_creation(self, wl, log):
        WatchList.save("Car", "x", None, wl, log, every="3m", looked=False)
        pending = WatchList.pending(datetime.now(), within=timedelta(minutes=55), directory=wl)
        assert [(item.name, moment) for item, moment in pending] == [("car", WatchList.stamp(WatchList.get("car", wl).created))]


@pytest.mark.unit
class TestIndexExcerpt:
    """The index line carries the start of the note: whole when it fits, cut at a word when not."""

    def _lines(self, wl):
        return {line.split("`")[1]: line for line in WatchList.render_index(wl, TODAY).splitlines()}

    def test_a_note_that_just_fits_is_whole_and_one_character_more_is_cut_at_the_word(self, wl, log):
        fits = "x" * 10 + " " + "y" * (WatchList.INDEX_NOTE_CHARS - 11)   # exactly the limit
        WatchList.save("Fits", fits, None, wl, log)
        WatchList.save("Over", fits + "z", None, wl, log)
        lines = self._lines(wl)
        assert lines["fits"].endswith(fits)
        assert lines["over"].endswith("x" * 10 + "…")

    @pytest.mark.xfail(strict=True, reason="_note_start cuts first[:INDEX_NOTE_CHARS] at its last space even when the "
                                           "last character is the end of a whole word, so a word ending exactly at the "
                                           "limit is dropped")
    def test_a_word_that_ends_exactly_at_the_limit_is_kept(self, wl, log):
        second = "y" * (WatchList.INDEX_NOTE_CHARS - 11)
        WatchList.save("Edge", "x" * 10 + " " + second + " more words follow", None, wl, log)
        assert second in self._lines(wl)["edge"]

    def test_a_bare_date_reached_today_reads_due_today_with_no_time(self, wl, log):
        WatchList.save("Plumber", "x", TODAY.isoformat(), wl, log)
        assert "(due today)" in self._lines(wl)["plumber"]
