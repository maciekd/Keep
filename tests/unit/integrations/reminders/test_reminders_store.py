"""The EventKit side of Reminders without EventKit: the parsing, the listing and the id lookup."""
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from integrations.reminders.RemindersStore import (Place, Reminder, RemindersStore, RemindersAccessError,
                                                   format_listing, parse_due, short_id, RECENTLY_COMPLETED_DAYS)

NOW = datetime(2026, 10, 11, 21, 0)
UNDEFINED = 2 ** 63 - 1  # NSDateComponentUndefined


class TestPlace:
    def test_semicolon_form_parses_into_a_geofence(self):
        place = Place.parse("ICA Maxi; 57.6834; 11.9798; 200; arrive")
        assert (place.label, place.latitude, place.longitude, place.radius, place.proximity) == \
               ("ICA Maxi", 57.6834, 11.9798, 200.0, "arrive")

    @pytest.mark.parametrize("text", ["Home|57.7|11.97|150|leave", "Home, 57.7, 11.97, 150, LEAVE"])
    def test_pipes_and_commas_separate_too_and_the_proximity_is_case_insensitive(self, text):
        place = Place.parse(text)
        assert place.proximity == "leave" and place.label == "Home"

    @pytest.mark.parametrize("text", [
        "Home; 57.7; 11.97; 150",             # no proximity
        "Home; 57.7; 11.97; 150; nearby",     # not arrive/leave
        "Home; north; 11.97; 150; arrive",    # not a number
        "Home; 97.0; 11.97; 150; arrive",     # latitude out of range
        "Home; 57.7; 11.97; 0; arrive",       # no radius
    ])
    def test_a_malformed_place_is_refused_with_a_reason(self, text):
        with pytest.raises(ValueError):
            Place.parse(text)


class TestParseDue:
    def test_a_time_is_a_timed_due(self):
        assert parse_due("2026-10-13 09:00") == (datetime(2026, 10, 13, 9, 0), True)

    def test_a_bare_date_is_a_day_without_alarm(self):
        assert parse_due("'2026-10-13'") == (datetime(2026, 10, 13), False)

    def test_empty_is_no_due_date(self):
        assert parse_due("  ") == (None, False)

    @pytest.mark.parametrize("text", ["tomorrow 9", "13/10/2026 09:00", "2026-10-13T09:00"])
    def test_another_shape_is_refused(self, text):
        with pytest.raises(ValueError):
            parse_due(text)


def reminder(rid="1A2B3C4D", title="Call Alice", due=None, timed=False, completed_at=None, notes="", place=""):
    return Reminder(rid, title, due, timed, completed_at, notes, place)


class TestListing:
    def test_an_overdue_open_reminder_is_marked_and_a_done_one_shows_when(self):
        overdue = reminder(due=NOW - timedelta(hours=2), timed=True).line(NOW)
        done = reminder(completed_at=NOW - timedelta(days=1)).line(NOW)
        assert "OVERDUE" in overdue and "2026-10-11 19:00" in overdue
        assert "OVERDUE" not in done and "done 2026-10-10 21:00" in done

    def test_a_day_due_shows_no_time_and_the_place_and_notes_follow(self):
        line = reminder(due=datetime(2026, 10, 13), place="Home (arrive)", notes="bring  the\nkeys").line(NOW)
        assert "due 2026-10-13" in line and "00:00" not in line
        assert "at Home (arrive)" in line and "notes: bring the keys" in line

    def test_open_ones_come_first_by_due_then_undated_then_the_recently_done_latest_first(self):
        items = [
            reminder("D1", "done early", completed_at=NOW - timedelta(days=3)),
            reminder("U1", "no date"),
            reminder("D2", "done late", completed_at=NOW - timedelta(hours=1)),
            reminder("O2", "later", due=NOW + timedelta(days=2), timed=True),
            reminder("O1", "sooner", due=NOW + timedelta(hours=1), timed=True),
        ]
        lines = format_listing(items, "Michelle", NOW).splitlines()
        assert "3 open, 2 completed" in lines[0]
        assert [l.split("] ")[1].split()[0] for l in lines[1:]] == ["O1", "O2", "U1", "D2", "D1"]

    def test_an_empty_list_says_so(self):
        assert "no open reminders" in format_listing([], "Michelle", NOW)


# --- the EventKit objects, faked just far enough -------------------------------------------

def components(year=UNDEFINED, month=UNDEFINED, day=UNDEFINED, hour=UNDEFINED, minute=UNDEFINED):
    return SimpleNamespace(year=lambda: year, month=lambda: month, day=lambda: day,
                           hour=lambda: hour, minute=lambda: minute)


def fake_item(identifier, title="x", due=None, completed_at=None, alarms=(), notes=None):
    completion = None if completed_at is None else SimpleNamespace(timeIntervalSince1970=lambda: completed_at.timestamp())
    return SimpleNamespace(calendarItemIdentifier=lambda: identifier, title=lambda: title,
                           dueDateComponents=lambda: due, isCompleted=lambda: completed_at is not None,
                           completionDate=lambda: completion, alarms=lambda: list(alarms), notes=lambda: notes,
                           setCompleted_=lambda v: None)


def geofence_alarm(label, proximity):
    return SimpleNamespace(structuredLocation=lambda: SimpleNamespace(title=lambda: label), proximity=lambda: proximity)


class TestConversion:
    def test_a_timed_due_keeps_its_time_and_a_day_due_has_none(self):
        assert RemindersStore._components_to_datetime(components(2026, 10, 13, 9, 30)) == (datetime(2026, 10, 13, 9, 30), True)
        assert RemindersStore._components_to_datetime(components(2026, 10, 13)) == (datetime(2026, 10, 13), False)
        assert RemindersStore._components_to_datetime(None) == (None, False)

    def test_an_item_becomes_a_reminder_with_its_short_id_and_geofence(self):
        item = fake_item("2EC7F761-BB07-59D0-9D83-0C012B8FF3A1", "Buy flowers", components(2026, 10, 13, 9, 0),
                         alarms=[SimpleNamespace(structuredLocation=lambda: None, proximity=lambda: 0),
                                 geofence_alarm("Blomsterlandet", 2)])
        made = RemindersStore._to_reminder(item)
        assert made.id == "2EC7F761" and made.title == "Buy flowers"
        assert made.due == datetime(2026, 10, 13, 9, 0) and made.timed
        assert made.place == "Blomsterlandet (leave)" and not made.completed

    def test_a_completed_item_carries_its_completion_moment(self):
        when = datetime(2026, 10, 10, 14, 2)
        made = RemindersStore._to_reminder(fake_item("ABCDEF01-0000", completed_at=when))
        assert made.completed and made.completed_at == when


class FakeStore(RemindersStore):
    """The store with the EventKit calls replaced: one list holding the given items."""

    def __init__(self, items, list_exists=True):
        super().__init__("Michelle")
        self.items = items
        self.list_exists = list_exists
        self.saved = []
        self.removed = []
        self._store = SimpleNamespace(
            predicateForRemindersInCalendars_=lambda cals: "all",
            saveReminder_commit_error_=lambda item, commit, err: (self.saved.append(item) or True, None),
            removeReminder_commit_error_=lambda item, commit, err: (self.removed.append(item) or True, None),
        )

    def _calendar(self, create):
        return "michelle" if self.list_exists else None

    def _fetch(self, predicate):
        return list(self.items)


class TestLookupAndOperations:
    def test_a_short_id_resolves_by_prefix_case_insensitively(self):
        store = FakeStore([fake_item("2EC7F761-BB07", "a"), fake_item("8A158B9F-F44B", "b")])
        assert store._item("8a158b9f").title() == "b"
        assert store._item("8A158B9F-F44B").title() == "b"

    def test_an_unknown_or_ambiguous_id_is_refused(self):
        store = FakeStore([fake_item("2EC7F761-BB07"), fake_item("2EC7F761-0000")])
        with pytest.raises(RemindersAccessError, match="no reminder"):
            store._item("FFFFFFFF")
        with pytest.raises(RemindersAccessError, match="more of the id"):
            store._item("2EC7F761")
        with pytest.raises(RemindersAccessError):
            store._item("")

    def test_a_missing_list_lists_nothing_but_refuses_a_lookup(self):
        store = FakeStore([], list_exists=False)
        assert store.list(NOW) == []
        with pytest.raises(RemindersAccessError, match="no 'Michelle' list"):
            store.complete("2EC7F761")

    def test_list_keeps_open_and_recently_completed_only(self):
        old = NOW - timedelta(days=RECENTLY_COMPLETED_DAYS + 1)
        fresh = NOW - timedelta(days=1)
        store = FakeStore([fake_item("AAAAAAAA-1", "open"), fake_item("BBBBBBBB-1", "old", completed_at=old),
                           fake_item("CCCCCCCC-1", "fresh", completed_at=fresh)])
        assert [r.title for r in store.list(NOW)] == ["open", "fresh"]

    def test_complete_saves_and_refuses_a_second_time(self):
        store = FakeStore([fake_item("2EC7F761-BB07", "a"), fake_item("8A158B9F-F44B", "b", completed_at=NOW)])
        done = store.complete("2ec7f761")
        assert done.title == "a" and len(store.saved) == 1
        with pytest.raises(RemindersAccessError, match="already completed"):
            store.complete("8A158B9F")

    def test_delete_removes_the_item_and_hands_back_what_it_was(self):
        store = FakeStore([fake_item("2EC7F761-BB07", "a")])
        gone = store.delete("2EC7F761")
        assert gone.title == "a" and store.removed[0].title() == "a"

    def test_short_id_is_the_upper_cased_head_of_the_uuid(self):
        assert short_id("2ec7f761-bb07-59d0") == "2EC7F761"
