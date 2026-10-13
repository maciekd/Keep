"""GraphCalendarAccount against a fake of Graph v1.0: the calendars, the calendarView window
asked in the home timezone and paged to the end, the event posted with that timezone, and an
event deleted or moved by its id."""
from datetime import date, datetime
from unittest.mock import patch

import pytest
import pytz

from assistant.integration.calendar.Event import Event
from integrations.microsoft.GraphCalendarAccount import GraphCalendarAccount
from integrations.microsoft.GraphClient import BASE_URL, GraphClient, GraphError

from .graph_fakes import FakeGraph, StubAuth

REQUEST = "integrations.microsoft.GraphClient.requests.request"
CALENDARS = {"value": [{"id": "c-1", "name": "Calendar"}, {"id": "c-2", "name": "Birthdays"}]}
STOCKHOLM = pytz.timezone("Europe/Stockholm")


def view(*events, next_link=None):
    body = {"value": [{"subject": title, "isAllDay": all_day, **dict(more),
                       "start": {"dateTime": start, "timeZone": "Europe/Stockholm"},
                       "end": {"dateTime": end, "timeZone": "Europe/Stockholm"}} for title, start, end, all_day, *more in events]}
    if next_link:
        body["@odata.nextLink"] = next_link
    return body


@pytest.fixture
def graph():
    fake = FakeGraph()
    with patch(REQUEST, fake):
        yield fake


def account(shown=(), default="Calendar", tz="Europe/Stockholm") -> GraphCalendarAccount:
    return GraphCalendarAccount("Work", GraphClient(StubAuth()), lambda: list(shown), lambda: default, lambda: tz)


@pytest.mark.unit
class TestCalendars:
    def test_every_calendar_is_named_and_the_check_reads_them(self, graph):
        graph.route("GET", "/me/calendars", CALENDARS)
        acc = account(shown=["Calendar"], default="Birthdays")

        assert [c.name for c in acc.calendars()] == ["Calendar", "Birthdays"]
        assert acc.check() is True and acc.service == "Microsoft 365"
        assert (acc.shown_calendars, acc.default_calendar) == (["Calendar"], "Birthdays")


@pytest.mark.unit
class TestEvents:
    def test_the_view_is_asked_in_the_home_timezone_and_the_times_come_back_aware(self, graph):
        graph.route("GET", "/me/calendars", CALENDARS)
        graph.route("GET", "/me/calendars/c-1/calendarView",
                    view(("Standup", "2026-06-15T09:00:00.0000000", "2026-06-15T09:30:00.0000000", False)))

        events = account().events(["calendar"], date(2026, 6, 15), date(2026, 6, 22))

        call, = graph.sent("GET", "/me/calendars/c-1/calendarView")
        assert call.headers["Prefer"] == 'outlook.timezone="Europe/Stockholm"'
        assert (call.params["startDateTime"], call.params["endDateTime"]) == ("2026-06-15T00:00:00", "2026-06-22T00:00:00")
        assert {"id", "subject", "type"} <= set(call.params["$select"].split(",")) and call.params["$top"] == 250
        event, = events
        assert event.start == datetime(2026, 6, 15, 7, 0, tzinfo=pytz.UTC) and event.end.tzinfo is not None
        assert (event.title, event.calendar, event.account, event.all_day) == ("Standup", "Calendar", "Work", False)
        assert not graph.sent("GET", "/me/calendars/c-2/calendarView")

    def test_the_id_comes_back_and_an_occurrence_of_a_series_is_recurring(self, graph):
        graph.route("GET", "/me/calendars", CALENDARS)
        graph.route("GET", "/me/calendars/c-1/calendarView",
                    view(("Dentist", "2026-06-15T09:00:00.0000000", "2026-06-15T10:00:00.0000000", False, ("id", "ev-1"), ("type", "singleInstance")),
                         ("Standup", "2026-06-16T09:00:00.0000000", "2026-06-16T09:15:00.0000000", False, ("id", "ev-2"), ("type", "occurrence"))))

        dentist, standup = account().events(["Calendar"], date(2026, 6, 15), date(2026, 6, 22))

        assert (dentist.id, dentist.recurring) == ("ev-1", False) and (standup.id, standup.recurring) == ("ev-2", True)

    def test_an_all_day_event_becomes_midnight_utc_bounds(self, graph):
        graph.route("GET", "/me/calendars", CALENDARS)
        graph.route("GET", "/me/calendars/c-2/calendarView",
                    view(("Frances' birthday", "2026-06-16T00:00:00.0000000", "2026-06-17T00:00:00.0000000", True)))

        event, = account().events(["Birthdays"], date(2026, 6, 15), date(2026, 6, 22))

        assert event.all_day and event.start == datetime(2026, 6, 16, tzinfo=pytz.UTC) and event.end == datetime(2026, 6, 17, tzinfo=pytz.UTC)

    def test_the_pages_of_a_view_are_followed_to_the_end(self, graph):
        graph.route("GET", "/me/calendars", CALENDARS)
        second = f"{BASE_URL}/me/calendars/c-1/calendarView?$skip=250"
        graph.route("GET", "/me/calendars/c-1/calendarView",
                    view(("One", "2026-06-15T09:00:00", "2026-06-15T10:00:00", False), next_link=second))
        graph.route("GET", "/me/calendars/c-1/calendarView?$skip=250",
                    view(("Two", "2026-06-16T09:00:00", "2026-06-16T10:00:00", False)))

        events = account().events(["Calendar"], date(2026, 6, 15), date(2026, 6, 22))

        assert [e.title for e in events] == ["One", "Two"]

    def test_a_zone_name_pytz_does_not_know_falls_back_to_the_home_zone(self, graph):
        graph.route("GET", "/me/calendars", CALENDARS)
        graph.route("GET", "/me/calendars/c-1/calendarView", {"value": [{
            "subject": "Call", "isAllDay": False,
            "start": {"dateTime": "2026-06-15T09:00:00", "timeZone": "W. Europe Standard Time"},
            "end": {"dateTime": "2026-06-15T10:00:00", "timeZone": "W. Europe Standard Time"}}]})

        event, = account().events(["Calendar"], date(2026, 6, 15), date(2026, 6, 22))

        assert event.start == STOCKHOLM.localize(datetime(2026, 6, 15, 9, 0))


@pytest.mark.unit
class TestCreate:
    def test_a_timed_event_is_posted_in_the_home_timezone(self, graph):
        graph.route("GET", "/me/calendars", CALENDARS)
        graph.route("POST", "/me/calendars/c-1/events", {"id": "ev-1"}, status=201)
        event = Event(datetime(2026, 6, 15, 7, 0, tzinfo=pytz.UTC), datetime(2026, 6, 15, 8, 30, tzinfo=pytz.UTC), "Dentist Charlotte")

        account().create_event("Calendar", event)

        posted, = graph.sent("POST", "/me/calendars/c-1/events")
        assert posted.json["subject"] == "Dentist Charlotte" and posted.json["isAllDay"] is False
        assert posted.json["start"] == {"dateTime": "2026-06-15T09:00:00", "timeZone": "Europe/Stockholm"}
        assert posted.json["end"] == {"dateTime": "2026-06-15T10:30:00", "timeZone": "Europe/Stockholm"}

    def test_an_all_day_event_spans_midnight_to_the_next_midnight(self, graph):
        graph.route("GET", "/me/calendars", CALENDARS)
        graph.route("POST", "/me/calendars/c-2/events", {"id": "ev-2"}, status=201)
        event = Event(datetime(2026, 6, 16, tzinfo=pytz.UTC), datetime(2026, 6, 16, tzinfo=pytz.UTC), "Ingeborg visits", all_day=True)

        account().create_event("Birthdays", event)

        posted, = graph.sent("POST", "/me/calendars/c-2/events")
        assert posted.json["isAllDay"] is True
        assert (posted.json["start"]["dateTime"], posted.json["end"]["dateTime"]) == ("2026-06-16T00:00:00", "2026-06-17T00:00:00")

    def test_an_unknown_calendar_names_the_known_ones(self, graph):
        graph.route("GET", "/me/calendars", CALENDARS)
        with pytest.raises(ValueError) as raised:
            account().create_event("Kids", Event(datetime(2026, 6, 15, tzinfo=pytz.UTC), datetime(2026, 6, 15, tzinfo=pytz.UTC), "x"))
        assert "Birthdays" in str(raised.value) and not graph.sent("POST", "/me/calendars/c-1/events")


def _read(id="ev-1"):
    return Event(datetime(2026, 6, 15, 7, 0, tzinfo=pytz.UTC), datetime(2026, 6, 15, 8, 30, tzinfo=pytz.UTC), "Dentist Charlotte",
                 calendar="Calendar", account="Work", id=id)


@pytest.mark.unit
class TestDeleteAndMove:
    def test_the_event_is_deleted_by_its_id(self, graph):
        graph.route("GET", "/me/calendars", CALENDARS)
        graph.route("DELETE", "/me/events/ev-1", None, status=204)

        account().delete_event("Calendar", _read())

        assert len(graph.sent("DELETE", "/me/events/ev-1")) == 1

    def test_a_move_patches_the_bounds_in_the_home_timezone(self, graph):
        graph.route("GET", "/me/calendars", CALENDARS)
        graph.route("PATCH", "/me/events/ev-1", {"id": "ev-1"})

        account().move_event("Calendar", _read(), datetime(2026, 6, 18, 8, 0, tzinfo=pytz.UTC),
                             datetime(2026, 6, 18, 9, 30, tzinfo=pytz.UTC), all_day=False)

        patched, = graph.sent("PATCH", "/me/events/ev-1")
        assert patched.json == {"isAllDay": False,
                                "start": {"dateTime": "2026-06-18T10:00:00", "timeZone": "Europe/Stockholm"},
                                "end": {"dateTime": "2026-06-18T11:30:00", "timeZone": "Europe/Stockholm"}}

    def test_an_all_day_move_runs_midnight_to_the_next_midnight(self, graph):
        graph.route("GET", "/me/calendars", CALENDARS)
        graph.route("PATCH", "/me/events/ev-1", {"id": "ev-1"})

        account().move_event("Calendar", _read(), datetime(2026, 6, 18, tzinfo=pytz.UTC), datetime(2026, 6, 18, tzinfo=pytz.UTC), all_day=True)

        patched, = graph.sent("PATCH", "/me/events/ev-1")
        assert patched.json["isAllDay"] is True
        assert (patched.json["start"]["dateTime"], patched.json["end"]["dateTime"]) == ("2026-06-18T00:00:00", "2026-06-19T00:00:00")

    def test_a_gone_event_an_unknown_calendar_or_no_id_raises_a_value_error(self, graph):
        graph.route("GET", "/me/calendars", CALENDARS)
        with pytest.raises(ValueError, match="no longer"):
            account().delete_event("Calendar", _read(id="gone"))
        with pytest.raises(ValueError, match="Calendar 'Kids' not found"):
            account().delete_event("Kids", _read())
        with pytest.raises(ValueError, match="no event id"):
            account().move_event("Calendar", _read(id=""), datetime(2026, 6, 18, tzinfo=pytz.UTC), datetime(2026, 6, 18, tzinfo=pytz.UTC), True)
        assert not [c for c in graph.calls if c.method == "PATCH"]

    def test_a_backend_error_is_not_reported_as_a_gone_event(self, graph):
        graph.route("GET", "/me/calendars", CALENDARS)
        graph.route("DELETE", "/me/events/ev-1", {"error": {"code": "InternalServerError", "message": "boom"}}, status=500)
        with pytest.raises(GraphError, match="boom") as raised:
            account().delete_event("Calendar", _read())
        assert raised.value.status == 500
