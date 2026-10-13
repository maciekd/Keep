"""GoogleCalendarAccount against a fake of Calendar v3: the calendar list with its primary
aliases, the events window asked in the home timezone and paged to the end, the event posted
with that timezone, an event deleted or moved by its id, and the 401 / rate-limit rules."""
from datetime import date, datetime
from unittest.mock import patch

import pytest
import pytz

from assistant.integration.calendar.Event import Event
from integrations.google.GoogleAuth import NotSignedIn
from integrations.google.GoogleCalendarAccount import GoogleCalendarAccount, GoogleCalendarError

from .google_fakes import FakeGoogle, FakeResponse, StubAuth, all_day, calendar, calendar_list, events_page, timed

REQUEST = "integrations.google.GoogleCalendarAccount.requests.request"
SLEEP = "integrations.google.GoogleCalendarAccount.time.sleep"
PRIMARY_PATH = "/calendars/marcus%40example.com/events"
FAMILY_PATH = "/calendars/family-id%40group.calendar.google.com/events"
KIDS_PATH = "/calendars/kids-id%40group.calendar.google.com/events"
CALENDARS = calendar_list(calendar("marcus@example.com", "marcus@example.com", primary=True),
                          calendar("Family", "family-id@group.calendar.google.com"))
STOCKHOLM = pytz.timezone("Europe/Stockholm")


@pytest.fixture
def google():
    fake = FakeGoogle()
    fake.route("GET", "/users/me/calendarList", CALENDARS)
    with patch(REQUEST, fake):
        yield fake


def account(shown=(), default="primary", tz="Europe/Stockholm", auth=None) -> GoogleCalendarAccount:
    return GoogleCalendarAccount("Personal", auth or StubAuth(), lambda: list(shown), lambda: default, lambda: tz,
                                 lambda: "marcus@example.com")


@pytest.mark.unit
class TestCalendars:
    def test_every_calendar_is_named_and_the_check_reads_one_entry_of_the_list(self, google):
        acc = account(shown=["Family"], default="Family")

        assert [c.name for c in acc.calendars()] == ["marcus@example.com", "Family"]
        assert acc.check() is True and acc.service == "Google"
        assert (acc.shown_calendars, acc.default_calendar) == (["Family"], "Family")
        assert google.sent("GET", "/users/me/calendarList")[-1].params == {"maxResults": 1}
        assert google.calls[0].headers["Authorization"] == "Bearer ya29.tok"

    def test_the_primary_calendar_answers_to_primary_and_to_the_address_and_once_only(self, google):
        google.route("GET", PRIMARY_PATH, events_page(timed("Standup", "2026-06-15T09:00:00+02:00", "2026-06-15T09:30:00+02:00")))

        events = account().events(["primary", "marcus@example.com", "Nobody's"], date(2026, 6, 15), date(2026, 6, 22))

        assert [e.title for e in events] == ["Standup"] and events[0].calendar == "marcus@example.com"
        assert len(google.sent("GET", PRIMARY_PATH)) == 1


@pytest.mark.unit
class TestEvents:
    def test_the_window_is_asked_in_the_home_timezone_as_single_instances_and_the_times_come_back_aware(self, google):
        google.route("GET", FAMILY_PATH, events_page(timed("Dentist Charlotte", "2026-06-15T09:00:00+02:00", "2026-06-15T10:30:00+02:00")))

        events = account().events(["family"], date(2026, 6, 15), date(2026, 6, 22))

        call, = google.sent("GET", FAMILY_PATH)
        assert (call.params["timeMin"], call.params["timeMax"]) == ("2026-06-15T00:00:00+02:00", "2026-06-22T00:00:00+02:00")
        assert (call.params["singleEvents"], call.params["orderBy"], call.params["timeZone"]) == ("true", "startTime", "Europe/Stockholm")
        assert call.params["maxResults"] == 250
        event, = events
        assert event.start == datetime(2026, 6, 15, 7, 0, tzinfo=pytz.UTC) and event.end == datetime(2026, 6, 15, 8, 30, tzinfo=pytz.UTC)
        assert (event.title, event.calendar, event.account, event.all_day) == ("Dentist Charlotte", "Family", "Personal", False)
        assert not google.sent("GET", PRIMARY_PATH)

    def test_an_all_day_event_becomes_midnight_utc_bounds_and_a_zulu_time_is_utc(self, google):
        google.route("GET", FAMILY_PATH, events_page(all_day("Frances' birthday", "2026-06-16", "2026-06-17"),
                                                     timed("Call Ingeborg", "2026-06-16T12:00:00Z", "2026-06-16T12:30:00Z")))

        birthday, call = account().events(["Family"], date(2026, 6, 15), date(2026, 6, 22))

        assert birthday.all_day and birthday.start == datetime(2026, 6, 16, tzinfo=pytz.UTC) and birthday.end == datetime(2026, 6, 17, tzinfo=pytz.UTC)
        assert not call.all_day and call.start == STOCKHOLM.localize(datetime(2026, 6, 16, 14, 0))

    def test_the_id_comes_back_and_an_instance_of_a_series_is_recurring(self, google):
        google.route("GET", FAMILY_PATH, events_page(
            timed("Dentist", "2026-06-15T09:00:00+02:00", "2026-06-15T10:00:00+02:00", id="ev-1"),
            timed("Standup", "2026-06-16T09:00:00+02:00", "2026-06-16T09:15:00+02:00", id="ev-2_20260616T070000Z", recurringEventId="ev-2")))

        dentist, standup = account().events(["Family"], date(2026, 6, 15), date(2026, 6, 22))

        assert (dentist.id, dentist.recurring) == ("ev-1", False)
        assert (standup.id, standup.recurring) == ("ev-2_20260616T070000Z", True)

    def test_the_pages_are_followed_through_next_page_token_and_cancelled_instances_left_out(self, google):
        pages = {None: events_page(timed("One", "2026-06-15T09:00:00+02:00", "2026-06-15T10:00:00+02:00"), next_page_token="p2"),
                 "p2": events_page(timed("Gone", "2026-06-15T11:00:00+02:00", "2026-06-15T12:00:00+02:00", status="cancelled"),
                                   timed("Two", "2026-06-16T09:00:00+02:00", "2026-06-16T10:00:00+02:00"))}
        google.route("GET", FAMILY_PATH, lambda params, json: FakeResponse(200, pages[params.get("pageToken")]))

        events = account().events(["Family"], date(2026, 6, 15), date(2026, 6, 22))

        assert [e.title for e in events] == ["One", "Two"]
        first, second = google.sent("GET", FAMILY_PATH)
        assert "pageToken" not in first.params and second.params["pageToken"] == "p2" and second.params["singleEvents"] == "true"


@pytest.mark.unit
class TestCreate:
    def test_a_timed_event_is_posted_in_the_home_timezone(self, google):
        google.route("POST", FAMILY_PATH, {"id": "ev-1"})
        event = Event(datetime(2026, 6, 15, 7, 0, tzinfo=pytz.UTC), datetime(2026, 6, 15, 8, 30, tzinfo=pytz.UTC), "Dentist Charlotte")

        account().create_event("Family", event)

        posted, = google.sent("POST", FAMILY_PATH)
        assert posted.json["summary"] == "Dentist Charlotte"
        assert posted.json["start"] == {"dateTime": "2026-06-15T09:00:00", "timeZone": "Europe/Stockholm"}
        assert posted.json["end"] == {"dateTime": "2026-06-15T10:30:00", "timeZone": "Europe/Stockholm"}

    def test_an_all_day_event_is_posted_as_dates_ending_the_day_after(self, google):
        google.route("POST", PRIMARY_PATH, {"id": "ev-2"})
        event = Event(datetime(2026, 6, 16, tzinfo=pytz.UTC), datetime(2026, 6, 16, tzinfo=pytz.UTC), "Ingeborg visits", all_day=True)

        account().create_event("primary", event)

        posted, = google.sent("POST", PRIMARY_PATH)
        assert posted.json["start"] == {"date": "2026-06-16"} and posted.json["end"] == {"date": "2026-06-17"}

    def test_an_unknown_calendar_names_the_known_ones(self, google):
        with pytest.raises(ValueError) as raised:
            account().create_event("Kids", Event(datetime(2026, 6, 15, tzinfo=pytz.UTC), datetime(2026, 6, 15, tzinfo=pytz.UTC), "x"))
        assert "Family" in str(raised.value) and not [c for c in google.calls if c.method == "POST"]


def _read(id="ev-1", title="Dentist Charlotte"):
    return Event(datetime(2026, 6, 15, 7, 0, tzinfo=pytz.UTC), datetime(2026, 6, 15, 8, 30, tzinfo=pytz.UTC), title,
                 calendar="Family", account="Personal", id=id)


@pytest.mark.unit
class TestDeleteAndMove:
    def test_the_event_is_deleted_by_its_id_under_its_calendar(self, google):
        google.route("DELETE", FAMILY_PATH + "/ev-1", None, status=204)

        account().delete_event("Family", _read())

        assert len(google.sent("DELETE", FAMILY_PATH + "/ev-1")) == 1

    def test_a_move_patches_the_bounds_in_the_home_timezone(self, google):
        google.route("PATCH", FAMILY_PATH + "/ev-1", {"id": "ev-1"})

        account().move_event("Family", _read(), datetime(2026, 6, 18, 8, 0, tzinfo=pytz.UTC),
                             datetime(2026, 6, 18, 9, 30, tzinfo=pytz.UTC), all_day=False)

        patched, = google.sent("PATCH", FAMILY_PATH + "/ev-1")
        assert patched.json == {"start": {"dateTime": "2026-06-18T10:00:00", "timeZone": "Europe/Stockholm"},
                                "end": {"dateTime": "2026-06-18T11:30:00", "timeZone": "Europe/Stockholm"}}

    def test_an_all_day_move_patches_dates_ending_the_day_after(self, google):
        google.route("PATCH", FAMILY_PATH + "/ev-1", {"id": "ev-1"})

        account().move_event("Family", _read(), datetime(2026, 6, 18, tzinfo=pytz.UTC), datetime(2026, 6, 18, tzinfo=pytz.UTC), all_day=True)

        patched, = google.sent("PATCH", FAMILY_PATH + "/ev-1")
        assert patched.json == {"start": {"date": "2026-06-18"}, "end": {"date": "2026-06-19"}}

    def test_a_gone_event_an_unknown_calendar_or_no_id_raises_a_value_error(self, google):
        with pytest.raises(ValueError, match="no longer"):
            account().delete_event("Family", _read(id="gone"))
        with pytest.raises(ValueError, match="Calendar 'Kids' not found"):
            account().move_event("Kids", _read(), datetime(2026, 6, 18, tzinfo=pytz.UTC), datetime(2026, 6, 18, tzinfo=pytz.UTC), True)
        with pytest.raises(ValueError, match="no event id"):
            account().delete_event("Family", _read(id=""))
        assert not [c for c in google.calls if c.method in ("DELETE", "PATCH") and c.path != FAMILY_PATH + "/gone"]

    def test_a_410_is_a_gone_event_but_another_error_stays_the_backends(self, google):
        """Only "not there any more" is the user's business; a backend failure must surface as
        one, not be reported as the event having been deleted by someone else."""
        google.route("DELETE", FAMILY_PATH + "/ev-1", {"error": {"code": 410, "message": "Resource has been deleted"}}, status=410)
        with pytest.raises(ValueError, match="no longer"):
            account().delete_event("Family", _read())

        google.route("PATCH", FAMILY_PATH + "/ev-1", {"error": {"code": 500, "message": "Backend Error"}}, status=500)
        with pytest.raises(GoogleCalendarError, match="Backend Error") as raised:
            account().move_event("Family", _read(), datetime(2026, 6, 18, 8, 0, tzinfo=pytz.UTC),
                                 datetime(2026, 6, 18, 9, 30, tzinfo=pytz.UTC), all_day=False)
        assert raised.value.status == 500


@pytest.mark.unit
class TestFailures:
    def test_a_401_is_the_sign_in_hint(self, google):
        google.route("GET", "/users/me/calendarList", {"error": {"code": 401, "message": "Invalid Credentials"}}, status=401)
        with pytest.raises(NotSignedIn) as raised:
            account().check()
        assert 'google_login("Personal")' in str(raised.value)

    def test_a_rate_limit_is_retried_once_after_a_second_and_then_given_up(self, google):
        limited = {"error": {"code": 403, "message": "Rate Limit Exceeded", "errors": [{"reason": "rateLimitExceeded"}]}}
        answers = [FakeResponse(403, limited), FakeResponse(200, events_page(timed("One", "2026-06-15T09:00:00+02:00", "2026-06-15T10:00:00+02:00")))]
        google.route("GET", FAMILY_PATH, lambda params, json: answers.pop(0))
        with patch(SLEEP) as sleep:
            events = account().events(["Family"], date(2026, 6, 15), date(2026, 6, 22))
        assert [e.title for e in events] == ["One"] and sleep.call_args.args == (1,)

        google.route("GET", FAMILY_PATH, {"error": {"code": 429, "message": "Too Many Requests"}}, status=429)
        with patch(SLEEP), pytest.raises(GoogleCalendarError) as raised:
            account().events(["Family"], date(2026, 6, 15), date(2026, 6, 22))
        assert raised.value.status == 429 and len(google.sent("GET", FAMILY_PATH)) == 4

    def test_another_error_carries_googles_message(self, google):
        google.route("GET", FAMILY_PATH, {"error": {"code": 404, "message": "Not Found", "errors": [{"reason": "notFound"}]}}, status=404)
        with pytest.raises(GoogleCalendarError, match="Not Found"):
            account().events(["Family"], date(2026, 6, 15), date(2026, 6, 22))

    def test_an_account_without_a_token_never_reaches_the_api(self, google):
        with pytest.raises(NotSignedIn):
            account(auth=StubAuth(signed_in=False)).calendars()
        assert google.calls == []


@pytest.mark.unit
class TestNamesAndBounds:
    def test_a_renamed_calendar_goes_by_its_override_not_the_owners_summary(self, google):
        google.route("GET", "/users/me/calendarList", calendar_list(
            calendar("marcus@example.com", "marcus@example.com", primary=True),
            calendar("family-xyz", "kids-id@group.calendar.google.com", summaryOverride="Kids")))
        google.route("GET", KIDS_PATH, events_page(timed("Swim lesson", "2026-06-17T16:00:00+02:00", "2026-06-17T17:00:00+02:00")))
        acc = account()

        assert [c.name for c in acc.calendars()] == ["marcus@example.com", "Kids"]
        events = acc.events(["kids", "family-xyz"], date(2026, 6, 15), date(2026, 6, 22))

        assert [(e.title, e.calendar) for e in events] == [("Swim lesson", "Kids")]
        assert len(google.sent("GET", KIDS_PATH)) == 1 and len(google.sent("GET", "/users/me/calendarList")) == 1

    def test_an_all_day_event_over_several_days_keeps_its_exclusive_end(self, google):
        """The Event's all-day end is already the day after the last one (the renderer's convention,
        the same shape Google sends back), so it is posted as is — only a zero-length one is widened."""
        google.route("POST", FAMILY_PATH, {"id": "ev-3"})
        event = Event(datetime(2026, 6, 16, tzinfo=pytz.UTC), datetime(2026, 6, 19, tzinfo=pytz.UTC), "Camp", all_day=True)

        account().create_event("Family", event)

        posted, = google.sent("POST", FAMILY_PATH)
        assert posted.json["start"] == {"date": "2026-06-16"} and posted.json["end"] == {"date": "2026-06-19"}

    def test_a_403_that_is_not_a_rate_limit_fails_at_once_without_a_retry(self, google):
        forbidden = {"error": {"code": 403, "message": "Insufficient Permission", "errors": [{"reason": "insufficientPermissions"}]}}
        google.route("GET", FAMILY_PATH, forbidden, status=403)

        with patch(SLEEP) as sleep, pytest.raises(GoogleCalendarError) as raised:
            account().events(["Family"], date(2026, 6, 15), date(2026, 6, 22))

        assert raised.value.status == 403 and "Insufficient Permission" in str(raised.value)
        assert len(google.sent("GET", FAMILY_PATH)) == 1 and not sleep.called
