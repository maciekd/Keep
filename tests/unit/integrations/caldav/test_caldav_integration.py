"""The CalDAV folder: the account that turns the library's calendars and vobject events into
CalendarInfo and Event and deletes or moves one by its UID, the integration that builds one per
`caldav.accounts` entry, and the icalendar pins the library needs."""
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import pytest
import pytz
import vobject
from caldav.lib.error import NotFoundError, ReportError

from assistant.integration.calendar.CalendarSource import CalendarSource
from assistant.integration.calendar.Event import Event
from integrations.caldav.CalDavCalendarAccount import CalDavCalendarAccount

CALDAV = 'integrations.caldav.CalDavCalendarAccount.caldav'
CONFIG = 'integrations.caldav.CalDAVIntegration.ConfigManager.get_instance'
UTC = pytz.UTC


def _calendar(name, *events, components=("VEVENT",), day=date(2025, 6, 15)):
    """A library calendar holding `(title, start)` events on `day`; a start may be an hour, a
    vobject string, a plain date or a naive datetime."""
    calendar = Mock()
    calendar.name = name
    if components is None:
        calendar.get_supported_components.side_effect = RuntimeError("no props")
    else:
        calendar.get_supported_components.return_value = list(components)
    found = []
    for title, start in events:
        if isinstance(start, int):
            start, end = (datetime(day.year, day.month, day.day, start, 0, tzinfo=UTC),
                          datetime(day.year, day.month, day.day, start + 1, 0, tzinfo=UTC))
        else:
            start, end = start
        vevent = Mock()
        vevent.contents = {'dtstart': [Mock(value=start)], 'uid': [Mock(value=f"uid-{title}")]}
        if end is not None:
            vevent.contents['dtend'] = [Mock(value=end)]
        vevent.summary = Mock(value=title)
        event = Mock()
        event.vobject_instance.vevent = vevent
        found.append(event)
    calendar.date_search.return_value = found
    return calendar


ICS = ("BEGIN:VCALENDAR\nVERSION:2.0\nBEGIN:VEVENT\nUID:uid-1\nSUMMARY:Dentist\n{bounds}\nSEQUENCE:0\n"
       "END:VEVENT\nEND:VCALENDAR\n")


def _resource(bounds="DTSTART;TZID=Europe/Stockholm:20250615T110000\nDTEND;TZID=Europe/Stockholm:20250615T120000"):
    """A library event object over a real vobject, the way `event_by_uid` answers."""
    resource = Mock()
    resource.vobject_instance = vobject.readOne(ICS.format(bounds=bounds))
    return resource


def _stored(resource) -> dict:
    """The DTSTART / DTEND lines the resource would PUT."""
    lines = resource.vobject_instance.serialize().splitlines()
    return {line.split(";")[0].split(":")[0]: line for line in lines if line.startswith(("DTSTART", "DTEND", "DURATION"))}


def _serve(mock_caldav, *calendars):
    mock_caldav.DAVClient.return_value.principal.return_value.calendars.return_value = list(calendars)


def _account(shown=(), default="Home", username="m@mlab.test", password="app-password"):
    return CalDavCalendarAccount("iCloud", "https://caldav.example/", lambda: (username, password),
                                 lambda: list(shown), lambda: default)


@pytest.mark.unit
class TestCalendars:
    @patch(CALDAV)
    def test_names_every_calendar_and_marks_a_reminders_list(self, mock_caldav):
        _serve(mock_caldav, _calendar("Home"), _calendar("Shopping", components=("VTODO",)), _calendar("Mystery", components=None))

        infos = _account().calendars()

        assert [(c.name, c.holds_events) for c in infos] == [("Home", True), ("Shopping", False), ("Mystery", True)]
        mock_caldav.DAVClient.assert_called_once_with("https://caldav.example/", username="m@mlab.test", password="app-password")

    def test_missing_credentials_raise_before_any_connection(self):
        with patch(CALDAV) as mock_caldav:
            with pytest.raises(ValueError):
                _account(password="").calendars()
            mock_caldav.DAVClient.assert_not_called()


@pytest.mark.unit
class TestEvents:
    @patch(CALDAV)
    def test_only_the_named_calendars_are_searched_whatever_their_case(self, mock_caldav):
        home, work = _calendar("Home", ("Dentist Charlotte", 9)), _calendar("Work", ("Budget review", 13))
        _serve(mock_caldav, home, work)

        events = _account().events(["work"], date(2025, 6, 15), date(2025, 6, 20))

        assert [e.title for e in events] == ["Budget review"]
        assert events[0].calendar == "Work" and events[0].account == "iCloud"
        home.date_search.assert_not_called()
        work.date_search.assert_called_once_with(start=datetime(2025, 6, 15), end=datetime(2025, 6, 20))

    @patch(CALDAV)
    def test_the_calendars_of_the_last_listing_are_reused_without_a_second_lookup(self, mock_caldav):
        _serve(mock_caldav, _calendar("Home", ("Dentist", 9)))
        account = _account()

        account.calendars()
        account.events(["Home"], date(2025, 6, 15), date(2025, 6, 20))

        assert mock_caldav.DAVClient.call_count == 1

    @patch(CALDAV)
    def test_a_vobject_string_time_is_utc(self, mock_caldav):
        _serve(mock_caldav, _calendar("Home", ("String Time", ("2025-06-15T10:00:00Z", "2025-06-15T11:00:00Z"))))

        event, = _account().events(["Home"], date(2025, 6, 15), date(2025, 6, 20))

        assert event.start == datetime(2025, 6, 15, 10, 0, tzinfo=UTC) and not event.all_day

    @patch(CALDAV)
    def test_a_plain_date_is_an_all_day_event_at_midnight_utc(self, mock_caldav):
        _serve(mock_caldav, _calendar("Home", ("All Day", (date(2025, 6, 15), date(2025, 6, 16)))))

        event, = _account().events(["Home"], date(2025, 6, 15), date(2025, 6, 20))

        assert event.all_day
        assert (event.start, event.end) == (datetime(2025, 6, 15, tzinfo=UTC), datetime(2025, 6, 16, tzinfo=UTC))

    @patch(CALDAV)
    def test_a_naive_datetime_is_taken_as_utc(self, mock_caldav):
        _serve(mock_caldav, _calendar("Home", ("Naive", (datetime(2025, 6, 15, 10, 0), datetime(2025, 6, 15, 11, 0)))))

        event, = _account().events(["Home"], date(2025, 6, 15), date(2025, 6, 20))

        assert event.start.tzinfo is not None and event.start.utcoffset() == timedelta(0)

    @patch(CALDAV)
    def test_the_uid_is_the_events_id_and_a_series_or_an_occurrence_of_one_is_recurring(self, mock_caldav):
        home = _calendar("Home", ("Single", 9), ("Weekly", 10), ("Occurrence", 11))
        home.date_search.return_value[1].vobject_instance.vevent.contents['rrule'] = [Mock(value="FREQ=WEEKLY")]
        home.date_search.return_value[2].vobject_instance.vevent.contents['recurrence-id'] = [Mock(value=datetime(2025, 6, 15, 11))]
        _serve(mock_caldav, home)

        single, weekly, occurrence = _account().events(["Home"], date(2025, 6, 15), date(2025, 6, 20))

        assert single.id == "uid-Single" and not single.recurring
        assert weekly.recurring and occurrence.recurring

    @patch(CALDAV)
    def test_a_duration_instead_of_an_end_gives_the_end_and_neither_makes_an_instant(self, mock_caldav):
        home = _calendar("Home", ("Timed", (datetime(2025, 6, 15, 10, 0, tzinfo=UTC), None)),
                         ("Bare", (datetime(2025, 6, 15, 12, 0, tzinfo=UTC), None)))
        home.date_search.return_value[0].vobject_instance.vevent.contents['duration'] = [Mock(value=timedelta(minutes=90))]
        _serve(mock_caldav, home)

        timed, bare = _account().events(["Home"], date(2025, 6, 15), date(2025, 6, 20))

        assert timed.end == datetime(2025, 6, 15, 11, 30, tzinfo=UTC) and bare.end == bare.start


@pytest.mark.unit
class TestCreateEvent:
    def _event(self, title="Dentist"):
        tz = pytz.timezone("Europe/Stockholm")
        return Event(tz.localize(datetime(2025, 10, 28, 10, 0)), tz.localize(datetime(2025, 10, 28, 11, 0)), title)

    @patch(CALDAV)
    def test_saves_the_event_in_the_named_calendar(self, mock_caldav):
        home, family = _calendar("Home"), _calendar("Family")
        _serve(mock_caldav, home, family)
        event = self._event()

        _account().create_event("Family", event)

        family.save_event.assert_called_once_with(dtstart=event.start, dtend=event.end, summary="Dentist")
        home.save_event.assert_not_called()

    @patch(CALDAV)
    def test_an_unknown_calendar_raises_and_names_the_available_ones(self, mock_caldav):
        _serve(mock_caldav, _calendar("Work"))

        with pytest.raises(ValueError, match="Calendar 'Home' not found") as raised:
            _account().create_event("Home", self._event())
        assert "Work" in str(raised.value)


def _read(title="Dentist", recurring=False, uid="uid-1"):
    tz = pytz.timezone("Europe/Stockholm")
    return Event(tz.localize(datetime(2025, 6, 15, 11, 0)), tz.localize(datetime(2025, 6, 15, 12, 0)), title,
                 calendar="Home", account="iCloud", id=uid, recurring=recurring)


@pytest.mark.unit
class TestDeleteEvent:
    @patch(CALDAV)
    def test_the_resource_found_by_the_uid_in_the_named_calendar_is_deleted(self, mock_caldav):
        home, family = _calendar("Home"), _calendar("Family")
        _serve(mock_caldav, home, family)
        resource = _resource()
        home.event_by_uid.return_value = resource

        _account().delete_event("Home", _read())

        home.event_by_uid.assert_called_once_with("uid-1")
        resource.delete.assert_called_once_with()
        family.event_by_uid.assert_not_called()

    @patch(CALDAV)
    def test_a_recurring_event_is_refused_before_the_lookup_and_a_series_master_after_it(self, mock_caldav):
        home = _calendar("Home")
        _serve(mock_caldav, home)
        home.event_by_uid.return_value = _resource(bounds="DTSTART:20250615T090000Z\nDTEND:20250615T100000Z\nRRULE:FREQ=WEEKLY")

        with pytest.raises(ValueError, match="recurring"):
            _account().delete_event("Home", _read(recurring=True))
        home.event_by_uid.assert_not_called()
        with pytest.raises(ValueError, match="recurring"):
            _account().delete_event("Home", _read())
        home.event_by_uid.return_value.delete.assert_not_called()

    @patch(CALDAV)
    def test_an_event_without_a_uid_an_unknown_calendar_or_a_gone_event_raises(self, mock_caldav):
        home = _calendar("Home")
        _serve(mock_caldav, home)
        with pytest.raises(ValueError, match="no UID"):
            _account().delete_event("Home", _read(uid=""))
        with pytest.raises(ValueError, match="Calendar 'Work' not found"):
            _account().delete_event("Work", _read())
        home.event_by_uid.side_effect = NotFoundError("404 Not Found")
        with pytest.raises(ValueError, match="no longer"):
            _account().delete_event("Home", _read())

    @patch(CALDAV)
    def test_an_event_read_earlier_is_fetched_by_the_href_that_read_kept_not_by_a_uid_report(self, mock_caldav):
        """iCloud answers the UID REPORT with 412 for every event (issue #76); the href of the
        read the user acted on needs no query."""
        home = _calendar("Home", ("Dentist", 11))
        home.date_search.return_value[0].url = "https://caldav.example/home/abc.ics"
        _serve(mock_caldav, home)
        home.event_by_url.return_value = _resource()
        account = _account()
        account.events(["Home"], date(2025, 6, 15), date(2025, 6, 16))

        account.delete_event("Home", _read(uid="uid-Dentist"))

        home.event_by_url.assert_called_once_with("https://caldav.example/home/abc.ics")
        home.event_by_uid.assert_not_called()
        home.event_by_url.return_value.delete.assert_called_once_with()

    @patch(CALDAV)
    def test_a_lookup_the_server_refuses_is_an_error_not_a_gone_event(self, mock_caldav):
        home = _calendar("Home")
        _serve(mock_caldav, home)
        home.event_by_uid.side_effect = ReportError("ReportError at '412 Precondition Failed'")
        with pytest.raises(RuntimeError, match="refused") as caught:
            _account().delete_event("Home", _read())
        assert "412" in str(caught.value) and "no longer" not in str(caught.value)

    @patch(CALDAV)
    def test_a_deleted_event_s_href_is_forgotten_so_a_repeat_asks_by_uid_and_finds_it_gone(self, mock_caldav):
        home = _calendar("Home", ("Dentist", 11))
        home.date_search.return_value[0].url = "https://caldav.example/home/abc.ics"
        _serve(mock_caldav, home)
        home.event_by_url.return_value = _resource()
        home.event_by_uid.side_effect = NotFoundError("404 Not Found")
        account = _account()
        account.events(["Home"], date(2025, 6, 15), date(2025, 6, 16))

        account.delete_event("Home", _read(uid="uid-Dentist"))
        with pytest.raises(ValueError, match="no longer"):
            account.delete_event("Home", _read(uid="uid-Dentist"))

        home.event_by_url.assert_called_once()
        home.event_by_uid.assert_called_once_with("uid-Dentist")


@pytest.mark.unit
class TestMoveEvent:
    UTC_BOUNDS = (datetime(2025, 6, 18, 8, 0, tzinfo=UTC), datetime(2025, 6, 18, 9, 30, tzinfo=UTC))

    @patch(CALDAV)
    def test_an_event_gone_from_the_server_since_the_read_is_said_gone_without_a_uid_query(self, mock_caldav):
        home = _calendar("Home", ("Dentist", 11))
        home.date_search.return_value[0].url = "https://caldav.example/home/abc.ics"
        _serve(mock_caldav, home)
        home.event_by_url.side_effect = NotFoundError("404 Not Found")
        account = _account()
        account.events(["Home"], date(2025, 6, 15), date(2025, 6, 16))

        with pytest.raises(ValueError, match="no longer"):
            account.move_event("Home", _read(uid="uid-Dentist"), *self.UTC_BOUNDS, all_day=False)
        home.event_by_uid.assert_not_called()

    @patch(CALDAV)
    def test_timed_bounds_are_written_in_utc_without_the_old_tzid_and_the_resource_saved(self, mock_caldav):
        home = _calendar("Home")
        _serve(mock_caldav, home)
        resource = _resource()
        home.event_by_uid.return_value = resource

        _account().move_event("Home", _read(), *self.UTC_BOUNDS, all_day=False)

        resource.save.assert_called_once_with()
        assert _stored(resource) == {"DTSTART": "DTSTART:20250618T080000Z", "DTEND": "DTEND:20250618T093000Z"}

    @patch(CALDAV)
    def test_all_day_bounds_are_written_as_dates(self, mock_caldav):
        home = _calendar("Home")
        _serve(mock_caldav, home)
        resource = _resource()
        home.event_by_uid.return_value = resource

        _account().move_event("Home", _read(), datetime(2025, 6, 18, tzinfo=UTC), datetime(2025, 6, 19, tzinfo=UTC), all_day=True)

        assert _stored(resource) == {"DTSTART": "DTSTART;VALUE=DATE:20250618", "DTEND": "DTEND;VALUE=DATE:20250619"}

    @patch(CALDAV)
    def test_a_duration_gives_way_to_an_end_and_an_all_day_event_can_become_a_timed_one(self, mock_caldav):
        home = _calendar("Home")
        _serve(mock_caldav, home)
        with_duration = _resource(bounds="DTSTART:20250615T090000Z\nDURATION:PT1H")
        all_day = _resource(bounds="DTSTART;VALUE=DATE:20250615\nDTEND;VALUE=DATE:20250616")
        home.event_by_uid.side_effect = [with_duration, all_day]

        _account().move_event("Home", _read(), *self.UTC_BOUNDS, all_day=False)
        _account().move_event("Home", _read(), *self.UTC_BOUNDS, all_day=False)

        assert _stored(with_duration) == {"DTSTART": "DTSTART:20250618T080000Z", "DTEND": "DTEND:20250618T093000Z"}
        assert _stored(all_day) == {"DTSTART": "DTSTART:20250618T080000Z", "DTEND": "DTEND:20250618T093000Z"}

    @patch(CALDAV)
    def test_a_recurring_event_is_refused_unsaved(self, mock_caldav):
        home = _calendar("Home")
        _serve(mock_caldav, home)
        resource = _resource(bounds="DTSTART:20250615T090000Z\nDTEND:20250615T100000Z\nRRULE:FREQ=WEEKLY")
        home.event_by_uid.return_value = resource

        with pytest.raises(ValueError, match="recurring"):
            _account().move_event("Home", _read(), *self.UTC_BOUNDS, all_day=False)
        resource.save.assert_not_called()


@pytest.mark.unit
class TestCheck:
    @patch(CALDAV)
    def test_reads_today_from_the_first_calendar(self, mock_caldav):
        home = _calendar("Home")
        _serve(mock_caldav, home)

        assert _account().check() is True

        start, end = home.date_search.call_args.kwargs["start"], home.date_search.call_args.kwargs["end"]
        assert start.date() == date.today() and (end - start).days == 1

    @patch(CALDAV)
    def test_a_connection_failure_raises_for_the_monitor(self, mock_caldav):
        mock_caldav.DAVClient.side_effect = ConnectionError("failed")
        with pytest.raises(ConnectionError):
            _account().check()


def _config(accounts):
    """ConfigManager.get as the integration reads it, serving `caldav.accounts`."""
    config = MagicMock()
    config.get.side_effect = lambda key, *default: accounts if key == "caldav.accounts" else (default[0] if default else None)
    return patch(CONFIG, return_value=config)


@pytest.mark.unit
class TestCalDAVIntegration:
    @pytest.fixture(autouse=True)
    def fresh(self):
        from integrations.caldav.CalDAVIntegration import CalDAVIntegration
        CalDAVIntegration.reset_instance()
        yield
        CalDAVIntegration.reset_instance()

    def _integration(self):
        from integrations.caldav.CalDAVIntegration import CalDAVIntegration
        return CalDAVIntegration()

    def test_builds_one_account_per_entry_the_icloud_preset_giving_the_url_a_contract_with_caldav_accounts(self):
        entries = [{"name": "Family", "preset": "icloud", "username": "m@mlab.test", "password": "app-password",
                    "shown_calendars": ["Home, Family", "Work"], "default_calendar": "Family"},
                   {"name": "Work", "url": "https://caldav.example/", "username": "marcus@example.com", "password": "pw"}]
        with _config(entries):
            integration = self._integration()
            family, work = integration.calendar_accounts()
            assert isinstance(integration, CalendarSource) and integration.config_keys == ["caldav.accounts"]
            assert family.name == "Family" and family._url == "https://caldav.icloud.com/" and family.service == "CalDAV"
            assert family._credentials() == ("m@mlab.test", "app-password")
            assert family.shown_calendars == ["Home", "Family", "Work"] and family.default_calendar == "Family"
            assert work._url == "https://caldav.example/" and work._credentials() == ("marcus@example.com", "pw")
            assert work.shown_calendars == [] and work.default_calendar == "Home"

    def test_the_credentials_are_read_from_config_at_call_time(self):
        entries = [{"name": "Family", "preset": "icloud", "username": "m@mlab.test", "password": "old"}]
        with _config(entries):
            account, = self._integration().calendar_accounts()
            entries[0]["password"] = "new"
            assert account._credentials() == ("m@mlab.test", "new")

    def test_an_entry_without_a_preset_or_url_an_unknown_preset_or_a_repeated_name_is_skipped(self):
        entries = [{"name": "Family", "username": "m@mlab.test", "password": "pw"},
                   {"name": "Work", "preset": "exchange", "username": "x", "password": "y"},
                   {"name": "Kids", "preset": "icloud", "username": "k@example.com", "password": "pw"},
                   {"name": "kids", "preset": "icloud", "username": "other@example.com", "password": "pw"}]
        with _config(entries):
            accounts = self._integration().calendar_accounts()
        assert [a.name for a in accounts] == ["Kids"]

    def test_the_example_config_lists_shown_calendars_as_plain_values_so_config_set_can_add_one_by_index(self):
        from common import ConfigSchema
        assert ConfigSchema.is_scalar_list("caldav.accounts[0].shown_calendars")
        assert ConfigSchema.example_value("caldav.accounts[0].shown_calendars") == []

    def test_health_checks_every_account_and_names_the_failing_ones(self):
        entries = [{"name": "Family", "preset": "icloud", "username": "a", "password": "b"},
                   {"name": "Work", "preset": "icloud", "username": "c", "password": "d"}]
        with _config(entries):
            integration = self._integration()
        family, work = integration.calendar_accounts()
        with patch.object(family, "check", return_value=True), patch.object(work, "check", side_effect=ConnectionError("down")):
            with pytest.raises(RuntimeError) as raised:
                integration.check_health()
        assert "Work" in str(raised.value) and "down" in str(raised.value) and "Family" not in str(raised.value)
        assert [(a["name"], a["state"]) for a in integration.accounts_status()] == [("Family", "ok"), ("Work", "error")]
        with patch.object(family, "check", return_value=True), patch.object(work, "check", return_value=True):
            assert integration.check_health() is True
        assert all(a["state"] == "ok" for a in integration.accounts_status())

    def test_without_an_account_health_fails_and_the_status_is_empty(self):
        with _config([]):
            integration = self._integration()
            assert integration.check_health() is False and integration.accounts_status() == []


class TestIcalendarDependencyCompatibility:
    """Guards the icalendar/recurring-ical-events pins that caldav 1.3.9 needs.

    These exercise the *real* installed libraries (no mocking), so they fail if
    icalendar is bumped to 6.0+ (component_factory becomes a non-subscriptable
    module, breaking event creation) or recurring-ical-events to 3.x (which then
    requires icalendar>=6.1, breaking recurrence expansion on the read path).
    """

    def test_component_factory_is_subscriptable(self):
        import icalendar
        # caldav/lib/vcal.py does icalendar.cal.component_factory["VEVENT"]()
        component = icalendar.cal.component_factory["VEVENT"]()
        assert component is not None

    def test_recurring_ical_events_imports(self):
        import recurring_ical_events
        assert hasattr(recurring_ical_events, "of")
