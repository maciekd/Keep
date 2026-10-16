"""A CalendarAccount over CalDAV: the account's calendars, their events, a new event and the
deletion or move of one, parsed from and written as vobject through the `caldav` library."""
import logging
from datetime import datetime, timedelta, date
from typing import Callable, Optional

import caldav
import pytz
from caldav.lib.error import NotFoundError

from assistant.integration.calendar.CalendarAccount import CalendarAccount, CalendarInfo
from assistant.integration.calendar.Event import Event
from common.logging import info, trace


class CalDavCalendarAccount(CalendarAccount):
    """One CalDAV login. `credentials` is read on every call so a password set through
    config_set works without a rebuild; the calendar objects of the last `calendars()` are kept
    so the events and create calls that follow need no second principal lookup."""

    def __init__(self, name: str, url: str, credentials: Callable[[], tuple[str, str]],
                 shown_calendars: Callable[[], list[str]], default_calendar: Callable[[], str]):
        self._name = name
        self._url = url
        self._credentials = credentials
        self._shown_calendars = shown_calendars
        self._default_calendar = default_calendar
        self._objects: Optional[list] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def service(self) -> str:
        return "CalDAV"

    @property
    def shown_calendars(self) -> list[str]:
        return self._shown_calendars()

    @property
    def default_calendar(self) -> str:
        return self._default_calendar()

    def _calendar_objects(self, refresh: bool = False) -> list:
        """Every calendar of the account as the library's objects, looked up again on `refresh`."""
        if self._objects is not None and not refresh:
            return self._objects
        logging.getLogger("root").setLevel(logging.CRITICAL)
        username, password = self._credentials()
        if not username or not password:
            raise ValueError("Calendar username or password not found in config")
        client = caldav.DAVClient(self._url, username=username, password=password)
        self._objects = client.principal().calendars()
        trace(f"{self.name}: {len(self._objects)} calendars on {self._url}")
        return self._objects

    @staticmethod
    def _holds_events(calendar) -> bool:
        """False for a Reminders list, which iCloud serves next to the calendars."""
        try:
            return "VEVENT" in calendar.get_supported_components()
        except Exception as e:
            trace(f"Calendar '{calendar.name}': supported components unknown ({e}), treating it as a calendar")
            return True

    def calendars(self) -> list[CalendarInfo]:
        return [CalendarInfo(str(c.name), self._holds_events(c)) for c in self._calendar_objects(refresh=True)]

    def events(self, calendar_names: list[str], start: date, end: date) -> list[Event]:
        wanted = {name.lower() for name in calendar_names}
        found = []
        for calendar in self._calendar_objects():
            if str(calendar.name).lower() not in wanted:
                continue
            events = calendar.date_search(
                start=datetime.combine(start, datetime.min.time()),
                end=datetime.combine(end, datetime.min.time())
            )
            for event in events:
                vevent = event.vobject_instance.vevent
                dtstart, all_day = self._as_datetime(vevent.contents['dtstart'][0].value)
                dtend = self._end(vevent, dtstart)
                found.append(Event(dtstart, dtend, vevent.summary.value, calendar=str(calendar.name),
                                   account=self.name, all_day=all_day, id=self._uid(vevent),
                                   recurring=self._is_recurring(vevent)))
        trace(f"{self.name}: {len(found)} event(s) in {sorted(wanted)} between {start} and {end}")
        return found

    @classmethod
    def _end(cls, vevent, dtstart: datetime) -> datetime:
        """DTEND, or DTSTART plus DURATION when the event carries that instead, or DTSTART
        itself when it carries neither (an instant, the way RFC 5545 reads a bare DTSTART)."""
        if 'dtend' in vevent.contents:
            return cls._as_datetime(vevent.contents['dtend'][0].value)[0]
        if 'duration' in vevent.contents:
            return dtstart + vevent.contents['duration'][0].value
        return dtstart

    @staticmethod
    def _uid(vevent) -> str:
        return str(vevent.contents['uid'][0].value) if 'uid' in vevent.contents else ""

    @staticmethod
    def _is_recurring(vevent) -> bool:
        """A series (RRULE / RDATE) or one expanded occurrence of one (RECURRENCE-ID)."""
        return any(key in vevent.contents for key in ('rrule', 'rdate', 'recurrence-id'))

    @staticmethod
    def _as_datetime(value) -> tuple[datetime, bool]:
        """A vobject date value as a tz-aware datetime, and whether it was a plain date (all-day):
        an ISO string is UTC, a date is its midnight UTC, a naive datetime is taken as UTC."""
        if isinstance(value, str):
            return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.UTC), False
        if isinstance(value, date) and not isinstance(value, datetime):
            return datetime.combine(value, datetime.min.time()).replace(tzinfo=pytz.UTC), True
        if value.tzinfo is None:
            return value.replace(tzinfo=pytz.UTC), False
        return value, False

    def create_event(self, calendar_name: str, event: Event) -> None:
        calendar = self._calendar(calendar_name)
        calendar.save_event(dtstart=event.start, dtend=event.end, summary=event.title)
        info(f"{self.name}: event '{event.title}' saved in '{calendar_name}'")

    def _calendar(self, calendar_name: str):
        calendar = next((c for c in self._calendar_objects() if c.name == calendar_name), None)
        if calendar is None:
            raise ValueError(f"Calendar '{calendar_name}' not found. Available: {[c.name for c in self._calendar_objects()]}")
        return calendar

    LOOKUP_WINDOW = timedelta(days=1)

    def _resource(self, calendar_name: str, event: Event, verb: str):
        """The library's object for an event read earlier: a date search around the event's
        start, as the listing does, and the object whose UID is the event's. iCloud answers
        412 Precondition Failed to the UID prop-filter `event_by_uid` sends, so that lookup is
        never used. Refuses an event of a series, whose resource is the whole series (saving
        one expanded occurrence would overwrite the series with that instance, deleting it
        would delete every one)."""
        if event.recurring:
            raise ValueError(f"'{event.title}' is a recurring event, which cannot be {verb} yet — only single events can")
        if not event.id:
            raise ValueError(f"'{event.title}' has no UID, so it cannot be {verb}")
        calendar = self._calendar(calendar_name)
        try:
            candidates = calendar.date_search(start=event.start - self.LOOKUP_WINDOW,
                                              end=event.start + self.LOOKUP_WINDOW, expand=False)
        except NotFoundError as e:
            info(f"{self.name}: event '{event.title}' ({event.id}) not found in '{calendar_name}': {e}")
            raise ValueError(f"'{event.title}' is no longer in '{calendar_name}'") from e
        except Exception as e:
            info(f"{self.name}: lookup of event '{event.title}' ({event.id}) in '{calendar_name}' refused: {type(e).__name__}: {e}")
            raise ValueError(f"{self.name} refused the lookup of '{event.title}' in '{calendar_name}': {type(e).__name__}: {e}") from e
        resource = next((c for c in candidates if self._uid(c.vobject_instance.vevent) == event.id), None)
        if resource is None:
            info(f"{self.name}: event '{event.title}' ({event.id}) not among the {len(candidates)} event(s) of '{calendar_name}' "
                 f"around {event.start.isoformat()}")
            raise ValueError(f"'{event.title}' is no longer in '{calendar_name}'")
        if self._is_recurring(resource.vobject_instance.vevent):
            raise ValueError(f"'{event.title}' is a recurring event, which cannot be {verb} yet — only single events can")
        return resource

    def delete_event(self, calendar_name: str, event: Event) -> None:
        self._resource(calendar_name, event, "deleted").delete()
        info(f"{self.name}: event '{event.title}' ({event.id}) deleted from '{calendar_name}'")

    def move_event(self, calendar_name: str, event: Event, start: datetime, end: datetime, all_day: bool) -> None:
        """Rewrites DTSTART and DTEND (a DURATION gives way to a DTEND) and PUTs the event
        back; the library bumps SEQUENCE so invitees see the change. Timed bounds are written
        in UTC, the form `events()` reads back; all-day ones as plain dates."""
        resource = self._resource(calendar_name, event, "moved")
        vevent = resource.vobject_instance.vevent
        if 'duration' in vevent.contents:
            del vevent.contents['duration']
        if 'dtend' not in vevent.contents:
            vevent.add('dtend')
        # A TZID or VALUE=DATE parameter left over from the old value would contradict the new one.
        for key in ('dtstart', 'dtend'):
            vevent.contents[key][0].params.clear()
        if all_day:
            vevent.dtstart.value, vevent.dtend.value = start.date(), end.date()
        else:
            vevent.dtstart.value, vevent.dtend.value = start.astimezone(pytz.UTC), end.astimezone(pytz.UTC)
        resource.save()
        info(f"{self.name}: event '{event.title}' ({event.id}) in '{calendar_name}' moved to {start.isoformat()} - {end.isoformat()}"
             f"{' (all day)' if all_day else ''}")

    def check(self) -> bool:
        calendars = self._calendar_objects(refresh=True)
        if calendars:
            today = date.today()
            calendars[0].date_search(
                start=datetime.combine(today, datetime.min.time()),
                end=datetime.combine(today + timedelta(days=1), datetime.min.time())
            )
        return True
