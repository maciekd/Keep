import logging
from datetime import datetime, timedelta, date
from typing import List

import caldav
import pytz

from common.ConfigManager import config
from ..Integration import Integration
from common.logging import info, error, trace

# How far ahead the prompt's calendar section and `load_calendar` look.
CALENDAR_DAYS_AHEAD = 8

NO_EVENTS = "(No upcoming events)"
# What the prompt carries when the fetch failed — never NO_EVENTS, which would read as an empty week.
EVENTS_UNAVAILABLE = "(Calendar events could not be loaded)"


class CalDAVIntegration(Integration):
    """Reads and creates calendar events on the iCloud CalDAV server."""

    @property
    def name(self) -> str:
        return "CalDAV"

    @property
    def config_keys(self) -> List[str]:
        return ["calendar"]

    def check_health(self) -> bool:
        if not config.get('calendar.username') or not config.get('calendar.password'):
            return False
        calendars = self._calendars()
        if calendars:
            today = date.today()
            calendars[0].date_search(
                start=datetime.combine(today, datetime.min.time()),
                end=datetime.combine(today + timedelta(days=1), datetime.min.time())
            )
        return True

    @staticmethod
    def _calendars() -> list:
        """Every calendar of the configured iCloud account."""
        logging.getLogger("root").setLevel(logging.CRITICAL)
        username = config.get('calendar.username', "")
        password = config.get('calendar.password', "")
        if not username or not password:
            raise ValueError("Calendar username or password not found in config")
        client = caldav.DAVClient('https://caldav.icloud.com/', username=username, password=password)
        return client.principal().calendars()

    @staticmethod
    def shown_calendar_names() -> list[str]:
        """`calendar.shown_calendars`: the calendars the prompt carries. Empty means all of them."""
        # An empty default, not None: ConfigManager.get raises KeyError for an absent key without one.
        configured = config.get('calendar.shown_calendars', [])
        if isinstance(configured, str):
            configured = configured.split(",")
        if not isinstance(configured, (list, tuple)):
            return []
        return [str(name).strip() for name in configured if str(name).strip()]

    @staticmethod
    def _holds_events(calendar) -> bool:
        """False for a Reminders list, which iCloud serves next to the calendars."""
        try:
            return "VEVENT" in calendar.get_supported_components()
        except Exception as e:
            trace(f"Calendar '{calendar.name}': supported components unknown ({e}), treating it as a calendar")
            return True

    @staticmethod
    def _render_events(calendars: list, start_date: date, end_date: date) -> str:
        def extract_time(ev, key) -> datetime:
            time_value = ev.vobject_instance.vevent.contents[key][0].value
            if isinstance(time_value, str):
                dt = datetime.strptime(time_value, "%Y-%m-%dT%H:%M:%SZ")
                return dt.replace(tzinfo=pytz.UTC)
            return time_value

        def make_sure_is_datetime(dt) -> datetime:
            if isinstance(dt, date) and not isinstance(dt, datetime):
                return datetime.combine(dt, datetime.min.time()).replace(tzinfo=pytz.UTC)
            if isinstance(dt, datetime) and dt.tzinfo is None:
                return dt.replace(tzinfo=pytz.UTC)
            return dt

        parsed_events = []
        for calendar in calendars:
            events = calendar.date_search(
                start=datetime.combine(start_date, datetime.min.time()),
                end=datetime.combine(end_date, datetime.min.time())
            )
            for event in events:
                time_start = extract_time(event, 'dtstart')
                time_end = extract_time(event, 'dtend')
                title = event.vobject_instance.vevent.summary.value
                parsed_events.append((make_sure_is_datetime(time_start), make_sure_is_datetime(time_end), title))

        parsed_events.sort(key=lambda x: x[0])

        local_tz = pytz.timezone(config.get_home_timezone())
        start_dt = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=pytz.UTC)
        end_dt = datetime.combine(end_date, datetime.min.time()).replace(tzinfo=pytz.UTC)

        lines = []
        for time_start, time_end, title in parsed_events:
            if start_dt <= time_start <= end_dt or start_dt <= time_end <= end_dt:
                local_start = time_start.astimezone(local_tz)
                local_end = time_end.astimezone(local_tz)
                title = title.replace("\n", " ")
                if local_start.date() == local_end.date():
                    lines.append(f"{local_start.strftime('%Y-%m-%d')} {local_start.strftime('%H:%M')} - {local_end.strftime('%H:%M')}: {title}")
                else:
                    lines.append(f"{local_start.strftime('%Y-%m-%d %H:%M')} - {local_end.strftime('%Y-%m-%d %H:%M')}: {title}")

        return "\n".join(lines) if lines else NO_EVENTS

    @staticmethod
    def fetch_calendar_events(start_date: date, end_date: date) -> str:
        """The events of the shown calendars, followed by the names of the ones left out."""
        try:
            if not config.get('calendar.username', "") or not config.get('calendar.password', ""):
                return "(Calendar not configured)"

            calendars = CalDAVIntegration._calendars()
            shown_names = {name.lower() for name in CalDAVIntegration.shown_calendar_names()}
            if not shown_names:
                trace(f"calendar.shown_calendars not set, showing all {len(calendars)} calendars")
                return CalDAVIntegration._render_events(calendars, start_date, end_date)

            shown = [c for c in calendars if str(c.name).lower() in shown_names]
            missing = shown_names - {str(c.name).lower() for c in shown}
            if missing:
                info(f"calendar.shown_calendars names calendars the account does not have: {sorted(missing)}. "
                     f"Available: {[c.name for c in calendars]}")
            hidden = [c.name for c in calendars
                      if str(c.name).lower() not in shown_names and CalDAVIntegration._holds_events(c)]
            trace(f"Calendars shown: {[c.name for c in shown]}, not shown: {hidden}")

            summary = CalDAVIntegration._render_events(shown, start_date, end_date)
            if hidden:
                summary += "\n\nCalendars not shown here (`load_calendar(name)` loads one): " + ", ".join(hidden)
            return summary

        except Exception as e:
            error(f"Error fetching calendar events, the prompt gets '{EVENTS_UNAVAILABLE}': {type(e).__name__}: {e}")
            return EVENTS_UNAVAILABLE

    @staticmethod
    def load_calendars(names: list[str]) -> str:
        """The coming week of the named calendars, each under its own header; the `load_calendar` command."""
        start_date = date.today()
        end_date = start_date + timedelta(days=CALENDAR_DAYS_AHEAD)
        calendars = CalDAVIntegration._calendars()
        by_name = {str(c.name).lower(): c for c in calendars}
        parts = []
        for name in names:
            calendar = by_name.get(name.strip().strip("'\"").lower())
            if calendar is None:
                info(f"load_calendar: no calendar named '{name}'")
                available = ", ".join(c.name for c in calendars if CalDAVIntegration._holds_events(c)) or "none"
                parts.append(f"load_calendar failed: no calendar named '{name}'. Available: {available}")
                continue
            info(f"Loading calendar '{calendar.name}' {start_date} to {end_date}")
            parts.append(f"Calendar '{calendar.name}', {start_date} to {end_date}:\n"
                         + CalDAVIntegration._render_events([calendar], start_date, end_date))
        return "\n\n".join(parts)

    @staticmethod
    def create_calendar_event(title: str, date_str: str, start_time: str, end_time: str) -> None:
        local_tz = pytz.timezone(config.get_home_timezone())
        dt_start = local_tz.localize(datetime.strptime(f"{date_str} {start_time}", "%Y-%m-%d %H:%M"))
        dt_end = local_tz.localize(datetime.strptime(f"{date_str} {end_time}", "%Y-%m-%d %H:%M"))

        calendars = CalDAVIntegration._calendars()
        target_name = config.get('calendar.default_calendar', 'Home')
        calendar = next((cal for cal in calendars if cal.name == target_name), None)
        if calendar is None:
            raise ValueError(f"Calendar '{target_name}' not found. Available: {[cal.name for cal in calendars]}")

        calendar.save_event(dtstart=dt_start, dtend=dt_end, summary=title)
        info(f"Created calendar event: {title} on {date_str} {start_time}-{end_time}")


fetch_calendar_events = CalDAVIntegration.fetch_calendar_events
create_calendar_event = CalDAVIntegration.create_calendar_event
