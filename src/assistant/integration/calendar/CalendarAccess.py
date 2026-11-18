"""The one calendar the prompt section, the commands and the memory curator read: every calendar
account of every enabled and configured integration, presented as one.

`fetch` is the CALENDAR EVENTS COMING WEEK text — one header line per account naming the
calendars it shows and the ones left out, then one chronological list of the shown calendars'
events across accounts, each line tagged with its calendar when more than one is shown; an
account that fails is named as such while the others still show, and the whole week is
EVENTS_UNAVAILABLE only when none could be read (never NO_EVENTS, which would read as an empty
week). `coming_week` is the same for the prompt section, except that it raises
CalendarUnavailable then: the section retries and keeps the week it has rather than trading it
for a placeholder over an outage that passes. `load` is the `load_calendar` command, `create` the `create_event` one, `delete` and
`move` the `delete_event` / `move_event` ones; all name a calendar across accounts,
`Account/Name` when two accounts have one by that name. An existing event is named the way the
user names it, by title and date — `_find` reads that day and insists on one match — and goes
back to the account that read it, whose handle its `id` is."""
from datetime import date, datetime, timedelta
from typing import Optional

import pytz

from common.ConfigManager import config
from common.logging import info, error, trace
from .CalendarAccount import CalendarAccount, CalendarInfo
from .CalendarSource import CalendarSource
from .Event import Event, covers, render, when

# How far ahead the prompt's calendar section and `load_calendar` look.
CALENDAR_DAYS_AHEAD = 8
# What the prompt carries when the fetch failed.
EVENTS_UNAVAILABLE = "(Calendar events could not be loaded)"
NOT_CONFIGURED = "(Calendar not configured)"
NO_ACCOUNT_HINT = ("No calendar account is configured. Add one under caldav.accounts (name, preset icloud or a url, "
                   "username, password) — see Setting things up: list_integrations, config_help(\"caldav\"), config_request.")


class CalendarUnavailable(Exception):
    """None of the calendar accounts could be read."""


class CalendarAccess:
    # Which accounts the last read could not reach, and with what — the process's, since an
    # access is built per read.
    _reported_failing: frozenset = frozenset()

    def __init__(self, accounts: list[CalendarAccount], starting: bool = False):
        self._accounts = list(accounts)
        # True while the launcher has not registered the integrations yet (the prompt sections
        # start before them): the section polls again soon instead of showing "not configured".
        self.starting = starting

    @classmethod
    def current(cls) -> "CalendarAccess":
        """The accounts of every registered CalendarSource whose config keys are set — resolved
        per call, so a restarted or newly configured integration is what the next read uses."""
        from assistant.integration.IntegrationMonitor import IntegrationMonitor, DisabledIntegration, UnavailableIntegration
        monitor = IntegrationMonitor.get_instance()
        accounts: list[CalendarAccount] = []
        sources = monitor.of_type(CalendarSource)
        for integration in sources:
            missing = integration.missing_config_keys()
            if missing:
                trace(f"Calendar: {integration.name} is enabled but not configured ({', '.join(missing)}), no accounts from it")
                continue
            accounts.extend(integration.calendar_accounts())
        if accounts:
            return cls(accounts)
        placeholders = [p for p in monitor.of_type(DisabledIntegration) + monitor.of_type(UnavailableIntegration)
                        if issubclass(p.cls, CalendarSource)]
        if not sources and not placeholders:
            trace("Calendar: no calendar integration registered yet — the integrations are still starting")
            return cls([], starting=True)
        trace("Calendar: no calendar account configured")
        return cls([])

    @property
    def accounts(self) -> list[CalendarAccount]:
        return list(self._accounts)

    def coming_week(self) -> str:
        """The prompt section's week. Raises CalendarUnavailable when no account could be read."""
        start = date.today()
        return self._read(start, start + timedelta(days=CALENDAR_DAYS_AHEAD))

    def fetch(self, start_date: date, end_date: date) -> str:
        """One header line per account — the calendars shown and the ones left out — then the
        events of the shown ones in one list."""
        try:
            return self._read(start_date, end_date)
        except CalendarUnavailable as e:
            error(f"Calendar: {e}, the reader gets '{EVENTS_UNAVAILABLE}'")
            return EVENTS_UNAVAILABLE

    def _read(self, start_date: date, end_date: date) -> str:
        if not self._accounts:
            info(f"Calendar: no account configured, the prompt gets '{NOT_CONFIGURED}'")
            return NOT_CONFIGURED
        headers, events, failures, shown_total = [], [], [], 0
        for account in self._accounts:
            try:
                header, shown = self._header(account)
                events.extend(account.events([c.name for c in shown], start_date, end_date))
                headers.append(header)
                shown_total += sum(1 for c in shown if c.holds_events)
            except Exception as e:
                failures.append((account, e))
                headers.append(f"({account.name}: calendar could not be loaded)")
        described = "; ".join(f"{account.name}: {type(e).__name__}: {e}" for account, e in failures)
        if len(failures) == len(self._accounts):
            # Not logged here: whoever catches it knows whether this is an attempt or the outcome.
            raise CalendarUnavailable(f"no calendar account could be read ({described})") from failures[-1][1]
        self._report(failures, described)
        trace(f"Calendar: {len(events)} event(s) from {len(self._accounts) - len(failures)} account(s), {shown_total} calendar(s) shown")
        return "\n".join(headers) + "\n\n" + render(events, start_date, end_date, tag_calendars=shown_total > 1)

    def _report(self, failures: list[tuple[CalendarAccount, Exception]], described: str):
        """Some accounts read and some not: an error when which ones fail, or how, changes —
        the week is read every few minutes, and an expired sign-in lasts until somebody signs in."""
        failing = frozenset((account.name, type(e).__name__) for account, e in failures)
        text = f"Calendar: {len(failures)} of {len(self._accounts)} account(s) could not be read ({described})"
        if failing == CalendarAccess._reported_failing:
            if failing:
                info(f"{text} — as already reported")
            return
        CalendarAccess._reported_failing = failing
        if failing:
            error(text, outside=True)
        else:
            info(f"Calendar: every account can be read again ({len(self._accounts)})")

    @staticmethod
    def _label(account: CalendarAccount) -> str:
        return f"{account.name} ({account.service})" if account.service else account.name

    @classmethod
    def _header(cls, account: CalendarAccount) -> tuple[str, list[CalendarInfo]]:
        """`Family (CalDAV): Home, Kids — not shown: Birthdays`, and the calendars whose events are shown."""
        calendars = account.calendars()
        if account.shown_calendars:
            shown, missing = [], []
            for wanted in account.shown_calendars:
                calendar = cls._match(account, calendars, wanted)
                if calendar is None:
                    missing.append(wanted)
                elif calendar not in shown:
                    shown.append(calendar)
            if missing:
                info(f"{account.name}: shown_calendars names calendars the account does not have: {missing}. "
                     f"Available: {[c.name for c in calendars]}")
        else:
            trace(f"{account.name}: shown_calendars not set, showing all {len(calendars)} calendars")
            shown = calendars
        # A Reminders list among the shown ones is searched (it holds no events) but not named.
        named = [c for c in calendars if c.holds_events]
        shown_labels = [c.name for c in named if c in shown]
        hidden_labels = [c.name for c in named if c not in shown]
        trace(f"{account.name}: calendars shown: {shown_labels}, not shown: {hidden_labels}")
        header = f"{cls._label(account)}: " + (", ".join(shown_labels) or "none shown")
        if hidden_labels:
            header += " — not shown: " + ", ".join(hidden_labels)
        return header, shown

    @staticmethod
    def _match(account: CalendarAccount, calendars: list[CalendarInfo], wanted: str) -> Optional[CalendarInfo]:
        """The calendar `wanted` names among `calendars`: a listed name whatever its case, else an
        alias the account resolves itself (Google's "primary"), else None."""
        lowered = wanted.strip().lower()
        for calendar in calendars:
            if calendar.name.lower() == lowered:
                return calendar
        listed = account.listed_name(wanted)
        if listed is not None:
            trace(f"{account.name}: '{wanted}' is the account's alias of calendar '{listed}'")
            return next((c for c in calendars if c.name == listed), None)
        return None

    def _resolve(self, name: str) -> tuple[Optional[tuple[CalendarAccount, CalendarInfo]], str]:
        """The account and calendar `name` means — `Kids`, or `Family/Kids` when two accounts
        have a Kids — or None and the reason; the reason names what exists."""
        wanted = name.strip().strip("'\"")
        account_part, _, calendar_part = wanted.rpartition("/")
        found: list[tuple[CalendarAccount, CalendarInfo]] = []
        available: list[str] = []
        for account in self._accounts:
            if account_part and account.name.lower() != account_part.strip().lower():
                continue
            calendars = account.calendars()
            for calendar in calendars:
                if calendar.holds_events:
                    available.append(f"{account.name}/{calendar.name}" if len(self._accounts) > 1 else calendar.name)
            matched = self._match(account, calendars, calendar_part if account_part else wanted)
            if matched is not None:
                found.append((account, matched))
        if not found:
            info(f"Calendar: no calendar named '{wanted}'")
            return None, f"no calendar named '{wanted}'. Available: {', '.join(available) or 'none'}"
        if len(found) > 1:
            choices = ", ".join(f"{a.name}/{c.name}" for a, c in found)
            info(f"Calendar: '{wanted}' is ambiguous: {choices}")
            return None, f"'{wanted}' is in several accounts — name one: {choices}"
        return found[0], ""

    def load(self, names: list[str]) -> str:
        """The coming week of the named calendars, each under its own header; the `load_calendar` command."""
        if not self._accounts:
            raise ValueError(NO_ACCOUNT_HINT)
        start_date = date.today()
        end_date = start_date + timedelta(days=CALENDAR_DAYS_AHEAD)
        parts = []
        for name in names:
            found, reason = self._resolve(name)
            if found is None:
                parts.append(f"load_calendar failed: {reason}")
                continue
            account, calendar = found
            info(f"Loading calendar '{calendar.name}' of {account.name} {start_date} to {end_date}")
            label = f"{account.name}/{calendar.name}" if len(self._accounts) > 1 else calendar.name
            parts.append(f"Calendar '{label}', {start_date} to {end_date}:\n"
                         + render(account.events([calendar.name], start_date, end_date), start_date, end_date))
        return "\n\n".join(parts)

    def _candidates(self, day: date, calendar_name: str) -> list[tuple[CalendarAccount, Event]]:
        """The events on `day` of the named calendar, or of every shown calendar of every account
        when none is named (the ones the prompt lists, so what the user sees is what is found);
        read with a day of margin each side since the backends search in UTC."""
        local_tz = pytz.timezone(config.get_home_timezone())
        start, end = day - timedelta(days=1), day + timedelta(days=2)
        read: list[tuple[CalendarAccount, Event]] = []
        if calendar_name.strip():
            found, reason = self._resolve(calendar_name)
            if found is None:
                raise ValueError(reason)
            account, calendar = found
            read.extend((account, e) for e in account.events([calendar.name], start, end))
        else:
            for account in self._accounts:
                _, shown = self._header(account)
                read.extend((account, e) for e in account.events([c.name for c in shown], start, end))
        return [(account, e) for account, e in read if covers(e, day, local_tz)]

    def _find(self, title: str, date_str: str, calendar_name: str = "") -> tuple[CalendarAccount, Event]:
        """The one event titled `title` on the date: the title whatever its case, else the one
        event whose title contains it. Raises ValueError naming the day's events when none or
        several match."""
        if not self._accounts:
            raise ValueError(NO_ACCOUNT_HINT)
        day = date.fromisoformat(date_str.strip())
        local_tz = pytz.timezone(config.get_home_timezone())
        wanted = title.strip().strip("'\"").lower()
        candidates = self._candidates(day, calendar_name)
        matches = [(a, e) for a, e in candidates if e.title.strip().lower() == wanted]
        if not matches:
            matches = [(a, e) for a, e in candidates if wanted and wanted in e.title.lower()]
        listed = "; ".join(f"'{e.title}' ({when(e, local_tz)}, {a.name}/{e.calendar})" for a, e in matches or candidates)
        where = f" in {calendar_name.strip()}" if calendar_name.strip() else ""
        if not matches:
            info(f"Calendar: no event titled '{title}' on {day}{where} among {len(candidates)} candidate(s)")
            raise ValueError(f"no event titled '{title}' on {day}{where}. That day holds: {listed or 'nothing'}")
        if len(matches) > 1:
            info(f"Calendar: '{title}' on {day}{where} is ambiguous: {listed}")
            raise ValueError(f"'{title}' on {day}{where} matches several events — name one exactly: {listed}")
        account, event = matches[0]
        trace(f"Calendar: '{title}' on {day}{where} is '{event.title}' ({when(event, local_tz)}) of {account.name}/{event.calendar}")
        return account, event

    def delete(self, title: str, date_str: str, calendar_name: str = "") -> str:
        """Deletes the one event titled `title` on the date (the `delete_event` command) and
        says which one went; raises ValueError when it is not one event or the account refuses."""
        account, event = self._find(title, date_str, calendar_name)
        local_tz = pytz.timezone(config.get_home_timezone())
        account.delete_event(event.calendar, event)
        info(f"Deleted calendar event '{event.title}' ({when(event, local_tz)}) from {account.name}/{event.calendar}")
        return f"Deleted '{event.title}' ({when(event, local_tz)}) from {account.name}/{event.calendar}"

    def move(self, title: str, date_str: str, new_date: str, start_time: str = "", end_time: str = "",
             calendar_name: str = "") -> str:
        """Moves the one event titled `title` on the date to `new_date` (the `move_event`
        command): at the given times, or without them at its own time of day and length
        (an all-day event stays all-day). Says what moved where; raises ValueError when it is
        not one event or the account refuses."""
        account, event = self._find(title, date_str, calendar_name)
        local_tz = pytz.timezone(config.get_home_timezone())
        day = date.fromisoformat(new_date.strip())
        if start_time.strip() or end_time.strip():
            if not (start_time.strip() and end_time.strip()):
                raise ValueError("a move with times needs both the start and the end time")
            start = local_tz.localize(datetime.strptime(f"{day} {start_time.strip()}", "%Y-%m-%d %H:%M"))
            end = local_tz.localize(datetime.strptime(f"{day} {end_time.strip()}", "%Y-%m-%d %H:%M"))
            all_day = False
        elif event.all_day:
            days = max((event.end - event.start).days, 1)
            start = datetime.combine(day, datetime.min.time()).replace(tzinfo=pytz.UTC)
            end, all_day = start + timedelta(days=days), True
        else:
            local_start = event.start.astimezone(local_tz)
            start = local_tz.localize(datetime.combine(day, local_start.time()))
            end, all_day = start + (event.end - event.start), False
        moved = Event(start, end, event.title, event.calendar, event.account, all_day, event.id, event.recurring)
        account.move_event(event.calendar, event, start, end, all_day)
        info(f"Moved calendar event '{event.title}' from {when(event, local_tz)} to {when(moved, local_tz)} "
             f"in {account.name}/{event.calendar}")
        return f"Moved '{event.title}' from {when(event, local_tz)} to {when(moved, local_tz)} in {account.name}/{event.calendar}"

    def create(self, title: str, date_str: str, start_time: str, end_time: str, calendar_name: str = "") -> str:
        """An event in the named calendar, else the first account's default calendar (matched
        to a listed name whatever its case, or handed over as an alias the account knows);
        returns the `Account/Calendar` it went to and raises ValueError when there is no
        account or no such calendar."""
        if not self._accounts:
            raise ValueError(NO_ACCOUNT_HINT)
        local_tz = pytz.timezone(config.get_home_timezone())
        dt_start = local_tz.localize(datetime.strptime(f"{date_str} {start_time}", "%Y-%m-%d %H:%M"))
        dt_end = local_tz.localize(datetime.strptime(f"{date_str} {end_time}", "%Y-%m-%d %H:%M"))
        if calendar_name.strip():
            found, reason = self._resolve(calendar_name)
            if found is None:
                raise ValueError(reason)
            account, calendar = found
            target_name = calendar.name
        else:
            account = self._accounts[0]
            target_name = account.default_calendar
            calendars = account.calendars()
            if not target_name:
                target_name = next((c.name for c in calendars if c.holds_events), "")
                trace(f"{account.name}: no default calendar configured, using '{target_name}'")
            listed = next((c.name for c in calendars if c.name.lower() == target_name.lower()), None)
            if listed is None:
                # Not a listed name: an alias the account resolves itself (Google's "primary"),
                # or unknown — the account's create_event raises for that, naming what exists.
                trace(f"{account.name}: default calendar '{target_name}' is not a listed name, the account resolves it")
            target_name = listed or target_name
        account.create_event(target_name, Event(dt_start, dt_end, title, calendar=target_name, account=account.name))
        info(f"Created calendar event: {title} on {date_str} {start_time}-{end_time} in {account.name}/{target_name}")
        return f"{account.name}/{target_name}"
