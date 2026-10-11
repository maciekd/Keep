"""Keep's own list in Reminders.app through EventKit: a reminder with a time alarm or a geofence alarm is created, listed, completed and deleted.

AppleScript (what the shopping list goes through) can make a reminder with a due date but no
location alarm; EventKit (`EKReminder`, `EKAlarm`, `EKStructuredLocation`, via PyObjC) can
set both, and a time alarm set here rings on the user's iPhone through iCloud like one made
in the app. The frameworks are imported on first use so the module loads anywhere and the
unit tests run without them; access is the Reminders privacy permission of the app Keep runs
under (Terminal, say), asked for by macOS on the first call when it is not yet granted.

The id the LLM sees and hands back is the first eight hex digits of the item's UUID: short
enough for a chat line, unique within one list. A full UUID is accepted too.
"""
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from common.logging import info, trace, error

ID_LENGTH = 8
RECENTLY_COMPLETED_DAYS = 7
DUE_FORMAT = "%Y-%m-%d %H:%M"
DATE_FORMAT = "%Y-%m-%d"
ACCESS_PROMPT_TIMEOUT_SECONDS = 90
FETCH_TIMEOUT_SECONDS = 20
PROXIMITIES = ("arrive", "leave")
# EKAuthorizationStatus: 0 not determined, 1 restricted, 2 denied, 3 full access (authorized
# before macOS 14), 4 write only (macOS 14+) — listing needs 3.
AUTHORIZATION_HINTS = {
    1: "the Reminders permission is restricted on this Mac",
    2: "the Reminders permission was denied — allow it for the app Keep runs under (Terminal) in "
       "System Settings › Privacy & Security › Reminders",
    4: "the Reminders permission is write-only — switch it to full access in "
       "System Settings › Privacy & Security › Reminders",
}
# NSDateComponents reports an unset field as NSIntegerMax.
_UNDEFINED_COMPONENT = 2 ** 62


@dataclass(frozen=True)
class Place:
    """A geofence: the alarm fires when the phone arrives at or leaves the circle."""
    label: str
    latitude: float
    longitude: float
    radius: float
    proximity: str

    @classmethod
    def parse(cls, text: str) -> "Place":
        """`label; latitude; longitude; radius_m; arrive|leave` — `|` or `,` separate too."""
        parts = [p.strip() for p in re.split(r"[;|]", text)]
        if len(parts) < 5:
            parts = [p.strip() for p in text.split(",")]
        if len(parts) != 5:
            raise ValueError(f"place must be 'label; latitude; longitude; radius_m; arrive|leave', got '{text}'")
        label, lat, lon, radius, proximity = parts
        proximity = proximity.lower().strip("'\"")
        if proximity not in PROXIMITIES:
            raise ValueError(f"place must end with 'arrive' or 'leave', got '{proximity}'")
        try:
            latitude, longitude, radius_m = float(lat), float(lon), float(radius)
        except ValueError:
            raise ValueError(f"place needs numeric latitude, longitude and radius, got '{lat}', '{lon}', '{radius}'")
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError(f"place coordinates out of range: {latitude}, {longitude}")
        if radius_m <= 0:
            raise ValueError(f"place radius must be metres above zero, got {radius}")
        return cls(label.strip("'\""), latitude, longitude, radius_m, proximity)

    def describe(self) -> str:
        return f"{self.label} ({self.proximity}, {self.radius:.0f} m)"


def parse_due(text: str) -> tuple[Optional[datetime], bool]:
    """The due moment and whether it carries a time: 'YYYY-MM-DD HH:MM' does, 'YYYY-MM-DD'
    is a day with no alarm. Empty text is no due date."""
    text = text.strip().strip("'\"")
    if not text:
        return None, False
    for fmt, timed in ((DUE_FORMAT, True), (DATE_FORMAT, False)):
        try:
            return datetime.strptime(text, fmt), timed
        except ValueError:
            continue
    raise ValueError(f"due must be 'YYYY-MM-DD HH:MM' or 'YYYY-MM-DD', got '{text}'")


@dataclass(frozen=True)
class Reminder:
    id: str
    title: str
    due: Optional[datetime]
    timed: bool
    completed_at: Optional[datetime]
    notes: str = ""
    place: str = ""

    @property
    def completed(self) -> bool:
        return self.completed_at is not None

    def line(self, now: datetime) -> str:
        """One line for the LLM: state, id, title, due and what the alarm is tied to."""
        if self.completed:
            state = f"[done {self.completed_at.strftime(DUE_FORMAT)}]"
        elif self.due and self.due < now:
            state = "[open, OVERDUE]"
        else:
            state = "[open]"
        parts = [f"{state} {self.id} {self.title}"]
        if self.due:
            parts.append("due " + self.due.strftime(DUE_FORMAT if self.timed else DATE_FORMAT))
        if self.place:
            parts.append("at " + self.place)
        if self.notes:
            parts.append("notes: " + " ".join(self.notes.split()))
        return " — ".join(parts)


def short_id(identifier: str) -> str:
    return identifier[:ID_LENGTH].upper()


def format_listing(reminders: list[Reminder], list_name: str, now: datetime) -> str:
    """Open ones first (soonest due first, undated last), then the recently completed, latest
    first; an empty list says so."""
    open_items = sorted((r for r in reminders if not r.completed),
                        key=lambda r: (r.due is None, r.due or now, r.title.lower()))
    done = sorted((r for r in reminders if r.completed), key=lambda r: r.completed_at, reverse=True)
    if not open_items and not done:
        return f"The '{list_name}' list has no open reminders and none completed in the last {RECENTLY_COMPLETED_DAYS} days."
    lines = [f"Reminders in '{list_name}' ({len(open_items)} open, {len(done)} completed in the last "
             f"{RECENTLY_COMPLETED_DAYS} days):"]
    lines += [r.line(now) for r in open_items + done]
    return "\n".join(lines)


class RemindersAccessError(RuntimeError):
    """Reminders cannot be reached: no permission, no such reminder, or EventKit said no."""


class RemindersStore:
    """One list in Reminders.app; every method raises RemindersAccessError with the reason
    when it cannot do its job."""

    def __init__(self, list_name: str):
        self.list_name = list_name
        self._store = None
        self._lock = threading.Lock()

    # --- EventKit plumbing -------------------------------------------------------------

    @staticmethod
    def _eventkit():
        import EventKit
        return EventKit

    def _event_store(self):
        """The EKEventStore, built and authorized on first use. macOS asks the user the first
        time the app Keep runs under touches Reminders."""
        if self._store is not None:
            return self._store
        ek = self._eventkit()
        status = ek.EKEventStore.authorizationStatusForEntityType_(ek.EKEntityTypeReminder)
        store = ek.EKEventStore.alloc().init()
        if status == 0:
            info("Reminders access not decided yet — asking macOS (a permission prompt may show on the Mac)")
            answered = threading.Event()
            outcome = {}

            def _done(granted, err):
                outcome["granted"], outcome["error"] = bool(granted), err
                answered.set()

            if hasattr(store, "requestFullAccessToRemindersWithCompletion_"):
                store.requestFullAccessToRemindersWithCompletion_(_done)
            else:
                store.requestAccessToEntityType_completion_(ek.EKEntityTypeReminder, _done)
            if not answered.wait(ACCESS_PROMPT_TIMEOUT_SECONDS):
                raise RemindersAccessError("the Reminders permission prompt on the Mac was not answered in time")
            if not outcome.get("granted"):
                raise RemindersAccessError(AUTHORIZATION_HINTS[2])
            info("Reminders access granted")
        elif status in AUTHORIZATION_HINTS:
            raise RemindersAccessError(AUTHORIZATION_HINTS[status])
        self._store = store
        return store

    def _calendar(self, create: bool):
        """The EKCalendar of the list, made in the iCloud account on first use when asked to."""
        ek = self._eventkit()
        store = self._event_store()
        for calendar in store.calendarsForEntityType_(ek.EKEntityTypeReminder) or []:
            if calendar.title() == self.list_name:
                return calendar
        if not create:
            return None
        default = store.defaultCalendarForNewReminders()
        source = default.source() if default is not None else None
        if source is None:
            sources = list(store.sources() or [])
            caldav = [s for s in sources if s.sourceType() == ek.EKSourceTypeCalDAV]
            source = (caldav or sources or [None])[0]
        if source is None:
            raise RemindersAccessError("no Reminders account to create the list in")
        calendar = ek.EKCalendar.calendarForEntityType_eventStore_(ek.EKEntityTypeReminder, store)
        calendar.setTitle_(self.list_name)
        calendar.setSource_(source)
        ok, err = store.saveCalendar_commit_error_(calendar, True, None)
        if not ok:
            raise RemindersAccessError(f"could not create the '{self.list_name}' list: {err}")
        info(f"Created the Reminders list '{self.list_name}' in {source.title()}")
        return calendar

    def _fetch(self, predicate) -> list:
        store = self._event_store()
        done = threading.Event()
        found: list = []

        def _collect(items):
            found.extend(items or [])
            done.set()

        store.fetchRemindersMatchingPredicate_completion_(predicate, _collect)
        if not done.wait(FETCH_TIMEOUT_SECONDS):
            raise RemindersAccessError("Reminders did not answer in time")
        return found

    def _item(self, reminder_id: str):
        """The EKReminder in the list whose UUID starts with *reminder_id*."""
        wanted = reminder_id.strip().strip("'\"").upper()
        if not wanted:
            raise RemindersAccessError("a reminder id is needed")
        calendar = self._calendar(create=False)
        if calendar is None:
            raise RemindersAccessError(f"there is no '{self.list_name}' list in Reminders")
        store = self._event_store()
        items = self._fetch(store.predicateForRemindersInCalendars_([calendar]))
        matches = [i for i in items if i.calendarItemIdentifier().upper().startswith(wanted)]
        if not matches:
            raise RemindersAccessError(f"no reminder {reminder_id} in the '{self.list_name}' list — list_reminders shows the ids")
        if len(matches) > 1:
            raise RemindersAccessError(f"{len(matches)} reminders start with {reminder_id}; give more of the id")
        return matches[0]

    @staticmethod
    def _components_to_datetime(components) -> tuple[Optional[datetime], bool]:
        if components is None:
            return None, False
        fields = [components.year(), components.month(), components.day(), components.hour(), components.minute()]
        if any(f >= _UNDEFINED_COMPONENT or f < 0 for f in fields[:3]):
            return None, False
        year, month, day, hour, minute = fields
        timed = 0 <= hour < _UNDEFINED_COMPONENT and 0 <= minute < _UNDEFINED_COMPONENT
        try:
            return datetime(year, month, day, hour if timed else 0, minute if timed else 0), timed
        except ValueError:
            return None, False

    @classmethod
    def _to_reminder(cls, item) -> Reminder:
        due, timed = cls._components_to_datetime(item.dueDateComponents())
        completed_at = None
        if item.isCompleted() and item.completionDate() is not None:
            completed_at = datetime.fromtimestamp(item.completionDate().timeIntervalSince1970())
        place = ""
        for alarm in item.alarms() or []:
            location = alarm.structuredLocation()
            if location is not None:
                proximity = "leave" if alarm.proximity() == 2 else "arrive"
                place = f"{location.title()} ({proximity})"
                break
        return Reminder(short_id(item.calendarItemIdentifier()), item.title() or "", due, timed, completed_at,
                        item.notes() or "", place)

    # --- the operations ------------------------------------------------------------------

    def create(self, title: str, due: str = "", notes: str = "", place: str = "") -> Reminder:
        """A new reminder in the list; a timed due date becomes an alarm at that time, a place a
        geofence alarm. Raises ValueError for a malformed due or place."""
        title = title.strip()
        if not title:
            raise ValueError("a reminder needs a title")
        due_at, timed = parse_due(due)
        geofence = Place.parse(place) if place.strip() else None
        ek = self._eventkit()
        import Foundation
        with self._lock:
            store = self._event_store()
            calendar = self._calendar(create=True)
            reminder = ek.EKReminder.reminderWithEventStore_(store)
            reminder.setTitle_(title)
            reminder.setCalendar_(calendar)
            if notes.strip():
                reminder.setNotes_(notes.strip())
            if due_at is not None:
                components = Foundation.NSDateComponents.alloc().init()
                components.setCalendar_(Foundation.NSCalendar.currentCalendar())
                components.setYear_(due_at.year)
                components.setMonth_(due_at.month)
                components.setDay_(due_at.day)
                if timed:
                    components.setHour_(due_at.hour)
                    components.setMinute_(due_at.minute)
                reminder.setDueDateComponents_(components)
                if timed:
                    when = Foundation.NSDate.dateWithTimeIntervalSince1970_(due_at.timestamp())
                    reminder.addAlarm_(ek.EKAlarm.alarmWithAbsoluteDate_(when))
            if geofence is not None:
                import CoreLocation
                location = ek.EKStructuredLocation.locationWithTitle_(geofence.label)
                location.setGeoLocation_(CoreLocation.CLLocation.alloc().initWithLatitude_longitude_(
                    geofence.latitude, geofence.longitude))
                location.setRadius_(geofence.radius)
                alarm = ek.EKAlarm.alloc().init()
                alarm.setStructuredLocation_(location)
                alarm.setProximity_(ek.EKAlarmProximityLeave if geofence.proximity == "leave" else ek.EKAlarmProximityEnter)
                reminder.addAlarm_(alarm)
            ok, err = store.saveReminder_commit_error_(reminder, True, None)
            if not ok:
                raise RemindersAccessError(f"Reminders refused the reminder: {err}")
            made = self._to_reminder(reminder)
        info(f"Created reminder {made.id} '{title}' in '{self.list_name}'"
             + (f", due {due_at.strftime(DUE_FORMAT if timed else DATE_FORMAT)}" if due_at else "")
             + (f", at {geofence.describe()}" if geofence else ""))
        return made

    def list(self, now: Optional[datetime] = None) -> list[Reminder]:
        """The open reminders of the list and those completed in the last RECENTLY_COMPLETED_DAYS."""
        now = now or datetime.now()
        with self._lock:
            calendar = self._calendar(create=False)
            if calendar is None:
                info(f"list_reminders: no '{self.list_name}' list in Reminders yet")
                return []
            items = self._fetch(self._event_store().predicateForRemindersInCalendars_([calendar]))
            reminders = [self._to_reminder(item) for item in items]
        since = now - timedelta(days=RECENTLY_COMPLETED_DAYS)
        kept = [r for r in reminders if not r.completed or r.completed_at >= since]
        trace(f"list_reminders: {len(reminders)} in '{self.list_name}', {len(kept)} open or recently completed")
        return kept

    def complete(self, reminder_id: str) -> Reminder:
        with self._lock:
            item = self._item(reminder_id)
            if item.isCompleted():
                raise RemindersAccessError(f"reminder {short_id(item.calendarItemIdentifier())} '{item.title()}' is already completed")
            item.setCompleted_(True)
            ok, err = self._event_store().saveReminder_commit_error_(item, True, None)
            if not ok:
                raise RemindersAccessError(f"Reminders refused the change: {err}")
            done = self._to_reminder(item)
        info(f"Completed reminder {done.id} '{done.title}'")
        return done

    def delete(self, reminder_id: str) -> Reminder:
        with self._lock:
            item = self._item(reminder_id)
            gone = self._to_reminder(item)
            ok, err = self._event_store().removeReminder_commit_error_(item, True, None)
            if not ok:
                raise RemindersAccessError(f"Reminders refused the deletion: {err}")
        info(f"Deleted reminder {gone.id} '{gone.title}'")
        return gone
