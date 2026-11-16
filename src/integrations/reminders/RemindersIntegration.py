import threading
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Optional

from common.ConfigManager import config
from assistant.integration.Integration import Integration
from common.logging import error, info, trace

try:
    import EventKit
except ImportError:   # not a Mac, or pyobjc missing: check_health says so
    EventKit = None

ACCESS_HINT = ("Reminders access was not granted. Allow it for the app the server runs in (Terminal) under "
               "System Settings → Privacy & Security → Reminders, then restart the integration.")


@dataclass
class ReminderItem:
    title: str
    due: Optional[datetime | date] = None
    note: str = ""

    def render(self) -> str:
        parts = [self.title]
        if self.due is not None:
            parts.append(f"(due {self.due:%Y-%m-%d %H:%M})" if isinstance(self.due, datetime)
                         else f"(due {self.due:%Y-%m-%d})")
        if self.note:
            parts.append(f"— {self.note}")
        return " ".join(parts)


@dataclass
class ReminderList:
    name: str
    shared: bool = False
    items: list[ReminderItem] = field(default_factory=list)


class RemindersIntegration(Integration):
    """Apple Reminders: the family's to-do and shopping lists, read and edited.

    Lists what is open in every Reminders list, adds a to-do to any list or to the shopping
    list, and ticks one off when it is done.

    Goes through EventKit (pyobjc), not AppleScript: an AppleScript call launches and drives
    the Reminders app, half a second warm and seconds cold, while EventKit writes to the
    reminders store in-process in milliseconds. macOS asks once for Reminders access, for the
    process that started the server (Terminal); until the prompt is answered the first call
    waits. One EKEventStore serves the monitor's health checks and the commands, under a lock,
    since EventKit objects are not thread-safe. Reading is asynchronous in EventKit
    (fetchRemindersMatchingPredicate_completion_), so a fetch waits on the completion with a
    timeout; only open reminders are ever fetched, never the completed history. Two lists of
    the same name: the shared one wins, the shopping list being the one shared with the family."""
    sensitive = True
    mac_only = True

    ACCESS_TIMEOUT = 30   # seconds to wait for the user to answer macOS's access prompt
    FETCH_TIMEOUT = 10    # seconds to wait for the store to answer a fetch

    def __init__(self):
        super().__init__()
        self._store = None
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        return "Reminders"

    @property
    def config_keys(self) -> List[str]:
        return ["reminders"]

    def check_health(self) -> bool:
        with self._lock:
            store = self._event_store()
            self._calendar(store, self._shopping_list_name())
        return True

    @staticmethod
    def _shopping_list_name() -> str:
        return config.get("reminders.shopping_list_name")

    def add_reminder_item(self, item: str, list_name: str = None) -> str:
        try:
            with self._lock:
                if not list_name:
                    list_name = self._shopping_list_name()
                store = self._event_store()
                calendar = self._calendar(store, list_name)
                reminder = EventKit.EKReminder.reminderWithEventStore_(store)
                reminder.setTitle_(item)
                reminder.setCalendar_(calendar)
                ok, err = store.saveReminder_commit_error_(reminder, True, None)
            if not ok:
                message = err.localizedDescription() if err else "EventKit refused the save"
                error(f"Reminders: saving {item!r} to {list_name!r} failed: {message}")
                return f"Failed to add reminder: {message}"
            info(f"Reminders: added {item!r} to list {list_name!r}")
            return ""
        except Exception as e:
            error(f"Failed to add reminder: {e}")
            return f"Failed to add reminder: {e}"

    def open_reminders(self, list_name: str = None) -> list[ReminderList]:
        """The open reminders of every list, or of the one named: the shopping list first,
        the rest by name. Raises when the store cannot be read or the named list is missing."""
        with self._lock:
            store = self._event_store()
            if list_name:
                calendars = [self._calendar(store, list_name)]
            else:
                calendars = self._all_lists(store)
            reminders = self._fetch(store, calendars)
        by_title = {}
        for calendar in calendars:
            by_title.setdefault(calendar.title(), ReminderList(calendar.title(), _is_shared(calendar)))
        for reminder in reminders:
            target = by_title.setdefault(reminder.calendar().title(), ReminderList(reminder.calendar().title()))
            target.items.append(_item_of(reminder))
        shopping = self._shopping_list_name()
        lists = sorted(by_title.values(), key=lambda l: (l.name != shopping, l.name.casefold()))
        total = sum(len(l.items) for l in lists)
        if total:
            info(f"Reminders: {total} open item(s) across {len(lists)} list(s)")
        else:
            info(f"Reminders: nothing open in {len(lists)} list(s)")
        return lists

    def complete_reminder(self, title: str, list_name: str) -> str:
        """Ticks off the open reminder called `title` in the list `list_name`. The title is
        matched exactly, then regardless of case, then as the one item containing it; anything
        else is refused with the items to choose from."""
        try:
            with self._lock:
                store = self._event_store()
                calendar = self._calendar(store, list_name)
                reminders = self._fetch(store, [calendar])
                match = _match_title(reminders, title)
                match.setCompleted_(True)
                ok, err = store.saveReminder_commit_error_(match, True, None)
            if not ok:
                message = err.localizedDescription() if err else "EventKit refused the save"
                error(f"Reminders: completing {title!r} in {list_name!r} failed: {message}")
                return f"Failed to complete reminder: {message}"
            info(f"Reminders: completed {match.title()!r} in list {list_name!r}")
            return ""
        except Exception as e:
            error(f"Failed to complete reminder: {e}")
            return f"Failed to complete reminder: {e}"

    def _fetch(self, store, calendars) -> list:
        """The open reminders of `calendars`, fetched synchronously: EventKit answers on a
        queue of its own, so the completion is waited for, FETCH_TIMEOUT at most."""
        predicate = store.predicateForIncompleteRemindersWithDueDateStarting_ending_calendars_(None, None, calendars)
        answered = threading.Event()
        found = []

        def completion(reminders):
            found.extend(reminders or [])
            answered.set()

        trace(f"Reminders: fetching open reminders of {len(calendars)} list(s)")
        store.fetchRemindersMatchingPredicate_completion_(predicate, completion)
        if not answered.wait(self.FETCH_TIMEOUT):
            raise RuntimeError(f"the reminders store did not answer within {self.FETCH_TIMEOUT}s")
        trace(f"Reminders: fetched {len(found)} open reminder(s)")
        return found

    def _event_store(self):
        """The one EKEventStore, built on first use once macOS has granted Reminders access.
        Raises when EventKit is missing or access was denied, with what to do about it."""
        if self._store is not None:
            return self._store
        if EventKit is None:
            raise RuntimeError("EventKit is not available: Apple Reminders needs macOS and pyobjc")
        store = EventKit.EKEventStore.alloc().init()
        answered = threading.Event()
        verdict = {}

        def completion(granted, err):
            verdict["granted"] = granted
            verdict["error"] = err
            answered.set()

        trace("Reminders: requesting access to the reminders store")
        if hasattr(store, "requestFullAccessToRemindersWithCompletion_"):   # macOS 14+
            store.requestFullAccessToRemindersWithCompletion_(completion)
        else:
            store.requestAccessToEntityType_completion_(EventKit.EKEntityTypeReminder, completion)
        if not answered.wait(self.ACCESS_TIMEOUT):
            raise RuntimeError("Reminders access prompt not answered within "
                               f"{self.ACCESS_TIMEOUT}s — answer it on the server's screen")
        if not verdict.get("granted"):
            detail = verdict.get("error")
            raise RuntimeError(f"{ACCESS_HINT} ({detail.localizedDescription()})" if detail else ACCESS_HINT)
        info("Reminders: access granted, store ready")
        self._store = store
        return store

    @staticmethod
    def _all_lists(store) -> list:
        return list(store.calendarsForEntityType_(EventKit.EKEntityTypeReminder))

    @classmethod
    def _calendar(cls, store, list_name: str):
        """The reminders list called list_name, exactly or regardless of case; a shared one
        before an unshared namesake. Raises, naming the lists there are, when none has the name."""
        lists = cls._all_lists(store)
        candidates = [c for c in lists if c.title() == list_name]
        if not candidates:
            wanted = (list_name or "").strip().casefold()
            candidates = [c for c in lists if c.title().casefold() == wanted]
        if not candidates:
            names = sorted({c.title() for c in lists})
            raise LookupError(f"No Reminders list called {list_name!r}; the lists are: {', '.join(names)}")
        if len(candidates) > 1:
            trace(f"Reminders: {len(candidates)} lists called {list_name!r}, preferring a shared one")
            candidates.sort(key=lambda c: not _is_shared(c))
        return candidates[0]

    def _on_reset(self) -> None:
        self._store = None


def _is_shared(calendar) -> bool:
    sharing_status = getattr(calendar, "sharingStatus", None)   # EKCalendarSharingStatusNotShared is 0
    try:
        return bool(sharing_status and sharing_status())
    except Exception:
        return False


def _item_of(reminder) -> ReminderItem:
    note = reminder.notes()
    note = note.strip().split("\n", 1)[0] if isinstance(note, str) else ""
    return ReminderItem(reminder.title(), _due_of(reminder), note)


# NSDateComponentUndefined: the value of a component the due date does not carry (a day with no time).
_UNDEFINED = 0x7FFFFFFFFFFFFFFF


def _due_of(reminder) -> Optional[datetime | date]:
    """The due moment of a reminder: a date when the components carry no time of day, None
    when it has no due date at all."""
    components = reminder.dueDateComponents()
    if components is None:
        return None
    try:
        year, month, day = int(components.year()), int(components.month()), int(components.day())
        hour, minute = int(components.hour()), int(components.minute())
    except (TypeError, ValueError, AttributeError):
        return None
    if hour == _UNDEFINED or not 0 <= hour <= 23:
        return date(year, month, day)
    return datetime(year, month, day, hour, minute if 0 <= minute <= 59 else 0)


def _match_title(reminders, title: str):
    """The one reminder called `title`: exactly, else regardless of case, else the one whose
    title contains it. Raises naming the candidates when there are several, the open items
    when there are none."""
    wanted = title.strip()
    for matcher in (lambda t: t == wanted,
                    lambda t: t.casefold() == wanted.casefold(),
                    lambda t: wanted.casefold() in t.casefold()):
        found = [r for r in reminders if matcher(r.title())]
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            names = ", ".join(repr(r.title()) for r in found)
            raise LookupError(f"{len(found)} open reminders match {title!r}: {names} — name one exactly")
    if not reminders:
        raise LookupError(f"No open reminder called {title!r}: the list has nothing open")
    names = ", ".join(repr(r.title()) for r in reminders)
    raise LookupError(f"No open reminder called {title!r}; the open items are: {names}")


def render_lists(lists: list[ReminderList]) -> str:
    """The open reminders as the LLM receives them: one heading per list with its open count
    and whether it is shared, one line per item; one sentence when nothing is open anywhere."""
    if not any(l.items for l in lists):
        names = ", ".join(l.name for l in lists)
        return f"No open reminders in any list ({names})." if names else "No Reminders lists."
    blocks = []
    for l in lists:
        shared = ", shared" if l.shared else ""
        if not l.items:
            blocks.append(f"## {l.name} (empty{shared})")
            continue
        lines = [f"## {l.name} ({len(l.items)} open{shared})"]
        lines += [f"- {item.render()}" for item in l.items]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def add_reminder_item(item: str, list_name: str = None) -> str:
    return RemindersIntegration.get_instance().add_reminder_item(item, list_name)


def complete_reminder(title: str, list_name: str) -> str:
    return RemindersIntegration.get_instance().complete_reminder(title, list_name)


def list_open_reminders(list_name: str = None) -> str:
    try:
        return render_lists(RemindersIntegration.get_instance().open_reminders(list_name))
    except Exception as e:
        error(f"Failed to list reminders: {e}")
        return f"Failed to list reminders: {e}"
