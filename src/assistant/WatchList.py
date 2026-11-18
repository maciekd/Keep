"""memory/watchlist/ — everything the assistant is keeping an eye on, one small file per open item.

An item is something with a state that will change or end: an order on its way, an invoice to
pay, an appointment, an open decision, a reminder. It answers to its name, like a procedure
does. Its note is the item's *current* state and is replaced when the item is saved again, so
an item lives in one place instead of as a trail of remembered sentences. Deleting an item
removes its file; every replaced or deleted version is copied to `log/memory/watchlist/`
first, which is a debugging trail the LLM never reads.

An item may carry a due date, which may name a time (`due`: "2026-11-08" or "2026-11-08 17:00");
the due is a moment the assistant acts at — `due_moment` reads it, a bare date as that morning
(`BARE_DATE_HOUR`) — and the item is due for a poll from then until it is looked at again (checked,
re-dated or deleted), so a reminder the check failed to deliver comes back on the next one instead
of being lost, and an item that only waits for its date is looked at when the date comes.
An item may carry a check cadence (`every`: "1d", "6h", "weekly" — free text the LLM reads)
and always carries when it was last looked at (`checked`, with the time of day): `updated`
moves when the state changes, `checked` whenever someone looks (the assistant's `save_watch` is a
look; the curator's nightly rewrite is not). A check that finds nothing
stamps `checked` with `checked_watch` and leaves the note alone, so the note never turns into
a trail again. `interval` and `stamp` read the cadence and the stamp: `due_for_poll` is the
code's arithmetic on the two (the autonomous check gets only the items it names in full, which
keeps that prompt small; it passes half its own interval as the tolerance, so an item is polled
by the check that falls nearest its cadence), `pending` the moments the manager's due watcher wakes
for, and the TUI draws when the next check falls from them.

The `save_watch`, `checked_watch` and `delete_watch` commands and the memory curator's command
block all go through `run`. The prompt's watch-list section is only an index (`render_index`: a
line per item); `load_watch` returns whole items (`load`); the curator, which works through the
whole list, gets `render`, and the autonomous check `render_due` — the items whose poll is due.
The TUI's Watch list tab draws the items themselves (`items`).
"""
import math
import re
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from common.logging import info, trace
from common.Paths import MEMORY_DIR, MEMORY_LOG_DIR
from common.vault.Redactor import redact_for_storage

WATCHLIST_DIR = MEMORY_DIR / "watchlist"
LOG_DIR = MEMORY_LOG_DIR / "watchlist"
COMMANDS = ("save_watch", "checked_watch", "delete_watch")
# An item nobody has touched this long, or whose due date passed this long ago, is marked
# STALE in the prompt: the assistant and the curator then know which ones to look at.
STALE_AFTER_DAYS = 14
OVERDUE_AFTER_DAYS = 2
# The hour a due date without a time is acted at.
BARE_DATE_HOUR = 9
EMPTY = "Nothing is being watched."
NOTHING_DUE = "No item is due for a poll."
# The header of a loaded item, as `load_watch(name)` returns it.
HEADER_PREFIX = "# Watch item: "
# How much of an item's note its index line carries.
INDEX_NOTE_CHARS = 48
# The longest note a save may write (about 3 KB of text): a note is the item's current state, and one
# that keeps every past finding grows by each check and is rewritten whole on every save.
NOTE_WORD_LIMIT = 500

lock = threading.RLock()


@dataclass
class Item:
    name: str
    note: str = ""
    due: str = ""
    created: str = ""
    updated: str = ""
    every: str = ""
    checked: str = ""

    def text(self, closing: str = "") -> str:
        header = [f"# {self.name}", f"Due: {self.due or '-'}", f"Every: {self.every or '-'}",
                  f"Created: {self.created}", f"Updated: {self.updated}", f"Checked: {self.checked or '-'}"]
        return "\n".join(header) + (f"\n{closing}" if closing else "") + f"\n\n{self.note.strip()}\n"


def slug(name: str) -> str:
    """The file name an item answers to: 'TV delivery' and "tv-delivery" are the same one."""
    return re.sub(r"[^a-z0-9]+", "-", name.strip().strip("\"'`#").lower()).strip("-")


def _path(name: str, directory: Path) -> Path:
    return directory / f"{slug(name)}.md"


def _parse(path: Path) -> Item | None:
    """The item a file holds; None for an unreadable file."""
    try:
        header, _, note = path.read_text(encoding="utf-8").partition("\n\n")
    except OSError as e:
        trace(f"Watch list: cannot read {path.name}: {e}")
        return None
    fields = {"due": "", "created": "", "updated": "", "every": "", "checked": ""}
    for line in header.splitlines():
        if ":" in line and not line.startswith("#"):
            key, value = line.split(":", 1)
            if key.lower() in fields:
                fields[key.lower()] = value.strip()
    for key in ("due", "every", "checked"):
        if fields[key] == "-":
            fields[key] = ""
    return Item(name=path.stem, note=note.strip(), **fields)


def items(directory: Path = None) -> list[Item]:
    """The open items, oldest first."""
    directory = directory or WATCHLIST_DIR
    with lock:
        if not directory.is_dir():
            return []
        found = [item for path in directory.glob("*.md") if (item := _parse(path)) is not None]
    return sorted(found, key=lambda item: (item.created, item.name))


def get(name: str, directory: Path = None) -> Item | None:
    directory = directory or WATCHLIST_DIR
    if not slug(name):
        return None
    path = _path(name, directory)
    with lock:
        return _parse(path) if path.is_file() else None


def _log_copy(item: Item, log_dir: Path, closing: str = ""):
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    (log_dir / f"{item.name}.{stamp}.md").write_text(item.text(closing), encoding="utf-8")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _given(value: str | None) -> str | None:
    """A header value as the caller meant it: None (or blank) keeps the old one, '-' clears it."""
    return None if value is None or not value.strip() else value.strip()


def save(name: str, note: str, due: str | None = None, directory: Path = None, log_dir: Path = None,
         every: str | None = None, looked: bool = True) -> Item | None:
    """Start watching `name`, or replace the note of the item that already answers to it. The
    due date and the check cadence are kept unless given ('-' clears either). The assistant's
    save is a look at the item, so it stamps `checked` too; the curator's nightly rewrite is
    not one (`looked=False`) and leaves the stamp alone, or the next check would be postponed
    by a whole cadence. None when the name has no letters or digits."""
    directory = directory or WATCHLIST_DIR
    if not slug(name):
        return None
    today = date.today().isoformat()
    due, every = _given(due), _given(every)
    note = redact_for_storage(note)
    with lock:
        item = get(name, directory)
        if item is None:
            item = Item(slug(name), note.strip(), "" if due in (None, "-") else due, today, today,
                        "" if every in (None, "-") else every)
            directory.mkdir(parents=True, exist_ok=True)
            info(f"Watch list: '{item.name}' added" + (f", due {item.due}" if item.due else "")
                 + (f", checked every {item.every}" if item.every else ""))
        else:
            _log_copy(item, log_dir or LOG_DIR)
            item.note = note.strip()
            if due is not None:
                item.due = "" if due == "-" else due
            if every is not None:
                item.every = "" if every == "-" else every
            item.updated = today
            info(f"Watch list: '{item.name}' updated" + (f", checked every {item.every}" if item.every else ""))
        if looked:
            item.checked = _now()
        else:
            trace(f"Watch list: '{item.name}' saved without a look — checked stays {item.checked or 'unset'}")
        _path(name, directory).write_text(item.text(), encoding="utf-8")
    return item


def check(name: str, directory: Path = None) -> Item | None:
    """Record that the item was looked at just now and nothing changed: only `checked` moves,
    the note and `updated` stay. None when there is no such item."""
    directory = directory or WATCHLIST_DIR
    with lock:
        item = get(name, directory)
        if item is None:
            info(f"Watch list: no item '{slug(name) or name.strip()}' to mark checked")
            return None
        item.checked = _now()
        _path(name, directory).write_text(item.text(), encoding="utf-8")
    info(f"Watch list: '{item.name}' checked, nothing new" + (f" (every {item.every})" if item.every else ""))
    return item


def delete(name: str, outcome: str = "", directory: Path = None, log_dir: Path = None) -> Item | None:
    """Remove the item's file, its last state and the outcome kept in the log; None when there is no such item."""
    directory = directory or WATCHLIST_DIR
    outcome = redact_for_storage(outcome).strip()
    with lock:
        item = get(name, directory)
        if item is None:
            info(f"Watch list: no item '{slug(name) or name.strip()}' to delete")
            return None
        _log_copy(item, log_dir or LOG_DIR, closing=f"Closed: {date.today().isoformat()}\nOutcome: {outcome or '-'}")
        _path(name, directory).unlink()
    info(f"Watch list: '{item.name}' deleted" + (f" — {outcome}" if outcome else ""))
    return item


def _days_since(text: str, today: date) -> int | None:
    """Days from the ISO date `text` opens with to today; None when it opens with none."""
    match = re.match(r"\d{4}-\d{2}-\d{2}", text.strip())
    if not match:
        return None
    try:
        return (today - date.fromisoformat(match.group(0))).days
    except ValueError:
        return None


_UNIT_SECONDS = {"m": 60, "h": 3600, "d": 86400, "w": 7 * 86400, "mo": 30 * 86400}
_UNIT_WORDS = (("mo", "mo|months?"), ("m", "m|mins?|minutes?"), ("h", "h|hrs?|hours?"), ("d", "d|days?"),
               ("w", "w|wks?|weeks?"))
_UNIT_PATTERN = "|".join(f"(?P<{unit}>{words})" for unit, words in _UNIT_WORDS)
_AMOUNT = re.compile(rf"(\d+(?:\.\d+)?)\s*(?:{_UNIT_PATTERN})\b")
_BARE_UNIT = re.compile(r"\b(?:(?P<h>hour(?:ly)?)|(?P<d>da(?:il)?y|night(?:ly)?)|(?P<w>week(?:ly)?)|(?P<mo>month(?:ly)?))\b")


def interval(every: str) -> timedelta | None:
    """The time between two checks a cadence asks for: "6h", "1d 12h", "weekly", "every 2 days",
    "every day". None for an empty cadence or one this cannot read — it is free text."""
    text = every.strip().lower()
    seconds = sum(float(m.group(1)) * _UNIT_SECONDS[m.lastgroup] for m in _AMOUNT.finditer(text))
    if not seconds and (bare := _BARE_UNIT.search(text)):
        seconds = _UNIT_SECONDS[bare.lastgroup]
    return timedelta(seconds=seconds) if seconds > 0 else None


def stamp(text: str) -> datetime | None:
    """The moment a header value names: `checked`'s date and time, or a bare date (midnight).
    None when it opens with neither."""
    match = re.match(r"(\d{4}-\d{2}-\d{2})(?:[ T](\d{2}:\d{2}))?", text.strip())
    if not match:
        return None
    try:
        return datetime.fromisoformat(f"{match.group(1)} {match.group(2) or '00:00'}")
    except ValueError:
        return None


def describe_duration(delta: timedelta) -> str:
    """'just now', '3 min', '2 h 15 min' or '1 d 4 h' — how long a moment has been reached."""
    minutes = int(delta.total_seconds() // 60)
    if minutes < 1:
        return "just now"
    days, minutes = divmod(minutes, 24 * 60)
    hours, minutes = divmod(minutes, 60)
    if days:
        return f"{days} d {hours} h"
    if hours:
        return f"{hours} h {minutes} min"
    return f"{minutes} min"


def staleness(item: Item, today: date = None) -> str:
    """Why the item needs a look, or '' when it does not. A check is a touch: an item polled on
    its cadence that keeps finding nothing is current, however old its note."""
    today = today or date.today()
    overdue = _days_since(item.due, today)
    if overdue is not None and overdue >= OVERDUE_AFTER_DAYS:
        return f"STALE: due date passed {overdue} days ago"
    touched = [d for d in (_days_since(item.updated, today), _days_since(item.checked, today)) if d is not None]
    if touched and min(touched) >= STALE_AFTER_DAYS:
        return f"STALE: not updated or checked for {min(touched)} days"
    return ""


def _block(item: Item, today: date = None, now: datetime = None) -> str:
    """One item in full: its name and facts on a line, the note under it. A due time the item
    has reached without a look since is said so, with how long ago — the check then knows a
    reminder is a repeat and that the item comes back until it is looked at."""
    now = now or datetime.now()
    facts = ([f"due {item.due}"] if item.due else []) + [f"updated {item.updated}"]
    if (reached := fell_due(item, now)) is not None:
        facts[0] += f" — reached {describe_duration(now - reached)} ago, not looked at since"
    if item.every:
        facts.append(f"check every {item.every}")
    if item.checked:
        facts.append(f"last checked {item.checked}")
    if stale := staleness(item, today):
        facts.append(stale)
    return f"`{item.name}` ({', '.join(facts)})" + (f"\n{item.note}" if item.note else "")


def render(directory: Path = None, today: date = None) -> str:
    """The open items in full, as the curator reads them."""
    return "\n\n".join(_block(item, today) for item in items(directory)) or EMPTY


def due_moment(item: Item) -> datetime | None:
    """The moment the due names: its time, or the morning of a bare date. None for free text,
    which fires nothing."""
    text = item.due.strip()
    if re.match(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}", text):
        return stamp(text)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text) and (day := stamp(text)):
        return day + timedelta(hours=BARE_DATE_HOUR)
    return None


def fell_due(item: Item, now: datetime = None) -> datetime | None:
    """The due moment the item has reached without a look since: the moment, or None when the
    item has none, it is still ahead, or the item was checked, saved or re-dated after it."""
    moment = due_moment(item)
    if moment is None or moment > (now or datetime.now()):
        return None
    checked = stamp(item.checked)
    return moment if checked is None or checked < moment else None


def due_for_poll(item: Item, now: datetime = None, tolerance: timedelta = timedelta(0)) -> bool:
    """Whether the item has reached its due time without a look since (`fell_due`), or its check
    cadence has elapsed since it was last checked. An item without a cadence is otherwise never
    polled; one never checked, or whose cadence or stamp cannot be read, is due — the LLM is the
    one to make sense of it. `tolerance` is how much short of the cadence still counts (never
    more than half the cadence): a caller that looks every 55 minutes and stamps `checked` a few
    minutes into the look would otherwise find an hourly item not yet due and poll it every
    other time. The due time takes no tolerance — the watcher wakes for it on the minute."""
    now = now or datetime.now()
    if fell_due(item, now) is not None:
        return True
    if not item.every.strip():
        return False
    every, checked = interval(item.every), stamp(item.checked)
    if every is None or checked is None:
        return True
    return now - checked >= every - min(tolerance, every / 2)


def pending(now: datetime = None, within: timedelta = None, directory: Path = None) -> list[tuple[Item, datetime]]:
    """The moments the due watcher wakes for, soonest first: every due time not yet looked at
    since (ahead or already reached), and the next poll of every item whose cadence is shorter
    than `within` — the periodic check serves the longer ones. A cadence item never checked
    polls at its creation. Each moment names the one wake it earns: the watcher fires once per
    (item, moment) and the periodic checks carry an item left unlooked-at after that."""
    now = now or datetime.now()
    found = []
    for item in items(directory):
        moment = due_moment(item)
        checked = stamp(item.checked)
        if moment is not None and (checked is None or checked < moment):
            found.append((item, moment))
        every = interval(item.every)
        if every is not None and within is not None and every < within:
            found.append((item, (checked + every) if checked else (stamp(item.created) or now)))
    found.sort(key=lambda pair: pair[1])
    return found


def render_due(directory: Path = None, now: datetime = None, tolerance: timedelta = timedelta(0)) -> str:
    """The items whose poll is due, in full, as the autonomous check reads them: the check
    carries only what it must act on and reads anything else with `load_watch`."""
    open_items = items(directory)
    due = [item for item in open_items if due_for_poll(item, now, tolerance)]
    if not due:
        trace(f"Watch list: none of the {len(open_items)} open items is due for a poll")
        return NOTHING_DUE
    info(f"Watch list: {len(due)} of {len(open_items)} open items due for a poll: "
         + ", ".join(item.name for item in due))
    now = now or datetime.now()
    return "\n\n".join(_block(item, now.date(), now) for item in due)


def index_start(note: str) -> str:
    """The start of a text, cut at a word once it runs past what an index line carries — of a
    watch item's note here, of a procedure's `When:` line in the procedures' index."""
    first = " ".join(note.strip().split("\n", 1)[0].split())
    if len(first) <= INDEX_NOTE_CHARS:
        return first
    # One char past the limit, so a word ending exactly at it is cut after, not inside
    return first[:INDEX_NOTE_CHARS + 1].rsplit(" ", 1)[0].rstrip(" ,;:—-(") + "…"


def _due_fact(item: Item, today: date) -> str:
    """'due today', 'due today 17:30' or 'overdue 3 d' — and '' for an item whose day has not
    come, so the index says nothing about a date the assistant need not act on yet."""
    days = _days_since(item.due, today)
    if days is None or days < 0:
        return ""
    if days == 0:
        clock = item.due.strip()[10:].strip()
        return f"due today {clock}".rstrip()
    return f"overdue {days} d"


def render_index(directory: Path = None, today: date = None) -> str:
    """The `{WATCHLIST}` text: one line per open item — its name, whether it is due today or
    overdue (a date still ahead is not shown), the idle-staleness mark and the start of its
    note. Neither the cadence nor the check stamp, so a poll that only stamps an item leaves
    the prompt as it was."""
    today = today or date.today()
    lines = []
    for item in items(directory):
        facts = [fact for fact in (_due_fact(item, today),) if fact]
        if (stale := staleness(item, today)) and "due date" not in stale:
            facts.append(stale)
        start = index_start(item.note)
        lines.append(f"- `{item.name}`" + (f" ({', '.join(facts)})" if facts else "") + (f" — {start}" if start else ""))
    return "\n".join(lines) or EMPTY


def load(name: str, directory: Path = None, today: date = None) -> str:
    """What `load_watch(name)` answers: the item in full, or which items exist instead."""
    item = get(name, directory)
    if item is None:
        info(f"Watch list: '{name.strip()}' asked for, no such item")
        open_names = ", ".join(i.name for i in items(directory)) or "none"
        return f"load_watch failed: no item named '{slug(name) or name.strip()}'. Open: {open_names}"
    trace(f"Watch list: '{item.name}' loaded ({len(item.note)} chars)")
    return f"{HEADER_PREFIX}{_block(item, today)}\n"


def version(directory: Path = None) -> tuple:
    """Changes whenever the rendered list would: on any save or delete, and at midnight
    (staleness is counted in days)."""
    directory = directory or WATCHLIST_DIR
    try:
        files = tuple(sorted((p.name, p.stat().st_mtime_ns) for p in directory.glob("*.md")))
    except OSError:
        files = ()
    return files, time.strftime("%Y-%m-%d")


def run(command: str, args: list[str], directory: Path = None, log_dir: Path = None,
        looked: bool = True) -> str:
    """One watch-list command with the arguments as the command parser split them. Returns ''
    when it ran and the reason when it could not — the error the LLM is told. `looked` is
    whether a save counts as a check of the item (the assistant's does, the curator's does not)."""
    args = [a.strip() for a in args]
    if command == "save_watch":
        if len(args) < 2 or not args[1]:
            return "save_watch failed: expected 'name, note, due, every' parameters"
        if (words := len(args[1].split())) > NOTE_WORD_LIMIT:
            cut = math.ceil((words - NOTE_WORD_LIMIT) * 100 / words)
            kept = "the item keeps its previous note" if get(args[0], directory) else "the item was not created"
            info(f"Watch list: save of '{slug(args[0]) or args[0]}' refused — note is {words} words, "
                 f"limit {NOTE_WORD_LIMIT}")
            return (f"save_watch failed: this note is too large — {words} words, and a watch item's note may "
                    f"be at most {NOTE_WORD_LIMIT} words. Nothing was saved, {kept}. Condense it by at least "
                    f"{cut}% and save it again.")
        if save(args[0], args[1], args[2] if len(args) > 2 else None, directory, log_dir,
                every=args[3] if len(args) > 3 else None, looked=looked) is None:
            return "save_watch failed: the name must contain letters or digits"
        return ""
    if command == "checked_watch":
        if not args or not args[0]:
            return "checked_watch failed: expected the item's name"
        if check(args[0], directory) is None:
            open_names = ", ".join(item.name for item in items(directory)) or "none"
            return f"checked_watch failed: no item named '{slug(args[0]) or args[0]}'. Open: {open_names}"
        return ""
    if command == "delete_watch":
        if not args or not args[0]:
            return "delete_watch failed: expected the item's name"
        if delete(args[0], ", ".join(args[1:]), directory, log_dir) is None:
            open_names = ", ".join(item.name for item in items(directory)) or "none"
            return f"delete_watch failed: no item named '{slug(args[0]) or args[0]}'. Open: {open_names}"
        return ""
    return f"{command} is not a watch-list command"
