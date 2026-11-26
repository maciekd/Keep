"""The phone app's chat as the server keeps it: one ordered log of items, where every change
is "the item with this id is now this" under a rising sequence number, so an app that says
the last number it saw is sent exactly what it missed.

No Quart and no hub in here. An item is a dict with `id`, `kind`, `at` (when it first
appeared) and `order` (the sequence number it first appeared under — the list is drawn in
that order, since a change gives an old item a new sequence number). Whatever else it holds
is the transport's business. The working line is not an item: it is sent to whoever listens
and never logged. A listener's stream opens with `{"log": …}`, then what it missed, then
`{"live": true}`.

The log is capped (`CAP_ITEMS`, `MAX_AGE_SECONDS`) and kept in `<directory>/app_chat.json`,
the photos and videos beside it. `log` names this file's lifetime: an app whose cursor was
counted in another log, or is older than the log reaches, is told to start over."""
import json
import mimetypes
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

from common.logging import error, info, trace

CAP_ITEMS = 500
MAX_AGE_SECONDS = 7 * 24 * 3600
MAX_TOMBSTONES = 200
SAVE_DELAY_SECONDS = 2.0
FILE_NAME = "app_chat.json"
MEDIA_FOLDER = "app_chat_media"
MEDIA_ROUTE = "/app/media/"
_MEDIA_ID = re.compile(r"^[0-9a-f]{32}\.[a-z0-9]{2,5}$")


def new_id() -> str:
    return uuid.uuid4().hex


class Subscription:
    """What one listener has not been sent yet. A newer line about the same item replaces
    the older one, so a listener that reads slower than a reply streams gets the newest
    state and nothing in between."""

    def __init__(self, wake: Callable[[], None]):
        self._wake = wake
        self._pending: dict = {}
        self._lock = threading.Lock()
        self.ended = False      # the server is stopping: the stream of this listener ends

    def end(self) -> None:
        self.ended = True
        self._wake()

    def push(self, key, line: dict) -> None:
        with self._lock:
            self._pending.pop(key, None)
            self._pending[key] = line
        self._wake()

    def drain(self) -> list[dict]:
        with self._lock:
            lines = list(self._pending.values())
            self._pending.clear()
        return lines


class Transcript:
    """The item log of the app chat: upsert, remove, replay from a cursor, listeners, a cap
    and a file."""

    def __init__(self, directory: Path, autosave: bool = True, clock: Callable[[], float] = time.time):
        self._path = Path(directory) / FILE_NAME
        self._media = Path(directory) / MEDIA_FOLDER
        self._autosave = autosave
        self._clock = clock
        self._lock = threading.RLock()
        self._entries: dict[str, dict] = {}     # id → {"seq", "item"}, in the order the items appeared
        self._removed: dict[str, int] = {}      # id → the sequence number it was removed under
        self._seq = 0
        self._floor = 0     # a cursor below this has missed something the log no longer holds
        self._working: Optional[dict] = None
        self._subscriptions: list[Subscription] = []
        self._save_timer: Optional[threading.Timer] = None
        self._closed = False        # the file says nothing follows, and nothing has since
        self._ended = False
        self.log = new_id()
        self._load()

    # --- items --------------------------------------------------------------------------------

    def upsert(self, item_id: str, kind: str, **fields) -> str:
        """The item is now *fields*. A new id is appended to the list."""
        with self._lock:
            self._seq += 1
            known = self._entries.get(item_id)
            if known is None:
                item = {"id": item_id, "kind": kind, "at": self._clock(), "order": self._seq, **fields}
            else:
                old = known["item"]
                item = {"id": item_id, "kind": kind, "at": old["at"], "order": old["order"], **fields}
            self._entries[item_id] = {"seq": self._seq, "item": item}
            self._removed.pop(item_id, None)
            line = {"seq": self._seq, "item": item}
            if known is None:
                self._trim()
            self._publish(("item", item_id), line)
            self._changed()
        return item_id

    def remove(self, item_id: str) -> None:
        with self._lock:
            if self._entries.pop(item_id, None) is None:
                trace(f"App transcript: nothing to remove under {item_id}")
                return
            self._seq += 1
            self._removed[item_id] = self._seq
            while len(self._removed) > MAX_TOMBSTONES:
                oldest = next(iter(self._removed))
                self._floor = max(self._floor, self._removed.pop(oldest))
            self._publish(("item", item_id), {"seq": self._seq, "remove": item_id})
            self._changed()

    def item(self, item_id: str) -> Optional[dict]:
        with self._lock:
            entry = self._entries.get(item_id)
            return dict(entry["item"]) if entry else None

    def items(self) -> list[dict]:
        with self._lock:
            return [dict(entry["item"]) for entry in self._entries.values()]

    def _trim(self) -> None:
        horizon = self._clock() - MAX_AGE_SECONDS
        dropped = 0
        while self._entries:
            oldest_id = next(iter(self._entries))
            oldest = self._entries[oldest_id]
            if len(self._entries) <= CAP_ITEMS and oldest["item"]["at"] >= horizon:
                break
            del self._entries[oldest_id]
            self._floor = max(self._floor, oldest["seq"])
            self._forget_media(oldest["item"])
            dropped += 1
        if dropped:
            info(f"App transcript: {dropped} old item(s) dropped, {len(self._entries)} kept")

    # --- the working line -------------------------------------------------------------------

    def set_working(self, working: Optional[dict]) -> None:
        with self._lock:
            if working == self._working:
                return
            self._working = working
            self._publish(("working",), {"working": working})

    # --- listeners --------------------------------------------------------------------------

    def subscribe(self, after: int, log: str, wake: Callable[[], None]) -> tuple[list[dict], Subscription]:
        """The lines that bring a listener at cursor *after* of *log* up to date, and the
        subscription the later ones arrive in."""
        with self._lock:
            subscription = Subscription(wake)
            subscription.ended = self._ended
            self._subscriptions.append(subscription)
            lines = [{"log": self.log}] + self._replay(after, log)
            if self._working is not None:
                lines.append({"working": self._working})
            lines.append({"live": True})    # what follows happens while the listener watches
            info(f"App transcript: a listener joined at {after} ({len(lines) - 2} line(s) to catch up, "
                 f"{len(self._subscriptions)} listening)")
            return lines, subscription

    def unsubscribe(self, subscription: Subscription) -> None:
        with self._lock:
            if subscription in self._subscriptions:
                self._subscriptions.remove(subscription)
            info(f"App transcript: a listener left ({len(self._subscriptions)} listening)")

    def end_listeners(self) -> None:
        """The server is stopping: every listener's stream ends, and one that joins from now
        on gets what it missed and ends too."""
        with self._lock:
            self._ended = True
            info(f"App transcript: ending the stream of {len(self._subscriptions)} listener(s)")
            for subscription in self._subscriptions:
                subscription.end()

    def listeners(self) -> int:
        with self._lock:
            return len(self._subscriptions)

    def _replay(self, after: int, log: str) -> list[dict]:
        stale = bool(after) and (log != self.log or after > self._seq or after < self._floor)
        if stale:
            info(f"App transcript: cursor {after} of log {log[:8] or '-'} cannot be continued, replaying everything")
        if stale or not after:
            lines = [{"seq": e["seq"], "item": e["item"]} for e in self._entries.values()]
            lines.sort(key=lambda line: line["seq"])
            return ([{"reset": True}] if stale else []) + lines
        lines = [{"seq": e["seq"], "item": e["item"]} for e in self._entries.values() if e["seq"] > after]
        lines += [{"seq": seq, "remove": item_id} for item_id, seq in self._removed.items() if seq > after]
        lines.sort(key=lambda line: line["seq"])
        return lines

    def _publish(self, key, line: dict) -> None:
        for subscription in self._subscriptions:
            subscription.push(key, line)

    # --- photos and videos ------------------------------------------------------------------

    def add_media(self, data: bytes, extension: str) -> str:
        """Keep *data* beside the log; returns the address an item names it by."""
        media_id = f"{new_id()}.{extension}"
        self._media.mkdir(parents=True, exist_ok=True)
        (self._media / media_id).write_bytes(data)
        trace(f"App transcript: kept {len(data)} bytes as {media_id}")
        return MEDIA_ROUTE + media_id

    def media(self, media_id: str) -> Optional[tuple[bytes, str]]:
        """The bytes and the content type behind a media id; None for an id that is not one."""
        if not _MEDIA_ID.match(media_id):
            info(f"App transcript: refused a media id that is not one ({len(media_id)} chars)")
            return None
        path = self._media / media_id
        if not path.is_file():
            trace(f"App transcript: no media {media_id}")
            return None
        return path.read_bytes(), mimetypes.guess_type(media_id)[0] or "application/octet-stream"

    def _forget_media(self, item: dict) -> None:
        """The files an item names: its own, a video's still, the pictures of a user's message."""
        addresses = [item.get("url"), item.get("poster")] + [image.get("url") for image in item.get("images") or []]
        for address in addresses:
            address = address or ""
            if address.startswith(MEDIA_ROUTE) and _MEDIA_ID.match(address[len(MEDIA_ROUTE):]):
                (self._media / address[len(MEDIA_ROUTE):]).unlink(missing_ok=True)

    # --- the file -----------------------------------------------------------------------------

    def _changed(self) -> None:
        if self._closed:
            # A change after the closing save (a reply cancelled by the stop settles late): the
            # file must not stay closed behind it, or the next run would continue this log
            # below a number a phone already holds.
            info("App transcript: changed after it was closed, saving it as open again")
            self._closed = False
            self.flush()
            return
        if not self._autosave or self._save_timer is not None:
            return
        self._save_timer = threading.Timer(SAVE_DELAY_SECONDS, self.flush)
        self._save_timer.name = "appchatsave"
        self._save_timer.daemon = True
        self._save_timer.start()

    def flush(self, closed: bool = False) -> None:
        """Write the log to its file now. *closed* says nothing follows: only such a file is
        the whole log, so only it is continued under its name by the next run."""
        with self._lock:
            if self._save_timer is not None:
                self._save_timer.cancel()
                self._save_timer = None
            if self._closed and not closed:
                trace("App transcript: a save due after the closing one, nothing has changed since")
                return
            self._closed = closed
            state = {"log": self.log, "closed": closed, "seq": self._seq, "floor": self._floor,
                     "removed": self._removed, "entries": list(self._entries.values())}
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                scratch = self._path.with_suffix(".tmp")
                scratch.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
                os.replace(scratch, self._path)
                trace(f"App transcript: saved {len(self._entries)} item(s) at {self._seq}")
            except OSError as e:
                error(f"App transcript: could not save {self._path}: {e}")

    def _load(self) -> None:
        if not self._path.is_file():
            info(f"App transcript: no {self._path.name} yet, starting an empty log")
            return
        try:
            state = json.loads(self._path.read_text(encoding="utf-8"))
            self._entries = {entry["item"]["id"]: entry for entry in state["entries"]}
            self._removed = {item_id: int(seq) for item_id, seq in state["removed"].items()}
            self._seq, self._floor, log = int(state["seq"]), int(state["floor"]), state["log"]
            closed = state.get("closed") is True
            with self._lock:
                self._trim()    # reads every field a later use counts on, so a file of another shape fails here
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
            error(f"App transcript: {self._path} is unreadable ({e!r}), starting an empty log")
            self._entries, self._removed, self._seq, self._floor = {}, {}, 0, 0
            return
        if closed:
            self.log = log
            self.flush()    # open again: a run that is killed must not leave a file that says closed
        else:
            # The file was saved up to SAVE_DELAY_SECONDS behind a run that never closed it, so
            # numbers above its last may be in a phone already: under a new name every cursor
            # starts over instead of skipping what reuses them.
            info(f"App transcript: {self._path.name} was not closed by the last run, continuing as a new log")
        info(f"App transcript: {len(self._entries)} item(s) loaded, at {self._seq}")
