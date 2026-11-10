"""The Feed contract: an Integration that produces history rows over time (iMessage, personal
Telegram, later Photos and a training sheet). A feed only knows how to read its source from a
cursor onwards; FeedRunner owns the thread, the persisted cursor, the catch-up / live states
and the progress numbers (plans/FEED_PLAN.md)."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

FEED_ROLE = "system"    # the history role every feed row carries; with the source it names a feed's rows


@dataclass
class FeedEntry:
    timestamp: float            # when it happened at the source, never when we saw it
    content: str
    location: Optional[str] = None
    role: str = FEED_ROLE


@dataclass
class FeedBatch:
    entries: list[FeedEntry] = field(default_factory=list)
    cursor: Optional[str] = None    # persisted by the runner once the entries are stored
    exhausted: bool = True          # False: more is waiting, fetch again without sleeping
    remaining: Optional[int] = None    # what the feed knows is still waiting after this batch (a sweep's count)


class FeedSourceAway(Exception):
    """Raised by `open_feed`, `initial_cursor` or `fetch` of a feed whose source cannot be reached
    right now and is not broken — a host on another Mac that sleeps, restarts or updates itself.
    The runner keeps the feed's place and asks again for as long as it takes, instead of
    counting the failure towards giving up: what the source holds waits behind the cursor."""


class Feed(ABC):
    """What a feed implements. The cursor is an opaque string the feed alone interprets
    (a ROWID, a timestamp, a JSON blob); the runner stores whatever `fetch` hands back."""

    poll_interval: float = 5.0          # seconds between fetches once the feed is live
    default_backfill_months: int = 1    # horizon when feeds.<slug>.backfill_months is not set
    item_noun: str = "items"            # plural, what the dashboard counts: "4202 messages"
    # The version of what the cursor means. A feed bumps it when a stored cursor no longer
    # says what it said when it was written (iMessage: a ROWID that once meant "everything
    # below is read", now "everything below dated in the horizon"); the runner then catches
    # up from the horizon once more and drops what the history already holds a row for at the
    # same moment — not by the dedup index, which a row stored live under another content
    # (a described photo, a contact's earlier name) would pass (`FeedRunner._drop_reread`).
    cursor_generation: int = 0
    # What the cursor is a position in, for a feed that can be pointed at another source under
    # the same name (iMessage: the Messages account its ROWIDs are of). The runner keeps it with
    # the cursor and hands it back to `cursor_lost`; None for a feed with one source for good.
    cursor_source: Optional[str] = None
    # The runner's horizon, set before `initial_cursor` and kept current — the persisted one
    # after a restart, since a horizon only moves back. A feed whose cursor is not in time
    # order (an iMessage ROWID: Messages in iCloud writes the history in any order) keeps
    # the items from before it out of `fetch` and `remaining`.
    horizon_ts: Optional[float] = None

    @property
    @abstractmethod
    def feed_name(self) -> str:
        """The history `source` of every row this feed writes, e.g. "iMessage"."""
        ...

    @property
    @abstractmethod
    def config_slug(self) -> str:
        """Key under `feeds:` in config.yaml, e.g. "imessage"."""
        ...

    @abstractmethod
    def initial_cursor(self, horizon_ts: float) -> Optional[str]:
        """The cursor to start from when nothing is persisted: just before `horizon_ts`."""
        ...

    @abstractmethod
    def fetch(self, cursor: Optional[str], backfill: bool) -> FeedBatch:
        """Everything after `cursor`, up to the feed's own batch size. `backfill` is True
        while the runner is catching up: skip the side effects a months-old entry must not
        trigger (2FA forwarding, media downloads)."""
        ...

    def remaining(self, cursor: Optional[str]) -> Optional[int]:
        """How many items are still ahead of `cursor`, when that is cheap to know; None otherwise."""
        return None

    def cursor_lost(self, cursor: Optional[str], source: Optional[str]) -> Optional[str]:
        """Why the stored cursor is no place in what the feed reads now, None (the default) when
        it is. `source` is the `cursor_source` it was stored with — None for a row from before
        sources were kept. The runner asks before it resumes and catches up from the horizon
        again on a reason: a cursor of another source would skip what lies before it there, or
        sit past the end and leave a live, healthy feed that reads nothing."""
        return None

    def sweep(self, horizon_ts: float, cursor: Optional[str]) -> Optional[FeedBatch]:
        """A batch of the items between the horizon and `cursor` that the feed has no row for:
        a batch the provider refused halfway, rows a migration forgot — holes `fetch` never
        comes back to, since it reads only past the cursor. The runner asks while the feed is
        live and idle, every `FeedRunner.SWEEP_SECONDS` and straight away after a batch that
        was not `exhausted`; the batch's `cursor` is ignored (the cursor stays) and its
        `remaining` is shown as the sweep's progress. None (the default): the feed cannot tell
        a hole from a row it has, so nothing is swept."""
        return None

    def count_stored(self) -> Optional[int]:
        """How many items the feed holds, for the status counts; None (the default) counts its
        rows in the history. A feed whose real store is elsewhere (Photos: the index table,
        which predates its history rows) answers from there, so stored and `remaining` are
        measured against the same thing."""
        return None

    def cursor_reached(self, cursor: Optional[str], target: Optional[str]) -> bool:
        """Whether `cursor` is at or past `target` — the runner's test for a gap read up to the
        resume cursor. Every feed here writes its cursor as a number (a ROWID, a timestamp), so
        the default compares numerically and falls back to equality; a feed with another shape
        overrides. No target (a feed without a cursor) is reached at once."""
        if target is None:
            return True
        if cursor is None:
            return False
        try:
            return float(cursor) >= float(target)
        except ValueError:
            return cursor == target

    def in_flight(self) -> int:
        """Items of the batch being fetched that are already done (Photos: described but not
        yet handed over). The runner takes them off `remaining` so a long batch shows its
        progress item by item instead of once at the end. Must be cheap: called per frame."""
        return 0

    def open_feed(self) -> bool:
        """Connect. Return False when the feed cannot run now (no session file, no database);
        the runner then stops with `last_error` = the reason in `open_error`."""
        return True

    def close_feed(self):
        """Disconnect; called on stop and restart."""
        pass

    @property
    def open_error(self) -> Optional[str]:
        """Why `open_feed` returned False, for the status line."""
        return None
