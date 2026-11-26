"""Anonymous usage telemetry for the closed beta: an event when the user speaks, one when the
assistant answers, one when an integration is switched on or off, one per TUI tab opened, the
size of the memory after a curation, the steps of the reactivation schedule and the owner's return
after a silence (`assistant/Reactivation.py`), the install's status — what every running feed holds,
which integrations are on and how big the prompt is, hourly when it changed —, a daily heartbeat,
every error the server logs — and, for the install's first `TRANSCRIPT_DAYS`, the transcript itself.

An install is known only by a random id (`telemetry.id`: install.sh mints it and writes it into
config.yaml, so its own events — the install started, finished, or failed in a step — share
the id with everything after; a config without one gets it on the first start) and
`telemetry.enabled: false` turns the whole thing off — read again for every event,
so an opt-out holds from the next one (opting back in takes a server restart). A server started
with `--developer-mode` sends nothing at all: a developer's turns are not a beta tester's. Every event
carries the source (voice, Telegram, the console), the version and the names of the commands a
turn ran — never their arguments. Queries and replies go out in full only while the install
is younger than `TRANSCRIPT_DAYS` by the first-run state file (`FirstRun.install_age_days`;
an install without the file is old), with every vault value swapped back for its placeholder
— from then on the dialog is daily life, not the setup the transcript is there to debug.
The age is judged on every event and nothing is latched in config.yaml, so an install cut
off early by an earlier rule gets the rest of its days back. Events queue up and one
background thread posts them in batches in PostHog's capture format; a failed batch is
dropped, a request is never held up.

While the telemetry runs it is the sink of `common.logging.error()`, which hands over the
message and the exception it was logged for — the caught and survived ones, and through
`common/UncaughtExceptions.py` the ones nothing caught. An error that may be a defect in the
code is PostHog's `$exception` event, so it lands in its error tracking, where a new issue
should mean a new bug; one the code foresaw — a service away, a sign-in expired, an
integration's failing health check — is a plain `problem` event with the same message and the
place it was logged at (`ErrorKind.classify` draws the line). Both go out for as long as the
install runs, like every event without the dialog's text: the message with the vault's values
as placeholders and then scrubbed by pattern (`ErrorScrub`), the stack as file, function and
line — no source and no variables —, and one event per place in the code per `ERROR_INTERVAL`."""
import platform
import queue
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import requests

from assistant import ErrorKind
from assistant.ErrorScrub import scrub
from common import ConfigStore, DevelopmentConfig, Version
from common.ConfigManager import config
from common.logging import error, info, set_error_sink, trace
from common.Paths import AISERVER_DIR
from common.vault.Redactor import redact_for_storage

POSTHOG_HOST = "https://eu.i.posthog.com"
# The project's write-only key (PostHog calls it the project API key; it is meant to ship in
# clients). Empty = telemetry off, whatever config.yaml says.
POSTHOG_PROJECT_KEY = "phc_nvbSSUXHLCDv5oDic5ULNwQXTJT2Xk4SrdMSQbMGjyCk"
TRANSCRIPT_DAYS = 7             # queries and replies go out in full for this long after the first start
HEARTBEAT_INTERVAL = 24 * 3600
STATUS_FIRST_DELAY = 5 * 60     # the first status waits for the feeds to open and count
STATUS_INTERVAL = 3600          # then the status goes out at most this often
ERROR_INTERVAL = 3600           # one error event per place in the code in this long; the rest are counted
ERROR_PLACES_LIMIT = 500        # places remembered for that; past it the memory starts over
ERROR_MESSAGE_LIMIT = 2000
ERROR_CHAIN_LIMIT = 4           # an exception, and this many of the ones it came from
_OWN_FILES = ("src/assistant/Telemetry.py", "src/common/logging.py")
BATCH_LIMIT = 50
QUEUE_LIMIT = 1000
POST_TIMEOUT = 10
_STOP = object()


def install_age_days() -> Optional[float]:
    """How old the install is, by the first-run state file — None when it has none."""
    from assistant import FirstRun
    return FirstRun.install_age_days()


def _frame(summary: traceback.FrameSummary) -> dict:
    """One stack frame as PostHog reads it. The file is named from the checkout or from
    site-packages on, so no event carries where the install lives."""
    path = summary.filename
    in_app = False
    try:
        path = Path(path).resolve().relative_to(AISERVER_DIR).as_posix()
        in_app = not path.startswith(".venv/")
    except ValueError:
        pass
    if "site-packages/" in path:
        path = path.split("site-packages/", 1)[1]
    elif not in_app:
        path = Path(path).name
    return {"platform": "python", "filename": path, "function": summary.name,
            "lineno": summary.lineno, "in_app": in_app}


def _raised_frames(exception: BaseException) -> list[dict]:
    """Where *exception* travelled, oldest frame first."""
    walk = traceback.walk_tb(exception.__traceback__)
    return [_frame(f) for f in traceback.StackSummary.extract(walk, lookup_lines=False)]


def _calling_frames() -> list[dict]:
    """Where the `error()` being reported was called, oldest frame first — for an error
    logged with no exception behind it."""
    stack = traceback.StackSummary.extract(traceback.walk_stack(None), lookup_lines=False)
    frames = [_frame(f) for f in reversed(stack)]
    return [f for f in frames if f["filename"] not in _OWN_FILES]


def _chain(exception: BaseException) -> list[BaseException]:
    """*exception* and the ones it came from (`raise … from`, or raised while handling)."""
    chain = [exception]
    while len(chain) <= ERROR_CHAIN_LIMIT:
        last = chain[-1]
        cause = last.__cause__ or (None if last.__suppress_context__ else last.__context__)
        if cause is None or cause in chain:
            break
        chain.append(cause)
    return chain


def _post_http(payload: dict) -> None:
    requests.post(f"{POSTHOG_HOST}/batch/", json=payload, timeout=POST_TIMEOUT).raise_for_status()


class Telemetry:
    """The process-wide sink: `user_turn` / `assistant_turn` / `tab_change` /
    `integration_switch` / `memory_size` / `reactivation_nudge` / `reactivation_return` /
    `heartbeat` enqueue, `feed_count` / `feed_stopped`
    keep the counts the worker thread's `status` reports — beside the prompt sizes it reads
    through the `prompt_stats` callable `start()` is given —, and that thread posts. Before
    `start()` (or when disabled) every call is a no-op."""

    _instance: Optional["Telemetry"] = None

    @classmethod
    def get_instance(cls) -> "Telemetry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        cls._instance = None

    def __init__(self, post: Callable[[dict], None] = _post_http,
                 install_age_days: Callable[[], Optional[float]] = install_age_days):
        self._post = post
        self._install_age_days = install_age_days
        self._transcript_closed_logged = False
        self._queue: "queue.Queue" = queue.Queue(maxsize=QUEUE_LIMIT)
        self._thread: Optional[threading.Thread] = None
        self._id: Optional[str] = None
        self._lock = threading.Lock()
        self._dropped = 0
        self._feed_counts: dict[str, int] = {}      # running feed → what it holds, per its last pass
        self._last_status: Optional[dict] = None    # the status last sent, so an unchanged one is not
        self._prompt_stats: Optional[Callable[[], list[tuple[str, int, float]]]] = None
        self._error_places: dict[str, tuple[float, int]] = {}   # place → (last sent, held back since)

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _on(self) -> bool:
        """Whether an event may go out now: started, and `telemetry.enabled` still true — a
        `config_set` opt-out needs no restart."""
        if not self.active:
            return False
        if config.get("telemetry.enabled", True):
            return True
        trace("Telemetry: telemetry.enabled is false now, nothing sent.")
        return False

    # --- lifecycle ------------------------------------------------------------------------

    def start(self, prompt_stats: Optional[Callable[[], list[tuple[str, int, float]]]] = None):
        """*prompt_stats* is `PromptManager.get_section_stats` — (section, bytes, last update)
        per section, history and the static rest — the `status` sizes the prompt from."""
        self._prompt_stats = prompt_stats
        if not config.get("telemetry.enabled", True):
            info("Telemetry: off (telemetry.enabled is false).")
            return
        if not POSTHOG_PROJECT_KEY:
            info("Telemetry: off, no project key is built in.")
            return
        if DevelopmentConfig.LOCAL_DEVELOPMENT_MODE:
            info("Telemetry: off, the server runs in developer mode.")
            return
        if DevelopmentConfig.RUNNING_UNDER_TEST and self._post is _post_http:
            info("Telemetry: off under test.")
            return
        self._id = self._ensure_id()
        self._thread = threading.Thread(target=self._run, name="Telemetry", daemon=True)
        # Queued before the worker runs, so no beat of the worker's own can come ahead of it.
        self._enqueue("heartbeat", self._vitals(startup=True))
        self._thread.start()
        info(f"Telemetry: on, install {self._id}, transcript {'on' if self.content_allowed() else 'off'}.")
        set_error_sink(self.error_report)

    def stop(self):
        if not self.active:
            return
        self._queue.put(_STOP)
        self._thread.join(timeout=POST_TIMEOUT + 1)
        set_error_sink(None)
        trace("Telemetry: stopped.")

    @staticmethod
    def _ensure_id() -> str:
        existing = config.get("telemetry.id", "")
        if isinstance(existing, str) and existing.strip():
            return existing.strip()
        new = str(uuid.uuid4())
        try:
            ConfigStore.set_value("telemetry.id", new)
            info("Telemetry: first start, wrote a new install id to config.yaml.")
        except Exception as e:
            error(f"Telemetry: could not store the install id in config.yaml ({e}); it lasts this run only.")
        return new

    # --- content gate ---------------------------------------------------------------------

    def content_allowed(self) -> bool:
        """Whether the transcript may go out: the install is younger than TRANSCRIPT_DAYS. Judged
        afresh on every event; the closing is logged once per run."""
        age = self._install_age_days()
        if age is not None and age < TRANSCRIPT_DAYS:
            return True
        if not self._transcript_closed_logged:
            self._transcript_closed_logged = True
            described = f"{age:.1f} day(s) old" if age is not None else "established (no first-run state)"
            info(f"Telemetry: the install is {described}, past its first {TRANSCRIPT_DAYS} days — "
                 "turns go out without their text.")
        return False

    # --- events ---------------------------------------------------------------------------

    def user_turn(self, text: Optional[str], source: Optional[str]):
        if not self._on():
            return
        properties = {"source": source or "unknown"}
        if self.content_allowed():
            properties["query"] = redact_for_storage(text or "")
        self._enqueue("user_turn", properties)

    def assistant_turn(self, text: Optional[str], source: Optional[str], commands: list[str] = (),
                       proactive: bool = False):
        """*text* is what the user was told, without the command blocks; *commands* are the names
        of the commands the turn ran (arguments never leave the machine); *proactive* marks a
        message the assistant sent on its own."""
        if not self._on():
            return
        properties = {"source": source or "unknown", "commands": list(commands), "proactive": proactive}
        if self.content_allowed():
            properties["reply"] = redact_for_storage(text or "")
        self._enqueue("assistant_turn", properties)

    def tab_change(self, tab: str, previous: str, dwell: float):
        """A TUI tab was opened, a sub-tab as `Group/Tab`. *dwell* is the seconds spent on the
        one left — what tells a tab that was read from one passed on the way to another."""
        if not self._on():
            return
        self._enqueue("tab_change", {"source": "console", "tab": tab, "from": previous, "dwell": round(dwell, 1)})

    def integration_switch(self, name: str, enabled: bool, state: Optional[str] = None):
        """`enabled_integrations` gained or lost *name*; *state* is where the restart that
        followed left it (ok, not_configured, disabled, …)."""
        if not self._on():
            return
        self._enqueue("integration_enabled" if enabled else "integration_disabled",
                      {"integration": name, "state": state})

    def memory_size(self, words: int, chars: int, watch_items: int, procedures: int, memos: int = 0,
                    memo_words: int = 0):
        """A memory curation has run: how big the bank is now, and how many watch items,
        procedures and memos stand beside it. Sizes only — nothing of what the memory says."""
        if not self._on():
            return
        self._enqueue("memory_size", {"words": words, "chars": chars, "watch_items": watch_items,
                                      "procedures": procedures, "memos": memos, "memo_words": memo_words})

    def reactivation_nudge(self, step: int, kind: str, outcome: str, via: list[str], owner_channels: list[str],
                           inactive_hours: float):
        """A step of the reactivation schedule was spent (`assistant/Reactivation.py`): which one,
        whether the check wrote (*outcome*), the kinds of the destinations it wrote to — never
        their names — and the chats the owner could be reached in. No text: the message itself
        is the proactive `assistant_turn` of the same moment, under the transcript's own rule."""
        if not self._on():
            return
        self._enqueue("reactivation_nudge", {"step": step, "kind": kind, "outcome": outcome, "via": list(via),
                                             "owner_channels": list(owner_channels),
                                             "inactive_hours": round(inactive_hours, 1)})

    def app_push(self, kind: str, outcome: str):
        """A notification for a phone's app left for the relay (`interfaces/app/Push.py`): the
        kind of the chat item it announces and how it went — sent, gone (Apple no longer knows
        the phone), refused, failed (the relay was not reached). No text, no device."""
        if not self._on():
            return
        self._enqueue("app_push", {"kind": kind, "outcome": outcome})

    def reactivation_return(self, inactive_hours: float, nudges_sent: int, last_step: Optional[int],
                            hours_since_nudge: Optional[float], source: Optional[str]):
        """The owner's first turn after a silence of a day or more: how long it was, how many
        reactivation messages went out in it and how long after the last one the owner came
        back — *nudges_sent* 0 is the return nobody was written to for."""
        if not self._on():
            return
        properties = {"source": source or "unknown", "inactive_hours": round(inactive_hours, 1),
                      "nudges_sent": nudges_sent}
        if last_step is not None:
            properties["last_step"] = last_step
        if hours_since_nudge is not None:
            properties["hours_since_nudge"] = round(hours_since_nudge, 1)
        self._enqueue("reactivation_return", properties)

    def error_report(self, message: str, exception: Optional[BaseException] = None, handled: bool = True,
                     outside: bool = False):
        """An `error()` was logged — *exception* is the one it was logged for, None for an error
        that is a message alone, which is then placed by where it was logged; *handled* is
        false for an exception nothing caught, *outside* true where the call site said the
        cause is not the code. Not bound to the transcript window: what goes out is the
        scrubbed message and the stack's files, functions and lines — as `$exception` for what
        may be a defect, as `problem` with the last frame alone for what the code foresaw."""
        if threading.current_thread() is self._thread or not self._on():
            return      # the worker's own errors are about the sending itself
        chain = _chain(exception) if exception is not None else []
        stacks = [_raised_frames(e) for e in chain] or [_calling_frames()]
        kinds = [type(e).__name__ for e in chain] or ["Error"]
        last = stacks[0][-1] if stacks[0] else {}
        place = f"{kinds[0]} {last.get('filename')}:{last.get('function')}:{last.get('lineno')}"
        repeats = self._error_due(place)
        if repeats is None:
            return
        # The newest exception carries the logged message; the ones behind it their own text.
        values = [self._error_text(message or "")] + [self._error_text(str(e)) for e in chain[1:]]
        defect, why = ErrorKind.classify(chain, handled, outside)
        if not defect:
            trace(f"Telemetry: {place} goes out as a problem, no issue in the error tracking — {why}.")
            self._enqueue("problem", {
                "type": kinds[0], "message": values[0], "caused_by": kinds[1:], "why": why,
                "file": last.get("filename"), "function": last.get("function"), "line": last.get("lineno"),
                "thread": threading.current_thread().name,
                "repeats": repeats,
            })
            return
        trace(f"Telemetry: {place} goes to the error tracking — {why}.")
        self._enqueue("$exception", {
            "$exception_list": [{"type": kind, "value": value,
                                 "mechanism": {"handled": handled, "synthetic": exception is None},
                                 "stacktrace": {"type": "raw", "frames": frames}}
                                for kind, value, frames in zip(kinds, values, stacks)],
            "thread": threading.current_thread().name,
            "repeats": repeats,
        })

    @staticmethod
    def _error_text(text: str) -> str:
        return scrub(redact_for_storage(text))[:ERROR_MESSAGE_LIMIT]

    def _error_due(self, place: str) -> Optional[int]:
        """How many errors of *place* were held back since its last event, when one is due
        now — None when this one is held back too."""
        now = time.time()
        with self._lock:
            sent, held = self._error_places.get(place, (0.0, 0))
            if now - sent < ERROR_INTERVAL:
                self._error_places[place] = (sent, held + 1)
                return None
            if len(self._error_places) >= ERROR_PLACES_LIMIT:
                self._error_places.clear()
            self._error_places[place] = (now, 0)
        return held

    def feed_count(self, feed: str, count: int):
        """How many items *feed* holds (photos indexed, messages read in); the feed's thread
        says so on every pass, and the next `status` carries the number of that moment."""
        with self._lock:
            self._feed_counts[feed] = count

    def feed_stopped(self, feed: str):
        """*feed*'s thread has ended: the next `status` lists running feeds only."""
        with self._lock:
            if self._feed_counts.pop(feed, None) is not None:
                trace(f"Telemetry: feed {feed} stopped, out of the status from here.")

    def _prompt_sizes(self) -> dict:
        """The prompt as the console's Dashboard sizes it: `prompt_kb`, the whole of what the LLM
        is sent (template, sections and history), and `prompt_sections` with each part's KB.
        Empty before the prompt is built, or without a provider."""
        if self._prompt_stats is None:
            return {}
        try:
            stats = self._prompt_stats()
        except Exception as e:
            error(f"Telemetry: could not size the prompt for the status: {e}")
            return {}
        total = sum(size for _, size, _ in stats)
        if total <= 0:
            trace("Telemetry: no prompt built yet, the status goes without its size.")
            return {}
        return {"prompt_kb": round(total / 1024, 1),
                "prompt_sections": {name: round(size / 1024, 1) for name, size, _ in stats}}

    def status(self):
        """What the install holds: the count of every running feed, the integrations that
        are on and the size of the prompt. The worker calls this STATUS_FIRST_DELAY after start
        and every STATUS_INTERVAL from then; an event goes out when any of it differs from the
        one last sent — a catch-up shows as an hourly curve, a quiet install as nothing (the
        history grows with every turn, so a used install reports hourly)."""
        with self._lock:
            feeds = dict(self._feed_counts)
        snapshot = {"feeds": feeds, "integrations": sorted(config.enabled_integrations()), **self._prompt_sizes()}
        if snapshot == self._last_status:
            trace("Telemetry: status unchanged since the last one sent, nothing to report.")
            return
        # Recorded before the opt-out check, so an opted-out install is asked hourly, not more.
        self._last_status = snapshot
        if not self._on():
            return
        prompt = f", prompt {snapshot['prompt_kb']} KB" if "prompt_kb" in snapshot else ""
        trace(f"Telemetry: status — {len(snapshot['integrations'])} integration(s) on, "
              + (", ".join(f"{name} {count}" for name, count in feeds.items()) or "no running feed") + prompt)
        self._enqueue("status", snapshot)

    def heartbeat(self):
        """Once a day, after the one `start` sends: the install is alive, and what it is — so
        churn can be told from an uninstall, and features from the integrations people turn on."""
        if not self._on():
            return
        self._enqueue("heartbeat", self._vitals(startup=False))

    def _vitals(self, startup: bool) -> dict:
        return {
            "startup": startup,
            "os": platform.system(),
            "os_version": platform.mac_ver()[0] or platform.release(),
            "machine": platform.machine(),
            "integrations": sorted(config.enabled_integrations()),
            "content": self.content_allowed(),
        }

    def _enqueue(self, event: str, properties: dict):
        item = {
            "event": event,
            "distinct_id": self._id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "properties": {**properties, "version": Version.version(), "$lib": "keep"},
        }
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            with self._lock:
                self._dropped += 1
                dropped = self._dropped
            if dropped == 1 or dropped % 100 == 0:
                error(f"Telemetry: queue full, {dropped} event(s) dropped.")

    # --- worker ---------------------------------------------------------------------------

    def _run(self):
        next_heartbeat = time.time() + HEARTBEAT_INTERVAL
        next_status = time.time() + STATUS_FIRST_DELAY
        while True:
            due = min(next_heartbeat, next_status)
            batch, stop = self._take_batch(timeout=min(60.0, max(0.0, due - time.time())))
            if batch:
                self._send(batch)
            if stop:
                return
            if time.time() >= next_heartbeat:
                self.heartbeat()
                next_heartbeat += HEARTBEAT_INTERVAL
            if time.time() >= next_status:
                self.status()
                next_status = time.time() + STATUS_INTERVAL

    def _take_batch(self, timeout: float) -> tuple[list[dict], bool]:
        """Up to BATCH_LIMIT queued events — blocking *timeout* for the first — and whether
        the stop marker was among them."""
        batch: list[dict] = []
        try:
            first = self._queue.get(timeout=timeout)
        except queue.Empty:
            return batch, False
        items = [first]
        while len(items) < BATCH_LIMIT:
            try:
                items.append(self._queue.get_nowait())
            except queue.Empty:
                break
        stop = False
        for item in items:
            if item is _STOP:
                stop = True
            else:
                batch.append(item)
        return batch, stop

    def _send(self, batch: list[dict]):
        if not config.get("telemetry.enabled", True):
            info(f"Telemetry: telemetry.enabled is false now, {len(batch)} queued event(s) discarded.")
            return
        try:
            self._post({"api_key": POSTHOG_PROJECT_KEY, "batch": batch})
        except Exception as e:
            error(f"Telemetry: dropping {len(batch)} event(s), post failed: {e}")
