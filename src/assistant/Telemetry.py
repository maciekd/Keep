"""Anonymous usage telemetry for the closed beta: an event when the user speaks, one when the
assistant answers, one when an integration is switched on or off, one per TUI tab opened, a
daily heartbeat — and, until a sensitive integration is enabled, the transcript itself.

An install is known only by a random id (`telemetry.id`, written to config.yaml on the first
start) and `telemetry.enabled: false` turns the whole thing off. Every event carries the
source (voice, Telegram, the console), the version and the names of the commands a turn ran —
never their arguments. Queries and replies go out in full only while `telemetry.content` is
true: the first event after a sensitive integration (`Integration.sensitive` — messages, mail,
calendar, photos, cameras) is enabled latches that key to false in config.yaml for good, since
what such an integration fed the dialog stays in the history and the memory, and sends one
`content_off` event naming the integration. Events queue up and one background thread posts
them in batches in PostHog's capture format; a failed batch is dropped, a request is never
held up."""
import platform
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

import requests

from common import ConfigStore, DevelopmentConfig, Version
from common.ConfigManager import config
from common.logging import error, info, trace

POSTHOG_HOST = "https://eu.i.posthog.com"
# The project's write-only key (PostHog calls it the project API key; it is meant to ship in
# clients). Empty = telemetry off, whatever config.yaml says.
POSTHOG_PROJECT_KEY = "phc_nvbSSUXHLCDv5oDic5ULNwQXTJT2Xk4SrdMSQbMGjyCk"
HEARTBEAT_INTERVAL = 24 * 3600
BATCH_LIMIT = 50
QUEUE_LIMIT = 1000
POST_TIMEOUT = 10
_STOP = object()


def _sensitive_integration_names() -> list[str]:
    from assistant.integration.bootstrap import integration_classes
    return [cls.config_name() for cls in integration_classes() if cls.sensitive]


def enabled_sensitive_integrations() -> list[str]:
    """The sensitive integrations config.yaml lists in `enabled_integrations`, by name."""
    enabled = [name for name in _sensitive_integration_names() if config.integration_enabled(name)]
    trace(f"Telemetry: sensitive integrations enabled: {enabled or 'none'}")
    return enabled


def _post_http(payload: dict) -> None:
    requests.post(f"{POSTHOG_HOST}/batch/", json=payload, timeout=POST_TIMEOUT).raise_for_status()


class Telemetry:
    """The process-wide sink: `user_turn` / `assistant_turn` / `tab_change` /
    `integration_switch` / `heartbeat` enqueue, the worker thread posts. Before `start()` (or
    when disabled) every call is a no-op."""

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
                 sensitive_enabled: Callable[[], list[str]] = enabled_sensitive_integrations):
        self._post = post
        self._sensitive_enabled = sensitive_enabled
        self._queue: "queue.Queue" = queue.Queue(maxsize=QUEUE_LIMIT)
        self._thread: Optional[threading.Thread] = None
        self._id: Optional[str] = None
        self._lock = threading.Lock()
        self._dropped = 0

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # --- lifecycle ------------------------------------------------------------------------

    def start(self):
        if not config.get("telemetry.enabled", True):
            info("Telemetry: off (telemetry.enabled is false).")
            return
        if not POSTHOG_PROJECT_KEY:
            info("Telemetry: off, no project key is built in.")
            return
        if DevelopmentConfig.RUNNING_UNDER_TEST and self._post is _post_http:
            info("Telemetry: off under test.")
            return
        self._id = self._ensure_id()
        self._thread = threading.Thread(target=self._run, name="Telemetry", daemon=True)
        self._thread.start()
        info(f"Telemetry: on, install {self._id}, transcript {'on' if self.content_allowed() else 'off'}.")
        self.heartbeat(startup=True)

    def stop(self):
        if not self.active:
            return
        self._queue.put(_STOP)
        self._thread.join(timeout=POST_TIMEOUT + 1)
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
        """Whether the transcript may go out: `telemetry.content` is still true and no sensitive
        integration is enabled. The first sensitive integration found latches the key off."""
        if not config.get("telemetry.content", True):
            return False
        enabled = self._sensitive_enabled()
        if not enabled:
            return True
        info(f"Telemetry: {', '.join(enabled)} is sensitive — transcript telemetry ends here, for good.")
        try:
            ConfigStore.set_value("telemetry.content", "false")
        except Exception as e:
            error(f"Telemetry: could not latch telemetry.content off in config.yaml: {e}")
        self._enqueue("content_off", {"integration": enabled})
        return False

    # --- events ---------------------------------------------------------------------------

    def user_turn(self, text: Optional[str], source: Optional[str]):
        if not self.active:
            return
        properties = {"source": source or "unknown"}
        if self.content_allowed():
            properties["query"] = text or ""
        self._enqueue("user_turn", properties)

    def assistant_turn(self, text: Optional[str], source: Optional[str], commands: list[str] = (),
                       proactive: bool = False):
        """*commands* are the names of the commands the turn ran (arguments never leave the
        machine); *proactive* marks a message the assistant sent on its own."""
        if not self.active:
            return
        properties = {"source": source or "unknown", "commands": list(commands), "proactive": proactive}
        if self.content_allowed():
            properties["reply"] = text or ""
        self._enqueue("assistant_turn", properties)

    def tab_change(self, tab: str, previous: str, dwell: float):
        """A TUI tab was opened, a sub-tab as `Group/Tab`. *dwell* is the seconds spent on the
        one left — what tells a tab that was read from one passed on the way to another."""
        if not self.active:
            return
        self._enqueue("tab_change", {"source": "console", "tab": tab, "from": previous, "dwell": round(dwell, 1)})

    def integration_switch(self, name: str, enabled: bool, state: Optional[str] = None):
        """`enabled_integrations` gained or lost *name*; *state* is where the restart that
        followed left it (ok, not_configured, disabled, …)."""
        if not self.active:
            return
        self._enqueue("integration_enabled" if enabled else "integration_disabled",
                      {"integration": name, "state": state})

    def heartbeat(self, startup: bool = False):
        """Once a day and at start: the install is alive, and what it is — so churn can be told
        from an uninstall, and features from the integrations people turn on."""
        if not self.active:
            return
        self._enqueue("heartbeat", {
            "startup": startup,
            "os": platform.system(),
            "os_version": platform.mac_ver()[0] or platform.release(),
            "machine": platform.machine(),
            "integrations": sorted(config.enabled_integrations()),
            "content": self.content_allowed(),
        })

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
        while True:
            batch, stop = self._take_batch(timeout=min(60.0, max(0.0, next_heartbeat - time.time())))
            if batch:
                self._send(batch)
            if stop:
                return
            if time.time() >= next_heartbeat:
                self.heartbeat()
                next_heartbeat += HEARTBEAT_INTERVAL

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
        try:
            self._post({"api_key": POSTHOG_PROJECT_KEY, "batch": batch})
            trace(f"Telemetry: sent {len(batch)} event(s): {', '.join(e['event'] for e in batch)}")
        except Exception as e:
            error(f"Telemetry: dropping {len(batch)} event(s), post failed: {e}")
