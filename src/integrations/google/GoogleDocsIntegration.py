"""The edits of listed Google Docs, as a feed of diffs."""
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from assistant.integration.Accounts import STATE_ERROR, STATE_OK, STATE_UNKNOWN, named_entries, slug
from assistant.integration.Feed import Feed, FeedBatch, FeedEntry
from assistant.integration.Integration import Integration
from common.ConfigManager import ConfigManager
from common.Paths import CACHE_DIR
from common.logging import error, info, trace
from integrations.google.DocDiff import describe_change, tracking_note
from integrations.google.GoogleAuth import (SCOPE_DRIVE_READONLY, GoogleAuth, NotSignedIn, not_signed_in_hint,
                                            sign_in_again_hint)
from integrations.google.GoogleDrive import DocHead, GoogleDrive, GoogleDriveError, Revision
from integrations.google.GoogleIntegration import ACCOUNTS_KEY, CLIENT_ID_KEY, SECTION, GoogleIntegration, account_scopes, docs_of

FEED_NAME = "Google Docs"
CONFIG_SLUG = "google_docs"
POLL_INTERVAL = 600          # seconds between polls; feeds.google_docs.poll_seconds overrides it
POLL_SECONDS_KEY = f"feeds.{CONFIG_SLUG}.poll_seconds"
TEXT_CACHE_DIR = CACHE_DIR / "google_docs"
NO_DOCS = "no document listed under google.accounts[].docs"


class WatchedDoc:
    """One document of one account: its id, the name the rows call it, the text of the last
    revision handled, cached on disk (`cache/google_docs/<id>.txt`), and the revision list of
    the head last seen, so a plan and the fetch after it list the revisions once."""

    def __init__(self, account: str, drive: GoogleDrive, doc_id: str, name: str):
        self.account = account
        self.drive = drive
        self.id = doc_id
        self.name = name or doc_id
        self.problem: Optional[str] = None     # why the doc is left out of the poll, when it is
        self._revisions: Optional[tuple[float, list[Revision]]] = None    # (head modifiedTime, the list)

    @property
    def label(self) -> str:
        return f"«{self.name}» ({self.account})"

    @property
    def cache_path(self) -> Path:
        return TEXT_CACHE_DIR / f"{slug(self.id)}.txt"

    def cached_text(self) -> Optional[str]:
        try:
            return self.cache_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None

    def cache(self, text: str) -> None:
        TEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, self.cache_path)

    def revisions(self, head: DocHead) -> list[Revision]:
        """The revisions as of `head`, listed once per head."""
        if self._revisions is None or self._revisions[0] != head.modified_ts:
            self._revisions = (head.modified_ts, self.drive.revisions(self.id))
        return self._revisions[1]


@dataclass
class Step:
    """One export's worth of work: the revisions of one day folded into one row, diffed at
    the day's last revision. `baseline` is the revision before the horizon to export first,
    when the doc is seen for the first time and has one; `tracking` is the first sight without
    one — the target becomes the baseline and the row says the doc is now tracked."""
    doc: WatchedDoc
    target: Revision
    revisions: list[Revision]
    baseline: Optional[Revision] = None
    tracking: bool = False

    @property
    def authors(self) -> str:
        names = list(dict.fromkeys(r.author for r in self.revisions))
        return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def day_of(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts))


class GoogleDocsIntegration(Integration, Feed):
    """The edits of listed Google Docs, as a feed of diffs.

    Follows the Google Docs listed under google.accounts[].docs and writes one row per day a
    document was edited — who changed it and what changed, as a diff — so Keep knows what was
    added or rewritten without ever reading the whole document.

    An Integration plus Feed in the google folder, beside GoogleIntegration, whose sign-in
    (google_login) and token files it reads; the Google integration must be on. The source is
    the Drive revision list of each doc (one revision per editing session, with its author),
    the text/plain export of a revision diffed against the text of the one before, cached under
    cache/google_docs/ — Google merges and prunes old revisions, so the cache is the baseline,
    never an older revision. Exporting a large doc costs Google ~25 s whatever the revision
    (433k chars in 27 s measured), so the unit of work is a day: the revisions of one day fold
    into one row diffed at the day's last revision, each fetch does one such step, and
    `remaining` counts the steps ahead so the catch-up shows progress and a restart resumes at
    the last export. A doc whose head moved within the last poll interval is held: the session
    may still be open, and one row per session beats one per poll. The cursor is a JSON map of
    doc id → epoch of the last revision handled; a doc the map does not know starts at the
    backfill horizon, with the newest revision before it as the baseline when there is one.
    Every poll costs one files.get per doc; the revisions are listed only when modifiedTime moved."""
    sensitive = True
    default_backfill_months = 1
    item_noun = "changes"

    def __init__(self):
        Integration.__init__(self)
        self._docs: list[WatchedDoc] = self._build_docs()
        self._open_error: Optional[str] = None
        self._opened = False
        self._lock = threading.Lock()

    def _on_reset(self):
        from assistant.integration.FeedRunner import Feeds
        Feeds.get_instance().stop(self.feed_name)

    @property
    def poll_interval(self) -> float:
        """Seconds between polls, read live: a change in config.yaml is in force at the next poll."""
        value = ConfigManager.get_instance().get(POLL_SECONDS_KEY, POLL_INTERVAL)
        try:
            seconds = float(value)
            if seconds > 0:
                return seconds
        except (TypeError, ValueError):
            pass
        info(f"Google Docs: {POLL_SECONDS_KEY}={value!r} is not a number of seconds, polling every {POLL_INTERVAL}s")
        return float(POLL_INTERVAL)

    @property
    def name(self) -> str:
        return FEED_NAME

    @property
    def config_keys(self) -> List[str]:
        return [CLIENT_ID_KEY, ACCOUNTS_KEY]

    @classmethod
    def wipe_targets(cls) -> list[tuple[Path, str]]:
        return [(TEXT_CACHE_DIR, "Google Docs text cache (a copy of every tracked document)")]

    # --- building ----------------------------------------------------------------------------

    @classmethod
    def _build_docs(cls) -> list[WatchedDoc]:
        config = ConfigManager.get_instance()
        docs: list[WatchedDoc] = []
        for index, entry in named_entries(config.get(ACCOUNTS_KEY, []), SECTION):
            listed = docs_of(entry)
            if not listed:
                continue
            name = str(entry["name"]).strip()
            address = lambda i=index: str(GoogleIntegration._entry(i).get("address") or "")
            auth = GoogleAuth(name, GoogleIntegration._client_settings, lambda e=entry: account_scopes(e), address)
            drive = GoogleDrive(name, auth)
            docs += [WatchedDoc(name, drive, doc["id"], doc["name"]) for doc in listed]
        info(f"Google Docs: {len(docs)} document(s) listed: {', '.join(d.label for d in docs) or 'none'}")
        return docs

    def docs(self) -> list[WatchedDoc]:
        return list(self._docs)

    # --- health ------------------------------------------------------------------------------

    def check_health(self) -> bool:
        if self._open_error:
            raise RuntimeError(self._open_error)
        return self._opened

    def accounts_status(self) -> list[dict]:
        status = []
        for doc in self._docs:
            problem = doc.problem or self._auth_problem(doc)
            state = STATE_ERROR if problem else (STATE_OK if self._opened else STATE_UNKNOWN)
            status.append({"name": doc.label, "state": state, "last_error": problem})
        return status

    @staticmethod
    def _auth_problem(doc: WatchedDoc) -> Optional[str]:
        auth = doc.drive.auth
        if not auth.signed_in():
            return not_signed_in_hint(doc.account)
        if auth.needs_sign_in_again():
            return sign_in_again_hint(doc.account)
        if not auth.has_scope(SCOPE_DRIVE_READONLY):
            return f"signed in without the Drive scope — {sign_in_again_hint(doc.account)}"
        return None

    # --- Feed --------------------------------------------------------------------------------

    @property
    def feed_name(self) -> str:
        return FEED_NAME

    @property
    def config_slug(self) -> str:
        return CONFIG_SLUG

    @property
    def open_error(self) -> Optional[str]:
        return self._open_error

    def open_feed(self) -> bool:
        if not self._docs:
            self._open_error = NO_DOCS
            info(f"Google Docs: {NO_DOCS}")
            return False
        problems = []
        for doc in self._docs:
            doc.problem = self._auth_problem(doc)
            if doc.problem is None:
                try:
                    head = doc.drive.head(doc.id)
                    if not head.is_document:
                        doc.problem = f"not a Google Doc ({head.mime_type or 'unknown type'})"
                    elif not doc.name or doc.name == doc.id:
                        doc.name = head.name or doc.id
                except (NotSignedIn, GoogleDriveError) as e:
                    doc.problem = str(e)
                except Exception as e:
                    doc.problem = f"{type(e).__name__}: {e}"
            if doc.problem:
                info(f"Google Docs: {doc.label} is left out — {doc.problem}")
                problems.append(f"{doc.label}: {doc.problem}")
            else:
                info(f"Google Docs: following {doc.label}")
        if len(problems) == len(self._docs):
            self._open_error = "; ".join(problems)
            return False
        self._open_error = None
        self._opened = True
        return True

    def close_feed(self):
        self._opened = False

    def initial_cursor(self, horizon_ts: float) -> Optional[str]:
        return "{}"

    def cursor_reached(self, cursor: Optional[str], target: Optional[str]) -> bool:
        have, want = self._cursor_map(cursor), self._cursor_map(target)
        return all(have.get(doc_id, 0.0) >= ts for doc_id, ts in want.items())

    @staticmethod
    def _cursor_map(cursor: Optional[str]) -> dict[str, float]:
        if not cursor:
            return {}
        try:
            loaded = json.loads(cursor)
            return {str(k): float(v) for k, v in loaded.items()} if isinstance(loaded, dict) else {}
        except (ValueError, TypeError):
            info(f"Google Docs: cursor {cursor!r} is not a map, starting over from the horizon")
            return {}

    def remaining(self, cursor: Optional[str]) -> Optional[int]:
        return len(self._steps(self._cursor_map(cursor)))

    def fetch(self, cursor: Optional[str], backfill: bool) -> FeedBatch:
        positions = self._cursor_map(cursor)
        steps = self._steps(positions)
        if not steps:
            return FeedBatch(entries=[], cursor=json.dumps(positions), exhausted=True)
        step = steps[0]
        entries = self._run(step)
        positions[step.doc.id] = step.target.modified_ts
        left = len(steps) - 1
        if left:
            info(f"Google Docs: {left} day(s) of edits still to read")
        return FeedBatch(entries=entries, cursor=json.dumps(positions), exhausted=left == 0, remaining=left)

    # --- the plan ----------------------------------------------------------------------------

    def _steps(self, positions: dict[str, float]) -> list[Step]:
        """The steps ahead of the cursor over every doc, in config order."""
        from assistant.integration.FeedRunner import Feeds
        steps: list[Step] = []
        now = time.time()
        for doc in self._docs:
            if doc.problem:
                continue
            since = positions.get(doc.id)
            if since is None:
                since = Feeds.horizon_for(self)
                trace(f"Google Docs: {doc.label} is new to the cursor, reading from the horizon")
            steps += self._plan(doc, since, now)
        return steps

    def _plan(self, doc: WatchedDoc, since: float, now: float) -> list[Step]:
        head = doc.drive.head(doc.id)
        if head.modified_ts <= since:
            trace(f"Google Docs: {doc.label} unchanged since the cursor, nothing to do")
            return []
        revisions = doc.revisions(head)
        new = [r for r in revisions if r.modified_ts > since]
        if not new:
            trace(f"Google Docs: {doc.label} moved but has no revision past the cursor yet")
            return []
        steps: list[Step] = []
        baseline: Optional[Revision] = None
        if doc.cached_text() is None:
            before = [r for r in revisions if r.modified_ts <= since]
            if before:
                baseline = before[-1]
            else:
                first = new.pop(0)
                steps.append(Step(doc, first, [first], tracking=True))
        days: dict[str, list[Revision]] = {}
        for revision in new:
            days.setdefault(day_of(revision.modified_ts), []).append(revision)
        for group in days.values():
            steps.append(Step(doc, group[-1], group, baseline=baseline if not steps else None))
        if steps and now - head.modified_ts < self.poll_interval:
            held = steps.pop()
            trace(f"Google Docs: {doc.label} was edited {int(now - head.modified_ts)} s ago, the session may be open — "
                  f"{len(held.revisions)} revision(s) of {day_of(held.target.modified_ts)} wait")
        return steps

    def _run(self, step: Step) -> list[FeedEntry]:
        doc = step.doc
        with self._lock:
            if step.tracking:
                return self._track(doc, step.target)
            baseline = doc.cached_text()
            if baseline is None:
                if step.baseline is None:
                    info(f"Google Docs: {doc.label} has no cached text and no baseline revision, tracking from {step.target.id}")
                    return self._track(doc, step.target)
                info(f"Google Docs: {doc.label} seen for the first time, revision {step.baseline.id} before the horizon is the baseline")
                baseline = doc.drive.export(doc.id, step.baseline)
            text = doc.drive.export(doc.id, step.target)
            content = describe_change(doc.name, step.authors, baseline, text, edits=len(step.revisions))
            doc.cache(text)
        day = day_of(step.target.modified_ts)
        if content is None:
            trace(f"Google Docs: the {len(step.revisions)} revision(s) of {doc.label} on {day} by {step.authors} changed no paragraph")
            return []
        info(f"Google Docs: {doc.label} on {day}, {len(step.revisions)} revision(s) by {step.authors} — {content.splitlines()[0]}")
        return [FeedEntry(step.target.modified_ts, content, location=doc.name)]

    @staticmethod
    def _track(doc: WatchedDoc, target: Revision) -> list[FeedEntry]:
        """The first row of a doc with no baseline: its text is cached and noted, not diffed.
        Called with the lock held — it takes none itself, so the fallback above cannot deadlock."""
        text = doc.drive.export(doc.id, target)
        doc.cache(text)
        info(f"Google Docs: {doc.label} seen for the first time with no earlier revision, now tracking from {target.id}")
        return [FeedEntry(target.modified_ts, tracking_note(doc.name, text), location=doc.name)]
