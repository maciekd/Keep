"""The Photos library as a Feed (plans/FEED_PLAN.md): `fetch` lists the assets created after the
cursor, indexes the unindexed ones through PhotoProcessor and hands their history rows to
FeedRunner; `sweep`, asked while the feed is live, does the same for the assets behind the
cursor that have no row (a batch the provider refused, rows a migration forgot). The `photos`
table is the ledger, so re-reading a stretch of the library costs a query and nothing else; the
cursor (an Apple creation timestamp) only decides where the next poll starts."""
import os
import sqlite3
import threading
import time

from typing import List, Optional

from assistant.integration.Feed import Feed, FeedBatch, FeedEntry
from assistant.integration.Integration import Integration, IntegrationState
from assistant.llm import tier_api_key_name
from common.ConfigManager import config
from common.DevelopmentConfig import LOCAL_DEVELOPMENT_MODE
from common.FloxyDatabase import FloxyDatabase, PHOTO_HISTORY_SOURCE
from common.logging import info, error, fine, trace, note
from .PhotoProcessor import (PhotoProcessor, PHOTOS_SQLITE_PATH, SYNDICATION_SAVED_ASSET_TYPE,
                             ASSET_COLUMNS, APPLE_EPOCH_OFFSET, asset_kind_filter, asset_row)

POLL_INTERVAL = 10
FETCH_BATCH_SIZE = 100
# A just-taken photo has no ZCLOUDASSETGUID until its iCloud upload completes, and the index
# only stores portable GUIDs, so the cursor holds below such a photo and re-reads it every
# poll. An asset still without a GUID after this long is not uploading any more (iCloud Photos
# off, a stuck upload): the cursor moves past it and the Photos tab's bulk indexing is what
# finds it later.
GUID_HOLD_SECONDS = 24 * 3600
ICLOUD_CHECK_INTERVAL_SECONDS = 60 * 60  # Check iCloud download once per hour


def assets_after_query() -> str:
    """Assets (stills, and videos when enabled) created after the cursor, oldest first."""
    return f"""
SELECT {ASSET_COLUMNS}
FROM ZASSET a
LEFT JOIN ZCLOUDMASTER m ON m.Z_PK = a.ZMASTER
WHERE {asset_kind_filter()} AND a.ZTRASHEDSTATE = 0 AND a.ZSAVEDASSETTYPE != {SYNDICATION_SAVED_ASSET_TYPE}
  AND a.ZDATECREATED > ?
ORDER BY a.ZDATECREATED ASC
LIMIT ?
"""


class PhotosIntegration(Integration, Feed):
    """The Photos library, indexed by a vision model and searchable by words.

    Indexes the macOS Photos library with a vision model, so Keep can search the family's photos
    and videos and show them. What it indexes also lands in Keep's history, as a feed.

    Pours one history row per asset into the dialog history, as a Feed. check_health() verifies
    the library database and, once an hour, that iCloud photo downloads still work."""
    sensitive = True

    poll_interval = POLL_INTERVAL

    def __init__(self):
        super().__init__()
        self._processor: Optional[PhotoProcessor] = None
        self._open_error: Optional[str] = None
        # history rows the processor produced (its worker threads, the iCloud retries) — handed
        # over with the batch that is being fetched
        self._pending_entries: list[FeedEntry] = []
        self._entries_lock = threading.Lock()
        self._held: set[str] = set()     # uuids the cursor is holding for (no iCloud GUID yet)
        self._last_icloud_check: Optional[float] = None

    def _on_reset(self):
        from assistant.integration.FeedRunner import Feeds
        Feeds.get_instance().stop(self.feed_name)

    @property
    def name(self) -> str:
        return "Photos"

    @property
    def config_keys(self) -> List[str]:
        """The key of whichever backend the `photo_indexing` tier runs on (Gemini by
        default). Hardcoding a provider here once had list_integrations tag Photos with
        the Anthropic key while it indexed on Gemini Flash, and the FTUE assistant "fixed"
        a problem that did not exist."""
        key = tier_api_key_name('photo_indexing')
        return [key] if key else []

    def _provider_refused(self, reason: str, out_of_credits: bool, cursor: Optional[str]):
        """A rate limit clears by itself: the feed keeps polling (every POLL_INTERVAL) and each
        poll is refused without a call until the backoff passes, so nothing to tell anyone. An
        empty prepaid account needs the user: the feed pauses, the integration shows RATE_LIMITED
        with the reason (the Catalog tab, the prompt's INTEGRATIONS section), and the household
        hears it once — the health check probes the key and resumes, with one more note."""
        if not out_of_credits:
            trace(f"Photos: cursor held at {cursor} — {reason}")
            return
        if self.state == IntegrationState.RATE_LIMITED:
            trace("Photos: still out of credits, feed stays paused")
            return
        from assistant.integration.FeedRunner import Feeds
        from assistant.SystemNote import system_note
        self.state = IntegrationState.RATE_LIMITED
        self._last_error = reason
        runner = Feeds.get_instance().runner(self.feed_name)
        if runner is not None:
            runner.pause()
        waiting = self.remaining(cursor)
        note(f"Photos: indexing paused — {reason}")
        system_note("Photos", (
            "Photo indexing is paused: the Gemini API key's prepaid credits are used up (HTTP 402)."
            + (f" {waiting} photos and videos are waiting." if waiting else "")
            + " Nothing is lost — once the Google AI Studio project is topped up, indexing resumes "
              "by itself within 15 minutes."))

    def _provider_back(self):
        from assistant.integration.FeedRunner import Feeds
        from assistant.SystemNote import system_note
        self._last_error = None
        runner = Feeds.get_instance().runner(self.feed_name)
        if runner is not None:
            runner.resume()
        note("Photos: the Gemini key takes requests again — indexing resumed")
        system_note("Photos", "Photo indexing has resumed: the Gemini API key accepts requests again.")

    def check_health(self) -> bool:
        if self.state == IntegrationState.RATE_LIMITED:
            processor = self._processor or PhotoProcessor()
            if not processor._get_llm().probe():
                raise RuntimeError(self._last_error or "the indexing LLM still refuses requests")
            self._provider_back()
        if not os.path.exists(PHOTOS_SQLITE_PATH):
            raise FileNotFoundError(f"Photos database not found at {PHOTOS_SQLITE_PATH}")
        with sqlite3.connect(f"file:{PHOTOS_SQLITE_PATH}?mode=ro", uri=True) as conn:
            conn.execute("SELECT 1 FROM ZASSET LIMIT 1")
        now = time.time()
        if (self._last_icloud_check is None
                or now - self._last_icloud_check >= ICLOUD_CHECK_INTERVAL_SECONDS):
            self._last_icloud_check = now
            self._check_icloud_download()
        return True

    def _check_icloud_download(self):
        """Pick a random photo with a cloud GUID and verify iCloud download works."""
        if not config.get("icloud", {}):
            info("Photos: iCloud check skipped — iCloud not configured")
            return
        try:
            if not os.path.exists(PHOTOS_SQLITE_PATH):
                error("Photos: iCloud check skipped — Photos database not found")
                return
            conn = sqlite3.connect(f"file:{PHOTOS_SQLITE_PATH}?mode=ro", uri=True)
            try:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT a.ZUUID, m.ZCLOUDMASTERGUID, a.ZLIBRARYSCOPE "
                    "FROM ZASSET a "
                    "LEFT JOIN ZCLOUDMASTER m ON m.Z_PK = a.ZMASTER "
                    "WHERE a.ZKIND = 0 AND a.ZTRASHEDSTATE = 0 AND a.ZHIDDEN = 0 "
                    "AND m.ZCLOUDMASTERGUID IS NOT NULL "
                    "ORDER BY RANDOM() LIMIT 1")
                row = cursor.fetchone()
            finally:
                conn.close()

            if not row:
                info("Photos: iCloud check skipped — no photos with cloud GUID found")
                return

            uuid = row["ZUUID"]
            cloud_guid = row["ZCLOUDMASTERGUID"]
            library_scope = row["ZLIBRARYSCOPE"] or 0
            info(f"Photos: iCloud download check — testing with {uuid}")

            from .ICloudPhotoDownloader import ICloudPhotoDownloader
            result = ICloudPhotoDownloader.get_instance().download_photo_bytes(
                cloud_guid, library_scope)

            if result:
                info(f"Photos: iCloud download check OK ({len(result)} bytes)")
            else:
                error(f"Photos: iCloud download check FAILED — got no data for {uuid}")
        except Exception as e:
            error(f"Photos: iCloud download check FAILED — {e}")

    # --- Feed ---------------------------------------------------------------------------------

    @property
    def feed_name(self) -> str:
        return PHOTO_HISTORY_SOURCE

    @property
    def config_slug(self) -> str:
        return "photos"

    @property
    def open_error(self) -> Optional[str]:
        return self._open_error

    def open_feed(self) -> bool:
        if not os.path.exists(PHOTOS_SQLITE_PATH):
            self._open_error = f"Photos database not found at {PHOTOS_SQLITE_PATH}"
            error(f"Photos: {self._open_error}")
            return False
        self._open_error = None
        self._processor = PhotoProcessor(entry_sink=self._queue_entry)
        self._held = set()
        self._processor.remove_hidden_assets()
        info("Photos feed opened")
        return True

    def close_feed(self):
        """A batch that was cut short (stop while the workers ran) has its photos in the ledger
        but their history rows only here — write them now, or they are never written."""
        entries = self._drain_entries()
        if entries:
            info(f"Photos: writing {len(entries)} history row(s) left over from the last batch")
            FloxyDatabase.get_instance().add_history_entries(
                [(e.role, e.content, e.timestamp, e.location) for e in entries], PHOTO_HISTORY_SOURCE)

    def _catch_up_blocker(self) -> Optional[str]:
        """Why the catch-up must not index anything, or None. The catch-up is bulk work and
        never runs on a development machine."""
        # if LOCAL_DEVELOPMENT_MODE:    testing FTUT indexing now so keep it on for now
        #     return "development mode"
        return None

    def initial_cursor(self, horizon_ts: float) -> Optional[str]:
        """The horizon as an Apple creation timestamp — or now, when the catch-up is not
        allowed to run: new photos still get indexed one by one as they arrive."""
        blocker = self._catch_up_blocker()
        if blocker:
            note(f"Photos: not catching up on older photos — {blocker}; indexing new photos only")
            return str(time.time() - APPLE_EPOCH_OFFSET)
        return str(horizon_ts - APPLE_EPOCH_OFFSET)

    def remaining(self, cursor: Optional[str]) -> Optional[int]:
        processor = self._processor or PhotoProcessor()
        return processor.count_unindexed_photos(after_apple_ts=float(cursor or 0))

    def count_stored(self) -> Optional[int]:
        """The index table, not the history: 73k assets were indexed before a photo got a
        history row (2026-03), and `remaining` is measured against the same table."""
        return FloxyDatabase.get_instance().get_photo_count()

    def in_flight(self) -> int:
        """Assets of the current batch the workers have finished: each one queued its history row."""
        with self._entries_lock:
            return len(self._pending_entries)

    def fetch(self, cursor: Optional[str], backfill: bool) -> FeedBatch:
        processor = self._processor
        processor._retry_icloud_pending()
        after = float(cursor or 0)
        rows = self._assets_after(after, FETCH_BATCH_SIZE)
        todo = []
        if rows:
            indexed = FloxyDatabase.get_instance().get_indexed_photo_guids()
            todo = [r for r in rows if r["cloud_asset_guid"] and r["cloud_asset_guid"] not in indexed]
            if not todo:
                trace(f"Photos: {len(rows)} asset(s) after the cursor, all indexed or awaiting iCloud")
        if self._index(todo, " (catching up)" if backfill else "", batched=backfill, cursor=cursor):
            # The provider refused (out of credits, rate limited): the cursor stays, so the rows
            # nobody described come round again, and the batch hands over what did get stored.
            return FeedBatch(self._drain_entries(), cursor=cursor, exhausted=True)
        new_cursor, held = self.advance_cursor(after, rows, time.time())
        if held != self._held:
            if held:
                info(f"Photos: holding the cursor — {len(held)} asset(s) have no iCloud asset GUID yet "
                     f"(still uploading), re-polling until they sync")
            else:
                fine("Photos: every held asset has synced, cursor moving on")
            self._held = held
        # A full batch means more is waiting — unless the cursor stayed put (held below an asset
        # still uploading): the next fetch would return these same rows, so wait for the poll.
        exhausted = len(rows) < FETCH_BATCH_SIZE or new_cursor == after
        if not exhausted:
            trace(f"Photos: full batch of {len(rows)}, cursor {after} → {new_cursor}, fetching on")
        elif len(rows) >= FETCH_BATCH_SIZE:
            trace(f"Photos: full batch but the cursor is held at {after}, nothing more until the next poll")
        return FeedBatch(self._drain_entries(), cursor=str(new_cursor), exhausted=exhausted)

    def _index(self, todo: list, why: str, batched: bool, cursor: Optional[str]) -> bool:
        """Index `todo` (nothing to do is fine) and tell whether the provider refused: it is
        then marked (`_provider_refused`, which pauses the feed when the credits are gone) and
        the caller must leave the cursor where it is, so the rows nobody described come round
        again. With `batched` the stills go through the batch API — the catch-up and the sweep
        are the work nobody waits for."""
        processor = self._processor
        refused = []
        if todo:
            num_videos = sum(1 for r in todo if r["kind"] == "video")
            info(f"Photos: {len(todo) - num_videos} photo(s) and {num_videos} video(s) to index{why}")
            processor.process_rows(todo, batched=batched)
            refused = processor.refused_last_run()
        reason = processor.provider_unavailable()
        if refused and not reason:
            # A parallel worker's success or a lapsed backoff cleared the shared outage before
            # the batch was over; the rows refused meanwhile still have no description.
            reason = f"the indexing backend refused {len(refused)} asset(s) during the batch"
        if reason:
            self._provider_refused(reason, processor.provider_out_of_credits(), cursor)
            return True
        return False

    def sweep(self, horizon_ts: float, cursor: Optional[str]) -> FeedBatch:
        """The assets between the horizon and the cursor that have no index row: a batch the
        provider refused halfway (a 402 or 429 before the cursor learned to hold), rows a
        migration forgot. `fetch` reads only past the cursor and never comes back for them, so
        the live feed indexes them in the background, oldest first, at the backfill's price.
        Re-reading the stretch costs a query: everything with a row is left alone."""
        processor = self._processor
        since, until = horizon_ts - APPLE_EPOCH_OFFSET, float(cursor or 0)
        waiting = processor.count_unindexed_photos(after_apple_ts=since, before_apple_ts=until)
        if not waiting:
            trace("Photos: sweep found nothing unindexed between the horizon and the cursor")
            return FeedBatch([], cursor=cursor, exhausted=True, remaining=0)
        rows = processor.unindexed_between(since, until, FETCH_BATCH_SIZE)
        note(f"Photos: {waiting} asset(s) behind the cursor have no index row (back to "
             f"{time.strftime('%Y-%m-%d', time.localtime(horizon_ts))}), indexing {len(rows)} of them now")
        if self._index(rows, " (sweeping behind the cursor)", batched=True, cursor=cursor):
            return FeedBatch(self._drain_entries(), cursor=cursor, exhausted=True, remaining=waiting)
        left = processor.count_unindexed_photos(after_apple_ts=since, before_apple_ts=until)
        if left >= waiting:
            # nothing got a row (every download failed, the batch job is still unanswered): the
            # same rows would come straight back, so wait for the next sweep instead
            info(f"Photos: the sweep made no progress ({left} still unindexed), trying again later")
            return FeedBatch(self._drain_entries(), cursor=cursor, exhausted=True, remaining=left)
        return FeedBatch(self._drain_entries(), cursor=cursor, exhausted=left == 0 or len(rows) < FETCH_BATCH_SIZE,
                         remaining=left)

    @staticmethod
    def advance_cursor(current: float, rows: list, now: float) -> tuple[float, set[str]]:
        """The cursor after `rows` (creation date ASC, from a strict `ZDATECREATED > cursor`)
        and the uuids it is holding for.

        The cursor moves across assets that have an iCloud asset GUID, and across GUID-less
        ones older than GUID_HOLD_SECONDS, but stays strictly below the earliest asset that is
        still uploading. Anchoring on that asset's timestamp rather than on the iteration
        position matters because the poll's ORDER BY has no tiebreaker: a GUID-bearing sibling
        at the same timestamp iterated first would otherwise push the cursor onto the shared
        timestamp and the strict `>` would exclude the held asset from every later poll.
        Returns (cursor, held uuids)."""
        hold_before_apple = now - APPLE_EPOCH_OFFSET - GUID_HOLD_SECONDS
        held = {r["uuid"] for r in rows
                if not r["cloud_asset_guid"] and r["creation_date_apple"] is not None
                and r["creation_date_apple"] > hold_before_apple}
        held_ts = [r["creation_date_apple"] for r in rows if r["uuid"] in held]
        # Ceiling the cursor may approach but never reach: the earliest held asset's timestamp
        # (so it and any equal-timestamp siblings survive the strict `>`), else unbounded.
        ceiling = min(held_ts) if held_ts else None

        cursor = current
        for row in rows:
            ts = row["creation_date_apple"]
            if ts is None or row["uuid"] in held:
                continue
            if ceiling is not None and ts >= ceiling:
                continue
            if ts > cursor:
                cursor = ts
        return cursor, held

    @staticmethod
    def _assets_after(cursor: float, limit: int) -> list:
        conn = sqlite3.connect(f"file:{PHOTOS_SQLITE_PATH}?mode=ro", uri=True)
        try:
            conn.row_factory = sqlite3.Row
            return [asset_row(r) for r in conn.execute(assets_after_query(), (cursor, limit)).fetchall()]
        finally:
            conn.close()

    def _queue_entry(self, entry: FeedEntry):
        with self._entries_lock:
            self._pending_entries.append(entry)

    def _drain_entries(self) -> list[FeedEntry]:
        with self._entries_lock:
            entries = self._pending_entries
            self._pending_entries = []
        return entries
