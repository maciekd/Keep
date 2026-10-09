import sqlite3
from contextlib import closing
import threading
import time
from typing import Optional, List, Tuple

import re

from common.logging import info, error, trace, note

# Bumped when a one-off data migration in migrate_from_old_db_versions() must run against
# existing databases. SQLite keeps it in the file header (PRAGMA user_version); a fresh
# database is stamped with the current value before the migrations run, so it never replays them.
SCHEMA_VERSION = 3
# `source` of the history entries PhotoProcessor writes for indexed assets; the purge of hidden
# assets removes those rows by it.
PHOTO_HISTORY_SOURCE = "iCloud Photos"
NO_INTERACTIONS = "(No interactions in this period)"   # get_interaction_history with nothing in the period


def _tokenize_query(query: str) -> List[str]:
    return [t for t in query.strip().split() if t]


def _build_fts_query(query: str) -> str:
    tokens = _tokenize_query(query)
    parts = []
    for t in tokens:
        if re.search(r'[^a-zA-Z0-9]', t):
            # FTS5 escapes embedded double quotes by doubling them inside a quoted phrase.
            escaped = t.replace('"', '""')
            parts.append(f'"{escaped}"')
        else:
            parts.append(t)
    return " OR ".join(parts)


def _count_token_coverage(content: str, tokens: List[str]) -> int:
    if not content or not tokens:
        return 0
    lowered = content.lower()
    return sum(1 for t in tokens if t.lower() in lowered)

"""
To query the database:
SELECT 
    datetime(timestamp, 'unixepoch', 'localtime') as local_time,
    query,
    commands,
    response
FROM usage_stats
ORDER BY timestamp ASC;
"""


class FloxyDatabase:
    """Owns the floxy.db SQLite store for usage stats, response cache, dialog history and photo index.

    Singleton; every operation takes the instance lock so it is safe to call from any
    thread, and schema migrations run on first access.
    """

    _instance: Optional['FloxyDatabase'] = None
    # Relative to the cwd (AIServer/). The unit suite points it at a temp file so no test can
    # write into the developer's live database.
    DB_PATH = "floxy.db"

    @classmethod
    def get_instance(cls) -> 'FloxyDatabase':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        cls._instance = None

    def __init__(self):
        self.db_path = self.DB_PATH
        self.lock = threading.Lock()
        self._init_database()

    def _init_database(self):
        """Initialize the database and create tables if they don't exist."""
        with self.lock:
            info("Floxy database initializing...")

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            fresh = cursor.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'usage_stats'"
            ).fetchone()[0] == 0
            if fresh:
                # A new file has user_version 0 like the oldest databases; stamp it before the
                # migrations look so they see a current schema instead of one to upgrade.
                cursor.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                info(f"Fresh database, creating schema version {SCHEMA_VERSION} at {self.db_path}")

            # Create the main stats table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS usage_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    query TEXT NOT NULL,
                    commands TEXT,
                    response TEXT NOT NULL,
                    cached BOOLEAN DEFAULT 0,
                    not_cachable INTEGER DEFAULT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    source TEXT DEFAULT NULL,
                    location TEXT DEFAULT NULL
                )
            ''')

            self.migrate_from_old_db_versions(cursor)

            # Create indexes for better query performance
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_query 
                ON usage_stats(query)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON usage_stats(timestamp)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_cached
                ON usage_stats(cached)
            ''')

            # Photos table for iCloud Photos integration.
            # asset_guid is the iCloud asset GUID (ZASSET.ZCLOUDASSETGUID) — identical on
            # every machine synced to the same iCloud library, unlike the machine-local
            # PhotoKit UUID. This keeps the database portable between computers.
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS photos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset_guid TEXT NOT NULL UNIQUE,
                    creation_date REAL NOT NULL,
                    latitude REAL,
                    longitude REAL,
                    location_text TEXT,
                    albums TEXT,
                    persons TEXT,
                    photo_type TEXT,
                    short_description TEXT NOT NULL,
                    detailed_description TEXT NOT NULL,
                    pretty_score INTEGER NOT NULL,
                    family_importance_score INTEGER NOT NULL,
                    indexed_at REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    duration_seconds REAL
                )
            ''')

            cursor.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS photos_fts USING fts5(
                    short_description, detailed_description, location_text,
                    persons, albums, photo_type,
                    content='photos', content_rowid='id'
                )
            ''')

            # Triggers to keep FTS in sync
            cursor.execute('''
                CREATE TRIGGER IF NOT EXISTS photos_ai AFTER INSERT ON photos BEGIN
                    INSERT INTO photos_fts(rowid, short_description, detailed_description, location_text, persons, albums, photo_type)
                    VALUES (new.id, new.short_description, new.detailed_description, new.location_text, new.persons, new.albums, new.photo_type);
                END
            ''')
            cursor.execute('''
                CREATE TRIGGER IF NOT EXISTS photos_ad AFTER DELETE ON photos BEGIN
                    INSERT INTO photos_fts(photos_fts, rowid, short_description, detailed_description, location_text, persons, albums, photo_type)
                    VALUES ('delete', old.id, old.short_description, old.detailed_description, old.location_text, old.persons, old.albums, old.photo_type);
                END
            ''')
            cursor.execute('''
                CREATE TRIGGER IF NOT EXISTS photos_au AFTER UPDATE ON photos BEGIN
                    INSERT INTO photos_fts(photos_fts, rowid, short_description, detailed_description, location_text, persons, albums, photo_type)
                    VALUES ('delete', old.id, old.short_description, old.detailed_description, old.location_text, old.persons, old.albums, old.photo_type);
                    INSERT INTO photos_fts(rowid, short_description, detailed_description, location_text, persons, albums, photo_type)
                    VALUES (new.id, new.short_description, new.detailed_description, new.location_text, new.persons, new.albums, new.photo_type);
                END
            ''')

            cursor.execute('CREATE INDEX IF NOT EXISTS idx_photos_asset_guid ON photos(asset_guid)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_photos_creation_date ON photos(creation_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_photos_pretty_score ON photos(pretty_score)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_photos_family_score ON photos(family_importance_score)')

            # Conversation history table (replaces history.json)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    source TEXT DEFAULT NULL,
                    location TEXT DEFAULT NULL
                )
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_history_timestamp
                ON history(timestamp)
            ''')

            cursor.execute('''
                CREATE UNIQUE INDEX IF NOT EXISTS idx_history_dedup
                ON history(timestamp, role, content)
            ''')

            cursor.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS history_fts USING fts5(
                    content, role, source, location,
                    content='history', content_rowid='id'
                )
            ''')

            cursor.execute('''
                CREATE TRIGGER IF NOT EXISTS history_fts_ai AFTER INSERT ON history BEGIN
                    INSERT INTO history_fts(rowid, content, role, source, location)
                    VALUES (new.id, new.content, new.role, new.source, new.location);
                END
            ''')
            cursor.execute('''
                CREATE TRIGGER IF NOT EXISTS history_fts_ad AFTER DELETE ON history BEGIN
                    INSERT INTO history_fts(history_fts, rowid, content, role, source, location)
                    VALUES ('delete', old.id, old.content, old.role, old.source, old.location);
                END
            ''')
            cursor.execute('''
                CREATE TRIGGER IF NOT EXISTS history_fts_au AFTER UPDATE ON history BEGIN
                    INSERT INTO history_fts(history_fts, rowid, content, role, source, location)
                    VALUES ('delete', old.id, old.content, old.role, old.source, old.location);
                    INSERT INTO history_fts(rowid, content, role, source, location)
                    VALUES (new.id, new.content, new.role, new.source, new.location);
                END
            ''')

            # A feed's rows are counted straight from here (source = feed name, role = system):
            # the dashboard's totals and the catch-up progress.
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_history_source
                ON history(source, role)
            ''')

            # Where every feed (iMessage, Telegram, ...) got to: FeedRunner's only state.
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feeds (
                    name TEXT PRIMARY KEY,
                    cursor TEXT,
                    horizon REAL,
                    live_since REAL,
                    last_entry_ts REAL,
                    gap_until REAL,
                    resume_cursor TEXT,
                    gap_covered_until REAL,
                    updated_at REAL NOT NULL
                )
            ''')
            feed_columns = [col[1] for col in cursor.execute('PRAGMA table_info(feeds)').fetchall()]
            if 'gap_until' not in feed_columns:
                cursor.execute('ALTER TABLE feeds ADD COLUMN gap_until REAL')
                cursor.execute('ALTER TABLE feeds ADD COLUMN resume_cursor TEXT')
                note("Database migration: added 'gap_until' and 'resume_cursor' columns to feeds "
                     "(a horizon moved back re-reads only the gap).")
            if 'gap_covered_until' not in feed_columns:
                cursor.execute('ALTER TABLE feeds ADD COLUMN gap_covered_until REAL')
                note("Database migration: added 'gap_covered_until' column to feeds "
                     "(the newest entry stored before a horizon move-back; a gap read drops only up to it).")
            for column in ('entries', 'total'):
                # counted from the history since 2026-09: a lifetime counter drifted on purges and
                # started at 0 for rows older than the feeds table
                if column in feed_columns:
                    cursor.execute(f'ALTER TABLE feeds DROP COLUMN {column}')
                    note(f"Database migration: dropped '{column}' column from feeds (counts come from the history).")

            fts_count = cursor.execute("SELECT COUNT(*) FROM history_fts").fetchone()[0]
            history_count = cursor.execute("SELECT COUNT(*) FROM history").fetchone()[0]
            if fts_count == 0 and history_count > 0:
                note(f"History FTS index is empty but history has {history_count} entries — rebuilding...")
                cursor.execute("INSERT INTO history_fts(history_fts) VALUES('rebuild')")

            conn.commit()
            conn.close()

    @staticmethod
    def migrate_from_old_db_versions(cursor):
        cursor.execute('''
                PRAGMA table_info(usage_stats)
            ''')
        columns = [col[1] for col in cursor.fetchall()]
        if 'cached' not in columns:
            cursor.execute('''
                           ALTER TABLE usage_stats
                               ADD COLUMN cached BOOLEAN DEFAULT 0
                           ''')
            note("Database migration: added 'cached' column to existing database.")
        if 'not_cachable' not in columns:
            cursor.execute('''
                           ALTER TABLE usage_stats
                               ADD COLUMN not_cachable INTEGER DEFAULT NULL
                           ''')
            note("Database migration: added 'not_cachable' column to existing database.")
        if 'source' not in columns:
            cursor.execute('''
                           ALTER TABLE usage_stats
                               ADD COLUMN source TEXT DEFAULT NULL
                           ''')
            note("Database migration: added 'source' column to existing database.")
        if 'location' not in columns:
            cursor.execute('''
                           ALTER TABLE usage_stats
                               ADD COLUMN location TEXT DEFAULT NULL
                           ''')
            note("Database migration: added 'location' column to existing database.")

        cursor.execute('''
                PRAGMA table_info(photos)
            ''')
        photo_columns = [col[1] for col in cursor.fetchall()]
        if 'local_identifier' in photo_columns:
            cursor.execute('ALTER TABLE photos RENAME COLUMN local_identifier TO asset_guid')
            cursor.execute('DROP INDEX IF EXISTS idx_photos_local_id')
            note("Database migration: renamed photos.local_identifier to asset_guid. "
                 "Run migrate_photos_to_guid.py to convert legacy UUID values to iCloud asset GUIDs.")
        if photo_columns and 'duration_seconds' not in photo_columns:  # table exists, pre-video schema
            cursor.execute('ALTER TABLE photos ADD COLUMN duration_seconds REAL')
            note("Database migration: added 'duration_seconds' column to photos (videos).")
        version = cursor.execute("PRAGMA user_version").fetchone()[0]
        if photo_columns and version < 1:
            # One-off re-queues from the 2026-09 video indexing work. The rows they delete are
            # re-created with the same text by today's code, so they must run exactly once.
            # Videos skipped under the 20 MB inline cap fit the Files API path; the observer's
            # backlog query re-indexes them once the rows are gone.
            cursor.execute("DELETE FROM photos WHERE photo_type = 'skipped' "
                           "AND short_description = 'Skipped (video too large)'")
            if cursor.rowcount:
                note(f"Database migration: re-queued {cursor.rowcount} videos skipped as too large "
                     f"for the old inline video limit.")
            # Sub-2-second clips used to fail with INVALID_ARGUMENT because Gemini's default
            # 1 frame/s sampling saw no frames; they now get an fps override, so retry them.
            cursor.execute("DELETE FROM photos WHERE photo_type = 'skipped' "
                           "AND short_description = 'Skipped (LLM analysis failed)' AND duration_seconds < 2")
            if cursor.rowcount:
                note(f"Database migration: re-queued {cursor.rowcount} short videos whose LLM analysis "
                     f"failed before the short-clip fps override.")
            # Slo-mo clips keep their renditions on the iCloud asset record, which the downloader
            # only started consulting on 2026-09-03; those rows are videos (a duration) skipped as
            # download failures. Stills skipped for the same reason are unaffected and stay.
            cursor.execute("DELETE FROM photos WHERE photo_type = 'skipped' "
                           "AND short_description = 'Skipped (iCloud download failed)' "
                           "AND duration_seconds IS NOT NULL")
            if cursor.rowcount:
                note(f"Database migration: re-queued {cursor.rowcount} videos whose iCloud download failed "
                     f"before the asset-record rendition fallback.")
        has_history = cursor.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'history'").fetchone()[0]
        if has_history and version < 2:
            # History roles are stored lowercase since 2026-09 (add_history_entry lowercases);
            # before that a Telegram first name landed as "Alex", the voice fallback as "User"
            # and the console owner as "user", so a query by role had to guess the spelling.
            # A row whose lowercased key would collide with one already there is a duplicate
            # the dedup index only missed by case; it goes first so the UPDATE cannot violate it.
            cursor.execute('''
                DELETE FROM history WHERE id IN (
                    SELECT later.id FROM history later JOIN history earlier
                      ON earlier.timestamp = later.timestamp AND earlier.content = later.content
                     AND lower(earlier.role) = lower(later.role) AND earlier.id < later.id)
            ''')
            if cursor.rowcount:
                note(f"Database migration: dropped {cursor.rowcount} history rows duplicated only by role case.")
            cursor.execute("UPDATE history SET role = lower(role) WHERE role != lower(role)")
            note(f"Database migration: lowercased the role of {cursor.rowcount} history rows.")
        if photo_columns and version < 3:
            # Until 2026-10 a still whose LLM call failed — including a 429 or 402 from the
            # provider, i.e. a rate limit or an empty prepaid account — was stored as skipped and
            # never looked at again; a beta install lost 2,368 photos that way in twenty minutes.
            # A provider refusal now leaves the row unindexed, so the rows it already wrote are
            # dropped. They sit behind the feed's cursor, which fetches only assets after it;
            # the live feed's sweep (`Feed.sweep`, every FeedRunner.SWEEP_SECONDS) is what finds
            # them. Videos keep theirs: the sub-2-second ones were re-queued above.
            cursor.execute("DELETE FROM photos WHERE photo_type = 'skipped' "
                           "AND short_description = 'Skipped (LLM analysis failed)' AND duration_seconds IS NULL")
            if cursor.rowcount:
                note(f"Database migration: forgot {cursor.rowcount} photos skipped when the LLM call failed "
                     f"(a rate limit or empty credits is no longer a verdict on the photo). The feed's "
                     f"sweep indexes them again once it is live.")
        if version < SCHEMA_VERSION:
            cursor.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            trace(f"Database schema version {version} -> {SCHEMA_VERSION}")

    def submit(self, query: str, commands: str, response: str, cached: bool = False,
               source: str | None = None, location: str | None = None):
        """
        Submit a new usage record to the database.

        Args:
            query: The user's query (e.g., "Turn off the kitchen lights")
            commands: The commands executed (can be empty string if no commands)
            response: The text response given to the user
            cached: Whether the response was served from LLM cache
            source: Request source (e.g., "VoiceToVoice", "TextChat")
            location: Physical location of the request (e.g., "Kitchen")
        """
        timestamp = time.time()

        if response.strip() == "":
            trace("LLM replied with empty message. Assume the AI did not want to reply with anything.")
            return

        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO usage_stats (timestamp, query, commands, response, cached, source, location)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (timestamp, query, commands, response, cached, source, location))

                conn.commit()
                conn.close()

                trace(f"Usage stats stored for query: '{query}'.")

            except sqlite3.Error as e:
                error(f"Error recording usage stats: {e}")

    def get_response_counts_for_query(self, query: str) -> List[Tuple[str, int]]:
        """Returns (response, row_count) per unique cachable response. The row counts carry
        the evidence weight: three rows that all answered the same way are three independent
        confirmations the command mapping is stable, so the cache-eligibility threshold sums
        the counts and the majority vote weighs each response by its count — collapsing
        duplicates into a single vote would let one stray wording tie a 2-vs-1 majority."""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # Trailing sentence punctuation (.?!) is stripped from both the stored
                # query and the lookup query so transcriptions differing only by a final
                # "." — the new outpost STT appends one, the old one didn't — still share
                # a cache key. RTRIM on the column normalizes existing rows without a migration.
                cursor.execute('''
                    SELECT response, COUNT(*)
                    FROM usage_stats
                    WHERE RTRIM(LOWER(query), '.?! ') = RTRIM(LOWER(?), '.?! ') AND cached = 0
                      AND (not_cachable IS NULL OR not_cachable = 0)
                      AND response IS NOT NULL AND response <> ''
                    GROUP BY response;
                ''', (query,))

                results = cursor.fetchall()
                conn.close()

                return [(row[0], row[1]) for row in results]

            except sqlite3.Error as e:
                error(f"Error retrieving response counts: {e}")
                return []

    def get_unaudited_cache_entries(self, limit: int = 100) -> List[Tuple[int, str, str]]:
        """Returns (id, query, response) tuples where cached=0 AND not_cachable IS NULL."""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, query, response
                    FROM usage_stats
                    WHERE cached = 0 AND not_cachable IS NULL
                    LIMIT ?
                ''', (limit,))
                results = cursor.fetchall()
                conn.close()
                return results
            except sqlite3.Error as e:
                error(f"Error fetching unaudited cache entries: {e}")
                return []

    def mark_not_cachable(self, entry_id: int):
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('UPDATE usage_stats SET not_cachable = 1 WHERE id = ?', (entry_id,))
                conn.commit()
                conn.close()
            except sqlite3.Error as e:
                error(f"Error marking entry {entry_id} as not cachable: {e}")

    def mark_responses_not_cachable(self, query: str, responses: List[str]) -> int:
        """Mark every cachable row matching (query, response) as not_cachable — used when a
        served cached response turns out to be bad (e.g. it commands a nonexistent device).
        Query matching uses the same punctuation normalization as get_response_counts_for_query
        so all key variants of the query are covered. Returns the number of rows marked."""
        if not responses:
            info("No responses given to mark_responses_not_cachable — nothing to do.")
            return 0
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                placeholders = ", ".join("?" for _ in responses)
                cursor.execute(f'''
                    UPDATE usage_stats SET not_cachable = 1
                    WHERE RTRIM(LOWER(query), '.?! ') = RTRIM(LOWER(?), '.?! ')
                      AND cached = 0 AND response IN ({placeholders})
                ''', (query, *responses))
                marked = cursor.rowcount
                conn.commit()
                conn.close()
                return marked
            except sqlite3.Error as e:
                error(f"Error marking responses not cachable for query '{query}': {e}")
                return 0

    def mark_audited(self, entry_id: int):
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('UPDATE usage_stats SET not_cachable = 0 WHERE id = ?', (entry_id,))
                conn.commit()
                conn.close()
            except sqlite3.Error as e:
                error(f"Error marking entry {entry_id} as audited: {e}")

    def add_history_entry(self, role: str, content: str, timestamp: float,
                          source: str | None = None, location: str | None = None):
        """Insert a single conversation history entry. The role is stored lowercase, whatever
        the caller's spelling: a user row's role is the speaker's name and it arrives as the
        Telegram profile spells it, as the voice profile's lowercase id or as the "User"
        fallback, and one spelling per person keeps the role queries and the memory
        curator's grouping honest. The renderers capitalize it on the way out."""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR IGNORE INTO history (role, content, timestamp, source, location)
                    VALUES (?, ?, ?, ?, ?)
                ''', (role.lower(), content, timestamp, source, location))
                conn.commit()
                conn.close()
            except sqlite3.Error as e:
                error(f"Error adding history entry: {e}")

    def add_history_entries(self, entries: list[tuple], source: str | None = None) -> int:
        """Insert many (role, content, timestamp, location) rows in one transaction, roles
        lowercased as in add_history_entry; returns how
        many were new (the dedup index ignores rows already there). A database error is
        logged and re-raised: the caller (a feed) must not mistake it for "all duplicates"
        and move its cursor past rows that never landed."""
        if not entries:
            return 0
        with self.lock:
            try:
                with closing(sqlite3.connect(self.db_path)) as conn:
                    cursor = conn.cursor()
                    inserted = 0
                    for role, content, timestamp, location in entries:
                        # rowcount is 0 for an ignored duplicate (total_changes would count the FTS triggers)
                        cursor.execute('''
                            INSERT OR IGNORE INTO history (role, content, timestamp, source, location)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (role.lower(), content, timestamp, source, location))
                        inserted += cursor.rowcount
                    conn.commit()
                    return inserted
            except sqlite3.Error as e:
                error(f"Error adding {len(entries)} history entries from {source}: {e}")
                raise

    def count_history_entries(self, source: str, role: str) -> int:
        """How many rows `source` has stored under `role` — a feed's total (idx_history_source)."""
        with self.lock:
            try:
                with closing(sqlite3.connect(self.db_path)) as conn:
                    row = conn.execute('SELECT COUNT(*) FROM history WHERE source = ? AND role = ?',
                                       (source, role)).fetchone()
                    return row[0]
            except sqlite3.Error as e:
                error(f"Error counting history entries from {source}: {e}")
                raise

    def history_timestamps(self, source: str, timestamps: list[float]) -> set[float]:
        """Which of `timestamps` already have a row under `source` — a feed's test for an
        entry it stored before (under any role: a live row may carry a media description the
        backfill's row for the same moment does not, so the dedup index is no test). Chunked
        under SQLite's parameter limit."""
        found: set[float] = set()
        if not timestamps:
            return found
        with self.lock:
            try:
                with closing(sqlite3.connect(self.db_path)) as conn:
                    for start in range(0, len(timestamps), 500):
                        chunk = timestamps[start:start + 500]
                        placeholders = ",".join("?" * len(chunk))
                        rows = conn.execute(f'SELECT timestamp FROM history WHERE source = ? '
                                            f'AND timestamp IN ({placeholders})', (source, *chunk)).fetchall()
                        found.update(row[0] for row in rows)
            except sqlite3.Error as e:
                error(f"Error looking up {len(timestamps)} history timestamps from {source}: {e}")
                raise
        return found

    # --- feeds: FeedRunner's cursor per feed name ---

    _FEED_COLUMNS = ("cursor", "horizon", "live_since", "last_entry_ts", "gap_until", "resume_cursor",
                     "gap_covered_until")

    def get_feed(self, name: str) -> dict | None:
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                row = conn.execute('SELECT * FROM feeds WHERE name = ?', (name,)).fetchone()
                conn.close()
                return dict(row) if row else None
            except sqlite3.Error as e:
                error(f"Error reading feed '{name}': {e}")
                return None

    def save_feed(self, name: str, **fields):
        """Create or update the feed's row; only the given columns change."""
        unknown = set(fields) - set(self._FEED_COLUMNS)
        if unknown:
            raise ValueError(f"unknown feed columns: {sorted(unknown)}")
        now = time.time()
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                conn.execute('INSERT OR IGNORE INTO feeds (name, updated_at) VALUES (?, ?)', (name, now))
                if fields:
                    assignments = ", ".join(f"{column} = ?" for column in fields)
                    conn.execute(f'UPDATE feeds SET {assignments}, updated_at = ? WHERE name = ?',
                                 (*fields.values(), now, name))
                else:
                    conn.execute('UPDATE feeds SET updated_at = ? WHERE name = ?', (now, name))
                conn.commit()
                conn.close()
            except sqlite3.Error as e:
                error(f"Error saving feed '{name}': {e}")

    def delete_feed(self, name: str) -> bool:
        """Forget where a feed got to, so its next start catches up from the horizon again."""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                deleted = conn.execute('DELETE FROM feeds WHERE name = ?', (name,)).rowcount
                conn.commit()
                conn.close()
                return deleted > 0
            except sqlite3.Error as e:
                error(f"Error deleting feed '{name}': {e}")
                return False

    def delete_todays_history(self) -> int:
        """Delete all conversation history entries from today (since midnight). Returns count deleted."""
        from datetime import datetime
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('DELETE FROM history WHERE timestamp >= ?', (today_start,))
                deleted_count = cursor.rowcount
                conn.commit()
                conn.close()
                info(f"Deleted {deleted_count} history entries for today")
                return deleted_count
            except sqlite3.Error as e:
                error(f"Error deleting today's history: {e}")
                return 0

    def get_recent_history(self, limit: int = 100) -> List[dict]:
        """Get today's conversation history entries (since midnight), oldest first."""
        from datetime import datetime
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM (
                        SELECT role, content, timestamp, source, location
                        FROM history
                        WHERE timestamp >= ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    ) ORDER BY timestamp ASC
                ''', (today_start, limit))
                rows = cursor.fetchall()
                conn.close()
                return [dict(row) for row in rows]
            except sqlite3.Error as e:
                error(f"Error retrieving history: {e}")
                return []

    def get_history_range(self, from_ts: float, to_ts: float, limit: int = 200) -> List[dict]:
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT role, content, timestamp, source, location
                    FROM history
                    WHERE timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp ASC
                    LIMIT ?
                ''', (from_ts, to_ts, limit))
                rows = cursor.fetchall()
                conn.close()
                info(f"History range query returned {len(rows)} results")
                return [dict(row) for row in rows]
            except sqlite3.Error as e:
                error(f"Error retrieving history range: {e}")
                return []

    HISTORY_SORT_OPTIONS = {
        "relevance": "rank",
        "newest": "h.timestamp DESC",
        "oldest": "h.timestamp ASC",
    }

    def search_history(self, query: str, sort: str = "relevance", limit: int = 30) -> List[dict]:
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                tokens = _tokenize_query(query)
                fts_query = _build_fts_query(query)
                info(f"Searching history: query='{query}', fts='{fts_query}', sort='{sort}', tokens={len(tokens)}")
                results = self._run_history_fts(cursor, tokens, fts_query, sort, limit)

                conn.close()
                info(f"History search returned {len(results)} results")
                return results
            except sqlite3.Error as e:
                error(f"Error searching history: {e}")
                return []

    @staticmethod
    def _run_history_fts(cursor, tokens: List[str], fts_query: str, sort: str, limit: int) -> List[dict]:
        # Overfetch generously so Python-side coverage re-ranking has the full candidate
        # set to work with. FTS5's PERCENT_RANK normalization throws away absolute quality,
        # so coverage * 1000 + bm25_score is what actually orders results.
        overfetch = 2000
        cursor.execute('''
            SELECT h.id, h.role, h.content, h.timestamp, h.source, h.location,
                   (1.0 - PERCENT_RANK() OVER (ORDER BY rank)) * 100 AS bm25_score
            FROM history h
            JOIN history_fts ON history_fts.rowid = h.id
            WHERE history_fts MATCH ?
              AND length(h.content) > 20
            ORDER BY rank
            LIMIT ?
        ''', (fts_query, overfetch))
        rows = [dict(r) for r in cursor.fetchall()]

        for r in rows:
            r["coverage"] = _count_token_coverage(r["content"], tokens)

        # Coverage dominates; bm25 percentile breaks ties.
        for r in rows:
            r["fts_relevance"] = r["coverage"] * 1000 + r.pop("bm25_score", 0.0)

        if sort == "newest":
            rows.sort(key=lambda r: (r["coverage"], r["timestamp"]), reverse=True)
        elif sort == "oldest":
            rows.sort(key=lambda r: (-r["coverage"], r["timestamp"]))
        else:
            rows.sort(key=lambda r: r["fts_relevance"], reverse=True)

        return rows[:limit]

    def is_query_common(self, query: str) -> bool:
        """
        Check if a query is common (appears more than 2 times in the database).

        Args:
            query: The query to check

        Returns:
            True if the query appears more than 2 times, False otherwise
        """
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute('''
                    SELECT COUNT(*) as count
                    FROM usage_stats
                    WHERE query = ? AND cached = 0
                ''', (query,))

                result = cursor.fetchone()
                conn.close()

                count = result[0] if result else 0
                return count > 2

            except sqlite3.Error as e:
                error(f"Error checking if query is common: {e}")
                return False

    # --- Photos integration methods ---

    def store_photo(self, asset_guid: str, creation_date: float,
                    latitude: Optional[float], longitude: Optional[float],
                    location_text: Optional[str], albums: Optional[str],
                    persons: Optional[str], photo_type: Optional[str],
                    short_description: str, detailed_description: str,
                    pretty_score: int, family_importance_score: int,
                    indexed_at: float, duration_seconds: Optional[float] = None) -> Optional[int]:
        """Insert one indexed asset (a still, or a video when duration_seconds is set).
        Returns the new row id, or None when the GUID was already stored."""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR IGNORE INTO photos
                    (asset_guid, creation_date, latitude, longitude, location_text,
                     albums, persons, photo_type, short_description, detailed_description,
                     pretty_score, family_importance_score, indexed_at, duration_seconds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (asset_guid, creation_date, latitude, longitude, location_text,
                      albums, persons, photo_type, short_description, detailed_description,
                      pretty_score, family_importance_score, indexed_at, duration_seconds))
                photo_id = cursor.lastrowid if cursor.rowcount > 0 else None
                conn.commit()
                conn.close()
                return photo_id
            except sqlite3.Error as e:
                error(f"Error storing photo {asset_guid}: {e}")
                return None

    def photo_exists(self, asset_guid: str) -> bool:
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT 1 FROM photos WHERE asset_guid = ?', (asset_guid,))
                result = cursor.fetchone()
                conn.close()
                return result is not None
            except sqlite3.Error as e:
                error(f"Error checking photo existence: {e}")
                return False

    SEARCH_SORT_OPTIONS = {
        "relevance": "rank",
        "pretty": "p.pretty_score DESC",
        "family": "p.family_importance_score DESC",
        "best": "(p.pretty_score + p.family_importance_score) DESC",
    }

    # FTS sort options blend relevance (via PERCENT_RANK over bm25) into scoring
    # so photos matching more search terms rank significantly higher.
    # fts_relevance is 0-100 where 100 = best match in the result set.
    FTS_SEARCH_SORT_OPTIONS = {
        "relevance": "rank",
        "pretty": "(fts_relevance + p.pretty_score) DESC",
        "family": "(fts_relevance + p.family_importance_score) DESC",
        "best": "(fts_relevance + p.pretty_score + p.family_importance_score) DESC",
    }

    def search_photos(self, query: str, limit: int = 20, sort_by: str = "relevance") -> List[dict]:
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                if query.strip() == "*" or not query.strip():
                    order_clause = self.SEARCH_SORT_OPTIONS.get(sort_by, "rank")
                    if sort_by == "relevance":
                        order_clause = "p.creation_date DESC"
                    info(f"Browsing all photos sorted by {order_clause}")
                    cursor.execute(f'''
                        SELECT p.* FROM photos p
                        ORDER BY {order_clause}
                        LIMIT ?
                    ''', (limit,))
                else:
                    order_clause = self.FTS_SEARCH_SORT_OPTIONS.get(sort_by, "rank")
                    fts_query = _build_fts_query(query)
                    info(f"Searching photos: query='{query}', sort_by='{sort_by}', order='{order_clause}'")
                    trace(f"Photos FTS query: '{fts_query}'")
                    cursor.execute(f'''
                        SELECT p.*,
                               (1.0 - PERCENT_RANK() OVER (ORDER BY rank)) * 100 AS fts_relevance
                        FROM photos p
                        JOIN photos_fts ON photos_fts.rowid = p.id
                        WHERE photos_fts MATCH ?
                        ORDER BY {order_clause}
                        LIMIT ?
                    ''', (fts_query, limit))
                results = [dict(row) for row in cursor.fetchall()]
                conn.close()
                return results
            except sqlite3.Error as e:
                error(f"Error searching photos: {e}")
                return []

    def get_photo_by_id(self, photo_id: int) -> Optional[dict]:
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM photos WHERE id = ?', (photo_id,))
                row = cursor.fetchone()
                conn.close()
                if row:
                    return dict(row)
                trace(f"Photo with id {photo_id} not found")
                return None
            except sqlite3.Error as e:
                error(f"Error getting photo by id: {e}")
                return None

    def get_photo_description(self, asset_guid: str) -> Optional[str]:
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT short_description FROM photos WHERE asset_guid = ?',
                               (asset_guid,))
                row = cursor.fetchone()
                conn.close()
                return row[0] if row else None
            except sqlite3.Error as e:
                error(f"Error getting photo description: {e}")
                return None

    def delete_photos(self, asset_guids) -> int:
        """Remove the indexed rows for the given asset GUIDs, plus the "New photo taken" history
        entries that carry their descriptions (the FTS triggers follow). Returns the photo row count."""
        guids = list(asset_guids)
        if not guids:
            return 0
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                removed = history_removed = 0
                for i in range(0, len(guids), 500):
                    chunk = guids[i:i + 500]
                    placeholders = ','.join('?' * len(chunk))
                    photo_ids = [row[0] for row in
                                 cursor.execute(f"SELECT id FROM photos WHERE asset_guid IN ({placeholders})", chunk)]
                    for photo_id in photo_ids:
                        cursor.execute("DELETE FROM history WHERE source = ? AND content LIKE ?",
                                       (PHOTO_HISTORY_SOURCE, f"%[photo_id={photo_id}]%"))
                        history_removed += cursor.rowcount
                    cursor.execute(f"DELETE FROM photos WHERE asset_guid IN ({placeholders})", chunk)
                    removed += cursor.rowcount
                conn.commit()
                conn.close()
                trace(f"Deleted {removed} photo row(s) and {history_removed} matching history entr(y/ies)")
                return removed
            except sqlite3.Error as e:
                error(f"Error deleting photos: {e}")
                return 0

    def get_indexed_photo_guids(self) -> set:
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT asset_guid FROM photos')
                guids = {row[0] for row in cursor.fetchall()}
                conn.close()
                return guids
            except sqlite3.Error as e:
                error(f"Error getting indexed photo GUIDs: {e}")
                return set()

    def get_photo_count(self) -> int:
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM photos')
                count = cursor.fetchone()[0]
                conn.close()
                return count
            except sqlite3.Error as e:
                error(f"Error getting photo count: {e}")
                return 0

    PHOTO_SORT_COLUMNS = {"creation_date", "pretty_score", "family_importance_score", "indexed_at"}

    def get_photos_paginated(self, offset: int = 0, limit: int = 1,
                             sort_by: str = "creation_date", sort_desc: bool = True) -> List[dict]:
        if sort_by not in self.PHOTO_SORT_COLUMNS:
            sort_by = "creation_date"
        direction = "DESC" if sort_desc else "ASC"
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    f'SELECT * FROM photos ORDER BY {sort_by} {direction} LIMIT ? OFFSET ?',
                    (limit, offset))
                results = [dict(row) for row in cursor.fetchall()]
                conn.close()
                return results
            except sqlite3.Error as e:
                error(f"Error getting paginated photos: {e}")
                return []

    def get_interaction_history(self, start_date, end_date) -> str:
        """
        Get interaction history from the history table, formatted for memory curation.
        Includes user/assistant messages and system events (photo descriptions, home automation, etc).

        Args:
            start_date: Start (inclusive) — a datetime, or a date meaning midnight at its start
            end_date: End (exclusive) — a datetime, or a date meaning midnight at its start

        Returns:
            Formatted string with timestamps, roles, and content
        """
        from datetime import datetime

        def as_timestamp(moment):
            if not isinstance(moment, datetime):
                moment = datetime.combine(moment, datetime.min.time())
            return moment.timestamp()

        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                start_ts = as_timestamp(start_date)
                end_ts = as_timestamp(end_date)

                cursor.execute('''
                    SELECT datetime(timestamp, 'unixepoch', 'localtime') as local_time,
                           role, content, source, location
                    FROM history
                    WHERE timestamp >= ? AND timestamp < ?
                    ORDER BY timestamp ASC
                ''', (start_ts, end_ts))

                rows = cursor.fetchall()
                conn.close()

                if not rows:
                    return NO_INTERACTIONS

                lines = []
                for local_time, role, content, source, location in rows:
                    if role == "user":
                        loc = f" [{location}]" if location else ""
                        lines.append(f"[{local_time}]{loc} User: {content}")
                    elif role == "assistant":
                        lines.append(f"[{local_time}] Assistant: {content}")
                    elif role == "system":
                        src = f" ({source})" if source else ""
                        lines.append(f"[{local_time}] System{src}: {content}")
                    else:
                        lines.append(f"[{local_time}] {role.capitalize()}: {content}")
                    lines.append("")

                return "\n".join(lines)

            except sqlite3.Error as e:
                error(f"Error retrieving interaction history: {e}")
                return NO_INTERACTIONS


