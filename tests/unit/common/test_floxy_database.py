import pathlib
import sqlite3
import threading
import time
from datetime import date, datetime, timedelta

import pytest

from common.FloxyDatabase import FloxyDatabase


@pytest.fixture
def floxy_db(tmp_path):
    db = FloxyDatabase.__new__(FloxyDatabase)
    db.db_path = str(tmp_path / "floxy_test.db")
    db.lock = threading.Lock()
    db._init_database()
    return db


@pytest.mark.unit
def test_response_counts_ignore_trailing_punctuation(floxy_db):
    # The new outpost STT appends a trailing "." that the old one didn't; both forms
    # must share a cache key so stored no-dot responses still answer dotted queries.
    floxy_db.submit("Turn off the kitchen lights", "", "Done A", cached=False)
    floxy_db.submit("Turn off the kitchen lights.", "", "Done B", cached=False)
    floxy_db.submit("turn off the kitchen lights?", "", "Done C", cached=False)

    dotted = floxy_db.get_response_counts_for_query("Turn off the kitchen lights.")
    undotted = floxy_db.get_response_counts_for_query("turn off the kitchen lights")

    assert sorted(dotted) == [("Done A", 1), ("Done B", 1), ("Done C", 1)]
    assert sorted(undotted) == [("Done A", 1), ("Done B", 1), ("Done C", 1)]


@pytest.mark.unit
def test_response_counts_count_duplicate_rows(floxy_db):
    # Three identical answers are three confirmations the mapping is stable; the count
    # must see all three even though the responses collapse to one unique wording.
    floxy_db.submit("Turn off the kitchen lights", "execute(Off)", "Done.", cached=False)
    floxy_db.submit("turn off the kitchen lights.", "execute(Off)", "Done.", cached=False)
    floxy_db.submit("Turn off the kitchen lights", "execute(Off)", "Done.", cached=False)

    assert floxy_db.get_response_counts_for_query("Turn off the kitchen lights") == [("Done.", 3)]


@pytest.mark.unit
def test_response_counts_exclude_cached_and_not_cachable(floxy_db):
    floxy_db.submit("Q", "execute(Off)", "Done.", cached=False)
    floxy_db.submit("Q", "", "Served from cache", cached=True)
    floxy_db.submit("Q", "", "already off", cached=False)
    floxy_db.mark_not_cachable(floxy_db.get_unaudited_cache_entries()[-1][0])

    assert floxy_db.get_response_counts_for_query("Q") == [("Done.", 1)]


@pytest.mark.unit
def test_mark_responses_not_cachable_by_query_and_response(floxy_db):
    floxy_db.submit("Dim the lights", "", "Bad.", cached=False)
    floxy_db.submit("Dim the lights", "", "Bad.", cached=False)
    floxy_db.submit("Dim the lights", "", "Good.", cached=False)

    marked = floxy_db.mark_responses_not_cachable("Dim the lights", ["Bad."])

    assert marked == 2
    assert floxy_db.get_response_counts_for_query("Dim the lights") == [("Good.", 1)]


@pytest.mark.unit
def test_mark_responses_not_cachable_normalizes_query_punctuation(floxy_db):
    # Rows stored with/without a trailing "." share a cache key (see the response-counts
    # tests) — invalidation must cover all key variants or the bad row keeps serving.
    floxy_db.submit("Turn on lights", "", "Bad.", cached=False)
    floxy_db.submit("Turn on lights.", "", "Bad.", cached=False)

    marked = floxy_db.mark_responses_not_cachable("turn on lights.", ["Bad."])

    assert marked == 2
    assert floxy_db.get_response_counts_for_query("Turn on lights") == []


@pytest.mark.unit
def test_mark_responses_not_cachable_ignores_cached_rows_and_empty_list(floxy_db):
    floxy_db.submit("Q", "", "Text.", cached=True)

    assert floxy_db.mark_responses_not_cachable("Q", ["Text."]) == 0
    assert floxy_db.mark_responses_not_cachable("Q", []) == 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_row_count(db):
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM usage_stats")
    count = cursor.fetchone()[0]
    conn.close()
    return count


# ---------------------------------------------------------------------------
# New tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_submit_rejects_empty_response(floxy_db):
    floxy_db.submit("q", "", "", False)
    floxy_db.submit("q", "", "  ", False)
    assert get_row_count(floxy_db) == 0


@pytest.mark.unit
def test_get_interaction_history(floxy_db):
    now = time.time()
    floxy_db.add_history_entry("user", "hello", now, location="Kitchen")
    floxy_db.add_history_entry("assistant", "hi there", now + 1)
    floxy_db.add_history_entry("system", "Photo: family at dinner", now + 2, source="iCloud Photos")

    yesterday = date.today() - timedelta(days=1)
    tomorrow = date.today() + timedelta(days=1)
    result = floxy_db.get_interaction_history(yesterday, tomorrow)

    assert "[Kitchen] User: hello" in result
    assert "Assistant: hi there" in result
    assert "System (iCloud Photos): Photo: family at dinner" in result


@pytest.mark.unit
def test_get_interaction_history_accepts_datetimes(floxy_db):
    # The curator's first-run window is "since 14:03", not a whole day: a datetime bound
    # is used as is, so an entry a second before the start is left out.
    start = datetime.now().replace(microsecond=0)
    floxy_db.add_history_entry("user", "before the window", start.timestamp() - 1)
    floxy_db.add_history_entry("user", "inside the window", start.timestamp() + 1)
    result = floxy_db.get_interaction_history(start, start + timedelta(minutes=5))
    assert "inside the window" in result
    assert "before the window" not in result


@pytest.mark.unit
def test_get_unaudited_cache_entries(floxy_db):
    # (a) uncached + unaudited (not_cachable IS NULL) — should be returned
    floxy_db.submit("q1", "", "resp_a", cached=False)

    # (b) cached — should NOT be returned
    floxy_db.submit("q2", "", "resp_b", cached=True)

    # (c) uncached + marked not_cachable — should NOT be returned
    floxy_db.submit("q3", "", "resp_c", cached=False)
    # Get the id of the last insert so we can mark it
    conn = sqlite3.connect(floxy_db.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM usage_stats WHERE query='q3'")
    row_id = cursor.fetchone()[0]
    conn.close()
    floxy_db.mark_not_cachable(row_id)

    entries = floxy_db.get_unaudited_cache_entries()
    queries = [e[1] for e in entries]
    assert "q1" in queries
    assert "q2" not in queries
    assert "q3" not in queries


@pytest.mark.unit
def test_mark_audited_keeps_in_unique_responses(floxy_db):
    floxy_db.submit("test_q2", "", "test_r2", cached=False)

    conn = sqlite3.connect(floxy_db.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM usage_stats WHERE query='test_q2'")
    row_id = cursor.fetchone()[0]
    conn.close()

    floxy_db.mark_audited(row_id)

    responses = [r for r, _ in floxy_db.get_response_counts_for_query("test_q2")]
    assert "test_r2" in responses


@pytest.mark.unit
def test_migrate_adds_missing_columns(tmp_path):
    db_path = str(tmp_path / "old_schema.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE usage_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            query TEXT NOT NULL,
            commands TEXT,
            response TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

    db = FloxyDatabase.__new__(FloxyDatabase)
    db.db_path = db_path
    db.lock = threading.Lock()
    db._init_database()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(usage_stats)")
    columns = [col[1] for col in cursor.fetchall()]
    conn.close()

    assert "cached" in columns
    assert "not_cachable" in columns
    assert "source" in columns
    assert "location" in columns


def _mark_pre_video_schema(db_path: str):
    """Rewind the schema version so the one-off video re-queues run on the next open."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA user_version = 0")
    conn.commit()
    conn.close()


@pytest.mark.unit
def test_migrate_requeues_videos_skipped_as_too_large(tmp_path):
    db_path = str(tmp_path / "video_skips.db")
    first = FloxyDatabase.__new__(FloxyDatabase)
    first.db_path = db_path
    first.lock = threading.Lock()
    first._init_database()
    for guid, kind, desc in [("v-large", "skipped", "Skipped (video too large)"),
                             ("v-long", "skipped", "Skipped (video too long)"),
                             ("p-large", "skipped", "Skipped (too large)"),
                             ("v-ok", "video", "A dog runs")]:
        first.store_photo(asset_guid=guid, creation_date=1.0, latitude=None, longitude=None,
                          location_text="", albums="", persons="", photo_type=kind,
                          short_description=desc, detailed_description=desc,
                          pretty_score=0, family_importance_score=0, indexed_at=1.0)

    _mark_pre_video_schema(db_path)
    second = FloxyDatabase.__new__(FloxyDatabase)
    second.db_path = db_path
    second.lock = threading.Lock()
    second._init_database()

    conn = sqlite3.connect(db_path)
    remaining = {row[0] for row in conn.execute("SELECT asset_guid FROM photos")}
    fts_rows = conn.execute("SELECT count(*) FROM photos_fts WHERE photos_fts MATCH '\"too large\"'").fetchone()[0]
    conn.close()
    assert remaining == {"v-long", "p-large", "v-ok"}
    assert fts_rows == 1  # the still's row stays, the video's FTS entry went with it


@pytest.mark.unit
def test_history_roles_are_stored_lowercase(tmp_path):
    db = FloxyDatabase.__new__(FloxyDatabase)
    db.db_path = str(tmp_path / "roles.db")
    db.lock = threading.Lock()
    db._init_database()
    db.add_history_entry("Marcus", "hi", 1.0, "Telegram")
    db.add_history_entry("User", "lights", 2.0, "VoiceToVoice")
    db.add_history_entries([("Charlotte", "hej", 3.0, None)], source="Telegram")

    conn = sqlite3.connect(db.db_path)
    roles = [row[0] for row in conn.execute("SELECT role FROM history ORDER BY timestamp")]
    fts = [row[0] for row in conn.execute("SELECT role FROM history_fts ORDER BY rowid")]
    conn.close()
    assert roles == ["marcus", "user", "charlotte"]
    assert fts == roles


@pytest.mark.unit
def test_migrate_lowercases_history_roles_and_drops_case_duplicates(tmp_path):
    db_path = str(tmp_path / "mixed_roles.db")
    first = FloxyDatabase.__new__(FloxyDatabase)
    first.db_path = db_path
    first.lock = threading.Lock()
    first._init_database()
    conn = sqlite3.connect(db_path)
    conn.executemany("INSERT INTO history (role, content, timestamp, source) VALUES (?, ?, ?, ?)", [
        ("Marcus", "hi", 1.0, "Telegram"),
        ("User", "lights", 2.0, "VoiceToVoice"),
        ("user", "lights", 2.0, "VoiceToVoice"),   # the same row the dedup index missed by case
        ("assistant", "done", 2.001, None),
    ])
    conn.commit()
    conn.close()

    _mark_pre_video_schema(db_path)
    second = FloxyDatabase.__new__(FloxyDatabase)
    second.db_path = db_path
    second.lock = threading.Lock()
    second._init_database()

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT role, content FROM history ORDER BY timestamp").fetchall()
    fts = conn.execute("SELECT role FROM history_fts WHERE role = 'marcus'").fetchall()
    conn.close()
    assert rows == [("marcus", "hi"), ("user", "lights"), ("assistant", "done")]
    assert fts == [("marcus",)]


@pytest.mark.unit
def test_migrate_requeues_short_videos_that_failed_llm_analysis(tmp_path):
    db_path = str(tmp_path / "short_video_skips.db")
    first = FloxyDatabase.__new__(FloxyDatabase)
    first.db_path = db_path
    first.lock = threading.Lock()
    first._init_database()
    for guid, kind, desc, duration in [("v-tiny", "skipped", "Skipped (LLM analysis failed)", 0.1),
                                       ("v-short", "skipped", "Skipped (LLM analysis failed)", 1.9),
                                       ("v-long-failed", "skipped", "Skipped (LLM analysis failed)", 7.0),
                                       ("v-no-duration", "skipped", "Skipped (LLM analysis failed)", None),
                                       ("v-too-long", "skipped", "Skipped (video too long)", 0.5),
                                       ("v-ok", "video", "A dog runs", 0.4)]:
        first.store_photo(asset_guid=guid, creation_date=1.0, latitude=None, longitude=None,
                          location_text="", albums="", persons="", photo_type=kind,
                          short_description=desc, detailed_description=desc,
                          pretty_score=0, family_importance_score=0, indexed_at=1.0,
                          duration_seconds=duration)

    _mark_pre_video_schema(db_path)
    second = FloxyDatabase.__new__(FloxyDatabase)
    second.db_path = db_path
    second.lock = threading.Lock()
    second._init_database()

    conn = sqlite3.connect(db_path)
    remaining = {row[0] for row in conn.execute("SELECT asset_guid FROM photos")}
    conn.close()
    assert remaining == {"v-long-failed", "v-no-duration", "v-too-long", "v-ok"}


@pytest.mark.unit
def test_migrate_requeues_videos_whose_icloud_download_failed(tmp_path):
    db_path = str(tmp_path / "video_download_skips.db")
    first = FloxyDatabase.__new__(FloxyDatabase)
    first.db_path = db_path
    first.lock = threading.Lock()
    first._init_database()
    for guid, kind, desc, duration in [("v-slomo", "skipped", "Skipped (iCloud download failed)", 47.2),
                                       ("v-tiny", "skipped", "Skipped (iCloud download failed)", 0.002),
                                       ("p-failed", "skipped", "Skipped (iCloud download failed)", None),
                                       ("v-gone", "skipped", "Skipped (not in iCloud library)", 3.0),
                                       ("v-ok", "video", "A dog runs", 0.4)]:
        first.store_photo(asset_guid=guid, creation_date=1.0, latitude=None, longitude=None,
                          location_text="", albums="", persons="", photo_type=kind,
                          short_description=desc, detailed_description=desc,
                          pretty_score=0, family_importance_score=0, indexed_at=1.0,
                          duration_seconds=duration)

    _mark_pre_video_schema(db_path)
    second = FloxyDatabase.__new__(FloxyDatabase)
    second.db_path = db_path
    second.lock = threading.Lock()
    second._init_database()

    conn = sqlite3.connect(db_path)
    remaining = {row[0] for row in conn.execute("SELECT asset_guid FROM photos")}
    conn.close()
    assert remaining == {"p-failed", "v-gone", "v-ok"}


@pytest.mark.unit
def test_video_requeue_migrations_run_once(tmp_path):
    """Today's code re-creates the very rows the re-queue deletes remove, so replaying them on
    every start would re-download and re-analyze the same clips forever."""
    from common.FloxyDatabase import SCHEMA_VERSION
    db_path = str(tmp_path / "versioned.db")

    def open_db():
        db = FloxyDatabase.__new__(FloxyDatabase)
        db.db_path = db_path
        db.lock = threading.Lock()
        db._init_database()
        return db

    def store_skip(db, guid, desc, duration):
        db.store_photo(asset_guid=guid, creation_date=1.0, latitude=None, longitude=None,
                       location_text="", albums="", persons="", photo_type="skipped",
                       short_description=desc, detailed_description="", pretty_score=0,
                       family_importance_score=0, indexed_at=1.0, duration_seconds=duration)

    first = open_db()
    conn = sqlite3.connect(db_path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION  # fresh DB, nothing to replay
    conn.close()
    _mark_pre_video_schema(db_path)
    store_skip(first, "v-old-large", "Skipped (video too large)", 30.0)

    open_db()  # the one-off re-queue
    assert first.get_indexed_photo_guids() == set()

    store_skip(first, "v-large", "Skipped (video too large)", 30.0)
    store_skip(first, "v-short-failed", "Skipped (LLM analysis failed)", 0.5)
    store_skip(first, "v-download-failed", "Skipped (iCloud download failed)", 12.0)
    open_db()  # any later start leaves today's terminal skips alone
    assert first.get_indexed_photo_guids() == {"v-large", "v-short-failed", "v-download-failed"}


@pytest.mark.unit
def test_submit_with_source_and_location(floxy_db):
    floxy_db.submit("Hello", "", "Hi there", cached=False, source="VoiceToVoice", location="Kitchen")
    conn = sqlite3.connect(floxy_db.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT source, location FROM usage_stats WHERE query='Hello'")
    row = cursor.fetchone()
    conn.close()
    assert row == ("VoiceToVoice", "Kitchen")


@pytest.mark.unit
def test_store_photo_duplicate_returns_none(floxy_db):
    kwargs = dict(
        asset_guid="uuid-dup/L0/001", creation_date=time.time(),
        latitude=None, longitude=None, location_text="",
        albums="", persons="", photo_type="photo",
        short_description="Dup test", detailed_description="",
        pretty_score=50, family_importance_score=50, indexed_at=time.time(),
    )
    first_id = floxy_db.store_photo(**kwargs)
    assert first_id is not None
    second_id = floxy_db.store_photo(**kwargs)
    assert second_id is None


# ---------------------------------------------------------------------------
# search_history (FTS)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_search_history_finds_matching_entries(floxy_db):
    now = time.time()
    floxy_db.add_history_entry("user", "We went to the beach and had a wonderful time", now - 100)
    floxy_db.add_history_entry("assistant", "That sounds like a great day at the beach!", now - 50)
    floxy_db.add_history_entry("user", "Can you turn on the kitchen lights please", now)

    results = floxy_db.search_history("beach")

    assert len(results) == 2
    contents = [r["content"] for r in results]
    assert any("beach" in c for c in contents)
    assert all("kitchen" not in c for c in contents)


@pytest.mark.unit
def test_search_history_skips_short_entries(floxy_db):
    now = time.time()
    floxy_db.add_history_entry("user", "beach", now)  # too short (<=30 chars)
    floxy_db.add_history_entry("user", "We went to the beach and it was a beautiful sunny day", now + 1)

    results = floxy_db.search_history("beach")

    assert len(results) == 1
    assert "beautiful sunny day" in results[0]["content"]


@pytest.mark.unit
def test_search_history_sort_newest(floxy_db):
    floxy_db.add_history_entry("user", "Old conversation about the garden and flowers", 1000000.0)
    floxy_db.add_history_entry("user", "Recent conversation about the garden and trees", 2000000.0)

    results = floxy_db.search_history("garden", sort="newest")

    assert len(results) == 2
    assert "Recent" in results[0]["content"]
    assert "Old" in results[1]["content"]


@pytest.mark.unit
def test_search_history_respects_limit(floxy_db):
    now = time.time()
    for i in range(10):
        floxy_db.add_history_entry("user", f"Entry number {i} about the beach and ocean waves", now + i)

    results = floxy_db.search_history("beach", limit=3)

    assert len(results) == 3


# ---------------------------------------------------------------------------
# delete_todays_history
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_delete_todays_history(floxy_db):
    now = time.time()
    yesterday = now - 86400 * 2
    floxy_db.add_history_entry("user", "Old entry", yesterday)
    floxy_db.add_history_entry("user", "Today entry 1", now)
    floxy_db.add_history_entry("assistant", "Today entry 2", now + 1)

    deleted = floxy_db.delete_todays_history()

    assert deleted == 2
    remaining = floxy_db.get_recent_history()
    assert len(remaining) == 0

    yesterday_date = date.today() - timedelta(days=3)
    tomorrow = date.today() + timedelta(days=1)
    history = floxy_db.get_interaction_history(yesterday_date, tomorrow)
    assert "Old entry" in history


# ---------------------------------------------------------------------------
# search_photos (FTS)
# ---------------------------------------------------------------------------

def _store_test_photo(db, local_id, creation_date, short_desc, detailed_desc,
                      location_text="Home", persons=None, albums=None,
                      photo_type="photo", pretty_score=50, family_score=50):
    return db.store_photo(
        asset_guid=local_id, creation_date=creation_date,
        latitude=None, longitude=None, location_text=location_text,
        albums=albums, persons=persons, photo_type=photo_type,
        short_description=short_desc, detailed_description=detailed_desc,
        pretty_score=pretty_score, family_importance_score=family_score,
        indexed_at=time.time(),
    )


@pytest.mark.unit
def test_search_photos_fts_finds_matching(floxy_db):
    _store_test_photo(floxy_db, "uuid1/L0/001", 1710460800.0,
                      "Family dinner at restaurant", "A nice family dinner")
    _store_test_photo(floxy_db, "uuid2/L0/001", 1710460900.0,
                      "Sunset over the ocean", "Beautiful sunset colors")

    results = floxy_db.search_photos("dinner")

    assert len(results) == 1
    assert "dinner" in results[0]["short_description"].lower()


@pytest.mark.unit
def test_search_photos_fts_searches_persons_and_location(floxy_db):
    _store_test_photo(floxy_db, "uuid1/L0/001", 1710460800.0,
                      "Group photo", "People posing",
                      location_text="Frankfurt", persons="Alice, Bob")

    by_person = floxy_db.search_photos("Alice")
    assert len(by_person) == 1

    by_location = floxy_db.search_photos("Frankfurt")
    assert len(by_location) == 1


@pytest.mark.unit
def test_search_photos_wildcard_returns_all(floxy_db):
    _store_test_photo(floxy_db, "a/L0/001", 1710460800.0, "Photo A", "Desc A")
    _store_test_photo(floxy_db, "b/L0/001", 1710460900.0, "Photo B", "Desc B")

    results = floxy_db.search_photos("*")
    assert len(results) == 2


@pytest.mark.unit
def test_search_photos_sort_by_pretty(floxy_db):
    _store_test_photo(floxy_db, "a/L0/001", 1710460800.0,
                      "Beach sunset photo", "A sunset on a beach",
                      pretty_score=30)
    _store_test_photo(floxy_db, "b/L0/001", 1710460900.0,
                      "Beach sunrise photo", "A sunrise on a beach",
                      pretty_score=95)

    results = floxy_db.search_photos("beach", sort_by="pretty")
    assert len(results) == 2
    assert results[0]["pretty_score"] >= results[1]["pretty_score"]


@pytest.mark.unit
def test_search_photos_respects_limit(floxy_db):
    for i in range(5):
        _store_test_photo(floxy_db, f"uuid{i}/L0/001", 1710460800.0 + i,
                          f"Beach photo number {i}", f"Beach description {i}")

    results = floxy_db.search_photos("beach", limit=2)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# get_history_range
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_get_history_range_returns_entries_in_window(floxy_db):
    floxy_db.add_history_entry("user", "before range", 1000.0)
    floxy_db.add_history_entry("user", "inside range", 2000.0)
    floxy_db.add_history_entry("assistant", "also inside", 2500.0)
    floxy_db.add_history_entry("user", "after range", 4000.0)

    results = floxy_db.get_history_range(1500.0, 3000.0)

    assert len(results) == 2
    contents = [r["content"] for r in results]
    assert "inside range" in contents
    assert "also inside" in contents
    assert "before range" not in contents
    assert "after range" not in contents


@pytest.mark.unit
def test_get_history_range_returns_dicts_with_expected_keys(floxy_db):
    floxy_db.add_history_entry("user", "hello world", 2000.0, source="VoiceToVoice", location="Kitchen")

    results = floxy_db.get_history_range(1000.0, 3000.0)

    assert len(results) == 1
    r = results[0]
    assert r["role"] == "user"
    assert r["content"] == "hello world"
    assert r["timestamp"] == 2000.0
    assert r["source"] == "VoiceToVoice"
    assert r["location"] == "Kitchen"


@pytest.mark.unit
def test_get_history_range_ordered_asc(floxy_db):
    floxy_db.add_history_entry("user", "second", 2000.0)
    floxy_db.add_history_entry("user", "first", 1000.0)
    floxy_db.add_history_entry("user", "third", 3000.0)

    results = floxy_db.get_history_range(500.0, 3500.0)

    assert [r["content"] for r in results] == ["first", "second", "third"]


@pytest.mark.unit
def test_get_history_range_respects_limit(floxy_db):
    for i in range(10):
        floxy_db.add_history_entry("user", f"entry {i}", 1000.0 + i)

    results = floxy_db.get_history_range(999.0, 2000.0, limit=3)

    assert len(results) == 3


@pytest.mark.unit
def test_get_history_range_inclusive_boundaries(floxy_db):
    floxy_db.add_history_entry("user", "at start", 1000.0)
    floxy_db.add_history_entry("user", "at end", 2000.0)

    results = floxy_db.get_history_range(1000.0, 2000.0)

    assert len(results) == 2


# _build_fts_query — special character quoting (commit c58a8f5)

from common.FloxyDatabase import _build_fts_query, _count_token_coverage


@pytest.mark.unit
def test_build_fts_query_plain_words():
    assert _build_fts_query("hello world") == "hello OR world"


@pytest.mark.unit
def test_build_fts_query_quotes_hyphenated_tokens():
    result = _build_fts_query("wi-fi router")
    assert result == '"wi-fi" OR router'


@pytest.mark.unit
def test_build_fts_query_quotes_dots_and_slashes():
    result = _build_fts_query("config.yaml /etc/hosts")
    assert '"config.yaml"' in result
    assert '"/etc/hosts"' in result


@pytest.mark.unit
def test_build_fts_query_single_special_token():
    assert _build_fts_query("hello-world") == '"hello-world"'


@pytest.mark.unit
def test_build_fts_query_empty_string():
    assert _build_fts_query("") == ""



# ---------------------------------------------------------------------------
# _count_token_coverage
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_count_token_coverage_all_match():
    assert _count_token_coverage("the quick brown fox", ["quick", "fox"]) == 2


@pytest.mark.unit
def test_count_token_coverage_partial_match():
    assert _count_token_coverage("the quick brown fox", ["quick", "zebra"]) == 1


@pytest.mark.unit
def test_count_token_coverage_case_insensitive():
    assert _count_token_coverage("The Quick Brown Fox", ["quick", "FOX"]) == 2


@pytest.mark.unit
def test_count_token_coverage_empty_inputs():
    assert _count_token_coverage("", ["foo"]) == 0
    assert _count_token_coverage("hello", []) == 0
    assert _count_token_coverage(None, ["foo"]) == 0


# ---------------------------------------------------------------------------
# search_history coverage re-ranking
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_search_history_coverage_ranks_higher(floxy_db):
    """Rows matching more query tokens should rank above rows matching fewer."""
    now = time.time()
    floxy_db.add_history_entry(
        "user",
        "I went to the beach and enjoyed the sunshine and waves today it was great",
        now - 100,
    )
    floxy_db.add_history_entry(
        "user",
        "The beach was wonderful with sunshine and we played in the waves and built sandcastles",
        now - 50,
    )

    results = floxy_db.search_history("beach sunshine waves sandcastles")

    assert len(results) == 2
    assert results[0]["coverage"] >= results[1]["coverage"]


@pytest.mark.unit
def test_search_history_multi_token_query_keeps_single_coverage_rows(floxy_db):
    """Regression for 48aa9d59: a >=3-token query must NOT drop rows that
    match only one token. The previous implementation hard-filtered to
    coverage>=2 on multi-token queries, which silently killed cross-language
    queries like "Frances cykelhjälm bike helmet hjälm" where the source row is
    written in one language and only matches a single token. Coverage-based
    re-ranking still floats multi-match rows to the top — that's tested
    above — but the single-coverage tail must stay reachable."""
    now = time.time()
    floxy_db.add_history_entry(
        "user",
        "Frances wears a bike helmet in size 52-54 and is still growing pretty fast lately",
        now - 100,
    )
    floxy_db.add_history_entry(
        "user",
        "Quick reminder about a totally unrelated grocery list for the weekend run",
        now - 50,
    )

    results = floxy_db.search_history("Frances cykelhjälm bike helmet hjälm")

    contents = [r["content"] for r in results]
    assert any("52-54" in c for c in contents), (
        "row matching only one of 5 query tokens must still surface — "
        "quorum filter regression would drop it"
    )
    assert all("grocery" not in c for c in contents), (
        "FTS-non-matching row must still be excluded"
    )


@pytest.mark.unit
def test_search_history_content_floor_is_20_not_30(floxy_db):
    """Regression for 48aa9d59: the length(content) > 20 floor (down from 30)
    is load-bearing for short answer-shaped replies. A 25-char content that
    matches the query must now survive — under the old >30 floor it was
    silently excluded."""
    now = time.time()
    # 25 chars, matches the query, would have been excluded under old floor.
    short_match = "Frances helmet 52-54 ok!!"
    assert 20 < len(short_match) <= 30
    floxy_db.add_history_entry("user", short_match, now)

    results = floxy_db.search_history("Frances")

    contents = [r["content"] for r in results]
    assert short_match in contents, (
        f"25-char matching row must survive the >20 floor; got {contents}"
    )


@pytest.mark.unit
def test_search_history_overfetch_covers_broad_query(floxy_db):
    """Regression for 48aa9d59: the overfetch bump from min(max(limit*3, 60), 200)
    to a flat 2000 is what actually unblocked the production bug. FTS returned
    2104 candidate rows for a broad single-name query and the answer row sat at
    rank 988 — past the old 200-row cap, so it never entered the Python-side
    candidate pool no matter how the re-ranking was tuned.

    The two sibling tests above pin the quorum and content-floor invariants but
    leave the overfetch silently regressable: someone reading "min(max(limit*3,
    60), 200) → 2000" could easily revert it as 'too expensive' with no test
    failing. Drive a request whose result set exceeds the OLD 200-row ceiling
    so any cap below ~250 trips the assertion."""
    now = time.time()
    # 250 matching rows — comfortably past the old 200 cap, well under the new
    # 2000 ceiling. Keeps the test fast (<1s) while still pinning the invariant.
    floxy_db.add_history_entries([
        ("user", f"Frances did activity number {i} on a regular afternoon today", now - i, None)
        for i in range(250)
    ])

    results = floxy_db.search_history("Frances", limit=250)

    assert len(results) == 250, (
        f"expected 250 rows for broad query with limit=250, got {len(results)} — "
        "an overfetch regression (e.g. back to the old 200-row cap) would silently "
        "truncate the candidate set and re-open the rank-988 recall failure"
    )


@pytest.mark.unit
def test_fresh_database_runs_no_migrations(tmp_path, monkeypatch):
    """A blank file used to log 'added source/location column to existing database' and
    'schema version 0 -> 1' because the CREATE TABLE lagged the migrations and nothing
    stamped user_version on creation."""
    import common.FloxyDatabase as module
    notes = []
    monkeypatch.setattr(module, "note", lambda msg: notes.append(msg))
    monkeypatch.setattr(module, "trace", lambda msg: notes.append(msg))

    db = FloxyDatabase.__new__(FloxyDatabase)
    db.db_path = str(tmp_path / "blank.db")
    db.lock = threading.Lock()
    db._init_database()

    assert [n for n in notes if "migration" in n.lower() or "schema version" in n.lower()] == []
    conn = sqlite3.connect(db.db_path)
    columns = [col[1] for col in conn.execute("PRAGMA table_info(usage_stats)")]
    assert conn.execute("PRAGMA user_version").fetchone()[0] == module.SCHEMA_VERSION
    conn.close()
    assert {"source", "location"} <= set(columns)


@pytest.mark.unit
def test_batch_insert_raises_on_a_database_error(floxy_db):
    # A feed must see the failure: a silent 0 reads as "all duplicates" and moves the cursor
    # past rows that never landed.
    floxy_db.db_path = str(pathlib.Path(floxy_db.db_path).parent / "missing" / "nowhere.db")
    with pytest.raises(sqlite3.Error):
        floxy_db.add_history_entries([("system", "a", 1.0, None)], "iMessage")
