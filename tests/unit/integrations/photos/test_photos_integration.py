import json
import sqlite3
import sys
import threading
import time
import types
from unittest.mock import Mock, patch, MagicMock

import pytest

from common.FloxyDatabase import FloxyDatabase


# ---------------------------------------------------------------------------
# Database fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def floxy_db(tmp_path):
    db = FloxyDatabase.__new__(FloxyDatabase)
    db.db_path = str(tmp_path / "floxy_test.db")
    db.lock = threading.Lock()
    db._init_database()
    return db


# ---------------------------------------------------------------------------
# FloxyDatabase — photos table
# ---------------------------------------------------------------------------

class TestPhotosDatabase:

    def _store_sample(self, db, local_id="ABC/123", **overrides):
        defaults = dict(
            asset_guid=local_id,
            creation_date=1700000000.0,
            latitude=59.9,
            longitude=10.7,
            location_text="Oslo/Norway",
            albums="Vacation",
            persons="Alice, Bob",
            photo_type="photo",
            short_description="A sunset over the fjord",
            detailed_description="A beautiful sunset with orange and pink hues over the Norwegian fjord.",
            pretty_score=85,
            family_importance_score=60,
            indexed_at=time.time(),
        )
        defaults.update(overrides)
        db.store_photo(**defaults)

    @pytest.mark.unit
    def test_store_and_exists(self, floxy_db):
        assert not floxy_db.photo_exists("ABC/123")
        self._store_sample(floxy_db)
        assert floxy_db.photo_exists("ABC/123")

    @pytest.mark.unit
    def test_delete_photos_removes_rows_and_fts_entries(self, floxy_db):
        self._store_sample(floxy_db, local_id="KEEP/1", short_description="a red barn")
        self._store_sample(floxy_db, local_id="GONE/1", short_description="a blue boat")
        self._store_sample(floxy_db, local_id="GONE/2", short_description="a blue kayak")

        assert floxy_db.delete_photos({"GONE/1", "GONE/2", "NEVER/EXISTED"}) == 2
        assert floxy_db.delete_photos(set()) == 0

        assert floxy_db.get_indexed_photo_guids() == {"KEEP/1"}
        assert [r["asset_guid"] for r in floxy_db.search_photos("blue")] == []
        assert [r["asset_guid"] for r in floxy_db.search_photos("barn")] == ["KEEP/1"]

    @pytest.mark.unit
    def test_delete_photos_purges_their_history_entries(self, floxy_db):
        """A hidden asset's description also lives in history (search, prompt context, memory
        curation); the purge has to take that with it, and nothing else."""
        from common.FloxyDatabase import PHOTO_HISTORY_SOURCE
        self._store_sample(floxy_db, local_id="KEEP/1", short_description="a red barn")
        self._store_sample(floxy_db, local_id="GONE/1", short_description="a blue boat")
        conn = sqlite3.connect(floxy_db.db_path)
        ids = dict(conn.execute("SELECT asset_guid, id FROM photos").fetchall())
        conn.close()
        floxy_db.add_history_entry("system", f"New photo taken: a red barn [photo_id={ids['KEEP/1']}]\n\nA barn.",
                                   1000.0, source=PHOTO_HISTORY_SOURCE)
        floxy_db.add_history_entry("system", f"New photo taken: a blue boat [photo_id={ids['GONE/1']}]\n\nA boat.",
                                   1001.0, source=PHOTO_HISTORY_SOURCE)
        floxy_db.add_history_entry("system", f"New photo taken: decoy [photo_id={ids['GONE/1']}0]\n\nLonger id.",
                                   1002.0, source=PHOTO_HISTORY_SOURCE)
        floxy_db.add_history_entry("user", f"show me photo_id={ids['GONE/1']} please", 1003.0)

        assert floxy_db.delete_photos({"GONE/1"}) == 1

        remaining = [r["content"] for r in floxy_db.search_history("photo")]
        assert not any("a blue boat" in c for c in remaining)
        assert any("a red barn" in c for c in remaining)
        assert any("decoy" in c for c in remaining)
        assert any(c.startswith("show me") for c in remaining)

    @pytest.mark.unit
    def test_delete_photos_spans_the_sqlite_parameter_chunks(self, floxy_db):
        """A hidden-asset purge can name more GUIDs than one IN clause takes, so the delete
        works in chunks of 500; the boundary must not lose the tail or its history."""
        from common.FloxyDatabase import PHOTO_HISTORY_SOURCE
        guids = [f"BULK/{i}" for i in range(501)]
        self._store_sample(floxy_db, local_id="KEEP/1", short_description="a red barn")
        # The bulk rows are copies of that one, written in a single transaction: 501 calls of
        # store_photo are 501 commits.
        conn = sqlite3.connect(floxy_db.db_path)
        copied = ", ".join(row[1] for row in conn.execute("PRAGMA table_info(photos)")
                           if row[1] not in ("id", "asset_guid"))
        conn.executemany(f"INSERT INTO photos (asset_guid, {copied}) "
                         f"SELECT ?, {copied} FROM photos WHERE asset_guid = 'KEEP/1'", [(guid,) for guid in guids])
        conn.commit()
        last_id = conn.execute("SELECT id FROM photos WHERE asset_guid = 'BULK/500'").fetchone()[0]
        conn.close()
        floxy_db.add_history_entry("system", f"New photo taken: clip [photo_id={last_id}]\n\nThe last one.",
                                   1000.0, source=PHOTO_HISTORY_SOURCE)

        assert floxy_db.delete_photos(guids) == 501

        assert floxy_db.get_indexed_photo_guids() == {"KEEP/1"}
        assert not any("The last one" in r["content"] for r in floxy_db.search_history("clip"))

    @pytest.mark.unit
    def test_duplicate_insert_ignored(self, floxy_db):
        self._store_sample(floxy_db, local_id="DUP/1")
        self._store_sample(floxy_db, local_id="DUP/1", short_description="different")
        # Should still have only one row
        conn = sqlite3.connect(floxy_db.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM photos WHERE asset_guid='DUP/1'")
        assert cursor.fetchone()[0] == 1
        conn.close()

    @pytest.mark.unit
    def test_search_photos_fts(self, floxy_db):
        self._store_sample(floxy_db, local_id="S/1", short_description="Sunset at the beach",
                           detailed_description="Golden sunset over a sandy beach with waves")
        self._store_sample(floxy_db, local_id="S/2", short_description="Cat on the couch",
                           detailed_description="Orange tabby cat sleeping on a blue couch")

        results = floxy_db.search_photos("sunset")
        assert len(results) == 1
        assert results[0]["asset_guid"] == "S/1"

        results = floxy_db.search_photos("cat")
        assert len(results) == 1
        assert results[0]["asset_guid"] == "S/2"

    @pytest.mark.unit
    def test_search_photos_location(self, floxy_db):
        self._store_sample(floxy_db, local_id="L/1", location_text="Paris/France")
        self._store_sample(floxy_db, local_id="L/2", location_text="Tokyo/Japan")

        results = floxy_db.search_photos("Paris")
        assert len(results) == 1
        assert results[0]["asset_guid"] == "L/1"

    @pytest.mark.unit
    def test_search_photos_persons(self, floxy_db):
        self._store_sample(floxy_db, local_id="P/1", persons="Alice")
        self._store_sample(floxy_db, local_id="P/2", persons="Bob")

        results = floxy_db.search_photos("Alice")
        assert len(results) == 1
        assert results[0]["asset_guid"] == "P/1"

    @pytest.mark.unit
    def test_search_photos_sort_by_pretty(self, floxy_db):
        self._store_sample(floxy_db, local_id="SP/1", short_description="Beach sunset",
                           detailed_description="A beach sunset", pretty_score=50, family_importance_score=10)
        self._store_sample(floxy_db, local_id="SP/2", short_description="Beach party",
                           detailed_description="A beach party", pretty_score=95, family_importance_score=20)
        self._store_sample(floxy_db, local_id="SP/3", short_description="Beach walk",
                           detailed_description="A beach walk", pretty_score=75, family_importance_score=30)

        results = floxy_db.search_photos("beach", sort_by="pretty")
        assert len(results) == 3
        assert results[0]["asset_guid"] == "SP/2"
        assert results[1]["asset_guid"] == "SP/3"
        assert results[2]["asset_guid"] == "SP/1"

    @pytest.mark.unit
    def test_search_photos_sort_by_family(self, floxy_db):
        self._store_sample(floxy_db, local_id="SF/1", short_description="Garden flowers",
                           detailed_description="Garden flowers", pretty_score=80, family_importance_score=20)
        self._store_sample(floxy_db, local_id="SF/2", short_description="Garden party",
                           detailed_description="Garden party", pretty_score=60, family_importance_score=95)
        self._store_sample(floxy_db, local_id="SF/3", short_description="Garden path",
                           detailed_description="Garden path", pretty_score=70, family_importance_score=60)

        results = floxy_db.search_photos("garden", sort_by="family")
        assert len(results) == 3
        assert results[0]["asset_guid"] == "SF/2"
        assert results[1]["asset_guid"] == "SF/3"
        assert results[2]["asset_guid"] == "SF/1"

    @pytest.mark.unit
    def test_search_photos_sort_by_best(self, floxy_db):
        self._store_sample(floxy_db, local_id="SB/1", short_description="Mountain view",
                           detailed_description="Mountain view", pretty_score=40, family_importance_score=30)  # 70
        self._store_sample(floxy_db, local_id="SB/2", short_description="Mountain hike",
                           detailed_description="Mountain hike", pretty_score=80, family_importance_score=90)  # 170
        self._store_sample(floxy_db, local_id="SB/3", short_description="Mountain lake",
                           detailed_description="Mountain lake", pretty_score=70, family_importance_score=50)  # 120

        results = floxy_db.search_photos("mountain", sort_by="best")
        assert len(results) == 3
        assert results[0]["asset_guid"] == "SB/2"
        assert results[1]["asset_guid"] == "SB/3"
        assert results[2]["asset_guid"] == "SB/1"

    @pytest.mark.unit
    def test_search_photos_invalid_sort_by_falls_back(self, floxy_db):
        self._store_sample(floxy_db, local_id="SI/1", short_description="Lake view",
                           detailed_description="Lake view")
        results = floxy_db.search_photos("lake", sort_by="DROP TABLE photos")
        assert len(results) == 1

    @pytest.mark.unit
    def test_search_photos_empty_result(self, floxy_db):
        self._store_sample(floxy_db)
        results = floxy_db.search_photos("nonexistent_term_xyz")
        assert results == []

    @pytest.mark.unit
    def test_search_photos_wildcard_returns_all_sorted(self, floxy_db):
        self._store_sample(floxy_db, local_id="BK/1", short_description="Ugly snap", pretty_score=10, family_importance_score=20)
        self._store_sample(floxy_db, local_id="BK/2", short_description="Pretty sunset", pretty_score=95, family_importance_score=30)
        self._store_sample(floxy_db, local_id="BK/3", short_description="Family reunion", pretty_score=40, family_importance_score=90)
        results = floxy_db.search_photos("*", sort_by="pretty")
        assert len(results) == 3
        assert results[0]["short_description"] == "Pretty sunset"

        results = floxy_db.search_photos("*", sort_by="family")
        assert results[0]["short_description"] == "Family reunion"

        results = floxy_db.search_photos("*", sort_by="best")
        assert results[0]["short_description"] == "Family reunion"

    @pytest.mark.unit
    def test_search_photos_multiword_uses_or(self, floxy_db):
        self._store_sample(floxy_db, local_id="OR/1", short_description="Cute kitten",
                           detailed_description="A cute kitten playing")
        self._store_sample(floxy_db, local_id="OR/2", short_description="Wild animals",
                           detailed_description="Animals in the wild")
        self._store_sample(floxy_db, local_id="OR/3", short_description="Boring landscape",
                           detailed_description="A plain boring landscape")

        results = floxy_db.search_photos("cute animals")
        ids = {r["asset_guid"] for r in results}
        assert "OR/1" in ids
        assert "OR/2" in ids
        assert "OR/3" not in ids

    @pytest.mark.unit
    def test_search_photos_fts_relevance_blended_into_sort(self, floxy_db):
        """Photos matching more search terms should rank higher even with lower scores."""
        self._store_sample(floxy_db, local_id="REL/1", short_description="Beach sunset Italy",
                           detailed_description="Sunset at Italian beach with Alice and Carol",
                           persons="Alice, Carol", pretty_score=60, family_importance_score=60)
        self._store_sample(floxy_db, local_id="REL/2", short_description="Family dinner Italy",
                           detailed_description="Beautiful family dinner in Italy",
                           pretty_score=98, family_importance_score=98)

        results = floxy_db.search_photos("Alice Carol Italy", sort_by="best")
        assert results[0]["asset_guid"] == "REL/1"

    @pytest.mark.unit
    def test_search_photos_limit(self, floxy_db):
        for i in range(5):
            self._store_sample(floxy_db, local_id=f"LIM/{i}",
                               short_description=f"Beach scene {i}")
        results = floxy_db.search_photos("beach", limit=3)
        assert len(results) == 3

    # --- get_photo_count / get_photos_paginated ---

    @pytest.mark.unit
    def test_photo_count_after_inserts(self, floxy_db):
        for i in range(3):
            self._store_sample(floxy_db, local_id=f"CNT/{i}")
        assert floxy_db.get_photo_count() == 3

    @pytest.mark.unit
    def test_paginated_default_sort_by_date_desc(self, floxy_db):
        self._store_sample(floxy_db, local_id="PG/1", creation_date=1000.0)
        self._store_sample(floxy_db, local_id="PG/2", creation_date=3000.0)
        self._store_sample(floxy_db, local_id="PG/3", creation_date=2000.0)

        first = floxy_db.get_photos_paginated(offset=0, limit=1)
        assert len(first) == 1
        assert first[0]["asset_guid"] == "PG/2"

        second = floxy_db.get_photos_paginated(offset=1, limit=1)
        assert second[0]["asset_guid"] == "PG/3"

    @pytest.mark.unit
    def test_paginated_sort_asc(self, floxy_db):
        self._store_sample(floxy_db, local_id="ASC/1", creation_date=1000.0)
        self._store_sample(floxy_db, local_id="ASC/2", creation_date=3000.0)
        results = floxy_db.get_photos_paginated(offset=0, limit=1, sort_desc=False)
        assert results[0]["asset_guid"] == "ASC/1"

    @pytest.mark.unit
    def test_paginated_sort_by_pretty_score(self, floxy_db):
        self._store_sample(floxy_db, local_id="PS/1", pretty_score=50)
        self._store_sample(floxy_db, local_id="PS/2", pretty_score=90)
        self._store_sample(floxy_db, local_id="PS/3", pretty_score=70)

        results = floxy_db.get_photos_paginated(offset=0, limit=2, sort_by="pretty_score")
        assert results[0]["asset_guid"] == "PS/2"
        assert results[1]["asset_guid"] == "PS/3"

    @pytest.mark.unit
    def test_paginated_invalid_sort_column_falls_back(self, floxy_db):
        self._store_sample(floxy_db, local_id="FB/1", creation_date=1000.0)
        self._store_sample(floxy_db, local_id="FB/2", creation_date=2000.0)
        results = floxy_db.get_photos_paginated(offset=0, limit=1, sort_by="DROP TABLE photos")
        assert len(results) == 1
        assert results[0]["asset_guid"] == "FB/2"

    @pytest.mark.unit
    def test_paginated_offset_beyond_count(self, floxy_db):
        self._store_sample(floxy_db, local_id="OOB/1")
        results = floxy_db.get_photos_paginated(offset=100, limit=1)
        assert results == []

    # --- get_photo_by_id ---

    @pytest.mark.unit
    def test_get_photo_by_id(self, floxy_db):
        self._store_sample(floxy_db, local_id="BYID/L0/001", short_description="A lovely sunset")
        conn = sqlite3.connect(floxy_db.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM photos WHERE asset_guid = 'BYID/L0/001'")
        photo_id = cursor.fetchone()[0]
        conn.close()

        result = floxy_db.get_photo_by_id(photo_id)
        assert result is not None
        assert result["asset_guid"] == "BYID/L0/001"
        assert result["short_description"] == "A lovely sunset"

    @pytest.mark.unit
    def test_get_photo_by_id_not_found(self, floxy_db):
        result = floxy_db.get_photo_by_id(99999)
        assert result is None

    # --- get_indexed_photo_guids ---

    @pytest.mark.unit
    def test_get_indexed_photo_guids_returns_stored_guids(self, floxy_db):
        self._store_sample(floxy_db, local_id="AAA-BBB")
        self._store_sample(floxy_db, local_id="CCC-DDD")
        result = floxy_db.get_indexed_photo_guids()
        assert result == {"AAA-BBB", "CCC-DDD"}

    # --- get_photo_description ---

    @pytest.mark.unit
    def test_get_photo_description_exists(self, floxy_db):
        self._store_sample(floxy_db, local_id="DESC/L0/001", short_description="A lovely cat")
        assert floxy_db.get_photo_description("DESC/L0/001") == "A lovely cat"

    @pytest.mark.unit
    def test_get_photo_description_missing(self, floxy_db):
        assert floxy_db.get_photo_description("NOPE/L0/001") is None


# ---------------------------------------------------------------------------
# PhotosTab
# ---------------------------------------------------------------------------

class TestPhotosTab:

    @pytest.mark.unit
    def test_sort_cycling(self):
        from integrations.photos.PhotosTab import PhotosTab, SORT_MODES
        from unittest.mock import patch, MagicMock

        tab = PhotosTab()
        tab._count = 10
        tab._loading = False

        with patch.object(tab, '_load_data'):
            initial_idx = tab._sort_mode_idx
            for i in range(1, len(SORT_MODES) + 1):
                tab.handle_input('\x13')
                assert tab._sort_mode_idx == (initial_idx + i) % len(SORT_MODES)
                assert tab._index == 0

    @pytest.mark.unit
    def test_navigation_clamps(self):
        from integrations.photos.PhotosTab import PhotosTab
        from unittest.mock import patch, MagicMock

        tab = PhotosTab()
        tab._count = 5
        tab._index = 0

        with patch.object(tab, '_load_data'):
            tab._navigate(-10)
            assert tab._index == 0

            tab._index = 4
            tab._navigate(10)
            assert tab._index == 4

    @pytest.mark.unit
    def test_home_end(self):
        from integrations.photos.PhotosTab import PhotosTab
        from unittest.mock import patch, MagicMock

        tab = PhotosTab()
        tab._count = 100
        tab._index = 50

        with patch.object(tab, '_load_data'):
            tab.handle_input('home')
            assert tab._index == 0

            tab.handle_input('end')
            assert tab._index == 99

    @pytest.mark.unit
    def test_type_label_includes_duration_for_videos(self):
        from integrations.photos.PhotosTab import PhotosTab
        assert PhotosTab._type_label({"photo_type": "video, slo-mo", "duration_seconds": 12.4}) == "video, slo-mo \u00b7 12s"
        assert PhotosTab._type_label({"photo_type": "photo", "duration_seconds": None}) == "photo"
        assert PhotosTab._type_label({}) == "\u2014"

    @pytest.mark.unit
    def test_format_date(self):
        from integrations.photos.PhotosTab import PhotosTab
        assert PhotosTab._format_date(None) == "\u2014"
        formatted = PhotosTab._format_date(1700000000.0)
        assert "2023" in formatted


# ---------------------------------------------------------------------------
# PhotoProcessor
# ---------------------------------------------------------------------------

class TestImageBytesForAsset:
    """The Photos tab asks for a picture by the cloud asset GUID the photos table stores; the
    processor has to find the file (and iCloud identity) behind it in the library database."""

    MODULE = "integrations.photos.PhotoProcessor"

    @pytest.fixture
    def library(self, tmp_path):
        path = tmp_path / "Photos.sqlite"
        conn = sqlite3.connect(path)
        conn.executescript("""
            CREATE TABLE ZCLOUDMASTER (Z_PK INTEGER PRIMARY KEY, ZCLOUDMASTERGUID TEXT);
            CREATE TABLE ZASSET (Z_PK INTEGER PRIMARY KEY, ZUUID TEXT, ZFILENAME TEXT, ZDIRECTORY TEXT,
                                 ZCLOUDASSETGUID TEXT, ZLIBRARYSCOPE INTEGER, ZMASTER INTEGER);
            INSERT INTO ZCLOUDMASTER VALUES (7, 'MASTER-FAKE-0001');
            INSERT INTO ZASSET VALUES (1, 'UUID-FAKE-0001', 'IMG_0001.HEIC', '3/02', 'ASSET-FAKE-0001', NULL, 7);
            INSERT INTO ZASSET VALUES (2, 'UUID-FAKE-0002', 'IMG_0002.JPG', '4/11', 'ASSET-FAKE-0002', 1, NULL);
        """)
        conn.commit()
        conn.close()
        with patch(f"{self.MODULE}.PHOTOS_SQLITE_PATH", str(path)):
            yield path

    def test_the_row_behind_the_guid_names_the_file_and_its_icloud_identity(self, library):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        with patch.object(PhotoProcessor, "_get_image_bytes", return_value=b"\xff\xd8jpeg") as fetch:
            assert PhotoProcessor.image_bytes_for_asset("ASSET-FAKE-0001") == b"\xff\xd8jpeg"
        assert fetch.call_args.args == ("3/02", "IMG_0001.HEIC")
        assert fetch.call_args.kwargs == {"cloud_master_guid": "MASTER-FAKE-0001", "cloud_asset_guid": "ASSET-FAKE-0001",
                                          "library_scope": 0}   # NULL scope means the personal library

    def test_an_asset_without_a_cloud_master_still_resolves_through_its_own_guid(self, library):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        with patch.object(PhotoProcessor, "_get_image_bytes", return_value=b"jpeg") as fetch:
            PhotoProcessor.image_bytes_for_asset("ASSET-FAKE-0002")
        assert fetch.call_args.kwargs["cloud_master_guid"] is None
        assert fetch.call_args.kwargs["cloud_asset_guid"] == "ASSET-FAKE-0002"
        assert fetch.call_args.kwargs["library_scope"] == 1

    def test_a_guid_the_library_does_not_know_is_none_without_touching_the_disk(self, library):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        with patch.object(PhotoProcessor, "_get_image_bytes") as fetch:
            assert PhotoProcessor.image_bytes_for_asset("ASSET-FAKE-9999") is None
        fetch.assert_not_called()

    def test_a_missing_library_is_none_not_an_exception(self, tmp_path):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        with patch(f"{self.MODULE}.PHOTOS_SQLITE_PATH", str(tmp_path / "no-Photos.sqlite")), \
             patch.object(PhotoProcessor, "_get_image_bytes") as fetch:
            assert PhotoProcessor.image_bytes_for_asset("ASSET-FAKE-0001") is None
        fetch.assert_not_called()


class TestPhotoProcessorUuidDedup:

    @pytest.fixture(autouse=True)
    def clean_processing_set(self):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        PhotoProcessor._processing_uuids.clear()
        yield
        PhotoProcessor._processing_uuids.clear()

    @pytest.mark.unit
    def test_concurrent_same_uuid_only_one_processes(self):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        processor_a = PhotoProcessor()
        processor_b = PhotoProcessor()

        row = {"uuid": "RACE-UUID", "filename": "IMG.heic", "directory": "sub",
               "creation_ts": 1700000000.0, "latitude": 59.0, "longitude": 11.0,
               "kind_subtype": 0, "cloud_master_guid": None, "cloud_asset_guid": "LLM-FAIL-GUID"}

        llm_started = threading.Event()
        llm_proceed = threading.Event()
        results = {}

        original_inner = PhotoProcessor._process_one_inner

        def slow_inner(self_ref, r):
            llm_started.set()
            llm_proceed.wait(timeout=5)
            return original_inner(self_ref, r)

        with patch.object(PhotoProcessor, '_process_one_inner', slow_inner), \
             patch.object(PhotoProcessor, '_icloud_placeholder_type', return_value=None), \
             patch.object(PhotoProcessor, '_get_image_bytes', return_value=b"\xff\xd8fake"), \
             patch.object(PhotoProcessor, '_describe_subtypes', return_value="photo"), \
             patch.object(PhotoProcessor, '_get_albums', return_value=""), \
             patch.object(PhotoProcessor, '_get_persons', return_value=""), \
             patch.object(PhotoProcessor, '_get_location', return_value="Home"), \
             patch.object(PhotoProcessor, '_analyze_with_llm', return_value={
                 "short_description": "test", "detailed_description": "test desc",
                 "pretty_score": 50, "family_importance_score": 50}), \
             patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db:
            mock_db.return_value.photo_exists.return_value = False
            mock_db.return_value.store_photo.return_value = 42

            def run_a():
                results["a"] = processor_a._process_one(row)

            def run_b():
                llm_started.wait(timeout=5)
                results["b"] = processor_b._process_one(row)

            t_a = threading.Thread(target=run_a)
            t_b = threading.Thread(target=run_b)
            t_a.start()
            t_b.start()

            t_b.join(timeout=5)
            assert results.get("b") is False

            llm_proceed.set()
            t_a.join(timeout=5)

        assert "a" in results
        assert results["b"] is False

    @pytest.mark.unit
    def test_uuid_released_after_processing(self):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        processor = PhotoProcessor()
        row = {"uuid": "RELEASE-UUID", "filename": "IMG.heic", "directory": "sub",
               "creation_ts": 1700000000.0, "latitude": None, "longitude": None,
               "kind_subtype": 0, "cloud_master_guid": None, "cloud_asset_guid": "LLM-FAIL-GUID"}

        with patch.object(PhotoProcessor, '_icloud_placeholder_type', return_value=None), \
             patch.object(PhotoProcessor, '_get_image_bytes', return_value=b"\xff\xd8fake"), \
             patch.object(PhotoProcessor, '_describe_subtypes', return_value="photo"), \
             patch.object(PhotoProcessor, '_get_albums', return_value=""), \
             patch.object(PhotoProcessor, '_get_persons', return_value=""), \
             patch.object(PhotoProcessor, '_get_location', return_value=""), \
             patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db:
            mock_db.return_value.photo_exists.return_value = False
            mock_db.return_value.store_photo.return_value = 1

            mock_llm = MagicMock()
            mock_llm.think.return_value = '{"short_description":"test","detailed_description":"test desc","pretty_score":50,"family_importance_score":50}'
            mock_llm.background_wait_seconds.return_value = 0.0
            processor._llm = mock_llm

            processor._process_one(row)

        assert "RELEASE-UUID" not in PhotoProcessor._processing_uuids

    @pytest.mark.unit
    def test_uuid_released_on_exception(self):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        processor = PhotoProcessor()
        row = {"uuid": "EXCEPT-UUID", "filename": "IMG.heic", "directory": "sub",
               "creation_ts": 1700000000.0, "latitude": None, "longitude": None,
               "kind_subtype": 0, "cloud_master_guid": None, "cloud_asset_guid": "LLM-FAIL-GUID"}

        with patch.object(PhotoProcessor, '_process_one_inner', side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                processor._process_one(row)

        assert "EXCEPT-UUID" not in PhotoProcessor._processing_uuids

    @pytest.mark.unit
    def test_different_uuids_process_concurrently(self):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        processor = PhotoProcessor()

        row_a = {"uuid": "UUID-A", "filename": "A.heic", "directory": "sub",
                 "creation_ts": 1700000000.0, "latitude": None, "longitude": None,
                 "kind_subtype": 0, "cloud_master_guid": None, "cloud_asset_guid": "LLM-FAIL-GUID"}
        row_b = {"uuid": "UUID-B", "filename": "B.heic", "directory": "sub",
                 "creation_ts": 1700000001.0, "latitude": None, "longitude": None,
                 "kind_subtype": 0, "cloud_master_guid": None, "cloud_asset_guid": "LLM-FAIL-GUID"}

        both_running = threading.Barrier(2, timeout=5)
        results = {}

        original_inner = PhotoProcessor._process_one_inner

        def barrier_inner(self_ref, r):
            both_running.wait()
            return False

        with patch.object(PhotoProcessor, '_process_one_inner', barrier_inner):
            def run(key, row):
                results[key] = processor._process_one(row)

            t_a = threading.Thread(target=run, args=("a", row_a))
            t_b = threading.Thread(target=run, args=("b", row_b))
            t_a.start()
            t_b.start()
            t_a.join(timeout=5)
            t_b.join(timeout=5)

        assert results.get("a") is False
        assert results.get("b") is False


class TestPhotoProcessorParseAnalysis:

    @pytest.mark.unit
    def test_valid_json(self):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        response = '{"short_description":"A cat","detailed_description":"A fluffy cat on a mat","pretty_score":75,"family_importance_score":40}'
        result = PhotoProcessor._parse_analysis(response)
        assert result["short_description"] == "A cat"
        assert result["pretty_score"] == 75
        assert result["family_importance_score"] == 40

    @pytest.mark.unit
    def test_json_embedded_in_text(self):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        response = 'Here is the analysis:\n{"short_description":"Dog","detailed_description":"A happy dog","pretty_score":"80","family_importance_score":"90"}\nDone.'
        result = PhotoProcessor._parse_analysis(response)
        assert result["short_description"] == "Dog"
        assert result["pretty_score"] == 80

    @pytest.mark.unit
    def test_missing_key_raises(self):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        response = '{"short_description":"Only this"}'
        with pytest.raises(KeyError):
            PhotoProcessor._parse_analysis(response)

    @pytest.mark.unit
    def test_no_json_raises(self):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        with pytest.raises(ValueError):
            PhotoProcessor._parse_analysis("no json here")

    @pytest.mark.unit
    def test_unescaped_quote_in_transcribed_text_does_not_fail_the_photo(self):
        """Photos share the description parser's repair: an inch mark the model forgot to escape."""
        from integrations.photos.PhotoProcessor import PhotoProcessor
        response = ('{"short_description": "Fabric label", '
                    '"detailed_description": "The label reads Width: 140cm (55") on its second line.", '
                    '"pretty_score": "35", "family_importance_score": 10}')
        result = PhotoProcessor._parse_analysis(response)
        assert result["detailed_description"] == 'The label reads Width: 140cm (55") on its second line.'
        assert (result["pretty_score"], result["family_importance_score"]) == (35, 10)


class TestPhotoProcessorDescribeSubtypes:

    @pytest.mark.unit
    def test_plain_photo(self):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        assert PhotoProcessor._describe_subtypes(0) == "photo"

    @pytest.mark.unit
    def test_screenshot(self):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        assert "screenshot" in PhotoProcessor._describe_subtypes(1 << 3)

    @pytest.mark.unit
    def test_hdr_panorama(self):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        result = PhotoProcessor._describe_subtypes((1 << 0) | (1 << 1))
        assert "panorama" in result
        assert "HDR" in result

    @pytest.mark.unit
    def test_live_photo_deduplicates(self):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        # Both bit 3 and bit 4 map to "live"
        result = PhotoProcessor._describe_subtypes((1 << 3) | (1 << 4) | (1 << 5))
        assert result.count("live") == 1


class TestAppleLocation:
    """The place text comes from Photos.sqlite: the moment title and Apple's reverse-location
    plist (an NSKeyedArchiver archive of PLRevGeoLocationInfo → CNPostalAddress)."""

    @staticmethod
    def _archive(city="Göteborg", country="Sweden", is_home=False, state="Västra Götaland County"):
        import plistlib
        UID = plistlib.UID
        objects = [
            "$null",
            {"$class": UID(2), "postalAddress": UID(3), "isHome": is_home, "countryCode": UID(8)},
            {"$classname": "PLRevGeoLocationInfo", "$classes": ["PLRevGeoLocationInfo", "NSObject"]},
            {"$class": UID(4), "_city": UID(5), "_country": UID(6), "_state": UID(7),
             "_street": UID(0), "_ISOCountryCode": UID(8)},
            {"$classname": "CNPostalAddress", "$classes": ["CNPostalAddress", "NSObject"]},
            city, country, state, "SE",
        ]
        return plistlib.dumps({"$version": 100000, "$archiver": "NSKeyedArchiver",
                               "$top": {"root": UID(1)}, "$objects": objects}, fmt=plistlib.FMT_BINARY)

    @pytest.mark.unit
    def test_postal_address_reads_the_archive(self):
        from integrations.photos.PhotoProcessor import postal_address
        fields = postal_address(self._archive())
        assert fields["city"] == "Göteborg"
        assert fields["country"] == "Sweden"
        assert fields["state"] == "Västra Götaland County"
        assert fields["isHome"] is False
        assert "street" not in fields

    @pytest.mark.unit
    def test_postal_address_tolerates_missing_or_broken_data(self):
        from integrations.photos.PhotoProcessor import postal_address
        assert postal_address(None) == {}
        assert postal_address(b"") == {}
        assert postal_address(b"not a plist") == {}

    @pytest.mark.unit
    def test_moment_title_leads_city_and_country(self):
        from integrations.photos.PhotoProcessor import apple_location
        assert apple_location("Östra Haga", self._archive()) == "Östra Haga, Göteborg, Sweden"

    @pytest.mark.unit
    def test_title_equal_to_the_city_is_not_repeated(self):
        from integrations.photos.PhotoProcessor import apple_location
        assert apple_location("Göteborg", self._archive()) == "Göteborg, Sweden"

    @pytest.mark.unit
    def test_non_breaking_spaces_do_not_defeat_the_dedup(self):
        from integrations.photos.PhotoProcessor import apple_location
        archive = self._archive(city="Östra Haga")
        assert apple_location("Östra\xa0Haga", archive) == "Östra Haga, Sweden"

    @pytest.mark.unit
    def test_home_wins_from_either_source(self):
        from integrations.photos.PhotoProcessor import apple_location
        assert apple_location("Home", self._archive()) == "Home"
        assert apple_location("Göteborg", self._archive(is_home=True)) == "Home"

    @pytest.mark.unit
    def test_title_alone_when_there_is_no_reverse_location(self):
        from integrations.photos.PhotoProcessor import apple_location
        assert apple_location("Paris", None) == "Paris"
        assert apple_location(None, None) == ""
        assert apple_location("  ", b"") == ""

    @pytest.mark.unit
    def test_get_location_queries_photos_db(self, tmp_path):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        db_path = str(tmp_path / "Photos.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE ZASSET (Z_PK INTEGER, ZUUID TEXT, ZMOMENT INTEGER)")
        conn.execute("CREATE TABLE ZMOMENT (Z_PK INTEGER, ZTITLE TEXT)")
        conn.execute("CREATE TABLE ZADDITIONALASSETATTRIBUTES (Z_PK INTEGER, ZASSET INTEGER, ZREVERSELOCATIONDATA BLOB)")
        conn.execute("INSERT INTO ZASSET VALUES (1, 'uuid-1', 7)")
        conn.execute("INSERT INTO ZASSET VALUES (2, 'uuid-2', NULL)")
        conn.execute("INSERT INTO ZMOMENT VALUES (7, 'Östra Haga')")
        conn.execute("INSERT INTO ZADDITIONALASSETATTRIBUTES VALUES (1, 1, ?)", (self._archive(),))
        conn.commit()
        conn.close()

        with patch('integrations.photos.PhotoProcessor.PHOTOS_SQLITE_PATH', db_path):
            assert PhotoProcessor._get_location("uuid-1") == "Östra Haga, Göteborg, Sweden"
            assert PhotoProcessor._get_location("uuid-2") == ""
            assert PhotoProcessor._get_location("missing") == ""


class TestPhotoProcessorICloudPlaceholderType:

    @pytest.mark.unit
    def test_local_file_returns_none(self, tmp_path):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        local_file = tmp_path / "IMG_1234.heic"
        local_file.write_bytes(b"image data" * 200)
        result = PhotoProcessor._icloud_placeholder_type(str(local_file))
        assert result is None

    @pytest.mark.unit
    def test_tiny_file_returns_truncated(self, tmp_path):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        stub = tmp_path / "IMG_1234.heic"
        stub.write_bytes(b"\xff\xd8\xff\xe0converted_jpeg")
        assert PhotoProcessor._icloud_placeholder_type(str(stub)) == "truncated"

    @pytest.mark.unit
    def test_missing_file_with_icloud_stub_returns_stub(self, tmp_path):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        original = tmp_path / "IMG_1234.heic"
        stub = tmp_path / ".IMG_1234.heic.icloud"
        stub.write_bytes(b"stub")
        result = PhotoProcessor._icloud_placeholder_type(str(original))
        assert result == "stub"

    @pytest.mark.unit
    def test_missing_file_no_stub_returns_evicted(self, tmp_path):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        original = tmp_path / "IMG_1234.heic"
        result = PhotoProcessor._icloud_placeholder_type(str(original))
        assert result == "evicted"


class TestPhotoProcessorICloudDownload:

    @pytest.mark.unit
    def test_process_one_calls_brctl_for_evicted(self):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        processor = PhotoProcessor()
        row = {"uuid": "EVICT-UUID", "filename": "IMG.heic", "directory": "sub",
               "creation_ts": 1700000000.0, "latitude": None, "longitude": None,
               "kind_subtype": 0, "cloud_master_guid": "AbcCloudGuid123",
               "cloud_asset_guid": "EVICT-GUID"}

        with patch.object(PhotoProcessor, '_icloud_placeholder_type', return_value='evicted'), \
             patch.object(PhotoProcessor, '_try_icloud_download', return_value=None), \
             patch.object(PhotoProcessor, '_trigger_icloud_download_brctl') as mock_brctl, \
             patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db:
            mock_db.return_value.photo_exists.return_value = False
            needs_retry = processor._process_one(row)
            assert needs_retry is True
            mock_brctl.assert_called_once()

    @pytest.mark.unit
    def test_process_one_retries_truncated_original_without_brctl(self):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        processor = PhotoProcessor()
        row = {"uuid": "TRUNC-UUID", "filename": "IMG.heic", "directory": "sub",
               "creation_ts": 1700000000.0, "latitude": None, "longitude": None,
               "kind_subtype": 0, "cloud_master_guid": "AbcCloudGuid123",
               "cloud_asset_guid": "TRUNC-GUID"}

        with patch.object(PhotoProcessor, '_icloud_placeholder_type', return_value='truncated'), \
             patch.object(PhotoProcessor, '_try_icloud_download', return_value=None) as mock_dl, \
             patch.object(PhotoProcessor, '_trigger_icloud_download_brctl') as mock_brctl, \
             patch.object(PhotoProcessor, '_get_image_bytes') as mock_local, \
             patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db:
            mock_db.return_value.photo_exists.return_value = False
            needs_retry = processor._process_one(row)
            assert needs_retry is True
            mock_dl.assert_called_once()
            mock_brctl.assert_not_called()
            mock_local.assert_not_called()
            mock_db.return_value.store_photo.assert_not_called()

    @pytest.mark.unit
    def test_process_one_defers_when_no_asset_guid(self):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        processor = PhotoProcessor()
        row = {"uuid": "ORPHAN-UUID", "filename": "IMG.heic", "directory": "sub",
               "creation_ts": 1700000000.0, "latitude": None, "longitude": None,
               "kind_subtype": 0, "cloud_master_guid": None, "cloud_asset_guid": None}

        with patch.object(PhotoProcessor, '_icloud_placeholder_type', return_value='evicted'), \
             patch.object(PhotoProcessor, '_trigger_icloud_download_brctl') as mock_brctl, \
             patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db:
            mock_db.return_value.photo_exists.return_value = False
            needs_retry = processor._process_one(row)
            assert needs_retry is False
            mock_brctl.assert_not_called()
            mock_db.return_value.store_photo.assert_not_called()

    @pytest.mark.unit
    def test_process_one_skips_permanently_when_not_in_cloud(self):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        from integrations.photos.ICloudPhotoDownloader import PhotoNotInCloudError
        processor = PhotoProcessor()
        row = {"uuid": "DATA-UUID", "filename": "IMG.heic", "directory": "sub",
               "creation_ts": 1700000000.0, "latitude": None, "longitude": None,
               "kind_subtype": 0, "cloud_master_guid": "AbcCloudGuid456",
               "cloud_asset_guid": "DATA-GUID"}

        with patch.object(PhotoProcessor, '_icloud_placeholder_type', return_value='dataless'), \
             patch.object(PhotoProcessor, '_try_icloud_download',
                          side_effect=PhotoNotInCloudError("not found")), \
             patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db:
            mock_db.return_value.photo_exists.return_value = False
            needs_retry = processor._process_one(row)
            assert needs_retry is False
            mock_db.return_value.store_photo.assert_called_once()

    @pytest.mark.unit
    def test_process_one_retries_dataless_on_transient_failure(self):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        processor = PhotoProcessor()
        row = {"uuid": "DATA-UUID", "filename": "IMG.heic", "directory": "sub",
               "creation_ts": 1700000000.0, "latitude": None, "longitude": None,
               "kind_subtype": 0, "cloud_master_guid": "AbcCloudGuid456",
               "cloud_asset_guid": "DATA-GUID"}

        with patch.object(PhotoProcessor, '_icloud_placeholder_type', return_value='dataless'), \
             patch.object(PhotoProcessor, '_try_icloud_download', return_value=None), \
             patch.object(PhotoProcessor, '_trigger_icloud_download') as mock_dl, \
             patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db:
            mock_db.return_value.photo_exists.return_value = False
            needs_retry = processor._process_one(row)
            assert needs_retry is True
            mock_dl.assert_called_once()

    @pytest.mark.unit
    def test_bulk_unprocessed_tracks_icloud_pending(self):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        processor = PhotoProcessor()
        evicted_row = {"uuid": "EVICT-1", "filename": "IMG.heic", "directory": "sub",
                       "creation_ts": 1700000000.0, "latitude": None, "longitude": None,
                       "kind_subtype": 0}

        with patch.object(processor, '_find_unprocessed', return_value=([evicted_row], "")), \
             patch.object(processor, '_process_one', return_value=True), \
             patch.object(processor, '_retry_icloud_pending'):
            processor.process_bulk_unprocessed(count=10)
            assert "EVICT-1" in processor._pending_icloud
            assert processor._pending_icloud["EVICT-1"][1] == 0  # retry count

    @pytest.mark.unit
    def test_bulk_unprocessed_calls_retry_icloud_pending(self):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        processor = PhotoProcessor()

        with patch.object(processor, '_find_unprocessed', return_value=([], "No photos")), \
             patch.object(processor, '_retry_icloud_pending') as mock_retry:
            processor.process_bulk_unprocessed(count=10)
            mock_retry.assert_called_once()

class TestPhotoProcessorCorruptImage:

    @pytest.mark.unit
    def test_truncated_image_stored_as_skipped(self):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        from integrations.photos.ICloudPhotoDownloader import CorruptImageError
        processor = PhotoProcessor()
        row = {"uuid": "CORRUPT-UUID", "filename": "IMG.heic", "directory": "sub",
               "creation_ts": 1700000000.0, "latitude": 59.0, "longitude": 11.0,
               "kind_subtype": 0, "cloud_master_guid": "CloudGuid789",
               "cloud_asset_guid": "CORRUPT-GUID", "library_scope": 0}

        with patch.object(PhotoProcessor, '_icloud_placeholder_type', return_value='evicted'), \
             patch.object(PhotoProcessor, '_try_icloud_download',
                          side_effect=CorruptImageError("image file is truncated (42 bytes not processed)")), \
             patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db:
            mock_db.return_value.photo_exists.return_value = False
            needs_retry = processor._process_one(row)
            assert needs_retry is False
            mock_db.return_value.store_photo.assert_called_once()
            call_kwargs = mock_db.return_value.store_photo.call_args
            assert "corrupt image" in call_kwargs.kwargs.get("short_description", call_kwargs[1].get("short_description", ""))

    @pytest.mark.unit
    def test_corrupt_image_not_retained_in_pending(self):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        from integrations.photos.ICloudPhotoDownloader import CorruptImageError
        processor = PhotoProcessor()
        row = {"uuid": "CORRUPT-RETRY", "filename": "IMG.heic", "directory": "sub",
               "creation_ts": 1700000000.0, "latitude": None, "longitude": None,
               "kind_subtype": 0, "cloud_master_guid": "CloudGuid999",
               "cloud_asset_guid": None, "library_scope": 0}

        processor._pending_icloud["CORRUPT-RETRY"] = (row, 5)

        with patch.object(processor, '_process_one', side_effect=CorruptImageError("truncated")):
            processor._retry_icloud_pending()

        assert "CORRUPT-RETRY" not in processor._pending_icloud


class TestPhotoProcessorRetryExhaustion:

    @pytest.mark.unit
    def test_max_retries_stores_skipped(self):
        from integrations.photos.PhotoProcessor import PhotoProcessor, MAX_ICLOUD_RETRIES
        processor = PhotoProcessor()
        row = {"uuid": "EXHAUST-UUID", "filename": "IMG.heic", "directory": "sub",
               "creation_ts": 1700000000.0, "latitude": 59.0, "longitude": 11.0,
               "kind_subtype": 0, "cloud_master_guid": "CloudGuid123",
               "cloud_asset_guid": "EXHAUST-GUID"}

        processor._pending_icloud["EXHAUST-UUID"] = (row, MAX_ICLOUD_RETRIES)

        with patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db:
            mock_db.return_value.photo_exists.return_value = False
            processor._retry_icloud_pending()

        assert "EXHAUST-UUID" not in processor._pending_icloud
        mock_db.return_value.store_photo.assert_called_once()
        call_kwargs = mock_db.return_value.store_photo.call_args
        assert "iCloud download failed" in call_kwargs.kwargs.get(
            "short_description", call_kwargs[1].get("short_description", ""))

    @pytest.mark.unit
    def test_max_retries_keeps_the_video_duration_on_the_skipped_row(self):
        """The Photos tab labels a skipped row by its kind; without the duration a
        video that never downloaded would show up as a still."""
        from integrations.photos.PhotoProcessor import PhotoProcessor, MAX_ICLOUD_RETRIES
        processor = PhotoProcessor()
        row = {"uuid": "EXHAUST-VID", "filename": "IMG.mov", "directory": "sub",
               "creation_ts": 1700000000.0, "latitude": None, "longitude": None,
               "kind": "video", "duration": 12.5, "kind_subtype": 0,
               "cloud_master_guid": "CloudGuid789", "cloud_asset_guid": "EXHAUST-VID-GUID"}
        processor._pending_icloud["EXHAUST-VID"] = (row, MAX_ICLOUD_RETRIES)

        with patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db:
            mock_db.return_value.photo_exists.return_value = False
            processor._retry_icloud_pending()

        assert mock_db.return_value.store_photo.call_args.kwargs["duration_seconds"] == 12.5

    @pytest.mark.unit
    def test_max_retries_skips_if_already_exists(self):
        from integrations.photos.PhotoProcessor import PhotoProcessor, MAX_ICLOUD_RETRIES
        processor = PhotoProcessor()
        row = {"uuid": "EXISTS-UUID", "filename": "IMG.heic", "directory": "sub",
               "creation_ts": 1700000000.0, "latitude": None, "longitude": None,
               "kind_subtype": 0, "cloud_master_guid": "CloudGuid456",
               "cloud_asset_guid": None}

        processor._pending_icloud["EXISTS-UUID"] = (row, MAX_ICLOUD_RETRIES)

        with patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db:
            mock_db.return_value.photo_exists.return_value = True
            processor._retry_icloud_pending()

        assert "EXISTS-UUID" not in processor._pending_icloud
        mock_db.return_value.store_photo.assert_not_called()

    @pytest.mark.unit
    def test_below_max_retries_increments_and_retains(self):
        from integrations.photos.PhotoProcessor import PhotoProcessor, MAX_ICLOUD_RETRIES
        processor = PhotoProcessor()
        row = {"uuid": "RETRY-UUID", "filename": "IMG.heic", "directory": "sub",
               "creation_ts": 1700000000.0, "latitude": None, "longitude": None,
               "kind_subtype": 0}

        processor._pending_icloud["RETRY-UUID"] = (row, MAX_ICLOUD_RETRIES - 1)

        with patch.object(processor, '_process_one', return_value=True):
            processor._retry_icloud_pending()

        assert "RETRY-UUID" in processor._pending_icloud
        assert processor._pending_icloud["RETRY-UUID"][1] == MAX_ICLOUD_RETRIES

    @pytest.mark.unit
    def test_a_refusal_by_the_backend_keeps_the_asset_at_the_same_count(self):
        """A 402/429 on the retry is not a verdict on the asset: it stays, and does not burn
        one of its iCloud retries — the cursor has passed it, so this set is its only way in."""
        from assistant.llm.AbstractLLM import LLMUnavailable
        from integrations.photos.PhotoProcessor import PhotoProcessor
        processor = PhotoProcessor()
        row = {"uuid": "REFUSED-UUID", "filename": "IMG.heic", "directory": "sub",
               "creation_ts": 1700000000.0, "latitude": None, "longitude": None, "kind_subtype": 0}
        processor._pending_icloud["REFUSED-UUID"] = (row, 2)
        with patch.object(processor, 'provider_unavailable', return_value=None), \
             patch.object(processor, '_process_one', side_effect=LLMUnavailable("out of credits")):
            processor._retry_icloud_pending()
        assert processor._pending_icloud["REFUSED-UUID"] == (row, 2)

    @pytest.mark.unit
    def test_nothing_is_downloaded_while_the_backend_refuses(self):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        processor = PhotoProcessor()
        row = {"uuid": "WAIT-UUID", "filename": "IMG.heic"}
        processor._pending_icloud["WAIT-UUID"] = (row, 1)
        with patch.object(processor, 'provider_unavailable', return_value="rate limited"), \
             patch.object(processor, '_process_one') as process_one:
            processor._retry_icloud_pending()
        process_one.assert_not_called()
        assert processor._pending_icloud["WAIT-UUID"] == (row, 1)


class TestPhotoProcessorLlmFailure:

    @pytest.mark.unit
    def test_llm_failure_stores_skipped(self):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        processor = PhotoProcessor()
        row = {"uuid": "LLM-FAIL-UUID", "filename": "IMG.jpg", "directory": "sub",
               "creation_ts": 1700000000.0, "latitude": 59.0, "longitude": 11.0,
               "kind_subtype": 0, "cloud_master_guid": None, "cloud_asset_guid": "LLM-FAIL-GUID"}

        with patch.object(PhotoProcessor, '_icloud_placeholder_type', return_value=None), \
             patch.object(PhotoProcessor, '_get_image_bytes', return_value=b"\xff\xd8fake"), \
             patch.object(PhotoProcessor, '_describe_subtypes', return_value="photo"), \
             patch.object(PhotoProcessor, '_get_albums', return_value=""), \
             patch.object(PhotoProcessor, '_get_persons', return_value=""), \
             patch.object(PhotoProcessor, '_get_location', return_value="Oslo"), \
             patch.object(processor, '_analyze_with_llm', return_value=None), \
             patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db:
            mock_db.return_value.photo_exists.return_value = False
            needs_retry = processor._process_one(row)

        assert needs_retry is False
        mock_db.return_value.store_photo.assert_called_once()
        call_kwargs = mock_db.return_value.store_photo.call_args
        assert "LLM analysis failed" in call_kwargs.kwargs.get(
            "short_description", call_kwargs[1].get("short_description", ""))


class TestPhotoProcessorHistoryDedup:
    """The 'New photo taken' history entry must only be written by the thread that
    actually inserted the photo row — concurrent processors that lost the INSERT OR
    IGNORE race get photo_id=None and must not append a duplicate history entry."""

    @pytest.fixture(autouse=True)
    def clean_processing_set(self):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        PhotoProcessor._processing_uuids.clear()
        yield
        PhotoProcessor._processing_uuids.clear()

    def _row(self):
        return {"uuid": "HIST-UUID", "filename": "IMG.heic", "directory": "sub",
                "creation_ts": 1700000000.0, "latitude": 59.0, "longitude": 11.0,
                "kind_subtype": 0, "cloud_master_guid": None, "cloud_asset_guid": "LLM-FAIL-GUID"}

    def _run(self, store_photo_return):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        processor = PhotoProcessor()
        analysis = {"short_description": "A test photo",
                    "detailed_description": "A detailed test description",
                    "pretty_score": 50, "family_importance_score": 50}

        with patch.object(PhotoProcessor, '_icloud_placeholder_type', return_value=None), \
             patch.object(PhotoProcessor, '_get_image_bytes', return_value=b"\xff\xd8fake"), \
             patch.object(PhotoProcessor, '_describe_subtypes', return_value="photo"), \
             patch.object(PhotoProcessor, '_get_albums', return_value=""), \
             patch.object(PhotoProcessor, '_get_persons', return_value=""), \
             patch.object(PhotoProcessor, '_get_location', return_value="Home"), \
             patch.object(processor, '_analyze_with_llm', return_value=analysis), \
             patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db:
            mock_db.return_value.photo_exists.return_value = False
            mock_db.return_value.store_photo.return_value = store_photo_return
            processor._process_one(self._row())
        return mock_db.return_value

    @pytest.mark.unit
    def test_history_entry_written_when_photo_stored(self):
        db = self._run(store_photo_return=99)
        db.add_history_entry.assert_called_once()
        content = db.add_history_entry.call_args[0][1]
        assert "New photo taken" in content
        assert "[photo_id=99]" in content
        # delete_photos purges the entry by this source and the [photo_id=N] marker; the
        # writer drifting from either leaves hidden assets described in history forever.
        from common.FloxyDatabase import PHOTO_HISTORY_SOURCE
        assert db.add_history_entry.call_args.kwargs["source"] == PHOTO_HISTORY_SOURCE

    @pytest.mark.unit
    def test_history_entry_skipped_when_photo_insert_ignored(self):
        # store_photo returns None when INSERT OR IGNORE hit the UNIQUE constraint —
        # another thread already stored this photo, so we must not write history.
        db = self._run(store_photo_return=None)
        db.store_photo.assert_called_once()
        db.add_history_entry.assert_not_called()

    @pytest.mark.unit
    def test_history_entry_goes_to_the_sink_when_the_processor_has_one(self):
        """The Photos feed builds its processor with a sink: the row is handed to FeedRunner
        with the batch instead of being written here."""
        from assistant.integration.Feed import FeedEntry
        from integrations.photos.PhotoProcessor import PhotoProcessor
        sunk = []
        analysis = {"short_description": "A test photo",
                    "detailed_description": "A detailed test description",
                    "pretty_score": 50, "family_importance_score": 50}
        with patch.object(PhotoProcessor, '_icloud_placeholder_type', return_value=None), \
             patch.object(PhotoProcessor, '_get_image_bytes', return_value=b"\xff\xd8fake"), \
             patch.object(PhotoProcessor, '_describe_subtypes', return_value="photo"), \
             patch.object(PhotoProcessor, '_get_albums', return_value=""), \
             patch.object(PhotoProcessor, '_get_persons', return_value=""), \
             patch.object(PhotoProcessor, '_get_location', return_value="Home"), \
             patch.object(PhotoProcessor, '_analyze_with_llm', return_value=analysis), \
             patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db:
            mock_db.return_value.photo_exists.return_value = False
            mock_db.return_value.store_photo.return_value = 99
            PhotoProcessor(entry_sink=sunk.append)._process_one(self._row())
        db = mock_db.return_value
        db.add_history_entry.assert_not_called()
        assert len(sunk) == 1
        entry = sunk[0]
        assert isinstance(entry, FeedEntry)
        assert entry.timestamp == 1700000000.0
        assert entry.role == "system"
        assert "New photo taken" in entry.content and "[photo_id=99]" in entry.content


class TestGetImageBytesICloudFallback:

    @pytest.mark.unit
    def test_placeholder_triggers_icloud_download(self):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        jpeg_bytes = b"\xff\xd8\xff\xe0icloud_download"

        with patch.object(PhotoProcessor, '_icloud_placeholder_type', return_value="stub"), \
             patch('integrations.photos.ICloudPhotoDownloader.ICloudPhotoDownloader') as MockDL:
            MockDL.get_instance.return_value.download_photo_bytes.return_value = jpeg_bytes
            result = PhotoProcessor._get_image_bytes(
                "sub", "IMG.heic",
                cloud_master_guid="CloudGuid", library_scope=0,
            )

        assert result == jpeg_bytes
        MockDL.get_instance.return_value.download_photo_bytes.assert_called_once_with("CloudGuid", 0)

    @pytest.mark.unit
    def test_placeholder_uses_cloud_asset_guid_fallback(self):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        jpeg_bytes = b"\xff\xd8\xff\xe0icloud_asset"

        with patch.object(PhotoProcessor, '_icloud_placeholder_type', return_value="evicted"), \
             patch('integrations.photos.ICloudPhotoDownloader.ICloudPhotoDownloader') as MockDL:
            MockDL.get_instance.return_value.download_photo_bytes.return_value = jpeg_bytes
            result = PhotoProcessor._get_image_bytes(
                "sub", "IMG.heic",
                cloud_master_guid=None, cloud_asset_guid="AssetGuid", library_scope=1,
            )

        assert result == jpeg_bytes
        MockDL.get_instance.return_value.download_photo_bytes.assert_called_once_with("AssetGuid", 1)

    @pytest.mark.unit
    def test_placeholder_no_guid_no_download_attempt(self):
        from integrations.photos.PhotoProcessor import PhotoProcessor

        with patch.object(PhotoProcessor, '_icloud_placeholder_type',
                          side_effect=["stub", "stub"]), \
             patch('integrations.photos.ICloudPhotoDownloader.ICloudPhotoDownloader') as MockDL:
            result = PhotoProcessor._get_image_bytes("sub", "IMG.heic")

        MockDL.get_instance.return_value.download_photo_bytes.assert_not_called()
        assert result is None


# ---------------------------------------------------------------------------
# PhotosIntegration — the Integration half
# ---------------------------------------------------------------------------

PI = 'integrations.photos.PhotosIntegration'


class TestPhotosIntegration:

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        from integrations.photos.PhotosIntegration import PhotosIntegration
        PhotosIntegration.reset_instance()
        yield
        PhotosIntegration.reset_instance()

    @staticmethod
    def _healthy_db():
        """check_health's db probe passes: the path exists and the connection answers."""
        conn = MagicMock()
        conn.__enter__.return_value = conn
        return patch(f'{PI}.os.path.exists', return_value=True), patch(f'{PI}.sqlite3.connect', return_value=conn)

    @pytest.mark.unit
    def test_name_and_config_keys_follow_the_photo_indexing_tier(self):
        # list_integrations once tagged Photos with the Anthropic key while it indexed on
        # Gemini Flash (the tier default), and the FTUE assistant "rescued" the user from a
        # $500 backfill that was never going to happen. The key must be the tier backend's.
        from integrations.photos.PhotosIntegration import PhotosIntegration
        integration = PhotosIntegration()
        assert integration.name == "Photos"
        with patch('assistant.llm.parse_llm_config', return_value=('GeminiLLM', 'gemini-3.8-flash')):
            assert integration.config_keys == ["api_keys.gemini"]
        with patch('assistant.llm.parse_llm_config', return_value=('ClaudeLLM', None)):
            assert integration.config_keys == ["api_keys.anthropic"]
        with patch('assistant.llm.parse_llm_config', return_value=('OllamaLLM', None)):
            assert integration.config_keys == []

    @pytest.mark.unit
    def test_is_a_feed_under_the_history_source_name(self):
        from assistant.integration.Feed import Feed
        from integrations.photos.PhotosIntegration import PhotosIntegration, POLL_INTERVAL
        from common.FloxyDatabase import PHOTO_HISTORY_SOURCE
        integration = PhotosIntegration()
        assert isinstance(integration, Feed)
        # The feed name is the history `source`: the Chat tab, search_history and the
        # hidden-asset purge all match rows on it.
        assert integration.feed_name == PHOTO_HISTORY_SOURCE
        assert integration.config_slug == "photos"
        assert integration.poll_interval == POLL_INTERVAL

    @pytest.mark.unit
    def test_in_flight_is_the_number_of_history_rows_queued_by_the_batch(self):
        from assistant.integration.Feed import FeedEntry
        from integrations.photos.PhotosIntegration import PhotosIntegration
        integration = PhotosIntegration()
        assert integration.in_flight() == 0
        integration._queue_entry(FeedEntry(1.0, "a"))
        integration._queue_entry(FeedEntry(2.0, "b"))
        assert integration.in_flight() == 2
        integration._drain_entries()
        assert integration.in_flight() == 0, "handed over with the batch, nothing is in flight"

    @pytest.mark.unit
    def test_check_health_raises_without_the_photos_database(self):
        from integrations.photos.PhotosIntegration import PhotosIntegration
        integration = PhotosIntegration()
        with patch(f'{PI}.os.path.exists', return_value=False), \
             patch.object(integration, '_check_icloud_download') as mock_check:
            with pytest.raises(FileNotFoundError):
                integration.check_health()
        mock_check.assert_not_called()

    @pytest.mark.unit
    def test_check_health_propagates_a_broken_database(self):
        from integrations.photos.PhotosIntegration import PhotosIntegration
        integration = PhotosIntegration()
        with patch(f'{PI}.os.path.exists', return_value=True), \
             patch(f'{PI}.sqlite3.connect', side_effect=sqlite3.OperationalError("locked")), \
             patch.object(integration, '_check_icloud_download') as mock_check:
            with pytest.raises(sqlite3.OperationalError):
                integration.check_health()
        mock_check.assert_not_called()

    @pytest.mark.unit
    def test_icloud_check_skipped_within_interval(self):
        from integrations.photos.PhotosIntegration import PhotosIntegration
        integration = PhotosIntegration()
        exists, connect = self._healthy_db()
        with exists, connect, patch.object(integration, '_check_icloud_download') as mock_check:
            integration.check_health()
            mock_check.assert_called_once()
            integration.check_health()
            mock_check.assert_called_once()  # not called again

    @pytest.mark.unit
    def test_icloud_check_runs_again_after_interval(self):
        import time as _time
        from integrations.photos.PhotosIntegration import (
            PhotosIntegration, ICLOUD_CHECK_INTERVAL_SECONDS)
        integration = PhotosIntegration()
        exists, connect = self._healthy_db()
        with exists, connect, patch.object(integration, '_check_icloud_download') as mock_check:
            integration.check_health()
            assert mock_check.call_count == 1
            integration._last_icloud_check = _time.time() - ICLOUD_CHECK_INTERVAL_SECONDS - 1
            integration.check_health()
            assert mock_check.call_count == 2

    @pytest.mark.unit
    def test_icloud_check_download_success(self, tmp_path):
        from integrations.photos.PhotosIntegration import PhotosIntegration
        integration = PhotosIntegration()

        db_path = str(tmp_path / "Photos.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE ZASSET (Z_PK INTEGER, ZUUID TEXT, ZCLOUDASSETGUID TEXT, "
                     "ZLIBRARYSCOPE INTEGER, ZKIND INTEGER, ZTRASHEDSTATE INTEGER, ZMASTER INTEGER, ZHIDDEN INTEGER)")
        conn.execute("CREATE TABLE ZCLOUDMASTER (Z_PK INTEGER, ZCLOUDMASTERGUID TEXT)")
        conn.execute("INSERT INTO ZCLOUDMASTER VALUES (1, 'cloud-guid-123')")
        conn.execute("INSERT INTO ZASSET VALUES (1, 'uuid-abc', 'asset-guid', 0, 0, 0, 1, 0)")
        conn.commit()
        conn.close()

        mock_downloader = Mock()
        mock_downloader.download_photo_bytes.return_value = b'\xff\xd8\xff' + b'\x00' * 100

        with patch(f'{PI}.PHOTOS_SQLITE_PATH', db_path), \
             patch(f'{PI}.os.path.exists', return_value=True), \
             patch(f'{PI}.config.get', return_value={"apple_id": "x"}), \
             patch('integrations.photos.ICloudPhotoDownloader.ICloudPhotoDownloader.get_instance',
                   return_value=mock_downloader):
            integration._check_icloud_download()

        mock_downloader.download_photo_bytes.assert_called_once_with('cloud-guid-123', 0)

    @pytest.mark.unit
    def test_icloud_check_download_failure_logs_error(self, tmp_path):
        from integrations.photos.PhotosIntegration import PhotosIntegration
        integration = PhotosIntegration()

        db_path = str(tmp_path / "Photos.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE ZASSET (Z_PK INTEGER, ZUUID TEXT, ZCLOUDASSETGUID TEXT, "
                     "ZLIBRARYSCOPE INTEGER, ZKIND INTEGER, ZTRASHEDSTATE INTEGER, ZMASTER INTEGER, ZHIDDEN INTEGER)")
        conn.execute("CREATE TABLE ZCLOUDMASTER (Z_PK INTEGER, ZCLOUDMASTERGUID TEXT)")
        conn.execute("INSERT INTO ZCLOUDMASTER VALUES (1, 'cloud-guid-456')")
        conn.execute("INSERT INTO ZASSET VALUES (1, 'uuid-def', 'asset-guid', 0, 0, 0, 1, 0)")
        conn.commit()
        conn.close()

        mock_downloader = Mock()
        mock_downloader.download_photo_bytes.return_value = None

        with patch(f'{PI}.PHOTOS_SQLITE_PATH', db_path), \
             patch(f'{PI}.os.path.exists', return_value=True), \
             patch(f'{PI}.config.get', return_value={"apple_id": "x"}), \
             patch('integrations.photos.ICloudPhotoDownloader.ICloudPhotoDownloader.get_instance',
                   return_value=mock_downloader), \
             patch(f'{PI}.error') as mock_error:
            integration._check_icloud_download()

        mock_error.assert_called_once()
        assert "FAILED" in mock_error.call_args[0][0]

    @pytest.mark.unit
    def test_icloud_check_skipped_when_icloud_not_configured(self):
        from integrations.photos.PhotosIntegration import PhotosIntegration
        integration = PhotosIntegration()
        with patch(f'{PI}.config.get', return_value={}), \
             patch('integrations.photos.ICloudPhotoDownloader.ICloudPhotoDownloader.get_instance') as dl:
            integration._check_icloud_download()
        dl.assert_not_called()


# ---------------------------------------------------------------------------
# PhotosIntegration — the Feed half
# ---------------------------------------------------------------------------

class TestPhotosFeed:
    """The runner's view of the library: open, the first cursor, one fetch, close."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        from integrations.photos.PhotosIntegration import PhotosIntegration
        PhotosIntegration.reset_instance()
        yield
        PhotosIntegration.reset_instance()

    @staticmethod
    def _row(uuid, creation, guid, kind="photo"):
        return {"uuid": uuid, "creation_date_apple": creation, "cloud_asset_guid": guid, "kind": kind}

    @staticmethod
    def _opened(processor=None):
        """A PhotosIntegration whose feed is open on a mocked processor. Returns
        (integration, processor); the sink the real processor would call is exposed as
        `integration._queue_entry`."""
        from integrations.photos.PhotosIntegration import PhotosIntegration
        integration = PhotosIntegration()
        if processor is None:
            processor = Mock()
            processor.provider_unavailable.return_value = None    # the backend takes requests
            processor.refused_last_run.return_value = []
        with patch(f'{PI}.os.path.exists', return_value=True), \
             patch(f'{PI}.PhotoProcessor', return_value=processor):
            assert integration.open_feed() is True
        return integration, processor

    @pytest.mark.unit
    def test_open_feed_fails_without_the_database(self):
        from integrations.photos.PhotosIntegration import PhotosIntegration
        integration = PhotosIntegration()
        with patch(f'{PI}.os.path.exists', return_value=False), \
             patch(f'{PI}.PhotoProcessor') as processor_cls:
            assert integration.open_feed() is False
        assert "not found" in integration.open_error
        processor_cls.assert_not_called()

    @pytest.mark.unit
    def test_open_feed_builds_a_processor_with_the_entry_sink_and_purges_hidden_assets(self):
        """Assets hidden after they were indexed are only ever removed here, when the feed
        opens — the fetch query just stops seeing them."""
        from integrations.photos.PhotosIntegration import PhotosIntegration
        integration = PhotosIntegration()
        processor = Mock()
        with patch(f'{PI}.os.path.exists', return_value=True), \
             patch(f'{PI}.PhotoProcessor', return_value=processor) as processor_cls:
            assert integration.open_feed() is True
        assert integration.open_error is None
        processor_cls.assert_called_once_with(entry_sink=integration._queue_entry)
        processor.remove_hidden_assets.assert_called_once()

    @pytest.mark.unit
    def test_fetch_indexes_only_guid_bearing_unindexed_rows_and_retries_icloud_first(self):
        from integrations.photos.PhotosIntegration import PhotosIntegration
        integration, processor = self._opened()
        order = []
        processor._retry_icloud_pending.side_effect = lambda: order.append("retry")
        processor.process_rows.side_effect = lambda rows, batched: order.append("process")
        rows = [self._row("done", 100.0, "G-DONE"), self._row("new", 200.0, "G-NEW"),
                self._row("uploading", time.time() - 978307200 - 60, None)]

        with patch.object(PhotosIntegration, '_assets_after', return_value=rows) as assets_after, \
             patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db:
            mock_db.return_value.get_indexed_photo_guids.return_value = {"G-DONE"}
            batch = integration.fetch("50.0", backfill=True)

        assets_after.assert_called_once_with(50.0, PhotosIntegration.fetch.__globals__["FETCH_BATCH_SIZE"])
        assert order == ["retry", "process"]
        handed = processor.process_rows.call_args.args[0]
        assert [r["uuid"] for r in handed] == ["new"]
        assert processor.process_rows.call_args.kwargs["batched"] is True, "the catch-up is the batch API's work"
        assert batch.exhausted is True
        assert batch.entries == []

    @pytest.mark.unit
    def test_fetch_hands_over_the_entries_the_processor_queued(self):
        """process_rows runs the workers to completion; every history row they produced
        went through the sink and comes back in this batch — none before, none after."""
        from assistant.integration.Feed import FeedEntry
        from integrations.photos.PhotosIntegration import PhotosIntegration
        integration, processor = self._opened()
        entries = [FeedEntry(1_700_000_000.0, "New photo taken: a barn"),
                   FeedEntry(1_700_000_100.0, "New photo taken: a boat")]
        processor.process_rows.side_effect = lambda rows, batched: [integration._queue_entry(e) for e in entries]
        rows = [self._row("a", 100.0, "G-A"), self._row("b", 200.0, "G-B")]

        with patch.object(PhotosIntegration, '_assets_after', return_value=rows), \
             patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db:
            mock_db.return_value.get_indexed_photo_guids.return_value = set()
            batch = integration.fetch("0", backfill=False)
            second = integration.fetch("200.0", backfill=False)

        assert all(c.kwargs["batched"] is False for c in processor.process_rows.call_args_list), \
            "a live fetch never waits on a batch job: new photos are described within seconds"
        assert batch.entries == entries
        assert batch.cursor == "200.0"
        assert second.entries == entries  # the fake queues again on every call
        assert integration._drain_entries() == []

    @pytest.mark.unit
    def test_fetch_is_exhausted_only_below_a_full_batch(self):
        from integrations.photos.PhotosIntegration import PhotosIntegration, FETCH_BATCH_SIZE
        integration, processor = self._opened()
        full = [self._row(f"u{i}", float(i), f"G{i}") for i in range(FETCH_BATCH_SIZE)]

        with patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db:
            mock_db.return_value.get_indexed_photo_guids.return_value = {r["cloud_asset_guid"] for r in full}
            with patch.object(PhotosIntegration, '_assets_after', return_value=full):
                assert integration.fetch("0", backfill=True).exhausted is False
            with patch.object(PhotosIntegration, '_assets_after', return_value=full[:-1]):
                assert integration.fetch("0", backfill=True).exhausted is True
            with patch.object(PhotosIntegration, '_assets_after', return_value=[]):
                empty = integration.fetch("0", backfill=False)
        assert empty.exhausted is True
        assert empty.cursor == "0.0"
        processor.process_rows.assert_not_called()

    @pytest.mark.unit
    def test_a_full_batch_whose_cursor_is_held_is_exhausted(self):
        """A full page after the cursor, its first asset still uploading: the cursor cannot
        move, so the very same rows would come back at once — the runner must sleep until the
        next poll, not fetch again in a tight loop."""
        from integrations.photos.PhotosIntegration import PhotosIntegration, FETCH_BATCH_SIZE
        integration, processor = self._opened()
        young = time.time() - 978307200 - 60
        rows = [self._row("uploading", young, None)] + \
               [self._row(f"u{i}", young + i, f"G{i}") for i in range(1, FETCH_BATCH_SIZE)]

        with patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db:
            mock_db.return_value.get_indexed_photo_guids.return_value = {r["cloud_asset_guid"] for r in rows[1:]}
            with patch.object(PhotosIntegration, '_assets_after', return_value=rows):
                batch = integration.fetch("0", backfill=True)

        assert float(batch.cursor) == 0.0
        assert batch.exhausted is True
        processor.process_rows.assert_not_called()

    @pytest.mark.unit
    def test_fetch_holds_the_cursor_below_a_photo_still_uploading(self):
        """A poll returning a GUID-less newest photo leaves the cursor below it, so the very
        next poll still sees it."""
        from integrations.photos.PhotosIntegration import PhotosIntegration
        integration, processor = self._opened()
        young = time.time() - 978307200 - 60
        rows = [self._row("g1", young - 100, "g1"), self._row("uploading", young, None)]

        with patch.object(PhotosIntegration, '_assets_after', return_value=rows), \
             patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db:
            mock_db.return_value.get_indexed_photo_guids.return_value = {"g1"}
            batch = integration.fetch("0", backfill=False)

        assert float(batch.cursor) == young - 100
        processor.process_rows.assert_not_called()

    @pytest.mark.unit
    def test_close_feed_writes_leftover_entries_straight_to_the_history(self):
        """A batch cut short by a stop has its photos in the ledger already; the history rows
        only exist in the queue, so close writes them or they are lost for good."""
        from assistant.integration.Feed import FeedEntry
        from common.FloxyDatabase import PHOTO_HISTORY_SOURCE
        integration, _ = self._opened()
        integration._queue_entry(FeedEntry(1_700_000_000.0, "New photo taken: late"))

        with patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db:
            integration.close_feed()
            integration.close_feed()

        mock_db.return_value.add_history_entries.assert_called_once_with(
            [("system", "New photo taken: late", 1_700_000_000.0, None)], PHOTO_HISTORY_SOURCE)

# ---------------------------------------------------------------------------
# PhotosIntegration.advance_cursor — the iCloud GUID hold
# ---------------------------------------------------------------------------

class TestPhotosFeedCursor:
    """A just-taken photo has no ZCLOUDASSETGUID until its iCloud upload finishes. The
    cursor must NOT move past such a photo — otherwise it falls behind the cursor and is
    never re-examined by the feed."""

    NOW = 1_700_000_000.0

    @staticmethod
    def _row(creation: float, guid, uuid=None):
        return {"uuid": uuid or f"u{creation}-{guid}", "creation_date_apple": creation, "cloud_asset_guid": guid}

    @staticmethod
    def _young(offset_seconds: float) -> float:
        """An Apple timestamp `offset_seconds` before NOW — well inside the hold window."""
        from integrations.photos.PhotoProcessor import APPLE_EPOCH_OFFSET
        return TestPhotosFeedCursor.NOW - APPLE_EPOCH_OFFSET - offset_seconds

    def _advance(self, current, rows):
        from integrations.photos.PhotosIntegration import PhotosIntegration
        return PhotosIntegration.advance_cursor(current, rows, self.NOW)

    @pytest.mark.unit
    def test_all_rows_have_guid_advances_to_max(self):
        rows = [self._row(self._young(300), "g1"), self._row(self._young(200), "g2"),
                self._row(self._young(100), "g3")]
        cursor, held = self._advance(0.0, rows)
        assert cursor == self._young(100)
        assert held == set()

    @pytest.mark.unit
    def test_trailing_guidless_holds_cursor_below_it(self):
        # Newest photo still uploading (no GUID) — must keep getting re-polled.
        rows = [self._row(self._young(300), "g1"), self._row(self._young(200), "g2"),
                self._row(self._young(100), None, uuid="up")]
        cursor, held = self._advance(0.0, rows)
        assert cursor == self._young(200)
        assert held == {"up"}

    @pytest.mark.unit
    def test_guidless_in_middle_stops_advancement(self):
        # A GUID-less photo with a GUID-having photo created after it: the cursor must
        # not jump to the later photo, or the GUID-less one is lost.
        rows = [self._row(self._young(300), "g1"), self._row(self._young(200), None, uuid="up"),
                self._row(self._young(100), "g3")]
        cursor, held = self._advance(0.0, rows)
        assert cursor == self._young(300)
        assert held == {"up"}

    @pytest.mark.unit
    def test_equal_timestamp_guid_before_guidless_holds_below_shared_ts(self):
        # ORDER BY ZDATECREATED has no tiebreaker, so a GUID-having row can be iterated
        # ahead of a GUID-less sibling at the SAME timestamp. The cursor must NOT land
        # on that shared timestamp, or the strict `> cursor` poll drops the held one.
        rows = [self._row(self._young(300), "g1"), self._row(self._young(200), "g2"),
                self._row(self._young(200), None, uuid="up")]
        cursor, held = self._advance(0.0, rows)
        assert cursor == self._young(300)
        assert held == {"up"}

    @pytest.mark.unit
    def test_equal_timestamp_order_independent(self):
        # Reversed tie order yields the same held cursor — result is order-independent.
        rows = [self._row(self._young(300), "g1"), self._row(self._young(200), None, uuid="up"),
                self._row(self._young(200), "g2")]
        cursor, held = self._advance(0.0, rows)
        assert cursor == self._young(300)
        assert held == {"up"}

    @pytest.mark.unit
    def test_leading_guidless_keeps_current_cursor(self):
        rows = [self._row(self._young(250), None, uuid="a"), self._row(self._young(150), None, uuid="b")]
        current = self._young(280)
        cursor, held = self._advance(current, rows)
        assert cursor == current
        assert held == {"a", "b"}

    @pytest.mark.unit
    def test_guidless_older_than_the_hold_window_does_not_hold(self):
        """An asset still without a GUID a day later is not uploading any more (iCloud Photos
        off, a stuck upload); holding for it would pin the catch-up forever. The cursor
        moves past it and the Photos tab's bulk indexing is what finds it later."""
        from integrations.photos.PhotosIntegration import GUID_HOLD_SECONDS
        stale = self._young(GUID_HOLD_SECONDS + 1)
        fresh = self._young(GUID_HOLD_SECONDS - 1)
        rows = [self._row(stale - 50, "g1"), self._row(stale, None, uuid="stuck"),
                self._row(stale + 10, "g3")]
        cursor, held = self._advance(0.0, rows)
        assert cursor == stale + 10
        assert held == set()
        # Just inside the window it still holds.
        rows = [self._row(fresh - 50, "g1"), self._row(fresh, None, uuid="uploading")]
        cursor, held = self._advance(0.0, rows)
        assert cursor == fresh - 50
        assert held == {"uploading"}

    @pytest.mark.unit
    def test_row_without_creation_date_neither_moves_nor_holds(self):
        rows = [self._row(self._young(100), "g1"), self._row(None, None, uuid="undated")]
        cursor, held = self._advance(0.0, rows)
        assert cursor == self._young(100)
        assert held == set()


# ---------------------------------------------------------------------------
# PhotosIntegration._assets_after — the fetch query
# ---------------------------------------------------------------------------

class TestPhotosFeedQuery:
    """Messages "Shared with You" syndication assets (ZSAVEDASSETTYPE = 12) appear in
    Photos.sqlite but never upload to the user's iCloud library, so they never get a
    ZCLOUDASSETGUID. If fetched, they'd hold the cursor for a day at a time (the feed
    holds it below any young GUID-less photo waiting for a GUID that will never arrive)."""

    @staticmethod
    def _make_photos_db(tmp_path):
        db_path = str(tmp_path / "Photos.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE ZASSET (Z_PK INTEGER, ZUUID TEXT, ZFILENAME TEXT, ZDIRECTORY TEXT, "
            "ZDATECREATED REAL, ZLATITUDE REAL, ZLONGITUDE REAL, ZKINDSUBTYPE INTEGER, "
            "ZWIDTH INTEGER, ZHEIGHT INTEGER, ZCLOUDASSETGUID TEXT, ZLIBRARYSCOPE INTEGER, "
            "ZKIND INTEGER, ZTRASHEDSTATE INTEGER, ZMASTER INTEGER, ZSAVEDASSETTYPE INTEGER, "
            "ZDURATION REAL, ZHIDDEN INTEGER)")
        conn.execute("CREATE TABLE ZCLOUDMASTER (Z_PK INTEGER, ZCLOUDMASTERGUID TEXT)")
        # Normal synced photo at t=100.
        conn.execute(
            "INSERT INTO ZASSET VALUES (1, 'uuid-normal', 'a.heic', '0/00', 100.0, NULL, NULL, "
            "0, 100, 100, 'asset-guid-1', 0, 0, 0, NULL, 6, NULL, 0)")
        # Syndication (Shared with You) photo at t=200 — GUID-less forever.
        conn.execute(
            "INSERT INTO ZASSET VALUES (2, 'uuid-synd', 'b.heic', '0/00', 200.0, NULL, NULL, "
            "2, 100, 100, NULL, NULL, 0, 0, NULL, 12, NULL, 0)")
        # Synced 12.5 s slo-mo video at t=50.
        conn.execute(
            "INSERT INTO ZASSET VALUES (3, 'uuid-video', 'c.mov', '0/00', 50.0, NULL, NULL, "
            "101, 1920, 1080, 'asset-guid-3', 0, 1, 0, NULL, 6, 12.5, 0)")
        # Hidden (Hidden album) photo at t=300 — never indexed.
        conn.execute(
            "INSERT INTO ZASSET VALUES (4, 'uuid-hidden', 'd.heic', '0/00', 300.0, NULL, NULL, "
            "0, 100, 100, 'asset-guid-4', 0, 0, 0, NULL, 6, NULL, 1)")
        conn.commit()
        conn.close()
        return db_path

    @staticmethod
    def _assets_after(db_path, cursor, limit=100):
        from integrations.photos.PhotosIntegration import PhotosIntegration
        with patch(f'{PI}.PHOTOS_SQLITE_PATH', db_path):
            return PhotosIntegration._assets_after(cursor, limit)

    @pytest.mark.unit
    def test_query_excludes_syndication_assets(self, tmp_path):
        db_path = self._make_photos_db(tmp_path)
        rows = self._assets_after(db_path, 0.0)
        assert [r["uuid"] for r in rows] == ["uuid-video", "uuid-normal"]

    @pytest.mark.unit
    def test_query_starts_strictly_after_the_cursor_and_honours_the_limit(self, tmp_path):
        db_path = self._make_photos_db(tmp_path)
        assert [r["uuid"] for r in self._assets_after(db_path, 50.0)] == ["uuid-normal"]
        assert [r["uuid"] for r in self._assets_after(db_path, 100.0)] == []
        assert [r["uuid"] for r in self._assets_after(db_path, 0.0, limit=1)] == ["uuid-video"]

    @pytest.mark.unit
    def test_query_rows_carry_kind_and_duration(self, tmp_path):
        db_path = self._make_photos_db(tmp_path)
        rows = {r["uuid"]: r for r in self._assets_after(db_path, 0.0)}
        assert rows["uuid-video"]["kind"] == "video"
        assert rows["uuid-video"]["duration"] == 12.5
        assert rows["uuid-video"]["kind_subtype"] == 101
        assert rows["uuid-normal"]["kind"] == "photo"
        assert rows["uuid-normal"]["duration"] is None

    @pytest.mark.unit
    def test_query_is_stills_only_when_videos_disabled(self, tmp_path):
        from integrations.photos.PhotoProcessor import VideoSettings
        db_path = self._make_photos_db(tmp_path)
        disabled = VideoSettings(enabled=False, max_bytes=1, max_duration_seconds=1)
        with patch('integrations.photos.PhotoProcessor.video_settings', return_value=disabled):
            rows = self._assets_after(db_path, 0.0)
        assert [r["uuid"] for r in rows] == ["uuid-normal"]

    @pytest.mark.unit
    def test_remaining_count_ignores_syndication_assets_and_starts_after_the_cursor(self, tmp_path):
        """`remaining()` is this count: it must agree with what fetch will read, or the
        dashboard's progress never reaches 100 %."""
        from integrations.photos.PhotoProcessor import PhotoProcessor
        db_path = self._make_photos_db(tmp_path)
        conn = sqlite3.connect(db_path)
        # A syndication asset that somehow got a GUID: still not ours to index.
        conn.execute(
            "INSERT INTO ZASSET VALUES (5, 'uuid-synd-guid', 'e.heic', '0/00', 400.0, NULL, NULL, "
            "0, 100, 100, 'asset-guid-5', 0, 0, 0, NULL, 12, NULL, 0)")
        conn.commit()
        conn.close()
        with patch('integrations.photos.PhotoProcessor.PHOTOS_SQLITE_PATH', db_path), \
             patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db:
            mock_db.return_value.get_indexed_photo_guids.return_value = set()
            processor = PhotoProcessor()
            # video (t=50) and normal (t=100); hidden (t=300) and both syndication rows are out
            assert processor.count_unindexed_photos(after_apple_ts=0.0) == 2
            assert processor.count_unindexed_photos(after_apple_ts=50.0) == 1
            assert processor.count_unindexed_photos(after_apple_ts=100.0) == 0
            mock_db.return_value.get_indexed_photo_guids.return_value = {"asset-guid-1"}
            assert processor.count_unindexed_photos(after_apple_ts=0.0) == 1


# ---------------------------------------------------------------------------
# PhotoProcessor._find_unprocessed — real query coverage
# ---------------------------------------------------------------------------

class TestPhotoProcessorFindUnprocessed:
    """Every other test mocks _find_unprocessed away, so its three hand-written SQL
    variants (default / Favorites / named album) run against a real fake Photos.sqlite
    here. Syndication rows get a synthetic GUID on purpose: exclusion must come from
    the SQL's ZSAVEDASSETTYPE filter, not from the GUID-less deferral in the row loop."""

    @staticmethod
    def _make_photos_db(tmp_path):
        db_path = str(tmp_path / "Photos.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE ZASSET (Z_PK INTEGER, ZUUID TEXT, ZFILENAME TEXT, ZDIRECTORY TEXT, "
            "ZDATECREATED REAL, ZLATITUDE REAL, ZLONGITUDE REAL, ZKINDSUBTYPE INTEGER, "
            "ZCLOUDASSETGUID TEXT, ZLIBRARYSCOPE INTEGER, ZKIND INTEGER, ZTRASHEDSTATE INTEGER, "
            "ZMASTER INTEGER, ZSAVEDASSETTYPE INTEGER, ZFAVORITE INTEGER, ZDURATION REAL, "
            "ZHIDDEN INTEGER)")
        conn.execute("CREATE TABLE ZCLOUDMASTER (Z_PK INTEGER, ZCLOUDMASTERGUID TEXT)")
        conn.execute("CREATE TABLE Z_33ASSETS (Z_3ASSETS INTEGER, Z_33ALBUMS INTEGER)")
        conn.execute("CREATE TABLE ZGENERICALBUM (Z_PK INTEGER, ZTITLE TEXT)")
        return conn, db_path

    @staticmethod
    def _insert_asset(conn, pk, uuid, ts, guid, saved_type=6, favorite=0, kind=0, duration=None, hidden=0):
        conn.execute(
            "INSERT INTO ZASSET VALUES (?, ?, 'IMG.heic', '0/00', ?, NULL, NULL, 0, ?, "
            "NULL, ?, 0, NULL, ?, ?, ?, ?)",
            (pk, uuid, ts, guid, kind, saved_type, favorite, duration, hidden))

    def _find(self, db_path, indexed_guids, album=None, limit=10):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        with patch('integrations.photos.PhotoProcessor.PHOTOS_SQLITE_PATH', db_path), \
             patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db:
            mock_db.return_value.get_indexed_photo_guids.return_value = indexed_guids
            return PhotoProcessor()._find_unprocessed(album=album, limit=limit)

    @pytest.mark.unit
    def test_default_variant_skips_indexed_deferred_and_syndication(self, tmp_path):
        conn, db_path = self._make_photos_db(tmp_path)
        self._insert_asset(conn, 1, 'uuid-indexed', 100.0, 'G-INDEXED')
        self._insert_asset(conn, 2, 'uuid-new', 200.0, 'G-NEW')
        self._insert_asset(conn, 3, 'uuid-pending', 300.0, None)
        self._insert_asset(conn, 4, 'uuid-synd', 400.0, 'G-SYND', saved_type=12)
        conn.commit()
        conn.close()

        rows, err = self._find(db_path, {'G-INDEXED'})

        assert err == ""
        assert [r["uuid"] for r in rows] == ['uuid-new']
        # 978307200 is the Apple-epoch offset applied to ZDATECREATED.
        assert rows[0]["creation_ts"] == 200.0 + 978307200
        assert rows[0]["cloud_asset_guid"] == 'G-NEW'

    @pytest.mark.unit
    def test_hidden_assets_are_never_backlog(self, tmp_path):
        conn, db_path = self._make_photos_db(tmp_path)
        self._insert_asset(conn, 1, 'uuid-hidden', 100.0, 'G-HIDDEN', hidden=1)
        self._insert_asset(conn, 2, 'uuid-hidden-fav', 200.0, 'G-HIDDEN-FAV', hidden=1, favorite=1)
        self._insert_asset(conn, 3, 'uuid-visible', 300.0, 'G-VISIBLE')
        conn.commit()
        conn.close()

        rows, _ = self._find(db_path, set())
        fav_rows, _ = self._find(db_path, set(), album="Favorites")

        assert [r["uuid"] for r in rows] == ['uuid-visible']
        assert fav_rows == []

    @pytest.mark.unit
    def test_remove_hidden_assets_drops_indexed_hidden_rows_only(self, tmp_path):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        conn, db_path = self._make_photos_db(tmp_path)
        self._insert_asset(conn, 1, 'uuid-hidden-indexed', 100.0, 'G-HIDDEN-INDEXED', hidden=1)
        self._insert_asset(conn, 2, 'uuid-hidden-new', 200.0, 'G-HIDDEN-NEW', hidden=1)
        self._insert_asset(conn, 3, 'uuid-visible', 300.0, 'G-VISIBLE')
        conn.commit()
        conn.close()

        with patch('integrations.photos.PhotoProcessor.PHOTOS_SQLITE_PATH', db_path), \
             patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db:
            mock_db.return_value.get_indexed_photo_guids.return_value = {'G-HIDDEN-INDEXED', 'G-VISIBLE'}
            mock_db.return_value.delete_photos.return_value = 1
            removed = PhotoProcessor().remove_hidden_assets()

        assert removed == 1
        mock_db.return_value.delete_photos.assert_called_once_with({'G-HIDDEN-INDEXED'})

    @pytest.mark.unit
    def test_remove_hidden_assets_no_indexed_hidden_is_noop(self, tmp_path):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        conn, db_path = self._make_photos_db(tmp_path)
        self._insert_asset(conn, 1, 'uuid-hidden', 100.0, 'G-HIDDEN', hidden=1)
        conn.commit()
        conn.close()

        with patch('integrations.photos.PhotoProcessor.PHOTOS_SQLITE_PATH', db_path), \
             patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db:
            mock_db.return_value.get_indexed_photo_guids.return_value = {'G-VISIBLE'}
            removed = PhotoProcessor().remove_hidden_assets()

        assert removed == 0
        mock_db.return_value.delete_photos.assert_not_called()

    @pytest.mark.unit
    def test_only_syndication_assets_means_nothing_awaiting_sync(self, tmp_path):
        """The prod bug: GUID-less syndication assets were perpetually counted as
        'awaiting sync'. Post-fix the SQL drops them, so nothing is reported deferred
        and the result is a clean 'no unprocessed photos'."""
        conn, db_path = self._make_photos_db(tmp_path)
        self._insert_asset(conn, 1, 'uuid-synd', 100.0, None, saved_type=12)
        conn.commit()
        conn.close()

        with patch('integrations.photos.PhotoProcessor.info') as mock_info:
            rows, err = self._find(db_path, set())

        assert rows == []
        assert err == "No unprocessed photos found"
        deferred_logs = [c for c in mock_info.call_args_list if 'deferred' in str(c.args[0])]
        assert deferred_logs == []

    @pytest.mark.unit
    def test_favorites_variant_excludes_syndication(self, tmp_path):
        conn, db_path = self._make_photos_db(tmp_path)
        self._insert_asset(conn, 1, 'uuid-fav', 100.0, 'G-FAV', favorite=1)
        self._insert_asset(conn, 2, 'uuid-notfav', 200.0, 'G-NOTFAV')
        self._insert_asset(conn, 3, 'uuid-fav-synd', 300.0, 'G-FAV-SYND', saved_type=12, favorite=1)
        conn.commit()
        conn.close()

        rows, _ = self._find(db_path, set(), album="Favorites")

        assert [r["uuid"] for r in rows] == ['uuid-fav']

    @pytest.mark.unit
    def test_album_variant_excludes_syndication(self, tmp_path):
        conn, db_path = self._make_photos_db(tmp_path)
        self._insert_asset(conn, 1, 'uuid-in-album', 100.0, 'G-ALBUM')
        self._insert_asset(conn, 2, 'uuid-elsewhere', 200.0, 'G-ELSEWHERE')
        self._insert_asset(conn, 3, 'uuid-album-synd', 300.0, 'G-ALBUM-SYND', saved_type=12)
        conn.execute("INSERT INTO ZGENERICALBUM VALUES (50, 'Trip')")
        conn.execute("INSERT INTO Z_33ASSETS VALUES (1, 50)")
        conn.execute("INSERT INTO Z_33ASSETS VALUES (3, 50)")
        conn.commit()
        conn.close()

        rows, _ = self._find(db_path, set(), album="Trip")

        assert [r["uuid"] for r in rows] == ['uuid-in-album']

    @pytest.mark.unit
    def test_videos_included_with_kind_and_duration(self, tmp_path):
        conn, db_path = self._make_photos_db(tmp_path)
        self._insert_asset(conn, 1, 'uuid-photo', 100.0, 'G-PHOTO')
        self._insert_asset(conn, 2, 'uuid-video', 200.0, 'G-VIDEO', kind=1, duration=42.0)
        conn.commit()
        conn.close()

        rows, _ = self._find(db_path, set())

        assert [(r["uuid"], r["kind"], r["duration"]) for r in rows] == \
            [('uuid-video', 'video', 42.0), ('uuid-photo', 'photo', None)]

    @pytest.mark.unit
    def test_videos_excluded_when_disabled(self, tmp_path):
        from integrations.photos.PhotoProcessor import VideoSettings
        conn, db_path = self._make_photos_db(tmp_path)
        self._insert_asset(conn, 1, 'uuid-photo', 100.0, 'G-PHOTO')
        self._insert_asset(conn, 2, 'uuid-video', 200.0, 'G-VIDEO', kind=1, duration=42.0)
        conn.commit()
        conn.close()
        disabled = VideoSettings(enabled=False, max_bytes=1, max_duration_seconds=1)

        with patch('integrations.photos.PhotoProcessor.video_settings', return_value=disabled):
            rows, _ = self._find(db_path, set())

        assert [r["uuid"] for r in rows] == ['uuid-photo']

    @pytest.mark.unit
    def test_count_unindexed_by_kind(self, tmp_path):
        from integrations.photos.PhotoProcessor import PhotoProcessor, KIND_PHOTO, KIND_VIDEO
        conn, db_path = self._make_photos_db(tmp_path)
        self._insert_asset(conn, 1, 'uuid-photo', 100.0, 'G-PHOTO')
        self._insert_asset(conn, 2, 'uuid-photo-done', 150.0, 'G-PHOTO-DONE')
        self._insert_asset(conn, 3, 'uuid-video', 200.0, 'G-VIDEO', kind=1, duration=42.0)
        self._insert_asset(conn, 4, 'uuid-video-done', 250.0, 'G-VIDEO-DONE', kind=1, duration=3.0)
        conn.commit()
        conn.close()

        with patch('integrations.photos.PhotoProcessor.PHOTOS_SQLITE_PATH', db_path), \
             patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db:
            mock_db.return_value.get_indexed_photo_guids.return_value = {'G-PHOTO-DONE', 'G-VIDEO-DONE'}
            processor = PhotoProcessor()
            assert processor.count_unindexed_photos() == 2
            assert processor.count_unindexed_photos(kinds=(KIND_PHOTO,)) == 1
            assert processor.count_unindexed_photos(kinds=(KIND_VIDEO,)) == 1


# ---------------------------------------------------------------------------
# Videos — database column
# ---------------------------------------------------------------------------

class TestPhotosDatabaseVideos:

    @pytest.mark.unit
    def test_duration_round_trips(self, floxy_db):
        floxy_db.store_photo(
            asset_guid="VID-1", creation_date=1700000000.0, latitude=None, longitude=None,
            location_text="Home", albums="", persons="Frances", photo_type="video, slo-mo",
            short_description="Frances on the trampoline", detailed_description="Bouncing in the garden",
            pretty_score=70, family_importance_score=80, indexed_at=time.time(), duration_seconds=12.5)
        row = floxy_db.search_photos("trampoline")[0]
        assert row["duration_seconds"] == 12.5
        assert row["photo_type"] == "video, slo-mo"

    @pytest.mark.unit
    def test_migration_adds_duration_to_old_schema(self, tmp_path):
        db_path = str(tmp_path / "old.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""CREATE TABLE photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT, asset_guid TEXT NOT NULL UNIQUE,
            creation_date REAL NOT NULL, latitude REAL, longitude REAL, location_text TEXT,
            albums TEXT, persons TEXT, photo_type TEXT, short_description TEXT NOT NULL,
            detailed_description TEXT NOT NULL, pretty_score INTEGER NOT NULL,
            family_importance_score INTEGER NOT NULL, indexed_at REAL NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP)""")
        conn.commit()
        conn.close()

        db = FloxyDatabase.__new__(FloxyDatabase)
        db.db_path = db_path
        db.lock = threading.Lock()
        db._init_database()

        conn = sqlite3.connect(db_path)
        columns = [c[1] for c in conn.execute("PRAGMA table_info(photos)").fetchall()]
        conn.close()
        assert "duration_seconds" in columns


# ---------------------------------------------------------------------------
# Videos — PhotoProcessor
# ---------------------------------------------------------------------------

class _FakeVideoLLM:
    def __init__(self, responses, supports=True):
        self.responses = list(responses)
        self.supports = supports
        self.calls = []
        self.fps_calls = []

    def supports_video(self):
        return self.supports

    def name(self):
        return "FakeVideoLLM"

    def think_with_video(self, system_message, user_message, video_bytes, mime_type, fps=None):
        self.calls.append((system_message + user_message, video_bytes, mime_type))
        self.fps_calls.append(fps)
        return self.responses.pop(0)


VIDEO_RESULT = {"short_description": "Frances sings", "detailed_description": "Frances sings in the kitchen",
                "spoken_content": "Happy birthday to you", "pretty_score": 60, "family_importance_score": 90}


class TestPhotoProcessorVideo:

    @pytest.fixture(autouse=True)
    def clean_processing_set(self):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        PhotoProcessor._processing_uuids.clear()
        yield
        PhotoProcessor._processing_uuids.clear()

    @staticmethod
    def _row(duration=12.5, subtype=101):
        return {"uuid": "VID-UUID", "filename": "clip.mov", "directory": "0/00",
                "creation_ts": 1700000000.0, "latitude": 57.7, "longitude": 11.9,
                "kind_subtype": subtype, "kind": "video", "duration": duration,
                "cloud_master_guid": "MASTER-1", "cloud_asset_guid": "VID-GUID", "library_scope": 0}

    def _run(self, row, llm, download=None, download_error=None, store_photo_return=7):
        from integrations.photos.PhotoProcessor import PhotoProcessor, VideoSettings
        processor = PhotoProcessor()
        processor._llm = llm
        settings = VideoSettings(enabled=True, max_bytes=20 * 1024 * 1024, max_duration_seconds=300)
        get_bytes = Mock(return_value=download, side_effect=download_error)

        with patch('integrations.photos.PhotoProcessor.video_settings', return_value=settings), \
             patch.object(PhotoProcessor, '_get_video_bytes', get_bytes), \
             patch.object(PhotoProcessor, '_get_albums', return_value="Birthdays"), \
             patch.object(PhotoProcessor, '_get_persons', return_value="Frances"), \
             patch.object(PhotoProcessor, '_get_location', return_value="Home"), \
             patch.object(PhotoProcessor, '_read_memory', return_value=""), \
             patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db:
            mock_db.return_value.photo_exists.return_value = False
            mock_db.return_value.store_photo.return_value = store_photo_return
            needs_retry = processor._process_one(row)
        return needs_retry, mock_db.return_value, get_bytes

    @pytest.mark.unit
    def test_indexes_video_with_type_duration_and_spoken_history(self):
        llm = _FakeVideoLLM([json.dumps(VIDEO_RESULT)])
        needs_retry, db, get_bytes = self._run(self._row(), llm, download=(b"mp4bytes", "video/mp4"))

        assert needs_retry is False
        get_bytes.assert_called_once_with(self._row(), 20 * 1024 * 1024)
        kwargs = db.store_photo.call_args.kwargs
        assert kwargs["photo_type"] == "video, slo-mo"
        assert kwargs["duration_seconds"] == 12.5
        assert kwargs["short_description"] == "Frances sings"
        assert kwargs["family_importance_score"] == 90
        content = db.add_history_entry.call_args[0][1]
        assert content.startswith("New video taken: Frances sings (Frances, Home, 57.70000,11.90000, video, slo-mo, 12s) [photo_id=7]")
        assert content.endswith("Spoken: Happy birthday to you")
        system, video_bytes, mime = llm.calls[0]
        assert (video_bytes, mime) == (b"mp4bytes", "video/mp4")
        assert "Duration: 12 seconds" in system and "Video type: video, slo-mo" in system
        assert "Frances" in system and "Birthdays" in system

    @pytest.mark.unit
    def test_normal_length_clip_uses_gemini_default_sampling(self):
        llm = _FakeVideoLLM([json.dumps(VIDEO_RESULT)])
        self._run(self._row(duration=12.5), llm, download=(b"mp4bytes", "video/mp4"))
        assert llm.fps_calls == [None]

    @pytest.mark.unit
    def test_sub_second_time_lapse_gets_fps_override(self):
        llm = _FakeVideoLLM([json.dumps(VIDEO_RESULT)])
        needs_retry, db, _ = self._run(self._row(duration=0.1, subtype=102), llm, download=(b"mov", "video/quicktime"))

        assert needs_retry is False
        assert llm.fps_calls == [20.0]
        assert db.store_photo.call_args.kwargs["photo_type"] == "video, time-lapse"

    @pytest.mark.unit
    def test_long_spoken_content_is_truncated_in_history(self):
        from integrations.photos.PhotoProcessor import MAX_SPOKEN_CONTENT_CHARS
        speech = "la " * 1000
        result = {**VIDEO_RESULT, "spoken_content": speech}
        llm = _FakeVideoLLM([json.dumps(result)])
        _, db, _ = self._run(self._row(), llm, download=(b"mp4bytes", "video/mp4"))

        content = db.add_history_entry.call_args[0][1]
        spoken = content.split("Spoken: ", 1)[1]
        assert spoken.endswith("…") and len(spoken) <= MAX_SPOKEN_CONTENT_CHARS + 1
        assert len(speech) > MAX_SPOKEN_CONTENT_CHARS

    @pytest.mark.unit
    def test_history_without_spoken_line_when_silent(self):
        result = {**VIDEO_RESULT, "spoken_content": ""}
        llm = _FakeVideoLLM([json.dumps(result)])
        _, db, _ = self._run(self._row(subtype=0), llm, download=(b"v", "video/mp4"))
        content = db.add_history_entry.call_args[0][1]
        assert "Spoken:" not in content
        assert "(Frances, Home, 57.70000,11.90000, 12s)" in content
        assert db.store_photo.call_args.kwargs["photo_type"] == "video"

    @pytest.mark.unit
    def test_too_long_is_stored_skipped_without_download(self):
        llm = _FakeVideoLLM([])
        needs_retry, db, get_bytes = self._run(self._row(duration=400), llm)
        assert needs_retry is False
        get_bytes.assert_not_called()
        kwargs = db.store_photo.call_args.kwargs
        assert kwargs["short_description"] == "Skipped (video too long)"
        assert kwargs["photo_type"] == "skipped"
        assert kwargs["duration_seconds"] == 400

    @pytest.mark.unit
    def test_llm_without_video_input_leaves_clip_unindexed(self):
        llm = _FakeVideoLLM([], supports=False)
        needs_retry, db, get_bytes = self._run(self._row(), llm)
        assert needs_retry is False
        get_bytes.assert_not_called()
        db.store_photo.assert_not_called()

    @pytest.mark.unit
    def test_too_large_is_stored_skipped(self):
        from integrations.photos.ICloudPhotoDownloader import VideoTooLargeError
        llm = _FakeVideoLLM([])
        needs_retry, db, _ = self._run(self._row(), llm, download_error=VideoTooLargeError(41_000_000, 20_000_000))
        assert needs_retry is False
        assert db.store_photo.call_args.kwargs["short_description"] == "Skipped (video too large)"
        assert llm.calls == []

    @pytest.mark.unit
    def test_not_in_cloud_is_stored_skipped(self):
        from integrations.photos.ICloudPhotoDownloader import PhotoNotInCloudError
        llm = _FakeVideoLLM([])
        needs_retry, db, _ = self._run(self._row(), llm, download_error=PhotoNotInCloudError("gone"))
        assert needs_retry is False
        assert db.store_photo.call_args.kwargs["short_description"] == "Skipped (not in iCloud library)"

    @pytest.mark.unit
    def test_no_rendition_is_skipped_permanently_not_retried(self):
        from integrations.photos.ICloudPhotoDownloader import NoVideoRenditionError
        llm = _FakeVideoLLM([])
        needs_retry, db, _ = self._run(self._row(duration=0.002, subtype=0),
                                       llm, download_error=NoVideoRenditionError(["MASTER-1", "VID-GUID"]))
        assert needs_retry is False
        assert llm.calls == []
        assert db.store_photo.call_args.kwargs["short_description"] == "Skipped (no video rendition in iCloud)"

    @pytest.mark.unit
    def test_video_download_passes_the_asset_guid_for_the_rendition_fallback(self):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        row = {**self._row(), "cloud_master_guid": "MASTER-1", "cloud_asset_guid": "VID-GUID", "library_scope": 15}
        with patch.object(PhotoProcessor, '_icloud_placeholder_type', return_value="evicted"), \
             patch('integrations.photos.ICloudPhotoDownloader.ICloudPhotoDownloader.get_instance') as dl:
            dl.return_value.download_video_bytes.return_value = (b"v", "video/mp4")
            assert PhotoProcessor._get_video_bytes(row, 1000) == (b"v", "video/mp4")
        dl.return_value.download_video_bytes.assert_called_once_with("MASTER-1", 15, max_bytes=1000, cloud_asset_guid="VID-GUID")

    @pytest.mark.unit
    def test_download_failure_requests_retry(self):
        llm = _FakeVideoLLM([])
        needs_retry, db, _ = self._run(self._row(), llm, download=None)
        assert needs_retry is True
        db.store_photo.assert_not_called()

    @pytest.mark.unit
    def test_llm_failure_is_stored_skipped(self):
        llm = _FakeVideoLLM(["not json", "still not json"])
        needs_retry, db, _ = self._run(self._row(), llm, download=(b"v", "video/mp4"))
        assert needs_retry is False
        assert db.store_photo.call_args.kwargs["short_description"] == "Skipped (LLM analysis failed)"
        assert len(llm.calls) == 2

    @pytest.mark.unit
    def test_parse_retry_then_success(self):
        llm = _FakeVideoLLM(["garbage", json.dumps(VIDEO_RESULT)])
        _, db, _ = self._run(self._row(), llm, download=(b"v", "video/mp4"))
        assert db.store_photo.call_args.kwargs["short_description"] == "Frances sings"

    @pytest.mark.unit
    def test_lost_insert_race_writes_no_history(self):
        llm = _FakeVideoLLM([json.dumps(VIDEO_RESULT)])
        _, db, _ = self._run(self._row(), llm, download=(b"v", "video/mp4"), store_photo_return=None)
        db.add_history_entry.assert_not_called()


class TestPhotoProcessorDescribeVideoSubtype:

    @pytest.mark.unit
    @pytest.mark.parametrize("subtype, expected", [
        (0, "video"), (None, "video"), (100, "video"),
        (101, "video, slo-mo"), (102, "video, time-lapse"), (103, "video, cinematic"),
    ])
    def test_labels(self, subtype, expected):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        assert PhotoProcessor._describe_video_subtype(subtype) == expected


class TestPhotoProcessorGetVideoBytes:

    @staticmethod
    def _row(tmp_path, name="clip.mov"):
        return {"uuid": "VID", "filename": name, "directory": "0/00",
                "cloud_master_guid": "MASTER-1", "cloud_asset_guid": "ASSET-1", "library_scope": 3}

    @pytest.mark.unit
    def test_local_original_used_when_it_fits(self, tmp_path):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        (tmp_path / "0/00").mkdir(parents=True)
        clip = b"small clip" * 200
        (tmp_path / "0/00/clip.mov").write_bytes(clip)

        with patch('integrations.photos.PhotoProcessor.PHOTOS_ORIGINALS_DIR', str(tmp_path)), \
             patch('integrations.photos.ICloudPhotoDownloader.ICloudPhotoDownloader.get_instance') as dl:
            result = PhotoProcessor._get_video_bytes(self._row(tmp_path), max_bytes=len(clip) + 1)

        assert result == (clip, "video/quicktime")
        dl.assert_not_called()

    @pytest.mark.unit
    def test_large_local_original_falls_back_to_icloud_rendition(self, tmp_path):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        (tmp_path / "0/00").mkdir(parents=True)
        (tmp_path / "0/00/clip.mov").write_bytes(b"x" * 5000)

        with patch('integrations.photos.PhotoProcessor.PHOTOS_ORIGINALS_DIR', str(tmp_path)), \
             patch('integrations.photos.ICloudPhotoDownloader.ICloudPhotoDownloader.get_instance') as dl:
            dl.return_value.download_video_bytes.return_value = (b"rendition", "video/mp4")
            result = PhotoProcessor._get_video_bytes(self._row(tmp_path), max_bytes=100)

        assert result == (b"rendition", "video/mp4")
        dl.return_value.download_video_bytes.assert_called_once_with("MASTER-1", 3, max_bytes=100, cloud_asset_guid="ASSET-1")

    @pytest.mark.unit
    def test_evicted_original_goes_to_icloud_without_brctl(self, tmp_path):
        from integrations.photos.PhotoProcessor import PhotoProcessor

        with patch('integrations.photos.PhotoProcessor.PHOTOS_ORIGINALS_DIR', str(tmp_path)), \
             patch('integrations.photos.PhotoProcessor.subprocess.run') as run, \
             patch('integrations.photos.ICloudPhotoDownloader.ICloudPhotoDownloader.get_instance') as dl:
            dl.return_value.download_video_bytes.return_value = None
            result = PhotoProcessor._get_video_bytes(self._row(tmp_path), max_bytes=100)

        assert result is None
        run.assert_not_called()

class TestPhotoProcessorBulkVideoThrottle:

    @pytest.mark.unit
    def test_videos_and_photos_use_separate_semaphores(self):
        """Videos get their own, smaller fan-out so a video-heavy batch never occupies
        the 30 stills threads with 10 MB downloads and 20 s LLM calls."""
        from integrations.photos import PhotoProcessor as module
        rows = [{"uuid": f"V{i}", "filename": "c.mov", "kind": "video", "cloud_asset_guid": f"G{i}"} for i in range(8)]
        in_flight = []
        peak = [0]
        lock = threading.Lock()
        both_slots_taken = threading.Barrier(2)

        def fake_process(row):
            with lock:
                in_flight.append(row["uuid"])
                peak[0] = max(peak[0], len(in_flight))
            both_slots_taken.wait(5)    # each clip stays in flight until a second one is, too
            with lock:
                in_flight.remove(row["uuid"])
            return False

        processor = module.PhotoProcessor()
        with patch.object(module, 'VIDEO_THREADS', 2), \
             patch.object(processor, '_find_unprocessed', return_value=(rows, "")), \
             patch.object(processor, '_process_one', side_effect=fake_process), \
             patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db:
            mock_db.return_value.photo_exists.return_value = True
            mock_db.return_value.get_photo_description.return_value = "desc"
            status, _ = processor.process_bulk_unprocessed(count=8)

        assert status == "ok"
        assert peak[0] == 2


# ---------------------------------------------------------------------------
# PhotoProcessor — one indexing LLM, always Gemini 3.8 Flash, vault-wrapped
# ---------------------------------------------------------------------------

class TestPhotoProcessorIndexingLlm:
    """Backfill, new photos and videos all go to the `photo_indexing` tier — Gemini 3.8 Flash
    unless config says otherwise — whatever the backlog size. The prompt carries all of
    memory.md, so the LLM must come from create_llm(): the VaultLLM wrapper is what keeps the
    vaulted values out of the provider."""

    @staticmethod
    def _fake_backends(*names):
        """create_llm() imports the backend module by class name; a bare module per name stands
        in so the test never loads a provider SDK (google.genai alone is ~0.3 s). Returns the
        context manager and the fake class per name."""
        classes = {name: Mock(name=name) for name in names}
        modules = {}
        for name, cls in classes.items():
            module = types.ModuleType(f"assistant.llm.{name}")
            setattr(module, name, cls)
            modules[module.__name__] = module
        return patch.dict(sys.modules, modules), classes

    @pytest.mark.unit
    @pytest.mark.parametrize("unindexed", [10, 500])
    def test_indexing_llm_defaults_to_gemini_flash_and_is_vault_wrapped(self, unindexed):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        from assistant.llm.VaultLLM import VaultLLM
        from common.ConfigManager import config
        real_get = config.get

        def no_ai_tiers(key, default=None):
            if key.startswith('ai.'):
                raise KeyError(key)
            return real_get(key, default)

        backend = Mock()
        fakes, classes = self._fake_backends("GeminiLLM", "ClaudeLLM", "GrokLLM")
        classes["GeminiLLM"].return_value = backend
        with fakes, \
             patch.object(PhotoProcessor, 'count_unindexed_photos', return_value=unindexed), \
             patch('assistant.llm.config.get', side_effect=no_ai_tiers):
            processor = PhotoProcessor()
            llm = processor._get_llm()
            assert processor._get_llm() is llm
        assert isinstance(llm, VaultLLM) and llm.llm is backend
        classes["GeminiLLM"].assert_called_once_with(model="gemini-3.8-flash")
        backend.initialize.assert_called_once()
        classes["ClaudeLLM"].assert_not_called()
        classes["GrokLLM"].assert_not_called()

    @pytest.mark.unit
    def test_a_configured_photo_indexing_tier_is_honoured(self):
        """`ai.photo_indexing` steers the indexer like any other tier; PhotoProcessor binds
        parse_llm_config by name, so the patch has to land on its own module."""
        from integrations.photos.PhotoProcessor import PhotoProcessor
        backend = Mock()
        fakes, classes = self._fake_backends("ClaudeLLM", "GeminiLLM")
        classes["ClaudeLLM"].return_value = backend
        with fakes, \
             patch('integrations.photos.PhotoProcessor.parse_llm_config',
                   return_value=("ClaudeLLM", "claude-opus-5")) as tier:
            llm = PhotoProcessor()._get_llm()
        tier.assert_called_once_with('photo_indexing')
        classes["ClaudeLLM"].assert_called_once_with(model="claude-opus-5")
        classes["GeminiLLM"].assert_not_called()
        assert llm.llm is backend


@pytest.mark.unit
class TestLocalBackendIndexing:
    def test_a_preempted_image_is_resent_not_marked_failed(self):
        from assistant.llm.AbstractLLM import LLMPreempted
        from integrations.photos.PhotoProcessor import PhotoProcessor
        llm = Mock()
        llm.think.side_effect = [LLMPreempted("evicted"), LLMPreempted("evicted again"),
                                 '{"short_description":"a","detailed_description":"b","pretty_score":1,"family_importance_score":2}']
        llm.background_wait_seconds.side_effect = [4.0, 6.0, 2.5]
        processor = PhotoProcessor()
        processor._llm = llm
        PhotoProcessor._worker.queued_s = 0.0
        with patch.object(PhotoProcessor, '_read_memory', return_value=""):
            result = processor._analyze_with_llm(b"jpeg", "2026-01-01", "", "", "", "photo")
        assert result["pretty_score"] == 1
        assert llm.think.call_count == 3
        assert all(call.kwargs["background"] is True for call in llm.think.call_args_list)
        assert PhotoProcessor._worker.queued_s == 12.5, "the photo's queue time is the sum over every resend"

    @pytest.mark.parametrize("local, expected", [(True, 2), (False, 30)])
    def test_worker_count_follows_the_backend(self, local, expected):
        from integrations.photos import PhotoProcessor as module
        processor = module.PhotoProcessor()
        processor._llm = Mock(is_local=Mock(return_value=local), unavailable=Mock(return_value=None))
        with patch.object(module.threading, 'Semaphore', wraps=threading.Semaphore) as semaphore:
            processor.process_rows([])
        assert semaphore.call_args_list[0].args == (expected,)


@pytest.mark.unit
class TestProviderRefusal:
    """A 429 or 402 is the provider's state, not the photo's: nothing is stored as skipped, the
    row stays in the queue, and the feed holds its cursor so the row comes round again."""

    @staticmethod
    def _row():
        return {"uuid": "U1", "filename": "p.heic", "kind": "photo", "cloud_asset_guid": "G1", "creation_ts": 1.0,
                "latitude": None, "longitude": None, "kind_subtype": 0, "directory": "0", "duration": None}

    def test_a_still_the_provider_refused_is_not_stored_as_skipped(self):
        from assistant.llm.AbstractLLM import LLMUnavailable
        from integrations.photos import PhotoProcessor as module
        processor = module.PhotoProcessor()
        llm = Mock()
        llm.think.side_effect = LLMUnavailable("out of credits")
        llm.background_wait_seconds.return_value = 0.0
        processor._llm = llm
        prepared = module._PreparedPhoto(row=self._row(), jpeg=b"j", photo_type="photo", albums="", persons="",
                                         location_text="", user_message="u", started=0.0)
        with patch.object(processor, '_prepare_photo', return_value=(prepared, False)), \
             patch.object(module.PhotoProcessor, '_read_memory', return_value=""), \
             patch.object(processor, '_store_skipped') as skipped, \
             patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db:
            mock_db.return_value.photo_exists.return_value = False
            with pytest.raises(LLMUnavailable):
                processor._process_one(self._row())
        skipped.assert_not_called()
        mock_db.return_value.store_photo.assert_not_called()
        assert llm.think.call_count == 1, "no retry of a refusal the backend already knows about"
        assert not module.PhotoProcessor._processing_uuids

    def test_a_clip_the_provider_refused_is_not_stored_as_skipped(self):
        from assistant.llm.AbstractLLM import LLMUnavailable
        from integrations.photos import PhotoProcessor as module
        processor = module.PhotoProcessor()
        llm = Mock()
        llm.think_with_video.side_effect = LLMUnavailable("rate limited")
        processor._llm = llm
        with patch.object(module.PhotoProcessor, '_read_memory', return_value=""), \
             pytest.raises(LLMUnavailable):
            processor._analyze_video_with_llm(b"mp4", "video/mp4", "2026-01-01", "", "", "", "video", 5.0)

    def test_a_refused_backend_means_no_downloads_and_the_rows_stay(self):
        from integrations.photos import PhotoProcessor as module
        processor = module.PhotoProcessor()
        processor._llm = Mock(unavailable=Mock(return_value="out of credits"))
        with patch.object(processor, '_process_one') as process_one:
            succeeded, failed = processor.process_rows([self._row()], batched=True)
        process_one.assert_not_called()
        assert succeeded == [] and failed == ["p.heic"]
        assert processor.refused_last_run() == ["U1"]

    def test_a_row_refused_mid_batch_is_in_neither_verdict_and_the_next_run_forgets_it(self):
        """A parallel worker's success can clear the shared outage before the batch is over,
        so the feed cannot ask the provider afterwards — the batch itself must name the rows
        it never described, and only for the run that refused them."""
        from assistant.llm.AbstractLLM import LLMUnavailable
        from integrations.photos import PhotoProcessor as module
        processor = module.PhotoProcessor()
        processor._llm = Mock(is_local=Mock(return_value=False), unavailable=Mock(return_value=None))
        refused_row = dict(self._row(), uuid="U2", filename="q.heic", cloud_asset_guid="G2")

        def process_one(row):
            if row["uuid"] == "U2":
                raise LLMUnavailable("rate limited")
            return False

        with patch.object(processor, '_process_one', side_effect=process_one), \
             patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db:
            mock_db.return_value.photo_exists.return_value = True
            mock_db.return_value.get_photo_description.return_value = "a barn"
            succeeded, failed = processor.process_rows([self._row(), refused_row])
            first_run = processor.refused_last_run()
            processor.process_rows([self._row()])

        assert (succeeded, failed) == (["a barn"], [])
        assert first_run == ["U2"]
        assert processor._pending_icloud == {}
        assert processor.refused_last_run() == []

    def test_the_backfill_batch_switch_in_config_wins_over_the_module_default(self):
        """`feeds.photos.backfill_batch` is the config.yaml key; the module constant is only
        what an install without the key gets."""
        from integrations.photos import PhotoProcessor as module
        from common.ConfigManager import ConfigManager
        settings = {}
        instance = Mock()
        instance.get.side_effect = lambda key, default=None: settings.get(key, default)
        with patch.object(ConfigManager, 'get_instance', return_value=instance), \
             patch.object(module, 'BACKFILL_BATCH', True):
            assert module.PhotoProcessor._batch_enabled() is True
            settings['feeds.photos.backfill_batch'] = False
            assert module.PhotoProcessor._batch_enabled() is False


@pytest.mark.unit
class TestPhotosFeedSweep:
    """The live feed's sweep: the assets between the horizon and the cursor without an index
    row, indexed at the backfill's price with the cursor left where it is."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        from integrations.photos.PhotosIntegration import PhotosIntegration
        PhotosIntegration.reset_instance()
        yield
        PhotosIntegration.reset_instance()

    HORIZON = 1_600_000_000.0
    CURSOR = "750000000.0"

    def _sweep(self, counts, rows, reason=None, refused=()):
        integration, processor = TestPhotosFeed._opened()
        processor.count_unindexed_photos.side_effect = list(counts)
        processor.unindexed_between.return_value = rows
        processor.provider_unavailable.return_value = reason
        processor.provider_out_of_credits.return_value = False
        processor.refused_last_run.return_value = list(refused)
        with patch.object(integration, '_provider_refused') as refused_hook:
            batch = integration.sweep(self.HORIZON, self.CURSOR)
        return batch, processor, refused_hook

    def test_the_window_runs_from_the_horizon_to_the_cursor_in_apple_time(self):
        from integrations.photos.PhotoProcessor import APPLE_EPOCH_OFFSET
        from integrations.photos.PhotosIntegration import FETCH_BATCH_SIZE
        rows = [TestPhotosFeed._row("a", 600.0, "G-A")]
        batch, processor, _ = self._sweep([1, 0], rows)
        since, until = self.HORIZON - APPLE_EPOCH_OFFSET, float(self.CURSOR)
        assert processor.count_unindexed_photos.call_args_list[0].kwargs == {"after_apple_ts": since,
                                                                             "before_apple_ts": until}
        processor.unindexed_between.assert_called_once_with(since, until, FETCH_BATCH_SIZE)
        handed = processor.process_rows.call_args
        assert handed.args[0] == rows and handed.kwargs["batched"] is True, "a sweep is backfill work"
        assert batch.cursor == self.CURSOR

    def test_the_entries_the_processor_queued_come_back_and_the_count_is_the_leftover(self):
        from assistant.integration.Feed import FeedEntry
        integration, processor = TestPhotosFeed._opened()
        entries = [FeedEntry(1_600_000_500.0, "New photo taken: a barn")]
        processor.count_unindexed_photos.side_effect = [5, 4]
        processor.unindexed_between.return_value = [TestPhotosFeed._row("a", 600.0, "G-A")]
        processor.process_rows.side_effect = lambda rows, batched: [integration._queue_entry(e) for e in entries]
        batch = integration.sweep(self.HORIZON, self.CURSOR)
        assert batch.entries == entries
        assert batch.remaining == 4
        assert integration._drain_entries() == []

    def test_a_full_batch_with_progress_is_not_exhausted(self):
        from integrations.photos.PhotosIntegration import FETCH_BATCH_SIZE
        rows = [TestPhotosFeed._row(f"u{i}", float(i), f"G{i}") for i in range(FETCH_BATCH_SIZE)]
        batch, _, _ = self._sweep([300, 200], rows)
        assert batch.exhausted is False and batch.remaining == 200

    def test_a_short_batch_or_an_emptied_window_is_exhausted(self):
        rows = [TestPhotosFeed._row("a", 600.0, "G-A"), TestPhotosFeed._row("b", 700.0, "G-B")]
        short, _, _ = self._sweep([2, 0], rows)
        assert short.exhausted is True and short.remaining == 0

    def test_no_progress_means_an_exhausted_batch_so_the_same_rows_do_not_spin(self):
        rows = [TestPhotosFeed._row(f"u{i}", float(i), f"G{i}") for i in range(100)]
        batch, _, _ = self._sweep([300, 300], rows)
        assert batch.exhausted is True and batch.remaining == 300

    def test_nothing_behind_the_cursor_touches_no_asset(self):
        batch, processor, _ = self._sweep([0], [])
        assert (batch.exhausted, batch.remaining, batch.entries) == (True, 0, [])
        processor.unindexed_between.assert_not_called()
        processor.process_rows.assert_not_called()

    def test_a_provider_refusal_is_reported_and_leaves_the_whole_count_waiting(self):
        rows = [TestPhotosFeed._row("a", 600.0, "G-A")]
        batch, processor, refused_hook = self._sweep([40], rows, reason="prepaid credits are depleted (402)")
        refused_hook.assert_called_once()
        assert refused_hook.call_args.args[2] == self.CURSOR
        assert batch.exhausted is True and batch.remaining == 40
        assert processor.count_unindexed_photos.call_count == 1, "no progress check after a refusal"

    def test_rows_refused_mid_batch_count_as_a_refusal_too(self):
        rows = [TestPhotosFeed._row("a", 600.0, "G-A")]
        batch, _, refused_hook = self._sweep([40], rows, reason=None, refused=["a"])
        refused_hook.assert_called_once()
        assert batch.exhausted is True and batch.remaining == 40


@pytest.mark.unit
class TestPhotoProcessorUnindexedBetween:
    """`unindexed_between` and the bounded `count_unindexed_photos` agree on the stretch behind
    the feed's cursor: the GUID-bearing assets inside (after, before] without an index row."""

    def _library(self, tmp_path):
        conn, db_path = TestPhotoProcessorFindUnprocessed._make_photos_db(tmp_path)
        insert = TestPhotoProcessorFindUnprocessed._insert_asset
        insert(conn, 1, 'uuid-before', 100.0, 'G-BEFORE')          # at the exclusive start
        insert(conn, 2, 'uuid-indexed', 200.0, 'G-INDEXED')
        insert(conn, 3, 'uuid-hole-late', 400.0, 'G-HOLE-LATE')
        insert(conn, 4, 'uuid-hole-early', 300.0, 'G-HOLE-EARLY')
        insert(conn, 5, 'uuid-pending', 350.0, None)
        insert(conn, 6, 'uuid-synd', 360.0, 'G-SYND', saved_type=12)
        insert(conn, 7, 'uuid-hidden', 370.0, 'G-HIDDEN', hidden=1)
        insert(conn, 8, 'uuid-edge', 500.0, 'G-EDGE')              # at the inclusive end
        insert(conn, 9, 'uuid-after', 600.0, 'G-AFTER')
        conn.commit()
        conn.close()
        return db_path

    def _with_library(self, db_path):
        return patch('integrations.photos.PhotoProcessor.PHOTOS_SQLITE_PATH', db_path), \
            patch('common.FloxyDatabase.FloxyDatabase.get_instance')

    def test_only_the_holes_inside_the_window_oldest_first(self, tmp_path):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        db_path = self._library(tmp_path)
        lib, db = self._with_library(db_path)
        with lib, db as mock_db:
            mock_db.return_value.get_indexed_photo_guids.return_value = {'G-INDEXED'}
            rows = PhotoProcessor().unindexed_between(100.0, 500.0, limit=10)
            count = PhotoProcessor().count_unindexed_photos(after_apple_ts=100.0, before_apple_ts=500.0)
        assert [r["uuid"] for r in rows] == ['uuid-hole-early', 'uuid-hole-late', 'uuid-edge']
        assert count == len(rows)

    def test_the_limit_takes_the_oldest(self, tmp_path):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        db_path = self._library(tmp_path)
        lib, db = self._with_library(db_path)
        with lib, db as mock_db:
            mock_db.return_value.get_indexed_photo_guids.return_value = set()
            rows = PhotoProcessor().unindexed_between(100.0, 500.0, limit=2)
        assert [r["uuid"] for r in rows] == ['uuid-indexed', 'uuid-hole-early']

    def test_the_unbounded_count_still_covers_everything_after_the_cursor(self, tmp_path):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        db_path = self._library(tmp_path)
        lib, db = self._with_library(db_path)
        with lib, db as mock_db:
            mock_db.return_value.get_indexed_photo_guids.return_value = set()
            after = PhotoProcessor().count_unindexed_photos(after_apple_ts=500.0)
            window = PhotoProcessor().count_unindexed_photos(after_apple_ts=100.0, before_apple_ts=500.0)
            everything = PhotoProcessor().count_unindexed_photos()
        assert (after, window) == (1, 4)
        assert everything == after + window + 1, "the asset at the exclusive start is the one neither sees"


@pytest.mark.unit
class TestPhotosFeedProviderRefusal:
    """The feed's side of a refusal: the cursor is held, an empty prepaid account pauses the feed
    and tells the household once, and the health check brings it back with one more note."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        from integrations.photos.PhotosIntegration import PhotosIntegration
        PhotosIntegration.reset_instance()
        yield
        PhotosIntegration.reset_instance()

    @staticmethod
    def _opened(reason, out_of_credits):
        from integrations.photos.PhotosIntegration import PhotosIntegration
        integration = PhotosIntegration()
        processor = Mock()
        processor.provider_unavailable.return_value = reason
        processor.provider_out_of_credits.return_value = out_of_credits
        processor.refused_last_run.return_value = []
        with patch(f'{PI}.os.path.exists', return_value=True), \
             patch(f'{PI}.PhotoProcessor', return_value=processor):
            assert integration.open_feed() is True
        return integration, processor

    def _fetch(self, integration, cursor="500.0"):
        from integrations.photos.PhotosIntegration import PhotosIntegration
        rows = [TestPhotosFeed._row("a", 600.0, "G-A"), TestPhotosFeed._row("b", 700.0, "G-B")]
        with patch.object(PhotosIntegration, '_assets_after', return_value=rows), \
             patch.object(integration, 'remaining', return_value=120), \
             patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db, \
             patch('assistant.integration.FeedRunner.Feeds.get_instance') as feeds, \
             patch('assistant.SystemNote.system_note') as note:
            mock_db.return_value.get_indexed_photo_guids.return_value = set()
            batch = integration.fetch(cursor, backfill=True)
        return batch, feeds.return_value.runner.return_value, note

    def test_a_rate_limit_holds_the_cursor_and_bothers_nobody(self):
        from assistant.integration.Integration import IntegrationState
        integration, _ = self._opened("rate limited (429) — background requests wait 60s", out_of_credits=False)
        batch, runner, note = self._fetch(integration)
        assert batch.cursor == "500.0" and batch.exhausted is True
        runner.pause.assert_not_called()
        note.assert_not_called()
        assert integration.state != IntegrationState.RATE_LIMITED

    def test_empty_credits_pause_the_feed_and_tell_the_household_once(self):
        from assistant.integration.Integration import IntegrationState
        integration, _ = self._opened("prepaid credits are depleted (402)", out_of_credits=True)
        first, runner, note = self._fetch(integration)
        second, runner2, note2 = self._fetch(integration)
        assert first.cursor == second.cursor == "500.0"
        assert integration.state == IntegrationState.RATE_LIMITED
        assert "402" in integration.last_error
        runner.pause.assert_called_once()
        runner2.pause.assert_not_called()
        note.assert_called_once()
        note2.assert_not_called()
        text = note.call_args.args[1]
        assert "120" in text and "402" in text, "the note carries how much is waiting and what happened"

    def test_rows_refused_mid_batch_hold_the_cursor_after_the_outage_cleared(self):
        """A parallel worker's success (or a lapsed backoff) clears the shared outage before
        the batch is over: the rows refused meanwhile still have no description, so the cursor
        must not move past them."""
        from assistant.integration.Integration import IntegrationState
        integration, processor = self._opened(None, out_of_credits=False)
        processor.refused_last_run.return_value = ["a"]
        batch, runner, note = self._fetch(integration)
        assert batch.cursor == "500.0" and batch.exhausted is True
        runner.pause.assert_not_called()
        note.assert_not_called()
        assert integration.state != IntegrationState.RATE_LIMITED

    def test_health_check_probes_the_key_and_resumes_with_one_note(self):
        from assistant.integration.Integration import IntegrationState
        integration, processor = self._opened("prepaid credits are depleted (402)", out_of_credits=True)
        self._fetch(integration)
        llm = processor._get_llm.return_value
        llm.probe.side_effect = [False, True]
        with patch('assistant.integration.FeedRunner.Feeds.get_instance') as feeds, \
             patch('assistant.SystemNote.system_note') as note, \
             patch(f'{PI}.os.path.exists', return_value=True), \
             patch(f'{PI}.sqlite3.connect'), \
             patch.object(integration, '_check_icloud_download'):
            with pytest.raises(RuntimeError):
                integration.check_health()
            feeds.return_value.runner.return_value.resume.assert_not_called()
            assert integration.check_health() is True
        feeds.return_value.runner.return_value.resume.assert_called_once()
        note.assert_called_once()
        assert integration.last_error is None
        assert integration.state == IntegrationState.RATE_LIMITED, "the monitor, not the check, flips the state to OK"

    def test_system_prompt_is_identical_across_photos_so_a_prefix_cache_covers_it(self):
        from integrations.photos.PhotoProcessor import PhotoProcessor
        reply = '{"short_description":"a","detailed_description":"b","pretty_score":1,"family_importance_score":2}'
        llm = Mock()
        llm.think.return_value = reply
        llm.think_with_video.return_value = reply
        processor = PhotoProcessor()
        processor._llm = llm
        with patch.object(PhotoProcessor, '_read_memory', return_value="Marcus lives at Storgatan 1"):
            processor._analyze_with_llm(b"jpeg", "2026-01-01 10:00", "Home", "Family", "Charlotte", "photo")
            processor._analyze_with_llm(b"jpeg", "2026-03-14 18:30", "IKEA, Kållered", "", "", "screenshot")
            processor._analyze_video_with_llm(b"mp4", "video/mp4", "2026-01-01 10:00", "Home", "", "Frances", "video", 12.0)
            processor._analyze_video_with_llm(b"mp4", "video/mp4", "2026-05-05 09:00", "Beach", "", "", "slow-motion", 3.0)
        photo_a, photo_b = (c.kwargs for c in llm.think.call_args_list)
        assert photo_a["system_message"] == photo_b["system_message"]
        assert "Marcus lives at Storgatan 1" in photo_a["system_message"]
        assert "2026-03-14 18:30" in photo_b["user_message"] and "Kållered" in photo_b["user_message"]
        assert "2026-03-14" not in photo_b["system_message"]
        video_a, video_b = (c.kwargs for c in llm.think_with_video.call_args_list)
        assert video_a["system_message"] == video_b["system_message"]
        assert "Beach" in video_b["user_message"] and "slow-motion" in video_b["user_message"]

    def test_the_first_run_stub_is_not_sent_as_memory(self):
        """Until the first curation the bank is the shipped stub: teaching notes and another
        family's examples, which would only mislead the vision model about who is in a photo."""
        from assistant import MemoryFile
        from integrations.photos.PhotoProcessor import PhotoProcessor
        stub = MemoryFile.FIRST_RUN_STUB.read_text()
        assert MemoryFile.is_placeholder(stub), "the shipped stub must be recognised as uncurated"
        with patch.object(MemoryFile, "read", return_value=stub):
            assert PhotoProcessor._read_memory() == ""
            assert "(no memory available)" in PhotoProcessor()._photo_system_prompt()
        with patch.object(MemoryFile, "read", return_value="Marcus lives at Storgatan 1"):
            assert PhotoProcessor._read_memory() == "Marcus lives at Storgatan 1"


REPLY = '{"short_description":"a barn","detailed_description":"red barn","pretty_score":40,"family_importance_score":10}'


@pytest.mark.unit
class TestBatchedBackfill:
    """The catch-up's stills go through the batch API (half price); whatever it does not answer,
    and every video, runs on the live path as before."""

    @staticmethod
    def _rows(n, kind="photo"):
        return [{"uuid": f"U{i}", "filename": f"p{i}.heic", "kind": kind, "cloud_asset_guid": f"G{i}",
                 "creation_ts": 1_700_000_000.0 + i, "latitude": None, "longitude": None, "kind_subtype": 0,
                 "directory": "0", "duration": None} for i in range(n)]

    @staticmethod
    def _prepared(module, row, jpeg=b"jpeg"):
        return module._PreparedPhoto(row=row, jpeg=jpeg, photo_type="photo", albums="", persons="Charlotte",
                                     location_text="Home", user_message=f"photo {row['uuid']}", started=0.0)

    def _processor(self, module, llm):
        llm.unavailable.return_value = None
        processor = module.PhotoProcessor()
        processor._llm = llm
        return processor

    def test_stills_go_out_as_one_batch_and_only_the_unanswered_run_live(self):
        from integrations.photos import PhotoProcessor as module
        rows = self._rows(3)
        llm = Mock(is_local=Mock(return_value=False))
        llm.think_batched_with_images.return_value = [REPLY, None, REPLY]
        processor = self._processor(module, llm)
        live = []
        with patch.object(processor, '_prepare_photo', side_effect=lambda row, db: (self._prepared(module, row), False)), \
             patch.object(module.PhotoProcessor, '_batch_enabled', return_value=True), \
             patch.object(processor, '_photo_system_prompt', return_value="SYSTEM"), \
             patch.object(processor, '_store_analysis', return_value=1) as store, \
             patch.object(processor, '_process_one', side_effect=lambda row: live.append(row["uuid"]) and False), \
             patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db:
            mock_db.return_value.photo_exists.return_value = True
            mock_db.return_value.get_photo_description.return_value = "desc"
            succeeded, failed = processor.process_rows(rows, batched=True)

        call = llm.think_batched_with_images.call_args
        assert call.args[0] == "SYSTEM"
        assert call.args[1] == [("photo U0", b"jpeg"), ("photo U1", b"jpeg"), ("photo U2", b"jpeg")]
        assert [c.args[1]["uuid"] for c in store.call_args_list] == ["U0", "U2"]
        assert live == ["U1"], "the one the batch left unanswered is indexed live, nothing is indexed twice"
        assert len(succeeded) == 3 and failed == []
        assert not module.PhotoProcessor._processing_uuids, "every uuid is released afterwards"

    def test_no_batch_api_means_every_still_runs_live(self):
        from integrations.photos import PhotoProcessor as module
        rows = self._rows(2)
        llm = Mock(is_local=Mock(return_value=False))
        llm.think_batched_with_images.return_value = None
        processor = self._processor(module, llm)
        live = []
        with patch.object(processor, '_prepare_photo', side_effect=lambda row, db: (self._prepared(module, row), False)), \
             patch.object(module.PhotoProcessor, '_batch_enabled', return_value=True), \
             patch.object(processor, '_photo_system_prompt', return_value="SYSTEM"), \
             patch.object(processor, '_process_one', side_effect=lambda row: live.append(row["uuid"]) and False), \
             patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db:
            mock_db.return_value.photo_exists.return_value = True
            mock_db.return_value.get_photo_description.return_value = "desc"
            processor.process_rows(rows, batched=True)
        assert sorted(live) == ["U0", "U1"]

    def test_a_still_waiting_on_icloud_is_settled_by_the_prepare_step_not_batched(self):
        from integrations.photos import PhotoProcessor as module
        rows = self._rows(2)
        llm = Mock(is_local=Mock(return_value=False))
        llm.think_batched_with_images.return_value = [REPLY]
        processor = self._processor(module, llm)

        def prepare(row, db):
            return (None, True) if row["uuid"] == "U0" else (self._prepared(module, row), False)

        with patch.object(processor, '_prepare_photo', side_effect=prepare), \
             patch.object(module.PhotoProcessor, '_batch_enabled', return_value=True), \
             patch.object(processor, '_photo_system_prompt', return_value="SYSTEM"), \
             patch.object(processor, '_store_analysis', return_value=1), \
             patch.object(processor, '_process_one') as process_one, \
             patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db:
            mock_db.return_value.photo_exists.return_value = True
            mock_db.return_value.get_photo_description.return_value = "desc"
            processor.process_rows(rows, batched=True)

        assert [u for u, _ in llm.think_batched_with_images.call_args.args[1]] == ["photo U1"]
        process_one.assert_not_called()
        assert "U0" in processor._pending_icloud, "the iCloud retry set is fed exactly as on the live path"

    def test_videos_and_a_live_fetch_never_touch_the_batch_api(self):
        from integrations.photos import PhotoProcessor as module
        llm = Mock(is_local=Mock(return_value=False))
        processor = self._processor(module, llm)
        with patch.object(processor, '_process_one', return_value=False), \
             patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db:
            mock_db.return_value.photo_exists.return_value = True
            mock_db.return_value.get_photo_description.return_value = "desc"
            processor.process_rows(self._rows(2, kind="video"), batched=True)
            processor.process_rows(self._rows(2), batched=False)
        llm.think_batched_with_images.assert_not_called()

    @pytest.mark.parametrize("local, enabled", [(True, True), (False, False)])
    def test_a_local_backend_or_the_config_switch_keeps_the_backfill_live(self, local, enabled):
        from integrations.photos import PhotoProcessor as module
        llm = Mock(is_local=Mock(return_value=local))
        processor = self._processor(module, llm)
        with patch.object(processor, '_process_one', return_value=False), \
             patch.object(module.PhotoProcessor, '_batch_enabled', return_value=enabled), \
             patch('common.FloxyDatabase.FloxyDatabase.get_instance') as mock_db:
            mock_db.return_value.photo_exists.return_value = True
            mock_db.return_value.get_photo_description.return_value = "desc"
            processor.process_rows(self._rows(2), batched=True)
        llm.think_batched_with_images.assert_not_called()
