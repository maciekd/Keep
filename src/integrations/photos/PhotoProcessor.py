import json
import mimetypes
import os
import plistlib
import re
import sqlite3
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Callable, Optional

from assistant import MemoryFile
from assistant.llm import create_llm, parse_llm_config
from assistant.integration.Feed import FeedEntry
from common.logging import info, error, trace, fine
from common.FloxyDatabase import FloxyDatabase, PHOTO_HISTORY_SOURCE
from assistant.llm.AbstractLLM import AbstractLLM, LLMPreempted, LLMUnavailable
from assistant.tools.ImageAnalyzer import parse_description_json
from assistant.tools.VideoAnalyzer import short_clip_fps
from .ICloudPhotoDownloader import (CorruptImageError, NoVideoRenditionError, PhotoNotInCloudError,
                                    VideoTooLargeError)

_FAMILY_CONTEXT = """# Current long term memory about the family and house
{current_memory}
"""

ANALYSIS_PROMPT_SYSTEM = """

You are a photo analyst for a family's personal photo library. Analyze the provided photo and return a JSON object.

""" + _FAMILY_CONTEXT + """
# Return ONLY a JSON object with these keys
- "short_description": A concise 1-sentence description of the photo (max 50 chars)
- "detailed_description": A RICH description covering subjects, setting, mood, colors, and notable details. Include any visible text, objects, landmarks, or activities. This will be used for search. Be verbose, detail everything in the photo with high accuracy. Around 200 words is good.
- "thinking": Detail your thinking about how you decide on the two scores below. Don't emit your scores here, only the thinking
- "pretty_score": 0-100 rating of aesthetic quality (facial expressions, composition, lighting, color, sharpness). 0=average snapshot, 80+=genuinely beautiful, 90+=exceptional
- "family_importance_score": 0-100 rating of likely importance to a family (milestones, gatherings, children, pets, holidays). 0=no family connection, 80+=significant moment, 90+=exceptional

## Scoring guidance
- pretty_score: penalize blurry, poorly lit, or cluttered photos. Reward good composition, nice facial expressions, good light, interesting perspectives. Generally what's considered a beautiful photo.
- family_importance_score: consider what the family would truly appreciate in a photo. penalize pictures with few family members in them

"""

# The per-asset fields come last, in the user message: everything before them is identical from
# photo to photo, so a local model's prefix cache covers the memory and the instructions and each
# request prefills only the image and these lines.
ANALYSIS_PROMPT_USER = """# Context about this photo
- Date taken: {date}
- Location: {location}
- Albums: {albums}
- People identified: {persons}
- Photo type: {photo_type}

Analyze this photo and return the JSON object as described."""

VIDEO_ANALYSIS_PROMPT_SYSTEM = """

You are a video analyst for a family's personal photo library. Watch the provided video, including its audio track, and return a JSON object.

""" + _FAMILY_CONTEXT + """
# Return ONLY a JSON object with these keys
- "short_description": A concise 1-sentence description of the video (max 50 chars)
- "detailed_description": A RICH description of what happens over the course of the clip: subjects, setting, actions, mood, and notable details. For clips longer than about 30 seconds, follow the timeline ("first ..., then ..., at the end ...") so that moments late in the clip are described too. Include any visible text, landmarks, or activities. This will be used for search.
- "spoken_content": What is said or sung in the video, quoted or closely summarized, covering the whole clip. Keep it under 1500 characters; summarize rather than transcribe long passages. An empty string if nobody speaks or sings.
- "thinking": Detail your thinking about how you decide on the two scores below. Don't emit your scores here, only the thinking
- "pretty_score": 0-100 rating of aesthetic quality (steady, well-lit, well-framed footage, nice expressions, clear sound). 0=average clip, 80+=genuinely beautiful, 90+=exceptional
- "family_importance_score": 0-100 rating of likely importance to a family (milestones, first words, songs, speeches at gatherings, children, pets, holidays). 0=no family connection, 80+=significant moment, 90+=exceptional

## Scoring guidance
- pretty_score: penalize shaky, dark, or cluttered footage. Reward good framing, nice facial expressions, good light, interesting moments. Generally what's considered a beautiful clip.
- family_importance_score: consider what the family would truly appreciate in a video. penalize clips with few family members in them

"""

VIDEO_ANALYSIS_PROMPT_USER = """# Context about this video
- Date taken: {date}
- Location: {location}
- Albums: {albums}
- People identified: {persons}
- Video type: {video_type}
- Duration: {duration} seconds

Analyze this video and return the JSON object as described."""

# The "thinking" field of the two prompts above. Sonnet 5.5, Opus 5.5 and Fable refuse a request
# that asks for the model's reasoning (`stop_reason: refusal`, reasoning_extraction) and return
# nothing else, so for such a backend the field is left out; the scores and their guidance stay.
REASONING_FIELD_LINE = re.compile(r'^- "thinking":.*\n', re.MULTILINE)


def without_reasoning_field(system_prompt: str) -> str:
    """The analysis prompt minus its "thinking" field, for a backend that refuses to be asked
    for its reasoning (`AbstractLLM.refuses_reasoning_requests`)."""
    return REASONING_FIELD_LINE.sub("", system_prompt)


PHOTOS_SQLITE_PATH = os.path.expanduser(
    "~/Pictures/Photos Library.photoslibrary/database/Photos.sqlite"
)

PHOTOS_ORIGINALS_DIR = os.path.expanduser(
    "~/Pictures/Photos Library.photoslibrary/originals"
)

# ZSAVEDASSETTYPE for Messages "Shared with You" syndication assets. These appear in
# Photos.sqlite but are not part of the user's iCloud library — they never upload and
# never get a ZCLOUDASSETGUID, so they can't be indexed and must be excluded everywhere.
SYNDICATION_SAVED_ASSET_TYPE = 12

PERSONS_QUERY = """
SELECT DISTINCT ZPERSON.ZDISPLAYNAME
FROM ZDETECTEDFACE
JOIN ZPERSON ON ZPERSON.Z_PK = ZDETECTEDFACE.ZPERSONFORFACE
JOIN ZASSET ON ZASSET.Z_PK = ZDETECTEDFACE.ZASSETFORFACE
WHERE ZASSET.ZUUID = ? AND ZPERSON.ZDISPLAYNAME IS NOT NULL
"""

ALBUMS_QUERY = """
SELECT DISTINCT g.ZTITLE
FROM Z_33ASSETS za
JOIN ZGENERICALBUM g ON g.Z_PK = za.Z_33ALBUMS
JOIN ZASSET a ON a.Z_PK = za.Z_3ASSETS
WHERE a.ZUUID = ? AND g.ZTITLE IS NOT NULL
"""

# Where a photo was taken, as Photos itself worked it out: the moment's title is the label
# the app shows ("Home", "Östra Haga", "Paris"), the reverse-location blob the full
# address Apple geocoded — no key and no network needed, and it covers assets without GPS too.
LOCATION_QUERY = """
SELECT mo.ZTITLE, aa.ZREVERSELOCATIONDATA
FROM ZASSET a
LEFT JOIN ZMOMENT mo ON mo.Z_PK = a.ZMOMENT
LEFT JOIN ZADDITIONALASSETATTRIBUTES aa ON aa.ZASSET = a.Z_PK
WHERE a.ZUUID = ?
"""

APPLE_EPOCH_OFFSET = 978307200
SF_DATALESS = 0x40000000
# An original smaller than this is not a photo but a stub left by a broken download: treated as
# not local, so the bytes come from iCloud instead of sips failing on it.
MIN_ORIGINAL_BYTES = 1024
MAX_ICLOUD_RETRIES = 30  # ~5 min at 10s poll interval

MEDIA_SUBTYPE_FLAGS = {
    (1 << 0): "panorama",
    (1 << 1): "HDR",
    (1 << 3): "screenshot",
    (1 << 4): "live",
    (1 << 5): "live",
    (1 << 10): "portrait",
}

# ZKIND values in Photos.sqlite.
KIND_PHOTO = 0
KIND_VIDEO = 1

# For videos ZKINDSUBTYPE is an enum, not the bit flags stills use.
VIDEO_SUBTYPE_LABELS = {
    101: "slo-mo",
    102: "time-lapse",
    103: "cinematic",
}

# Each video is a 10–20 MB download plus a 5–20 s video-LLM call; the stills fan-out of 30
# threads would saturate the uplink and trip Gemini's rate limits.
PHOTO_THREADS = 30
VIDEO_THREADS = 5
# A local backend (Ollama) serves one request at a time, so a wide fan-out only piles up
# requests it cannot cancel. Two workers: one image in the model while the other one's
# iCloud read, HEIC conversion and encoding happen, so the model never waits for a photo.
LOCAL_PHOTO_THREADS = 2
# The backfill (a feed fetch while the Photos feed is catching up) can send its stills through the
# provider's batch API at half price: one job per fetch of up to FETCH_BATCH_SIZE photos, a job
# that has not ended after BATCH_MAX_WAIT_SECONDS cancelled and its photos indexed live.
# Off by default (2026-10-07): the batch API promises an answer within 24 h and nothing sooner.
# A first-run test on a quiet afternoon left a 79-photo job in JOB_STATE_RUNNING for the whole
# 30 min while the console showed 0 indexed — work the live path does in well under a minute,
# 30 threads at ~9 s a photo. The discount is half of a few cents per fetch and tens of
# dollars once over a whole library, not worth a counter that stands still for half an hour while
# someone watches. Flip BACKFILL_BATCH (or `feeds.photos.backfill_batch: true`, which overrides
# it) to use the batch API again; before relying on it, make it submit-and-return with the job
# names persisted, so no feed thread sleeps on a queue with a 24 h contract.
BACKFILL_BATCH = False
BATCH_MAX_WAIT_SECONDS = 1800


@dataclass
class _PreparedPhoto:
    """A still with its bytes and context in hand, ready for the LLM — the point where the live
    path and the batch path part."""
    row: dict
    jpeg: bytes
    photo_type: str
    albums: str
    persons: str
    location_text: str
    user_message: str
    started: float

ASSET_COLUMNS = """a.ZUUID, a.ZFILENAME, a.ZDIRECTORY, a.ZDATECREATED, a.ZLATITUDE, a.ZLONGITUDE,
           a.ZKINDSUBTYPE, a.ZKIND, a.ZDURATION, m.ZCLOUDMASTERGUID, a.ZCLOUDASSETGUID,
           a.ZLIBRARYSCOPE"""


# The 720p iCloud rendition is ~0.45 MB/s, so 150 MB covers a clip at the default 300 s duration
# cap; clips over 19 MB go to Gemini through the Files API, so the byte cap is not a transport limit.
DEFAULT_VIDEO_MAX_SIZE_MB = 150
# Long clips can carry minutes of speech; the history entry and index row keep this much of it.
MAX_SPOKEN_CONTENT_CHARS = 1500

@dataclass(frozen=True)
class VideoSettings:
    enabled: bool
    max_bytes: int
    max_duration_seconds: float


def video_settings() -> VideoSettings:
    """The `photos.videos` config block; videos are indexed unless switched off."""
    from common.ConfigManager import ConfigManager
    cfg = ConfigManager.get_instance()
    return VideoSettings(
        enabled=bool(cfg.get('photos.videos.enabled', True)),
        max_bytes=int(cfg.get('photos.videos.max_size_mb', DEFAULT_VIDEO_MAX_SIZE_MB)) * 1024 * 1024,
        max_duration_seconds=float(cfg.get('photos.videos.max_duration_seconds', 300)),
    )


def truncate_spoken_content(spoken: Optional[str]) -> str:
    text = (spoken or "").strip()
    if len(text) > MAX_SPOKEN_CONTENT_CHARS:
        trace(f"Photos: spoken content truncated from {len(text)} to {MAX_SPOKEN_CONTENT_CHARS} characters")
        text = text[:MAX_SPOKEN_CONTENT_CHARS].rstrip() + "…"
    return text


def indexed_kinds() -> tuple[int, ...]:
    return (KIND_PHOTO, KIND_VIDEO) if video_settings().enabled else (KIND_PHOTO,)


def asset_kind_filter(kinds: tuple[int, ...] | None = None) -> str:
    """SQL fragment (alias `a`) selecting the asset kinds the index covers, minus hidden assets.

    Photos the user hid (the Hidden album, ZHIDDEN = 1) are never described or stored."""
    kinds = kinds or indexed_kinds()
    return f"a.ZKIND IN ({', '.join(str(k) for k in kinds)}) AND a.ZHIDDEN = 0"


HIDDEN_ASSETS_QUERY = "SELECT a.ZCLOUDASSETGUID FROM ZASSET a WHERE a.ZHIDDEN = 1 AND a.ZCLOUDASSETGUID IS NOT NULL"


def asset_row(r: sqlite3.Row) -> dict:
    """The dict the processor works on, from one ASSET_COLUMNS row. Shared with the observer."""
    lat = r["ZLATITUDE"]
    lon = r["ZLONGITUDE"]
    if lat is not None and (lat < -90 or lat > 90):
        lat = None
    if lon is not None and (lon < -180 or lon > 180):
        lon = None
    if lat is None or lon is None:
        lat = None
        lon = None
    is_video = r["ZKIND"] == KIND_VIDEO
    return {
        "uuid": r["ZUUID"],
        "filename": r["ZFILENAME"],
        "directory": r["ZDIRECTORY"],
        "creation_date_apple": r["ZDATECREATED"],
        "creation_ts": r["ZDATECREATED"] + APPLE_EPOCH_OFFSET if r["ZDATECREATED"] else time.time(),
        "latitude": lat,
        "longitude": lon,
        "kind_subtype": r["ZKINDSUBTYPE"],
        "kind": "video" if is_video else "photo",
        "duration": (r["ZDURATION"] or 0.0) if is_video else None,
        "cloud_master_guid": r["ZCLOUDMASTERGUID"],
        "cloud_asset_guid": r["ZCLOUDASSETGUID"],
        "library_scope": r["ZLIBRARYSCOPE"] or 0,
    }


def _plain_spaces(text: str) -> str:
    """Photos writes some place names with non-breaking spaces; one form for matching."""
    return " ".join(text.split())


def postal_address(reverse_location_data: Optional[bytes]) -> dict:
    """The CNPostalAddress fields (city, country, state, subLocality…) and the isHome flag
    from a ZREVERSELOCATIONDATA blob — an NSKeyedArchiver plist of a PLRevGeoLocationInfo.
    Empty when the blob is missing or not what we expect."""
    if not reverse_location_data:
        return {}
    try:
        archive = plistlib.loads(reverse_location_data)
        objects = archive["$objects"]

        def deref(value):
            return objects[value.data] if isinstance(value, plistlib.UID) else value

        def classname(obj) -> str:
            return deref(obj["$class"]).get("$classname", "") if isinstance(obj, dict) and "$class" in obj else ""

        fields = {}
        for obj in objects:
            if classname(obj) == "PLRevGeoLocationInfo":
                fields["isHome"] = bool(obj.get("isHome", False))
            elif classname(obj) == "CNPostalAddress":
                for key, value in obj.items():
                    if key.startswith("_"):
                        text = deref(value)
                        if isinstance(text, str) and text != "$null" and text:
                            fields[key[1:]] = _plain_spaces(text)
        return fields
    except Exception as e:
        trace(f"Photos: reverse-location data not readable: {e}")
        return {}


def apple_location(moment_title: Optional[str], reverse_location_data: Optional[bytes]) -> str:
    """"Home", or "<moment title>, <city>, <country>" without repeats — the text the LLM
    sees and the index searches. The moment title is what the Photos app shows for the
    day and place; city and country come from Apple's own reverse geocoding."""
    address = postal_address(reverse_location_data)
    title = _plain_spaces(moment_title or "")
    if address.get("isHome") or title == "Home":
        return "Home"
    parts = []
    for part in (title, address.get("city"), address.get("country")):
        if part and part not in parts:
            parts.append(part)
    return ", ".join(parts)


class PhotoProcessor:
    """Analyzes photos with a vision LLM and indexes the descriptions in the Floxy database.

    Photos not available locally are fetched from iCloud and retried across batches,
    then stored as skipped after MAX_ICLOUD_RETRIES.

    Every indexed asset also becomes one history row. With an `entry_sink` (the Photos feed)
    the row is handed over as a FeedEntry for FeedRunner to store with the batch; without
    one (the TUI's Photos tab, scripts) it is written to the history directly.
    """

    _processing_uuids: set = set()
    _processing_lock = threading.Lock()
    # .queued_s: how long the photo this worker thread is on spent waiting for the LLM slot.
    _worker = threading.local()

    def __init__(self, entry_sink: Optional[Callable[[FeedEntry], None]] = None):
        self._entry_sink = entry_sink
        self._llm = None             # the `photo_indexing` backend: the new assets, as the feed sees them
        self._backfill_llm = None    # the `photo_backfill` backend: the catch-up, the sweep, a bulk run
        self._clip_backend: Optional[AbstractLLM] = None   # the `video` tier's, for the clips those cannot take
        self._clip_backend_asked = False
        self._llm_lock = threading.Lock()
        self._pending_icloud: dict[str, tuple[dict, int]] = {}  # uuid -> (row, retry_count); the row carries its pass
        self._icloud_lock = threading.Lock()
        self._refused_last_run: list[str] = []    # uuids the backend refused in the last process_rows
        self._reasoning_logged: set[str] = set()  # backends reported as refusing the prompt's "thinking" field

    def remove_hidden_assets(self) -> int:
        """Drop indexed rows for assets the user has since hidden in Photos; returns how many."""
        try:
            if not os.path.exists(PHOTOS_SQLITE_PATH):
                return 0
            conn = sqlite3.connect(f"file:{PHOTOS_SQLITE_PATH}?mode=ro", uri=True)
            try:
                hidden_guids = {guid for (guid,) in conn.execute(HIDDEN_ASSETS_QUERY).fetchall()}
            finally:
                conn.close()
            db = FloxyDatabase.get_instance()
            indexed_hidden = hidden_guids & db.get_indexed_photo_guids()
            if not indexed_hidden:
                trace(f"Photos: none of the {len(hidden_guids)} hidden asset(s) are indexed, nothing to remove")
                return 0
            removed = db.delete_photos(indexed_hidden)
            info(f"Photos: removed {removed} hidden asset(s) from the index")
            return removed
        except Exception as e:
            error(f"Photos: could not remove hidden assets from the index: {e}")
            return 0

    def count_unindexed_photos(self, kinds: tuple[int, ...] | None = None,
                               after_apple_ts: float | None = None, before_apple_ts: float | None = None) -> int:
        """Count the assets of the given kinds (default: every indexed kind) not yet indexed,
        optionally only those created after `after_apple_ts` (the feed's cursor) and, with
        `before_apple_ts` too, at or before it (the feed's sweep: the horizon up to the cursor)."""
        try:
            db = FloxyDatabase.get_instance()
            indexed_guids = db.get_indexed_photo_guids()
            if not os.path.exists(PHOTOS_SQLITE_PATH):
                return 0
            conn = sqlite3.connect(f"file:{PHOTOS_SQLITE_PATH}?mode=ro", uri=True)
            try:
                cursor = conn.cursor()
                query = (f"SELECT a.ZCLOUDASSETGUID FROM ZASSET a WHERE {asset_kind_filter(kinds)} "
                         f"AND a.ZTRASHEDSTATE = 0 AND a.ZSAVEDASSETTYPE != {SYNDICATION_SAVED_ASSET_TYPE} "
                         "AND a.ZCLOUDASSETGUID IS NOT NULL")
                params: tuple = ()
                if after_apple_ts is not None:
                    query += " AND a.ZDATECREATED > ?"
                    params += (after_apple_ts,)
                if before_apple_ts is not None:
                    query += " AND a.ZDATECREATED <= ?"
                    params += (before_apple_ts,)
                cursor.execute(query, params)
                return sum(1 for (guid,) in cursor.fetchall() if guid not in indexed_guids)
            finally:
                conn.close()
        except Exception as e:
            trace(f"Photos: could not count unindexed photos: {e}")
            return 0

    @staticmethod
    def tier(backfill: bool) -> str:
        """The LLM tier a pass runs on: `photo_backfill` for the catch-up, the sweep and a bulk
        run, `photo_indexing` for the new assets the feed sees as they arrive."""
        return 'photo_backfill' if backfill else 'photo_indexing'

    def _get_llm(self, backfill: bool = False) -> AbstractLLM:
        """The indexing LLM of a pass: the live one (`photo_indexing`, Sonnet by default — a few
        new assets a day, described well) or the backfill's (`photo_backfill`, Haiku by default —
        thousands of stills nobody is waiting for). Built through create_llm() like every other
        backend: the analysis prompt carries all of the memory bank, and only the VaultLLM
        wrapper keeps its secrets at home. A Claude model takes stills only; a clip then goes
        to the `video` tier's backend (_clip_llm)."""
        llm = self._backfill_llm if backfill else self._llm
        if llm is None:
            with self._llm_lock:
                llm = self._backfill_llm if backfill else self._llm
                if llm is None:
                    tier = self.tier(backfill)
                    name, model = parse_llm_config(tier)
                    llm = create_llm(name, model)
                    llm.initialize()
                    trace(f"Photos: using {name} ({model}) for the {tier} tier")
                    if backfill:
                        self._backfill_llm = llm
                    else:
                        self._llm = llm
        return llm

    def _clip_llm(self) -> Optional[AbstractLLM]:
        """The backend for the clips the indexing backend cannot take: the `video` tier's
        (Gemini by default; Claude has no video input), or None when that one takes no video
        either or has no key yet — then the clips stay unindexed, said once."""
        if not self._clip_backend_asked:
            with self._llm_lock:
                if not self._clip_backend_asked:
                    self._clip_backend = self._build_clip_llm()
                    self._clip_backend_asked = True
        return self._clip_backend

    @staticmethod
    def _build_clip_llm() -> Optional[AbstractLLM]:
        try:
            name, model = parse_llm_config('video')
            llm = create_llm(name, model)
            if not llm.supports_video():
                info(f"Photos: the indexing backend takes no video and neither does the video tier's {llm.name()} — clips stay unindexed")
                return None
            missing = llm.missing_api_key()
            if missing:
                info(f"Photos: clips go to {llm.name()} once {missing} is set — until then they stay unindexed")
                return None
            llm.initialize()
            info(f"Photos: {llm.name()} (the video tier) describes the clips the indexing backend cannot take")
            return llm
        except Exception as e:
            error(f"Photos: the video tier's backend could not be built ({e}) — clips stay unindexed")
            return None

    def _built_llms(self) -> list[AbstractLLM]:
        """Every backend built so far, the clips' one included: a 402 or 429 on the clips
        has to hold the feed's cursor and pause it like one on the stills (2026-10-19 — the
        clip backend was left out, and a Gemini credit refusal froze the indexing in silence)."""
        return [llm for llm in (self._llm, self._backfill_llm, self._clip_backend) if llm is not None]

    def provider_unavailable(self) -> Optional[str]:
        """Why an indexing backend refuses every request right now (out of credits, rate
        limited), or None. Asked before a batch is downloaded and after one ran, so the feed
        can hold its cursor instead of moving past photos nobody described."""
        return next((reason for llm in self._built_llms() if (reason := llm.unavailable())), None)

    def provider_out_of_credits(self) -> bool:
        return any(llm.out_of_credits() for llm in self._built_llms())

    def probe_provider(self) -> bool:
        """Whether every indexing backend built so far takes requests again — the health
        check's way out of RATE_LIMITED. With none built yet the live one is built and asked."""
        return all(llm.probe() for llm in self._built_llms() or [self._get_llm()])

    def _indexing_backend_is_local(self, backfill: bool = False) -> bool:
        """Whether the pass's backend serves one request at a time. Decided before the first
        asset is touched, so it must not initialize a cloud client (that needs the API key) just
        to pick a worker count: like tier_api_key_name(), it builds the backend uninitialized."""
        llm = self._backfill_llm if backfill else self._llm
        if llm is not None:
            return llm.is_local()
        name, model = parse_llm_config(self.tier(backfill))
        return create_llm(name, model).is_local()

    def unindexed_between(self, after_apple_ts: float, before_apple_ts: float, limit: int) -> list[dict]:
        """The oldest `limit` assets created after `after_apple_ts` and at or before
        `before_apple_ts` that have no index row — the feed's sweep of the stretch behind its
        cursor. Assets still without an iCloud GUID are left out, as everywhere."""
        rows, _ = self._find_unprocessed(limit=limit, window=(after_apple_ts, before_apple_ts))
        return rows

    def _find_unprocessed(self, album: Optional[str] = None, limit: int = 1,
                          window: tuple[float, float] | None = None) -> tuple[list[dict], str]:
        """Find unprocessed photos, newest first — or, with `window` (Apple timestamps,
        exclusive start, inclusive end), the ones created inside it, oldest first. Returns
        (rows, error_message)."""
        db = FloxyDatabase.get_instance()
        indexed_guids = db.get_indexed_photo_guids()

        if not os.path.exists(PHOTOS_SQLITE_PATH):
            return [], "Apple Photos database not found"

        conn = sqlite3.connect(f"file:{PHOTOS_SQLITE_PATH}?mode=ro", uri=True)
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            kind_filter = asset_kind_filter()
            if album == "Favorites":
                cursor.execute(f"""
                    SELECT {ASSET_COLUMNS}
                    FROM ZASSET a
                    LEFT JOIN ZCLOUDMASTER m ON m.Z_PK = a.ZMASTER
                    WHERE {kind_filter} AND a.ZTRASHEDSTATE = 0 AND a.ZFAVORITE = 1
                      AND a.ZSAVEDASSETTYPE != {SYNDICATION_SAVED_ASSET_TYPE}
                    ORDER BY a.ZDATECREATED DESC
                """)
            elif album:
                cursor.execute(f"""
                    SELECT {ASSET_COLUMNS}
                    FROM ZASSET a
                    LEFT JOIN ZCLOUDMASTER m ON m.Z_PK = a.ZMASTER
                    JOIN Z_33ASSETS za ON za.Z_3ASSETS = a.Z_PK
                    JOIN ZGENERICALBUM g ON g.Z_PK = za.Z_33ALBUMS
                    WHERE {kind_filter} AND a.ZTRASHEDSTATE = 0 AND g.ZTITLE = ?
                      AND a.ZSAVEDASSETTYPE != {SYNDICATION_SAVED_ASSET_TYPE}
                    ORDER BY a.ZDATECREATED DESC
                """, (album,))
            elif window:
                cursor.execute(f"""
                    SELECT {ASSET_COLUMNS}
                    FROM ZASSET a
                    LEFT JOIN ZCLOUDMASTER m ON m.Z_PK = a.ZMASTER
                    WHERE {kind_filter} AND a.ZTRASHEDSTATE = 0
                      AND a.ZSAVEDASSETTYPE != {SYNDICATION_SAVED_ASSET_TYPE}
                      AND a.ZDATECREATED > ? AND a.ZDATECREATED <= ?
                    ORDER BY a.ZDATECREATED ASC
                """, window)
            else:
                cursor.execute(f"""
                    SELECT {ASSET_COLUMNS}
                    FROM ZASSET a
                    LEFT JOIN ZCLOUDMASTER m ON m.Z_PK = a.ZMASTER
                    WHERE {kind_filter} AND a.ZTRASHEDSTATE = 0
                      AND a.ZSAVEDASSETTYPE != {SYNDICATION_SAVED_ASSET_TYPE}
                    ORDER BY a.ZDATECREATED DESC
                """)

            db_rows = cursor.fetchall()

            target_rows = []
            already_indexed = 0
            awaiting_sync = 0
            for r in db_rows:
                if not r["ZCLOUDASSETGUID"]:
                    # No iCloud asset GUID yet — the photo hasn't finished uploading.
                    # Defer so the index only ever contains portable GUIDs.
                    awaiting_sync += 1
                    continue
                if r["ZCLOUDASSETGUID"] in indexed_guids:
                    already_indexed += 1
                    continue
                target_rows.append(asset_row(r))
                if len(target_rows) >= limit:
                    break
        finally:
            conn.close()

        if awaiting_sync:
            info(f"Photos: {awaiting_sync} photo(s) deferred — not yet uploaded to iCloud (no asset GUID)")

        if not target_rows:
            msg = f"No unprocessed photos in '{album}'" if album else "No unprocessed photos found"
            return [], msg

        return target_rows, ""

    def process_next_unprocessed(self, album: Optional[str] = None) -> tuple:
        try:
            target_rows, err = self._find_unprocessed(album=album, limit=1)
            if not target_rows:
                return ("none", err) if err else ("none", "No unprocessed photos found")

            target_row = target_rows[0]
            self._process_one(target_row)
            db = FloxyDatabase.get_instance()
            asset_guid = target_row["cloud_asset_guid"]
            if not db.photo_exists(asset_guid):
                return ("error", "Processing failed (photo not stored)")
            desc = db.get_photo_description(asset_guid) or asset_guid
            return ("ok", f"Indexed: {desc}")

        except Exception as e:
            error(f"Photos: process_next_unprocessed failed: {e}")
            return ("error", f"Error: {e}")

    def process_bulk_unprocessed(self, count: int = 20, album: Optional[str] = None,
                                 on_progress=None) -> tuple:
        try:
            self._retry_icloud_pending()

            target_rows, err = self._find_unprocessed(album=album, limit=count)
            if not target_rows:
                return ("none", err) if err else ("none", "No unprocessed photos found")

            succeeded, failed = self.process_rows(target_rows, on_progress=on_progress, batched=True)
            total = len(succeeded) + len(failed)
            if failed:
                return ("error", f"Indexed {len(succeeded)}/{total} ({len(failed)} failed)")
            return ("ok", f"Indexed {len(succeeded)} photos")

        except Exception as e:
            error(f"Photos: process_bulk_unprocessed failed: {e}")
            return ("error", f"Error: {e}")

    def process_rows(self, rows: list, on_progress=None, batched: bool = False) -> tuple[list[str], list[str]]:
        """Index `rows` concurrently (PHOTO_THREADS stills, VIDEO_THREADS clips at a time) and
        wait for all of them. Assets whose original is still downloading from iCloud go to the
        pending set for `_retry_icloud_pending`. With `batched` (the catch-up, the sweep, a bulk
        run) the rows go to the `photo_backfill` tier instead of `photo_indexing` — the pass
        travels with the row, so an iCloud retry keeps it — and the stills through the batch API
        first when that is on, only the ones it did not answer running one by one; videos
        always run one by one. Returns (descriptions stored, filenames failed); the rows the backend refused
        (402 / 429) are in neither verdict but in `refused_last_run()`, so the feed holds its
        cursor for them even when a parallel worker's success or a lapsed backoff has cleared
        the shared outage by the time the batch is over."""
        t0 = time.time()
        rows = [{**row, "backfill": batched} for row in rows]
        self._refused_last_run = []
        reason = self.provider_unavailable()
        if reason:
            info(f"Photos: {len(rows)} asset(s) left in the queue — {reason}")
            self._refused_last_run = [row["uuid"] for row in rows]
            return [], [row["filename"] for row in rows]
        num_videos = sum(1 for row in rows if row.get("kind") == "video")
        local = self._indexing_backend_is_local(batched)
        photo_threads = LOCAL_PHOTO_THREADS if local else PHOTO_THREADS
        info(f"Photos: processing {len(rows) - num_videos} photo(s) with {photo_threads} threads"
             + (f" and {num_videos} video(s) with {VIDEO_THREADS} threads" if num_videos else "")
             + (" — stills through the batch API first" if batched and not local else ""))
        succeeded: list[str] = []
        failed: list[str] = []
        refused: list[str] = []
        lock = threading.Lock()
        settled: dict[str, bool] = {}
        if batched and not local and self._batch_enabled():
            settled = self._process_stills_batched([row for row in rows if row.get("kind") != "video"])

        def worker(row):
            try:
                if row["uuid"] in settled:
                    needs_icloud = settled[row["uuid"]]
                else:
                    try:
                        needs_icloud = self._process_one(row)
                    except LLMUnavailable as e:
                        # Not a verdict on the asset: nothing is stored, the feed holds its
                        # cursor and the same row comes round again.
                        trace(f"Photos: {row['uuid']} left in the queue — {e}")
                        with lock:
                            refused.append(row["uuid"])
                        return
                if needs_icloud:
                    with self._icloud_lock:
                        if row["uuid"] not in self._pending_icloud:
                            self._pending_icloud[row["uuid"]] = (row, 0)
                    return
                db = FloxyDatabase.get_instance()
                asset_guid = row["cloud_asset_guid"]
                if db.photo_exists(asset_guid):
                    desc = db.get_photo_description(asset_guid) or row['filename']
                    with lock:
                        succeeded.append(desc)
                else:
                    with lock:
                        failed.append(row['filename'])
            except Exception as e:
                error(f"Photos: worker failed for {row.get('uuid', '?')}: {e}")
                with lock:
                    failed.append(row['filename'])
            if on_progress:
                on_progress(len(succeeded), len(failed), len(rows))

        semaphore = threading.Semaphore(photo_threads)
        video_semaphore = threading.Semaphore(VIDEO_THREADS)

        def throttled_worker(row):
            with video_semaphore if row.get("kind") == "video" else semaphore:
                worker(row)

        threads = [threading.Thread(target=throttled_worker, args=(row,), daemon=True) for row in rows]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total = len(succeeded) + len(failed)
        elapsed = time.time() - t0
        avg = elapsed / total if total > 0 else 0
        with self._icloud_lock:
            pending_count = len(self._pending_icloud)
        self._refused_last_run = refused
        info(f"Photos: processing done — {len(succeeded)}/{total} succeeded, "
             f"{elapsed:.1f}s total, {avg:.1f}s avg/asset"
             + (f", {pending_count} awaiting iCloud download" if pending_count else "")
             + (f", {len(refused)} refused by the backend and left in the queue" if refused else ""))
        return succeeded, failed

    def refused_last_run(self) -> list[str]:
        """The uuids the indexing backend refused (402 / 429) in the last `process_rows`: nothing
        was stored for them, and the feed must not move its cursor past them."""
        return list(self._refused_last_run)

    @staticmethod
    def _batch_enabled() -> bool:
        from common.ConfigManager import ConfigManager
        return bool(ConfigManager.get_instance().get('feeds.photos.backfill_batch', BACKFILL_BATCH))

    def _process_stills_batched(self, rows: list) -> dict[str, bool]:
        """The backfill's stills through the batch API. Every row is prepared (downloaded, its
        context read) on the usual fan-out, the prepared ones go out as one batch, and each
        answered photo is stored. Returns the rows this settled, uuid -> "needs an iCloud retry"
        (False for stored or skipped); a row that is missing — the batch did not answer it, its
        reply did not parse, there is no batch API — is left for the live path, which still has
        the full fan-out for it."""
        settled: dict[str, bool] = {}
        prepared: list[_PreparedPhoto] = []
        lock = threading.Lock()
        db = FloxyDatabase.get_instance()

        def prepare(row):
            uuid = row["uuid"]
            with PhotoProcessor._processing_lock:
                if uuid in PhotoProcessor._processing_uuids:
                    trace(f"Photos: {uuid} already being processed by another thread, skipping")
                    with lock:
                        settled[uuid] = False
                    return
                PhotoProcessor._processing_uuids.add(uuid)
            try:
                photo, retry = self._prepare_photo(row, db)
            except Exception as e:
                error(f"Photos: preparing {uuid} for the batch failed: {e}")
                photo, retry = None, False
            with lock:
                if photo is None:
                    settled[uuid] = retry
                    with PhotoProcessor._processing_lock:
                        PhotoProcessor._processing_uuids.discard(uuid)
                else:
                    prepared.append(photo)    # stays marked as processing until stored or handed back

        semaphore = threading.Semaphore(PHOTO_THREADS)

        def throttled(row):
            with semaphore:
                prepare(row)

        threads = [threading.Thread(target=throttled, args=(row,), daemon=True) for row in rows]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if not prepared:
            trace(f"Photos: nothing to batch — {len(settled)} row(s) settled while preparing")
            return settled

        size_mb = sum(len(p.jpeg) for p in prepared) / 1024 / 1024
        info(f"Photos: {len(prepared)} photo(s) ({size_mb:.1f} MB) go to the batch API, "
             f"{len(settled)} settled while preparing")
        t0 = time.time()
        llm = self._get_llm(backfill=True)
        system = self._photo_system_prompt(llm)
        try:
            replies = llm.think_batched_with_images(
                system, [(p.user_message, p.jpeg) for p in prepared], max_wait_seconds=BATCH_MAX_WAIT_SECONDS)
        except Exception as e:
            error(f"Photos: the batch failed ({e}) — these photos are indexed live")
            replies = None
        if replies is None:
            info(f"Photos: no batch answer for {len(prepared)} photo(s) — indexing them live")
            replies = [None] * len(prepared)

        stored = unanswered = unparsed = 0
        for photo, reply in zip(prepared, replies):
            uuid = photo.row["uuid"]
            result = None
            if reply:
                try:
                    result = self._parse_analysis(reply)
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    trace(f"Photos: batch reply for {uuid} did not parse ({e}), indexing it live")
                    unparsed += 1
            else:
                unanswered += 1
            try:
                if result is not None:
                    photo.started = time.time()
                    self._finish_photo(db, photo, result)
                    stored += 1
                    settled[uuid] = False
            except Exception as e:
                error(f"Photos: storing the batch result for {uuid} failed: {e}")
                settled[uuid] = False
            finally:
                with PhotoProcessor._processing_lock:
                    PhotoProcessor._processing_uuids.discard(uuid)
        info(f"Photos: batch done in {time.time() - t0:.0f}s — {stored} stored, "
             f"{unanswered} unanswered and {unparsed} unparsed go to the live path")
        return settled

    def _retry_icloud_pending(self):
        with self._icloud_lock:
            pending = dict(self._pending_icloud)
        if not pending:
            return
        reason = self.provider_unavailable()
        if reason:
            trace(f"Photos: {len(pending)} asset(s) awaiting iCloud stay in the retry set — {reason}")
            return

        retained = {}
        for uuid, (row, retries) in pending.items():
            if retries >= MAX_ICLOUD_RETRIES:
                info(f"Photos: giving up on iCloud download for {uuid} after {retries} retries, marking as skipped")
                db = FloxyDatabase.get_instance()
                asset_guid = row["cloud_asset_guid"]
                if asset_guid and not db.photo_exists(asset_guid):
                    self._store_skipped(db, asset_guid, row["creation_ts"], row, reason="iCloud download failed")
                continue
            try:
                still_needs = self._process_one(row)
                if still_needs:
                    retained[uuid] = (row, retries + 1)
            except LLMUnavailable as e:
                # The download may have worked, the description was refused: not a verdict on
                # the asset, so it keeps its place and its retry count.
                trace(f"Photos: iCloud asset {uuid} kept for the next round — {e}")
                retained[uuid] = (row, retries)
            except CorruptImageError:
                pass  # already stored as skipped inside _process_one
            except Exception as e:
                error(f"Photos: failed to process iCloud asset {uuid}: {e}")
                retained[uuid] = (row, retries + 1)

        with self._icloud_lock:
            self._pending_icloud = retained

    @staticmethod
    def _store_skipped(db, asset_guid: str, creation_ts: float, row: dict, reason: str = "too large"):
        db.store_photo(
            asset_guid=asset_guid, creation_date=creation_ts,
            latitude=row["latitude"], longitude=row["longitude"],
            location_text="", albums="", persons="",
            photo_type="skipped", short_description=f"Skipped ({reason})",
            detailed_description="", pretty_score=0, family_importance_score=0,
            indexed_at=time.time(), duration_seconds=row.get("duration"),
        )

    def _store_analysis(self, db, row: dict, result: dict, media_type: str, location_text: str,
                        albums: str, persons: str, pretty: int, family: int) -> Optional[int]:
        """Store the LLM result and, if this thread won the insert, the history entry.
        Returns the photo id, or None when another thread stored the asset first."""
        uuid = row["uuid"]
        lat = row["latitude"]
        lon = row["longitude"]
        is_video = row.get("kind") == "video"
        duration = row.get("duration") if is_video else None

        photo_id = db.store_photo(
            asset_guid=row["cloud_asset_guid"],
            creation_date=row["creation_ts"],
            latitude=lat,
            longitude=lon,
            location_text=location_text,
            albums=albums,
            persons=persons,
            photo_type=media_type,
            short_description=result["short_description"],
            detailed_description=result["detailed_description"],
            pretty_score=pretty,
            family_importance_score=family,
            indexed_at=time.time(),
            duration_seconds=duration,
        )

        # store_photo returns a real id only for the thread that actually inserted the row
        # (INSERT OR IGNORE). Threads that lost a concurrent race get None — they must not
        # write a history entry, or the photo shows up multiple times in the prompt context.
        if not photo_id:
            trace(f"Photos: {uuid} already stored by another thread, skipping history entry")
            return None

        meta_parts = [p for p in [persons, location_text,
                      f"{lat:.5f},{lon:.5f}" if lat is not None and lon is not None else "",
                      media_type if media_type not in ("photo", "video") else "",
                      f"{duration:.0f}s" if duration else ""] if p]
        meta = f" ({', '.join(meta_parts)})" if meta_parts else ""
        id_tag = f" [photo_id={photo_id}]"
        spoken = truncate_spoken_content(result.get("spoken_content"))
        history_content = (
            f"New {'video' if is_video else 'photo'} taken: {result['short_description']}{meta}{id_tag}\n\n"
            f"{result['detailed_description']}"
            + (f"\n\nSpoken: {spoken}" if spoken else "")
        )
        self._emit_entry(db, FeedEntry(row["creation_ts"], history_content))
        return photo_id

    def _emit_entry(self, db, entry: FeedEntry):
        if self._entry_sink is not None:
            self._entry_sink(entry)
        else:
            db.add_history_entry(entry.role, entry.content, entry.timestamp, source=PHOTO_HISTORY_SOURCE)

    def _process_one(self, row: dict) -> bool:
        """Process a single photo. Returns True if photo needs iCloud download retry."""
        uuid = row["uuid"]

        with PhotoProcessor._processing_lock:
            if uuid in PhotoProcessor._processing_uuids:
                trace(f"Photos: {uuid} already being processed by another thread, skipping")
                return False
            PhotoProcessor._processing_uuids.add(uuid)
        try:
            # An LLMUnavailable passes through: it is not a verdict on the asset, and each caller
            # keeps the row where it was (the batch's refused list, the iCloud retry set).
            return self._process_one_inner(row)
        finally:
            with PhotoProcessor._processing_lock:
                PhotoProcessor._processing_uuids.discard(uuid)

    def _process_one_inner(self, row: dict) -> bool:
        uuid = row["uuid"]
        asset_guid = row["cloud_asset_guid"]
        db = FloxyDatabase.get_instance()

        if not asset_guid:
            fine(f"Photos: {uuid} has no iCloud asset GUID yet (not uploaded), deferring indexing")
            return False

        if db.photo_exists(asset_guid):
            return False

        if row.get("kind") == "video":
            return self._process_video(row, db)

        photo, retry = self._prepare_photo(row, db)
        if photo is None:
            return retry
        PhotoProcessor._worker.queued_s = 0.0
        result = self._analyze_with_llm(photo.jpeg, self._home_time(row["creation_ts"]), photo.location_text,
                                        photo.albums, photo.persons, photo.photo_type,
                                        backfill=bool(row.get("backfill")))
        if result is None:
            error(f"Photos: {uuid} permanently failed LLM analysis, skipping")
            self._store_skipped(db, asset_guid, row["creation_ts"], row, reason="LLM analysis failed")
            return False
        self._finish_photo(db, photo, result)
        return False

    def _prepare_photo(self, row: dict, db) -> tuple[Optional[_PreparedPhoto], bool]:
        """The still's bytes and context, or (None, retry): True when its original is still
        coming from iCloud and the row must be tried again later, False when it was stored as
        skipped or cannot be indexed at all."""
        uuid = row["uuid"]
        asset_guid = row["cloud_asset_guid"]
        original_path = os.path.join(PHOTOS_ORIGINALS_DIR, row["directory"], row["filename"])
        placeholder = self._icloud_placeholder_type(original_path)

        jpeg_bytes = None
        if placeholder:

            try:
                jpeg_bytes = self._try_icloud_download(row)
            except CorruptImageError as e:
                error(f"Photos: {uuid} has permanently corrupt image data: {e}")
                self._store_skipped(db, asset_guid, row["creation_ts"], row, reason="corrupt image")
                return None, False
            except PhotoNotInCloudError:
                info(f"Photos: {uuid} not found in any iCloud zone, skipping permanently")
                self._store_skipped(db, asset_guid, row["creation_ts"], row, reason="not in iCloud library")
                return None, False
            if jpeg_bytes:
                fine(f"Photos: {uuid} downloaded via iCloud API ({placeholder})")
            else:
                trace("iCloud API failed — fall back to brctl/retry for dataless/evicted")
                if placeholder == "dataless":
                    self._trigger_icloud_download(original_path)
                    trace(f"Photos: {uuid} is iCloud-only (dataless), triggered download")
                    return None, True
                if placeholder == "evicted":
                    self._trigger_icloud_download_brctl(original_path)
                    trace(f"Photos: {uuid} not on disk yet (evicted), triggered brctl download, will retry")
                    return None, True
                if placeholder == "truncated":
                    # The file is "local" as far as iCloud is concerned, so no download to
                    # trigger — only the API can supply the bytes, try again later.
                    trace(f"Photos: {uuid} original is truncated on disk and the iCloud API failed, will retry")
                    return None, True
                # stub — no brctl fallback
                trace(f"Photos: {uuid} is iCloud-only (.icloud stub), cannot download")
                return None, False

        t0 = time.time()
        creation_ts = row["creation_ts"]
        photo_type = self._describe_subtypes(row["kind_subtype"])
        albums = self._get_albums(uuid)
        persons = self._get_persons(uuid)
        location_text = self._get_location(uuid)

        if jpeg_bytes is None:
            jpeg_bytes = self._get_image_bytes(row["directory"], row["filename"])
            if jpeg_bytes is None:
                trace(f"Photos: skipping {uuid}, could not extract image")
                return None, False

        max_size = 5 * 1024 * 1024
        if len(jpeg_bytes) > max_size:
            info(f"Photos: skipping {uuid}, image too large ({len(jpeg_bytes) / 1024 / 1024:.1f} MB)")
            self._store_skipped(db, asset_guid, creation_ts, row)
            return None, False

        user_message = self._photo_user_prompt(self._home_time(creation_ts), location_text, albums, persons, photo_type)
        return _PreparedPhoto(row=row, jpeg=jpeg_bytes, photo_type=photo_type, albums=albums, persons=persons,
                              location_text=location_text, user_message=user_message, started=t0), False

    def _finish_photo(self, db, photo: _PreparedPhoto, result: dict):
        """Store an analysed still and log it; `photo.started` dates the timing line."""
        row = photo.row
        uuid = row["uuid"]
        pretty = result["pretty_score"]
        family = result["family_importance_score"]

        if "screenshot" in photo.photo_type:
            pretty //= 2
            family //= 2

        if not self._store_analysis(db, row, result, photo.photo_type, photo.location_text, photo.albums,
                                    photo.persons, pretty, family):
            return

        lat = row["latitude"]
        lon = row["longitude"]
        elapsed = time.time() - photo.started
        coords = f"[{lat:.5f},{lon:.5f}]" if lat is not None and lon is not None else "no coords"
        # A local backend serves one image at a time, so a worker's clock includes the time its
        # photo queued behind the other worker's — say so, or every photo looks twice as slow.
        queued = getattr(PhotoProcessor._worker, "queued_s", 0.0)
        timing = f"in {elapsed:.1f}s" + (f" ({queued:.1f}s of it queued for the model)" if queued >= 1 else "")
        line = (f"Photo processed '{result['short_description']}' ({photo.photo_type}, [{photo.persons}], "
                f"{photo.location_text}, {coords}) {timing} ({uuid}).")
        if elapsed > 10:
            info(line)
        else:
            trace(line)

    def _process_video(self, row: dict, db) -> bool:
        """The video half of _process_one_inner. Returns True when the download should be
        retried later (transient iCloud failure), False when the clip is stored or dropped."""
        uuid = row["uuid"]
        asset_guid = row["cloud_asset_guid"]
        creation_ts = row["creation_ts"]
        duration = row.get("duration") or 0.0
        settings = video_settings()

        if duration > settings.max_duration_seconds:
            info(f"Photos: skipping video {uuid}, {duration:.0f}s is over the "
                 f"{settings.max_duration_seconds:.0f}s limit")
            self._store_skipped(db, asset_guid, creation_ts, row, reason="video too long")
            return False

        llm = self._get_llm(bool(row.get("backfill")))
        if not llm.supports_video():
            llm = self._clip_llm()
            if llm is None:
                # Not stored as skipped: only the backends leave the clip unindexed, and a configured one may take it.
                trace(f"Photos: {uuid} is a video and no backend at hand takes video input, leaving it unindexed")
                return False

        try:
            downloaded = self._get_video_bytes(row, settings.max_bytes)
        except VideoTooLargeError as e:
            info(f"Photos: skipping video {uuid} ({duration:.0f}s): {e}")
            self._store_skipped(db, asset_guid, creation_ts, row, reason="video too large")
            return False
        except PhotoNotInCloudError:
            info(f"Photos: video {uuid} not found in any iCloud zone, skipping permanently")
            self._store_skipped(db, asset_guid, creation_ts, row, reason="not in iCloud library")
            return False
        except NoVideoRenditionError as e:
            info(f"Photos: skipping video {uuid} ({duration:.1f}s) permanently, {e}")
            self._store_skipped(db, asset_guid, creation_ts, row, reason="no video rendition in iCloud")
            return False
        if downloaded is None:
            trace(f"Photos: video {uuid} could not be downloaded from iCloud, will retry")
            return True
        video_bytes, mime_type = downloaded

        t0 = time.time()
        lat = row["latitude"]
        lon = row["longitude"]
        video_type = self._describe_video_subtype(row["kind_subtype"])
        albums = self._get_albums(uuid)
        persons = self._get_persons(uuid)
        location_text = self._get_location(uuid)
        date_str = self._home_time(creation_ts)

        result = self._analyze_video_with_llm(video_bytes, mime_type, date_str, location_text,
                                              albums, persons, video_type, duration, llm=llm)
        if result is None:
            error(f"Photos: video {uuid} permanently failed LLM analysis, skipping")
            self._store_skipped(db, asset_guid, creation_ts, row, reason="LLM analysis failed")
            return False

        if not self._store_analysis(db, row, result, video_type, location_text, albums, persons,
                                    result["pretty_score"], result["family_importance_score"]):
            return False

        elapsed = time.time() - t0
        trace(f"video processed '{result['short_description']}' ({video_type}, {duration:.0f}s, "
              f"[{persons}], {location_text}) in {elapsed:.1f}s ({uuid}).")
        return False

    @staticmethod
    def _get_video_bytes(row: dict, max_bytes: int) -> Optional[tuple[bytes, str]]:
        """A clip small enough for the video LLM: the local original when it happens to be on
        disk and fits, otherwise an iCloud rendition (720p, else 360p). Never triggers a
        download of the original into the library — those are hundreds of MB."""
        uuid = row["uuid"]
        original_path = os.path.join(PHOTOS_ORIGINALS_DIR, row["directory"], row["filename"])
        if PhotoProcessor._icloud_placeholder_type(original_path) is None:
            size = os.path.getsize(original_path)
            if size <= max_bytes:
                fine(f"Photos: video {uuid} original is local and fits ({size / 1024 / 1024:.1f} MB)")
                with open(original_path, "rb") as f:
                    return f.read(), mimetypes.guess_type(row["filename"])[0] or "video/mp4"
            trace(f"Photos: video {uuid} original is local but {size / 1024 / 1024:.1f} MB, using an iCloud rendition")

        record_name = row.get("cloud_master_guid") or row.get("cloud_asset_guid")
        if not record_name:
            trace(f"Photos: video {uuid} has no cloud GUID, cannot use iCloud API")
            return None
        from .ICloudPhotoDownloader import ICloudPhotoDownloader
        return ICloudPhotoDownloader.get_instance().download_video_bytes(
            record_name, row.get("library_scope") or 0, max_bytes=max_bytes,
            cloud_asset_guid=row.get("cloud_asset_guid"))

    @staticmethod
    def _describe_video_subtype(subtype: int) -> str:
        label = VIDEO_SUBTYPE_LABELS.get(subtype or 0)
        return f"video, {label}" if label else "video"

    def _analyze_video_with_llm(self, video_bytes: bytes, mime_type: str, date_str: str, location: str,
                                albums: str, persons: str, video_type: str, duration: float,
                                llm: Optional[AbstractLLM] = None) -> Optional[dict]:
        llm = llm or self._get_llm()
        system = self._for_backend(VIDEO_ANALYSIS_PROMPT_SYSTEM.format(
            current_memory=self._escape_braces(self._read_memory() or "(no memory available)")), llm)
        user = VIDEO_ANALYSIS_PROMPT_USER.format(
            date=date_str,
            location=self._escape_braces(location or "unknown"),
            albums=self._escape_braces(albums or "none"),
            persons=self._escape_braces(persons or "none"),
            video_type=video_type,
            duration=f"{duration:.0f}",
        )
        fps = short_clip_fps(duration)
        if fps:
            info(f"Photos: {duration:.1f}s clip is too short for Gemini's 1 frame/s default, sampling at {fps:.0f} fps")

        for attempt in range(2):
            try:
                response = llm.think_with_video(
                    system_message=system,
                    user_message=user,
                    video_bytes=video_bytes,
                    mime_type=mime_type,
                    fps=fps,
                )
                return self._parse_analysis(response)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                if attempt == 0:
                    trace(f"Photos: video LLM response parse failed, retrying: {e}")
                    continue
                error(f"Photos: video LLM analysis failed after retry: {e}")
                return None
            except LLMUnavailable:
                raise    # the provider, not the clip: the caller leaves it in the queue
            except Exception as e:
                error(f"Photos: video LLM call failed: {e}")
                return None
        return None

    @staticmethod
    def _try_icloud_download(row: dict) -> Optional[bytes]:
        """Try to download photo bytes via iCloud API. Returns JPEG bytes or None.
        Raises CorruptImageError if the image data is permanently corrupt."""
        record_name = row.get("cloud_master_guid") or row.get("cloud_asset_guid")
        if not record_name:
            trace(f"Photos: {row.get('uuid', '?')} has no cloud GUID, cannot use iCloud API")
            return None
        try:
            from .ICloudPhotoDownloader import ICloudPhotoDownloader
            library_scope = row.get("library_scope") or 0
            return ICloudPhotoDownloader.get_instance().download_photo_bytes(record_name, library_scope)
        except (CorruptImageError, PhotoNotInCloudError):
            raise
        except Exception as e:
            trace(f"Photos: iCloud API download failed for {row.get('uuid', '?')}: {e}")
            return None

    @staticmethod
    def _describe_subtypes(subtypes: int) -> str:
        subtypes = subtypes or 0
        labels = []
        for flag, label in MEDIA_SUBTYPE_FLAGS.items():
            if subtypes & flag:
                labels.append(label)
        seen = set()
        unique = []
        for label in labels:
            if label not in seen:
                seen.add(label)
                unique.append(label)
        return ", ".join(unique) if unique else "photo"

    @staticmethod
    def _get_albums(uuid: str) -> str:
        try:
            if not os.path.exists(PHOTOS_SQLITE_PATH):
                return ""
            conn = sqlite3.connect(f"file:{PHOTOS_SQLITE_PATH}?mode=ro", uri=True)
            try:
                cursor = conn.cursor()
                cursor.execute(ALBUMS_QUERY, (uuid,))
                names = [row[0] for row in cursor.fetchall()]
                cursor.execute("SELECT ZFAVORITE FROM ZASSET WHERE ZUUID = ?", (uuid,))
                fav_row = cursor.fetchone()
                if fav_row and fav_row[0] == 1:
                    names.insert(0, "Favorites")
                return ", ".join(names)
            finally:
                conn.close()
        except Exception as e:
            trace(f"Photos: could not fetch albums: {e}")
            return ""

    @staticmethod
    def _get_persons(uuid: str) -> str:
        try:
            if not os.path.exists(PHOTOS_SQLITE_PATH):
                return ""
            conn = sqlite3.connect(f"file:{PHOTOS_SQLITE_PATH}?mode=ro", uri=True)
            try:
                cursor = conn.cursor()
                cursor.execute(PERSONS_QUERY, (uuid,))
                names = [row[0] for row in cursor.fetchall()]
                return ", ".join(names)
            finally:
                conn.close()
        except Exception as e:
            trace(f"Photos: could not fetch persons: {e}")
            return ""

    @staticmethod
    def _get_location(uuid: str) -> str:
        """The place Photos assigned to the asset, "" when it has none."""
        try:
            if not os.path.exists(PHOTOS_SQLITE_PATH):
                return ""
            conn = sqlite3.connect(f"file:{PHOTOS_SQLITE_PATH}?mode=ro", uri=True)
            try:
                row = conn.execute(LOCATION_QUERY, (uuid,)).fetchone()
            finally:
                conn.close()
            if row is None:
                return ""
            location = apple_location(row[0], row[1])
            if not location:
                trace(f"Photos: {uuid} has no place in Photos (no moment title or reverse-location data)")
            return location
        except Exception as e:
            trace(f"Photos: could not fetch location for {uuid}: {e}")
            return ""

    @staticmethod
    def _icloud_placeholder_type(path: str) -> Optional[str]:
        """Returns 'dataless' for Sonoma+ dataless files, 'stub' for pre-Sonoma .icloud stubs,
        'evicted' if file is absent (iCloud-only, never downloaded), 'truncated' for a present
        file too small to be a photo (a broken download), None if local."""
        if os.path.exists(path):
            try:
                st = os.lstat(path)
                if st.st_flags & SF_DATALESS:
                    return "dataless"
                if st.st_size < MIN_ORIGINAL_BYTES:
                    trace(f"Photos: {os.path.basename(path)} is only {st.st_size} bytes on disk, treating as truncated")
                    return "truncated"
            except OSError:
                pass
            return None
        stub = os.path.join(os.path.dirname(path), f".{os.path.basename(path)}.icloud")
        if os.path.exists(stub):
            return "stub"
        return "evicted"

    @staticmethod
    def _trigger_icloud_download(path: str):
        """Trigger iCloud materialization by reading the file in a background thread."""
        def _download():
            try:
                with open(path, "rb") as f:
                    f.read(1)
            except Exception:
                pass
        threading.Thread(target=_download, name="iCloudDownload", daemon=True).start()

    @staticmethod
    def _trigger_icloud_download_brctl(path: str):
        """Trigger iCloud download for evicted files (no local data) using brctl."""
        try:
            subprocess.run(["brctl", "download", path], capture_output=True, timeout=30)
        except Exception as e:
            trace(f"Photos: brctl download failed for {path}: {e}")

    # Where a library asset's file is and what iCloud knows it as, by its cloud asset GUID
    ASSET_FILE_SQL = """SELECT a.ZUUID, a.ZFILENAME, a.ZDIRECTORY, m.ZCLOUDMASTERGUID,
                               a.ZCLOUDASSETGUID, a.ZLIBRARYSCOPE
                        FROM ZASSET a
                        LEFT JOIN ZCLOUDMASTER m ON m.Z_PK = a.ZMASTER
                        WHERE a.ZCLOUDASSETGUID = ?"""

    @staticmethod
    def image_bytes_for_asset(asset_guid: str) -> Optional[bytes]:
        """The JPEG of library asset `asset_guid` (1024 px, fetched from iCloud when it is not
        on disk), or None when the library does not know the asset or its bytes cannot be had."""
        try:
            conn = sqlite3.connect(f"file:{PHOTOS_SQLITE_PATH}?mode=ro", uri=True)
            try:
                row = conn.execute(PhotoProcessor.ASSET_FILE_SQL, (asset_guid,)).fetchone()
            finally:
                conn.close()
        except sqlite3.Error as e:
            info(f"Photos: could not look up asset {asset_guid}: {e}")
            return None
        if not row:
            info(f"Photos: asset {asset_guid} is not in the library")
            return None
        _uuid, filename, directory, cloud_master_guid, cloud_asset_guid, library_scope = row
        return PhotoProcessor._get_image_bytes(directory, filename, cloud_master_guid=cloud_master_guid,
                                               cloud_asset_guid=cloud_asset_guid, library_scope=library_scope or 0)

    @staticmethod
    def _get_image_bytes(directory: str, filename: str,
                         cloud_master_guid: str = None, cloud_asset_guid: str = None,
                         library_scope: int = 0) -> Optional[bytes]:
        try:
            original_path = os.path.join(PHOTOS_ORIGINALS_DIR, directory, filename)
            placeholder = PhotoProcessor._icloud_placeholder_type(original_path)

            if placeholder:
                record_name = cloud_master_guid or cloud_asset_guid
                if record_name:
                    info(f"Photos: Telegram send — photo is {placeholder}, attempting iCloud download")
                    try:
                        from .ICloudPhotoDownloader import ICloudPhotoDownloader
                        jpeg_bytes = ICloudPhotoDownloader.get_instance().download_photo_bytes(
                            record_name, library_scope
                        )
                        if jpeg_bytes:
                            info(f"Photos: Telegram send — iCloud download succeeded for {filename}")
                            return jpeg_bytes
                    except Exception as e:
                        trace(f"Photos: iCloud API download failed for Telegram send: {e}")

                if placeholder in ("dataless", "evicted"):
                    trace(f"Photos: Telegram send — trying brctl download for {placeholder} file")
                    try:
                        subprocess.run(["brctl", "download", original_path], capture_output=True, timeout=30)
                        for _ in range(10):
                            time.sleep(0.5)
                            if PhotoProcessor._icloud_placeholder_type(original_path) is None:
                                break
                    except Exception as e:
                        trace(f"Photos: brctl download failed: {e}")

                if PhotoProcessor._icloud_placeholder_type(original_path) is not None:
                    trace(f"Photos: Telegram send — photo still not local after download attempts: {filename}")
                    return None

            # Convert and resize all formats to JPEG via sips
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                subprocess.run(
                    ["sips", "-s", "format", "jpeg", original_path, "--out", tmp_path, "-Z", "1024"],
                    capture_output=True, timeout=30, check=True,
                )
                with open(tmp_path, "rb") as f:
                    return f.read()
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except Exception as e:
            error(f"Photos: image extraction failed: {e}")
            return None

    _bank_is_stub: Optional[bool] = None   # the last verdict, so the per-photo read logs only a change

    @classmethod
    def _read_memory(cls) -> str:
        """The memory bank for the description prompts, or "" while the bank is still the first-run
        stub: its teaching notes and another family's examples would only mislead the vision model.
        Read per photo, so the verdict is logged when it changes, not every time."""
        try:
            bank = MemoryFile.read()
        except Exception as e:
            trace(f"Photos: could not read the memory bank: {e}")
            return ""
        stub = MemoryFile.is_placeholder(bank)
        if stub != cls._bank_is_stub:
            cls._bank_is_stub = stub
            info("Photos: the memory bank is still the first-run stub, describing without memory" if stub
                 else "Photos: the memory bank is curated, describing with it")
        return "" if stub else bank

    @staticmethod
    def _home_time(timestamp: float) -> str:
        from common.ConfigManager import ConfigManager
        home_tz = ZoneInfo(ConfigManager.get_instance().get_home_timezone())
        return datetime.fromtimestamp(timestamp, tz=home_tz).strftime("%Y-%m-%d %H:%M")

    @staticmethod
    def _escape_braces(text: str) -> str:
        return text.replace("{", "{{").replace("}", "}}")

    def _photo_system_prompt(self, llm: AbstractLLM) -> str:
        """The part of a still's prompt that is the same for every photo: the instructions and the
        memory bank. Identical text from photo to photo is what a provider's prefix or explicit
        cache keys on, so nothing per-photo may go in here."""
        return self._for_backend(ANALYSIS_PROMPT_SYSTEM.format(
            current_memory=self._escape_braces(self._read_memory() or "(no memory available)")), llm)

    def _for_backend(self, system: str, llm: AbstractLLM) -> str:
        """The prompt as `llm` can take it: whole for most, without the "thinking" field for a
        model that refuses to be asked for its reasoning (Sonnet, Opus and Fable; Haiku and Gemini
        answer it). The verdict is the backend's own, by model name, so it costs no request; the
        same text goes to a backend for every photo and clip, so its cached prefix stays stable."""
        if not llm.refuses_reasoning_requests():
            return system
        if llm.name() not in self._reasoning_logged:
            self._reasoning_logged.add(llm.name())
            info(f"Photos: {llm.name()} refuses a prompt that asks for its reasoning — indexing without the thinking field")
        return without_reasoning_field(system)

    def _photo_prompt(self, date_str: str, location: str, albums: str, persons: str,
                      photo_type: str, llm: AbstractLLM) -> tuple[str, str]:
        """(system, user) for one still on `llm`; the per-photo fields are all in the user half."""
        return self._photo_system_prompt(llm), self._photo_user_prompt(date_str, location, albums, persons, photo_type)

    def _photo_user_prompt(self, date_str: str, location: str, albums: str, persons: str, photo_type: str) -> str:
        return ANALYSIS_PROMPT_USER.format(
            date=date_str,
            location=self._escape_braces(location or "unknown"),
            albums=self._escape_braces(albums or "none"),
            persons=self._escape_braces(persons or "none"),
            photo_type=photo_type,
        )

    def _analyze_with_llm(self, jpeg_bytes: bytes, date_str: str, location: str,
                          albums: str, persons: str, photo_type: str, backfill: bool = False) -> Optional[dict]:
        llm = self._get_llm(backfill)
        system, user = self._photo_prompt(date_str, location, albums, persons, photo_type, llm)

        for attempt in range(2):
            try:
                response = self._think_background(llm, system, user, jpeg_bytes)
                return self._parse_analysis(response)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                if attempt == 0:
                    trace(f"Photos: LLM response parse failed, retrying: {e}")
                    continue
                error(f"Photos: LLM analysis failed after retry: {e}")
                return None
            except LLMUnavailable:
                raise    # the provider, not the photo: the caller leaves it in the queue
            except Exception as e:
                error(f"Photos: LLM call failed: {e}")
                return None
        return None

    @staticmethod
    def _think_background(llm: AbstractLLM, system: str, user: str, jpeg_bytes: bytes) -> str:
        """One image through the LLM as background work. A local backend evicts it when a user
        request arrives; the asset must not count as failed for that (the cursor has moved past
        it), so the same image is resent once the backend is idle again."""
        preempted = 0
        while True:
            try:
                return llm.think(system_message=system, user_message=user, images=[jpeg_bytes], background=True)
            except LLMPreempted:
                preempted += 1
                trace(f"Photos: image analysis preempted by a user request ({preempted}x), resending when idle")
            finally:
                queued = getattr(PhotoProcessor._worker, "queued_s", 0.0)
                PhotoProcessor._worker.queued_s = queued + llm.background_wait_seconds()

    @staticmethod
    def _parse_analysis(response: str) -> dict:
        data = parse_description_json(
            response, ("short_description", "detailed_description", "pretty_score", "family_importance_score"),
            "Photos")
        data["pretty_score"] = int(data["pretty_score"])
        data["family_importance_score"] = int(data["family_importance_score"])
        if data.get("thinking"):
            fine(f"  AI thinking ({data["short_description"]}): p:{data["pretty_score"]} f:{data["family_importance_score"]} {data['thinking']}")
        return data
