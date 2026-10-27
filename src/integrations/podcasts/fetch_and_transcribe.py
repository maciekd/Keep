"""The swappable source of the podcast pipeline: a YouTube playlist in, an episode bundle out.

This module is the one unit to replace when yt-dlp and Whisper give way to another source. Its
contract is `fetch_next(playlist_url, is_processed, out_root, slug_prefix)`: find the newest
episode `is_processed(video_id, slug)` says is not done, and write its bundle to
`out_root/<slug>/` —
    video_meta.json      video_id, slug, title, uploader, upload_date, duration_s, video_url,
                         chapters, transcript_source and timings (download_s, transcribe_s)
    transcript_raw.json  [{text, start, duration}], one entry per sentence; a `>>` opens a speaker turn
    chapters/NN_M-SS.txt one file per chapter: CHAPTER: / TIME: header, then ~60 s timestamped blocks
— and return the bundle's directory. Everything after it (generate_report.py) reads only the bundle.

The audio is YouTube's best audio-only stream (the podcast RSS has dynamically inserted ads, so
its timeline would not match the chapter markers), transcribed with mlx-whisper in a subprocess
(transcribe_audio.py) so the model's memory is freed when it is done, then deleted. YouTube's
auto-captions are used only for their `>>` speaker-turn markers, merged onto the Whisper word
timeline; without mlx-whisper, or when it fails, the captions are the transcript.

    python fetch_and_transcribe.py --playlist URL --state processed.json --out DIR [--prefix moonshots]
"""
import argparse
import bisect
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

try:
    from common.logging import error, info, note, trace
except ImportError:     # run as a script outside the server
    error = info = note = trace = lambda message: print(message, file=sys.stderr)

WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
WHISPER_TIMEOUT_S = 60 * 60
RECENT_WINDOW = 5
TRANSCRIPT_LANGS = ["en"]
BLOCK_SECONDS = 60
FALLBACK_SEGMENT_SECONDS = 15 * 60
SOURCE_WHISPER_TURNS = "whisper+youtube-turns"
SOURCE_WHISPER = "whisper"
SOURCE_YOUTUBE = "youtube"

EPISODE_NUM_RE = re.compile(r'(?:EP\s*#?|#)\s*(\d{2,5})\b', re.IGNORECASE)
SENTENCE_END_RE = re.compile(r'[.?!]["\')\]]*$')


class ToolMissing(Exception):
    """A tool the source needs is not installed (yt-dlp)."""


class NoTranscriptAvailable(Exception):
    """Neither YouTube captions nor a local transcription could be had for an episode."""


@dataclass(frozen=True)
class Candidate:
    video_id: str
    title: str
    slug: str


def episode_slug(prefix: str, title: str, video_id: str) -> str:
    """'...| EP #282' / '...| #281' / '... 280' -> 'moonshots-282'; the video id when the title has no number."""
    match = EPISODE_NUM_RE.search(title or "") or re.search(r'(\d{2,5})\s*$', (title or "").strip())
    return f"{prefix}-{match.group(1) if match else video_id}"


def _yt_dlp():
    try:
        import yt_dlp
    except ImportError as e:
        raise ToolMissing("yt-dlp is not installed — pip install -r src/integrations/podcasts/requirements.txt") from e
    return yt_dlp


def _ydl_opts(**opts) -> dict:
    """yt-dlp options plus a JavaScript runtime when one is on PATH: YouTube extraction without one
    is deprecated and already loses formats, and yt-dlp only looks for deno by itself."""
    for name in ("deno", "node", "bun"):
        path = shutil.which(name)
        if path:
            opts["js_runtimes"] = {name: {"path": path}}
            break
    return opts


def recent_episodes(playlist_url: str, slug_prefix: str, n: int = RECENT_WINDOW) -> list[Candidate]:
    """The newest `n` entries of the playlist, newest first."""
    with _yt_dlp().YoutubeDL(_ydl_opts(extract_flat=True, playlist_items=f"1-{n}", quiet=True,
                                       skip_download=True)) as ydl:
        listing = ydl.extract_info(playlist_url, download=False)
    entries = [e for e in (listing.get("entries") or []) if e and e.get("id")]
    return [Candidate(e["id"], e.get("title") or "", episode_slug(slug_prefix, e.get("title") or "", e["id"]))
            for e in entries]


def fetch_next(playlist_url: str, is_processed: Callable[[str, str], bool], out_root: Path,
               slug_prefix: str, progress: Callable[[str], None] = lambda step: None,
               on_found: Callable[[Candidate], None] = lambda candidate: None) -> Optional[Path]:
    """The bundle of the newest episode of the playlist's recent window that is not processed; None
    when every one is. `on_found` hears of the episode before its download starts. Raises
    NoTranscriptAvailable when the chosen episode has no transcript."""
    candidates = recent_episodes(playlist_url, slug_prefix)
    pending = [c for c in candidates if not is_processed(c.video_id, c.slug)]
    if not pending:
        info(f"Podcasts: all {len(candidates)} recent episodes of {playlist_url} are processed — nothing new")
        return None
    chosen = pending[0]
    info(f"Podcasts: new episode {chosen.slug} ({chosen.video_id}): {chosen.title}")
    on_found(chosen)
    return fetch_episode(chosen.video_id, out_root, slug_prefix, progress)


def fetch_episode(video_id: str, out_root: Path, slug_prefix: str,
                  progress: Callable[[str], None] = lambda step: None) -> Path:
    """Writes the bundle of one video to `out_root/<slug>/` and returns that directory."""
    meta = _video_meta(video_id)
    meta["slug"] = episode_slug(slug_prefix, meta["title"], video_id)
    bundle = Path(out_root) / meta["slug"]
    bundle.mkdir(parents=True, exist_ok=True)

    captions = _captions(video_id)
    if captions is not None:
        _write_json(bundle / "transcript_youtube.json", captions)

    timings: dict[str, float] = {}
    transcript, source = None, None
    if importlib.util.find_spec("mlx_whisper") is None:
        error("Podcasts: mlx-whisper is not installed — the transcript falls back to the YouTube captions")
    else:
        transcript, source = _whisper_transcript(video_id, bundle, captions, timings, progress)
    if transcript is None:
        if captions is None:
            raise NoTranscriptAvailable(f"no YouTube captions and no Whisper transcript for {video_id}")
        transcript, source = captions, SOURCE_YOUTUBE

    meta["transcript_source"] = source
    meta["timings"] = timings
    _write_json(bundle / "transcript_raw.json", transcript)
    _write_json(bundle / "video_meta.json", meta)
    count = write_chapters(bundle, meta, transcript)
    info(f"Podcasts: bundle {bundle} written — {count} chapters, transcript from {source}, "
         f"{len(transcript)} entries")
    return bundle


def _video_meta(video_id: str) -> dict:
    url = f"https://www.youtube.com/watch?v={video_id}"
    with _yt_dlp().YoutubeDL(_ydl_opts(quiet=True, skip_download=True)) as ydl:
        found = ydl.extract_info(url, download=False)
    return {
        "video_id": video_id,
        "title": found.get("title") or video_id,
        "uploader": found.get("uploader") or found.get("channel"),
        "upload_date": found.get("upload_date"),
        "duration_s": found.get("duration"),
        "video_url": found.get("webpage_url") or url,
        "chapters": found.get("chapters") or [],
    }


def _captions(video_id: str) -> Optional[list[dict]]:
    """YouTube's captions, for their speaker turns (and as the fallback transcript); None without any."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        error("Podcasts: youtube-transcript-api is not installed — no speaker turns and no caption fallback")
        return None
    try:
        return YouTubeTranscriptApi().fetch(video_id, languages=TRANSCRIPT_LANGS).to_raw_data()
    except Exception as e:
        note(f"Podcasts: no YouTube captions for {video_id} ({type(e).__name__}) — no speaker-turn markers")
        return None


def _whisper_transcript(video_id, bundle: Path, captions, timings: dict, progress) -> tuple[Optional[list], Optional[str]]:
    audio = None
    try:
        progress("downloading audio")
        started = time.time()
        audio = download_audio(video_id, bundle)
        timings["download_s"] = round(time.time() - started, 1)
        info(f"Podcasts: audio of {video_id} downloaded in {timings['download_s']:.0f}s ({audio.stat().st_size >> 20} MB)")
        progress("transcribing")
        started = time.time()
        words = transcribe(audio, bundle / "transcript_whisper_words.json")
        timings["transcribe_s"] = round(time.time() - started, 1)
        info(f"Podcasts: {len(words)} words transcribed in {timings['transcribe_s']:.0f}s")
        transcript = words_to_entries(merge_turn_markers(words, caption_turn_times(captions or [])))
        return transcript, SOURCE_WHISPER_TURNS if captions else SOURCE_WHISPER
    except Exception as e:
        error(f"Podcasts: Whisper transcription of {video_id} failed, falling back to the captions: "
              f"{type(e).__name__}: {e}")
        return None, None
    finally:
        if audio is not None:
            audio.unlink(missing_ok=True)
            trace(f"Podcasts: audio {audio.name} deleted")


def download_audio(video_id: str, dest_dir: Path) -> Path:
    url = f"https://www.youtube.com/watch?v={video_id}"
    with _yt_dlp().YoutubeDL(_ydl_opts(quiet=True, noprogress=True, format="bestaudio",
                                       outtmpl=str(dest_dir / "audio.%(ext)s"))) as ydl:
        downloaded = ydl.extract_info(url, download=True)
        return Path(ydl.prepare_filename(downloaded))


def transcribe(audio: Path, out_path: Path) -> list[dict]:
    subprocess.run([sys.executable, str(Path(__file__).with_name("transcribe_audio.py")), str(audio),
                    str(out_path), WHISPER_MODEL], check=True, timeout=WHISPER_TIMEOUT_S,
                   capture_output=True, text=True)
    return json.loads(out_path.read_text(encoding="utf-8"))


def caption_turn_times(captions: list[dict]) -> list[float]:
    """The estimated audio time of every `>>` in the captions. A marker mid-entry is placed by its
    character offset across the entry's on-screen span (until the next entry starts): captions roll,
    so `duration` overlaps the next entry and is not the speaking span."""
    times = []
    for i, entry in enumerate(captions):
        text = entry["text"]
        if ">>" not in text:
            continue
        following = captions[i + 1]["start"] if i + 1 < len(captions) else entry["start"] + entry["duration"]
        span = max(following - entry["start"], 0.0)
        times += [entry["start"] + span * m.start() / max(len(text), 1) for m in re.finditer(r">>", text)]
    return times


def merge_turn_markers(words: list[dict], turn_times: list[float], max_snap_s: float = 3.0,
                       sentence_window_s: float = 1.5) -> list[dict]:
    """Puts a `>>` before the Whisper word where each caption turn begins: the nearest sentence start
    within `sentence_window_s`, else the nearest word within `max_snap_s`, else the turn is dropped.
    Two turns never share a word — quick back-and-forth puts them under a second apart."""
    starts = [w["start"] for w in words]
    marked: set[int] = set()
    for t in turn_times:
        candidates = range(max(bisect.bisect_left(starts, t - max_snap_s), 1),
                           bisect.bisect_right(starts, t + max_snap_s))
        if not candidates:
            continue
        at_sentence = [i for i in candidates if abs(starts[i] - t) <= sentence_window_s
                       and SENTENCE_END_RE.search(words[i - 1]["word"].strip())]
        ranked = sorted(at_sentence, key=lambda i: abs(starts[i] - t)) + sorted(candidates, key=lambda i: abs(starts[i] - t))
        free = next((i for i in ranked if i not in marked), None)
        if free is not None:
            marked.add(free)
    return [dict(w, word=" >> " + w["word"].lstrip()) if i in marked else w for i, w in enumerate(words)]


def words_to_entries(words: list[dict], max_words: int = 20) -> list[dict]:
    """Word list -> [{text, start, duration}], one entry per sentence, at most `max_words` long."""
    entries, current = [], []

    def flush():
        if current:
            entries.append({"text": "".join(w["word"] for w in current).strip(), "start": current[0]["start"],
                            "duration": round(current[-1]["end"] - current[0]["start"], 3)})
            current.clear()

    for word in words:
        if word["word"].lstrip().startswith(">>"):
            flush()
        current.append(word)
        if SENTENCE_END_RE.search(word["word"].strip()) or len(current) >= max_words:
            flush()
    flush()
    return entries


def fmt_time(seconds: float, sep: str = ":") -> str:
    """742 -> '12:22'; 7914 -> '2:11:54'."""
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}{sep}{minutes:02d}{sep}{secs:02d}" if hours else f"{minutes}{sep}{secs:02d}"


def chapters_of(meta: dict) -> list[dict]:
    """The video's chapter markers; fixed 15-minute segments for a video without any."""
    if meta.get("chapters"):
        return meta["chapters"]
    duration, segments, start = meta.get("duration_s") or 0, [], 0
    while start < duration:
        end = min(start + FALLBACK_SEGMENT_SECONDS, duration)
        segments.append({"start_time": start, "end_time": end,
                         "title": f"Segment {len(segments) + 1} ({fmt_time(start)}-{fmt_time(end)})"})
        start = end
    return segments


def group_into_blocks(entries: list[dict], block_seconds: int = BLOCK_SECONDS) -> list[tuple[float, str]]:
    blocks, block_start, texts = [], None, []
    for entry in entries:
        if block_start is None:
            block_start = entry["start"]
        elif entry["start"] - block_start >= block_seconds and texts:
            blocks.append((block_start, " ".join(texts)))
            block_start, texts = entry["start"], []
        text = entry["text"].replace("\n", " ").strip()
        if text:
            texts.append(text)
    if texts:
        blocks.append((block_start, " ".join(texts)))
    return blocks


def write_chapters(bundle: Path, meta: dict, transcript: list[dict]) -> int:
    """Writes chapters/NN_M-SS.txt, replacing the files of an earlier run; returns how many."""
    chapters_dir = bundle / "chapters"
    chapters_dir.mkdir(exist_ok=True)
    for stale in chapters_dir.glob("*.txt"):
        stale.unlink()
    written = 0
    for index, chapter in enumerate(chapters_of(meta)):
        start, end = chapter["start_time"], chapter["end_time"]
        blocks = group_into_blocks([e for e in transcript if start <= e["start"] < end])
        if not blocks:
            trace(f"Podcasts: chapter {chapter['title']!r} has no transcript, left out")
            continue
        body = "\n\n".join(f"[{fmt_time(ts)}]  {text}" for ts, text in blocks)
        (chapters_dir / f"{index:02d}_{fmt_time(start, sep='-')}.txt").write_text(
            f"CHAPTER: {chapter['title']}\nTIME: {fmt_time(start)} - {fmt_time(end)}\n\n\n{body}\n", encoding="utf-8")
        written += 1
    return written


def _write_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--playlist", required=True)
    parser.add_argument("--state", required=True, type=Path, help="processed.json ({'processed': [video ids]})")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--prefix", default="podcast")
    args = parser.parse_args()
    processed = set(json.loads(args.state.read_text()).get("processed", [])) if args.state.exists() else set()
    bundle = fetch_next(args.playlist, lambda video_id, slug: video_id in processed, args.out, args.prefix)
    print(bundle or "nothing new")


if __name__ == "__main__":
    main()
