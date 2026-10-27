import json
import sys
from unittest.mock import patch

import pytest

from integrations.podcasts import fetch_and_transcribe as source


@pytest.mark.parametrize("title, slug", [
    ("Robots Learn To Juggle | EP #412", "moonshots-412"),
    ("Robots Learn To Juggle | #87", "moonshots-87"),
    ("Robots Learn To Juggle 413", "moonshots-413"),
    ("Robots Learn To Juggle", "moonshots-abcDEF123"),
])
def test_the_slug_takes_the_episode_number_of_the_title_and_the_video_id_without_one(title, slug):
    assert source.episode_slug("moonshots", title, "abcDEF123") == slug


def words(*items):
    return [{"word": w, "start": s, "end": s + 0.4} for w, s in items]


def test_a_turn_marker_snaps_to_the_nearest_sentence_start():
    timeline = words((" Yes.", 10.0), (" I", 10.5), (" think", 11.0), (" so", 11.5))
    merged = source.merge_turn_markers(timeline, [10.9])
    assert [w["word"] for w in merged] == [" Yes.", " >> I", " think", " so"]


def test_a_turn_with_no_word_near_it_is_dropped():
    timeline = words((" Yes.", 10.0), (" I", 10.5))
    assert source.merge_turn_markers(timeline, [40.0]) == timeline


def test_two_turns_close_together_take_two_words():
    timeline = words((" Right.", 1.0), (" Yeah.", 1.5), (" Sure.", 2.0), (" Go", 2.5))
    merged = source.merge_turn_markers(timeline, [1.5, 1.6])
    assert sum(w["word"].lstrip().startswith(">>") for w in merged) == 2


def test_entries_break_at_sentences_and_at_turns():
    timeline = words((" One", 0.0), (" two.", 0.5), (" >> Three", 1.0), (" four", 1.5))
    entries = source.words_to_entries(timeline)
    assert [e["text"] for e in entries] == ["One two.", ">> Three four"]
    assert entries[1]["start"] == 1.0


def test_a_video_without_chapters_is_cut_into_fifteen_minute_segments():
    segments = source.chapters_of({"duration_s": 40 * 60})
    assert [(s["start_time"], s["end_time"]) for s in segments] == [(0, 900), (900, 1800), (1800, 2400)]


def test_the_chapter_files_are_named_by_index_and_start_and_skip_an_empty_chapter(tmp_path):
    meta = {"chapters": [{"title": "Intro", "start_time": 0, "end_time": 60},
                         {"title": "Silence", "start_time": 60, "end_time": 70},
                         {"title": "Fusion", "start_time": 70, "end_time": 4000}]}
    transcript = [{"text": "Hello.", "start": 1.0, "duration": 1}, {"text": "Plasma.", "start": 3700.0, "duration": 1}]
    (tmp_path / "chapters").mkdir()
    (tmp_path / "chapters" / "07_stale.txt").write_text("old")

    assert source.write_chapters(tmp_path, meta, transcript) == 2

    files = sorted(p.name for p in (tmp_path / "chapters").iterdir())
    assert files == ["00_0-00.txt", "02_1-10.txt"]
    fusion = (tmp_path / "chapters" / "02_1-10.txt").read_text()
    assert fusion.startswith("CHAPTER: Fusion\nTIME: 1:10 - 1:06:40") and "[1:01:40]  Plasma." in fusion


def test_fetch_next_takes_the_newest_episode_not_processed(tmp_path):
    candidates = [source.Candidate("new1", "A | EP #3", "show-3"), source.Candidate("old1", "B | EP #2", "show-2")]
    found = []
    with patch.object(source, "recent_episodes", return_value=candidates), \
            patch.object(source, "fetch_episode", return_value=tmp_path / "show-2") as fetch:
        bundle = source.fetch_next("https://example.com/pl", lambda vid, slug: slug == "show-3", tmp_path, "show",
                                   on_found=found.append)
    assert bundle == tmp_path / "show-2"
    assert fetch.call_args.args[0] == "old1"
    assert found == [candidates[1]]


def test_fetch_next_answers_none_when_everything_is_processed(tmp_path):
    with patch.object(source, "recent_episodes", return_value=[source.Candidate("v", "A", "show-v")]), \
            patch.object(source, "fetch_episode") as fetch:
        assert source.fetch_next("https://example.com/pl", lambda vid, slug: True, tmp_path, "show") is None
    fetch.assert_not_called()


def test_without_yt_dlp_the_source_says_what_is_missing():
    with patch.dict(sys.modules, {"yt_dlp": None}):
        with pytest.raises(source.ToolMissing, match="yt-dlp"):
            source.recent_episodes("https://example.com/pl", "show")


def video_meta(video_id):
    return {"video_id": video_id, "title": "Robots | EP #412", "uploader": "Example", "upload_date": "20261020",
            "duration_s": 120, "video_url": "https://www.youtube.com/watch?v=x",
            "chapters": [{"title": "All", "start_time": 0, "end_time": 120}]}


def test_without_mlx_whisper_the_captions_are_the_transcript(tmp_path):
    captions = [{"text": ">> Hello there.", "start": 0.0, "duration": 2.0}]
    with patch.object(source, "_video_meta", side_effect=video_meta), \
            patch.object(source, "_captions", return_value=captions), \
            patch.object(source.importlib.util, "find_spec", return_value=None), \
            patch.object(source, "download_audio") as download:
        bundle = source.fetch_episode("vid", tmp_path, "moonshots")
    download.assert_not_called()
    assert bundle == tmp_path / "moonshots-412"
    meta = json.loads((bundle / "video_meta.json").read_text())
    assert meta["transcript_source"] == source.SOURCE_YOUTUBE and meta["slug"] == "moonshots-412"
    assert json.loads((bundle / "transcript_raw.json").read_text()) == captions


def test_the_audio_is_deleted_after_the_transcription(tmp_path):
    def download(video_id, dest):
        audio = dest / "audio.webm"
        audio.write_bytes(b"\0" * 10)
        return audio

    whisper_words = [{"word": " Hello.", "start": 0.5, "end": 0.9}]
    with patch.object(source, "_video_meta", side_effect=video_meta), \
            patch.object(source, "_captions", return_value=[{"text": ">> Hello.", "start": 0.4, "duration": 1.0}]), \
            patch.object(source.importlib.util, "find_spec", return_value=object()), \
            patch.object(source, "download_audio", side_effect=download), \
            patch.object(source, "transcribe", return_value=whisper_words):
        bundle = source.fetch_episode("vid", tmp_path, "moonshots")
    assert not (bundle / "audio.webm").exists()
    meta = json.loads((bundle / "video_meta.json").read_text())
    assert meta["transcript_source"] == source.SOURCE_WHISPER_TURNS
    assert set(meta["timings"]) == {"download_s", "transcribe_s"}


def test_no_captions_and_no_transcription_is_no_transcript(tmp_path):
    with patch.object(source, "_video_meta", side_effect=video_meta), \
            patch.object(source, "_captions", return_value=None), \
            patch.object(source.importlib.util, "find_spec", return_value=None):
        with pytest.raises(source.NoTranscriptAvailable):
            source.fetch_episode("vid", tmp_path, "moonshots")
