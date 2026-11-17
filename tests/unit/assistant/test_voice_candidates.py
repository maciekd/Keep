"""The candidate store: the folder the voice discovery fills and the owner's answer empties."""
import json
from datetime import date
from pathlib import Path

import pytest

from assistant import VoiceCandidates


@pytest.fixture
def store(tmp_path):
    dirs = {"directory": tmp_path / "candidates", "voice_id_dir": tmp_path / "voice_id", "log_dir": tmp_path / "log"}
    return dirs


def _samples(tmp_path: Path, n: int) -> list[Path]:
    folder = tmp_path / "audio"
    folder.mkdir(exist_ok=True)
    paths = []
    for i in range(n):
        p = folder / f"what_s_for_dinner_{i}.wav"
        p.write_bytes(b"RIFF" + bytes([i]))
        paths.append(p)
    return paths


def _file(tmp_path, store, clips=120, days=4, transcripts=("what s for dinner", "turn off the lights"),
          centroid=(1.0, 0.0), samples=3):
    return VoiceCandidates.create(clips, days, "2026-10-08", "2026-10-11", "mostly 17–20h", list(transcripts),
                                  list(centroid), _samples(tmp_path, samples), directory=store["directory"],
                                  today=date(2026, 10, 12))


def test_a_filed_candidate_keeps_its_samples_and_facts(tmp_path, store):
    c = _file(tmp_path, store)
    assert c.id.startswith(VoiceCandidates.ID_PREFIX)
    assert len(c.samples) == 3 and all(p.read_bytes().startswith(b"RIFF") for p in c.samples)
    (again,) = VoiceCandidates.items(store["directory"])
    assert (again.id, again.clips, again.days, again.centroid) == (c.id, 120, 4, [1.0, 0.0])


def test_the_index_shows_the_facts_that_tell_the_voices_apart(tmp_path, store):
    c = _file(tmp_path, store)
    text = VoiceCandidates.render_index(store["directory"])
    assert c.id in text and "120 clips over 4 days" in text and "2026-10-08 to 2026-10-11" in text
    assert "what s for dinner" in text


def test_the_index_is_empty_without_candidates(store):
    assert VoiceCandidates.render_index(store["directory"]) == ""


def test_the_most_heard_voice_comes_first(tmp_path, store):
    small = _file(tmp_path, store, clips=30)
    big = _file(tmp_path, store, clips=300)
    assert [c.id for c in VoiceCandidates.items(store["directory"])] == [big.id, small.id]


def test_version_changes_on_every_change_the_prompt_would_show(tmp_path, store):
    seen = {VoiceCandidates.version(store["directory"])}
    c = _file(tmp_path, store)
    seen.add(VoiceCandidates.version(store["directory"]))
    VoiceCandidates.update(c, 200, 5, "2026-10-07", "2026-10-12", "", [])
    seen.add(VoiceCandidates.version(store["directory"]))
    assert len(seen) == 3


def test_version_changes_with_the_facts_even_when_the_files_time_does_not(tmp_path, store):
    """Linux stamps a file with the clock of the last timer tick: two writes a few milliseconds
    apart carry one mtime, and the prompt would keep the list it had."""
    import os
    c = _file(tmp_path, store)
    facts = c.folder / VoiceCandidates.FACTS_FILE
    stamp = facts.stat()
    before = VoiceCandidates.version(store["directory"])
    VoiceCandidates.update(c, 200, 5, "2026-10-07", "2026-10-12", "", [])
    os.utime(facts, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert facts.stat().st_mtime_ns == stamp.st_mtime_ns
    assert VoiceCandidates.version(store["directory"]) != before


def test_naming_moves_the_samples_into_the_persons_enrollment_folder(tmp_path, store):
    c = _file(tmp_path, store)
    assert VoiceCandidates.name(c.id, "Charlotte", **store) == ""
    enrolled = sorted((store["voice_id_dir"] / "charlotte").glob("*.wav"))
    assert len(enrolled) == 3 and all(p.name.startswith(c.id) for p in enrolled)
    assert VoiceCandidates.items(store["directory"]) == []
    trail = json.loads((store["log_dir"] / "named" / f"{c.id}.json").read_text())
    assert trail["name"] == "charlotte" and trail["samples"] == 3


def test_naming_an_enrolled_person_adds_to_their_folder(tmp_path, store):
    (store["voice_id_dir"] / "charlotte").mkdir(parents=True)
    (store["voice_id_dir"] / "charlotte" / "old.wav").write_bytes(b"RIFF")
    c = _file(tmp_path, store)
    assert VoiceCandidates.name(c.id, "charlotte", **store) == ""
    assert len(list((store["voice_id_dir"] / "charlotte").glob("*.wav"))) == 4


def test_run_refuses_a_missing_voice_or_argument_and_names_the_open_ones(tmp_path, store):
    c = _file(tmp_path, store)
    assert "expected" in VoiceCandidates.run("name_voice", [c.id], **store)
    refused = VoiceCandidates.run("name_voice", ["voice-nope", "Bob"], **store)
    assert "no voice" in refused and c.id in refused
    assert "letters or digits" in VoiceCandidates.run("name_voice", [c.id, "???"], **store)
    assert VoiceCandidates.run("name_voice", [f" `{c.id}` ", "Bob"], **store) == ""


@pytest.mark.parametrize("filename, text", [
    ("what_s_for_dinner__3.wav", "what s for dinner"),
    ("Michelle__89.wav", "Michelle"),
    ("I_m_a_weirdo_.wav", "I m a weirdo"),
])
def test_the_transcript_is_the_filename_without_its_counter(filename, text):
    assert VoiceCandidates.transcript_of(filename) == text


def test_a_long_transcript_is_cut_with_an_ellipsis():
    text = VoiceCandidates.transcript_of("word_" * 40 + ".wav")
    assert text.endswith("…") and len(text) <= VoiceCandidates.TRANSCRIPT_CHARS + 1


def test_a_folder_without_facts_is_not_a_candidate(tmp_path, store):
    store["directory"].mkdir()
    (store["directory"] / "voice-zzzz").mkdir()
    (store["directory"] / "voice-zzzz" / "voice-zzzz-01.wav").write_bytes(b"RIFF")
    assert VoiceCandidates.items(store["directory"]) == []


def test_the_unnamed_speaker_note_names_the_voice_and_asks_for_the_name():
    note = VoiceCandidates.unnamed_speaker_note("voice-7f3a")
    assert "voice-7f3a" in note and "ask" in note.lower() and "VOICES" in note
