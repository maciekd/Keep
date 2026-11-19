import os
import time
import wave

from common import audio_log
from common.audio_log import cleanup_audio_folder, read_wav_metadata, save_audio_file, save_pcm_wav
from tests.unit.waiting import FAILURE_BOUND


def _pcm(n_samples: int) -> bytes:
    # n_samples of 16-bit silence
    return b"\x00\x00" * n_samples


def test_save_pcm_wav_writes_readable_wav(tmp_path):
    pcm = _pcm(1600)
    path = save_pcm_wav(str(tmp_path), "what's the weather", pcm, sample_rate=16000)

    assert os.path.basename(path) == "what_s_the_weather.wav"
    with wave.open(path, "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
        assert wf.readframes(wf.getnframes()) == pcm


def test_metadata_travels_inside_the_wav_and_readers_still_decode_it(tmp_path):
    """The Whisper-span metadata must survive a clip being copied into a test set, so
    it lives in a RIFF chunk after the PCM — and both readers the harnesses use
    (soundfile, wave) must keep decoding the same samples with the chunk present."""
    import soundfile as sf

    pcm = _pcm(1601)  # odd sample count → odd-length JSON is the interesting alignment case
    meta = {"whisper_span_samples": 777, "decode": "spec"}
    path = save_pcm_wav(str(tmp_path), "hello", pcm, sample_rate=16000, metadata=meta)

    assert read_wav_metadata(path) == meta
    with wave.open(path, "rb") as wf:
        assert wf.readframes(wf.getnframes()) == pcm
    audio, sr = sf.read(path, dtype="int16")
    assert sr == 16000 and len(audio) == 1601
    with open(path, "rb") as f:
        assert int.from_bytes(f.read(8)[4:], "little") == os.path.getsize(path) - 8


def test_wav_without_metadata_reads_empty(tmp_path):
    path = save_pcm_wav(str(tmp_path), "plain", _pcm(10))
    assert read_wav_metadata(path) == {}
    (tmp_path / "junk.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAV")
    assert read_wav_metadata(str(tmp_path / "junk.wav")) == {}


def test_save_pcm_wav_dedupes_name_collision(tmp_path):
    p1 = save_pcm_wav(str(tmp_path), "hello", _pcm(10))
    p2 = save_pcm_wav(str(tmp_path), "hello", _pcm(10))
    p3 = save_pcm_wav(str(tmp_path), "hello", _pcm(10))

    assert os.path.basename(p1) == "hello.wav"
    assert os.path.basename(p2) == "hello_1.wav"
    assert os.path.basename(p3) == "hello_2.wav"
    assert os.path.exists(p1) and os.path.exists(p2) and os.path.exists(p3)


def test_save_pcm_wav_concurrent_same_name_never_overwrites(tmp_path):
    """Each save runs on its own thread now — the pick-a-name + write pair must be
    serialized or two mics saving the same phrase clobber each other's WAV."""
    import threading

    paths = []
    paths_lock = threading.Lock()

    def save():
        p = save_pcm_wav(str(tmp_path), "hello", _pcm(10))
        with paths_lock:
            paths.append(p)

    threads = [threading.Thread(target=save) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=FAILURE_BOUND)

    assert len(paths) == 8
    assert len(set(paths)) == 8
    assert all(os.path.exists(p) for p in paths)


def test_save_pcm_wav_caps_long_name_at_100_chars(tmp_path):
    # Long transcripts used to be written verbatim as the WAV basename, which
    # could blow past the filesystem's NAME_MAX (255 on most Linux/macOS). A
    # long user query like the one below would fail with ENAMETOOLONG before
    # any frames were written. Cap the sanitized base at 100 chars to keep
    # well under the limit and leave room for the "_NN" dedup suffix + ".wav".
    long_query = "tell me everything you know about the history of " * 5  # ~245 chars
    path = save_pcm_wav(str(tmp_path), long_query, _pcm(5))

    basename = os.path.basename(path)
    # 100-char base + ".wav" suffix → 104 chars total. Strictly bounded so a
    # future maintainer who tweaks the cap can't silently regress this.
    assert basename.endswith(".wav")
    assert len(basename) - len(".wav") == 100
    assert os.path.exists(path)


def test_save_pcm_wav_long_name_collision_still_dedupes_within_cap(tmp_path):
    # Two long queries with identical first-100 sanitized chars must still get
    # distinct filenames via the numeric-suffix dedup — otherwise the second
    # write would clobber the first. Proves the cap happens BEFORE the dedup
    # loop, not after (which would have re-introduced the length blow-up).
    long_query = "x" * 200
    p1 = save_pcm_wav(str(tmp_path), long_query, _pcm(5))
    p2 = save_pcm_wav(str(tmp_path), long_query, _pcm(5))

    assert p1 != p2
    assert os.path.exists(p1) and os.path.exists(p2)
    # Second file picks up the "_1" dedup suffix on top of the 100-char base.
    assert os.path.basename(p2).endswith("_1.wav")


def test_cleanup_runs_on_first_save_then_every_nth(tmp_path, monkeypatch):
    # Scanning a 100k-file folder on every save is too expensive for the
    # transcript path, so the cap is only enforced on the first save per
    # folder and every Nth after that.
    monkeypatch.setattr(audio_log, "CLEANUP_EVERY_N_SAVES", 5)
    calls = []
    monkeypatch.setattr(audio_log, "cleanup_audio_folder",
                        lambda folder, max_size_gb=10: calls.append(folder))

    for _ in range(11):
        save_pcm_wav(str(tmp_path), "hello", _pcm(5))

    # Saves 1, 5 and 10 trigger a pass; the rest skip it.
    assert calls == [str(tmp_path)] * 3


def test_cleanup_counter_is_per_folder(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(audio_log, "cleanup_audio_folder",
                        lambda folder, max_size_gb=10: calls.append(folder))

    a, b = tmp_path / "a", tmp_path / "b"
    save_pcm_wav(str(a), "hi", _pcm(5))
    save_pcm_wav(str(b), "hi", _pcm(5))

    # Each folder gets its own first-save cleanup rather than sharing a counter.
    assert calls == [str(a), str(b)]


def test_cleanup_deletes_oldest_files_over_cap(tmp_path):
    for i, name in enumerate(["old.wav", "mid.wav", "new.wav"]):
        p = tmp_path / name
        p.write_bytes(b"\x00" * 1024)
        mtime = time.time() - (100 - i)
        os.utime(p, (mtime, mtime))

    cleanup_audio_folder(str(tmp_path), max_size_gb=2 * 1024 / (1024 ** 3))

    assert not (tmp_path / "old.wav").exists()
    assert (tmp_path / "mid.wav").exists()
    assert (tmp_path / "new.wav").exists()


def test_cleanup_queued_behind_another_pass_runs_instead_of_skipping(tmp_path):
    # cleanup_audio_folder used to take its lock with blocking=False and silently
    # return when another pass held it, so a WavSave thread's pass could mask the
    # one that mattered and leave the folder over cap. A queued pass must wait for
    # the in-flight one and then run.
    import threading

    old = tmp_path / "old.wav"
    old.write_bytes(b"\x00" * 1024)
    mtime = time.time() - 100
    os.utime(old, (mtime, mtime))
    (tmp_path / "new.wav").write_bytes(b"\x00" * 1024)

    audio_log._cleanup_lock.acquire()   # an in-flight pass on another thread
    try:
        t = threading.Thread(target=cleanup_audio_folder,
                             args=(str(tmp_path),), kwargs={"max_size_gb": 1024 / (1024 ** 3)})
        t.start()
        t.join(timeout=0.05)
        # With the old skip behavior the thread would already be done here having
        # deleted nothing; it must still be waiting on the lock instead.
        assert t.is_alive()
        assert old.exists()
    finally:
        audio_log._cleanup_lock.release()
    t.join(timeout=FAILURE_BOUND)
    assert not t.is_alive()
    assert not old.exists()
    assert (tmp_path / "new.wav").exists()


def test_cleanup_tolerates_file_deleted_by_another_pass(tmp_path, monkeypatch):
    # Two WavSave threads used to race in cleanup and error-log ENOENT on the
    # files the other thread had already deleted. A vanished file must count
    # as freed space, not an error.
    for name in ["old.wav", "new.wav"]:
        (tmp_path / name).write_bytes(b"\x00" * 1024)
    old = tmp_path / "old.wav"
    mtime = time.time() - 100
    os.utime(old, (mtime, mtime))

    real_remove = os.remove

    def racing_remove(path):
        real_remove(path)
        raise FileNotFoundError(2, "No such file or directory", path)

    monkeypatch.setattr(os, "remove", racing_remove)
    errors = []
    monkeypatch.setattr(audio_log, "error", lambda msg: errors.append(msg))

    cleanup_audio_folder(str(tmp_path), max_size_gb=1024 / (1024 ** 3))

    assert not old.exists()
    assert errors == []


def test_an_encoded_clip_is_written_as_it_arrived_and_a_second_one_never_replaces_it(tmp_path):
    folder = tmp_path / "voice_notes"
    first = save_audio_file(str(folder), "turn on the lights", b"OggS-one", "ogg")
    second = save_audio_file(str(folder), "turn on the lights", b"OggS-two", "ogg")

    assert first != second and {os.path.dirname(first), os.path.dirname(second)} == {str(folder)}
    assert first.endswith(".ogg") and second.endswith(".ogg")
    with open(first, "rb") as f:
        assert f.read() == b"OggS-one"
    with open(second, "rb") as f:
        assert f.read() == b"OggS-two"
