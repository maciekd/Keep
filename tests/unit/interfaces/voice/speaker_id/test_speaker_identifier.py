import sys
import threading
import pytest
from unittest.mock import Mock, patch, MagicMock


# Save original modules before replacing them (numpy is used by other tests)
_original_modules = {
    key: sys.modules.get(key) for key in [
        'torch', 'torch.nn', 'torch.nn.functional',
        'torchaudio', 'torchaudio.transforms',
        'numpy',
        'speechbrain', 'speechbrain.inference', 'speechbrain.inference.speaker'
    ]
}

mock_torch = MagicMock()
mock_torch.Tensor = MagicMock
mock_torch.from_numpy = Mock(return_value=Mock())
mock_torch.mean = Mock(return_value=Mock())

mock_F = MagicMock()
mock_F.normalize = Mock(side_effect=lambda x, **kwargs: x)
mock_F.cosine_similarity = Mock(return_value=Mock(item=Mock(return_value=0.5)))

mock_torchaudio = MagicMock()
mock_np = MagicMock()

sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.nn.functional'] = mock_F
sys.modules['torchaudio'] = mock_torchaudio
sys.modules['torchaudio.transforms'] = MagicMock()
sys.modules['numpy'] = mock_np
sys.modules['speechbrain'] = MagicMock()
sys.modules['speechbrain.inference'] = MagicMock()
sys.modules['speechbrain.inference.speaker'] = MagicMock()


from interfaces.voice.speaker_id.SpeakerIdentifier import SpeakerIdentifier

# Restore original modules immediately after import to not affect other tests
for _key, _original in _original_modules.items():
    if _original is not None:
        sys.modules[_key] = _original
    elif _key in sys.modules:
        del sys.modules[_key]
del _key, _original


@pytest.fixture(autouse=True)
def reset_singleton():
    SpeakerIdentifier.reset_instance()
    yield
    SpeakerIdentifier.reset_instance()


@pytest.fixture
def mock_ml_globals():
    """Fixture that patches the module-level ML globals."""
    with patch('interfaces.voice.speaker_id.SpeakerIdentifier.np', mock_np), \
         patch('interfaces.voice.speaker_id.SpeakerIdentifier.torch', mock_torch), \
         patch('interfaces.voice.speaker_id.SpeakerIdentifier.F', mock_F), \
         patch('interfaces.voice.speaker_id.SpeakerIdentifier.torchaudio', mock_torchaudio):
        yield


def make_speaker_identifier():
    """Create a SpeakerIdentifier without running __init__ (avoids loading ML libs)."""
    si = object.__new__(SpeakerIdentifier)
    si._classifier = Mock()
    si._enrolled_users = {}
    si._identification_history = []
    si._history_lock = threading.Lock()
    return si


class TestGetLastIdentifiedUser:
    @patch('interfaces.voice.speaker_id.SpeakerIdentifier.time')
    def test_returns_most_voted_user(self, mock_time):
        mock_time.time.return_value = 1000.0
        si = make_speaker_identifier()
        si._identification_history = [
            ("alice", 0.8, 999.0),
            ("alice", 0.75, 998.0),
            ("bob", 0.7, 997.0),
        ]
        result = si.get_last_identified_user(max_age_seconds=10.0)
        assert result == "alice"

    @patch('interfaces.voice.speaker_id.SpeakerIdentifier.time')
    def test_returns_none_on_tie(self, mock_time):
        mock_time.time.return_value = 1000.0
        si = make_speaker_identifier()
        si._identification_history = [
            ("alice", 0.8, 999.0),
            ("bob", 0.75, 998.0),
        ]
        result = si.get_last_identified_user(max_age_seconds=10.0)
        assert result is None

    @patch('interfaces.voice.speaker_id.SpeakerIdentifier.time')
    def test_respects_max_age_seconds(self, mock_time):
        mock_time.time.return_value = 1000.0
        si = make_speaker_identifier()
        si._identification_history = [
            ("alice", 0.8, 999.0),  # 1 second old - within window
            ("alice", 0.75, 990.0),  # 10 seconds old - outside 5s window
            ("bob", 0.7, 998.0),  # 2 seconds old - within window
        ]
        # With max_age_seconds=5.0, only the 999.0 and 998.0 entries count
        result = si.get_last_identified_user(max_age_seconds=5.0)
        # alice and bob each have 1 vote -> tie
        assert result is None

    @patch('interfaces.voice.speaker_id.SpeakerIdentifier.time')
    def test_ignores_old_identifications(self, mock_time):
        mock_time.time.return_value = 1000.0
        si = make_speaker_identifier()
        si._identification_history = [
            ("alice", 0.8, 900.0),  # 100 seconds old - way outside window
        ]
        result = si.get_last_identified_user(max_age_seconds=5.0)
        assert result is None

    @patch('interfaces.voice.speaker_id.SpeakerIdentifier.time')
    def test_single_identification_returns_user(self, mock_time):
        mock_time.time.return_value = 1000.0
        si = make_speaker_identifier()
        si._identification_history = [
            ("alice", 0.8, 999.0),
        ]
        result = si.get_last_identified_user(max_age_seconds=5.0)
        assert result == "alice"


class TestPruneOldIdentifications:
    @patch('interfaces.voice.speaker_id.SpeakerIdentifier.time')
    def test_removes_old_entries(self, mock_time):
        mock_time.time.return_value = 1000.0
        si = make_speaker_identifier()
        # IDENTIFICATION_VOTE_WINDOW is 30 seconds
        si._identification_history = [
            ("alice", 0.8, 960.0),  # exactly 40 seconds old - should be removed
            ("bob", 0.75, 950.0),  # 50 seconds old - should be removed
        ]
        si._prune_old_identifications()
        assert len(si._identification_history) == 0

    @patch('interfaces.voice.speaker_id.SpeakerIdentifier.time')
    def test_keeps_recent_entries(self, mock_time):
        mock_time.time.return_value = 1000.0
        si = make_speaker_identifier()
        si._identification_history = [
            ("alice", 0.8, 995.0),  # 5 seconds old - keep
            ("bob", 0.75, 980.0),  # 20 seconds old - keep
        ]
        si._prune_old_identifications()
        assert len(si._identification_history) == 2

    @patch('interfaces.voice.speaker_id.SpeakerIdentifier.time')
    def test_boundary_at_vote_window(self, mock_time):
        mock_time.time.return_value = 1000.0
        si = make_speaker_identifier()
        # cutoff = 1000 - 30 = 970
        si._identification_history = [
            ("alice", 0.8, 970.0),  # exactly at boundary - keep (ts >= cutoff)
            ("bob", 0.75, 969.9),  # just before boundary - remove
        ]
        si._prune_old_identifications()
        assert len(si._identification_history) == 1
        assert si._identification_history[0][0] == "alice"


class TestIdentify:
    def test_returns_unknown_when_no_enrolled_users(self):
        si = make_speaker_identifier()
        si._enrolled_users = {}
        result = si.identify(b'\x00' * 100)
        assert result == ("unknown", 0.0)

    @patch('interfaces.voice.speaker_id.SpeakerIdentifier.time')
    def test_returns_unknown_below_threshold(self, mock_time, mock_ml_globals):
        mock_time.time.return_value = 1000.0
        si = make_speaker_identifier()

        mock_embedding = Mock()
        mock_embedding.unsqueeze = Mock(return_value=mock_embedding)
        mock_embedding.squeeze = Mock(return_value=mock_embedding)
        si._extract_embedding_from_pcm = Mock(return_value=mock_embedding)

        enrolled_emb = Mock()
        enrolled_emb.unsqueeze = Mock(return_value=enrolled_emb)
        si._enrolled_users = {"alice": [enrolled_emb]}

        mock_F.normalize.return_value = mock_embedding
        mock_F.cosine_similarity.return_value = Mock(item=Mock(return_value=0.2))

        result = si.identify(b'\x00' * 100)

        assert result[0] == "unknown"
        assert result[1] == 0.2

    @patch('interfaces.voice.speaker_id.SpeakerIdentifier.time')
    def test_returns_best_match_above_threshold(self, mock_time, mock_ml_globals):
        mock_time.time.return_value = 1000.0
        si = make_speaker_identifier()

        mock_embedding = Mock()
        mock_embedding.unsqueeze = Mock(return_value=mock_embedding)
        mock_embedding.squeeze = Mock(return_value=mock_embedding)
        si._extract_embedding_from_pcm = Mock(return_value=mock_embedding)

        enrolled_emb = Mock()
        enrolled_emb.unsqueeze = Mock(return_value=enrolled_emb)
        si._enrolled_users = {"alice": [enrolled_emb]}

        mock_F.normalize.return_value = mock_embedding
        mock_F.cosine_similarity.return_value = Mock(item=Mock(return_value=0.8))

        result = si.identify(b'\x00' * 100)

        assert result == ("alice", 0.8)

    @patch('interfaces.voice.speaker_id.SpeakerIdentifier.time')
    def test_records_identification_in_history(self, mock_time, mock_ml_globals):
        mock_time.time.return_value = 1000.0
        si = make_speaker_identifier()
        si._identification_history = []

        mock_embedding = Mock()
        mock_embedding.unsqueeze = Mock(return_value=mock_embedding)
        mock_embedding.squeeze = Mock(return_value=mock_embedding)
        si._extract_embedding_from_pcm = Mock(return_value=mock_embedding)

        enrolled_emb = Mock()
        enrolled_emb.unsqueeze = Mock(return_value=enrolled_emb)
        si._enrolled_users = {"alice": [enrolled_emb]}

        mock_F.normalize.return_value = mock_embedding
        mock_F.cosine_similarity.return_value = Mock(item=Mock(return_value=0.8))

        si.identify(b'\x00' * 100)

        assert len(si._identification_history) == 1
        assert si._identification_history[0][0] == "alice"
        assert si._identification_history[0][1] == 0.8
        assert si._identification_history[0][2] == 1000.0

    @patch('interfaces.voice.speaker_id.SpeakerIdentifier.time')
    def test_record_false_identifies_without_updating_history(self, mock_time, mock_ml_globals):
        # record=False computes and returns the speaker but leaves the vote history
        # untouched (used by speculative background identification).
        mock_time.time.return_value = 1000.0
        si = make_speaker_identifier()
        si._identification_history = []

        mock_embedding = Mock()
        mock_embedding.unsqueeze = Mock(return_value=mock_embedding)
        mock_embedding.squeeze = Mock(return_value=mock_embedding)
        si._extract_embedding_from_pcm = Mock(return_value=mock_embedding)

        enrolled_emb = Mock()
        enrolled_emb.unsqueeze = Mock(return_value=enrolled_emb)
        si._enrolled_users = {"alice": [enrolled_emb]}

        mock_F.normalize.return_value = mock_embedding
        mock_F.cosine_similarity.return_value = Mock(item=Mock(return_value=0.8))

        result = si.identify(b'\x00' * 100, record=False)

        assert result == ("alice", 0.8)
        assert si._identification_history == []

        # The deferred recorder then commits it, mirroring the wake-word path.
        si.record_identification("alice", 0.8)
        assert len(si._identification_history) == 1
        assert si._identification_history[0][:2] == ("alice", 0.8)

    def test_record_identification_ignores_unknown(self, mock_ml_globals):
        si = make_speaker_identifier()
        si._identification_history = []
        si.record_identification("unknown", 0.1)
        si.record_identification("", 0.0)
        assert si._identification_history == []

    @patch('interfaces.voice.speaker_id.SpeakerIdentifier.time')
    def test_k_nearest_averaging(self, mock_time, mock_ml_globals):
        mock_time.time.return_value = 1000.0
        si = make_speaker_identifier()

        mock_embedding = Mock()
        mock_embedding.unsqueeze = Mock(return_value=mock_embedding)
        mock_embedding.squeeze = Mock(return_value=mock_embedding)
        si._extract_embedding_from_pcm = Mock(return_value=mock_embedding)

        enrolled_embs = [Mock() for _ in range(5)]
        for emb in enrolled_embs:
            emb.unsqueeze = Mock(return_value=emb)
        si._enrolled_users = {"alice": enrolled_embs}

        # Return different similarities for each call
        similarity_values = [0.9, 0.8, 0.7, 0.6, 0.5]
        call_count = [0]
        def get_similarity():
            val = similarity_values[call_count[0] % len(similarity_values)]
            call_count[0] += 1
            return val

        mock_F.normalize.return_value = mock_embedding
        mock_F.cosine_similarity.return_value = Mock(item=Mock(side_effect=get_similarity))

        result = si.identify(b'\x00' * 100)

        # K_NEAREST=3, so top 3 are 0.9, 0.8, 0.7 -> average = 0.8
        assert result[0] == "alice"
        assert abs(result[1] - 0.8) < 0.01

    def test_returns_unknown_when_embedding_extraction_fails(self):
        si = make_speaker_identifier()
        si._enrolled_users = {"alice": [Mock()]}
        si._extract_embedding_from_pcm = Mock(return_value=None)

        result = si.identify(b'\x00' * 100)
        assert result == ("unknown", 0.0)

    @patch('interfaces.voice.speaker_id.SpeakerIdentifier.time')
    def test_selects_best_user_among_multiple(self, mock_time, mock_ml_globals):
        mock_time.time.return_value = 1000.0
        si = make_speaker_identifier()

        mock_embedding = Mock()
        mock_embedding.unsqueeze = Mock(return_value=mock_embedding)
        mock_embedding.squeeze = Mock(return_value=mock_embedding)
        si._extract_embedding_from_pcm = Mock(return_value=mock_embedding)

        alice_emb = Mock()
        alice_emb.unsqueeze = Mock(return_value=alice_emb)
        bob_emb = Mock()
        bob_emb.unsqueeze = Mock(return_value=bob_emb)
        si._enrolled_users = {"alice": [alice_emb], "bob": [bob_emb]}

        # alice gets 0.7, bob gets 0.9
        similarity_order = [0.7, 0.9]
        call_idx = [0]
        def get_sim():
            val = similarity_order[call_idx[0]]
            call_idx[0] += 1
            return val

        mock_F.normalize.return_value = mock_embedding
        mock_F.cosine_similarity.return_value = Mock(item=Mock(side_effect=get_sim))

        result = si.identify(b'\x00' * 100)

        assert result[0] == "bob"
        assert result[1] == 0.9


class TestExtractEmbeddingFromPcm:
    def test_unsupported_bit_depth_returns_none(self):
        si = make_speaker_identifier()
        result = si._extract_embedding_from_pcm(b'\x00' * 100, bit_depth=24)
        assert result is None

    def test_resamples_non_16khz_audio(self, mock_ml_globals):
        si = make_speaker_identifier()
        mock_result = Mock()
        mock_result.squeeze = Mock(return_value=Mock())
        si._classifier.encode_batch = Mock(return_value=mock_result)

        audio_data = b'\x00\x00' * 100

        mock_resampler = Mock()
        mock_resampler.return_value = Mock()
        mock_torchaudio.transforms.Resample.return_value = mock_resampler

        mock_array = Mock()
        mock_array.astype = Mock(return_value=mock_array)
        mock_array.__truediv__ = Mock(return_value=mock_array)
        mock_np.frombuffer.return_value = mock_array

        mock_tensor = Mock()
        mock_tensor.unsqueeze = Mock(return_value=mock_tensor)
        mock_torch.from_numpy.return_value = mock_tensor

        si._extract_embedding_from_pcm(audio_data, sample_rate=44100, bit_depth=16)

        mock_torchaudio.transforms.Resample.assert_called_with(44100, 16000)


class TestEnrollUsers:
    def test_skips_empty_user_dirs(self):
        si = make_speaker_identifier()
        mock_voice_id_path = Mock()
        mock_voice_id_path.exists.return_value = True

        mock_user_dir = Mock()
        mock_user_dir.is_dir.return_value = True
        mock_user_dir.name = "alice"
        mock_user_dir.glob.return_value = []  # No WAV files

        mock_voice_id_path.iterdir.return_value = [mock_user_dir]
        si.VOICE_ID_DIR = mock_voice_id_path
        si._ensure_model_loaded = Mock()

        si._enroll_users()

        assert "alice" not in si._enrolled_users

    def test_enrolls_users_with_wav_files(self, mock_ml_globals):
        si = make_speaker_identifier()
        mock_voice_id_path = Mock()
        mock_voice_id_path.exists.return_value = True

        mock_wav_file = Mock()
        mock_wav_file.name = "sample1.wav"

        mock_user_dir = Mock()
        mock_user_dir.is_dir.return_value = True
        mock_user_dir.name = "alice"
        mock_user_dir.glob.return_value = [mock_wav_file]

        mock_voice_id_path.iterdir.return_value = [mock_user_dir]
        si.VOICE_ID_DIR = mock_voice_id_path
        si._ensure_model_loaded = Mock()

        mock_embedding = Mock()
        si._extract_embedding_from_file = Mock(return_value=mock_embedding)
        mock_F.normalize.return_value = mock_embedding

        si._enroll_users()

        assert "alice" in si._enrolled_users
        assert len(si._enrolled_users["alice"]) == 1

    def test_skips_non_directory_entries(self):
        si = make_speaker_identifier()
        mock_voice_id_path = Mock()
        mock_voice_id_path.exists.return_value = True

        mock_file = Mock()
        mock_file.is_dir.return_value = False  # Not a directory

        mock_voice_id_path.iterdir.return_value = [mock_file]
        si.VOICE_ID_DIR = mock_voice_id_path
        si._ensure_model_loaded = Mock()

        si._enroll_users()

        assert si._enrolled_users == {}

    def test_handles_extraction_failure(self):
        si = make_speaker_identifier()
        mock_voice_id_path = Mock()
        mock_voice_id_path.exists.return_value = True

        mock_wav_file = Mock()
        mock_wav_file.name = "sample1.wav"

        mock_user_dir = Mock()
        mock_user_dir.is_dir.return_value = True
        mock_user_dir.name = "alice"
        mock_user_dir.glob.return_value = [mock_wav_file]

        mock_voice_id_path.iterdir.return_value = [mock_user_dir]
        si.VOICE_ID_DIR = mock_voice_id_path
        si._ensure_model_loaded = Mock()

        # Extraction returns None
        si._extract_embedding_from_file = Mock(return_value=None)

        si._enroll_users()

        assert "alice" not in si._enrolled_users

    def test_handles_extraction_exception(self):
        si = make_speaker_identifier()
        mock_voice_id_path = Mock()
        mock_voice_id_path.exists.return_value = True

        mock_wav_file = Mock()
        mock_wav_file.name = "sample1.wav"

        mock_user_dir = Mock()
        mock_user_dir.is_dir.return_value = True
        mock_user_dir.name = "alice"
        mock_user_dir.glob.return_value = [mock_wav_file]

        mock_voice_id_path.iterdir.return_value = [mock_user_dir]
        si.VOICE_ID_DIR = mock_voice_id_path
        si._ensure_model_loaded = Mock()

        # Extraction raises exception
        si._extract_embedding_from_file = Mock(side_effect=Exception("Audio error"))

        si._enroll_users()

        assert "alice" not in si._enrolled_users


        # If it tried to load, it would access speechbrain which would fail


class TestExtractEmbeddingFromFile:
    def test_converts_stereo_to_mono(self, mock_ml_globals):
        si = make_speaker_identifier()

        mock_signal = Mock()
        mock_signal.shape = [2, 16000]  # stereo - 2 channels
        mock_torchaudio.load.return_value = (mock_signal, 16000)
        mock_torch.mean.reset_mock()  # Reset from previous tests
        mock_torch.mean.return_value = mock_signal

        mock_embedding = Mock()
        mock_embedding.squeeze.return_value = mock_embedding
        si._classifier.encode_batch.return_value = mock_embedding

        from pathlib import Path
        si._extract_embedding_from_file(Path("/fake/path.wav"))

        mock_torch.mean.assert_called_once()

    def test_resamples_non_16khz_file(self, mock_ml_globals):
        si = make_speaker_identifier()

        mock_signal = Mock()
        mock_signal.shape = [1, 44100]  # mono at 44.1kHz
        mock_torchaudio.load.return_value = (mock_signal, 44100)

        mock_resampler = Mock()
        mock_resampler.return_value = mock_signal
        mock_torchaudio.transforms.Resample.return_value = mock_resampler

        mock_embedding = Mock()
        mock_embedding.squeeze.return_value = mock_embedding
        si._classifier.encode_batch.return_value = mock_embedding

        from pathlib import Path
        si._extract_embedding_from_file(Path("/fake/path.wav"))

        mock_torchaudio.transforms.Resample.assert_called_with(44100, 16000)


def _dir(name, wavs=(), subdirs=()):
    d = Mock()
    d.is_dir.return_value = True
    d.name = name
    d.glob.return_value = list(wavs)
    d.iterdir.return_value = list(subdirs)
    return d


def test_unnamed_voices_are_enrolled_under_their_id_but_are_not_persons(mock_ml_globals):
    """A candidate of the voice discovery (voice_id/candidates/<id>/) is recognised when it
    speaks, yet the discovery's own matching must not count it as a known person."""
    si = make_speaker_identifier()
    wav = Mock(); wav.name = "voice-7f3a-01.wav"
    root = Mock(); root.exists.return_value = True
    root.iterdir.return_value = [_dir("alice", [wav]), _dir("candidates", subdirs=[_dir("voice-7f3a", [wav]), _dir("voice-0000")])]
    si.VOICE_ID_DIR = root
    si._ensure_model_loaded = Mock()
    si._extract_embedding_from_file = Mock(return_value=Mock())
    mock_F.normalize.return_value = Mock()
    mock_torch.stack.return_value = Mock(numpy=lambda: "array")

    si._enroll_users()

    assert set(si._enrolled_users) == {"alice", "voice-7f3a"}
    assert si.is_candidate("voice-7f3a") and not si.is_candidate("alice")
    assert set(si.enrolled_embeddings()) == {"alice"}


def test_re_enrolling_swaps_the_set_in_whole(mock_ml_globals):
    """identify() on another mic thread must see the old set or the new one, never one being filled."""
    si = make_speaker_identifier()
    wav = Mock(); wav.name = "a.wav"
    root = Mock(); root.exists.return_value = True
    root.iterdir.return_value = [_dir("alice", [wav]), _dir("bob", [wav])]
    si.VOICE_ID_DIR = root
    si._ensure_model_loaded = Mock()
    seen_during = []
    before = si._enrolled_users = {"old": [Mock()]}

    def extract(path):
        seen_during.append(dict(si._enrolled_users))
        return Mock()
    si._extract_embedding_from_file = extract

    si._enroll_users()

    assert all(snapshot == before for snapshot in seen_during)
    assert set(si._enrolled_users) == {"alice", "bob"}


def test_a_reload_already_running_or_still_settling_is_not_started_again():
    import threading, time
    si = make_speaker_identifier()
    si._enrollment_seen = ()
    si._reload_lock = threading.Lock()
    si._enrollment_version = lambda: (("alice", int(time.time() * 1e9)),)
    si._enroll_users = Mock()
    si._reload_if_changed()                      # changed a moment ago: still settling
    si._enroll_users.assert_not_called()
    si._enrollment_version = lambda: (("alice", int((time.time() - 60) * 1e9)),)
    si._reload_lock.acquire()                    # a reload is running on another thread
    si._reload_if_changed()
    si._enroll_users.assert_not_called()
    si._reload_lock.release()
    si._reload_if_changed()
    deadline = time.time() + 2
    while not si._enroll_users.called and time.time() < deadline:
        time.sleep(0.01)
    si._enroll_users.assert_called_once()


class TestEnrollmentVersion:
    """What makes the identifier re-enroll: a person's folder written to, a candidate folder
    appearing or going — not a loose file in voice_id/."""

    def test_follows_the_person_and_candidate_folders_not_loose_files(self, tmp_path):
        import os
        import shutil
        si = make_speaker_identifier()
        si.VOICE_ID_DIR = tmp_path
        alice = tmp_path / "alice"
        alice.mkdir()
        start = si._enrollment_version()

        (tmp_path / "notes.txt").write_text("x")
        assert si._enrollment_version() == start, "a loose file at the top is nobody's enrollment"

        (alice / "a.wav").write_bytes(b"RIFF")
        os.utime(alice, (1_800_000_100, 1_800_000_100))
        after_wav = si._enrollment_version()
        assert after_wav != start

        candidate = tmp_path / SpeakerIdentifier.CANDIDATES_FOLDER / "voice-abcd"
        candidate.mkdir(parents=True)
        after_candidate = si._enrollment_version()
        assert after_candidate != after_wav
        assert "voice-abcd" in {name for name, _ in after_candidate}

        shutil.rmtree(candidate)
        assert si._enrollment_version() != after_candidate

    def test_a_missing_directory_is_an_empty_version(self, tmp_path):
        si = make_speaker_identifier()
        si.VOICE_ID_DIR = tmp_path / "nowhere"
        assert si._enrollment_version() == ()
