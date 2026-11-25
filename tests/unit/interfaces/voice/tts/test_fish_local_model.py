"""interfaces.voice.tts.FishLocalModel — the gates, the download, the install's own voice and the one thread that speaks."""
import queue
import threading

import pytest

from assistant.integration.Integration import HealthWarning
from common import Platform
from interfaces.voice.tts import FishLocalModel as fish
from interfaces.voice.tts.FishLocalModel import FishLocalModel, Settings
from tests.unit.interfaces.voice.tts.fish_fakes import SMALL, FakeModel, fish_home  # noqa: F401
from tests.unit.waiting import FAILURE_BOUND, wait_until

pytestmark = pytest.mark.unit


def _ready(settings=Settings()) -> FishLocalModel:
    run = FishLocalModel(settings)
    run.prepare()
    return run


def _drain(run, text, cancelled=lambda: False):
    out = queue.Queue()
    run._speak(text, out, cancelled)
    pieces = []
    while (piece := out.get_nowait()) is not None:
        pieces.append(piece)
    return pieces


class TestGate:
    @pytest.mark.parametrize("memory, carried", [(24, set()), (32, {"3bit"}), (48, {"3bit", "4bit"}),
                                                 (64, {"3bit", "4bit", "8bit"})])
    def test_each_build_is_offered_from_its_own_memory_floor(self, fish_home, monkeypatch, memory, carried):
        monkeypatch.setattr(Platform, "memory_gb", lambda: memory)
        assert {quant for quant in fish.BUILDS if not FishLocalModel.unmet(Settings(quant))} == carried

    def test_too_little_memory_is_said_with_both_numbers(self, fish_home, monkeypatch):
        monkeypatch.setattr(Platform, "memory_gb", lambda: 36)
        said = FishLocalModel.unmet(Settings("4bit"))
        assert "36" in said and "48" in said

    def test_a_machine_without_apple_silicon_is_refused(self, fish_home, monkeypatch):
        monkeypatch.setattr(Platform, "is_apple_silicon", lambda: False)
        assert FishLocalModel.unmet(Settings())

    def test_a_quant_there_is_no_build_of_is_refused_by_name(self, fish_home):
        assert "5bit" in FishLocalModel.unmet(Settings("5bit"))

    def test_the_download_needs_its_size_and_a_margin_of_free_disk(self, fish_home, monkeypatch):
        monkeypatch.setattr(fish, "GB", 100)                 # a build of 10 "GB"
        monkeypatch.setattr(fish, "free_gb", lambda: 10 + fish.DISK_MARGIN_GB - 1)
        said = FishLocalModel.unmet(Settings())
        assert str(10 + fish.DISK_MARGIN_GB) in said and str(10 + fish.DISK_MARGIN_GB - 1) in said
        monkeypatch.setattr(fish, "free_gb", lambda: 10 + fish.DISK_MARGIN_GB)
        assert not FishLocalModel.unmet(Settings())

    def test_what_is_on_disk_of_an_interrupted_download_counts(self, fish_home, monkeypatch):
        monkeypatch.setattr(fish, "GB", 100)
        monkeypatch.setattr(fish, "free_gb", lambda: fish.DISK_MARGIN_GB + 4)
        assert FishLocalModel.unmet(Settings())
        fish_home.fill("4bit", 600)                          # 4 of the 10 are left to fetch
        assert not FishLocalModel.unmet(Settings())

    def test_a_downloaded_build_needs_no_free_disk(self, fish_home, monkeypatch):
        fish_home.have()
        monkeypatch.setattr(fish, "free_gb", lambda: 0)
        assert not FishLocalModel.unmet(Settings())

    def test_a_voice_clip_must_exist_and_come_with_its_words(self, fish_home):
        clip = fish_home.tmp / "marcus.wav"
        assert str(clip) in FishLocalModel.unmet(Settings(voice=str(clip), voice_text="Hello there."))
        clip.write_bytes(b"RIFF")
        assert "voice_text" in FishLocalModel.unmet(Settings(voice=str(clip)))
        assert not FishLocalModel.unmet(Settings(voice=str(clip), voice_text="Hello there."))


class TestDownload:
    def test_a_first_start_fetches_the_pinned_revision_of_the_configured_build(self, fish_home):
        run = _ready(Settings("8bit"))
        assert fish_home.hub == [fish.BUILDS["8bit"]]
        assert run.state == fish.READY and fish.downloaded("8bit") and not fish.downloaded("4bit")

    def test_a_downloaded_build_starts_without_asking_the_hub(self, fish_home):
        """A local voice comes up without a network."""
        _ready()
        fish_home.hub_error = ConnectionError("offline")
        assert _ready().state == fish.READY
        assert len(fish_home.hub) == 1

    def test_a_build_of_another_revision_is_fetched_again(self, fish_home):
        fish_home.have()
        fish.marker("4bit").write_text("0" * 40)
        _ready()
        assert fish_home.hub == [fish.BUILDS["4bit"]]

    def test_a_failed_download_leaves_no_complete_build_and_gives_its_reason(self, fish_home):
        fish_home.hub_error = ConnectionError("the hub hung up")
        run = FishLocalModel(Settings())
        run._run()
        assert run.state == fish.FAILED and "the hub hung up" in run.reason
        assert not fish.downloaded("4bit")
        with pytest.raises(RuntimeError, match="the hub hung up"):
            run.health()

    def test_progress_is_what_is_on_disk_and_health_reports_it_as_a_warning(self, fish_home):
        run = FishLocalModel(Settings())
        run.state = fish.DOWNLOADING
        fish_home.fill("4bit", SMALL * 43 // 100)
        assert run.percent() == 43
        with pytest.raises(HealthWarning, match="43"):
            run.health()

    def test_the_part_files_of_an_interrupted_download_are_dropped_before_it_is_fetched_again(self, fish_home):
        fish_home.fill("4bit", SMALL // 4)
        stale = fish.model_dir("4bit") / ".cache" / "huggingface" / "download" / "weights.0a1b.incomplete"
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"0" * (SMALL // 2))
        run = FishLocalModel(Settings())
        assert run.percent() == 75
        seen = []
        fetch = fish._hub_download
        fish._hub_download = lambda build, target: (seen.append((stale.exists(), run.percent())), fetch(build, target))
        try:
            run.prepare()
        finally:
            fish._hub_download = fetch
        assert seen == [(False, 25)]

    def test_a_part_file_does_not_count_as_disk_the_download_no_longer_needs(self, fish_home, monkeypatch):
        monkeypatch.setattr(fish, "GB", 100)                 # a build of 10 "GB"
        stale = fish.model_dir("4bit") / "weights.0a1b.incomplete"
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"0" * (SMALL // 2))
        monkeypatch.setattr(fish, "free_gb", lambda: 10 + fish.DISK_MARGIN_GB - 1)
        assert FishLocalModel.unmet(Settings())

    def test_progress_never_passes_the_whole(self, fish_home):
        fish_home.fill("4bit", SMALL * 2)        # the hub's own bookkeeping lies beside the weights
        assert FishLocalModel(Settings()).percent() == 100


class TestOwnVoice:
    def test_the_first_start_makes_a_voice_without_a_reference_and_speaks_in_it_from_then_on(self, fish_home):
        run = _ready()
        clip, transcript = fish.own_voice()
        made = fish_home.model.calls[0]
        assert made.text == fish.VOICE_TEXT and made.ref_audio is None
        assert clip.is_file() and transcript.read_text() == fish.VOICE_TEXT
        _drain(run, "The car is charged.")
        assert all((c.ref_audio, c.ref_text) == (str(clip), fish.VOICE_TEXT) for c in fish_home.model.calls[1:])

    def test_the_next_start_reads_the_same_voice_instead_of_making_another(self, fish_home):
        _ready()
        made = fish.own_voice()[0].read_bytes()
        fish_home.model = FakeModel()
        _ready()
        assert fish.VOICE_TEXT not in [c.text for c in fish_home.model.calls]
        assert fish.own_voice()[0].read_bytes() == made

    def test_a_configured_clip_is_the_voice_and_no_other_is_made(self, fish_home):
        clip = fish_home.tmp / "charlotte.wav"
        clip.write_bytes(b"RIFF")
        run = _ready(Settings(voice=str(clip), voice_text="Good evening."))
        _drain(run, "Dinner is at seven.")
        assert {(c.ref_audio, c.ref_text) for c in fish_home.model.calls} == {(str(clip), "Good evening.")}
        assert not fish.own_voice()[0].exists()


class TestSpeaking:
    def test_the_audio_arrives_piece_by_piece_as_a_stream(self, fish_home):
        run = _ready()
        assert len(_drain(run, "The door is locked.")) == fish_home.model.pieces
        assert fish_home.model.calls[-1].stream

    def test_a_reply_killed_between_pieces_stops_the_generation(self, fish_home):
        run = _ready()
        seen = []
        closed_before = fish_home.model.closed

        def cancelled():
            seen.append(1)
            return len(seen) > 2          # asked before the start, then once per piece

        assert len(_drain(run, "A long story.", cancelled)) == 1
        assert fish_home.model.closed == closed_before + 1

    def test_a_reply_dropped_while_waiting_its_turn_is_never_generated(self, fish_home):
        run = _ready()
        assert _drain(run, "Too late.", lambda: True) == []
        assert fish_home.model.spoken() == []

    def test_a_text_the_model_refuses_ends_its_stream_and_the_next_one_is_spoken(self, fish_home):
        run = _ready()
        fish_home.model.fail = ValueError("text is empty")
        assert _drain(run, "…") == []
        fish_home.model.fail = None
        assert _drain(run, "The door is locked.")

    def test_nothing_is_spoken_before_the_model_is_ready(self, fish_home):
        run = FishLocalModel(Settings())
        run.state = fish.LOADING
        assert list(run.speak("Hello.", lambda: False)) == []

    def test_two_dialogs_take_turns_on_the_one_thread(self, fish_home):
        fish_home.have()
        run = FishLocalModel.ensure(Settings())
        assert wait_until(lambda: run.state == fish.READY)
        heard = {}

        def dialog(name, text):
            heard[name] = len(list(run.speak(text, lambda: False)))

        dialogs = [threading.Thread(target=dialog, args=(name, text))
                   for name, text in (("marcus", "The car is charged."), ("frances", "Dinner is at seven."))]
        for thread in dialogs:
            thread.start()
        for thread in dialogs:
            thread.join(FAILURE_BOUND)
        assert heard == {"marcus": fish_home.model.pieces, "frances": fish_home.model.pieces}
        assert sorted(fish_home.model.spoken()) == ["Dinner is at seven.", "The car is charged."]


class TestTheOneRun:
    def test_the_same_settings_get_the_same_run(self, fish_home):
        fish_home.have()
        assert FishLocalModel.ensure(Settings()) is FishLocalModel.ensure(Settings())

    def test_a_failed_run_is_replaced_and_the_download_resumes(self, fish_home):
        fish_home.hub_error = ConnectionError("offline")
        failed = FishLocalModel.ensure(Settings())
        assert wait_until(lambda: failed.state == fish.FAILED)
        fish_home.hub_error = None
        again = FishLocalModel.ensure(Settings())
        assert again is not failed and wait_until(lambda: again.state == fish.READY)

    def test_other_settings_stop_the_old_run_and_answer_whoever_waited_on_it(self, fish_home):
        fish_home.have("4bit")
        fish_home.have("8bit")
        old = FishLocalModel.ensure(Settings("4bit"))
        assert wait_until(lambda: old.state == fish.READY)
        new = FishLocalModel.ensure(Settings("8bit"))
        assert old.state == fish.STOPPED and FishLocalModel.current(Settings("4bit")) is None
        assert wait_until(lambda: new.state == fish.READY)
        assert list(old.speak("Anyone there?", lambda: False)) == []

    def test_a_job_queued_as_the_run_stops_is_answered_not_left_waiting(self, fish_home):
        run = _ready()
        out = queue.Queue()
        run._jobs.put(("Never spoken.", out, lambda: False))
        run._release()
        assert out.get_nowait() is None
