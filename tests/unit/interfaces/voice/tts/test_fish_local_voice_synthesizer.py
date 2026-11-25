"""interfaces.voice.tts.FishLocalVoiceSynthesizer — the engine in front of the local Drama 3 model."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from assistant.integration.Integration import HealthWarning, IntegrationState
from assistant.integration.IntegrationMonitor import IntegrationMonitor
from common import Platform
from interfaces.voice.tts import FishLocalModel as fish
from interfaces.voice.tts import FishLocalVoiceSynthesizer as module
from interfaces.voice.tts.FishLocalModel import FishLocalModel, Settings
from interfaces.voice.tts.FishLocalVoiceSynthesizer import FishLocalVoiceSynthesizer, TagBalance
from interfaces.voice.tts.SupertonicVoiceSynthesizer import SupertonicVoiceSynthesizer
from interfaces.voice.tts.VoiceFactory import ENGINES, create_voice
from tests.unit.interfaces.voice.tts.fish_fakes import fish_home  # noqa: F401
from tests.unit.interfaces.voice.tts.test_voice_factory import engines  # noqa: F401
from tests.unit.waiting import wait_until

pytestmark = pytest.mark.unit


@pytest.fixture
def section(monkeypatch):
    """`tts.fish_local` as the test sets it; empty is the defaults. `tts.emotions` is on."""
    values = {}
    real = module.config.get
    monkeypatch.setattr(module.config, "get", lambda key, default=None: (
        {"emotions": True, "fish_local": values} if key == "tts" else real(key, default)))
    return values


@pytest.fixture
def speaking(fish_home, section):
    """An engine over a model that is up."""
    fish_home.have()
    run = FishLocalModel.ensure(Settings())
    assert wait_until(lambda: run.state == fish.READY)
    return FishLocalVoiceSynthesizer()


def _say(engine, *chunks):
    """Run the synthesis over the text chunks; the PCM that reached the audio queue."""
    for chunk in chunks:
        engine.text_queue.put(chunk)
    engine.text_queue.put(None)
    with patch.object(type(engine), "END_OF_STREAM_GRACE_S", 0):
        engine._run_synthesis()
    audio = []
    while (piece := engine.audio_queue.get_nowait()) is not None:
        audio.append(piece)
    return audio


class TestTagBalance:
    def test_a_whisper_cut_in_two_is_whispered_in_both_units(self):
        tags = TagBalance()
        assert tags.balance("<whisper>The kids are asleep,") == "<whisper>The kids are asleep,</whisper>"
        assert tags.balance(" so I will be quiet.</whisper> Anyway.") == "<whisper> so I will be quiet.</whisper> Anyway."
        assert tags.balance(" Good night.") == " Good night."

    def test_a_unit_that_only_closes_the_last_ones_whisper_starts_with_no_empty_pair(self):
        tags = TagBalance()
        tags.balance("<whisper>The kids are asleep.")
        assert tags.balance("</whisper> [laugh] Anyway.") == " [laugh] Anyway."

    def test_a_tag_closed_where_none_is_open_is_dropped(self):
        assert TagBalance().balance("All done.</whisper>") == "All done."

    def test_nested_tags_close_in_their_own_order(self):
        tags = TagBalance()
        assert tags.balance("<soft><whisper>Shh,") == "<soft><whisper>Shh,</whisper></soft>"
        assert tags.balance(" quiet.</whisper> Louder now.</soft>") == "<soft><whisper> quiet.</whisper> Louder now.</soft>"

    def test_a_cue_cut_in_two_is_dropped_from_both_units(self):
        tags = TagBalance()
        assert tags.balance("The car [sad,") == "The car "
        assert tags.balance(" slowly] is charged.") == " is charged."

    def test_a_whole_cue_is_kept(self):
        assert TagBalance().balance("[sigh] It rained all night.") == "[sigh] It rained all night."

    def test_the_models_own_structure_tokens_never_reach_it(self):
        assert TagBalance().balance("Done.<|im_end|>") == "Done."


class TestReadiness:
    def test_until_the_model_is_up_the_next_engine_speaks_and_this_one_is_never_built(self, fish_home, section, engines, monkeypatch):
        for cls in ENGINES:
            if cls not in (FishLocalVoiceSynthesizer, SupertonicVoiceSynthesizer):
                engines.disable(cls)
        monkeypatch.setattr(FishLocalVoiceSynthesizer, "__init__",
                            lambda self: (_ for _ in ()).throw(AssertionError("built an engine that is not ready")))
        assert isinstance(create_voice("en").impl, SupertonicVoiceSynthesizer)

    def test_once_the_model_is_up_it_speaks_in_front_of_supertonic(self, speaking, engines):
        for cls in ENGINES:
            if cls not in (FishLocalVoiceSynthesizer, SupertonicVoiceSynthesizer):
                engines.disable(cls)
        voice = create_voice("en")
        assert isinstance(voice.impl, FishLocalVoiceSynthesizer) and voice.ready()

    def test_a_model_up_for_another_build_is_not_this_configs(self, speaking, section):
        section["quant"] = "8bit"
        assert not FishLocalVoiceSynthesizer.ready()

    def test_a_run_that_is_still_loading_is_not_ready_and_says_what_it_does(self, fish_home, section):
        fish_home.have()
        with patch.object(FishLocalModel, "start"):
            FishLocalModel.ensure(Settings()).state = fish.LOADING
        assert not FishLocalVoiceSynthesizer.ready()
        assert "loading" in FishLocalVoiceSynthesizer.not_ready_reason()


class TestTheMonitorsView:
    def test_a_mac_too_small_for_the_build_is_listed_not_configured_with_the_numbers(self, fish_home, section, on_a_mac, monkeypatch):
        monkeypatch.setattr(Platform, "memory_gb", lambda: 24)
        FishLocalVoiceSynthesizer.reset_instance()
        try:
            monitor = IntegrationMonitor()
            listed = monitor.register_class(FishLocalVoiceSynthesizer)
            assert listed.name == "Fish Local TTS" and listed.state == IntegrationState.NOT_CONFIGURED
            assert "24" in listed.last_error and "48" in listed.last_error
            assert FishLocalModel.current(Settings()) is None, "no download was started on a Mac that cannot carry it"
            assert monitor.check_now("Fish Local TTS") == IntegrationState.NOT_CONFIGURED, "no health check turns it into an error"
        finally:
            FishLocalVoiceSynthesizer.reset_instance()

    def test_a_reply_through_an_engine_that_cannot_run_ends_silent_instead_of_hanging(self, fish_home, section, monkeypatch):
        monkeypatch.setattr(Platform, "memory_gb", lambda: 24)
        assert _say(FishLocalVoiceSynthesizer(), "Hello there.") == []

    def test_switching_it_on_starts_the_download(self, fish_home, section, on_a_mac):
        FishLocalVoiceSynthesizer.reset_instance()
        try:
            IntegrationMonitor().register_class(FishLocalVoiceSynthesizer)
            assert wait_until(FishLocalVoiceSynthesizer.ready)
            assert fish_home.hub == [fish.BUILDS[fish.DEFAULT_QUANT]]
        finally:
            FishLocalVoiceSynthesizer.reset_instance()

    def test_a_download_in_flight_is_a_warning(self, fish_home, section):
        with patch.object(FishLocalModel, "start"):
            engine = FishLocalVoiceSynthesizer()
        engine.model.state = fish.DOWNLOADING
        with pytest.raises(HealthWarning):
            engine.check_health()

    def test_a_failed_run_is_reported_and_started_again_by_the_health_check(self, fish_home, section):
        fish_home.hub_error = ConnectionError("the hub hung up")
        engine = FishLocalVoiceSynthesizer()
        failed = engine.model
        assert wait_until(lambda: failed.state == fish.FAILED)
        fish_home.hub_error = None
        with pytest.raises(RuntimeError, match="the hub hung up"):
            engine.check_health()
        assert engine.model is not failed and wait_until(FishLocalVoiceSynthesizer.ready)
        assert engine.check_health() is True

    def test_a_restart_unloads_the_model(self, speaking):
        FishLocalVoiceSynthesizer.get_instance()
        FishLocalVoiceSynthesizer.reset_instance()
        assert FishLocalModel.current(Settings()) is None


class TestSynthesis:
    def test_each_unit_is_generated_and_reaches_the_pipeline_as_24k_pcm(self, speaking, fish_home):
        audio = _say(speaking, "Sure, the lights ", "are on. The door is locked.")
        assert fish_home.model.spoken() == ["Sure,", "the lights are on. The door is locked."]
        seconds_in = 2 * fish_home.model.pieces * 441 / 44100
        assert abs(sum(len(piece) for piece in audio) / 2 / 24000 - seconds_in) < 0.001
        assert np.frombuffer(audio[0], dtype=np.int16).max() > 0

    def test_a_whisper_cut_by_the_batcher_reaches_the_model_whole_in_each_unit(self, speaking, fish_home):
        _say(speaking, "<whisper>The kids are asleep, so I will be quiet.</whisper>")
        assert fish_home.model.spoken() == ["<whisper>The kids are asleep,</whisper>",
                                            "<whisper> so I will be quiet.</whisper>"]

    def test_with_emotions_off_the_paired_tags_are_taken_out_and_their_words_spoken(self, speaking, fish_home, monkeypatch):
        monkeypatch.setattr(speaking, "emotions_enabled", lambda: False)
        _say(speaking, "Fine. <whisper>The kids are asleep.</whisper>")
        assert fish_home.model.spoken() == ["Fine.", "The kids are asleep."]

    def test_an_empty_piece_from_the_model_is_not_handed_on_as_audio(self, speaking, fish_home, monkeypatch):
        pieces = [np.full(441, 0.2, dtype=np.float32), np.zeros(0, dtype=np.float32)]
        monkeypatch.setattr(fish_home.model, "generate",
                            lambda text, **_: (SimpleNamespace(audio=piece) for piece in pieces))
        assert [len(piece) > 0 for piece in _say(speaking, "The door is locked.")] == [True]

    def test_a_unit_with_nothing_to_say_is_not_generated(self, speaking, fish_home):
        assert _say(speaking, "…!") == []
        assert fish_home.model.spoken() == []

    def test_a_killed_reply_puts_no_more_audio_out(self, speaking, fish_home):
        speaking.text_queue.put("One. Two. Three.")
        speaking.text_queue.put(None)
        speaking.is_closed = True
        speaking._run_synthesis()
        assert speaking.audio_queue.get_nowait() is None
        assert fish_home.model.spoken() == []

    def test_a_voice_made_anew_gets_phrase_recordings_of_its_own(self, speaking):
        """The phrase cache is keyed by `voice_id`: a re-rolled voice must not answer in the old one's recordings."""
        import os
        before = FishLocalVoiceSynthesizer.voice_id()
        clip = fish.own_voice()[0]
        os.utime(clip, (clip.stat().st_atime, clip.stat().st_mtime + 60))
        assert FishLocalVoiceSynthesizer.voice_id() != before
