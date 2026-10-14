"""The room's speakers as a post_message destination: an announcement is synthesized and
played when the house is quiet, the mics are muted around it, and the registration follows
the local outposts' location. Playback, voice and the busy signal are stand-ins — no audio."""
from unittest.mock import patch

import pytest

from assistant import Destinations
from interfaces.voice import Announcer as announcer_module
from interfaces.voice.Announcer import Announcer, register_room, unregister_room


class FakePlayback:
    """Records the calls a reply makes on ServerPlayback, in order."""

    def __init__(self, is_killed):
        self.is_killed = is_killed
        self.calls: list = []
        self.chunks: list[bytes] = []

    def ensure_outposts_muted(self):
        self.calls.append("mute")

    def ensure_started(self):
        self.calls.append("start")

    def enqueue(self, chunk):
        self.chunks.append(chunk)

    def finish_reply(self, text):
        self.calls.append(("finish", text))

    def restore_outposts_if_muted(self):
        self.calls.append("restore")

    def stop(self):
        self.calls.append("stop")


class FakeVoice:
    def __init__(self, chunks=(b"a", b"b", b"c"), fail=False):
        self._chunks = list(chunks)
        self._fail = fail
        self.spoken: list[str] = []
        self.lang = None
        self.whisper_allowed = None
        self.closed = False

    def set_request_language(self, lang):
        self.lang = lang

    def set_whisper_allowed(self, allowed):
        self.whisper_allowed = allowed

    def start(self, text_stream):
        self.spoken.extend(text_stream)
        if self._fail:
            raise RuntimeError("synth down")
        return iter(self._chunks)

    def close(self):
        self.closed = True


def make(busy=lambda sender: False, voice=None, wait=0.05, poll=0.005):
    playbacks: list[FakePlayback] = []

    def playback_factory(is_killed):
        playbacks.append(FakePlayback(is_killed))
        return playbacks[-1]

    voice = voice if voice is not None else FakeVoice()
    announcer = Announcer(playback_factory, lambda lang: voice, busy, lang="en-US", wait_for_quiet=wait, poll=poll)
    return announcer, playbacks, voice


@pytest.fixture(autouse=True)
def clean_registry():
    Destinations.clear()
    yield
    Destinations.clear()


@pytest.mark.unit
class TestAnnounce:
    def test_every_chunk_is_played_and_the_reply_is_finished_with_the_text(self):
        announcer, playbacks, voice = make()
        assert announcer.announce("The pasta is done.") is True
        playback = playbacks[0]
        assert voice.spoken == ["The pasta is done."]
        assert playback.chunks == [b"a", b"b", b"c"]
        assert playback.calls.index("mute") < playback.calls.index("start")
        assert ("finish", "The pasta is done.") in playback.calls
        assert playback.calls[-2:] == ["restore", "stop"]
        assert voice.closed

    def test_the_night_whisper_is_allowed_and_the_house_language_is_set(self):
        announcer, _, voice = make()
        announcer.announce("Good night.")
        assert voice.whisper_allowed is True
        assert voice.lang == "en-US"

    def test_a_live_turn_makes_it_wait_and_then_speak(self):
        state = {"busy": 2}

        def busy(sender):
            state["busy"] -= 1
            return state["busy"] >= 0

        announcer, playbacks, _ = make(busy=busy)
        assert announcer.announce("Hello") is True
        assert playbacks[0].chunks

    def test_a_turn_that_never_ends_gives_up_without_playing(self):
        announcer, playbacks, voice = make(busy=lambda sender: True, wait=0.02)
        assert announcer.announce("Hello") is False
        assert playbacks == []
        assert voice.spoken == []

    def test_the_turn_posting_the_message_is_not_the_house_being_busy(self):
        # A spoken "tell the kitchen…" runs post_message inside its own voice turn: the busy
        # signal is asked about everyone but that turn, so it speaks instead of waiting on itself.
        turn = object()
        asked = []

        def busy(sender):
            asked.append(sender)
            return sender is not turn

        announcer, playbacks, _ = make(busy=busy, wait=0.02)
        assert announcer.announce("Hello", sender=turn) is True
        assert playbacks[0].chunks and asked == [turn]
        assert announcer.announce("Hello") is False, "outside a turn the busy house still wins"

    def test_outposts_are_restored_when_synthesis_raises(self):
        announcer, playbacks, _ = make(voice=FakeVoice(fail=True))
        assert announcer.announce("Hello") is False
        assert "restore" in playbacks[0].calls
        assert "stop" in playbacks[0].calls

    def test_a_voice_that_produces_no_audio_is_not_a_success(self):
        announcer, playbacks, _ = make(voice=FakeVoice(chunks=()))
        assert announcer.announce("Hello") is False
        assert not any(isinstance(c, tuple) and c[0] == "finish" for c in playbacks[0].calls)

    def test_no_tts_engine_means_nothing_is_said(self):
        announcer = Announcer(lambda k: FakePlayback(k), lambda lang: None, lambda sender: False, lang="en-US")
        assert announcer.announce("Hello") is False

    def test_empty_text_is_refused(self):
        announcer, playbacks, _ = make()
        assert announcer.announce("   ") is False
        assert playbacks == []

    def test_stopping_the_interface_cuts_an_announcement_short(self):
        announcer, playbacks, _ = make()

        class StoppingVoice(FakeVoice):
            def start(self_inner, text_stream):
                list(text_stream)
                announcer.stop()
                return iter([b"a", b"b"])

        announcer._voice_factory = lambda lang: StoppingVoice()
        assert announcer.announce("Hello") is False
        assert playbacks[0].chunks == []
        assert "restore" in playbacks[0].calls


@pytest.mark.unit
class TestRegistration:
    def _config(self, values):
        return patch.object(announcer_module.config, 'get',
                            side_effect=lambda key, default=None: values.get(key, default))

    def test_the_room_is_named_by_the_outposts_location_and_answers_to_speakers(self):
        announcer, playbacks, voice = make()
        with self._config({'local_outposts.location': 'Kitchen'}):
            assert register_room(announcer) == "kitchen"
        assert Destinations.find("Kitchen") is not None
        assert Destinations.find("speakers").kind == Destinations.KIND_VOICE
        assert Destinations.send("kitchen", "<b>The pasta</b> is done.") == ""
        assert playbacks[0].chunks and voice.spoken == ["The pasta is done."]

    def test_disabled_outposts_register_no_room(self):
        announcer, _, _ = make()
        with self._config({'local_outposts.enabled': False, 'local_outposts.location': 'Kitchen'}):
            assert register_room(announcer) is None
        assert Destinations.names() == []

    def test_no_location_registers_no_room(self):
        announcer, _, _ = make()
        with self._config({'local_outposts.location': ''}):
            assert register_room(announcer) is None
        assert Destinations.names() == []

    def test_unregister_drops_the_room_and_tolerates_none(self):
        announcer, _, _ = make()
        with self._config({'local_outposts.location': 'Kitchen'}):
            name = register_room(announcer)
        unregister_room(name)
        unregister_room(None)
        assert Destinations.names() == []
