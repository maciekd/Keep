"""The voice interface's room: the server speakers registered as a post_message destination."""
import pytest
from unittest.mock import Mock, patch

from assistant.ChannelProfile import SOURCE_VOICE
from interfaces.voice import VoiceInterface as module
from interfaces.voice.server_audio.ServerPlayback import ServerPlayback


def _interface(**attrs):
    vi = module.VoiceInterface.__new__(module.VoiceInterface)
    vi.running_as_testsuite = False
    vi.local_outpost_manager = Mock()
    vi.volume_controller = None
    vi.announcer = None
    vi._room = None
    for name, value in attrs.items():
        setattr(vi, name, value)
    return vi


@pytest.mark.unit
class TestRegisterRoom:
    def test_the_room_waits_for_a_voice_turn_or_a_speaking_reply_not_for_any_request(self):
        """A post_message from a Telegram turn runs inside that turn: an announcer waiting
        for any request would wait for itself. It waits for a voice turn, or for a reply
        that still holds the outpost mute (speaking)."""
        assistant = Mock()
        assistant.is_busy.side_effect = lambda source=None: source is None    # a chat turn is in flight
        vi = _interface()
        with patch.object(module, "Announcer") as announcer_cls, \
                patch.object(module, "register_room", return_value="kitchen") as register:
            vi._register_room(assistant)
        register.assert_called_once_with(announcer_cls.return_value)
        assert vi._room == "kitchen" and vi.announcer is announcer_cls.return_value

        busy = announcer_cls.call_args.kwargs["busy"]
        with patch.object(ServerPlayback, "_active_mute_generation", 0):
            assert busy() is False
            assistant.is_busy.side_effect = lambda source=None: source == SOURCE_VOICE
            assert busy() is True
        assistant.is_busy.side_effect = lambda source=None: False
        with patch.object(ServerPlayback, "_active_mute_generation", 3):
            assert busy() is True, "a reply still speaking holds the mute"

    @pytest.mark.parametrize("attrs", [{"running_as_testsuite": True}, {"local_outpost_manager": None}])
    def test_without_local_outposts_the_speakers_are_no_destination(self, attrs):
        vi = _interface(**attrs)
        with patch.object(module, "register_room") as register:
            vi._register_room(Mock())
        register.assert_not_called()
        assert vi.announcer is None and vi._room is None

    def test_stop_silences_the_announcer_and_drops_the_room(self):
        vi = _interface()
        with patch.object(module, "Announcer"), patch.object(module, "register_room", return_value="kitchen"):
            vi._register_room(Mock())
        with patch.object(module, "unregister_room") as unregister:
            vi.stop()
        vi.announcer.stop.assert_called_once()
        unregister.assert_called_once_with("kitchen")
        assert vi._room is None
