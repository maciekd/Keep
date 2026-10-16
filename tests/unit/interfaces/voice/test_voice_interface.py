"""The voice interface's room: the server speakers registered as a post_message destination,
busy for another voice turn or a reply still speaking, never for the turn that is posting."""
import pytest
from unittest.mock import Mock, patch

from assistant.ChannelProfile import SOURCE_VOICE
from interfaces.voice import VoiceInterface as module
from interfaces.voice.server_audio.ServerPlayback import ServerPlayback


def _interface(**attrs):
    vi = module.VoiceInterface.__new__(module.VoiceInterface)
    vi.running_as_testsuite = False
    vi.local_outpost_manager = Mock()
    vi.playback_factory = Mock()
    vi.volume_controller = None
    vi.voice_discovery = None
    vi.announcer = None
    vi._room = None
    for name, value in attrs.items():
        setattr(vi, name, value)
    return vi


@pytest.mark.unit
class TestRegisterRoom:
    def test_the_room_is_busy_for_another_voice_turn_or_a_speaking_reply_never_for_the_posting_turn(self):
        """A post_message from a phone outpost's voice turn runs inside that turn: an announcer
        waiting for every voice turn would wait for itself, so the sender is the one turn
        excepted. A reply still holding the outpost mute is speaking, and counts."""
        assistant = Mock()
        vi = _interface()
        with patch.object(module, "Announcer") as announcer_cls, \
                patch.object(module, "register_room", return_value="kitchen") as register:
            vi._register_room(assistant)
        register.assert_called_once_with(announcer_cls.return_value)
        assert vi._room == "kitchen" and vi.announcer is announcer_cls.return_value

        busy = announcer_cls.call_args.kwargs["busy"]
        sender = object()
        assistant.is_busy.side_effect = lambda source=None, except_ctx=None: (
            source == SOURCE_VOICE and except_ctx is not sender)
        with patch.object(ServerPlayback, "_active_mute_generation", 0):
            assert busy(sender) is False, "the turn posting is not the house being busy"
            assert busy(None) is True, "a voice turn of someone else is"
        assistant.is_busy.side_effect = lambda source=None, except_ctx=None: False
        with patch.object(ServerPlayback, "_active_mute_generation", 3):
            assert busy(sender) is True, "a reply still speaking holds the mute"

    @pytest.mark.parametrize("attrs", [{"running_as_testsuite": True}, {"local_outpost_manager": None}])
    def test_without_local_outposts_the_speakers_are_no_destination(self, attrs):
        vi = _interface(**attrs)
        with patch.object(module, "register_room") as register:
            vi._register_room(Mock())
        register.assert_not_called()
        assert vi.announcer is None and vi._room is None

    def test_stop_with_a_voice_discovery_running_unhooks_it_and_still_reaches_the_announcer(self):
        """A voice start outside the testsuite always has a discovery: its stop is the first
        thing stop() does, and the announcer, the room and the audio device come after it."""
        from assistant.commands.DebugCommands import DebugCommands
        vi = _interface(voice_discovery=Mock(), announcer=Mock(), _room="kitchen")
        DebugCommands.voice_discovery = vi.voice_discovery.run
        try:
            with patch.object(module, "unregister_room") as unregister:
                vi.stop()
        finally:
            DebugCommands.voice_discovery = None
        vi.voice_discovery.stop.assert_called_once()
        vi.announcer.stop.assert_called_once()
        unregister.assert_called_once_with("kitchen")

    def test_stop_silences_the_announcer_and_drops_the_room(self):
        vi = _interface()
        with patch.object(module, "Announcer"), patch.object(module, "register_room", return_value="kitchen"):
            vi._register_room(Mock())
        with patch.object(module, "unregister_room") as unregister:
            vi.stop()
        vi.announcer.stop.assert_called_once()
        unregister.assert_called_once_with("kitchen")
        assert vi._room is None
