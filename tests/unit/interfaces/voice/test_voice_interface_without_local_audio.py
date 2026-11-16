"""The voice interface on a server that is not a Mac: the phones' side starts, and nothing of
the server's own microphones and speakers is imported or opened."""
import asyncio
import importlib
import sys
from unittest.mock import Mock

import pytest
from quart import Quart

from common import Platform

MAC_ONLY_LIBRARIES = ("sounddevice", "mlx_whisper", "silero_vad", "lingua")
VOICE = "interfaces.voice"
LOCAL_HALF = (f"{VOICE}.server_audio.", f"{VOICE}.microphones.local_outposts.LocalOutpost")


@pytest.fixture
def voice_module(monkeypatch):
    """`interfaces.voice.VoiceInterface` imported afresh — the whole voice package with it — as
    *the caller's platform* imports it, with the libraries only a Mac installs refused. The
    modules the rest of the suite holds are put back afterwards."""
    held = {name: module for name, module in sys.modules.items() if name == VOICE or name.startswith(VOICE + ".")}
    package_attribute = getattr(sys.modules.get("interfaces"), "voice", None)
    for name in held:
        del sys.modules[name]
    for library in MAC_ONLY_LIBRARIES:
        monkeypatch.setitem(sys.modules, library, None)     # None makes `import` raise ImportError

    def load():
        return importlib.import_module(f"{VOICE}.VoiceInterface")

    try:
        yield load
    finally:
        for name in [n for n in sys.modules if n == VOICE or n.startswith(VOICE + ".")]:
            del sys.modules[name]
        sys.modules.update(held)
        if package_attribute is not None:
            sys.modules["interfaces"].voice = package_attribute


def _started(module, monkeypatch, app):
    """A VoiceInterface started the way the launcher starts it, the slow parts of the phones'
    side (the speaker models, spaCy, the TTS engines' registration) stubbed."""
    from assistant.commands.Commands import Commands
    from assistant.commands.DebugCommands import DebugCommands
    monkeypatch.setattr(Commands, "false_positive_reporter", None, raising=False)
    monkeypatch.setattr(DebugCommands, "voice_discovery", None, raising=False)
    monkeypatch.setattr(module, "preload_spacy", lambda: None)
    monkeypatch.setattr(module, "VoiceDiscovery", Mock())
    monkeypatch.setattr(module.VoiceInterface, "_init_speaker_identifier", staticmethod(lambda: None))
    monkeypatch.setattr(module.VoiceInterface, "_register_tts_integrations", staticmethod(lambda: None))
    interface = module.VoiceInterface(8888)
    interface.start(Mock(), app)
    return interface


@pytest.mark.unit
def test_on_linux_the_interface_starts_without_importing_the_servers_own_audio(voice_module, monkeypatch):
    monkeypatch.setattr(Platform, "current", lambda: Platform.LINUX)
    module = voice_module()
    app = Quart("voice-on-linux")

    interface = _started(module, monkeypatch, app)

    imported = sorted(name for name in sys.modules if name.startswith(LOCAL_HALF))
    assert imported == [], "the server's own audio was imported on a system that has none"
    assert "playback_factory" not in interface.handler_kwargs(), "a handler would build a ServerPlayback"
    assert interface.local_outpost_manager is None and interface.volume_controller is None
    assert interface.announcer is None, "the speakers must not be a post_message destination"
    assert interface.microphone_routes is not None, "the phones' vote and speaker id still need their routes"

    async def serve_and_stop():
        await interface.start_async()
        answer = await app.test_client().get("/end")     # the app's server-sound mode asks for the end sound
        await interface.stop_async()
        return answer.status_code

    assert asyncio.run(serve_and_stop()) == 204
    assert interface._outpost_server_task is None, "the Outpost TCP server serves the local microphones alone"
    interface.stop()


@pytest.mark.unit
def test_on_a_mac_the_same_start_brings_the_local_half(voice_module, monkeypatch):
    """The other side of the switch, so the test above cannot pass by the switch being stuck."""
    monkeypatch.setattr(Platform, "current", lambda: Platform.MAC)
    for library in MAC_ONLY_LIBRARIES[:1]:
        monkeypatch.setitem(sys.modules, library, Mock())     # the unit suite's sounddevice stand-in
    module = voice_module()
    monkeypatch.setattr(f"{VOICE}.microphones.local_outposts.bootstrap.start_local_outposts", lambda *a, **k: None)
    volume = Mock()
    monkeypatch.setattr(f"{VOICE}.server_audio.DynamicServerVolumeController.DynamicServerVolumeController.get_instance",
                        lambda **kwargs: volume)

    interface = _started(module, monkeypatch, Quart("voice-on-a-mac"))

    assert "playback_factory" in interface.handler_kwargs()
    assert interface.play_sound is not None and interface.volume_controller is volume
    volume.start.assert_called_once()
