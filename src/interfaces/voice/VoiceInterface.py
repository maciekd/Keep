"""The VoiceToVoice interface: microphones (remote outposts + local mics), cross-mic
voting, speaker identification, server-speaker playback, TTS and the iOS overlay channel.

Everything voice-related is imported here and nowhere else at process level: unless
`interfaces.voice.enabled: true` is set (voice is off by default) the launcher never imports
this module, so no audio, Whisper, torch or spaCy library is loaded (a Telegram-only install
runs on the base requirements alone)."""
import asyncio
import threading

from assistant.ChannelProfile import SOURCE_VOICE
from common.ConfigManager import config
from common.logging import info, trace
from interfaces.Interface import Interface
from interfaces.voice.Announcer import Announcer, register_room, unregister_room
from interfaces.voice.ios.TextStreamRegistry import TextStreamRegistry
from interfaces.voice.microphones.FalsePositiveFlood import report_false_positive_flood
from interfaces.voice.microphones.local_outposts import start_local_outposts
from interfaces.voice.microphones.OutpostServer import outpost_server
from interfaces.voice.microphones.speech_analysis.SentenceCompletenessDetector import preload_spacy
from interfaces.voice.microphones.speech_analysis.SpeechAnalyzer import SpeechAnalyzer
from interfaces.voice.server_audio.DynamicServerVolumeController import DynamicServerVolumeController
from interfaces.voice.server_audio.PlaySound import PlaySound
from interfaces.voice.server_audio.ServerPlayback import ServerPlayback
from interfaces.voice.speaker_id.VoiceDiscovery import VoiceDiscovery
from interfaces.voice.tts.VoiceFactory import create_voice
from interfaces.voice.VoiceTurnGate import VoiceTurnGate

OUTPOST_TCP_PORT = 43523


class VoiceInterface(Interface):
    """Runs the whole VoiceToVoice stack: microphones, cross-mic voting, speaker id, TTS and playback.

    Everything voice-related (Whisper, torch, spaCy, sounddevice) is imported here and nowhere
    else at process level, so a Telegram-only install boots on the base requirements alone.
    """

    name = "voice"
    enabled_by_default = False

    def __init__(self, port: int, running_as_testsuite: bool = False):
        self.port = port
        self.running_as_testsuite = running_as_testsuite
        self.speech_analyzer = SpeechAnalyzer(running_as_testsuite)
        self.play_sound = PlaySound()
        self.outpost_server = outpost_server
        self.text_stream = TextStreamRegistry.get_instance()
        self.volume_controller = None
        self.local_outpost_manager = None
        self.voice_discovery = None
        self.microphone_routes = None
        self._outpost_server_task = None
        self.announcer: Announcer | None = None
        self._room: str | None = None   # the destination name the speakers are registered under

    # --- RequestProcessor wiring ------------------------------------------------------

    def playback_factory(self, is_killed) -> ServerPlayback:
        """Per-handler server-speaker playback; the only place ServerPlayback is constructed."""
        return ServerPlayback(self.outpost_server, self.speech_analyzer, is_killed)

    def turn_gate_factory(self, is_killed, kill_reason) -> VoiceTurnGate:
        """Per-handler reply-timing gate; the only place VoiceTurnGate is constructed."""
        return VoiceTurnGate(self.speech_analyzer, is_killed, kill_reason)

    def handler_kwargs(self) -> dict:
        """The RequestProcessor constructor arguments this interface provides."""
        return dict(playback_factory=self.playback_factory, voice_factory=create_voice,
                    turn_gate_factory=self.turn_gate_factory, text_stream=self.text_stream)

    # --- lifecycle --------------------------------------------------------------------

    def start(self, assistant, app) -> None:
        from assistant.commands.Commands import Commands
        from assistant.commands.DebugCommands import DebugCommands
        from common.Language import configured_languages
        from interfaces.voice.ios.routes import register as register_ios_routes
        from interfaces.voice.microphones.routes import MicrophoneDeps, register as register_microphone_routes

        preload_spacy()
        Commands.false_positive_reporter = report_false_positive_flood
        self.local_outpost_manager = start_local_outposts(self.port, self.running_as_testsuite)
        self.volume_controller = DynamicServerVolumeController.get_instance(prompt_manager=assistant.prompt_manager)

        if self.running_as_testsuite:
            info("Voice interface in testsuite mode: no speaker id, volume controller or TTS monitoring.")
        else:
            threading.Thread(target=self._init_speaker_identifier, name="SpeakerIdInit", daemon=True).start()
            self.voice_discovery = VoiceDiscovery()
            self.voice_discovery.start()
            DebugCommands.voice_discovery = self.voice_discovery.run
            self.volume_controller.start()
            self._register_tts_integrations()

        self.microphone_routes = register_microphone_routes(app, MicrophoneDeps(
            running_handlers=assistant.running_handlers,
            speech_analyzer=self.speech_analyzer,
            play_sound=self.play_sound,
            wake_word=config.get('wake_word', 'Michelle'),
            configured_languages=configured_languages,
            volume_controller=self.volume_controller,
        ))
        register_ios_routes(app, self.text_stream)
        self._register_room(assistant)
        info("Voice interface started.")

    def _register_room(self, assistant):
        """The server speakers as a `post_message` destination, named by the outposts' location.
        Busy means a voice turn other than the one posting is being served, or a reply still
        holds the outpost mute (speaking) — not any request: a post_message from a Telegram
        turn runs inside that turn, and one from a phone outpost's voice turn inside that
        voice turn; waiting for either would wait for itself."""
        if self.running_as_testsuite or self.local_outpost_manager is None:
            trace("Voice: no local outposts, the speakers are not a destination")
            return
        self.announcer = Announcer(
            self.playback_factory, create_voice,
            busy=lambda sender: (assistant.is_busy(SOURCE_VOICE, except_ctx=sender)
                                 or ServerPlayback._active_mute_generation != 0))
        self._room = register_room(self.announcer)

    @staticmethod
    def _init_speaker_identifier():
        from interfaces.voice.speaker_id import SpeakerIdentifier
        trace("Initializing speaker identification...")
        SpeakerIdentifier.create_instance()

    @staticmethod
    def _register_tts_integrations():
        """Every TTS engine goes through the monitor like any other integration: by class,
        so `enabled_integrations` decides which are built (ElevenLabs is on by default),
        `list_integrations` shows the rest as disabled and `restart_integration` can rebuild
        one. The singleton is the health probe; VoiceFactory builds its own instances."""
        from assistant.integration.IntegrationMonitor import IntegrationMonitor
        from interfaces.voice.tts.VoiceFactory import ENGINES

        monitor = IntegrationMonitor.get_instance()
        for cls in ENGINES:
            monitor.register_class(cls)
        info(f"TTS engines registered with the integration monitor: {', '.join(c.config_name() for c in ENGINES)}")

    async def start_async(self) -> None:
        info(f"Starting Outpost server on port {OUTPOST_TCP_PORT}...")
        self._outpost_server_task = asyncio.create_task(self.outpost_server.start())

    async def stop_async(self) -> None:
        if self.local_outpost_manager is not None:
            self.local_outpost_manager.stop()
        await self.outpost_server.stop()
        if self._outpost_server_task is not None:
            self._outpost_server_task.cancel()
        info("Voice interface stopped.")

    def stop(self) -> None:
        """Stop all audio streams before Python teardown to prevent CoreAudio callback crashes."""
        if self.voice_discovery is not None:
            from assistant.commands.DebugCommands import DebugCommands
            self.voice_discovery.stop()
            DebugCommands.voice_discovery = None
        if self.announcer is not None:
            self.announcer.stop()
        unregister_room(self._room)
        self._room = None
        import sounddevice as sd
        try:
            sd.stop()
        except Exception:
            pass
        try:
            if self.volume_controller is not None:
                self.volume_controller.stop()
        except Exception:
            pass
