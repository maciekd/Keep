import base64
import json
import queue
import threading
from typing import Iterator, List

import requests as _requests

from .WebSocketVoiceSynthesizer import WebSocketVoiceSynthesizer
from assistant.integration.Integration import Integration
from common.logging import error, info, trace
from assistant.prompt.TimeOfDay import TimeOfDay
from common.ConfigManager import config


class ElevenLabsVoiceSynthesizer(WebSocketVoiceSynthesizer, Integration):
    """The default voice, from ElevenLabs, streamed as the answer is written.

    Speaks Keep's replies with ElevenLabs' TTS, streamed while the answer is still being
    written. It is the voice Keep uses unless another engine is enabled.

    Two wire protocols over the websocket API, chosen by the model: the v3/v4 family only
    streams over the text-to-dialogue socket (one registered voice, `inputs` frames,
    `close_socket` to end); the older flash/turbo/multilingual models speak the classic
    stream-input protocol (`bos`/`eos` text frames). `tts.elevenlabs.model` picks the model, so
    switching back to an old one is a config change."""

    CONFIG_SECTION = "elevenlabs"
    DEFAULT_MODEL = "eleven_v4_turbo"
    DIALOGUE_MODEL_PREFIXES = ("eleven_v3", "eleven_v4")

    # On cloud AI the engine every non-Swedish reply speaks through, so a voice install needs only
    # its key: on without an enabled_integrations entry, listed as "not configured" until then. On
    # local AI (`local_backend`) it is off unless listed — the replies would leave the machine.
    @classmethod
    def default_enabled(cls) -> bool:
        from assistant.llm import local_only_mode
        return not local_only_mode()

    @property
    def name(self) -> str:
        return "ElevenLabs TTS"

    @property
    def config_keys(self) -> List[str]:
        return ["api_keys.elevenlabs"]

    def check_health(self) -> bool:
        return _requests.get("https://api.elevenlabs.io/v1/user",
                             headers={"xi-api-key": self.api_key}, timeout=10).status_code == 200

    def __init__(self):
        WebSocketVoiceSynthesizer.__init__(self)
        Integration.__init__(self)
        self.api_key: str = config.get_api_key('elevenlabs')
        self.model_id: str = self.configured_voice("elevenlabs", self.DEFAULT_MODEL, key="model")
        self.optimize_streaming_latency: int = 4  # this seems to win around 100ms, haven't made any proper analysis though, just gut feeling
        self.output_format: str = "pcm_24000"
        # The voice registered on the open dialogue socket: v4 turbo allows one per connection,
        # so every text frame of that connection names this one, not whatever the clock says now.
        self.session_voice_id: str = ""

    def cache_variant(self) -> str:
        return self.model_id

    @property
    def SUPPORTS_EMOTION_TAGS(self) -> bool:  # noqa: N802 - overrides the class flag per model
        """The v3/v4 family performs audio tags ([laughs], [whispers]); the older models
        would read them out, so a config flip back to flash strips them again."""
        return self.uses_dialogue_socket

    @property
    def uses_dialogue_socket(self) -> bool:
        return self.model_id.startswith(self.DIALOGUE_MODEL_PREFIXES)

    def voice_id(self) -> str:
        if not self.whisper_allowed or TimeOfDay.get_current() == TimeOfDay.DAY:
            return WebSocketVoiceSynthesizer.configured_voice("elevenlabs", "QtY3JBOUKEB5xzrRfOKc")  # Maisie, normal speech
        # Night, morning and late evening we whisper on the house speakers
        return WebSocketVoiceSynthesizer.configured_voice("elevenlabs", "geVuXdUpU0lgUukWJCUE", key="night_voice")  # Shannon, ASMR American whisper

    def _get_ws_url(self) -> str:
        if self.uses_dialogue_socket:
            return f"wss://api.elevenlabs.io/v1/text-to-dialogue/stream-input?model_id={self.model_id}&output_format={self.output_format}"
        return f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id()}/stream-input?model_id={self.model_id}&optimize_streaming_latency={self.optimize_streaming_latency}&output_format={self.output_format}"

    def _get_ws_headers(self) -> dict:
        return {"xi-api-key": self.api_key}

    def start(self, llm_text_stream: Iterator[str]) -> Iterator[bytes]:
        """Set up the audio stream with text chunks from the LLM, following original API.
        This takes a few 100ms to do, but we're waiting for the LLM anyway."""
        self.llm_text_stream = llm_text_stream
        self._clear_queue(self.audio_queue)

        if not self.is_connected:
            self.connect()

        threading.Thread(target=self._feed_text_to_queue, daemon=True).start()
        self.audio_stream = self._audio_generator()
        return self.audio_stream

    def _on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)

            if data.get("error"):
                error(f"ElevenLabs ({self.model_id}) rejected the stream: {data['error']}")

            if "audio" in data and data["audio"] is not None:
                audio_bytes = base64.b64decode(data["audio"])
                self.audio_queue.put(audio_bytes)

            # The dialogue socket says is_final, the stream-input socket isFinal
            if data.get("isFinal", False) or data.get("is_final", False):
                import time
                time.sleep(1.0)
                self.audio_queue.put(None)
        except Exception as e:
            error(f"Error processing message: {e}")

    def _on_error(self, ws, e):
        """Handle WebSocket errors"""
        error_msg = str(e)
        if "Connection is already closed" not in error_msg and "'NoneType' object has no attribute 'sock'" not in error_msg:
            error(f"WebSocket error: {e}")
        self.audio_queue.put(None)

    def _on_open(self, ws):
        """Handle WebSocket connection open"""
        self.is_connected = True

        if self.uses_dialogue_socket:
            self.session_voice_id = self.voice_id()
            trace(f"ElevenLabs dialogue socket open: model {self.model_id}, voice {self.session_voice_id}")
            ws.send(json.dumps({
                "voices": [self.session_voice_id],
                "xi_api_key": self.api_key,
            }))
        else:
            info(f"ElevenLabs stream-input socket open: model {self.model_id}")
            ws.send(json.dumps({
                "text": " ",
                "voice_settings": {
                    "stability": 0.3,
                    "similarity_boost": 0.5
                },
                "generation_config": {
                    "chunk_length_schedule": [50, 50, 50, 50],
                    "auto_mode": True,
                },
                "xi_api_key": self.api_key,
                "bos": True
            }))

        threading.Thread(target=self._feed_text_thread, daemon=True).start()

    def _text_frame(self, text_chunk: str) -> dict:
        if self.uses_dialogue_socket:
            return {"inputs": [{"text": text_chunk, "voice_id": self.session_voice_id, "new_turn": False}]}
        return {"text": text_chunk, "try_trigger_generation": True, "bos": False}

    def _end_frame(self) -> dict:
        # close_socket flushes what the dialogue socket still buffers, then answers is_final
        if self.uses_dialogue_socket:
            return {"close_socket": True}
        return {"text": "", "eos": True}

    def _feed_text_thread(self):
        """Thread that feeds text to the WebSocket"""
        try:
            while not self.is_closed:
                try:
                    text_chunk = self.text_queue.get(timeout=0.5)

                    if text_chunk is None:
                        trace(f"ElevenLabs text stream ended; closing the {self.model_id} turn")
                        self.ws.send(json.dumps(self._end_frame()))
                        break

                    if not text_chunk.strip():
                        self.text_queue.task_done()
                        continue

                    self.ws.send(json.dumps(self._text_frame(text_chunk)))

                    self.text_queue.task_done()

                except queue.Empty:
                    continue
        except Exception as e:
            if "Connection is already closed" not in str(e):
                error(f"Error in feed text thread: {e}")
        finally:
            self.text_queue.put(None)
