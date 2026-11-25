import os
import queue
import re
import struct
import threading
from typing import Iterator, Generator

from .VoiceSynthesizer import VoiceSynthesizer
from assistant.integration.Integration import Integration
from common.logging import note, info, fine, error
from common.RequestKilledException import RequestKilledException
from common.ConfigManager import config

class CachedVoiceSynthesizer(VoiceSynthesizer):
    """Serves repeated phrases from a WAV cache and delegates misses to the wrapped VoiceSynthesizer."""

    def __init__(self, impl: VoiceSynthesizer):
        super().__init__()
        self.impl = impl
        self.cache_dir = "cache/voice"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_size_gb = config.get('voice_cache_size_in_gb', 2)
        # Whether the most recent start() served audio from the cache instead of the
        # underlying synthesizer — shown as "(cached)" on the Home tab's Voice synthesis
        # row. Set (cache-hit branch) before any cached chunk is queued, so it is
        # already accurate when the consumer times the first chunk.
        self.last_served_from_cache = False

    def voice_id(self) -> str:
        return self.impl.voice_id()

    def impl_name(self) -> str:
        return self.impl.impl_name()

    def ready(self) -> bool:
        return self.impl.ready()

    def not_ready_reason(self) -> str:
        return self.impl.not_ready_reason()

    def set_request_language(self, lang):
        super().set_request_language(lang)
        self.impl.set_request_language(lang)

    def set_whisper_allowed(self, allowed: bool):
        super().set_whisper_allowed(allowed)
        self.impl.set_whisper_allowed(allowed)

    def _missing_credentials(self) -> list[str]:
        """The wrapped engine's unset config keys (an api key, a service account); empty
        for a local engine or a synthesizer that is not an Integration."""
        if not isinstance(self.impl, Integration):
            return []
        return self.impl.missing_config_keys()

    def _get_cache_filename(self, text: str) -> str:
        clean_text = text.lower()
        clean_text = re.sub(r'[^a-z0-9]', '_', clean_text)
        impl_name = type(self.impl).__name__.lower()
        voice_id = self.impl.voice_id()
        clean_voice_id = re.sub(r'[^a-z0-9]', '_', voice_id.lower())
        variant = re.sub(r'[^a-z0-9]', '_', self.impl.cache_variant().lower())
        suffix = f"_{variant}" if variant else ""
        return os.path.join(self.cache_dir, f"{impl_name}_{clean_text}_{clean_voice_id}{suffix}.wav")

    def start(self, llm_text_stream: Iterator[str]) -> Iterator[bytes]:
        full_text = ""
        self.last_served_from_cache = False
        text_complete_event = threading.Event()
        audio_started_event = threading.Event()
        cache_used_event = threading.Event()
        text_queue = queue.Queue()
        audio_queue = queue.Queue()

        # Thread to collect all text
        def collect_text():
            nonlocal full_text
            try:
                for chunk in llm_text_stream:
                    full_text += chunk
                    text_queue.put(chunk)
                    self.debug("Got token length: {}".format(len(full_text)))
                self.debug("All Text received!")
            except RequestKilledException:
                fine("collect_text aborted: request was killed")
                return
            finally:
                text_queue.put(None)  # Signal end (always, so audio thread doesn't hang)
                text_complete_event.set()

            # Check cache if audio hasn't started
            if not audio_started_event.is_set():
                cache_file = self._get_cache_filename(full_text)
                self.debug("Cache file: {}".format(cache_file))
                if os.path.exists(cache_file):
                    cache_used_event.set()
                    self.last_served_from_cache = True
                    note("Voice response found in cache, serving from cache.")
                    # Touch the file to update access time
                    self._touch_cache_file(cache_file)
                    for chunk in self._read_cache(cache_file):
                        audio_queue.put(chunk)
                    audio_queue.put(None)  # Signal end
                else:
                    self.debug("Not found in cache.")

        # Thread to generate audio if not cached
        def generate_audio():
            def text_generator() -> Generator[str, None, None]:
                while True:
                    # Check if we should stop because cache was found
                    if cache_used_event.is_set():
                        self.debug("Cache found, stopping text generation")
                        break

                    chunk = text_queue.get()
                    if chunk is None:
                        break
                    yield chunk

            missing = self._missing_credentials()
            if missing:
                # Without this the backend opens a websocket with an empty key, the server
                # closes it, and the stream just ends: a silent reply with nothing in the log.
                error(f"TTS engine {self.impl_name()} cannot speak: {', '.join(missing)} is not set "
                      f"in config.yaml — the reply will be silent. Set it (config_request) or use "
                      f"a configured engine.")
                audio_queue.put(None)
                return
            self.debug("Starting audio generation from impl immediately")
            try:
                audio_stream = self.impl.start(text_generator())
                for audio_chunk in audio_stream:
                    # Check if cache was just found
                    if cache_used_event.is_set():
                        self.debug("Cache found, stopping audio generation")
                        # Close the underlying implementation
                        self.impl.close()
                        break

                    audio_started_event.set()
                    audio_queue.put(audio_chunk)

                if not cache_used_event.is_set():
                    audio_queue.put(None)  # Signal end
            except Exception as e:
                self.debug(f"Audio generation stopped: {e}")
                if not cache_used_event.is_set():
                    audio_queue.put(None)

        # Start both threads
        text_thread = threading.Thread(target=collect_text, daemon=True)
        text_thread.start()

        audio_thread = threading.Thread(target=generate_audio, daemon=True)
        audio_thread.start()

        # Yield audio chunks
        audio_chunks = []
        while True:
            chunk = audio_queue.get()
            if chunk is None:
                break
            if not cache_used_event.is_set():
                audio_chunks.append(chunk)
            self.debug(f"yielding audio chunk with length {len(chunk)}")
            yield chunk

        # Save to cache if we generated new audio
        if audio_chunks and not cache_used_event.is_set():
            text_complete_event.wait()  # Make sure we have full text
            if len(full_text) < 200:
                self.debug("Saving to cache")
                cache_file = self._get_cache_filename(full_text)
                audio_data = b''.join(audio_chunks)
                self._save_to_cache(cache_file, audio_data)
                # Cleanup old cache files if needed
                self._cleanup_cache_if_needed()

    @staticmethod
    def _read_cache(filename: str) -> Iterator[bytes]:
        with open(filename, 'rb') as f:
            # Skip WAV header (44 bytes)
            f.seek(44)
            chunk_size = 4096 * 10
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    @staticmethod
    def _save_to_cache(filename: str, audio_data: bytes):
        # Create WAV header for PCM 24kHz, 16-bit, mono
        sample_rate = 24000
        bits_per_sample = 16
        channels = 1
        data_size = len(audio_data)

        # WAV file header
        wav_header = b'RIFF'
        wav_header += struct.pack('<I', data_size + 36)  # File size - 8
        wav_header += b'WAVE'
        wav_header += b'fmt '
        wav_header += struct.pack('<I', 16)  # Subchunk size
        wav_header += struct.pack('<H', 1)  # Audio format (1 = PCM)
        wav_header += struct.pack('<H', channels)  # Number of channels
        wav_header += struct.pack('<I', sample_rate)  # Sample rate
        wav_header += struct.pack('<I', sample_rate * channels * bits_per_sample // 8)  # Byte rate
        wav_header += struct.pack('<H', channels * bits_per_sample // 8)  # Block align
        wav_header += struct.pack('<H', bits_per_sample)  # Bits per sample
        wav_header += b'data'
        wav_header += struct.pack('<I', data_size)  # Data chunk size

        with open(filename, 'wb') as f:
            f.write(wav_header)
            f.write(audio_data)

    def close(self):
        self.impl.close()

    def _join_threads(self):
        pass  # Decorator - impl handles its own threads

    def _touch_cache_file(self, file_path: str):
        """Update the access and modification time of a cache file."""
        try:
            os.utime(file_path, None)
        except Exception as e:
            self.debug(f"Failed to touch cache file: {e}")

    def _get_cache_size_bytes(self) -> int:
        """Calculate total size of cache directory in bytes."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self.cache_dir):
            for filename in filenames:
                if filename.endswith('.wav'):
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except OSError:
                        pass
        return total_size

    def _cleanup_cache_if_needed(self):
        """Check cache size and delete oldest files if over limit."""
        max_size_bytes = self.cache_size_gb * 1024 * 1024 * 1024
        current_size = self._get_cache_size_bytes()

        if current_size <= max_size_bytes:
            return

        # Get all cache files with their access times
        cache_files = []
        for dirpath, dirnames, filenames in os.walk(self.cache_dir):
            for filename in filenames:
                if filename.endswith('.wav'):
                    filepath = os.path.join(dirpath, filename)
                    try:
                        # Use access time for LRU
                        access_time = os.path.getatime(filepath)
                        file_size = os.path.getsize(filepath)
                        cache_files.append((access_time, filepath, file_size))
                    except OSError:
                        pass

        # Sort by access time (oldest first)
        cache_files.sort(key=lambda x: x[0])

        # Delete files until we're under the limit
        for access_time, filepath, file_size in cache_files:
            if current_size <= max_size_bytes:
                break
            try:
                os.remove(filepath)
                current_size -= file_size
                info(f"Removed old cache file: {os.path.basename(filepath)}")
            except OSError as e:
                self.debug(f"Failed to remove cache file {filepath}: {e}")

    def debug(self, msg: str):
        pass
