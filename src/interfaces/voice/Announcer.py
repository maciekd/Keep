"""Speaks a message through the server speakers outside any dialog turn, as a room's destination.

`post_message("kitchen", "The pasta is done.")` ends here: the voice interface registers one
`Announcer` under the local outposts' location (`local_outposts.location`), and the registry hands
it the plain words. An announcement is built from the same parts as a reply — the voice the
factory picks for the house language (the whispering night voice allowed, since these are the
house speakers), a `ServerPlayback` of its own, the outpost mute around the sound so the kitchen
mic never hears her and wakes her — but it has no request behind it, so it owns its own kill flag
(the interface stopping) and waits its turn: while a request is being served or a reply is still
speaking it holds for up to `WAIT_FOR_QUIET` seconds, then gives up and says so, and the command
result tells the model to post elsewhere.

The old repo spoke every autonomous message this way (ServerSpeachPlayback, 2025-11) and the
household got annoyed within two days; what to say aloud is now the model's choice per message,
so there is no daytime, presence or rate gate here.
"""
import threading
import time
from typing import Callable, Optional

from assistant.Destinations import Destination, KIND_VOICE, register, unregister
from common.ConfigManager import config
from common.Language import configured_language
from common.logging import error, info, trace

WAIT_FOR_QUIET = 20.0       # seconds an announcement waits for a running turn or reply to end
QUIET_POLL = 0.1
ALIASES = ("speakers",)


class Announcer:
    """One room's speakers as a destination: synthesizes a text and plays it when the house is quiet."""

    def __init__(self, playback_factory: Callable[[Callable[[], bool]], object],
                 voice_factory: Callable[[Optional[str]], object], busy: Callable[[object], bool],
                 lang: Optional[str] = None, wait_for_quiet: float = WAIT_FOR_QUIET, poll: float = QUIET_POLL):
        """*playback_factory* and *voice_factory* are the voice interface's (the ones every
        handler gets); *busy(sender)* says whether a request other than *sender* — the
        RequestContext of the turn posting, None outside a turn — is being served or a reply
        is speaking."""
        self._playback_factory = playback_factory
        self._voice_factory = voice_factory
        self._busy = busy
        self._lang = lang or configured_language()
        self._wait_for_quiet = wait_for_quiet
        self._poll = poll
        self._stopping = False
        self._lock = threading.Lock()   # one announcement at a time; a second waits for the first

    def stop(self):
        self._stopping = True

    def announce(self, text: str, sender=None) -> bool:
        """Say *text* through the speakers. True once it played to the end; False, logged, for
        everything else — never raises into the command. *sender* is the turn posting it: a
        voice turn's own post_message does not wait for that turn to end."""
        if not text or not text.strip():
            trace("Announcer: empty text, nothing to say")
            return False
        with self._lock:
            if not self._wait_for_quiet_house(sender):
                info(f"Announcer: the house was busy for {self._wait_for_quiet:.0f}s, not announcing: {text[:80]!r}")
                return False
            if self._stopping:
                info("Announcer: the voice interface is stopping, not announcing")
                return False
            return self._play(text)

    def _wait_for_quiet_house(self, sender) -> bool:
        deadline = time.time() + self._wait_for_quiet
        waited = False
        while self._busy(sender):
            if self._stopping or time.time() >= deadline:
                return False
            if not waited:
                trace("Announcer: a turn or a reply is in progress, waiting for quiet before speaking")
                waited = True
            time.sleep(self._poll)
        return True

    def _play(self, text: str) -> bool:
        voice = self._voice_factory(self._lang)
        if voice is None:
            info("Announcer: no TTS engine can speak, not announcing")
            return False
        voice.set_request_language(self._lang)
        voice.set_whisper_allowed(True)
        playback = self._playback_factory(lambda: self._stopping)
        chunks = 0
        try:
            playback.ensure_outposts_muted()
            playback.ensure_started()
            for chunk in voice.start(iter([text])):
                if self._stopping:
                    info("Announcer: stopping mid-announcement")
                    return False
                playback.enqueue(chunk)
                chunks += 1
            if chunks == 0:
                info("Announcer: the voice produced no audio, nothing was said")
                return False
            # Drains the audio and reopens the mics: 'listen' when the text asks a question.
            playback.finish_reply(text)
            info(f"Announcer: said {len(text)} chars in {chunks} chunks: {text[:80]!r}")
            return True
        except Exception as e:
            error(f"Announcer: announcement failed with {type(e).__name__}: {e}")
            return False
        finally:
            # A no-op after a finished reply; after a failure it is what reopens the mics.
            self._quietly(playback.restore_outposts_if_muted, "restore outposts")
            self._quietly(playback.stop, "stop playback")
            self._quietly(voice.close, "close voice")

    @staticmethod
    def _quietly(action: Callable[[], None], what: str):
        try:
            action()
        except Exception as e:
            trace(f"Announcer: {what} raised {type(e).__name__}: {e}")


def destination_name() -> Optional[str]:
    """The name the room's speakers answer to, or None when there is no room to register:
    local outposts off (the server has no mics of its own, so no speakers in a room either)
    or no `local_outposts.location` to name it by."""
    if not config.get('local_outposts.enabled', True):
        trace("Announcer: local outposts are disabled, no room to register as a destination")
        return None
    location = (config.get('local_outposts.location', "") or "").strip().lower()
    if not location:
        info("Announcer: local_outposts.location is not set, the speakers are not a destination")
        return None
    return location


def register_room(announcer: Announcer) -> Optional[str]:
    """Register *announcer* under the room's name. Returns the name, or None when not registered."""
    name = destination_name()
    if name is None:
        return None
    register(Destination(name=name, kind=KIND_VOICE, send=announcer.announce, aliases=ALIASES, sender_aware=True))
    return name


def unregister_room(name: Optional[str]):
    if name:
        unregister(name)
