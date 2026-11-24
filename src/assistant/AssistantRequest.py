import time
from dataclasses import dataclass
from typing import Any, Optional

from assistant.ChannelProfile import SOURCE_TEXT_CHAT, SOURCE_VOICE
from common.ConfigManager import config
from common.logging import trace

# The `user` of a turn the server itself starts (the first-run brief): the history stores
# it as a system row, which the prompt and the console render as "System (…)".
SYSTEM_USER = "system"


@dataclass
class RequestContext:
    """What the assistant needs to know about one request: who is asking, from where,
    through which channel (`source`, see ChannelProfile), in which language, and the
    delivery flags the interface asked for. Built by each interface; the HTTP interface
    uses `from_request_data`."""
    source: Optional[str] = None
    # The chat the request came from, in the interface's own id space (a Telegram chat id,
    # the TUI pane): command echoes are routed back to it. None outside the chat interfaces.
    conversation: Any = None
    # The request is from someone who is not one of the household's owners (a member of a
    # Telegram group the bot is in, not on the allow list): every command is refused for
    # the turn (CommandProcessor), and the chat layer states the facts above the message.
    guest: bool = False
    # Facts the chat layer states about the message (a guest, a group chat): sent in the
    # note row in front of the query, as the system's words, never inside the person's text.
    note: Optional[str] = None
    location: Optional[str] = None
    user: Optional[str] = None
    user_input: Optional[str] = None
    lang: Optional[str] = None
    text_only: bool = False
    server_sound: bool = False
    ai_override: Optional[str] = None
    test: bool = False
    images: Optional[list] = None
    engine: Optional[str] = None
    # Epoch of the moment the user actually stopped speaking, reconstructed on arrival
    # from the outpost's stt_ms (ms from speech end to transcript send). None for clients
    # that don't report it (phones).
    speech_end_epoch: Optional[float] = None
    # Set when a `post_message` of this turn went to the conversation it came from: the
    # reply is said, and the request owes no round to say it (RequestProcessor._reply_owed).
    answered_by_post: bool = False

    @property
    def display_name(self) -> str:
        return self.user or "User"

    @staticmethod
    def from_request_data(request_data) -> 'RequestContext':
        """HTTP-specific: User-Agent sniffing plus the `?text` / `?server_sound` / `?test`
        query flags. Belongs to interfaces/http; Phase 5 may move it there."""
        args = request_data.args or {}
        headers = request_data.headers or {}
        user_agent = headers.get('User-Agent', '')

        explicit_source = args.get('source')
        explicit_location = args.get('location')

        ios_app = 'CFNetwork' in user_agent
        android_app = 'okhttp' in user_agent.lower() or 'android' in user_agent.lower()

        if explicit_source is not None:
            source = explicit_source
        elif ios_app or android_app:
            source = SOURCE_VOICE
        else:
            source = SOURCE_TEXT_CHAT

        if explicit_location is not None:
            location = explicit_location
        elif android_app:
            location = config.get('clients.android_location', "") or None
        else:
            location = None

        json_body = getattr(request_data, 'json_body', None) or {}
        images = json_body.get('images') if json_body else None

        speech_end_epoch = None
        stt_ms = args.get('stt_ms')
        if stt_ms is not None:
            try:
                stt_s = float(stt_ms) / 1000.0
                if stt_s < 0:
                    # A negative stt_ms would put speech_end in the future, and the
                    # natural-timing wait sleeps uncapped until it — a stalled worker
                    # from one bad client value. Clamp to "speech just ended".
                    trace(f"Ignoring negative stt_ms={stt_ms!r} — clamping speech end to now.")
                    stt_s = 0.0
                speech_end_epoch = time.time() - stt_s
            except (ValueError, TypeError):
                pass

        return RequestContext(
            source=source,
            location=location,
            user=args.get('user'),
            user_input=args.get('query'),
            lang=args.get('lang'),
            text_only='text' in args,
            server_sound='server_sound' in args,
            ai_override=args.get('ai', None),
            test='test' in args,
            images=images if images else None,
            engine=args.get('engine'),
            speech_end_epoch=speech_end_epoch,
        )
