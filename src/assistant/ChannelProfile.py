"""Per-channel policy table — the one place the assistant core looks at *where* a request
came from. Everything that used to compare `RequestContext.source` against a string reads
a `ChannelProfile` field instead, so adding a channel is a table row, not a code hunt.

This is policy, not wire format: how an interface renders the event stream for a given
channel is its own business, keyed off `streams_tokens`.
"""
from dataclasses import dataclass
from typing import Optional

from common.logging import info, trace

SOURCE_VOICE = "VoiceToVoice"
SOURCE_TEXT_CHAT = "TextChat"
SOURCE_TELEGRAM = "Telegram"
SOURCE_CONSOLE = "Console"
SOURCE_IMESSAGE = "iMessage"
SOURCE_AUTONOMOUS = "autonomous"

PHOTO_TARGET_SCREEN = "screen"
PHOTO_TARGET_CHAT = "chat"

# The history windows (ChannelProfile.window): one conversation, rendered for a voice request
# with the voice preloads in it and for a text request without them (MessageHistory.visible).
WINDOW_VOICE = "voice"
WINDOW_TEXT = "text"


@dataclass(frozen=True)
class ChannelProfile:
    """Describes how the assistant core must treat requests arriving from one channel."""
    name: str
    # "fast" | "smart": the LLM tier when the request neither carries images nor overrides.
    llm_tier: str
    # Fold "Make sure your response is under 20 words!" onto the request.
    brevity_directive: bool
    # Forward <thinking> chunks to the observer hub (dashboards watching TextChat requests).
    forward_thinking: bool
    # Echo every executed command back to the chat the request came from (the user is
    # reading, not listening).
    command_notifications: bool
    # Send that echo as the chat's single-line ephemeral status line rather than a message.
    notification_ephemeral: bool
    # Also report the notification text to the observer hub.
    notify_observers_of_commands: bool
    # Where show_photo renders: the server's screen (spoken channel — only while `photos.show_on_screen`
    # is on, else the chat) or the chat (Telegram).
    photo_target: str
    # send_screenshot needs a screen to send to.
    screenshot_allowed: bool
    # The channel's consumer streams tokens live (else it takes the round's raw text at RoundEnd).
    streams_tokens: bool
    # Toolkits put in the LLM's window before every request from this channel, so the model
    # never spends a round on `load_toolkit` for them (assistant/commands/ToolkitPreloader.py).
    # Voice names `home`, `music`, `cars`, `weather` and `timers`: "turn on the lamp", "turn off
    # the music", "defrost the car", "will it rain tomorrow" and "turn on the TV for 15 minutes"
    # are what it is spoken to for, and a load_toolkit round there costs a second or two every
    # time — and a reply spoken before the round and another after it (the 2026-10-10 double
    # answer, a timer request without `timers` in the window). One whose integration is off
    # is skipped by itself; the `preload_toolkits.<channel>` config list replaces this default.
    preload_toolkits: tuple[str, ...] = ()
    # Toolkits the channel cannot do without, preloaded ahead of `preload_toolkits` and out of
    # the config list's reach: voice's is `voice`, the spoken dialog style and the false
    # positive rule, which the prompt carries nowhere else.
    fixed_toolkits: tuple[str, ...] = ()
    # The history window the channel's requests render (MessageHistory.get_history_messages).
    # The history is one conversation for every channel; what a preload puts in it is stamped
    # with the window it was loaded for and rendered for that window only, so one spoken
    # request does not leave voice's toolkits (8 KB of them) in every chat prompt after it.
    window: str = WINDOW_TEXT
    # What the channel asks of every reply beyond the prompt's dialog style, folded onto the
    # request with the other trailing directives (DialogManager): iMessage's plain short texts.
    style_directive: str = ""


VOICE_TO_VOICE = ChannelProfile(
    name=SOURCE_VOICE, llm_tier="fast", brevity_directive=True, forward_thinking=False,
    command_notifications=False, notification_ephemeral=False, notify_observers_of_commands=False,
    photo_target=PHOTO_TARGET_SCREEN, screenshot_allowed=False,
    streams_tokens=False, preload_toolkits=("home", "music", "cars", "weather", "timers"), fixed_toolkits=("voice",),
    window=WINDOW_VOICE,
)

TEXT_CHAT = ChannelProfile(
    name=SOURCE_TEXT_CHAT, llm_tier="smart", brevity_directive=False, forward_thinking=True,
    command_notifications=True, notification_ephemeral=False, notify_observers_of_commands=True,
    photo_target=PHOTO_TARGET_CHAT, screenshot_allowed=True,
    streams_tokens=True,
)

TELEGRAM = ChannelProfile(
    name=SOURCE_TELEGRAM, llm_tier="smart", brevity_directive=False, forward_thinking=False,
    command_notifications=True, notification_ephemeral=True, notify_observers_of_commands=False,
    photo_target=PHOTO_TARGET_CHAT, screenshot_allowed=True,
    streams_tokens=True,
)

# The TUI's Home tab chat: a chat transport on the hub like Telegram, so command echoes
# arrive through the Notifier (inline notices in the pane, not ephemeral) and thinking
# through the request's own event stream.
CONSOLE = ChannelProfile(
    name=SOURCE_CONSOLE, llm_tier="smart", brevity_directive=False, forward_thinking=False,
    command_notifications=True, notification_ephemeral=False, notify_observers_of_commands=False,
    photo_target=PHOTO_TARGET_CHAT, screenshot_allowed=True,
    streams_tokens=True,
)

# The assistant's own iMessage account (interfaces/imessage): a chat transport on the hub that
# can neither edit a bubble nor draw a card, so no command echo is sent there, and what the
# reply is written for — a phone's Messages app — is said with every request.
IMESSAGE = ChannelProfile(
    name=SOURCE_IMESSAGE, llm_tier="smart", brevity_directive=False, forward_thinking=False,
    command_notifications=False, notification_ephemeral=False, notify_observers_of_commands=False,
    photo_target=PHOTO_TARGET_CHAT, screenshot_allowed=True,
    streams_tokens=True,
    style_directive=("This conversation is over iMessage: write the way people text — short, plain text, "
                     "no HTML tags, no Markdown, no tables. A photo or a video cannot be sent there yet."),
)

# No dialog at all: the autonomous action manager running commands on its own.
AUTONOMOUS = ChannelProfile(
    name=SOURCE_AUTONOMOUS, llm_tier="smart", brevity_directive=False, forward_thinking=False,
    command_notifications=False, notification_ephemeral=False, notify_observers_of_commands=False,
    photo_target=PHOTO_TARGET_CHAT, screenshot_allowed=True,
    streams_tokens=False,
)

PROFILES = {
    SOURCE_VOICE: VOICE_TO_VOICE,
    SOURCE_TEXT_CHAT: TEXT_CHAT,
    SOURCE_TELEGRAM: TELEGRAM,
    SOURCE_CONSOLE: CONSOLE,
    SOURCE_IMESSAGE: IMESSAGE,
    SOURCE_AUTONOMOUS: AUTONOMOUS,
}

_reported_unknown = set()


def profile_for(source: Optional[str]) -> ChannelProfile:
    """The policy row for a request source. None means no dialog (autonomous); an unknown
    source is treated as a text chat and logged once."""
    if source is None:
        return AUTONOMOUS
    profile = PROFILES.get(source)
    if profile is not None:
        return profile
    if source not in _reported_unknown:
        _reported_unknown.add(source)
        info(f"Unknown request source {source!r} — using the {SOURCE_TEXT_CHAT} channel profile.")
    else:
        trace(f"Unknown request source {source!r} — using the {SOURCE_TEXT_CHAT} channel profile.")
    return TEXT_CHAT
