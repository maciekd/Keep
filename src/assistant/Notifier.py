"""
Notifier — the assistant's outbound side channel to the user.

The core (commands, agents, integrations) needs to push things at the user outside a
dialog turn: 2FA prompts, autonomous findings, photos, bug reports, and the approve/deny
gate for sensitive commands. Which chat app renders those is an interface concern, so the
core talks to whatever `set_notifier()` installed and never imports `interfaces.*`.

Default is `NullNotifier`: everything is logged and dropped, and `request_approval`
answers "unavailable" — a missing notifier must never let a sensitive command through.
"""
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from common.logging import info


class SecretRejected(Exception):
    """Raised by a secret request's `accept` callback when the value cannot be taken as
    typed (wrong shape, too short). The message is shown to the user, who is asked again."""


@runtime_checkable
class Notifier(Protocol):
    """Protocol for the side channel that pushes messages, photos and approvals to the user."""
    def send_message(self, text: str, disable_notification: bool = False,
                     disable_web_page_preview: bool = False, in_history: bool = False) -> bool:
        """True when the message was handed to a channel, False when it was dropped.
        *in_history* says the text is already a row in the dialog history, so a chat that
        renders the history (the TUI's Home tab) is skipped — it shows the row itself."""
        ...

    def send_command_notification(self, text: str, ephemeral: bool = False,
                                  source: Optional[str] = None, conversation: Any = None) -> Any:
        """Echo a command as it starts running. *source* and *conversation* (the
        RequestContext's) name the chat the request came from so the echo lands there and
        not in every chat; without them it goes everywhere the owner is reachable. Returns
        a handle for `complete_command_notification`, or None when nothing was shown."""
        ...

    def complete_command_notification(self, handle: Any, detail: str, result: Any = None) -> None:
        """The command behind `handle` finished: annotate its echo with *detail*
        ("done after 4s, received 345 chars"). *result* is what it handed back for the
        model — a text, a block list of texts and images, None for nothing — for a chat
        that shows it behind the echo. A None handle is ignored."""
        ...

    def send_photo(self, photo_bytes: bytes, caption: str = "",
                   source: Optional[str] = None, conversation: Any = None) -> bool:
        """*source* and *conversation* (the RequestContext's) name the chat of the request
        the photo answers, so it lands there — and in every owner chat when no chat
        transport carries that source (a voice request). Without them the photo is a push
        with no request behind it and goes wherever the transports route those."""
        ...

    def send_photo_album(self, photos: list[tuple[bytes, str]],
                         source: Optional[str] = None, conversation: Any = None) -> bool: ...

    def send_video(self, video_bytes: bytes, caption: str = "", thumbnail: Optional[bytes] = None,
                   source: Optional[str] = None, conversation: Any = None) -> bool:
        """*thumbnail* is an optional small JPEG poster frame shown before the clip is loaded."""
        ...

    def request_approval(self, title: str, details: str = "", timeout: Optional[float] = None,
                         source: Optional[str] = None, conversation: Any = None,
                         cancelled: Optional[Callable[[], bool]] = None) -> str:
        """Block until the user answers. Returns "approve", "deny", "timeout", "cancelled" or
        "unavailable". *source* and *conversation* (the RequestContext's) name the chat the
        request came from so the question is asked only there — the person who asked for the
        command is the one who answers for it; without them it is asked everywhere the owner
        is reachable (a voice request has no chat of its own). `cancelled()` is polled while
        waiting: true means the request behind the question is gone, the prompt is withdrawn
        and "cancelled" comes back."""
        ...

    def request_secret(self, title: str, details: str, accept: Callable[[str], str],
                       timeout: Optional[float] = None,
                       source: Optional[str] = None, conversation: Any = None) -> str:
        """Ask the user for one value through a masked prompt in the chat UIs; the value
        never enters a dialog. `accept(value)` runs on the server with what they typed and
        returns the confirmation to show them; it raises `SecretRejected(reason)` to have
        the reason shown and the user asked again (a few times). Blocks until the value
        was accepted, cancelled, every attempt was rejected, or the timeout. Returns
        "accepted", "cancel", "rejected", "timeout" or "unavailable".
        *source* and *conversation* (the RequestContext's) name the chat the request came
        from so the prompt opens only there; without them it opens everywhere the owner is
        reachable."""
        ...

    def markdown_to_html(self, md: str) -> str:
        """Render markdown into the notifier's rich-text dialect (Telegram HTML, plain text, …)."""
        ...

    def open_conversation(self, query: str) -> bool:
        """A turn the server starts (the first-run brief, the Full Disk Access grant seen):
        *query* goes to the assistant as a system turn in one owner conversation — the one
        somebody is sitting at when the chat layer can tell — and the reply is rendered
        there. Never blocks. False when no owner chat is reachable."""
        ...

    def owner_channels(self) -> list[tuple[str, bool]]:
        """What a push to the owner reaches now: each chat by its name ("Telegram", "Console")
        and whether somebody can see it at the moment — a console nobody has open is on the
        list and not attended."""
        ...

    def receive_chat_message(self, source: str, conversation: Any, sender: str, text: str, *,
                             guest: bool = False, sender_is_bot: bool = False,
                             chat_name: Optional[str] = None) -> bool:
        """A message a feed saw in a chat the chat layer serves (the owner's own Telegram
        account reading a group the bot is in): start a turn in *conversation* as if the
        transport with *source* had received it, so the reply goes out through that transport.
        *guest*: the sender is not one of the household's owners; *sender_is_bot*: the sender
        is another assistant. Never blocks. False when no transport carries *source*."""
        ...


class NullNotifier:
    """Installed when no chat interface is running: logs and drops everything; the send
    methods return False so commands can tell the LLM nothing reached the user."""

    def send_message(self, text: str, disable_notification: bool = False,
                     disable_web_page_preview: bool = False, in_history: bool = False) -> bool:
        info(f"no notifier configured — message dropped: {text[:120]!r}")
        return False

    def send_command_notification(self, text: str, ephemeral: bool = False,
                                  source: Optional[str] = None, conversation: Any = None) -> Any:
        info(f"no notifier configured — command notification dropped: {text[:120]!r}")
        return None

    def complete_command_notification(self, handle: Any, detail: str, result: Any = None) -> None:
        info(f"no notifier configured — command completion dropped: {detail!r}")

    def send_photo(self, photo_bytes: bytes, caption: str = "",
                   source: Optional[str] = None, conversation: Any = None) -> bool:
        info(f"no notifier configured — photo dropped ({len(photo_bytes)} bytes, caption {caption[:80]!r})")
        return False

    def send_photo_album(self, photos: list[tuple[bytes, str]],
                        source: Optional[str] = None, conversation: Any = None) -> bool:
        info(f"no notifier configured — photo album of {len(photos)} dropped")
        return False

    def send_video(self, video_bytes: bytes, caption: str = "", thumbnail: Optional[bytes] = None,
                   source: Optional[str] = None, conversation: Any = None) -> bool:
        info(f"no notifier configured — video dropped ({len(video_bytes)} bytes, caption {caption[:80]!r})")
        return False

    def request_approval(self, title: str, details: str = "", timeout: Optional[float] = None,
                         source: Optional[str] = None, conversation: Any = None,
                         cancelled: Optional[Callable[[], bool]] = None) -> str:
        info(f"no notifier configured — approval request for {title!r} dropped, answering 'unavailable'")
        return "unavailable"

    def request_secret(self, title: str, details: str, accept: Callable[[str], str],
                       timeout: Optional[float] = None,
                       source: Optional[str] = None, conversation: Any = None) -> str:
        info(f"no notifier configured — secret request for {title!r} dropped, answering 'unavailable'")
        return "unavailable"

    def receive_chat_message(self, source: str, conversation: Any, sender: str, text: str, *,
                             guest: bool = False, sender_is_bot: bool = False,
                             chat_name: Optional[str] = None) -> bool:
        info(f"no notifier configured — chat message from {sender} in {source} [{conversation}] dropped")
        return False

    def markdown_to_html(self, md: str) -> str:
        return md

    def open_conversation(self, query: str) -> bool:
        info(f"no notifier configured — conversation not opened: {query[:120]!r}")
        return False

    def owner_channels(self) -> list[tuple[str, bool]]:
        return []


_notifier: Notifier = NullNotifier()


def notifier() -> Notifier:
    return _notifier


def set_notifier(n: Optional[Notifier]) -> None:
    """Install the process-wide notifier; None restores the NullNotifier."""
    global _notifier
    _notifier = n if n is not None else NullNotifier()
    info(f"Notifier set to {type(_notifier).__name__}")
