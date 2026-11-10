"""The contract a chat app implements to become a Floxy chat interface.

A transport is deliberately thin: it turns platform updates into `IncomingMessage`s
(`hub.receive()`), and renders what the shared `ChatSession`/`ChatHub` ask it to. All text
it receives is the HTML dialect defined in `RichText` — render it however the platform can.
A photo it has to download first is announced with `hub.expect_photo()` before the download
(and released with `hub.photo_dropped()` if that fails), so an album is never dispatched
while one of its members is still on its way.

Secrets: while a secret prompt is open in a conversation, the next text the user sends
there is the value. A transport therefore calls `hub.claim_secret(self, conversation, text,
who)` before it does anything else with an incoming text — before logging it, caching it for
reply quotes, echoing it, or `hub.receive()` — and when that returns True treats the text as
consumed: not logged, not shown, not forwarded, and removed from the platform if it can be.

Threading: every method is synchronous and may be called from any thread (session worker
threads, the command worker, integration threads). A transport that owns an event loop
marshals onto it itself. `hub.receive()` never blocks, so calling it from a loop is safe.
"""
from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol

from interfaces.chat import RichText
from interfaces.chat.ChatTypes import Progress


class Draft(Protocol):
    """One reply bubble being written. `show_progress` and `update` are best-effort — a
    platform may throttle them, drop them during a rate-limit ban, or (no message editing)
    ignore them entirely; `finalize` is the one call that must deliver."""

    def show_progress(self, progress: Progress) -> None:
        """Nothing user-visible yet: show a spinner / "Thinking…" / typing state."""

    def update(self, html: str) -> bool:
        """Show the reply so far. True when it reached the screen."""

    def finalize(self, html: str) -> bool:
        """Deliver the bubble's final content (waiting out bans if it must). True on delivery."""

    def discard(self) -> None:
        """Remove whatever this draft put on screen (cancelled, or the reply was empty)."""


class ChatTransport(ABC):
    """One chat platform. Subclasses set `name`/`source`, implement the sends, and call
    `hub.receive(self, IncomingMessage)` for every user action."""

    name: str = ""            # log prefix, monitor name ("Telegram", "TUI", …)
    source: str = ""          # ChannelProfile key the requests carry (SOURCE_TELEGRAM, …)
    max_message_length: int = 0   # longest single message; 0 = unlimited
    shows_history: bool = False   # renders the dialog history itself, so a system row is not pushed to it
    # A reply cut short because a request elsewhere superseded it: True delivers what was written
    # so far (a bubble that is edited in place shows it already), False drops it (a platform that
    # cannot take a bubble back would send half a sentence only because of the kill).
    keeps_partial_reply: bool = True
    lang: str = "en"

    def start(self, hub) -> None:
        """Bring the platform up; keep `hub` to deliver incoming messages to."""

    def stop(self) -> None:
        pass

    @abstractmethod
    def owner_conversations(self) -> list:
        """Where the owner is reachable: approval prompts, the command status line and the
        secret prompts of a request from outside the chat layer go to every one of these."""

    def notification_conversations(self) -> list:
        """Where the Notifier's proactive pushes land — autonomous findings, system notes,
        photos, bug reports. Every owner conversation unless the platform is configured to
        route them to a subset (Telegram's `notify_chat_ids`)."""
        return self.owner_conversations()

    def attended(self) -> bool:
        """Whether somebody can see this chat right now. A phone app always can; the console
        only while a TUI client is connected. A turn the server starts goes to an attended
        chat first."""
        return True

    # --- one-shot sends: True when the platform accepted the message for delivery -----------

    @abstractmethod
    def send_text(self, conversation, html: str, *, silent: bool = False, preview: bool = True) -> bool: ...

    @abstractmethod
    def send_photo(self, conversation, data: bytes, caption: str = "") -> bool: ...

    @abstractmethod
    def send_photo_album(self, conversation, photos: list[tuple[bytes, str]]) -> bool: ...

    @abstractmethod
    def send_video(self, conversation, data: bytes, caption: str = "",
                   thumbnail: Optional[bytes] = None) -> bool: ...

    # --- the streamed reply ---------------------------------------------------------------

    @abstractmethod
    def begin_draft(self, conversation) -> Draft: ...

    def reply_started(self, conversation) -> None:
        """A reply is about to be produced in *conversation*; `reply_finished` follows once
        it is over, whatever happens. Optional; the TUI counts the replies in flight to
        reconcile its live bubbles with the dialog history."""

    def reply_finished(self, conversation) -> None:
        """The reply to the last incoming message is over — delivered, cancelled or failed —
        and no more drafts will be opened for it. Optional; the TUI uses it to reconcile
        its live bubbles with the dialog history."""

    def turned_away(self, conversation, message_ids: list) -> None:
        """A guest's turn from *conversation* was not served because a request was in flight
        — a guest never supersedes one — and was dropped without a word. *message_ids* are
        the `IncomingMessage.message_id`s the turn was made of. Called from the session's
        threads, and for a turn refused on the spot before `hub.receive` returns False.
        Optional: by default the message is gone, as in a Telegram group; a transport that
        can read the message again offers it later."""

    # --- the "• doing X" command echo ------------------------------------------------------

    def send_command_line(self, conversation, text: str) -> Any:
        """Echo a command that just started running. Returns a reference for
        `finish_command_line`, or None when the line cannot be annotated later. The
        default posts it as a silent message and forgets it."""
        self.send_text(conversation, RichText.command_line(text), silent=True, preview=False)
        return None

    def finish_command_line(self, conversation, ref, text: str, detail: str) -> None:
        """The command echoed as `ref` finished: show *detail* ("done after 4s, received
        345 chars") next to it. Optional."""

    # --- the single-line ephemeral "• doing X" status; None clears it -----------------------

    @abstractmethod
    def set_status_line(self, conversation, html: Optional[str]) -> None: ...

    # --- approvals ------------------------------------------------------------------------

    @abstractmethod
    def send_approval_prompt(self, conversation, token: str, title: str, details: str) -> Any:
        """Show an approve/deny question. Returns a reference for `close_approval_prompt`, or
        None when nothing was shown. The platform later calls
        `hub.resolve_approval(token, decision, who)` with the user's answer."""

    @abstractmethod
    def close_approval_prompt(self, conversation, prompt_ref, outcome_text: str) -> None:
        """The question is answered (or timed out): take the buttons away, show the outcome."""

    def asked_elsewhere(self, conversation, what: str, where: list[str]) -> None:
        """A turn from *conversation* needs an answer only the owner gives — *what* is
        "approval" or "secret" — and the question went to the owner's chats instead (*where*:
        the names of the transports asked), because this conversation is not one of them. A
        platform that shows no prompts at all says here that the turn is waiting. Optional:
        by default the chat is told nothing, as a Telegram group is not."""

    # --- secrets --------------------------------------------------------------------------

    @abstractmethod
    def send_secret_prompt(self, conversation, token: str, title: str, details: str) -> Any:
        """Ask for one value with a masked input. Returns a reference for
        `close_secret_prompt`, or None when nothing was shown. The answer comes back
        through `hub.claim_secret(...)` (see the module docstring); a cancel through
        `hub.cancel_secret(token, who)`. Called again with the same token when the value
        was refused: `details` then carries the reason, and a prompt still open for that
        token is updated in place rather than shown a second time."""

    @abstractmethod
    def close_secret_prompt(self, conversation, prompt_ref, outcome_text: str) -> None:
        """The prompt is over (accepted, rejected, cancelled, timed out, or about to be
        re-asked): stop taking input for it and show the outcome."""
