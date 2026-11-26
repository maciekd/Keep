"""The TUI as a chat transport: what the user types into the Home tab's chat pane goes to the
hub as an `IncomingMessage`, and everything the chat layer renders (drafts, notices, the
status line, approval and secret prompts) lands in the pane. One conversation ("tui"), always
the owner's — so Notifier broadcasts and prompts reach the console too. A secret typed into
the pane's masked field goes straight to `hub.claim_secret`: it is never a user entry, never
a history turn and never logged.

In-process, so no throttling or rate limits: every draft update is shown at once and the
pane's typewriter reveal smooths the stream. The pane also carries the dialog history
(`HistoryFeed`); the transport tells the feed when a console turn starts and when its
reply is over so the feed can recognise the history rows that turn produces.
"""
from typing import TYPE_CHECKING, Any, Optional

from assistant.ChannelProfile import SOURCE_CONSOLE
from common.ConfigManager import config
from common.logging import info, trace
from interfaces.chat import RichText
from interfaces.chat.ChatSession import ChatSession
from interfaces.chat.ChatTransport import ChatTransport
from interfaces.chat.ChatTypes import KIND_PHOTO, PHASE_THINKING, Attachment, IncomingMessage, Progress
from interfaces.console.widgets.ChatPane import ChatPane, Entry

if TYPE_CHECKING:
    from interfaces.console.HistoryFeed import HistoryFeed

CONVERSATION = "tui"


class ConsoleDraft:
    """One reply bubble in the pane. The working line shows until the first visible text;
    thinking chunks build a dim block of their own above the reply."""

    def __init__(self, pane: ChatPane):
        self._pane = pane
        self._entry: Optional[Entry] = None
        self._thinking: Optional[Entry] = None
        self._closed = False

    def show_progress(self, progress: Progress) -> None:
        if self._closed:
            return
        self._pane.set_working(progress)
        if progress.thinking_text:
            self._thinking = self._pane.set_thinking(self._thinking, progress.thinking_text,
                                                     done=progress.phase != PHASE_THINKING)

    def update(self, html: str) -> bool:
        if self._closed or not html.strip():
            return False
        self._pane.set_working(None)
        self._finish_thinking()
        if self._entry is None:
            self._entry = self._pane.add_assistant(html)
        else:
            self._pane.update_entry(self._entry, html)
        return True

    def finalize(self, html: str) -> bool:
        self._closed = True
        self._pane.set_working(None)
        self._finish_thinking()
        if self._entry is None:
            self._entry = self._pane.add_assistant(html, final=True)
        else:
            self._pane.update_entry(self._entry, html, final=True)
        return True

    def discard(self) -> None:
        self._closed = True
        self._pane.set_working(None)
        self._finish_thinking()
        if self._entry is not None:
            self._pane.remove_entry(self._entry)
            self._entry = None
        else:
            trace("Console draft discarded before anything was shown")

    def _finish_thinking(self):
        if self._thinking is not None and not self._thinking.final:
            self._thinking.final = True


class ConsoleTransport(ChatTransport):
    name = "Console"
    source = SOURCE_CONSOLE
    max_message_length = 0
    shows_history = True    # the HistoryFeed draws every system row into the same pane
    lang = "en"

    def __init__(self, pane: ChatPane, feed: Optional["HistoryFeed"] = None):
        self.pane = pane
        self.feed = feed
        self._hub = None
        self.has_clients = lambda: True   # ConsoleInterface points it at the TUI server's client set
        pane.bind(on_submit=self.submit, on_interrupt=self.interrupt, on_approval=self.answer_approval,
                  on_secret=self.answer_secret)

    def start(self, hub) -> None:
        self._hub = hub
        info("Console chat transport started (Home tab chat pane).")

    def stop(self) -> None:
        self._hub = None

    def owner_conversations(self) -> list:
        return [CONVERSATION]

    def attended(self) -> bool:
        """A TUI client is connected. With the terminal gone (the Full Disk Access grant quits
        it) a turn the server starts must go to Telegram, not to a pane nobody sees."""
        attended = bool(self.has_clients())
        if not attended:
            trace("Console chat: no TUI client connected — the console is not attended")
        return attended

    # --- from the pane ----------------------------------------------------------------------

    def submit(self, text: str, images: list = ()):
        """The composer's text and the pictures staged in it (`ChatPane.StagedImage`): a text
        message, or a photo message carrying every picture with the text as its caption —
        the session batches them into one request, as it does a Telegram album."""
        text = (text or "").strip()
        images = list(images)
        if not text and not images:
            trace("Console chat: empty submission ignored")
            return
        self.pane.add_user(text, [image.picture for image in images])
        if self._hub is None:
            info("Console chat: no chat hub attached, cannot send the message")
            self.pane.add_notice("Sorry, the chat is not connected to the assistant.")
            return
        info(f"Console chat: {text}" + (f" (+{len(images)} image(s))" if images else ""))
        if self.feed is not None:
            self.feed.user_submitted(text or ChatSession.IMAGES_ONLY_QUERY)
        if images:
            self._hub.receive(self, IncomingMessage(
                conversation=CONVERSATION, sender=config.get_owner_name(), kind=KIND_PHOTO, text=text,
                attachments=[Attachment(image.data, image.mime_type, "photo") for image in images]))
            return
        self._hub.receive(self, IncomingMessage(conversation=CONVERSATION, sender=config.get_owner_name(),
                                                text=text))

    def interrupt(self):
        if self._hub is None:
            trace("Console chat: interrupt with no hub attached, nothing to cancel")
            return
        info("Console chat: interrupting the in-flight reply")
        self._hub.session_for(self, CONVERSATION).cancel()

    def answer_approval(self, token: str, decision: str):
        if self._hub is None:
            trace("Console chat: approval answered with no hub attached, dropped")
            return
        if not self._hub.resolve_approval(token, decision, "console"):
            pending = self.pane.pending_approval
            if pending is not None and pending.token == token:
                self.pane.close_approval(pending, "no longer pending")

    def answer_secret(self, token: str, value: Optional[str]):
        """The pane's masked field was submitted (`value`) or dismissed (None)."""
        if self._hub is None:
            trace("Console chat: secret answered with no hub attached, dropped")
            return
        if value is None:
            taken = self._hub.cancel_secret(token, "console")
        else:
            taken = self._hub.claim_secret(self, CONVERSATION, value, "console")
        if not taken:
            pending = self.pane.pending_secret
            if pending is not None and pending.token == token:
                self.pane.close_secret(pending, "no longer pending")

    # --- ChatTransport ----------------------------------------------------------------------

    def send_text(self, conversation, html: str, *, silent: bool = False, preview: bool = True) -> bool:
        if not html.strip():
            trace("Console chat: empty text not shown")
            return False
        self.pane.add_notice(html)
        return True

    def send_photo(self, conversation, data: bytes, caption: str = "") -> bool:
        self.pane.add_media("photo", caption, len(data), data)
        return True

    def send_photo_album(self, conversation, photos: list[tuple[bytes, str]]) -> bool:
        for data, caption in photos:
            self.pane.add_media("photo", caption, len(data), data)
        return True

    def send_video(self, conversation, data: bytes, caption: str = "",
                   thumbnail: Optional[bytes] = None) -> bool:
        if thumbnail is None:
            trace("Console chat: video without a poster frame, shown as its caption alone")
        self.pane.add_media("video", caption, len(data), thumbnail)
        return True

    def begin_draft(self, conversation) -> ConsoleDraft:
        return ConsoleDraft(self.pane)

    def reply_started(self, conversation) -> None:
        if self.feed is not None:
            self.feed.turn_started()

    def reply_finished(self, conversation) -> None:
        if self.feed is not None:
            self.feed.turn_finished()

    def send_command_line(self, conversation, text: str) -> Any:
        return self.pane.add_notice(RichText.command_line(text))

    def finish_command_line(self, conversation, ref, text: str, detail: str, result=None) -> None:
        self.pane.set_detail(ref, detail)

    def set_status_line(self, conversation, html: Optional[str]) -> None:
        self.pane.set_status(html)

    def send_approval_prompt(self, conversation, token: str, title: str, details: str) -> Any:
        info(f"Console chat: showing approval prompt [{token[:8]}]: {title}")
        return self.pane.add_approval(token, title, details)

    def close_approval_prompt(self, conversation, prompt_ref, outcome_text: str) -> None:
        self.pane.close_approval(prompt_ref, outcome_text)

    def send_secret_prompt(self, conversation, token: str, title: str, details: str) -> Any:
        info(f"Console chat: showing secret prompt [{token[:8]}]: {title}")
        return self.pane.add_secret_prompt(token, title, details)

    def close_secret_prompt(self, conversation, prompt_ref, outcome_text: str) -> None:
        self.pane.close_secret(prompt_ref, outcome_text)
