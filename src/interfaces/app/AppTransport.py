"""The phone app as a chat transport: what the chat layer renders (drafts, command lines,
notices, photos, approval and secret prompts) becomes items of the `Transcript` the apps draw,
and what is typed in an app — with the pictures staged beside it — goes to the hub as an
`IncomingMessage`.

One conversation ("app"), always the owner's — as the console has "tui": whoever reaches the
HTTP address is the owner, and every phone shows the same list. A secret typed into an app's
masked field goes straight to `hub.claim_secret`: it is never an item and never logged.

The item kinds and their fields are a contract with the apps (`docs/plans/APP_CHAT_PLAN.md`).
"""
import io
import time
from typing import Any, Optional

from assistant.ChannelProfile import SOURCE_APP
from common.ConfigManager import config
from common.logging import info, trace
from interfaces.app.Transcript import Transcript, new_id
from interfaces.chat.ChatTransport import ChatTransport
from interfaces.chat.ChatTypes import KIND_PHOTO, PHASE_THINKING, PHASE_WAITING, Attachment, IncomingMessage, Progress
from interfaces.chat.MediaIntake import picture_as_jpeg

CONVERSATION = "app"
MAX_PICTURES = 10       # in one message, as many as the console's composer stages

KIND_USER = "user"
KIND_ASSISTANT = "assistant"
KIND_THINKING = "thinking"
KIND_COMMAND = "command"
KIND_NOTICE = "notice"
KIND_MEDIA = "media"
KIND_APPROVAL = "approval"
KIND_SECRET = "secret"

STATE_RUNNING = "running"
STATE_DONE = "done"
STATE_FAILED = "failed"

# What a restart leaves of an item nobody will finish any more.
INTERRUPTED = "interrupted"
NO_LONGER_PENDING = "No longer pending"


# The working line while a reply is being written or between two rounds: nothing to word,
# but a turn is still on — the apps show their stop button for as long as there is a line.
PHASE_REPLYING = "replying"
REPLYING = {"phase": PHASE_REPLYING, "elapsed": 0.0, "output_tokens": 0, "rate": 0.0, "prefill": 0.0, "commands": 0}


def working_facts(progress: Progress) -> dict:
    """The working line as the apps are sent it: the facts, worded by the app."""
    return {"phase": progress.phase, "elapsed": round(progress.elapsed, 1),
            "output_tokens": progress.output_tokens, "rate": round(progress.rate, 1),
            "prefill": round(progress.prefill, 1), "commands": progress.commands}


class PicturesRefused(Exception):
    """The pictures sent with a message cannot be taken: why, in a few words the app shows."""


class AppDraft:
    """One reply bubble as an `assistant` item, its thinking as an item of its own before it."""

    def __init__(self, transcript: Transcript):
        self._transcript = transcript
        self._id: Optional[str] = None
        self._thinking_id: Optional[str] = None
        self._thinking_shown: Optional[tuple] = None
        self._thinking_text = ""
        self._thinking_started = 0.0
        self._closed = False

    def show_progress(self, progress: Progress) -> None:
        if self._closed:
            return
        self._transcript.set_working(working_facts(progress))
        if progress.thinking_text:
            self._thinking_text = progress.thinking_text
            self._show_thinking(done=progress.phase != PHASE_THINKING)

    def update(self, html: str) -> bool:
        if self._closed or not html.strip():
            return False
        self._show(html, final=False)
        return True

    def finalize(self, html: str) -> bool:
        self._closed = True
        self._show(html, final=True)
        return True

    def discard(self) -> None:
        self._closed = True
        self._transcript.set_working(REPLYING)
        self._show_thinking(done=True)
        if self._id is None:
            trace("App draft discarded before anything was shown")
            return
        self._transcript.remove(self._id)
        self._id = None

    def _show(self, html: str, final: bool) -> None:
        self._transcript.set_working(REPLYING)
        self._show_thinking(done=True)
        if self._id is None:
            self._id = new_id()
        self._transcript.upsert(self._id, KIND_ASSISTANT, html=html, final=final)

    def _show_thinking(self, done: bool) -> None:
        if not self._thinking_text:
            return
        if self._thinking_id is None:
            self._thinking_id = new_id()
            self._thinking_started = time.monotonic()
        if self._thinking_shown is not None and self._thinking_shown[1]:
            return      # folded already: the seconds it took stand
        shown = (self._thinking_text, done)
        if shown == self._thinking_shown:
            return
        self._thinking_shown = shown
        self._transcript.upsert(self._thinking_id, KIND_THINKING, text=self._thinking_text, done=done,
                                seconds=round(time.monotonic() - self._thinking_started, 1))


class AppTransport(ChatTransport):
    name = "App"
    source = SOURCE_APP
    max_message_length = 0
    keeps_partial_reply = True
    lang = "en"

    def __init__(self, transcript: Transcript):
        self.transcript = transcript
        self._hub = None

    def start(self, hub) -> None:
        self._hub = hub
        self._settle()
        info("App chat transport started (/app, one conversation for every phone).")

    def stop(self) -> None:
        self._hub = None
        self.transcript.set_working(None)
        self.transcript.flush(closed=True)

    def _settle(self) -> None:
        """What the last run of the server left unfinished is closed, so no app shows a reply
        still streaming, a command still running or a question still open after a restart."""
        settled = 0
        for item in self.transcript.items():
            fields = {k: v for k, v in item.items() if k not in ("id", "kind", "at", "order")}
            kind = item["kind"]
            if kind == KIND_ASSISTANT and not fields.get("final"):
                fields["final"] = True
            elif kind == KIND_THINKING and not fields.get("done"):
                fields["done"] = True
            elif kind == KIND_COMMAND and fields.get("state") == STATE_RUNNING:
                fields.update(state=STATE_FAILED, detail=INTERRUPTED)
            elif kind in (KIND_APPROVAL, KIND_SECRET) and not fields.get("outcome"):
                fields["outcome"] = NO_LONGER_PENDING
            else:
                continue
            self.transcript.upsert(item["id"], kind, **fields)
            settled += 1
        if settled:
            info(f"App chat: {settled} item(s) the last run left unfinished were closed")
        else:
            trace("App chat: nothing left unfinished by the last run")

    def owner_conversations(self) -> list:
        return [CONVERSATION]

    def attended(self) -> bool:
        """A phone holds an event stream, so the list is on somebody's screen."""
        attended = self.transcript.listeners() > 0
        if not attended:
            trace("App chat: no phone is connected — the app is not attended")
        return attended

    # --- from the apps ------------------------------------------------------------------------

    def submit(self, item_id: str, text: str, user: str = "", pictures: list = ()) -> bool:
        """A message typed in an app, under the id the app gave its bubble, with the pictures
        staged beside it (their bytes): a text message, or a photo message carrying every
        picture with the text as its caption, as the console sends one. False for an empty
        message; `PicturesRefused` for pictures that cannot be taken. An id the list already
        holds is an app sending again what did arrive: taken, and no second turn."""
        text = (text or "").strip()
        pictures = list(pictures)
        if not text and not pictures:
            trace("App chat: empty message ignored")
            return False
        if item_id and self.transcript.item(item_id) is not None:
            info(f"App chat: message {item_id} is in the list already, sent again — no second turn")
            return True
        jpegs = self._as_jpegs(pictures)
        self.transcript.upsert(item_id or new_id(), KIND_USER, text=text,
                               images=[self._keep_picture(jpeg) for jpeg in jpegs])
        if self._hub is None:
            info("App chat: no chat hub attached, cannot send the message")
            self.send_text(CONVERSATION, "Sorry, the chat is not connected to the assistant.")
            return True
        info(f"App chat: {text or '[no text]'}" + (f" (+{len(jpegs)} picture(s))" if jpegs else ""))
        message = IncomingMessage(conversation=CONVERSATION, sender=user or config.get_owner_name(),
                                  message_id=item_id, text=text)
        if jpegs:
            message.kind = KIND_PHOTO
            message.attachments = [Attachment(jpeg, "image/jpeg", "photo") for jpeg in jpegs]
        self._hub.receive(self, message)
        return True

    @staticmethod
    def _as_jpegs(pictures: list) -> list:
        """The pictures as the vision tier reads them, whatever an app sent them as."""
        if len(pictures) > MAX_PICTURES:
            info(f"App chat: {len(pictures)} pictures in one message, more than the {MAX_PICTURES} it may carry — refused")
            raise PicturesRefused(f"more than {MAX_PICTURES} pictures")
        jpegs = [picture_as_jpeg(data) for data in pictures]
        unread = sum(jpeg is None for jpeg in jpegs)
        if unread:
            info(f"App chat: {unread} of {len(pictures)} picture(s) could not be read — the message is refused")
            raise PicturesRefused("a picture could not be read")
        return jpegs

    def interrupt(self) -> None:
        if self._hub is None:
            trace("App chat: stop with no hub attached, nothing to cancel")
            return
        info("App chat: stopping the in-flight reply")
        self._hub.session_for(self, CONVERSATION).cancel()

    def answer_approval(self, token: str, decision: str) -> bool:
        """False when the question is no longer open."""
        if self._hub is None:
            trace("App chat: approval answered with no hub attached, dropped")
            return False
        taken = self._hub.resolve_approval(token, decision, "app")
        if not taken:
            self._close_stale(KIND_APPROVAL, token)
        return taken

    def answer_secret(self, token: str, value: Optional[str]) -> bool:
        """An app's masked field was submitted (*value*) or dismissed (None)."""
        if self._hub is None:
            trace("App chat: secret answered with no hub attached, dropped")
            return False
        if value is None:
            taken = self._hub.cancel_secret(token, "app")
        else:
            taken = self._hub.claim_secret(self, CONVERSATION, value, "app", token=token)
        if not taken:
            self._close_stale(KIND_SECRET, token)
        return taken

    def _close_stale(self, kind: str, token: str) -> None:
        item = self.transcript.item(self._prompt_id(kind, token))
        if item is None or item.get("outcome"):
            trace(f"App chat: no open {kind} prompt [{token[:8]}] to close")
            return
        info(f"App chat: the {kind} prompt [{token[:8]}] is no longer pending, closing it")
        self._close_prompt(kind, item["id"], NO_LONGER_PENDING)

    # --- ChatTransport ----------------------------------------------------------------------

    def send_text(self, conversation, html: str, *, silent: bool = False, preview: bool = True) -> bool:
        if not html.strip():
            trace("App chat: empty text not shown")
            return False
        self.transcript.upsert(new_id(), KIND_NOTICE, html=html)
        return True

    def send_photo(self, conversation, data: bytes, caption: str = "") -> bool:
        self._add_photo(data, caption, album="")
        return True

    def send_photo_album(self, conversation, photos: list[tuple[bytes, str]]) -> bool:
        album = new_id()
        for data, caption in photos:
            self._add_photo(data, caption, album)
        return True

    def send_video(self, conversation, data: bytes, caption: str = "",
                   thumbnail: Optional[bytes] = None) -> bool:
        poster, width, height = "", 0, 0
        if thumbnail is None:
            trace("App chat: video without a poster frame")
        else:
            extension, width, height = self._picture_facts(thumbnail)
            poster = self.transcript.add_media(thumbnail, extension)
        self.transcript.upsert(new_id(), KIND_MEDIA, media="video", url=self.transcript.add_media(data, "mp4"),
                               poster=poster, caption=caption, width=width, height=height, album="")
        return True

    def _add_photo(self, data: bytes, caption: str, album: str) -> None:
        self.transcript.upsert(new_id(), KIND_MEDIA, media="photo", poster="", caption=caption, album=album,
                               **self._keep_picture(data))

    def _keep_picture(self, data: bytes) -> dict:
        """A picture kept beside the log, as an item names it: its address and its size."""
        extension, width, height = self._picture_facts(data)
        return {"url": self.transcript.add_media(data, extension), "width": width, "height": height}

    @staticmethod
    def _picture_facts(data: bytes) -> tuple[str, int, int]:
        """The file extension and the size of a picture, so an app can lay it out before
        the bytes arrive; a JPEG of unknown size when it cannot be read."""
        try:
            from PIL import Image
            with Image.open(io.BytesIO(data)) as picture:
                extension = {"JPEG": "jpg", "PNG": "png", "GIF": "gif", "WEBP": "webp"}.get(picture.format or "")
                if extension:
                    return extension, picture.width, picture.height
                trace(f"App chat: a picture in the format {picture.format}, served as it is")
                return "jpg", picture.width, picture.height
        except Exception as e:
            info(f"App chat: could not read a picture's size ({e}), sent without one")
            return "jpg", 0, 0

    def begin_draft(self, conversation) -> AppDraft:
        return AppDraft(self.transcript)

    def reply_started(self, conversation) -> None:
        self.transcript.set_working(working_facts(Progress(phase=PHASE_WAITING)))

    def reply_finished(self, conversation) -> None:
        self.transcript.set_working(None)

    def send_command_line(self, conversation, text: str) -> Any:
        return self.transcript.upsert(new_id(), KIND_COMMAND, text=text, state=STATE_RUNNING, detail="")

    def finish_command_line(self, conversation, ref, text: str, detail: str) -> None:
        # The contract carries no verdict, only CommandEcho.describe_completion's words.
        state = STATE_FAILED if detail.startswith("failed") else STATE_DONE
        self.transcript.upsert(ref, KIND_COMMAND, text=text, state=state, detail=detail)

    def set_status_line(self, conversation, html: Optional[str]) -> None:
        if html:
            trace("App chat: a status line is not shown, the app's command echoes are lines of their own")

    def send_approval_prompt(self, conversation, token: str, title: str, details: str) -> Any:
        info(f"App chat: showing approval prompt [{token[:8]}]: {title}")
        return self.transcript.upsert(self._prompt_id(KIND_APPROVAL, token), KIND_APPROVAL,
                                      token=token, title=title, details=details, outcome="")

    def close_approval_prompt(self, conversation, prompt_ref, outcome_text: str) -> None:
        self._close_prompt(KIND_APPROVAL, prompt_ref, outcome_text)

    def send_secret_prompt(self, conversation, token: str, title: str, details: str) -> Any:
        info(f"App chat: showing secret prompt [{token[:8]}]: {title}")
        return self.transcript.upsert(self._prompt_id(KIND_SECRET, token), KIND_SECRET,
                                      token=token, title=title, details=details, outcome="")

    def close_secret_prompt(self, conversation, prompt_ref, outcome_text: str) -> None:
        self._close_prompt(KIND_SECRET, prompt_ref, outcome_text)

    @staticmethod
    def _prompt_id(kind: str, token: str) -> str:
        """One item per question: asked again under the same token, the prompt is updated in place."""
        return f"{kind}-{token}"

    def _close_prompt(self, kind: str, item_id: str, outcome_text: str) -> None:
        item = self.transcript.item(item_id)
        if item is None:
            trace(f"App chat: the {kind} prompt {item_id} is gone from the log, nothing to close")
            return
        self.transcript.upsert(item_id, kind, token=item["token"], title=item["title"], details=item["details"],
                               outcome=outcome_text or "Closed")
