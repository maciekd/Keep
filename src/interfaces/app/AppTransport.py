"""The phone app as a chat transport: what the chat layer renders (drafts, command lines,
notices, photos, approval and secret prompts) becomes items of the `Transcript` the apps draw,
and what is typed in an app — with the pictures staged beside it — goes to the hub as an
`IncomingMessage`.

One conversation ("app"), always the owner's — as the console has "tui": whoever reaches the
HTTP address is the owner, and every phone shows the same list. A secret typed into an app's
masked field goes straight to `hub.claim_secret`: it is never an item and never logged.

What a command handed back for the model is kept beside the log when the command ends and
never sent with the chat: the item names the file, and an app asks for it when its row is
tapped (`docs/plans/APP_COMMAND_RESULT_PLAN.md`).

Some items are also a notification on the phones registered in `Devices`, sent by `Push`
(`docs/plans/APP_PUSH_PLAN.md`): a message Keep sends on its own and a question only the
owner answers, on every phone, while the app is the chat that alerts
(`ChatHub.alert_channel`); the end of a reply, on the phone that asked. Whether somebody is
looking is not tracked: a phone with the app on screen shows no notification.

The item kinds and their fields are a contract with the apps (`docs/plans/APP_CHAT_PLAN.md`).
"""
import base64
import io
import threading
import time
from html import escape
from typing import Any, Callable, Optional

from assistant.ChannelProfile import SOURCE_APP
from assistant.commands.DebugCommands import DebugCommands
from assistant.Destinations import plain_text
from common.ConfigManager import config
from common.logging import error, info, trace
from common.vault.Redactor import redact_for_storage
from interfaces.app.Devices import Devices
from interfaces.app.Push import Sender
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

# The most of a command's result an app is shown; what is cut says how long the whole was.
RESULT_CHARS = 100_000
RESULT_CUT = "\n\n[… cut here: the whole result is {chars:,} characters]"

# What stands over a question's notification; its text is the question's title, never more.
APPROVAL_NEEDED = "Approval needed"
SECRET_NEEDED = "Keep needs something from you"
# Said once in the app's chat when a phone's registration moves the alerts away from another chat.
ALERTS_MOVED = ("Messages Keep sends on its own now arrive as notifications of this app, and no longer in {chat}. "
                "To keep them in {chat}, ask Keep to set <code>notify_channel</code> to <code>{channel}</code>.")

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

    def __init__(self, transcript: Transcript, on_final: Optional[Callable[[str], None]] = None):
        self._transcript = transcript
        self._on_final = on_final
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
        if self._on_final is not None:
            self._on_final(html)
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
    channel = "app"

    def __init__(self, transcript: Transcript, devices: Optional[Devices] = None, sender: Optional[Sender] = None):
        self.transcript = transcript
        self.devices = devices if devices is not None else Devices()
        self._sender = sender if sender is not None else Sender(self.devices)
        self._hub = None
        # Whose reply is being written, for the notification at its end; guarded by _turns.
        self._turns = threading.Lock()
        self._replying = 0                      # replies in flight: a superseded one winds down beside its successor
        self._asked_by: Optional[str] = None    # the phone of the message no reply has started for yet
        self._answers: Optional[str] = None     # the phone the newest reply answers; None: the server started the turn
        self._turn = 0                          # counts the replies started: a bubble belongs to the one it was begun in
        self._final: Optional[str] = None       # the last bubble the newest reply finished

    def start(self, hub) -> None:
        self._hub = hub
        self._settle()
        info("App chat transport started (/app, one conversation for every phone).")

    def stop(self) -> None:
        self._hub = None
        self._sender.stop()
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

    # --- notifications ------------------------------------------------------------------------

    def alerts(self) -> bool:
        """A phone takes notifications: "the user has the app"."""
        return bool(self.devices.all())

    def alert(self, conversation, text: str) -> None:
        self._push(KIND_NOTICE, plain_text(text))

    def register_device(self, device_id: str, token: Optional[str], environment: str, key: str, name: str) -> None:
        """A phone says where its notifications go — *token* None: nowhere any more — and
        that it has just been looked at. `DeviceRefused` for fields the wire does not know."""
        if token is None:
            self.devices.forget(device_id)
            return
        before = self._alert_channel()
        self.devices.register(device_id, token, environment, key, name)
        if before is not None and before is not self and self._alert_channel() is self:
            # The other chat going quiet must not be a riddle.
            info(f"App chat: with this phone the app is the chat that alerts, no longer {before.name} — said in the chat")
            self.send_text(CONVERSATION, ALERTS_MOVED.format(chat=escape(before.name), channel=escape(before.channel)))

    def _alert_channel(self):
        return self._hub.alert_channel() if self._hub is not None else None

    def _push(self, kind: str, body: str, title: str = "", only: Optional[str] = None) -> None:
        """A notification to every registered phone, or to the phone *only* names."""
        devices = self.devices.all()
        if only is not None:
            devices = [device for device in devices if device.id == only]
        if not devices or not body.strip():
            trace(f"App chat: no notification for a {kind} — "
                  f"{'it has no text' if devices else 'no phone to send it to'}")
            return
        info(f"App chat: a {kind} goes to {len(devices)} phone(s) as a notification")
        for device in devices:
            self._sender.push(device, kind, body, title)

    def _push_question(self, kind: str, heading: str, title: str) -> None:
        """A question only the owner answers rings the phones while the app is the chat that
        alerts, and during a turn of the app's own: then it is asked nowhere else."""
        with self._turns:
            own_turn = self._replying > 0
        if own_turn or self._alert_channel() is self:
            self._push(kind, plain_text(title), heading)
        else:
            trace(f"App chat: the {kind} is not pushed — another chat alerts, and no turn of the app's is running")

    def _reply_final(self, turn: int, html: str) -> None:
        with self._turns:
            if turn == self._turn:
                self._final = html
            else:
                trace("App chat: a bubble of a superseded reply, not what the reply's end announces")

    # --- from the apps ------------------------------------------------------------------------

    def submit(self, item_id: str, text: str, user: str = "", pictures: list = (), device: str = "") -> bool:
        """A message typed in an app, under the id the app gave its bubble, with the pictures
        staged beside it (their bytes): a text message, or a photo message carrying every
        picture with the text as its caption, as the console sends one. *device* is the phone
        it was typed on, where the end of its reply is a notification. False for an empty
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
        # A debug command is answered without a reply: it must not pass for the next turn's asker.
        if jpegs or not DebugCommands.is_command(text):
            with self._turns:
                self._asked_by = device or ""
        if self._hub.receive(self, message) is False:
            with self._turns:
                self._asked_by = None       # dropped on the spot: no reply will start for it
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
        with self._turns:
            turn = self._turn
        return AppDraft(self.transcript, on_final=lambda html: self._reply_final(turn, html))

    def reply_started(self, conversation) -> None:
        with self._turns:
            self._replying += 1
            self._turn += 1
            self._final = None
            self._answers, self._asked_by = self._asked_by, None
        self.transcript.set_working(working_facts(Progress(phase=PHASE_WAITING)))

    def reply_finished(self, conversation) -> None:
        self.transcript.set_working(None)
        with self._turns:
            self._replying = max(0, self._replying - 1)
            if self._replying:
                trace("App chat: a superseded reply is over, its successor's end is the one announced")
                return
            final, self._final = self._final, None
            answers = self._answers
        if final is None:
            trace("App chat: the reply ended without a bubble, nothing to announce")
        elif answers == "":
            trace("App chat: the reply's end is not pushed — its message named no phone")
        else:
            # The asking phone alone, so a reply read on one device does not ring the others;
            # every phone for a turn the server started.
            self._push(KIND_ASSISTANT, plain_text(final), only=answers)

    def send_command_line(self, conversation, text: str) -> Any:
        return self.transcript.upsert(new_id(), KIND_COMMAND, text=text, state=STATE_RUNNING, detail="")

    def finish_command_line(self, conversation, ref, text: str, detail: str, result=None) -> None:
        # The contract carries no verdict, only CommandEcho.describe_completion's words.
        state = STATE_FAILED if detail.startswith("failed") else STATE_DONE
        self.transcript.upsert(ref, KIND_COMMAND, text=text, state=state, detail=detail,
                               **self._keep_result(text, result))

    def _keep_result(self, command: str, result) -> dict:
        """What a command handed back for the model, as its item names it: `result`, the
        address of the text kept beside the log ("" for none), and `images`, its pictures.
        Nothing of it is in the item, so a long result costs the chat nothing until an app
        asks for it. Keeping it never costs the row its ending."""
        try:
            text, pictures = self._result_parts(result)
            images = [self._keep_picture(picture) for picture in pictures]
            address = self.transcript.add_media(self._result_file(text), "txt") if text else ""
        except Exception as e:
            error(f"App chat: the result of '{command}' could not be kept ({e!r}), its row ends without one")
            return {"result": "", "images": []}
        if address or images:
            info(f"App chat: kept the result of '{command}' ({len(text)} chars, {len(images)} picture(s))")
        else:
            trace(f"App chat: '{command}' handed nothing back, its row has no result")
        return {"result": address, "images": images}

    @staticmethod
    def _result_parts(result) -> tuple[str, list[bytes]]:
        """The text and the pictures of a result: a text as it is, a block list's texts with
        a blank line between them and its images as their bytes."""
        if isinstance(result, str):
            return result.strip(), []
        if not isinstance(result, list):
            if result is not None:
                trace(f"App chat: a result that is neither text nor blocks ({type(result).__name__}), not kept")
            return "", []
        texts, pictures = [], []
        for block in result:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "image":
                data = base64.b64decode(block.get("source", block).get("data", ""))
                if data:
                    pictures.append(data)
                else:
                    trace("App chat: an image of a result without bytes, left out")
            elif block.get("text", "").strip():
                texts.append(block["text"].strip())
        return "\n\n".join(texts), pictures

    @staticmethod
    def _result_file(text: str) -> bytes:
        """A result's text as its file holds it: every vault value as its placeholder, and
        cut where it is longer than an app is shown."""
        text = redact_for_storage(text)
        if len(text) > RESULT_CHARS:
            info(f"App chat: a result of {len(text)} chars is kept as its first {RESULT_CHARS}")
            text = text[:RESULT_CHARS] + RESULT_CUT.format(chars=len(text))
        return text.encode("utf-8")

    def set_status_line(self, conversation, html: Optional[str]) -> None:
        if html:
            trace("App chat: a status line is not shown, the app's command echoes are lines of their own")

    def send_approval_prompt(self, conversation, token: str, title: str, details: str) -> Any:
        info(f"App chat: showing approval prompt [{token[:8]}]: {title}")
        self._push_question(KIND_APPROVAL, APPROVAL_NEEDED, title)
        return self.transcript.upsert(self._prompt_id(KIND_APPROVAL, token), KIND_APPROVAL,
                                      token=token, title=title, details=details, outcome="")

    def close_approval_prompt(self, conversation, prompt_ref, outcome_text: str) -> None:
        self._close_prompt(KIND_APPROVAL, prompt_ref, outcome_text)

    def send_secret_prompt(self, conversation, token: str, title: str, details: str) -> Any:
        info(f"App chat: showing secret prompt [{token[:8]}]: {title}")
        self._push_question(KIND_SECRET, SECRET_NEEDED, title)
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
