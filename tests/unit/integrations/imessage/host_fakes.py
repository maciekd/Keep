"""A MessagesHost that answers from lists: the integration's side of iMessage — what it makes of
the rows, what it tracks, what it tells the assistant — is tested over it, with no chat.db."""
import time
from typing import Optional

from common.imessage.MessagesHost import (Attachment, AttachmentData, Count, GroupChat, GroupList, HostHealth, Message,
                                          MessageBatch, MessagesHost, Recipient, SendResult, TypingResult)

RECENT = time.time() - 5
FUTURE = time.time() + 3600


def received(rowid, text, sender="Alice", handle="+1234567890", chat_name=None, ts=RECENT, attachments=()):
    return Message(rowid, ts, False, text, handle, sender, chat_name, attachments=list(attachments))


def sent(rowid, text, recipients=("Alice",), chat_name=None, ts=RECENT):
    return Message(rowid, ts, True, text, chat_name=chat_name, recipients=list(recipients))


def send_later(rowid, ts=FUTURE, is_from_me=True, state="", in_chat=True, text=None, recipients=("Alice",)):
    """A row still marked as a Send Later message; with `text`, the way `messages` reads it again."""
    return Message(rowid, ts, is_from_me, text, recipients=list(recipients), send_later=True,
                   schedule_state=state, in_chat=in_chat)


class FakeHost(MessagesHost):
    def __init__(self):
        self.rows: list[Message] = []           # what a poll finds, in ROWID order
        self.tracked: list[Message] = []        # what `messages` answers for the rows it is asked about
        self.scheduled_rows: list[Message] = []
        self.files: dict[int, bytes] = {}
        self.sizes: dict[int, int] = {}         # what `attachment_size` answers; a file not listed has no known size
        self.known: dict[str, Recipient] = {}   # recipient string, lowercased → what it resolves to
        self.health_answer = HostHealth(True, contacts=1)
        self.send_answer = SendResult(service="sent:imessage")
        self.start = 0
        self.error = ""                         # set: every read fails with it
        self.calls: list[tuple] = []
        self.group_chats: list[GroupChat] = []  # what `groups` lists
        self.chat_answer = SendResult(service="sent:imessage")      # how a `send_chat` goes
        self.is_away = False                    # what `away` says after a failed read
        self.typing_answer: Optional[TypingResult] = None   # how a `typing` goes; None: the dots do as asked

    def health(self, refresh: bool = False) -> HostHealth:
        self.calls.append(("health", refresh))
        return self.health_answer

    def cursor_at(self, horizon_ts: Optional[float]) -> Count:
        return Count(self.start, self.error)

    def last_rowid(self) -> Count:
        return Count(max([self.start] + [m.rowid for m in self.rows]), self.error)

    def away(self) -> bool:
        return self.is_away

    def remaining(self, after_rowid: int, horizon_ts: Optional[float]) -> Count:
        return Count(sum(1 for m in self.rows if m.rowid > after_rowid), self.error)

    def poll(self, after_rowid: int, horizon_ts: Optional[float], limit: int) -> MessageBatch:
        self.calls.append(("poll", after_rowid, horizon_ts, limit))
        if self.error:
            return MessageBatch(cursor=after_rowid, error=self.error)
        rows = [m for m in self.rows if m.rowid > after_rowid][:limit]
        return MessageBatch(rows, max([after_rowid] + [m.rowid for m in rows]), exhausted=len(rows) < limit)

    def messages(self, rowids: list[int]) -> MessageBatch:
        self.calls.append(("messages", list(rowids)))
        return MessageBatch([m for m in self.tracked if m.rowid in rowids], error=self.error)

    def scheduled(self) -> MessageBatch:
        self.calls.append(("scheduled",))
        return MessageBatch(list(self.scheduled_rows), error=self.error)

    def attachment(self, attachment_id: int) -> AttachmentData:
        raise AssertionError("the integration reads attachments through read_attachment")

    def attachment_size(self, attachment_id: int) -> Count:
        if attachment_id in self.sizes:
            return Count(self.sizes[attachment_id])
        return super().attachment_size(attachment_id)

    def read_attachment(self, attachment_id: int):
        if attachment_id in self.files:
            return self.files[attachment_id], ""
        return None, f"no file for attachment {attachment_id}"

    def resolve(self, recipient: str, owner: str = "") -> Recipient:
        self.calls.append(("resolve", recipient, owner))
        return self.known.get(recipient.lower(), Recipient(error=f"unknown contact: {recipient} (cache has 1 names)"))

    def send(self, recipient: str, body: str, owner: str = "") -> SendResult:
        self.calls.append(("send", recipient, body, owner))
        resolved = self.resolve(recipient, owner)
        if resolved.error:
            return SendResult(error=resolved.error, recipient=resolved)
        return SendResult(self.send_answer.error, resolved, self.send_answer.service, self.send_answer.unauthorized)

    def send_chat(self, chat: str, body: str) -> SendResult:
        self.calls.append(("send_chat", chat, body))
        return SendResult(self.chat_answer.error, None, self.chat_answer.service, self.chat_answer.unauthorized)

    def groups(self) -> GroupList:
        self.calls.append(("groups",))
        return GroupList(list(self.group_chats), self.error)

    def typing(self, chat: str, on: bool) -> TypingResult:
        self.calls.append(("typing", chat, on))
        return self.typing_answer or TypingResult(showing=on)

    def sent(self, chat: Optional[str] = None) -> list[str]:
        """The bodies `send_chat` was asked to send, to one chat or to any."""
        return [call[2] for call in self.calls if call[0] == "send_chat" and chat in (None, call[1])]


def photo(attachment_id=40, name="IMG_0001.jpg", mime_type="image/jpeg"):
    return Attachment(attachment_id, name, mime_type)
