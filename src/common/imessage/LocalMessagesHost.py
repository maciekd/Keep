import base64
from typing import Optional

from common.imessage.Contacts import Contacts
from common.imessage.MessagesDatabase import MessagesDatabase
from common.imessage.MessagesHost import (AttachmentData, Count, GroupList, HostHealth, MessageBatch, MessagesHost,
                                          Recipient, SendResult, TYPING_FAILED, TYPING_NO_NAME, TypingResult)
from common.imessage.MessagesSender import MessagesSender
from common.imessage.TypingIndicator import TypingIndicator
from common.logging import info


class LocalMessagesHost(MessagesHost):
    """The Messages account of this Mac user, reached in-process.

    One `MessagesDatabase`, one `Contacts`, one `MessagesSender` and one `TypingIndicator`
    behind the `MessagesHost` calls. Whatever they raise ends here as the error of the answer,
    logged — the caller sees values only, as it would over a wire."""

    def __init__(self, database: Optional[MessagesDatabase] = None, contacts: Optional[Contacts] = None,
                 sender: Optional[MessagesSender] = None, typing_indicator: Optional[TypingIndicator] = None):
        # `is None`, not truthiness: Contacts has a length, and an empty cache is still the one to use
        self.database = database if database is not None else MessagesDatabase()
        self.contacts = contacts if contacts is not None else Contacts(self.database)
        self.sender = sender if sender is not None else MessagesSender()
        self.typing_indicator = typing_indicator if typing_indicator is not None else TypingIndicator()
        self._unnamed: dict[str, str] = {}      # chat → why it has no sidebar name, as last logged

    @staticmethod
    def _failed(what: str, e: Exception) -> str:
        """The error of an answer. Logged at info: the consumer reports it as its own failure."""
        reason = f"{type(e).__name__}: {e}"
        info(f"iMessage host: {what} failed: {reason}")
        return reason

    def health(self, refresh: bool = False) -> HostHealth:
        reason, detail = self.database.check()
        if reason:
            info(f"iMessage host: not healthy ({reason}): {detail}")
            return HostHealth(False, reason, detail, len(self.contacts))
        if refresh:
            self.database.probe_schema()
            self.contacts.refresh()
            info(f"iMessage host: schema and contacts re-read ({len(self.contacts)} contact entries)")
        return HostHealth(True, contacts=len(self.contacts), typing=self.typing_indicator.blocked())

    def cursor_at(self, horizon_ts: Optional[float]) -> Count:
        try:
            return Count(self.database.cursor_at(horizon_ts))
        except Exception as e:
            return Count(error=self._failed("reading the first ROWID of the horizon", e))

    def last_rowid(self) -> Count:
        try:
            return Count(self.database.last_rowid())
        except Exception as e:
            return Count(error=self._failed("reading the last ROWID", e))

    def remaining(self, after_rowid: int, horizon_ts: Optional[float]) -> Count:
        try:
            return Count(self.database.count_after(after_rowid, horizon_ts))
        except Exception as e:
            return Count(error=self._failed("counting the rows ahead", e))

    def poll(self, after_rowid: int, horizon_ts: Optional[float], limit: int) -> MessageBatch:
        try:
            self.contacts.refresh_if_stale()
            messages, cursor, scanned = self.database.messages_after(after_rowid, horizon_ts, limit,
                                                                     self.contacts.name_for)
        except Exception as e:
            return MessageBatch(cursor=after_rowid, error=self._failed("the poll", e))
        return MessageBatch(messages, cursor, exhausted=scanned < limit)

    def messages(self, rowids: list[int]) -> MessageBatch:
        try:
            self.contacts.refresh_if_stale()
            return MessageBatch(self.database.messages(rowids, self.contacts.name_for))
        except Exception as e:
            return MessageBatch(error=self._failed(f"reading {len(rowids)} row(s)", e))

    def scheduled(self) -> MessageBatch:
        try:
            return MessageBatch(self.database.send_later())
        except Exception as e:
            return MessageBatch(error=self._failed("reading the Send Later rows", e))

    def attachment_size(self, attachment_id: int) -> Count:
        try:
            return Count(self.database.attachment_size(attachment_id))
        except Exception as e:
            return Count(error=self._failed(f"reading the size of attachment {attachment_id}", e))

    def read_attachment(self, attachment_id: int) -> tuple[Optional[bytes], str]:
        try:
            return self.database.read_attachment(attachment_id), ""
        except Exception as e:
            info(f"iMessage host: attachment {attachment_id} could not be read: {e}")
            return None, str(e)

    def attachment(self, attachment_id: int) -> AttachmentData:
        raw, reason = self.read_attachment(attachment_id)
        if raw is None:
            return AttachmentData(error=reason)
        return AttachmentData(base64.b64encode(raw).decode("ascii"))

    def resolve(self, recipient: str, owner: str = "") -> Recipient:
        try:
            return self.contacts.resolve(recipient, owner)
        except Exception as e:
            return Recipient(error=self._failed(f"resolving '{recipient}'", e))

    def send(self, recipient: str, body: str, owner: str = "") -> SendResult:
        try:
            refusal = self.sender.refusal(body)
            if refusal:
                return SendResult(error=refusal)
            resolved = self.resolve(recipient, owner)
            if resolved.error:
                info(f"iMessage send: could not resolve recipient '{recipient}': {resolved.error}")
                return SendResult(error=resolved.error, recipient=resolved)
            result = self.sender.send(resolved.handle, body)
            result.recipient = resolved
        except Exception as e:
            return SendResult(error=self._failed(f"the send to '{recipient}'", e))
        if not result.error:
            info(f"iMessage {result.service} to {resolved.group.describe() if resolved.group else resolved.handle}: "
                 f"{body!r}")
        return result

    def send_chat(self, chat: str, body: str) -> SendResult:
        try:
            refusal = self.sender.refusal(body, chat=True)
            if refusal:
                return SendResult(error=refusal)
            result = self.sender.send(self.sender.chat_target(chat), body, chat=True)
        except Exception as e:
            return SendResult(error=self._failed(f"the send to the chat {chat}", e))
        if not result.error:
            info(f"iMessage {result.service} to the chat {chat}: {body!r}")
        return result

    def groups(self) -> GroupList:
        try:
            self.contacts.refresh_if_stale()
            return GroupList(self.contacts.group_chats())
        except Exception as e:
            return GroupList(error=self._failed("reading the group chats", e))

    def typing(self, chat: str, on: bool) -> TypingResult:
        """The dots go on in the window the sidebar of this Mac names like the chat — a name
        only this side can know, since the sidebar shows this Mac's contacts — and nowhere
        when that name is not the chat's alone."""
        try:
            if not on:
                return self.typing_indicator.off(chat)
            name, why = self.contacts.sidebar_name(chat)
            if not name:
                if self._unnamed.get(chat) != why:
                    self._unnamed[chat] = why
                    info(f"iMessage host: no typing dots in {chat} — {why}. The reply is sent all the same.")
                return TypingResult(reason=TYPING_NO_NAME, detail=why)
            self._unnamed.pop(chat, None)
            return self.typing_indicator.on(chat, name)
        except Exception as e:
            return TypingResult(reason=TYPING_FAILED, error=self._failed(f"typing {'on' if on else 'off'} in {chat}", e))

    def close(self) -> None:
        self.typing_indicator.shutdown()
