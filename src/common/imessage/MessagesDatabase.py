import os
import sqlite3
from pathlib import Path
from typing import Callable, Iterable, Optional

from common.imessage.Handles import phone_suffix
from common.imessage.MessagesHost import (Attachment, GroupChat, Message, NO_ACCESS, NO_DATABASE,
                                          SCHEDULE_COMPLETED, SCHEDULE_FAILED)
from common.logging import info, trace, fine

CHAT_DB_PATH = os.path.expanduser("~/Library/Messages/chat.db")
APPLE_EPOCH_OFFSET = 978_307_200
GROUP_CHAT_STYLE = 43           # chat.style of a group conversation; 45 is a 1:1 chat

# chat.db send-later columns: schedule_type=2 marks a scheduled row (Apple resets it to 0
# on the actual send); schedule_state 3 = completed, 4 = failed (error set, never sent)
SCHEDULE_TYPE_SEND_LATER = 2
SCHEDULE_STATES = {3: SCHEDULE_COMPLETED, 4: SCHEDULE_FAILED}
# Send Later arrived with macOS 15; a Sonoma chat.db has neither column, and a query naming them
# failed on every poll there (a beta tester's Mac, 2026-10-16: the feed and the Apple 2FA codes it
# forwards were both dead from the first fetch)
SCHEDULE_COLUMNS = ("schedule_type", "schedule_state")
CHAT_ACTIVITY_CHUNK = 500      # handle rowids per chat.db query, well under SQLite's parameter limit

# What a handle is called when nobody resolves contacts: itself.
Namer = Callable[[Optional[str]], str]


def unnamed(handle: Optional[str]) -> str:
    return handle or "Unknown"


def apple_date(unix_ts: Optional[float]) -> int:
    """A unix time as chat.db stores dates (nanoseconds since 2001); 0 — before every row —
    for no time at all."""
    if unix_ts is None:
        return 0
    return int((unix_ts - APPLE_EPOCH_OFFSET) * 1_000_000_000)


def unix_time(apple_ts: int) -> float:
    return apple_ts / 1_000_000_000 + APPLE_EPOCH_OFFSET


def attributed_text(blob: bytes) -> Optional[str]:
    """Extract plain text from an NSAttributedString typedstream blob."""
    if not blob:
        return None
    try:
        import Foundation
        ns_data = Foundation.NSData.dataWithBytes_length_(blob, len(blob))
        attr_str = Foundation.NSUnarchiver.unarchiveObjectWithData_(ns_data)
        return str(attr_str.string()) if attr_str else None
    except Exception:
        return None


OBJECT_REPLACEMENT = "\ufffc"      # what the body holds where an attachment stands: a placeholder, not a word


def _body(text: Optional[str], attributed_body: Optional[bytes]) -> Optional[str]:
    if not text:
        text = attributed_text(attributed_body)
    return text.replace(OBJECT_REPLACEMENT, "").strip() if text else text


def _group_name(display_name: Optional[str]) -> Optional[str]:
    return display_name.strip() if display_name and display_name.strip() else None


def _tapback_target(associated_guid: Optional[str]) -> str:
    """The guid of the bubble a tapback was put on: chat.db stores it as `p:<part>/<guid>` (a
    part of a bubble) or `bp:<guid>` (the bubble whole)."""
    if not associated_guid:
        return ""
    return associated_guid.rsplit("/", 1)[-1].removeprefix("bp:")


class MessagesDatabase:
    """Reads chat.db, the SQLite database Messages.app keeps on a Mac.

    The rows after a ROWID within a horizon, decoded (the attributed body, the attachments, the
    Send Later columns), the group chats, and what the database knows about a handle. The file
    is opened read-only and per call; every method but the three that say otherwise raises
    what sqlite raises — `LocalMessagesHost` turns that into the error of its answer. A reader
    passes a `Namer` to have handles turned into contact names on the way out."""

    def __init__(self, path: Optional[str] = None):
        self.path = str(path or CHAT_DB_PATH)
        # chat.db has the Send Later columns (macOS 15+); assumed until `probe_schema` has read the table
        self._has_schedule_columns = True

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)

    def check(self) -> tuple[str, str]:
        """("", "") when the message table can be read, else (NO_DATABASE or NO_ACCESS, the
        fact). Never raises."""
        try:
            with self._connect() as conn:
                conn.execute("SELECT 1 FROM message LIMIT 1")
        except sqlite3.OperationalError as e:
            if not os.path.exists(self.path):
                return NO_DATABASE, f"{self.path} does not exist — Messages.app is not set up on this Mac"
            # sqlite says "unable to open database file" whether the file is missing or macOS
            # refused it; the file is there, so it is the TCC refusal
            return NO_ACCESS, f"cannot read {self.path}: {e}"
        except Exception as e:
            return NO_ACCESS, f"cannot read {self.path}: {e}"
        return "", ""

    def probe_schema(self) -> bool:
        """Whether this Mac's chat.db has the Send Later columns; without them every row is read
        as a plain message and no scheduled send is found. Never raises: a schema that cannot be
        read keeps the columns in the queries, so the read, not the probe, reports what is wrong."""
        self._has_schedule_columns = self._read_schedule_columns()
        return self._has_schedule_columns

    def _read_schedule_columns(self) -> bool:
        try:
            with self._connect() as conn:
                columns = {row[1] for row in conn.execute("PRAGMA table_info(message)")}
        except sqlite3.OperationalError as e:
            info(f"iMessage: could not read the message table's schema ({e}) — assuming the Send Later columns")
            return True
        missing = [c for c in SCHEDULE_COLUMNS if c not in columns]
        if missing:
            info(f"iMessage: chat.db has no {', '.join(missing)} column(s) (macOS before 15) — "
                 "send-later tracking off, messages read as usual")
            return False
        trace("iMessage: chat.db has the Send Later columns")
        return True

    def cursor_at(self, horizon_ts: Optional[float]) -> int:
        """The ROWID just before the first message dated at or after the horizon (past every
        row when there is none). Not the last ROWID from before the horizon: chat.db's ROWIDs
        are not in date order — Messages in iCloud downloads the history in chunks, newest
        first, old, recent again — and on such a library that ROWID sits near the end of the
        table and a catch-up from it skips years."""
        with self._connect() as conn:
            first = conn.execute("SELECT MIN(ROWID) - 1 FROM message WHERE date >= ?",
                                 (apple_date(horizon_ts),)).fetchone()[0]
            return first if first is not None else self._last_rowid(conn)

    def last_rowid(self) -> int:
        """The end of the message table, whatever its rows are dated (`_last_rowid`)."""
        with self._connect() as conn:
            return self._last_rowid(conn)

    @staticmethod
    def _last_rowid(conn) -> int:
        """The highest ROWID the message table ever gave out. chat.db numbers its messages with
        AUTOINCREMENT, so this is the table's counter: deleting the newest message lowers
        MAX(ROWID) and not this, and a cursor that stood on the deleted row is still a place in
        this database. A table without the counter answers with its highest row."""
        last = conn.execute("SELECT COALESCE(MAX(ROWID), 0) FROM message").fetchone()[0]
        try:
            counter = conn.execute("SELECT seq FROM sqlite_sequence WHERE name = 'message'").fetchone()
        except sqlite3.OperationalError:
            trace("iMessage: chat.db keeps no ROWID counter, the last row stands for the end of the table")
            return last
        return max(last, counter[0]) if counter and counter[0] is not None else last

    def count_after(self, after_rowid: int, horizon_ts: Optional[float]) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM message WHERE ROWID > ? AND date >= ? AND is_system_message = 0",
                               (after_rowid, apple_date(horizon_ts))).fetchone()
        return row[0]

    def _schedule_select(self) -> str:
        return ("m.schedule_type, m.schedule_state" if self._has_schedule_columns
                else "0 AS schedule_type, 0 AS schedule_state")

    def messages_after(self, after_rowid: int, horizon_ts: Optional[float], limit: int,
                       name: Namer = unnamed) -> tuple[list[Message], int, int]:
        """(messages, the highest ROWID read, rows read) for the rows after `after_rowid` dated
        at or after the horizon, at most `limit` of them. The date test is what keeps a row an
        iCloud sync writes late — a 2013 message at a fresh ROWID — out of the answer. A Send
        Later row comes without body or files (it is not sent yet); a row with neither text nor
        a file — on disk or still to arrive — is read and left out."""
        messages: list[Message] = []
        cursor, scanned = after_rowid, 0
        with self._connect() as conn:
            rows = conn.execute(f"""
                SELECT m.ROWID, m.text, m.attributedBody, m.date, m.is_from_me,
                       h.id AS handle_id,
                       c.display_name AS chat_name,
                       {self._schedule_select()}, cmj.chat_id,
                       m.guid, m.service, m.thread_originator_guid, m.associated_message_type,
                       m.associated_message_guid, c.guid AS chat_guid, c.style
                FROM message m
                LEFT JOIN handle h ON m.handle_id = h.ROWID
                LEFT JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
                LEFT JOIN chat c ON cmj.chat_id = c.ROWID
                WHERE m.ROWID > ?
                  AND m.date >= ?
                  AND m.is_system_message = 0
                ORDER BY m.ROWID
                LIMIT ?
            """, (after_rowid, apple_date(horizon_ts), limit))
            for (rowid, text, attributed_body, apple_ts, is_from_me, handle, chat_name,
                 schedule_type, schedule_state, chat_id, guid, service, thread_guid, associated_type,
                 associated_guid, chat_guid, chat_style) in rows:
                scanned += 1
                cursor = max(cursor, rowid)
                message = Message(rowid, unix_time(apple_ts), bool(is_from_me), handle=handle,
                                  chat_name=_group_name(chat_name),
                                  send_later=schedule_type == SCHEDULE_TYPE_SEND_LATER,
                                  schedule_state=SCHEDULE_STATES.get(schedule_state, ""),
                                  in_chat=chat_id is not None, guid=guid or "", chat_guid=chat_guid or "",
                                  is_group=chat_style == GROUP_CHAT_STYLE, service=service or "",
                                  tapback=associated_type or 0)
                if not message.send_later:
                    message.text = _body(text, attributed_body)
                    message.attachments, message.pending_attachments = self._attachments(conn, rowid)
                    if not message.text and not message.attachments and not message.pending_attachments:
                        continue
                    message.reply_to = _tapback_target(associated_guid) if message.tapback else thread_guid or ""
                    if message.reply_to:
                        message.reply_to_from_me, message.reply_to_text = self._referred(conn, message.reply_to, chat_id)
                    if not message.is_from_me:
                        message.sender = name(handle)
                    elif not message.chat_name:
                        message.recipients = self._recipients(conn, rowid, name)
                messages.append(message)
        return messages, cursor, scanned

    def messages(self, rowids: list[int], name: Namer = unnamed) -> list[Message]:
        """The given rows as they are now, with body and recipients and without attachments. A
        reschedule only moves the date on the same row, and Apple resets schedule_type when the
        message sends — so this is how a tracked Send Later row is seen to have gone out."""
        if not rowids:
            return []
        placeholders = ",".join("?" * len(rowids))
        with self._connect() as conn:
            rows = conn.execute(f"""
                SELECT m.ROWID, m.text, m.attributedBody, m.date, m.is_from_me,
                       c.display_name AS chat_name,
                       {self._schedule_select()}, cmj.chat_id
                FROM message m
                LEFT JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
                LEFT JOIN chat c ON cmj.chat_id = c.ROWID
                WHERE m.ROWID IN ({placeholders})
            """, tuple(rowids)).fetchall()
            messages = []
            for (rowid, text, attributed_body, apple_ts, is_from_me, chat_name,
                 schedule_type, schedule_state, chat_id) in rows:
                message = Message(rowid, unix_time(apple_ts), bool(is_from_me), text=_body(text, attributed_body),
                                  chat_name=_group_name(chat_name),
                                  send_later=schedule_type == SCHEDULE_TYPE_SEND_LATER,
                                  schedule_state=SCHEDULE_STATES.get(schedule_state, ""),
                                  in_chat=chat_id is not None)
                if not message.chat_name:
                    message.recipients = self._recipients(conn, rowid, name)
                messages.append(message)
        return messages

    def send_later(self) -> list[Message]:
        """Every row marked as a Send Later message, the copies iCloud leaves behind included —
        time, direction, state and whether a chat holds it. Empty on a chat.db without the columns."""
        if not self._has_schedule_columns:
            trace("iMessage: no Send Later columns, no scheduled message to find")
            return []
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT m.ROWID, m.date, m.is_from_me, m.schedule_state, cmj.chat_id
                FROM message m
                LEFT JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
                WHERE m.schedule_type = ?
            """, (SCHEDULE_TYPE_SEND_LATER,)).fetchall()
        return [Message(rowid, unix_time(apple_ts), bool(is_from_me), send_later=True,
                        schedule_state=SCHEDULE_STATES.get(schedule_state, ""), in_chat=chat_id is not None)
                for rowid, apple_ts, is_from_me, schedule_state, chat_id in rows]

    @staticmethod
    def _referred(conn, guid: str, chat_id: Optional[int]) -> tuple[bool, Optional[str]]:
        """(the account's own, its body) for the bubble a reply or a tapback refers to, looked
        up in the chat the reply is in: a guid names a bubble of that chat or nothing — never
        the text of another conversation. (False, None) when chat.db has no such row there."""
        row = conn.execute("""SELECT m.is_from_me, m.text, m.attributedBody FROM message m
                              JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
                              WHERE m.guid = ? AND cmj.chat_id = ?""", (guid, chat_id)).fetchone()
        if row is None:
            fine(f"iMessage: the bubble {guid} a message refers to is not in its chat in chat.db")
            return False, None
        return bool(row[0]), _body(row[1], row[2])

    @staticmethod
    def _recipients(conn, message_rowid: int, name: Namer) -> list[str]:
        """The other participants of a message's chat by name, sorted.

        Apple stores message.handle_id = 0 for outgoing messages, so the recipient
        cannot be read from the message row — it lives on the chat: one name for a 1:1
        chat, several for an unnamed group, none when no participants can be found."""
        rows = conn.execute("""
            SELECT DISTINCT h.id
            FROM chat_message_join cmj
            JOIN chat_handle_join chj ON cmj.chat_id = chj.chat_id
            JOIN handle h ON chj.handle_id = h.ROWID
            WHERE cmj.message_id = ?
        """, (message_rowid,)).fetchall()
        handles = [r[0] for r in rows if r[0]]
        if not handles:
            fine(f"iMessage: no chat participants found for message (rowid={message_rowid})")
        return sorted(name(h) for h in handles)

    @staticmethod
    def _expand(filename: str) -> str:
        return filename.replace("~", str(Path.home()), 1) if filename.startswith("~") else filename

    @classmethod
    def _attachments(cls, conn, message_rowid: int) -> tuple[list[Attachment], int]:
        """(the attachments of a message whose file is on this disk, of any type; how many
        more chat.db lists whose file is not there — Messages writes the row before a large
        attachment has finished downloading, so a reader may want to look again)."""
        rows = conn.execute("""
            SELECT a.ROWID, a.filename, a.mime_type
            FROM attachment a
            JOIN message_attachment_join maj ON a.ROWID = maj.attachment_id
            WHERE maj.message_id = ?
              AND a.filename IS NOT NULL
        """, (message_rowid,)).fetchall()
        attachments, missing = [], 0
        for attachment_id, filename, mime_type in rows:
            if filename.endswith(".pluginPayloadAttachment"):
                continue  # iMessage app payloads (link previews, stickers, Apple Pay…) — never a file to show
            expanded = cls._expand(filename)
            if os.path.isfile(expanded):
                attachments.append(Attachment(attachment_id, os.path.basename(expanded), mime_type))
            else:
                missing += 1
                fine(f"iMessage: attachment file not found (not downloaded yet, or removed): {expanded}")
        return attachments, missing

    def _attachment_path(self, attachment_id: int) -> str:
        """Where the file chat.db lists under this attachment id is — the id is the whole
        reference, so only a file Messages attached can be asked for."""
        with self._connect() as conn:
            row = conn.execute("SELECT filename FROM attachment WHERE ROWID = ?", (attachment_id,)).fetchone()
        if not row or not row[0]:
            raise FileNotFoundError(f"chat.db has no file for attachment {attachment_id}")
        return self._expand(row[0])

    def attachment_size(self, attachment_id: int) -> int:
        """The size of an attachment's file in bytes, without reading it."""
        return os.path.getsize(self._attachment_path(attachment_id))

    def read_attachment(self, attachment_id: int) -> bytes:
        with open(self._attachment_path(attachment_id), "rb") as f:
            return f.read()

    def chat_activity(self, handles: Iterable[str]) -> dict[str, tuple[bool, int, int]]:
        """What chat.db knows about the given handles: (seen on the iMessage service, Apple date of
        the latest message, message count), keyed by the handle id and, for phones, by its 9-digit
        suffix so a card's local-format number matches the E.164 handle Messages stores. Messages
        are counted through the chats the handle takes part in, group chats included — Apple stores
        handle_id = 0 on outgoing messages (see `_recipients`), so joining on it would
        count only the contact's replies and miss a conversation where the owner does the writing.
        Only the handle rows matching the wanted ids or suffixes are read. Empty when chat.db
        cannot be read — never raises."""
        wanted_ids = {h.lower() for h in handles}
        wanted_suffixes = {s for h in handles if "@" not in h and (s := phone_suffix(h))}
        if not wanted_ids:
            return {}
        activity: dict[str, tuple[bool, int, int]] = {}

        def merge(key, imessage, last_date, count):
            seen = activity.get(key, (False, 0, 0))
            activity[key] = (seen[0] or imessage, max(seen[1], last_date), seen[2] + count)

        matched: dict[int, tuple[str, str, str | None]] = {}
        counts: dict[int, tuple[int, int]] = {}
        try:
            with self._connect() as conn:
                for rowid, handle_id, service in conn.execute("SELECT ROWID, id, service FROM handle"):
                    if not handle_id:
                        continue
                    suffix = phone_suffix(handle_id) if "@" not in handle_id else None
                    if handle_id.lower() in wanted_ids or (suffix and suffix in wanted_suffixes):
                        matched[rowid] = (handle_id, service or "", suffix)
                rowids = list(matched)
                for start in range(0, len(rowids), CHAT_ACTIVITY_CHUNK):
                    chunk = rowids[start:start + CHAT_ACTIVITY_CHUNK]
                    marks = ",".join("?" * len(chunk))
                    for rowid, count, last_date in conn.execute(
                            "SELECT chj.handle_id, COUNT(m.ROWID), COALESCE(MAX(m.date), 0) "
                            "FROM chat_handle_join chj "
                            "JOIN chat_message_join cmj ON cmj.chat_id = chj.chat_id "
                            "JOIN message m ON m.ROWID = cmj.message_id "
                            f"WHERE chj.handle_id IN ({marks}) GROUP BY chj.handle_id", chunk):
                        counts[rowid] = (count or 0, last_date or 0)
        except Exception as e:
            info(f"iMessage: chat.db handle query failed, ranking contact handles by card only: {e}")
            return {}
        for rowid, (handle_id, service, suffix) in matched.items():
            imessage = service.lower() == "imessage"
            count, last_date = counts.get(rowid, (0, 0))
            merge(handle_id.lower(), imessage, last_date, count)
            if suffix:
                merge(suffix, imessage, last_date, count)
        trace(f"iMessage: chat.db knows {len(matched)} handle rows for {len(wanted_ids)} contested handles, "
              f"{len(counts)} with messages")
        return activity

    def group_chats(self, name: Namer = unnamed) -> list[GroupChat]:
        """Every group chat of chat.db with its members named. Read per lookup — a few dozen
        rows — so a group created a minute ago is already sendable. Empty when chat.db cannot
        be read — never raises."""
        try:
            with self._connect() as conn:
                member_rows = conn.execute("""
                    SELECT c.ROWID, c.guid, c.display_name, h.id
                    FROM chat c
                    JOIN chat_handle_join chj ON chj.chat_id = c.ROWID
                    JOIN handle h ON h.ROWID = chj.handle_id
                    WHERE c.style = ?
                """, (GROUP_CHAT_STYLE,)).fetchall()
                activity = dict(conn.execute("""
                    SELECT cmj.chat_id, MAX(m.date)
                    FROM chat c
                    JOIN chat_message_join cmj ON cmj.chat_id = c.ROWID
                    JOIN message m ON m.ROWID = cmj.message_id
                    WHERE c.style = ?
                    GROUP BY cmj.chat_id
                """, (GROUP_CHAT_STYLE,)).fetchall())
        except Exception as e:
            info(f"iMessage: chat.db group chat query failed, no group can be addressed: {e}")
            return []
        chats: dict[int, tuple[str, Optional[str], set[str], set[str]]] = {}
        for rowid, guid, display_name, handle in member_rows:
            if not guid:
                continue
            _, _, members, handles = chats.setdefault(rowid, (guid, (display_name or "").strip() or None, set(), set()))
            if handle:
                members.add(name(handle))
                handles.add(handle)
        groups = [GroupChat(guid, group_name, sorted(members), activity.get(rowid) or 0, sorted(handles))
                  for rowid, (guid, group_name, members, handles) in chats.items()]
        trace(f"iMessage: chat.db holds {len(groups)} group chats, {sum(1 for g in groups if g.name)} named")
        return groups
