import os
import re
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple
from assistant.SystemNote import system_note

from assistant.integration.Feed import Feed, FeedBatch, FeedEntry
from assistant.integration.Integration import Integration
from common.ConfigManager import config
from common.logging import info, trace, error, fine, note
from assistant.PromptInjectionDetector import is_prompt_injection
from assistant.tools.MediaAnalyzer import media_analyzer, media_kind, KIND_VIDEO
from common.RateLimiter import RateLimiter


CHAT_DB_PATH = os.path.expanduser("~/Library/Messages/chat.db")
# What the LLM relays when macOS keeps the server out of chat.db or the AddressBook. The grant
# goes to Terminal, the app macOS holds responsible for the server (start.sh launches it from
# there), and a process that was refused once stays refused — so a fresh process, not a reopen
# in this one. The prompt shows only the first ~80 characters of an error, so the action leads.
FULL_DISK_ACCESS_HINT = ("needs Full Disk Access for Terminal, then restart_server: System Settings > Privacy & "
                         "Security > Full Disk Access > turn on Terminal (the server is launched from it); the "
                         "running process keeps the refusal, so run restart_server afterwards — no need to quit Terminal")
APPLE_EPOCH_OFFSET = 978_307_200
POLL_INTERVAL = 5
FETCH_BATCH_SIZE = 500   # chat.db rows read per fetch; a shorter batch means the catch-up is over
CONTACT_CACHE_REFRESH_INTERVAL = 30 * 60
APPLE_2FA_PATTERN = re.compile(r"Apple Account Code is:\s*(\d{6})")
RING_2FA_PATTERN = re.compile(r"Your Ring verification code is:\s*(\d{6})")
TFA_CODE_MAX_AGE = 120  # only auto-submit 2FA codes from messages received within this many seconds

SEND_RATE_LIMIT_WINDOW = 60.0
SEND_RATE_LIMIT_MAX = 5
APPLESCRIPT_TIMEOUT = 15
APPLESCRIPT_RESULT_IMESSAGE = "sent:imessage"
APPLESCRIPT_RESULT_SMS = "sent:sms"

# chat.db send-later columns: schedule_type=2 marks a scheduled row (Apple resets it to 0
# on the actual send); schedule_state 3 = completed, 4 = failed (error set, never sent)
SCHEDULE_TYPE_SEND_LATER = 2
SCHEDULE_STATE_COMPLETED = 3
SCHEDULE_STATE_FAILED = 4
# a scheduled message still unsent this long after its send time is never going to send
# from here (Send Later fires from whichever device is online) — stop tracking it
SCHEDULED_GRACE_PERIOD = 7 * 24 * 3600

# Which of a contact's handles send_imessage picks, best first: one chat.db knows as an iMessage
# handle (phone or email), a mobile number (AddressBook label or a Swedish 07x number), any other
# phone (landline, unlabelled), an email Messages has never seen, a fax/pager. Inside a tier the
# handle with the most recent conversation wins.
TIER_IMESSAGE, TIER_MOBILE, TIER_OTHER_PHONE, TIER_UNKNOWN_EMAIL, TIER_FAX = range(5)
TIER_NAMES = ("iMessage handle", "mobile", "other phone", "email unknown to Messages", "fax")
MOBILE_LABEL_PATTERN = re.compile(r"iphone|mobil|cell", re.IGNORECASE)     # "Mobile", "mobil" (sv)
FAX_LABEL_PATTERN = re.compile(r"fax|pager", re.IGNORECASE)
SWEDISH_MOBILE_PATTERN = re.compile(r"^(?:\+46|0046)7\d{8}$|^07\d{8}$")


class IMessageIntegration(Integration, Feed):
    """iMessage and SMS on the server's Mac.

    Brings the messages that arrive on the server's Mac into Keep's history, as a feed, and
    sends an iMessage or SMS to a contact when asked. It also catches the 2FA codes Apple and
    Ring text.

    Reads the macOS iMessage database as a Feed (plans/FEED_PLAN.md): `fetch` turns the rows
    after the cursor (a chat.db ROWID) into history entries; FeedRunner owns the thread, the
    cursor and the catch-up. Sending is rate-limited; it also resolves contact names to handles,
    forwards the detected Apple/Ring 2FA codes to their integrations, and dispatches scheduled
    send-later messages."""
    sensitive = True

    poll_interval = POLL_INTERVAL
    item_noun = "messages"
    count_in_telemetry = True
    default_backfill_months = 60    # five years: chat.db rows are cheap, no LLM call per message

    def __init__(self):
        super().__init__()
        self._last_seen_rowid = 0
        self._scanned_rows = 0          # rows the last _poll_new_messages read (batch-size test)
        self._open_error: Optional[str] = None
        # history rows produced outside fetch (a sent message) — handed over with the next batch
        self._pending_entries: list[FeedEntry] = []
        self._contact_cache: dict[str, str] = {}
        self._name_to_handle: dict[str, str] = {}
        self._handle_reasons: dict[str, str] = {}     # name key → why its handle won (for the log)
        self._handle_to_full_name: dict[str, str] = {}
        self._contact_cache_time = 0.0
        self._rate_limiter = RateLimiter(SEND_RATE_LIMIT_MAX, SEND_RATE_LIMIT_WINDOW)
        self._notified_cache_empty = False
        # 2FA codes detected but held for one poll cycle — see _settle_and_forward_2fa
        self._pending_apple_code: Optional[Tuple[float, str]] = None
        self._pending_ring_code: Optional[Tuple[float, str]] = None
        # send-later messages composed but not yet sent — see _check_scheduled_messages
        self._scheduled_rowids: set[int] = set()
        # rowids already logged as past-due-but-unsent, so the wait is noted once
        self._past_due_noted: set[int] = set()

    def _on_reset(self):
        from assistant.integration.FeedRunner import Feeds
        Feeds.get_instance().stop(self.feed_name)

    @property
    def name(self) -> str:
        return "iMessage"

    @property
    def config_keys(self) -> List[str]:
        return []

    def check_health(self) -> bool:
        try:
            with sqlite3.connect(f"file:{CHAT_DB_PATH}?mode=ro", uri=True) as conn:
                conn.execute("SELECT 1 FROM message LIMIT 1")
        except sqlite3.OperationalError as e:
            if not os.path.exists(CHAT_DB_PATH):
                raise RuntimeError(f"{CHAT_DB_PATH} does not exist — Messages.app is not set up on this Mac") from e
            # sqlite says "unable to open database file" whether the file is missing or macOS
            # refused it; the file is there, so it is the TCC refusal
            raise PermissionError(f"{FULL_DISK_ACCESS_HINT} (cannot read {CHAT_DB_PATH}: {e})") from e
        return True

    # --- Feed ---------------------------------------------------------------------------------

    @property
    def feed_name(self) -> str:
        return "iMessage"

    @property
    def config_slug(self) -> str:
        return "imessage"

    @property
    def open_error(self) -> Optional[str]:
        return self._open_error

    def open_feed(self) -> bool:
        try:
            self.check_health()
        except Exception as e:
            self._open_error = str(e) if isinstance(e, (PermissionError, RuntimeError)) else \
                f"{FULL_DISK_ACCESS_HINT} (cannot read {CHAT_DB_PATH}: {e})"
            error(f"iMessage: {self._open_error}")
            return False
        self._open_error = None
        self._refresh_contact_cache()
        self._seed_scheduled_messages()
        info(f"iMessage feed opened (contact cache: {len(self._contact_cache)} entries)")
        return True

    def initial_cursor(self, horizon_ts: float) -> Optional[str]:
        """The last ROWID from before the horizon, so the catch-up starts right after it."""
        apple_ts = (horizon_ts - APPLE_EPOCH_OFFSET) * 1_000_000_000
        with sqlite3.connect(f"file:{CHAT_DB_PATH}?mode=ro", uri=True) as conn:
            row = conn.execute("SELECT COALESCE(MAX(ROWID), 0) FROM message WHERE date < ?",
                               (int(apple_ts),)).fetchone()
        fine(f"iMessage: initial ROWID {row[0]} (last message before "
             f"{time.strftime('%Y-%m-%d', time.localtime(horizon_ts))})")
        return str(row[0])

    def remaining(self, cursor: Optional[str]) -> Optional[int]:
        with sqlite3.connect(f"file:{CHAT_DB_PATH}?mode=ro", uri=True) as conn:
            row = conn.execute("SELECT COUNT(*) FROM message WHERE ROWID > ? AND is_system_message = 0",
                               (int(cursor or 0),)).fetchone()
        return row[0]

    def fetch(self, cursor: Optional[str], backfill: bool) -> FeedBatch:
        self._last_seen_rowid = int(cursor or 0)
        entries = self._poll_new_messages(backfill)
        if self._pending_entries:
            entries = self._pending_entries + entries
            self._pending_entries = []
        return FeedBatch(entries, cursor=str(self._last_seen_rowid),
                         exhausted=self._scanned_rows < FETCH_BATCH_SIZE)

    def _seed_scheduled_messages(self):
        """Track the send-later messages composed before this start, so their eventual send
        is reported once (see _check_scheduled_messages)."""
        try:
            with sqlite3.connect(f"file:{CHAT_DB_PATH}?mode=ro", uri=True) as conn:
                scheduled = conn.execute("""
                    SELECT m.ROWID, m.date, m.is_from_me, m.schedule_state, cmj.chat_id
                    FROM message m
                    LEFT JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
                    WHERE m.schedule_type = ?
                """, (SCHEDULE_TYPE_SEND_LATER,)).fetchall()

            now = time.time()
            pending, ignored = [], []
            for rowid, apple_ts, is_from_me, schedule_state, chat_id in scheduled:
                unix_ts = apple_ts / 1_000_000_000 + APPLE_EPOCH_OFFSET
                reason = self._not_pending_reason(is_from_me, chat_id is not None, schedule_state, unix_ts, now)
                (ignored if reason else pending).append((rowid, reason))
            if ignored:
                fine(f"iMessage: ignoring {len(ignored)} schedule_type=2 row(s) that are not pending "
                     f"send-later messages: {', '.join(f'{r} ({why})' for r, why in ignored)}")
            if pending:
                self._scheduled_rowids.update(r for r, _ in pending)
                trace(f"iMessage: tracking {len(pending)} pending send-later message(s) from before "
                      f"startup (rowids={[r for r, _ in pending]})")
            else:
                fine("iMessage: no pending send-later messages")
        except Exception as e:
            error(f"iMessage: failed to read scheduled messages: {e}")

    def _poll_new_messages(self, backfill: bool = False) -> list[FeedEntry]:
        """The history entries for the chat.db rows after `_last_seen_rowid` (at most
        FETCH_BATCH_SIZE of them); advances `_last_seen_rowid` past every row it read. With
        `backfill` (catching up on months of history) nothing reacts — no send-later tracking,
        media as placeholders instead of vision-LLM descriptions — except that a 2FA code
        still fresh enough to submit is forwarded."""
        if time.time() - self._contact_cache_time > CONTACT_CACHE_REFRESH_INTERVAL:
            self._refresh_contact_cache()

        entries: list[FeedEntry] = []
        new_max_rowid = self._last_seen_rowid
        self._scanned_rows = 0
        # (received_at, code) per service — newest wins; forwarded after the batch so an
        # older code in the same batch can never be submitted ahead of a newer one
        apple_code: Optional[Tuple[float, str]] = None
        ring_code: Optional[Tuple[float, str]] = None

        with sqlite3.connect(f"file:{CHAT_DB_PATH}?mode=ro", uri=True) as conn:
            if not backfill:
                entries.extend(self._check_scheduled_messages(conn))

            cursor = conn.execute("""
                SELECT m.ROWID, m.text, m.attributedBody, m.date, m.is_from_me,
                       h.id AS handle_id,
                       c.display_name AS chat_name,
                       m.schedule_type, m.schedule_state, cmj.chat_id
                FROM message m
                LEFT JOIN handle h ON m.handle_id = h.ROWID
                LEFT JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
                LEFT JOIN chat c ON cmj.chat_id = c.ROWID
                WHERE m.ROWID > ?
                  AND m.is_system_message = 0
                ORDER BY m.ROWID
                LIMIT ?
            """, (self._last_seen_rowid, FETCH_BATCH_SIZE))

            for (rowid, text, attributed_body, apple_ts, is_from_me, handle_id, chat_name,
                 schedule_type, schedule_state, chat_id) in cursor:
                self._scanned_rows += 1
                unix_ts = apple_ts / 1_000_000_000 + APPLE_EPOCH_OFFSET

                if schedule_type == SCHEDULE_TYPE_SEND_LATER:
                    # Send-later message: chat.db gets the row at compose time with the
                    # send time in m.date, and Apple resets schedule_type to 0 only when
                    # it ACTUALLY sends — so schedule_type, not the timestamp, is the
                    # truth. A past-due row still at 2 (delivery delay, Mac offline at
                    # fire time) is unsent and must be tracked, not announced. Defer
                    # reporting to _check_scheduled_messages, the sole emitter.
                    reason = self._not_pending_reason(
                        is_from_me, chat_id is not None, schedule_state, unix_ts, time.time())
                    if reason:
                        fine(f"iMessage: ignoring schedule_type=2 row that is {reason}, "
                             f"not a pending send-later message (rowid={rowid})")
                    elif backfill:
                        fine(f"iMessage: send-later message seen during catch-up, not tracked (rowid={rowid})")
                    elif rowid not in self._scheduled_rowids:
                        sends_at = time.strftime('%Y-%m-%d %H:%M', time.localtime(unix_ts))
                        info(f"iMessage: send-later message detected (rowid={rowid}, sends {sends_at}), deferring until sent")
                        self._scheduled_rowids.add(rowid)
                    new_max_rowid = max(new_max_rowid, rowid)
                    continue

                if not text:
                    text = self._extract_attributed_text(attributed_body)
                text = text.strip() if text else text

                media_attachments = self._get_media_attachments(conn, rowid)

                if not text and not media_attachments:
                    new_max_rowid = max(new_max_rowid, rowid)
                    continue
                if not is_from_me and text and is_prompt_injection(text):
                    info(f"iMessage: blocked prompt injection from {handle_id}")
                    new_max_rowid = max(new_max_rowid, rowid)
                    continue

                sender_name = self._resolve_contact(handle_id) if not is_from_me else config.get_owner_name()
                group_name = chat_name.strip() if chat_name and chat_name.strip() else None

                photo_desc = ""
                if media_attachments:
                    path, mime_type = media_attachments[0]
                    if backfill:
                        photo_desc = "[Video]" if media_kind(mime_type) == KIND_VIDEO else "[Photo]"
                    else:
                        photo_desc = self._analyze_media_attachment(path, mime_type, sender_name)

                if photo_desc and text:
                    body = f"{photo_desc}\nText message: {text}"
                else:
                    body = photo_desc or text or ""
                if not body:
                    new_max_rowid = max(new_max_rowid, rowid)
                    continue

                if is_from_me:
                    content = self._format_sent_content(conn, rowid, group_name, body)
                else:
                    target = group_name or config.get_owner_name()
                    content = f"iMessage received:\nFrom {sender_name} to {target}: {body}"

                # Also during catch-up: a code requested at startup lands in chat.db before
                # the feed goes live, so its row is a backfill row. The age check alone
                # keeps the old codes in the history from being submitted.
                if not is_from_me and text:
                    apple_match = APPLE_2FA_PATTERN.search(text)
                    ring_match = RING_2FA_PATTERN.search(text)
                    if apple_match or ring_match:
                        code_age = time.time() - unix_ts
                        if code_age > TFA_CODE_MAX_AGE:
                            if backfill:
                                fine(f"iMessage: old 2FA code in the catch-up, not submitted (rowid={rowid})")
                            else:
                                info(f"iMessage: ignoring stale 2FA code (age={code_age:.0f}s > "
                                     f"{TFA_CODE_MAX_AGE}s, rowid={rowid})")
                        else:
                            if backfill:
                                info(f"iMessage: fresh 2FA code found during catch-up "
                                     f"(age={code_age:.0f}s, rowid={rowid})")
                            if apple_match:
                                if apple_code:
                                    info("iMessage: multiple Apple 2FA codes in one poll batch — keeping the newest")
                                if not apple_code or unix_ts >= apple_code[0]:
                                    apple_code = (unix_ts, apple_match.group(1))
                            if ring_match:
                                if ring_code:
                                    info("iMessage: multiple Ring 2FA codes in one poll batch — keeping the newest")
                                if not ring_code or unix_ts >= ring_code[0]:
                                    ring_code = (unix_ts, ring_match.group(1))

                if not backfill:
                    trace(content)
                entries.append(FeedEntry(unix_ts, content))
                new_max_rowid = max(new_max_rowid, rowid)

        self._last_seen_rowid = new_max_rowid

        self._pending_apple_code = self._settle_and_forward_2fa(
            self._pending_apple_code, apple_code, self._forward_2fa_code, "Apple")
        self._pending_ring_code = self._settle_and_forward_2fa(
            self._pending_ring_code, ring_code, self._forward_ring_2fa_code, "Ring")
        return entries

    def _format_sent_content(self, conn, rowid: int, group_name: Optional[str], body: str) -> str:
        target = group_name or self._resolve_chat_recipient(conn, rowid)
        if target:
            return f"iMessage sent:\n{config.get_owner_name()} to {target}: {body}"
        return f"iMessage sent:\n{config.get_owner_name()}: {body}"

    @staticmethod
    def _not_pending_reason(is_from_me, has_chat: bool, schedule_state, unix_ts: float,
                            now: float) -> Optional[str]:
        """Why a schedule_type=2 row is NOT a genuine pending send-later message, or
        None if it is. Messages in iCloud (ck_sync_state=1) writes incoming, chat-less
        copies of an already-sent scheduled record that keep schedule_type=2 forever
        because Apple's scheduler never touches them — tracking those means logging
        them as "past due" on every restart. Only the owner's own rows attached to a
        chat, neither completed nor failed, and not long past their send time count."""
        if not is_from_me:
            return "an incoming iCloud sync duplicate"
        if not has_chat:
            return "not attached to any chat"
        if schedule_state == SCHEDULE_STATE_COMPLETED:
            return "already completed"
        if schedule_state == SCHEDULE_STATE_FAILED:
            return "marked failed"
        if unix_ts < now - SCHEDULED_GRACE_PERIOD:
            return f"more than {SCHEDULED_GRACE_PERIOD // 86400} days past its send time"
        return None

    def _untrack_scheduled(self, rowid: int):
        self._scheduled_rowids.discard(rowid)
        self._past_due_noted.discard(rowid)

    def _check_scheduled_messages(self, conn) -> list[FeedEntry]:
        """Re-check tracked send-later messages: emit once they actually send, drop
        them if cancelled, failed or long past due. Apple resets schedule_type to 0
        when a scheduled message sends; a reschedule only moves m.date on the same
        row, so no event until then."""
        entries: list[FeedEntry] = []
        if not self._scheduled_rowids:
            return entries

        placeholders = ",".join("?" * len(self._scheduled_rowids))
        rows = conn.execute(f"""
            SELECT m.ROWID, m.text, m.attributedBody, m.date, m.schedule_type,
                   c.display_name AS chat_name, m.schedule_state
            FROM message m
            LEFT JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
            LEFT JOIN chat c ON cmj.chat_id = c.ROWID
            WHERE m.ROWID IN ({placeholders})
        """, tuple(self._scheduled_rowids)).fetchall()

        now = time.time()
        found = set()
        for rowid, text, attributed_body, apple_ts, schedule_type, chat_name, schedule_state in rows:
            found.add(rowid)
            unix_ts = apple_ts / 1_000_000_000 + APPLE_EPOCH_OFFSET
            if schedule_state == SCHEDULE_STATE_FAILED:
                info(f"iMessage: scheduled message failed to send, no longer tracking (rowid={rowid})")
                self._untrack_scheduled(rowid)
                continue
            if schedule_type == SCHEDULE_TYPE_SEND_LATER and schedule_state != SCHEDULE_STATE_COMPLETED:
                # Still pending — even past its send time (delivery delay, Mac offline
                # at fire time). Announcing it as sent here would be a false report;
                # keep tracking until Apple flips schedule_type to 0, but not forever.
                if unix_ts < now - SCHEDULED_GRACE_PERIOD:
                    info(f"iMessage: scheduled message still unsent {(now - unix_ts) / 86400:.0f} days "
                         f"past its send time — giving up on it (rowid={rowid})")
                    self._untrack_scheduled(rowid)
                elif unix_ts <= now and rowid not in self._past_due_noted:
                    self._past_due_noted.add(rowid)
                    info(f"iMessage: scheduled message past its send time but not yet "
                         f"sent — still waiting (rowid={rowid})")
                continue

            self._untrack_scheduled(rowid)
            if not text:
                text = self._extract_attributed_text(attributed_body)
            text = text.strip() if text else text
            if not text:
                trace(f"iMessage: scheduled message sent with empty body, skipping (rowid={rowid})")
                continue

            group_name = chat_name.strip() if chat_name and chat_name.strip() else None
            content = self._format_sent_content(conn, rowid, group_name, text)
            info(f"iMessage: scheduled message sent (rowid={rowid})")
            trace(content)
            entries.append(FeedEntry(unix_ts, content))

        for rowid in self._scheduled_rowids - found:
            info(f"iMessage: scheduled message cancelled by user (rowid={rowid})")
            self._untrack_scheduled(rowid)
        return entries

    @staticmethod
    def _settle_and_forward_2fa(pending, fresh, forward, service) -> Optional[Tuple[float, str]]:
        """2FA codes settle for one poll cycle before submission. When two codes are
        sent back-to-back (each new request invalidates the previous code), the stale
        one usually arrives first — waiting one cycle lets the newer code supersede it
        instead of burning a verification attempt on a dead code."""
        if fresh:
            if pending:
                info(f"iMessage: newer {service} 2FA code arrived while one was settling — keeping the newest")
                return fresh if fresh[0] >= pending[0] else pending
            trace(f"iMessage: {service} 2FA code detected, settling one poll cycle before submitting")
            return fresh
        if pending:
            forward(pending[1], pending[0])
        return None

    @staticmethod
    def _forward_2fa_code(code: str, received_at: float):
        """Forward an auto-detected Apple 2FA code to iCloud integration."""
        try:
            from integrations.icloud.ICloudIntegration import ICloudIntegration
            icloud = ICloudIntegration.get_instance()
            icloud.submit_auto_2fa_code(code, received_at)
        except Exception as e:
            info(f"iMessage: could not forward 2FA code to iCloud: {e}")

    @staticmethod
    def _forward_ring_2fa_code(code: str, received_at: float):
        """Forward an auto-detected Ring 2FA code to Ring integration."""
        try:
            from integrations.ring.RingIntegration import RingIntegration
            ring = RingIntegration.get_instance()
            ring.submit_auto_2fa_code(code, received_at)
        except Exception as e:
            info(f"iMessage: could not forward 2FA code to Ring: {e}")

    @staticmethod
    def _extract_attributed_text(blob: bytes) -> str | None:
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

    def _resolve_contact(self, handle_id: str) -> str:
        """Resolve a handle (phone/email) to a contact name."""
        if not handle_id:
            return "Unknown"

        normalized = self._normalize_phone(handle_id)
        if normalized in self._contact_cache:
            return self._contact_cache[normalized]

        if handle_id in self._contact_cache:
            return self._contact_cache[handle_id]

        suffix = self._phone_suffix(handle_id)
        if suffix and suffix in self._contact_cache:
            return self._contact_cache[suffix]

        return handle_id

    def _resolve_chat_recipient(self, conn, message_rowid: int) -> Optional[str]:
        """Resolve the recipient(s) of a sent message from the chat's participants.

        Apple stores message.handle_id = 0 for outgoing messages, so the recipient
        cannot be read from the message row — it lives on the chat. Returns a single
        contact name for a 1:1 chat, a comma-joined list for an unnamed group, or
        None when no participants can be found.
        """
        rows = conn.execute("""
            SELECT DISTINCT h.id
            FROM chat_message_join cmj
            JOIN chat_handle_join chj ON cmj.chat_id = chj.chat_id
            JOIN handle h ON chj.handle_id = h.ROWID
            WHERE cmj.message_id = ?
        """, (message_rowid,)).fetchall()

        handles = [r[0] for r in rows if r[0]]
        if not handles:
            fine(f"iMessage: no chat participants found for sent message (rowid={message_rowid})")
            return None

        names = sorted(self._resolve_contact(h) for h in handles)
        return ", ".join(names)

    @staticmethod
    def _normalize_phone(phone: str) -> str:
        """Normalize phone number: keep leading + and digits only."""
        if not phone:
            return phone
        digits = re.sub(r'[^\d+]', '', phone)
        return digits

    @staticmethod
    def _phone_suffix(phone: str) -> str | None:
        """Extract last 9 digits for matching local (0xxx) vs international (+46xxx) formats."""
        if not phone:
            return None
        digits = re.sub(r'\D', '', phone)
        return digits[-9:] if len(digits) >= 7 else None

    @staticmethod
    def _get_media_attachments(conn, message_rowid: int) -> list[tuple[str, str]]:
        """Return (local file path, mime type) of the image and video attachments of a message."""
        rows = conn.execute("""
            SELECT a.filename, a.mime_type
            FROM attachment a
            JOIN message_attachment_join maj ON a.ROWID = maj.attachment_id
            WHERE maj.message_id = ?
              AND a.filename IS NOT NULL
        """, (message_rowid,)).fetchall()

        attachments = []
        for filename, mime_type in rows:
            if filename.endswith(".pluginPayloadAttachment"):
                continue  # iMessage app payloads (link previews, stickers, Apple Pay…) — never media
            if media_kind(mime_type) is None:
                trace(f"iMessage: skipping attachment of type {mime_type!r} ({filename})")
                continue
            expanded = filename.replace("~", str(Path.home()), 1) if filename.startswith("~") else filename
            if os.path.isfile(expanded):
                attachments.append((expanded, mime_type))
            else:
                fine(f"iMessage: attachment file not found: {expanded}")
        return attachments

    @staticmethod
    def _analyze_media_attachment(path: str, mime_type: str, sender_name: str) -> str:
        label = "Video" if media_kind(mime_type) == KIND_VIDEO else "Photo"
        try:
            with open(path, "rb") as f:
                raw_bytes = f.read()
        except OSError as e:
            error(f"iMessage: could not read {label.lower()} {path}: {e}")
            return f"[{label}]"
        return media_analyzer().describe_bytes(raw_bytes, mime_type=mime_type, filename=path,
                                               context=f"{label} sent via iMessage by {sender_name}")

    @classmethod
    def _add_candidate(cls, candidates: dict[str, list[tuple[str, Optional[str]]]], name: str,
                       raw_handle: str, label: Optional[str] = None):
        """Add a handle (with its AddressBook label) to the candidates of a name. The key is the
        lowercased, whitespace-collapsed name — callers pass first name, full name, etc. as
        separate calls; `_rank_candidates` picks one handle per name once the cards are read."""
        if not name or not raw_handle:
            return
        key = " ".join(name.split()).lower()
        handle = cls._prefer_handle(raw_handle)
        if not key or not handle:
            return
        entries = candidates.setdefault(key, [])
        if all(h != handle for h, _ in entries):
            entries.append((handle, label))

    @classmethod
    def _record_full_name(cls, mapping: dict[str, str], first: str | None, last: str | None, raw_handle: str):
        """Map handle → 'First Last' (or just 'First' if no last). Used to render canonical names
        in approval dialogs even when the caller passed only a first name."""
        if not first or not raw_handle:
            return
        handle = cls._prefer_handle(raw_handle)
        if not handle:
            return
        f = first.strip()
        l = (last or "").strip()
        if not f:
            return
        full = f"{f} {l}" if l else f
        mapping.setdefault(handle, full)

    def display_name_for_handle(self, handle: str) -> Optional[str]:
        """Canonical 'First Last' for a resolved handle, or None if unknown."""
        if not handle:
            return None
        return self._handle_to_full_name.get(handle)

    @staticmethod
    def _name_keys(first: str | None, last: str | None) -> list[str]:
        """Lookup keys to register for a contact: first name, and 'first last' when last is present.
        Last-name-only is intentionally omitted — too easy to collide across contacts."""
        keys: list[str] = []
        f = (first or "").strip()
        l = (last or "").strip()
        if f:
            keys.append(f)
        if f and l:
            keys.append(f"{f} {l}")
        return keys

    @staticmethod
    def _prefer_handle(raw_handle: str) -> Optional[str]:
        """Return a normalized handle suitable for AppleScript (E.164 phone or lowercase email)."""
        if not raw_handle:
            return None
        if "@" in raw_handle:
            return raw_handle.strip().lower()
        digits = re.sub(r'[^\d+]', '', raw_handle)
        return digits or None

    @staticmethod
    def _handle_priority(handle: str) -> int:
        """Lower is better. Prefer E.164 phones, then emails, then bare digits."""
        if not handle:
            return 99
        if handle.startswith("+"):
            return 0
        if "@" in handle:
            return 1
        return 2

    @staticmethod
    def _handle_tier(handle: str, label: Optional[str], activity: Optional[tuple[bool, int, int]]) -> int:
        """The preference tier of a handle (`TIER_*`, lower is better) given its AddressBook label
        and what chat.db knows about it."""
        if activity and activity[0]:
            return TIER_IMESSAGE
        if "@" in handle:
            return TIER_UNKNOWN_EMAIL
        if label and FAX_LABEL_PATTERN.search(label):
            return TIER_FAX
        if (label and MOBILE_LABEL_PATTERN.search(label)) or SWEDISH_MOBILE_PATTERN.match(handle):
            return TIER_MOBILE
        return TIER_OTHER_PHONE

    @classmethod
    def _chat_activity(cls) -> dict[str, tuple[bool, int, int]]:
        """What chat.db knows per handle: (seen on the iMessage service, Apple date of the latest
        message, message count), keyed by the handle id and, for phones, by its 9-digit suffix so a
        card's local-format number matches the E.164 handle Messages stores. Empty when chat.db
        cannot be read — the ranking then falls back to the card alone."""
        activity: dict[str, tuple[bool, int, int]] = {}

        def merge(key, imessage, last_date, count):
            seen = activity.get(key, (False, 0, 0))
            activity[key] = (seen[0] or imessage, max(seen[1], last_date), seen[2] + count)

        try:
            with sqlite3.connect(f"file:{CHAT_DB_PATH}?mode=ro", uri=True) as conn:
                rows = conn.execute(
                    "SELECT h.id, h.service, COUNT(m.ROWID), COALESCE(MAX(m.date), 0) "
                    "FROM handle h LEFT JOIN message m ON m.handle_id = h.ROWID "
                    "GROUP BY h.id, h.service"
                ).fetchall()
        except Exception as e:
            info(f"iMessage: chat.db handle query failed, ranking contact handles by card only: {e}")
            return activity
        for handle_id, service, count, last_date in rows:
            if not handle_id:
                continue
            imessage = (service or "").lower() == "imessage"
            merge(handle_id.lower(), imessage, last_date or 0, count or 0)
            suffix = cls._phone_suffix(handle_id) if "@" not in handle_id else None
            if suffix:
                merge(suffix, imessage, last_date or 0, count or 0)
        trace(f"iMessage: chat.db knows {len(rows)} handles")
        return activity

    @classmethod
    def _lookup_activity(cls, activity: dict[str, tuple[bool, int, int]], handle: str) \
            -> Optional[tuple[bool, int, int]]:
        found = activity.get(handle.lower())
        if found is None and "@" not in handle:
            suffix = cls._phone_suffix(handle)
            found = activity.get(suffix) if suffix else None
        return found

    @classmethod
    def _rank_candidates(cls, candidates: dict[str, list[tuple[str, Optional[str]]]]) \
            -> tuple[dict[str, str], dict[str, str]]:
        """Pick one handle per name: the best tier, then the most recent conversation in chat.db
        (then the most messages), then E.164 over an email over bare digits, then the card's
        order. Returns the name → handle map and, per name, why the handle won — the line
        `resolve_recipient` logs. chat.db is read only when some name has a choice to make."""
        reverse: dict[str, str] = {}
        reasons: dict[str, str] = {}
        contested = {key: entries for key, entries in candidates.items() if len(entries) > 1}
        activity = cls._chat_activity() if contested else {}
        for key, entries in candidates.items():
            ranked = []
            for index, (handle, label) in enumerate(entries):
                seen = cls._lookup_activity(activity, handle)
                tier = cls._handle_tier(handle, label, seen)
                last_date, count = (seen[1], seen[2]) if seen else (0, 0)
                ranked.append(((tier, -last_date, -count, cls._handle_priority(handle), index), handle))
            (tier, neg_date, neg_count, _, index), handle = min(ranked)
            reverse[key] = handle
            history = f", {-neg_count} messages" if neg_count else ", no chat history"
            reasons[key] = f"{TIER_NAMES[tier]}{history}, listed {index + 1}/{len(entries)} on the card"
        if contested:
            overridden = [f"{key}→{reverse[key]} ({reasons[key]})" for key, entries in contested.items()
                          if reverse[key] != entries[0][0]]
            shown = "; ".join(overridden[:20]) + (" …" if len(overridden) > 20 else "")
            fine(f"iMessage: {len(contested)} contacts with several handles, {len(overridden)} resolved to "
                 f"one other than the card's first" + (f": {shown}" if overridden else ""))
        return reverse, reasons

    @staticmethod
    def _looks_like_handle(s: str) -> bool:
        """Heuristic: does this look like a phone number or email rather than a contact name?"""
        if not s:
            return False
        if "@" in s:
            return True
        digits = re.sub(r'\D', '', s)
        return len(digits) >= 7

    def resolve_recipient(self, recipient: str) -> Tuple[Optional[str], Optional[str]]:
        """Resolve a recipient string (contact name or raw handle) to a sendable handle.
        Returns (handle, error). On success error is None; on failure handle is None
        and error is a short human-readable string."""
        if not recipient or not recipient.strip():
            return None, "empty recipient"
        candidate = recipient.strip()

        if self._looks_like_handle(candidate):
            normalized = self._prefer_handle(candidate)
            return (normalized or candidate), None

        if time.time() - self._contact_cache_time > CONTACT_CACHE_REFRESH_INTERVAL:
            self._refresh_contact_cache()

        key = " ".join(candidate.split()).lower()
        handle = self._name_to_handle.get(key)
        if handle:
            info(f"iMessage: resolved '{candidate}' → {handle} ({self._handle_reasons.get(key, 'only handle')})")
            return handle, None

        fallback = self._fallback_lookup(key)
        if fallback:
            info(f"iMessage: resolved '{candidate}' via forward-cache fallback → {fallback}")
            return fallback, None

        if not self._contact_cache and not self._name_to_handle:
            self._notify_contact_cache_empty()
            return None, (
                f"contact cache is empty — could not look up '{candidate}'. "
                f"Reading the AddressBook likely {FULL_DISK_ACCESS_HINT}."
            )
        return None, f"unknown contact: {candidate} (cache has {len(self._name_to_handle)} names)"

    def _fallback_lookup(self, name_lower: str) -> Optional[str]:
        """Last-resort: scan the forward (handle → name) cache for any handle whose name matches
        and rank them like the card's handles (without labels — the forward cache has none).
        Useful when the reverse cache missed for any reason."""
        if not self._contact_cache:
            return None
        candidates: dict[str, list[tuple[str, Optional[str]]]] = {}
        for raw_handle, contact_name in self._contact_cache.items():
            if contact_name and contact_name.strip().lower() == name_lower:
                self._add_candidate(candidates, name_lower, raw_handle)
        if not candidates:
            return None
        reverse, reasons = self._rank_candidates(candidates)
        trace(f"iMessage: fallback lookup of '{name_lower}' → {reverse[name_lower]} ({reasons[name_lower]})")
        return reverse[name_lower]

    @staticmethod
    def _applescript_quote(s: str) -> str:
        """Escape a string for safe interpolation inside AppleScript double quotes,
        breaking newlines into concatenated `& linefeed &` segments."""
        if s is None:
            return '""'
        parts = s.split("\n")
        escaped = [p.replace("\\", "\\\\").replace('"', '\\"') for p in parts]
        if len(escaped) == 1:
            return f'"{escaped[0]}"'
        return " & linefeed & ".join(f'"{p}"' for p in escaped)

    @classmethod
    def _build_send_applescript(cls, handle: str, body: str) -> str:
        """Build AppleScript that tries iMessage first then falls back to SMS."""
        target = cls._applescript_quote(handle)
        message = cls._applescript_quote(body)
        return f'''
            tell application "Messages"
                set targetBuddy to missing value
                try
                    set iService to 1st service whose service type = iMessage
                    set targetBuddy to buddy {target} of iService
                    send {message} to targetBuddy
                    return "{APPLESCRIPT_RESULT_IMESSAGE}"
                on error
                    try
                        set smsService to 1st service whose service type = SMS
                        set targetBuddy to buddy {target} of smsService
                        send {message} to targetBuddy
                        return "{APPLESCRIPT_RESULT_SMS}"
                    on error errMsg2 number errNum2
                        error errMsg2 number errNum2
                    end try
                end try
            end tell
        '''

    def send(self, recipient: str, body: str) -> str:
        """Send an iMessage/SMS via Messages.app AppleScript.
        Returns "" on success or a short error string on failure."""
        if not body or not body.strip():
            info("iMessage send skipped: empty body")
            return "empty message body"

        rate_err = self._rate_limiter.check()
        if rate_err:
            info(f"iMessage send blocked: {rate_err}")
            return rate_err

        handle, err = self.resolve_recipient(recipient)
        if err:
            info(f"iMessage send: could not resolve recipient '{recipient}': {err}")
            return err

        script = self._build_send_applescript(handle, body)
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=APPLESCRIPT_TIMEOUT
            )
        except subprocess.TimeoutExpired:
            error(f"iMessage send to {handle} timed out after {APPLESCRIPT_TIMEOUT}s")
            return "timeout sending iMessage"
        except Exception as e:
            error(f"iMessage send to {handle} raised: {e}")
            return f"{e}"

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if result.returncode != 0:
            if "-1743" in stderr or "not authorized" in stderr.lower():
                error(f"iMessage send unauthorized: grant Automation permission to control Messages. stderr={stderr}")
                self._notify_automation_permission_needed()
                return "not authorized to control Messages (System Settings → Privacy & Security → Automation)"
            error(f"iMessage send to {handle} failed: rc={result.returncode} stderr={stderr}")
            return stderr or f"osascript exit {result.returncode}"

        self._rate_limiter.record()
        service = stdout if stdout in (APPLESCRIPT_RESULT_IMESSAGE, APPLESCRIPT_RESULT_SMS) else "sent"
        info(f"iMessage {service} to {handle}: {body!r}")
        self._log_sent_to_history(recipient, handle, body)
        return ""

    def _notify_contact_cache_empty(self):
        if self._notified_cache_empty:
            return
        self._notified_cache_empty = True
        system_note(self.name,
            "iMessage: contact cache is empty — name-based recipients can't be resolved. "
            f"Likely cause: reading the AddressBook {FULL_DISK_ACCESS_HINT}. "
            "Sends to raw phone numbers / emails still work."
        )

    @classmethod
    def _notify_automation_permission_needed(cls):
        system_note("imessage",
            "iMessage send blocked: Keep is not authorized to control Messages. "
            "Grant access in System Settings → Privacy & Security → Automation, then retry."
        )

    def _log_sent_to_history(self, recipient: str, handle: str, body: str):
        """Mirror the read-path's `iMessage sent:` shape so the LLM sees its own outbound
        message; the entry rides along with the feed's next batch."""
        target_name = self._resolve_contact(handle) if handle != recipient else recipient
        if not target_name or target_name == handle:
            target_name = recipient
        content = f"iMessage sent:\n{config.get_owner_name()} to {target_name}: {body}"
        self._pending_entries.append(FeedEntry(time.time(), content))

    def _refresh_contact_cache(self):
        """Build phone/email -> name cache by reading the macOS AddressBook SQLite databases directly."""
        try:
            ab_root = Path.home() / "Library" / "Application Support" / "AddressBook"
            source_dbs = list((ab_root / "Sources").glob("*/AddressBook-v22.abcddb"))
            if not source_dbs:
                trace("iMessage: no AddressBook source databases found")
                self._contact_cache_time = time.time()
                return

            cache: dict[str, str] = {}
            candidates: dict[str, list[tuple[str, Optional[str]]]] = {}
            handle_to_full: dict[str, str] = {}
            for db_path in source_dbs:
                try:
                    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
                        for first, last, phone, label in conn.execute("""
                            SELECT r.ZFIRSTNAME, r.ZLASTNAME, p.ZFULLNUMBER, p.ZLABEL
                            FROM ZABCDRECORD r JOIN ZABCDPHONENUMBER p ON p.ZOWNER = r.Z_PK
                            WHERE r.ZFIRSTNAME IS NOT NULL AND p.ZFULLNUMBER IS NOT NULL
                            ORDER BY r.Z_PK, p.Z_PK
                        """):
                            normalized = self._normalize_phone(phone)
                            if normalized:
                                cache[normalized] = first
                            suffix = self._phone_suffix(phone)
                            if suffix:
                                cache[suffix] = first
                            for name_key in self._name_keys(first, last):
                                self._add_candidate(candidates, name_key, phone, label)
                            self._record_full_name(handle_to_full, first, last, phone)

                        for first, last, email in conn.execute("""
                            SELECT r.ZFIRSTNAME, r.ZLASTNAME, e.ZADDRESS
                            FROM ZABCDRECORD r JOIN ZABCDEMAILADDRESS e ON e.ZOWNER = r.Z_PK
                            WHERE r.ZFIRSTNAME IS NOT NULL AND e.ZADDRESS IS NOT NULL
                        """):
                            cache[email.lower()] = first
                            for name_key in self._name_keys(first, last):
                                self._add_candidate(candidates, name_key, email)
                            self._record_full_name(handle_to_full, first, last, email)
                except Exception as e:
                    fine(f"iMessage: skipping contact source {db_path.parent.name}: {e}")

            reverse, reasons = self._rank_candidates(candidates)
            self._contact_cache = cache
            self._name_to_handle = reverse
            self._handle_reasons = reasons
            self._handle_to_full_name = handle_to_full
            self._contact_cache_time = time.time()
            if cache:
                self._notified_cache_empty = False
            fine(f"iMessage: refreshed contact cache ({len(cache)} entries, {len(reverse)} reverse)")
        except Exception as e:
            trace(f"iMessage: failed to refresh contact cache: {e}")
            self._contact_cache_time = time.time()
