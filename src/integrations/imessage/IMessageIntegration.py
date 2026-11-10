import re
import time
from typing import List, Optional, Tuple
from assistant.SystemNote import system_note

from assistant.integration import SatelliteHints
from assistant.integration.Feed import Feed, FeedBatch, FeedEntry, FeedSourceAway
from assistant.integration.SmsCodeReader import SmsCodeReader
from assistant.integration.Integration import HealthWarning, Integration
from common.ConfigManager import config
from common.imessage import Hosts
from common.imessage.Handles import is_group_guid
from common.imessage.MessagesHost import (Attachment, GroupChat, HostHealth, Message, MessageBatch, MessagesHost,
                                          NO_ACCESS, NO_DATABASE, Recipient, SCHEDULE_COMPLETED, SCHEDULE_FAILED,
                                          UNREACHABLE)
from common.logging import info, trace, error, fine
from assistant.PromptInjectionDetector import is_prompt_injection
from assistant.tools.MediaAnalyzer import media_analyzer, media_kind, KIND_VIDEO


CONFIG_SECTION = "imessage"     # holds `host` and `host_token`: whose Messages this integration reads (common/imessage/Hosts)
POLL_INTERVAL = 5
FETCH_BATCH_SIZE = 500   # chat.db rows read per fetch; a shorter batch means the catch-up is over
APPLE_2FA_PATTERN = re.compile(r"Apple Account Code is:\s*(\d{6})")
RING_2FA_PATTERN = re.compile(r"Your Ring verification code is:\s*(\d{6})")
TFA_CODE_MAX_AGE = 120  # only auto-submit 2FA codes from messages received within this many seconds

# How often a file is asked for again because the host went away over it (a satellite relaunching
# between the poll and the read, a video too slow to cross) before its row is stored without it.
ATTACHMENT_TRIES = 3

# a scheduled message still unsent this long after its send time is never going to send
# from here (Send Later fires from whichever device is online) — stop tracking it
SCHEDULED_GRACE_PERIOD = 7 * 24 * 3600


class IMessageIntegration(Integration, Feed, SmsCodeReader):
    """The iMessages and SMS of the server's Mac, read as a feed and sent.

    Brings the messages that arrive on the server's Mac into Keep's history, as a feed, and
    sends an iMessage or SMS to a contact when asked. It also catches the 2FA codes Apple and
    Ring text.

    Everything that touches Messages.app goes through `self.host`, a `MessagesHost`
    (`common/imessage/` — chat.db, the AddressBook, the AppleScript sends); what is here is the
    assistant's side of it. `imessage.host` picks the host (`Hosts.build`): this Mac's own
    Messages, or a satellite on another Mac or macOS user — whose problems `SatelliteHints`
    puts into words, since the grants are then needed there and not here. As a Feed
    (plans/FEED_PLAN.md) `fetch` turns the rows the host returns after the cursor (a chat.db
    ROWID) into history entries; FeedRunner owns the thread, the cursor and the catch-up — the
    ROWID is one chat.db's, so the cursor is kept with its account (`cursor_source`) and given
    up when `imessage.host` names another or the database ends before it (`cursor_lost`). It
    screens incoming texts for prompt injection, has media described, forwards the detected
    Apple/Ring 2FA codes to their integrations, follows send-later messages until they send
    and echoes its own sends into the history."""
    sensitive = True

    poll_interval = POLL_INTERVAL
    item_noun = "messages"
    default_backfill_months = 60    # five years: chat.db rows are cheap, no LLM call per message
    cursor_generation = 1           # 1: a ROWID means "read" only for rows dated in the horizon

    def __init__(self):
        super().__init__()
        self.host: MessagesHost = Hosts.build(CONFIG_SECTION)
        self.cursor_source = Hosts.account(CONFIG_SECTION)      # whose chat.db the ROWIDs are of, as the host was built
        self._last_seen_rowid = 0
        self._exhausted = True          # the last poll read fewer rows than it asked for
        self._open_error: Optional[str] = None
        # history rows produced outside fetch (a sent message) — handed over with the next batch
        self._pending_entries: list[FeedEntry] = []
        # what the host resolved, by handle — the approval card and the history echo read it
        self._recipients: dict[str, Recipient] = {}
        self._notified_cache_empty = False
        # 2FA codes detected but held for one poll cycle — see _settle_and_forward_2fa
        self._pending_apple_code: Optional[Tuple[float, str]] = None
        self._pending_ring_code: Optional[Tuple[float, str]] = None
        # send-later messages composed but not yet sent — see _check_scheduled_messages
        self._scheduled_rowids: set[int] = set()
        # rowids already logged as past-due-but-unsent, so the wait is noted once
        self._past_due_noted: set[int] = set()
        self._rewind_noted = False      # a host answered a poll with a cursor behind the one it was asked from
        self._attachment_tries: dict[int, int] = {}     # attachment id → reads that found the host away
        # attachment id → its description, for the batch being read: a batch read again because
        # the host went away over a later file costs no second vision call for the earlier ones
        self._described: dict[int, str] = {}

    def _on_reset(self):
        from assistant.integration.FeedRunner import Feeds
        Feeds.get_instance().stop(self.feed_name)

    @property
    def name(self) -> str:
        return "iMessage"

    @property
    def config_keys(self) -> List[str]:
        """Nothing for this Mac's own Messages; the token when `imessage.host` names a satellite."""
        return [f"{CONFIG_SECTION}.{Hosts.TOKEN_KEY}"] if Hosts.is_remote(CONFIG_SECTION) else []

    def reads_sms_codes(self) -> bool:
        """The feed is the reader: live while its runner polls (or is still opening), dead when
        it failed, was stopped or never started — and while its source is away: a satellite
        whose Mac sleeps reads no code, however long the runner waits for it."""
        from assistant.integration.FeedRunner import Feeds, FeedState
        runner = Feeds.get_instance().runner(self.feed_name)
        if runner is None or runner.state not in (FeedState.STARTING, FeedState.CATCHING_UP, FeedState.LIVE):
            return False
        return not runner.away

    def _full_disk_access_hint(self) -> str:
        """The grant is needed where Messages is: on this Mac, or on the satellite's."""
        return SatelliteHints.full_disk_access(self.host.address)

    def _health_error(self, health: HostHealth) -> Optional[str]:
        """What an unhealthy host means to the user: no Messages on the Mac, or — for every
        other refusal of chat.db — the Full Disk Access grant, the action first. For a
        satellite, also that it cannot be reached or does not take the token."""
        if health.contacts:
            self._notified_cache_empty = False
        if health.ok:
            return None
        if self.host.address:
            return SatelliteHints.problem(self.host.address, health, CONFIG_SECTION, self.name)
        if health.reason == NO_DATABASE:
            return health.detail
        return f"{self._full_disk_access_hint()} ({health.detail})"

    def check_health(self) -> bool:
        health = self.host.health()
        problem = self._health_error(health)
        if problem:
            raise (PermissionError if health.reason == NO_ACCESS else RuntimeError)(problem)
        warning = SatelliteHints.warning(self.host.address, health.satellite) if health.satellite else ""
        if warning:
            raise HealthWarning(warning)
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
        health = self.host.health(refresh=True)
        self._open_error = self._health_error(health)
        if self._open_error and (health.reason == UNREACHABLE or self.host.away()):
            # a satellite that sleeps, restarts or is updating itself to a wire version this side
            # speaks: the runner asks again, the messages wait in its chat.db
            raise FeedSourceAway(self._open_error)
        if self._open_error:
            error(f"iMessage: {self._open_error}")
            return False
        self._seed_scheduled_messages()
        info(f"iMessage feed opened (contact cache: {health.contacts} entries)")
        return True

    def _checked(self, answer):
        """A host's answer, or its error raised for the runner to report and retry — as
        `FeedSourceAway` when the host could not be reached at all, which the runner waits out."""
        if answer.error:
            raise (FeedSourceAway if self.host.away() else RuntimeError)(answer.error)
        return answer

    def initial_cursor(self, horizon_ts: float) -> Optional[str]:
        """The ROWID just before the first message dated at or after the horizon (past every
        row when there is none), so the catch-up starts right there — see
        `MessagesDatabase.cursor_at` for why it is not the last ROWID from before the horizon."""
        self.horizon_ts = horizon_ts
        rowid = self._checked(self.host.cursor_at(horizon_ts)).value
        fine(f"iMessage: initial ROWID {rowid} (just before the first message dated "
             f"{time.strftime('%Y-%m-%d', time.localtime(horizon_ts))} or later)")
        return str(rowid)

    def cursor_lost(self, cursor: Optional[str], source: Optional[str]) -> Optional[str]:
        """A ROWID is a place in one chat.db. In another account's it skips whatever lies before
        it there, and past that database's end the feed would be live and read nothing — no
        messages, no 2FA codes — until the ROWIDs there have grown past it. A row stored before
        the account was kept is this Mac's own: there was no other."""
        stored = source or Hosts.LOCAL
        if stored != self.cursor_source:
            return (f"is a ROWID of the Messages account '{stored}', and {CONFIG_SECTION}.{Hosts.HOST_KEY} "
                    f"now names '{self.cursor_source}'")
        # asked as the table's counter, never by a date: one row dated past any date asked for
        # would end the table before it
        last = self._checked(self.host.last_rowid()).value
        if int(cursor or 0) > last:
            return f"is past the last row of chat.db (ROWID {last}): another database than the one it was written for"
        return None

    def remaining(self, cursor: Optional[str]) -> Optional[int]:
        return self._checked(self.host.remaining(int(cursor or 0), self.horizon_ts)).value

    def fetch(self, cursor: Optional[str], backfill: bool) -> FeedBatch:
        self._last_seen_rowid = int(cursor or 0)
        entries = self._poll_new_messages(backfill)
        if self._pending_entries:
            entries = self._pending_entries + entries
            self._pending_entries = []
        return FeedBatch(entries, cursor=str(self._last_seen_rowid), exhausted=self._exhausted)

    def _seed_scheduled_messages(self):
        """Track the send-later messages composed before this start, so their eventual send
        is reported once (see _check_scheduled_messages)."""
        batch = self.host.scheduled()
        if batch.error:
            error(f"iMessage: failed to read scheduled messages: {batch.error}")
            return
        now = time.time()
        pending, ignored = [], []
        for message in batch.messages:
            reason = self._not_pending_reason(message.is_from_me, message.in_chat, message.schedule_state,
                                              message.timestamp, now)
            (ignored if reason else pending).append((message.rowid, reason))
        if ignored:
            fine(f"iMessage: ignoring {len(ignored)} schedule_type=2 row(s) that are not pending "
                 f"send-later messages: {', '.join(f'{r} ({why})' for r, why in ignored)}")
        if pending:
            self._scheduled_rowids.update(r for r, _ in pending)
            trace(f"iMessage: tracking {len(pending)} pending send-later message(s) from before "
                  f"startup (rowids={[r for r, _ in pending]})")
        else:
            fine("iMessage: no pending send-later messages")

    def _poll_new_messages(self, backfill: bool = False) -> list[FeedEntry]:
        """The history entries for the messages the host has after `_last_seen_rowid` within
        the horizon (at most FETCH_BATCH_SIZE rows read); advances `_last_seen_rowid` past every
        row the host read. With `backfill` (catching up on months of history) nothing reacts —
        no send-later tracking, media as placeholders instead of vision-LLM descriptions —
        except that a 2FA code still fresh enough to submit is forwarded."""
        reported = [] if backfill else self._check_scheduled_messages()
        try:
            batch = self._checked(self.host.poll(self._last_seen_rowid, self.horizon_ts, FETCH_BATCH_SIZE))
            return reported + self._entries_of(batch, backfill)
        except Exception:
            # the sends just reported are no longer tracked: they ride with the next batch that goes through
            self._pending_entries.extend(reported)
            raise

    def _entries_of(self, batch: MessageBatch, backfill: bool) -> list[FeedEntry]:
        """The history entries of one polled batch; moves `_last_seen_rowid` past it. Raises
        `FeedSourceAway` — nothing moved, the batch is read again — when the host went away over
        one of its files."""
        entries: list[FeedEntry] = []
        # (received_at, code) per service — newest wins; forwarded after the batch so an
        # older code in the same batch can never be submitted ahead of a newer one
        apple_code: Optional[Tuple[float, str]] = None
        ring_code: Optional[Tuple[float, str]] = None
        for message in batch.messages:
            rowid, unix_ts, text = message.rowid, message.timestamp, message.text

            if message.send_later:
                self._note_send_later(message, backfill)
                continue

            media_attachments = self._media_attachments(message)
            if not text and not media_attachments:
                continue
            if not message.is_from_me and text and is_prompt_injection(text):
                info(f"iMessage: blocked prompt injection from {message.handle}")
                continue

            sender_name = message.sender if not message.is_from_me else config.get_owner_name()

            photo_desc = ""
            if media_attachments:
                attachment = media_attachments[0]
                if backfill:
                    photo_desc = "[Video]" if media_kind(attachment.mime_type) == KIND_VIDEO else "[Photo]"
                else:
                    photo_desc = self._analyze_media_attachment(attachment, sender_name)

            if photo_desc and text:
                body = f"{photo_desc}\nText message: {text}"
            else:
                body = photo_desc or text or ""
            if not body:
                continue

            if message.is_from_me:
                content = self._format_sent_content(message, body)
            else:
                target = message.chat_name or config.get_owner_name()
                content = f"iMessage received:\nFrom {sender_name} to {target}: {body}"

            # Also during catch-up: a code requested at startup lands in chat.db before
            # the feed goes live, so its row is a backfill row. The age check alone
            # keeps the old codes in the history from being submitted.
            if not message.is_from_me and text:
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

        self._described.clear()     # the batch went through: nothing of it is read again
        if batch.cursor < self._last_seen_rowid:
            # an answer without a cursor, or with an older one: following it would read history again
            if not self._rewind_noted:
                error(f"iMessage: the host answered the poll after ROWID {self._last_seen_rowid} with the cursor "
                      f"{batch.cursor} — the place is kept, a cursor never moves back")
            self._rewind_noted = True
        else:
            self._rewind_noted = False
            self._last_seen_rowid = batch.cursor
        self._exhausted = batch.exhausted

        self._pending_apple_code = self._settle_and_forward_2fa(
            self._pending_apple_code, apple_code, self._forward_2fa_code, "Apple")
        self._pending_ring_code = self._settle_and_forward_2fa(
            self._pending_ring_code, ring_code, self._forward_ring_2fa_code, "Ring")
        return entries

    def _note_send_later(self, message: Message, backfill: bool):
        """A send-later message among the new rows: chat.db gets the row at compose time with
        the send time as its date, and Apple un-marks it only when it ACTUALLY sends — so the
        mark, not the timestamp, is the truth. A past-due row still marked (delivery delay, Mac
        offline at fire time) is unsent and must be tracked, not announced. Reporting is left to
        _check_scheduled_messages, the sole emitter."""
        rowid = message.rowid
        reason = self._not_pending_reason(message.is_from_me, message.in_chat, message.schedule_state,
                                          message.timestamp, time.time())
        if reason:
            fine(f"iMessage: ignoring schedule_type=2 row that is {reason}, "
                 f"not a pending send-later message (rowid={rowid})")
        elif backfill:
            fine(f"iMessage: send-later message seen during catch-up, not tracked (rowid={rowid})")
        elif rowid not in self._scheduled_rowids:
            sends_at = time.strftime('%Y-%m-%d %H:%M', time.localtime(message.timestamp))
            info(f"iMessage: send-later message detected (rowid={rowid}, sends {sends_at}), deferring until sent")
            self._scheduled_rowids.add(rowid)

    @staticmethod
    def _format_sent_content(message: Message, body: str) -> str:
        target = message.chat_name or ", ".join(message.recipients)
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
        if schedule_state == SCHEDULE_COMPLETED:
            return "already completed"
        if schedule_state == SCHEDULE_FAILED:
            return "marked failed"
        if unix_ts < now - SCHEDULED_GRACE_PERIOD:
            return f"more than {SCHEDULED_GRACE_PERIOD // 86400} days past its send time"
        return None

    def _untrack_scheduled(self, rowid: int):
        self._scheduled_rowids.discard(rowid)
        self._past_due_noted.discard(rowid)

    def _check_scheduled_messages(self) -> list[FeedEntry]:
        """Re-check tracked send-later messages: emit once they actually send, drop
        them if cancelled, failed or long past due. Apple un-marks a scheduled message
        when it sends; a reschedule only moves the date on the same row, so no event
        until then."""
        entries: list[FeedEntry] = []
        if not self._scheduled_rowids:
            return entries

        batch = self._checked(self.host.messages(sorted(self._scheduled_rowids)))
        now = time.time()
        found = set()
        for message in batch.messages:
            rowid, unix_ts = message.rowid, message.timestamp
            found.add(rowid)
            if message.schedule_state == SCHEDULE_FAILED:
                info(f"iMessage: scheduled message failed to send, no longer tracking (rowid={rowid})")
                self._untrack_scheduled(rowid)
                continue
            if message.send_later and message.schedule_state != SCHEDULE_COMPLETED:
                # Still pending — even past its send time (delivery delay, Mac offline
                # at fire time). Announcing it as sent here would be a false report;
                # keep tracking until Apple un-marks it, but not forever.
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
            if not message.text:
                trace(f"iMessage: scheduled message sent with empty body, skipping (rowid={rowid})")
                continue

            content = self._format_sent_content(message, message.text)
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
    def _media_attachments(message: Message) -> list[Attachment]:
        """The image and video attachments of a message."""
        media = []
        for attachment in message.attachments:
            if media_kind(attachment.mime_type) is None:
                trace(f"iMessage: skipping attachment of type {attachment.mime_type!r} ({attachment.name})")
                continue
            media.append(attachment)
        return media

    def _analyze_media_attachment(self, attachment: Attachment, sender_name: str) -> str:
        label = "Video" if media_kind(attachment.mime_type) == KIND_VIDEO else "Photo"
        if attachment.id in self._described:
            trace(f"iMessage: {label.lower()} {attachment.name} was described before the host went away — "
                  f"not described again")
            return self._described[attachment.id]
        raw_bytes, reason = self.host.read_attachment(attachment.id)
        if raw_bytes is None and self.host.away():
            tries = self._attachment_tries[attachment.id] = self._attachment_tries.get(attachment.id, 0) + 1
            if tries < ATTACHMENT_TRIES:
                # the row is not stored as a bare placeholder for good: the runner keeps the cursor and asks again
                info(f"iMessage: the host went away over {label.lower()} {attachment.name} (try {tries} of "
                     f"{ATTACHMENT_TRIES}) — its message is read again when the host is back: {reason}")
                raise FeedSourceAway(reason)
            info(f"iMessage: {label.lower()} {attachment.name} did not cross in {ATTACHMENT_TRIES} tries — "
                 f"its message is stored without the description")
        self._attachment_tries.pop(attachment.id, None)
        if raw_bytes is None:
            error(f"iMessage: could not read {label.lower()} {attachment.name}: {reason}")
            return f"[{label}]"
        description = media_analyzer().describe_bytes(raw_bytes, mime_type=attachment.mime_type,
                                                      filename=attachment.name,
                                                      context=f"{label} sent via iMessage by {sender_name}")
        self._described[attachment.id] = description
        return description

    def _resolved(self, resolved: Recipient) -> Tuple[Optional[str], Optional[str]]:
        """What a host's resolution means here: (handle, None), remembered for the approval card
        and the history echo, or (None, error) — with the Full Disk Access hint and one system
        note when the host has no contacts at all."""
        if not resolved.error:
            self._recipients[resolved.handle] = resolved
            return resolved.handle, None
        if resolved.no_contacts:
            self._notify_contact_cache_empty()
            return None, f"{resolved.error}. Reading the AddressBook likely {self._full_disk_access_hint()}."
        return None, resolved.error

    def resolve_recipient(self, recipient: str) -> Tuple[Optional[str], Optional[str]]:
        """Resolve a recipient string (contact name, raw handle, group name or member list) to a
        sendable handle — for a group chat its guid, see `is_group_guid`. Returns (handle, error).
        On success error is None; on failure handle is None and error is a short human-readable
        string."""
        return self._resolved(self.host.resolve(recipient, config.get_owner_name()))

    @staticmethod
    def is_group_guid(handle: str) -> bool:
        """Whether a resolved handle is a group chat's guid rather than a buddy."""
        return is_group_guid(handle)

    def group_for_handle(self, handle: str) -> Optional[GroupChat]:
        """The group chat behind a guid `resolve_recipient` handed out, for the approval card and
        the history echo."""
        resolved = self._recipients.get(handle)
        return resolved.group if resolved else None

    def display_name_for_handle(self, handle: str) -> Optional[str]:
        """Canonical 'First Last' for a handle `resolve_recipient` handed out, or None if unknown."""
        resolved = self._recipients.get(handle)
        return resolved.full_name if resolved else None

    def send(self, recipient: str, body: str) -> str:
        """Send an iMessage/SMS through the host's Messages.app, to a contact or a group chat.
        Returns "" on success or a short error string on failure."""
        result = self.host.send(recipient, body, config.get_owner_name())
        if result.unauthorized:
            self._notify_automation_permission_needed()
        resolved = result.recipient
        if resolved and resolved.error:
            return self._resolved(resolved)[1]
        if result.error:
            info(f"iMessage: send to '{recipient}' failed: {result.error}")
            return result.error
        handle = self._resolved(resolved)[0] if resolved else recipient
        self._log_sent_to_history(recipient, handle, body)
        return ""

    def _notify_contact_cache_empty(self):
        if self._notified_cache_empty:
            return
        self._notified_cache_empty = True
        system_note(self.name,
            "iMessage: contact cache is empty — name-based recipients can't be resolved. "
            f"Likely cause: reading the AddressBook {self._full_disk_access_hint()}. "
            "Sends to raw phone numbers / emails still work."
        )

    def _notify_automation_permission_needed(self):
        system_note("imessage", SatelliteHints.automation(self.host.address) if self.host.address else
            "iMessage send blocked: Keep is not authorized to control Messages. "
            "Grant access in System Settings → Privacy & Security → Automation, then retry."
        )

    def _log_sent_to_history(self, recipient: str, handle: str, body: str):
        """Mirror the read-path's `iMessage sent:` shape so the LLM sees its own outbound
        message; the entry rides along with the feed's next batch. A group is named the way the
        feed names it: its display name, or its members."""
        resolved = self._recipients.get(handle)
        if resolved and resolved.group:
            target_name = resolved.group.label
        else:
            target_name = (resolved.name if resolved and handle != recipient else None) or recipient
        content = f"iMessage sent:\n{config.get_owner_name()} to {target_name}: {body}"
        self._pending_entries.append(FeedEntry(time.time(), content))
