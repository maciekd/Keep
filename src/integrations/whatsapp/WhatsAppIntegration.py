import logging
import os
import queue
import re
import sys
import threading
import time
import types
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from assistant.integration.Feed import Feed, FeedBatch, FeedEntry
from assistant.integration.Integration import Integration
from assistant.PromptInjectionDetector import is_prompt_injection
from assistant.tools.MediaAnalyzer import media_analyzer
import common.Platform as Platform
from common.ConfigManager import config
from common.logging import info, trace, error, fine, note
from common.Paths import AISERVER_DIR
from common.RateLimiter import RateLimiter
from integrations.whatsapp import WhatsAppMessages as msgs
from integrations.whatsapp.WhatsAppMessages import Content, Incoming, KIND_VIDEO

SESSION_PATH = AISERVER_DIR / "whatsapp.session"     # whatsmeow's SQLite store: the device keys
SESSION_SIDECARS = ("-journal", "-wal", "-shm")
POLL_INTERVAL = 2               # seconds between drains of the live queue
CONNECT_TIMEOUT = 30            # seconds for the socket to come up on open
QR_TIMEOUT = 30                 # seconds for WhatsApp to say "unpaired" when a login starts
PAIR_TIMEOUT = 180              # seconds the user has to type the pairing code into the phone
HISTORY_FIRST_CHUNK_WAIT = 120  # seconds after pairing to wait for the first history chunk
HISTORY_QUIET_SECONDS = 45      # no chunk for this long → the history sync is over
FULL_SYNC_SIZE_MB = 10240       # the phone's cap on the history it uploads at pairing
CONTACT_CACHE_REFRESH_INTERVAL = 30 * 60
SEND_RATE_LIMIT_WINDOW = 60.0
SEND_RATE_LIMIT_MAX = 5
LOGIN_HINT = ("link the server to the WhatsApp account with whatsapp_login(phone) from any chat, or in the "
              "console's WhatsApp tab")
PHONE_CHARS = re.compile(r"^[+\d][\d\s().-]{6,}$")
MIN_PHONE_DIGITS = 7


def import_neonize():
    """neonize imports python-magic at module level, and that needs the libmagic library the
    installer never puts on a Mac (on Linux it is the apt package libmagic1). The feed sends text only and
    neonize sniffs mime types only for the media it sends, so a stand-in built on mimetypes
    stands in for the library when it is missing. Deferred to a call: the folder must import
    without the 20 MB native library, and the unit suite never loads it."""
    try:
        import magic   # noqa: F401
    except ImportError:
        import mimetypes
        stub = types.ModuleType("magic")
        stub.from_buffer = lambda data, mime=False: "application/octet-stream"
        stub.from_file = lambda path, mime=False: mimetypes.guess_type(path)[0] or "application/octet-stream"
        sys.modules["magic"] = stub
        install = "brew install libmagic" if Platform.is_mac() else "sudo apt install libmagic1"
        note(f"WhatsApp: libmagic is not installed — media sends would need `{install}`; text is unaffected")
    root = logging.getLogger()
    level_before = root.level
    import neonize
    # neonize's import runs logging.basicConfig(level=INFO): the root logger goes from WARNING
    # to INFO and every stdlib logger in the process (httpx, google_genai, …) starts spilling
    # into the log through stderr. Put the root back and keep INFO for neonize's own loggers,
    # whose pairing and reconnect lines are the ones wanted.
    if root.level != level_before:
        root.setLevel(level_before)
        for name in ("whatsmeow", "Whatsmeow"):
            logging.getLogger(name).setLevel(logging.INFO)
        trace(f"WhatsApp: neonize raised the root logger to INFO on import — restored to {logging.getLevelName(level_before)}")
    return neonize


class WhatsAppIntegration(Integration, Feed):
    """The owner's WhatsApp as a linked device: chats as a feed, and sending.

    Brings the owner's WhatsApp chats into Keep's history, as a feed, and sends a WhatsApp
    message to a contact or a group when asked. Linking takes the owner's phone number and
    the code WhatsApp shows, typed into the phone under Linked devices.

    The account through neonize (whatsmeow, the WhatsApp Web protocol) as a Feed
    (plans/FEED_PLAN.md): `connect()` blocks for the life of the link, so it runs on a thread
    of its own and the events land on the Go callback thread, where a handler only queues.
    History comes once, right after pairing, in chunks the phone uploads — as deep as the
    device properties asked for at pairing time, so `feeds.whatsapp.backfill_months` is read
    then and a deeper horizon means re-pairing — and `fetch` holds the runner in catching-up
    until the chunks stop. Messages missed while the server was down replay on reconnect.
    The client never sends presence or read receipts: a linked device that says it is
    available makes the phone stop ringing for new messages. `login` pairs on the feed's own
    client and the feed starts in place, since the history arrives on the connection that
    paired and a reconnect would not replay it."""
    sensitive = True
    login_hint = LOGIN_HINT

    poll_interval = POLL_INTERVAL
    item_noun = "messages"
    default_backfill_months = 36

    def __init__(self):
        super().__init__()
        self._blacklist: list[str] = [kw.lower() for kw in config.get("whatsapp.blacklist", []) or []]
        self._client = None
        self._thread: Optional[threading.Thread] = None
        self._connected = threading.Event()
        self._unpaired = threading.Event()        # WhatsApp answered the connect with a QR: no device keys
        self._paired = threading.Event()
        self._queued = threading.Event()
        self._logged_out = False
        self._pair_error: Optional[str] = None
        self._open_error: Optional[str] = None
        self._connected_name: Optional[str] = None
        self._own_user: str = ""
        self._login_lock = threading.Lock()
        self._login_phone: Optional[str] = None
        self._login_attempt = 0          # counts the pairing codes issued; a waiter of an earlier one stands down
        self._paired_at: Optional[float] = None
        self._last_history_chunk: Optional[float] = None
        self._history_complete = False
        self._horizon_ts: float = 0.0
        self._live: "queue.Queue[Incoming]" = queue.Queue()
        self._pending_entries: list[FeedEntry] = []    # a sent message, handed over with the next batch
        self._person_names: dict[str, str] = {}
        self._group_names: dict[str, str] = {}
        self._lid_to_pn: dict[str, str] = {}
        self._chats_seen: set[str] = set()
        self._name_index: dict[str, set[str]] = {}       # lowercase name → phone-number users
        self._full_names: dict[str, str] = {}            # user → the contact's full name
        self._group_index: dict[str, str] = {}           # lowercase group name → group JID
        self._contact_cache_time = 0.0
        self._rate_limiter = RateLimiter(SEND_RATE_LIMIT_MAX, SEND_RATE_LIMIT_WINDOW)

    def _on_reset(self):
        self.shut_down()

    def shut_down(self):
        """Stop the feed thread and let the link go (the device stays linked on the account)."""
        from assistant.integration.FeedRunner import Feeds
        Feeds.get_instance().stop(self.feed_name)
        self._stop_client()

    former_names = ("WhatsApp",)

    @property
    def name(self) -> str:
        return "WhatsApp Feed"

    @property
    def config_keys(self) -> List[str]:
        return []

    def check_health(self) -> bool:
        if self._logged_out:
            raise RuntimeError(f"WhatsApp unlinked the server — {LOGIN_HINT}")
        client = self._client
        return bool(client) and self._connected.is_set() and bool(client.is_logged_in)

    def needs_2fa(self) -> bool:
        return not SESSION_PATH.exists()

    # --- //wipe --------------------------------------------------------------------------------

    @classmethod
    def wipe_targets(cls) -> list[tuple[Path, str]]:
        targets = [(SESSION_PATH, "WhatsApp link (whatsmeow session, unlinked on the account first)")]
        targets += [(SESSION_PATH.with_name(SESSION_PATH.name + s), "WhatsApp session sidecar") for s in SESSION_SIDECARS]
        return targets

    @classmethod
    def log_out(cls) -> str:
        """Unlink the device on the account before the session file goes: the file alone
        deleted leaves a linked device the phone still lists. The running client logs out
        when there is one, else a client of its own connects for it."""
        if not SESSION_PATH.exists():
            return "no session file"
        instance = cls.peek_instance()
        if instance is not None and instance._client is not None and instance._connected.is_set():
            client = instance._client
            try:
                client.logout()
                outcome = "unlinked on WhatsApp"
            except Exception as e:
                error(f"WhatsApp: log-out failed: {type(e).__name__}: {e}")
                outcome = f"not unlinked on WhatsApp ({type(e).__name__}: {e})"
            instance.shut_down()
            info(f"WhatsApp: {outcome}")
            return outcome
        if instance is not None:
            instance.shut_down()
        temporary = cls()
        try:
            temporary._start_client()
            if not temporary._connected.wait(CONNECT_TIMEOUT):
                return "not unlinked on WhatsApp: no connection within 30 s" \
                    if not temporary._unpaired.is_set() else "the session was not linked, nothing to unlink"
            temporary._client.logout()
            outcome = "unlinked on WhatsApp"
        except Exception as e:
            error(f"WhatsApp: log-out failed: {type(e).__name__}: {e}")
            outcome = f"not unlinked on WhatsApp ({type(e).__name__}: {e})"
        finally:
            temporary._stop_client()
        info(f"WhatsApp: {outcome}")
        return outcome

    # --- Feed ---------------------------------------------------------------------------------

    @property
    def feed_name(self) -> str:
        return "WhatsApp"

    @property
    def config_slug(self) -> str:
        return "whatsapp"

    @property
    def open_error(self) -> Optional[str]:
        return self._open_error

    def open_feed(self) -> bool:
        from assistant.integration.FeedRunner import Feeds
        if self.needs_2fa():
            self._open_error = f"not linked — {LOGIN_HINT}"
            info(f"WhatsApp: {self._open_error}")
            return False
        if self._pairing_pending():
            # the login owns the client until the phone answers: a pairing code is only good
            # on the socket that asked for it, so this must neither rebuild nor drop it
            self._open_error = "pairing in progress — the code goes into the phone first"
            info(f"WhatsApp: feed not opening, {self._open_error}")
            return False
        if self._client is None:
            self._start_client()
        elif self._client_thread_dead():
            # connect() raised (a bad session file, no network at boot): that client never
            # connects again, so waiting on it would fail every open until a manual restart
            info(f"WhatsApp: the client thread ended ({self._open_error or 'no error recorded'}) — rebuilding it")
            self._stop_client()
            self._start_client()
        if not self._connected.wait(CONNECT_TIMEOUT):
            if self._pairing_pending():
                self._open_error = "pairing in progress — the code goes into the phone first"
                info(f"WhatsApp: feed not opening, {self._open_error} (a login started on this client meanwhile)")
            elif self._unpaired.is_set():
                self._drop_session("the session file holds no linked device")
                self._open_error = f"not linked — {LOGIN_HINT}"
            else:
                self._open_error = f"no connection to WhatsApp within {CONNECT_TIMEOUT} s"
            info(f"WhatsApp: {self._open_error}")
            return False
        if self._logged_out:
            self._open_error = f"WhatsApp unlinked the server — {LOGIN_HINT}"
            return False
        self._open_error = None
        self._horizon_ts = Feeds.horizon_for(self)
        self._identify()
        info(f"WhatsApp feed opened as {self._connected_name}")
        return True

    def close_feed(self):
        pass    # the link outlives the runner: a restart of the feed must not drop a history sync in flight

    def initial_cursor(self, horizon_ts: float) -> Optional[str]:
        self._horizon_ts = horizon_ts
        return repr(float(horizon_ts))

    def fetch(self, cursor: Optional[str], backfill: bool) -> FeedBatch:
        if self._logged_out:
            raise RuntimeError(f"WhatsApp unlinked the server — {LOGIN_HINT}")
        since = float(cursor) if cursor else 0.0
        entries = self._drain(backfill)
        if not entries and self._history_in_progress():
            self._queued.wait(POLL_INTERVAL)
            entries = self._drain(backfill)
        # swap first: a send() appending between a read and a reset would be lost
        pending, self._pending_entries = self._pending_entries, []
        if pending:
            entries = pending + entries
        newest = max([since] + [e.timestamp for e in entries])
        return FeedBatch(entries, cursor=repr(newest), exhausted=not self._history_in_progress())

    def _history_in_progress(self) -> bool:
        """Whether the phone is still uploading the history of a pairing of this process: the
        runner stays in catching-up meanwhile instead of going live after the first drain."""
        if self._paired_at is None or self._history_complete:
            return False
        now = time.time()
        if self._last_history_chunk is None:
            return now - self._paired_at < HISTORY_FIRST_CHUNK_WAIT
        return now - self._last_history_chunk < HISTORY_QUIET_SECONDS

    def _drain(self, backfill: bool) -> list[FeedEntry]:
        self._queued.clear()
        entries: list[FeedEntry] = []
        dropped = 0
        while True:
            try:
                incoming = self._live.get_nowait()
            except queue.Empty:
                break
            if incoming.from_history and self._horizon_ts and incoming.timestamp < self._horizon_ts:
                dropped += 1
                continue
            try:
                content = self._format(incoming, analyze=not backfill and not incoming.from_history)
            except Exception as e:
                error(f"WhatsApp: could not format a message from {incoming.chat}: {type(e).__name__}: {e}")
                continue
            if content:
                entries.append(FeedEntry(incoming.timestamp, content))
        if dropped:
            fine(f"WhatsApp: dropped {dropped} history messages older than the horizon")
        return entries

    def _format(self, incoming: Incoming, analyze: bool) -> Optional[str]:
        if msgs.is_skipped_chat(incoming.chat):
            return None
        if incoming.edit:
            trace(f"WhatsApp: skipped an edit in {incoming.chat}")
            return None
        content = msgs.content_of(incoming.message)
        if content.empty:
            return None
        owner = config.get_owner_name()
        self._chats_seen.add(msgs.jid_parts(incoming.chat)[0])
        sender_name = self.person(incoming.sender or incoming.chat, incoming.push_name) if not incoming.from_me else owner
        group_name = self.group(incoming.chat) if msgs.is_group(incoming.chat) else None
        recipient_name = self.person(incoming.chat, "") if incoming.from_me and not group_name else None
        if self._is_blacklisted(sender_name, group_name, recipient_name):
            return None
        if not incoming.from_me and content.text and is_prompt_injection(content.text):
            info(f"WhatsApp: blocked prompt injection from {sender_name}")
            return None
        media_text = ""
        if content.media is not None:
            media_text = self._describe_media(content, sender_name) if analyze else msgs.placeholder(content.media.kind)
        return msgs.format_entry(incoming, self, owner, content, media_text)

    def _describe_media(self, content: Content, sender_name: str) -> str:
        media = content.media
        label = "Video" if media.kind == KIND_VIDEO else "Photo"
        fallback = msgs.placeholder(media.kind)
        try:
            data = self._client.download_any(media.message)
        except Exception as e:
            info(f"WhatsApp: {label.lower()} from {sender_name} could not be downloaded: {type(e).__name__}: {e}")
            return fallback
        if not data:
            return fallback
        if media.kind == KIND_VIDEO and len(data) > media_analyzer().video_size_limit():
            info(f"WhatsApp: video from {sender_name} is {len(data) / 1024 / 1024:.1f} MB, over the analysis cap")
            return fallback
        return media_analyzer().describe_bytes(data, mime_type=media.mimetype, fallback=fallback,
                                               context=f"{label} sent via WhatsApp by {sender_name}")

    def _is_blacklisted(self, sender_name: str, group_name: Optional[str], recipient_name: Optional[str]) -> bool:
        if not self._blacklist:
            return False
        haystacks = [h.lower() for h in (sender_name, group_name, recipient_name) if h]
        return any(kw in h for kw in self._blacklist for h in haystacks)

    # --- names (the `Names` the formatter takes) --------------------------------------------------

    def person(self, jid: str, push_name: str) -> str:
        """The contact's name for a JID: the address book's full name, else its first name,
        else the push name, else the number."""
        user, server = msgs.jid_parts(jid)
        if server == msgs.SERVER_LID:
            user = self._phone_user_of_lid(user) or user
        cached = self._person_names.get(user)
        if cached:
            return cached
        name = self._lookup_contact(user, server if server != msgs.SERVER_LID else msgs.SERVER_USER)
        if name:
            self._person_names[user] = name
            return name
        if push_name:
            return push_name
        return f"+{user}" if user.isdigit() else user

    def group(self, jid: str) -> str:
        user, _ = msgs.jid_parts(jid)
        cached = self._group_names.get(user)
        if cached:
            return cached
        name = ""
        if self._client is not None:
            try:
                name = self._client.get_group_info(self._jid(user, msgs.SERVER_GROUP)).GroupName.Name
            except Exception as e:
                trace(f"WhatsApp: no group info for {jid}: {type(e).__name__}: {e}")
        if name:
            self._group_names[user] = name
            return name
        return "group " + user

    def _phone_user_of_lid(self, lid_user: str) -> Optional[str]:
        pn = self._lid_to_pn.get(lid_user)
        if pn or self._client is None:
            return pn
        try:
            found = self._client.get_pn_from_lid(self._jid(lid_user, msgs.SERVER_LID))
            if found and found.User:
                self._lid_to_pn[lid_user] = found.User
                return found.User
        except Exception as e:
            trace(f"WhatsApp: no phone number for lid {lid_user}: {type(e).__name__}: {e}")
        return None

    def _lookup_contact(self, user: str, server: str) -> str:
        if self._client is None or not user:
            return ""
        try:
            found = self._client.contact.get_contact(self._jid(user, server))
        except Exception as e:
            trace(f"WhatsApp: contact lookup for {user} failed: {type(e).__name__}: {e}")
            return ""
        if not getattr(found, "Found", False):
            return ""
        return found.FullName or found.FirstName or found.PushName or found.BusinessName or ""

    # --- the client ------------------------------------------------------------------------------

    def _device_props(self):
        from neonize.proto.waCompanionReg.WAWebProtobufsCompanionReg_pb2 import DeviceProps
        from assistant.integration.FeedRunner import Feeds
        months = config.get(f"feeds.{self.config_slug}.backfill_months", self.default_backfill_months)
        try:
            days = max(1, int(float(months) * 30.5))
        except (TypeError, ValueError):
            days = self.default_backfill_months * 30
        trace(f"WhatsApp: a pairing asks for {days} days of history ({months} months)")
        history = DeviceProps.HistorySyncConfig(fullSyncDaysLimit=days, fullSyncSizeMbLimit=FULL_SYNC_SIZE_MB,
                                                storageQuotaMb=FULL_SYNC_SIZE_MB)
        return DeviceProps(os="Keep", platformType=DeviceProps.CHROME, requireFullSync=True, historySyncConfig=history)

    def _start_client(self):
        """Build the client, register the handlers and connect on a thread of its own."""
        import_neonize()
        from neonize.client import NewClient
        from neonize.events import (ConnectedEv, ConnectFailureEv, HistorySyncEv, LoggedOutEv, MessageEv,
                                    OfflineSyncCompletedEv, PairStatusEv, StreamReplacedEv, TemporaryBanEv,
                                    ClientOutdatedEv)
        self._connected.clear()
        self._unpaired.clear()
        self._logged_out = False
        client = NewClient(str(SESSION_PATH), props=self._device_props())
        client.event.qr(self._on_qr)       # the QR has a callback of its own, outside the event list
        client.event(PairStatusEv)(self._on_pair_status)
        client.event(ConnectedEv)(self._on_connected)
        client.event(ConnectFailureEv)(self._on_connect_failure)
        client.event(LoggedOutEv)(self._on_logged_out)
        client.event(StreamReplacedEv)(self._on_stream_replaced)
        client.event(TemporaryBanEv)(self._on_temporary_ban)
        client.event(ClientOutdatedEv)(self._on_client_outdated)
        client.event(HistorySyncEv)(self._on_history_sync)
        client.event(OfflineSyncCompletedEv)(self._on_offline_sync_completed)
        client.event(MessageEv)(self._on_message)
        self._client = client

        def run():
            try:
                client.connect()
            except Exception as e:
                error(f"WhatsApp: the client stopped: {type(e).__name__}: {e}")
                self._open_error = f"{type(e).__name__}: {e}"
            finally:
                self._connected.clear()
                trace("WhatsApp: client thread ended")

        self._thread = threading.Thread(target=run, name="WhatsAppClient", daemon=True)
        self._thread.start()
        info(f"WhatsApp: client started on {SESSION_PATH.name}")

    def _client_thread_dead(self) -> bool:
        """A client whose connect() returned or raised: nothing will ever set `_connected` on it."""
        return self._thread is not None and not self._thread.is_alive()

    def _stop_client(self):
        """Drop the socket, then cancel the Go side. `disconnect` is what ends the link's
        connection (at once); `stop` does not make the blocking `connect` return in neonize
        0.5.2, so the thread stays parked as a daemon — a restart builds a new client on the
        same session file and the old one holds no connection."""
        client, self._client = self._client, None
        self._connected.clear()
        if client is None:
            return
        for step in ("disconnect", "stop"):
            try:
                getattr(client, step)()
            except Exception as e:
                trace(f"WhatsApp: {step} raised {type(e).__name__}: {e}")
        if self._thread is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                trace("WhatsApp: the client thread is still parked in connect() after stop — left as a daemon")
            self._thread = None
        self._connected_name = None

    def _identify(self):
        try:
            me = self._client.get_me()
            self._own_user = msgs.jid_parts(f"{me.JID.User}@{me.JID.Server}")[0]
            self._connected_name = me.PushName or f"+{me.JID.User}"
        except Exception as e:
            trace(f"WhatsApp: get_me failed: {type(e).__name__}: {e}")
            self._connected_name = self._connected_name or "the linked account"

    def _drop_session(self, why: str):
        info(f"WhatsApp: dropping {SESSION_PATH.name} — {why}")
        self._stop_client()
        for path in [SESSION_PATH] + [SESSION_PATH.with_name(SESSION_PATH.name + s) for s in SESSION_SIDECARS]:
            try:
                path.unlink(missing_ok=True)
            except OSError as e:
                error(f"WhatsApp: could not delete {path.name}: {e}")

    # --- events, on the Go callback thread: queue and return ---------------------------------------

    def _on_qr(self, _client, _qr_data):
        """WhatsApp answered the connect with a QR to scan: no device keys in the session. The
        QR itself is not shown — the login pairs by phone number instead."""
        if not self._unpaired.is_set():
            info("WhatsApp: no linked device in the session — a pairing is needed")
        self._unpaired.set()

    def _on_pair_status(self, _client, event):
        if getattr(event, "Error", ""):
            self._pair_error = event.Error
            error(f"WhatsApp: pairing failed: {event.Error}")
            return
        self._paired_at = time.time()
        self._history_complete = False
        self._last_history_chunk = None
        self._paired.set()
        jid = getattr(event, "ID", None)
        info(f"WhatsApp: paired as {jid.User if jid is not None else 'the account'} on {getattr(event, 'Platform', '')}")

    def _on_connected(self, _client, _event):
        self._connected.set()
        self._logged_out = False
        info("WhatsApp: connected")

    def _on_connect_failure(self, _client, event):
        self._open_error = f"connection refused: {event}"
        error(f"WhatsApp: {self._open_error}")

    def _on_logged_out(self, _client, event):
        """The phone unlinked the server, or 14 days passed without the phone: the keys are
        void, so the file goes and the monitor's needs-2FA state carries the login hint."""
        self._logged_out = True
        self._connected.clear()
        reason = getattr(event, "Reason", "")
        error(f"WhatsApp: logged out by WhatsApp ({reason or 'no reason given'}) — {LOGIN_HINT}")
        threading.Thread(target=self._drop_session, args=(f"logged out ({reason or 'no reason'})",),
                         name="WhatsAppLogout", daemon=True).start()

    def _on_stream_replaced(self, _client, _event):
        error("WhatsApp: the stream was replaced — another client opened this session")

    def _on_temporary_ban(self, _client, event):
        error(f"WhatsApp: temporary ban: {event}")

    def _on_client_outdated(self, _client, _event):
        error("WhatsApp: the client is outdated — WhatsApp refuses this neonize version; bump the pin")

    def _on_offline_sync_completed(self, _client, event):
        fine(f"WhatsApp: offline messages delivered ({getattr(event, 'Count', '?')})")

    def _on_history_sync(self, _client, event):
        """A chunk of the history the phone uploads after pairing: every message of every
        conversation goes into the queue as a history row, newest chunk first as WhatsApp
        sends them; the horizon cut and the formatting happen on the feed thread."""
        data = event.Data
        self._last_history_chunk = time.time()
        for mapping in getattr(data, "phoneNumberToLidMappings", ()):
            self._lid_to_pn[msgs.jid_parts(mapping.lidJID)[0]] = msgs.jid_parts(mapping.pnJID)[0]
        for conversation in getattr(data, "conversations", ()):
            chat = conversation.ID
            if getattr(conversation, "lidJID", "") and getattr(conversation, "pnJID", ""):
                self._lid_to_pn[msgs.jid_parts(conversation.lidJID)[0]] = msgs.jid_parts(conversation.pnJID)[0]
                if msgs.jid_parts(chat)[1] == msgs.SERVER_LID:
                    chat = conversation.pnJID
            title = getattr(conversation, "name", "") or getattr(conversation, "displayName", "")
            if title and msgs.is_group(chat):
                self._group_names[msgs.jid_parts(chat)[0]] = title
            for row in getattr(conversation, "messages", ()):
                web = row.message
                key = web.key
                if key.fromMe:
                    sender = f"{self._own_user}@{msgs.SERVER_USER}" if self._own_user else ""
                else:
                    sender = key.participant or chat
                self._live.put(Incoming(timestamp=float(web.messageTimestamp), chat=chat, sender=sender,
                                        from_me=bool(key.fromMe), push_name=getattr(web, "pushName", "") or "",
                                        message=web.message, from_history=True))
        sync_type = getattr(data, "syncType", None)
        progress = getattr(data, "progress", 0)
        if sync_type == 2 and progress >= 100:     # FULL, the last chunk
            self._history_complete = True
        fine(f"WhatsApp: history chunk type {sync_type}, progress {progress}%, "
             f"{len(getattr(data, 'conversations', ()))} conversations, {self._live.qsize()} queued")
        self._queued.set()

    def _on_message(self, _client, event):
        source = event.Info.MessageSource
        chat = self._jid_string(source.Chat)
        sender = self._jid_string(source.Sender)
        alt = self._jid_string(getattr(source, "SenderAlt", None))
        if msgs.jid_parts(sender)[1] == msgs.SERVER_LID and alt and msgs.jid_parts(alt)[1] == msgs.SERVER_USER:
            self._lid_to_pn[msgs.jid_parts(sender)[0]] = msgs.jid_parts(alt)[0]
            sender = alt
        # neonize stamps a live event in milliseconds; a history-sync row (messageTimestamp)
        # is in seconds, and the feed, the horizon and the history rows all run on seconds.
        self._live.put(Incoming(timestamp=float(event.Info.Timestamp) / 1000.0, chat=chat, sender=sender,
                                from_me=bool(source.IsFromMe), push_name=event.Info.Pushname or "",
                                message=event.Message, from_history=False, edit=getattr(event.Info, "Edit", "") or ""))
        self._queued.set()

    @staticmethod
    def _jid(user: str, server: str):
        """A JID protobuf for the client's calls; the one place neonize's type is built."""
        import_neonize()
        from neonize.utils.jid import build_jid
        return build_jid(user, server)

    @staticmethod
    def _jid_string(jid) -> str:
        return f"{jid.User}@{jid.Server}" if jid is not None and jid.User else (jid.Server if jid is not None else "")

    # --- the login --------------------------------------------------------------------------------

    def login(self, phone: str) -> Tuple[Optional[str], Optional[str]]:
        """Ask WhatsApp for a pairing code for *phone* (E.164): the client comes up unpaired,
        the code comes back at once for the user to type into the phone, and `wait_paired`
        tells when the link landed. Returns (code, None), or (None, why not)."""
        digits = re.sub(r"\D", "", phone)
        if not phone.strip().startswith("+") or len(digits) < MIN_PHONE_DIGITS:
            return None, "WhatsApp login failed: expected the phone number with its country code, e.g. +46701234567"
        if not self._login_lock.acquire(blocking=False):
            info("WhatsApp login: refused, another login is in progress")
            return None, "WhatsApp login failed: another login is in progress (the console's WhatsApp tab?)"
        try:
            if self._client is not None and self._connected.is_set() and self._client.is_logged_in:
                return None, f"WhatsApp is already linked as {self._connected_name or 'the account'} — nothing to do"
            if self._client is not None:
                # an unpaired client's socket is gone once WhatsApp's pairing window closed
                # (PairPhone then fails with "websocket not connected"): a fresh one each time
                if self._login_phone is not None and not self._paired.is_set():
                    info(f"WhatsApp login: a code is still out for +{self._login_phone} — "
                         f"this login supersedes it, its waiter stands down")
                info("WhatsApp login: rebuilding the unpaired client for a fresh socket")
                self._stop_client()
            self._start_client()
            if not self._unpaired.wait(QR_TIMEOUT):
                if self._connected.is_set():
                    return None, "WhatsApp is already linked — nothing to do"
                return None, f"WhatsApp login failed: no answer from WhatsApp within {QR_TIMEOUT} s"
            self._paired.clear()
            self._pair_error = None
            try:
                code = self._client.PairPhone(digits, True)
            except Exception as e:
                error(f"WhatsApp login: pairing request failed: {type(e).__name__}: {e}")
                return None, f"WhatsApp login failed: WhatsApp refused the pairing request ({e})"
            self._login_phone = digits
            self._login_attempt += 1
            code = code if "-" in code or len(code) != 8 else f"{code[:4]}-{code[4:]}"
            info(f"WhatsApp login: pairing code issued for +{digits}, valid for a few minutes")
            return code, None
        finally:
            self._login_lock.release()

    def wait_paired(self, timeout: float = PAIR_TIMEOUT) -> Optional[str]:
        """Block until the phone accepted the code; None then, else what went wrong. A code
        that never landed ends the attempt: the client goes (its socket is dead by then) and
        the next login starts clean. A waiter whose code a newer login replaced stands down
        without touching the client — that one belongs to the newer attempt now."""
        attempt = self._login_attempt
        paired = self._paired.wait(timeout)
        if attempt != self._login_attempt:
            info(f"WhatsApp login: pairing attempt {attempt} was superseded by attempt {self._login_attempt}")
            return "a newer whatsapp_login replaced this code"
        if not paired:
            failure = self._pair_error or "the code was not entered in time — run whatsapp_login again for a new one"
            info(f"WhatsApp login: the pairing attempt is over ({failure}) — stopping the unpaired client")
            self._login_phone = None
            self._stop_client()
            return failure
        if not self._connected.wait(CONNECT_TIMEOUT):
            return "paired, but the connection did not come up"
        self._identify()
        return None

    def login_in_progress(self) -> bool:
        return self._login_lock.locked()

    def _pairing_pending(self) -> bool:
        """A login is asking for a code, or one is out and the phone has not answered yet."""
        return self._login_lock.locked() or (self._login_phone is not None and not self._paired.is_set()
                                             and not self._logged_out)

    def finish_login(self) -> Optional[dict]:
        """Start the feed on the client that just paired — in place, no rebuild, so the
        history sync the phone began on this connection is not lost."""
        from assistant.integration.IntegrationMonitor import IntegrationMonitor
        return IntegrationMonitor.get_instance().restart_in_place(self.name)

    def get_auth_status(self) -> dict:
        """The auth status for the TUI tab, from what the feed already knows."""
        if self._pairing_pending():
            return {"status": "needs_auth", "message": "Pairing in progress — type the code into the phone"}
        if self._logged_out:
            return {"status": "needs_auth", "message": "WhatsApp unlinked the server — press Ctrl-P to link again"}
        if self.needs_2fa():
            return {"status": "needs_auth", "message": "Not linked — press Ctrl-P to link the WhatsApp account"}
        if self._connected.is_set() and self._client is not None:
            return {"status": "authenticated", "message": f"Linked as {self._connected_name or 'the account'}"}
        if self._open_error:
            return {"status": "error", "message": self._open_error}
        return {"status": "connecting", "message": "Connecting..."}

    # --- sending ----------------------------------------------------------------------------------

    def _refresh_contact_cache(self):
        """The phone's address book as WhatsApp synced it, and the joined groups, as name
        indexes for `resolve_recipient`."""
        if self._client is None:
            return
        names: dict[str, set[str]] = {}
        full_names: dict[str, str] = {}
        try:
            for contact in self._client.contact.get_all_contacts():
                user, server = contact.JID.User, contact.JID.Server
                if server == msgs.SERVER_LID:
                    user = self._lid_to_pn.get(user) or self._phone_user_of_lid(user) or ""
                if not user or not user.isdigit():
                    continue
                full = contact.Info.FullName or contact.Info.FirstName or contact.Info.PushName or contact.Info.BusinessName
                if not full:
                    continue
                full_names[user] = full
                self._person_names.setdefault(user, full)
                for candidate in {full, contact.Info.FirstName, contact.Info.PushName, contact.Info.BusinessName}:
                    if candidate:
                        names.setdefault(" ".join(candidate.split()).lower(), set()).add(user)
        except Exception as e:
            error(f"WhatsApp: reading the contacts failed: {type(e).__name__}: {e}")
        groups: dict[str, str] = {}
        try:
            for group in self._client.get_joined_groups():
                title = group.GroupName.Name
                jid = self._jid_string(group.JID)
                if title:
                    groups[" ".join(title.split()).lower()] = jid
                    self._group_names[msgs.jid_parts(jid)[0]] = title
        except Exception as e:
            error(f"WhatsApp: reading the groups failed: {type(e).__name__}: {e}")
        self._name_index, self._full_names, self._group_index = names, full_names, groups
        self._contact_cache_time = time.time()
        info(f"WhatsApp: contact cache refreshed — {len(full_names)} contacts, {len(groups)} groups")

    def _ensure_contact_cache(self):
        if time.time() - self._contact_cache_time > CONTACT_CACHE_REFRESH_INTERVAL:
            self._refresh_contact_cache()

    @staticmethod
    def _looks_like_phone(text: str) -> bool:
        return bool(PHONE_CHARS.match(text)) and len(re.sub(r"\D", "", text)) >= MIN_PHONE_DIGITS

    def _normalize_phone(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """The digits of an international number; a local one is matched against the
        contacts' numbers by its suffix, since the country code is not known here."""
        digits = re.sub(r"\D", "", text)
        if text.strip().startswith("+"):
            return digits, None
        if digits.startswith("00"):
            return digits[2:], None
        if digits.startswith("0"):
            suffix = digits[1:]
            matches = {user for user in self._full_names if user.endswith(suffix)}
            if len(matches) == 1:
                return matches.pop(), None
            return None, (f"{text.strip()} has no country code and matches "
                          f"{'no' if not matches else 'several'} contact(s) — give it as +46…")
        return digits, None

    def resolve_recipient(self, recipient: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """(jid, display name, error) for a contact name, a group name or a number."""
        candidate = " ".join((recipient or "").split())
        if not candidate:
            return None, None, "empty recipient"
        if self._client is None or not self._connected.is_set():
            return None, None, f"WhatsApp is not connected — {LOGIN_HINT}"
        self._ensure_contact_cache()
        if self._looks_like_phone(candidate):
            user, err = self._normalize_phone(candidate)
            if err:
                return None, None, err
            jid, err = self._jid_on_whatsapp(user)
            if err:
                return None, None, err
            return jid, self._full_names.get(user) or f"+{user}", None
        key = candidate.lower()
        users = self._name_index.get(key, set())
        if len(users) == 1:
            user = next(iter(users))
            jid, err = self._jid_on_whatsapp(user)
            if err:
                return None, None, err
            info(f"WhatsApp: resolved '{candidate}' → +{user} ({self._full_names.get(user, '?')})")
            return jid, self._full_names.get(user, candidate), None
        if len(users) > 1:
            options = sorted(self._full_names.get(u, f"+{u}") for u in users)
            return None, None, f"'{candidate}' is ambiguous — say the full name: {', '.join(options)}"
        group_jid = self._group_index.get(key)
        if group_jid:
            info(f"WhatsApp: resolved '{candidate}' → group {group_jid}")
            return group_jid, candidate, None
        if not self._full_names and not self._group_index:
            return None, None, f"unknown contact: {candidate} (the contact list has not synced from the phone yet)"
        return None, None, f"unknown contact: {candidate} (cache has {len(self._full_names)} contacts, {len(self._group_index)} groups)"

    def _jid_on_whatsapp(self, user: str) -> Tuple[Optional[str], Optional[str]]:
        try:
            answers = list(self._client.is_on_whatsapp(f"+{user}"))
        except Exception as e:
            error(f"WhatsApp: is_on_whatsapp(+{user}) failed: {type(e).__name__}: {e}")
            return None, f"could not check +{user} on WhatsApp ({e})"
        if not answers or not answers[0].IsIn:
            return None, f"+{user} is not on WhatsApp"
        jid = self._jid_string(answers[0].JID)
        lid_user, server = msgs.jid_parts(jid)
        if server == msgs.SERVER_LID and lid_user:
            # an account with number privacy on answers with its lid: remember whose it is,
            # so the approval card and an incoming message still show the phone number
            trace(f"WhatsApp: +{user} is on WhatsApp as lid {lid_user}")
            self._lid_to_pn[lid_user] = user
        return jid or f"{user}@{msgs.SERVER_USER}", None

    def phone_for(self, jid: str) -> Optional[str]:
        """The phone number behind a resolved JID (`+46…`): the JID's own for a phone JID, the
        mapped one for a lid, None for a group or a lid nothing maps — never a lid dressed up
        as a number."""
        user, server = msgs.jid_parts(jid or "")
        if server == msgs.SERVER_USER and user.isdigit():
            return f"+{user}"
        if server == msgs.SERVER_LID:
            pn = self._phone_user_of_lid(user)
            return f"+{pn}" if pn else None
        return None

    def is_known(self, recipient: str) -> bool:
        """Whether the owner has a chat or an address-book entry behind *recipient* — the
        approval card says so when not, since messaging strangers first is what gets a
        number restricted."""
        candidate = " ".join((recipient or "").split())
        if not candidate or not self._looks_like_phone(candidate):
            return True     # a name resolves through the contacts or not at all
        user, _ = self._normalize_phone(candidate)
        return bool(user) and (user in self._full_names or user in self._chats_seen)

    def send(self, recipient: str, body: str) -> str:
        """Send *body* to a contact, a group or a number. "" on success, else why not."""
        if not body or not body.strip():
            info("WhatsApp send skipped: empty body")
            return "empty message body"
        rate_err = self._rate_limiter.check()
        if rate_err:
            info(f"WhatsApp send blocked: {rate_err}")
            return rate_err
        jid, display, err = self.resolve_recipient(recipient)
        if err:
            info(f"WhatsApp send: could not resolve recipient '{recipient}': {err}")
            return err
        try:
            user, server = msgs.jid_parts(jid)
            self._client.send_message(self._jid(user, server), body)
        except Exception as e:
            error(f"WhatsApp send to {display} ({jid}) failed: {type(e).__name__}: {e}")
            return f"WhatsApp send failed: {e}"
        self._rate_limiter.record()
        self._chats_seen.add(msgs.jid_parts(jid)[0])
        owner = config.get_owner_name()
        self._pending_entries.append(FeedEntry(time.time(), f"WhatsApp sent:\n{owner} to {display}: {body}"))
        self._queued.set()
        info(f"WhatsApp: sent {len(body)} chars to {display} ({jid})")
        return ""
