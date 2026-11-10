"""The chat hub: every registered chat transport, a `ChatSession` per conversation, and the
process-wide `Notifier` the assistant core pushes messages, photos, the command status line
and approval requests through.

One hub per process (`ChatHub.get_instance()`); each chat interface registers its transport
on it, so a notification fans out to every chat the owner is reachable in. The command
echo, the approval prompt and the secret prompt are the exceptions: they go back to the
chat whose request ran the command, so only the person who asked sees and answers the
question; a request from outside the chat layer (voice) is asked in every owner chat and
the first answer wins across all of them.

Secret requests work like approvals, but the answer is a value the user types and the
prompt opens only in the chat that asked (every owner chat when the request came from
outside the chat layer): the hub
hands it to the requester's `accept` callback and forgets it. The value is never logged,
never put in the transcript and never reaches a session — a transport asks `claim_secret`
before it treats an incoming text as chat.
"""
import concurrent.futures
import threading
import time
import uuid
from html import escape
from typing import Any, Callable, Optional

from assistant import Destinations
from assistant.Notifier import SecretRejected
from assistant.Telemetry import Telemetry
from assistant.integration.SingletonMixin import SingletonMixin
from common.ConfigManager import config
from common.logging import error, info, trace
from interfaces.chat import RichText
from interfaces.chat.ChatSession import ChatSession
from interfaces.chat.ChatTransport import ChatTransport
from interfaces.chat.ChatTypes import (DECISION_ACCEPTED, DECISION_APPROVE, DECISION_CANCEL, DECISION_CANCELLED,
                                       DECISION_REJECTED, DECISION_TIMEOUT, DECISION_UNAVAILABLE, IncomingMessage)


class _CommandEcho:
    """One running command's echo: the lines it put in the chats, to be annotated when the
    command ends, and when it started."""

    def __init__(self, text: str):
        self.text = text
        self.started = time.time()
        self.refs: list[tuple[ChatTransport, Any, Any]] = []


class ChatHub(SingletonMixin):
    DEFAULT_APPROVAL_TIMEOUT_SECONDS = 300.0
    APPROVAL_POLL_SECONDS = 1.0   # how often a pending approval looks at its request's kill flag
    STALE_COMMAND_SECONDS = 600.0   # a command echo never completed is forgotten after this
    DEFAULT_SECRET_TIMEOUT_SECONDS = 3600.0   # the user often has to set something up before answering
    SECRET_ATTEMPTS = 3
    MAX_TRANSCRIPT = 500

    def __init__(self, assistant=None):
        self.assistant = assistant
        self._transports: list[ChatTransport] = []
        self._sessions: dict[tuple, ChatSession] = {}
        self._lock = threading.Lock()
        self._transcript: list[tuple[float, str, Any, str, str]] = []
        # The running commands' echoes and, per (transport, conversation), when the last
        # command echoed there ended; both guarded by _lock.
        self._running_commands: list[_CommandEcho] = []
        self._last_command_end: dict[tuple, float] = {}
        self._pending_approvals: dict[str, dict] = {}
        self._approvals_lock = threading.Lock()
        timeout = config.get('chat.approval_timeout_seconds',
                             config.get('telegram_bot.approval_timeout_seconds', self.DEFAULT_APPROVAL_TIMEOUT_SECONDS))
        self._approval_timeout = float(timeout)
        # token → entry; guarded by _approvals_lock like the approvals.
        self._pending_secrets: dict[str, dict] = {}
        self._secret_timeout = float(config.get('chat.secret_request_timeout_seconds',
                                                self.DEFAULT_SECRET_TIMEOUT_SECONDS))

    def _on_reset(self):
        self.stop()

    # --- transports & sessions ------------------------------------------------------------

    def attach(self, assistant):
        self.assistant = assistant

    def register(self, transport: ChatTransport):
        with self._lock:
            if transport in self._transports:
                trace(f"Chat transport {transport.name} already registered, nothing to do.")
                return
            self._transports.append(transport)
        info(f"Chat transport {transport.name} registered (source={transport.source}).")
        self._register_owner_destination()

    def unregister(self, transport: ChatTransport):
        with self._lock:
            if transport not in self._transports:
                trace(f"Chat transport {transport.name} was not registered, nothing to do.")
                return
            self._transports.remove(transport)
            dropped = [key for key in self._sessions if key[0] is transport]
            sessions = [self._sessions.pop(key) for key in dropped]
        for session in sessions:
            session.cancel()
        info(f"Chat transport {transport.name} unregistered ({len(sessions)} session(s) dropped).")
        if not self.transports:
            Destinations.unregister(self.OWNER_DESTINATION)

    @property
    def transports(self) -> list[ChatTransport]:
        with self._lock:
            return list(self._transports)

    def session_for(self, transport: ChatTransport, conversation) -> ChatSession:
        key = (transport, conversation)
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                session = self._sessions[key] = ChatSession(self, transport, conversation)
                trace(f"Chat session opened: {transport.name} [{conversation}]")
            return session

    def receive(self, transport: ChatTransport, msg: IncomingMessage) -> bool:
        """A user action from a transport. Never blocks: the session works on its own thread.
        False when the session dropped it on the spot (`ChatSession.submit`)."""
        return self.session_for(transport, msg.conversation).submit(msg)

    def receive_chat_message(self, source: str, conversation: Any, sender: str, text: str, *,
                             guest: bool = False, sender_is_bot: bool = False,
                             chat_name: Optional[str] = None) -> bool:
        """Notifier: a feed saw *text* in *conversation* of the transport whose source is
        *source* (the owner's Telegram account reading a group the bot is in — the bot itself
        never receives a plain message there). The turn runs as if that transport had received
        the message, so the reply goes out through it. False when it never became a turn: no
        such transport, or the session dropped it (a guest while a reply is in flight)."""
        transport = next((t for t in self.transports if t.source == source), None)
        if transport is None:
            info(f"No {source} transport is running, a chat message from {sender} in [{conversation}] is dropped")
            return False
        info(f"{transport.name} [{conversation}]: message from {sender} handed over by a feed"
             f"{' (guest)' if guest else ''}{' (bot)' if sender_is_bot else ''}")
        taken = self.receive(transport, IncomingMessage(conversation=conversation, sender=sender, text=text,
                                                        chat_name=chat_name, guest=guest, sender_is_bot=sender_is_bot))
        if not taken:
            info(f"{transport.name} [{conversation}]: the message from {sender} was dropped by the session")
        return taken

    def open_conversation(self, query: str) -> bool:
        """A turn the server starts (the first-run brief, the Full Disk Access grant seen):
        *query* goes to the assistant as a system turn in one owner conversation — one turn,
        not one per chat, so the history gets a single exchange — and the reply is rendered
        there like any other. The first attended chat takes it (the console counts only with
        a TUI client connected: after the grant the terminal is gone and the user is on
        Telegram), else the first there is. Never blocks. False when no owner chat is
        reachable."""
        targets = self._targets()
        if not targets:
            info("No owner conversation to open, the assistant waits to be spoken to.")
            return False
        attended = [t for t in targets if t[0].attended()]
        if not attended:
            info(f"No attended owner conversation — opening the first one ({targets[0][0].name})")
        transport, conversation = (attended or targets)[0]
        info(f"Opening the conversation in {transport.name} [{conversation}]")
        self.session_for(transport, conversation).open(query)
        return True

    def expect_photo(self, transport: ChatTransport, conversation, sender: str, group_id=None):
        """A transport is downloading a photo from *conversation*: its session holds the album
        open until `receive` brings the photo or `photo_dropped` gives it up. Album members
        download concurrently, and the batch must not go out while one is still on its way."""
        self.session_for(transport, conversation).expect_photo(sender, group_id)

    def photo_dropped(self, transport: ChatTransport, conversation):
        self.session_for(transport, conversation).photo_dropped()

    def stop(self):
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.cancel()
        Destinations.unregister(self.OWNER_DESTINATION)

    # The destination `post_message("owner", ...)` reaches: the proactive route, exactly what
    # `send_message` pushes to (each transport's notification_conversations — Telegram's
    # notify_chat_ids), not every owner chat. Registered with the first transport, dropped
    # with the last.
    OWNER_DESTINATION = "owner"

    def _register_owner_destination(self):
        Destinations.register(Destinations.Destination(
            self.OWNER_DESTINATION, Destinations.KIND_CHAT, lambda text: self.send_message(text)))

    # --- transcript -----------------------------------------------------------------------

    def record(self, transport: ChatTransport, conversation, direction: str, text: str):
        with self._lock:
            self._transcript.append((time.time(), transport.name, conversation, direction, text))
            if len(self._transcript) > self.MAX_TRANSCRIPT:
                self._transcript = self._transcript[-self.MAX_TRANSCRIPT:]

    @property
    def transcript(self) -> list[tuple[float, str, Any, str, str]]:
        with self._lock:
            return list(self._transcript)

    # --- Notifier -------------------------------------------------------------------------

    def _targets(self, notifications: bool = False) -> list[tuple[ChatTransport, Any]]:
        """Every owner conversation on every transport, or with *notifications* the ones
        each transport routes its proactive pushes to."""
        targets = []
        for transport in self.transports:
            try:
                convs = transport.notification_conversations() if notifications else transport.owner_conversations()
                targets.extend((transport, conv) for conv in convs)
            except Exception as e:
                error(f"{transport.name}: {'notification' if notifications else 'owner'}_conversations failed: {e}")
        return targets

    def send_message(self, text: str, disable_notification: bool = False,
                     disable_web_page_preview: bool = False, in_history: bool = False) -> bool:
        if not text or not text.strip():
            trace("Empty message text, skipping chat send.")
            return False
        targets = self._targets(notifications=True)
        skipped = []
        if in_history:
            skipped = [t.name for t, _ in targets if t.shows_history]
            targets = [(t, conv) for t, conv in targets if not t.shows_history]
            if skipped:
                trace(f"Message is a history row already, not pushed to {', '.join(skipped)}")
        if not targets:
            if skipped:
                trace("Nothing to push: every chat draws the history row itself.")
                return True
            trace("No chat conversation to send to, message dropped.")
            return False
        accepted = self._send_text(targets, text, silent=disable_notification, preview=not disable_web_page_preview)
        if accepted:
            Telemetry.get_instance().assistant_turn(text, "notifier", proactive=True)
        return accepted

    def _send_text(self, targets: list[tuple[ChatTransport, Any]], text: str, *, silent: bool, preview: bool) -> bool:
        accepted = False
        for transport, conv in targets:
            chunks = RichText.split(text, transport.max_message_length) if transport.max_message_length else [text]
            for chunk in chunks:
                try:
                    accepted = transport.send_text(conv, chunk, silent=silent, preview=preview) or accepted
                except Exception as e:
                    error(f"{transport.name} [{conv}]: send_text failed: {e}")
        return accepted

    def _origin(self, source: Optional[str], conversation: Any, *,
                owners_only: bool = False) -> list[tuple[ChatTransport, Any]]:
        """The (transport, conversation) a request came from, when a registered transport
        carries that source; empty for requests from outside the chat layer. With
        *owners_only* the conversation must also be one the transport names as an owner's:
        a question that gates an action (an approval, a secret) is the owner's to answer,
        never the requester's — a bot any chat may talk to must not let a stranger approve
        their own send_email — so a non-owner origin is empty too, and the owner chats are
        asked instead."""
        if source is None or conversation is None:
            return []
        origin = [(transport, conversation) for transport in self.transports if transport.source == source]
        if not origin:
            trace(f"No chat transport registered for source {source!r}, every owner chat gets it instead.")
        elif owners_only:
            origin = [(t, c) for t, c in origin if c in self._owners_of(t)]
            if not origin:
                info(f"{source} chat {conversation} is not an owner conversation — the owner chats are asked instead.")
        return origin

    def _tell_origin(self, source: Optional[str], conversation: Any, what: str,
                     asked: list[tuple[ChatTransport, Any]]) -> None:
        """The question of a chat's own turn was shown in other chats (the conversation is not
        an owner's): its transport is told, so it can say the turn is waiting there
        (`ChatTransport.asked_elsewhere`). Nothing for a request from outside the chat layer."""
        if source is None or conversation is None:
            return
        where = list(dict.fromkeys(t.name for t, _ in asked))
        for transport in self.transports:
            if transport.source != source or (transport, conversation) in asked:
                continue
            try:
                transport.asked_elsewhere(conversation, what, where)
            except Exception as e:
                error(f"{transport.name} [{conversation}]: asked_elsewhere failed: {e}")

    @staticmethod
    def _owners_of(transport: ChatTransport) -> list:
        try:
            return list(transport.owner_conversations())
        except Exception as e:
            error(f"{transport.name}: owner_conversations failed: {e}")
            return []

    def send_command_notification(self, text: str, ephemeral: bool = False,
                                  source: Optional[str] = None, conversation: Any = None) -> Any:
        """Echo a command as it starts. It belongs to the request that ran it, so it goes back
        to that request's own chat (a TUI command must not post into Telegram, and a Telegram
        status line must not linger on the console); only a request from outside the chat
        layer is echoed to every owner conversation. Returns the handle
        `complete_command_notification` takes when the command ends (None for the
        ephemeral status line, which the reply clears itself)."""
        if not text or not text.strip():
            return None
        targets = self._origin(source, conversation) or self._targets()
        if not targets:
            trace(f"No chat conversation for the command echo {text!r}, dropped.")
            return None
        if ephemeral:
            body = RichText.command_line(text)
            for transport, conv in targets:
                try:
                    transport.set_status_line(conv, body)
                except Exception as e:
                    error(f"{transport.name} [{conv}]: set_status_line failed: {e}")
            return None
        echo = _CommandEcho(text)
        for transport, conv in targets:
            try:
                echo.refs.append((transport, conv, transport.send_command_line(conv, text)))
            except Exception as e:
                error(f"{transport.name} [{conv}]: send_command_line failed: {e}")
        with self._lock:
            self._running_commands.append(echo)
        return echo

    def complete_command_notification(self, handle: Any, detail: str) -> None:
        """The command behind `handle` ended: annotate its echo and stop counting it."""
        if not isinstance(handle, _CommandEcho):
            if handle is not None:
                trace(f"Ignoring an unknown command echo handle {handle!r}")
            return
        with self._lock:
            if handle in self._running_commands:
                self._running_commands.remove(handle)
            for transport, conv, _ in handle.refs:
                self._last_command_end[(id(transport), conv)] = time.time()
        for transport, conv, ref in handle.refs:
            try:
                transport.finish_command_line(conv, ref, handle.text, detail)
            except Exception as e:
                error(f"{transport.name} [{conv}]: finish_command_line failed: {e}")

    def command_activity(self, transport: ChatTransport, conversation: Any) -> tuple[int, float]:
        """What the commands echoed into a conversation are up to: (how many are running,
        when the earliest of them started — or, with none running, when the last one
        ended; 0.0 when no command has run yet). Drives the session's working line."""
        with self._lock:
            now = time.time()
            stale = [e for e in self._running_commands if now - e.started > self.STALE_COMMAND_SECONDS]
            for echo in stale:
                # A command that never reported back must not show "Running command…" forever.
                trace(f"Chat hub: forgetting the command echo {echo.text!r}, no completion in "
                      f"{self.STALE_COMMAND_SECONDS:.0f}s")
                self._running_commands.remove(echo)
            mine = [e for e in self._running_commands
                    if any(t is transport and c == conversation for t, c, _ in e.refs)]
            if mine:
                return len(mine), min(e.started for e in mine)
            return 0, self._last_command_end.get((id(transport), conversation), 0.0)

    def send_photo(self, photo_bytes: bytes, caption: str = "",
                   source: Optional[str] = None, conversation: Any = None) -> bool:
        return self._fan_out("send_photo", lambda t, c: t.send_photo(c, photo_bytes, caption), source, conversation)

    def send_photo_album(self, photos: list[tuple[bytes, str]],
                         source: Optional[str] = None, conversation: Any = None) -> bool:
        return self._fan_out("send_photo_album", lambda t, c: t.send_photo_album(c, photos), source, conversation)

    def send_video(self, video_bytes: bytes, caption: str = "", thumbnail: Optional[bytes] = None,
                   source: Optional[str] = None, conversation: Any = None) -> bool:
        return self._fan_out("send_video", lambda t, c: t.send_video(c, video_bytes, caption, thumbnail),
                             source, conversation)

    def _media_targets(self, source: Optional[str], conversation: Any) -> list[tuple[ChatTransport, Any]]:
        """Media answering a request goes back to the chat it came from, or to every owner
        chat when no transport carries its source (a voice request has no chat of its own);
        media with no request behind it (the autonomous check) is a proactive push and takes
        the transports' notification route, telegram_bot.notify_chat_ids."""
        if source is None:
            return self._targets(notifications=True)
        return self._origin(source, conversation) or self._targets()

    def _fan_out(self, what: str, send, source: Optional[str] = None, conversation: Any = None) -> bool:
        targets = self._media_targets(source, conversation)
        if not targets:
            trace(f"No chat conversation for {what}, dropped.")
            return False
        accepted = False
        for transport, conv in targets:
            try:
                accepted = bool(send(transport, conv)) or accepted
            except Exception as e:
                error(f"{transport.name} [{conv}]: {what} failed: {e}")
        return accepted

    @staticmethod
    def markdown_to_html(md: str) -> str:
        return RichText.markdown_to_html(md)

    # --- approvals ------------------------------------------------------------------------

    def request_approval(self, title: str, details: str = "", timeout: Optional[float] = None,
                         source: Optional[str] = None, conversation: Any = None,
                         cancelled: Optional[Callable[[], bool]] = None) -> str:
        """Ask the chat the request came from (every owner conversation for a request from
        outside the chat layer), block until the first answer, the timeout, or — when no chat
        can show the question — return "unavailable" at once. While waiting, `cancelled()` is
        polled: once it says the request behind the question is gone, the prompts are closed
        and "cancelled" comes back, so a card is never approved into a dead request.
        Returns "approve", "deny", "timeout", "cancelled" or "unavailable". Safe to call from
        any thread."""
        origin = self._origin(source, conversation, owners_only=True)
        targets = origin or self._targets()
        if not targets:
            info(f"Approval request skipped (no chat conversation to ask): {title}")
            return DECISION_UNAVAILABLE

        wait_seconds = float(timeout) if timeout is not None else self._approval_timeout
        token = uuid.uuid4().hex
        entry = {"future": concurrent.futures.Future(), "prompts": []}
        with self._approvals_lock:
            self._pending_approvals[token] = entry
        where = f"{source} chat {conversation}" if origin else f"{len(targets)} owner chat(s)"
        info(f"Approval requested [{token[:8]}] in {where}: {title} (timeout={wait_seconds:.0f}s)")

        for transport, conv in targets:
            try:
                ref = transport.send_approval_prompt(conv, token, title, details)
            except Exception as e:
                error(f"{transport.name} [{conv}]: approval prompt failed: {e}")
                ref = None
            if ref is not None:
                entry["prompts"].append((transport, conv, ref))
        if not entry["prompts"]:
            with self._approvals_lock:
                self._pending_approvals.pop(token, None)
            info(f"Approval [{token[:8]}] could not be shown anywhere, answering 'unavailable'")
            return DECISION_UNAVAILABLE
        if not origin:
            self._tell_origin(source, conversation, "approval", [(t, c) for t, c, _ in entry["prompts"]])

        result = self._await_approval(entry, wait_seconds, cancelled)
        if result == DECISION_TIMEOUT:
            info(f"Approval [{token[:8]}] timed out after {wait_seconds:.0f}s")
            self._close_prompts(entry, "⏱ Timed out")
        elif result == DECISION_CANCELLED:
            info(f"Approval [{token[:8]}] cancelled: the request behind it is gone")
            self._close_prompts(entry, "🚫 Cancelled (the request was superseded)")
        with self._approvals_lock:
            self._pending_approvals.pop(token, None)
        return result

    def _await_approval(self, entry: dict, wait_seconds: float,
                        cancelled: Optional[Callable[[], bool]]) -> str:
        """The user's answer, "timeout", or "cancelled" the moment `cancelled()` turns true;
        the answer wins when both arrive. The decision is sealed under the lock so a late tap
        cannot set_result on a future already given up on."""
        deadline = time.time() + wait_seconds
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                return entry["future"].result(timeout=min(remaining, self.APPROVAL_POLL_SECONDS))
            except concurrent.futures.TimeoutError:
                pass
            if cancelled is not None and cancelled():
                with self._approvals_lock:
                    if entry["future"].done():
                        return entry["future"].result()
                    entry["future"].set_result(DECISION_CANCELLED)
                return DECISION_CANCELLED
        with self._approvals_lock:
            if entry["future"].done():
                return entry["future"].result()
            entry["future"].set_result(DECISION_TIMEOUT)
        return DECISION_TIMEOUT

    def resolve_approval(self, token: str, decision: str, who: str = "user") -> bool:
        """A transport reports the user's answer. False when the request is unknown or
        already answered (the transport tells the user "already resolved")."""
        with self._approvals_lock:
            entry = self._pending_approvals.get(token)
            if entry is None or entry["future"].done():
                info(f"Approval [{token[:8]}] {decision} by {who} ignored: not pending")
                return False
            # Under the lock: two chats answering at once must not both pass the done()
            # check and race into set_result (InvalidStateError for the loser).
            entry["future"].set_result(decision)
        marker = "✅ Approved" if decision == DECISION_APPROVE else "❌ Denied"
        info(f"Approval [{token[:8]}] {decision} by {who}")
        self._close_prompts(entry, f"{marker} by {escape(who)}")
        return True

    @staticmethod
    def _close_prompts(entry: dict, outcome_text: str):
        for transport, conv, ref in entry["prompts"]:
            try:
                transport.close_approval_prompt(conv, ref, outcome_text)
            except Exception as e:
                trace(f"{transport.name} [{conv}]: closing approval prompt failed: {e}")

    def pending_approvals(self) -> int:
        with self._approvals_lock:
            return len(self._pending_approvals)

    # --- secrets --------------------------------------------------------------------------

    def request_secret(self, title: str, details: str, accept: Callable[[str], str],
                       timeout: Optional[float] = None,
                       source: Optional[str] = None, conversation: Any = None) -> str:
        """Open a masked prompt for one value in the chat the request came from (every
        owner conversation for a request from outside the chat layer) and block until
        `accept` took an answer, the user cancelled, every attempt was rejected, or the
        timeout; "unavailable" at once when no chat can show the prompt or one is already
        open in one of those chats. See `Notifier.request_secret` for the `accept` contract.
        Safe to call from any thread; the value never passes through here as anything but
        an argument to `accept`."""
        targets = self._origin(source, conversation, owners_only=True) or self._targets()
        if not targets:
            info(f"Secret request skipped (no chat conversation to ask): {title}")
            return DECISION_UNAVAILABLE

        wait_seconds = float(timeout) if timeout is not None else self._secret_timeout
        token = uuid.uuid4().hex
        entry = {"token": token, "future": concurrent.futures.Future(), "prompts": [], "targets": targets,
                 "accept": accept, "attempts": 0, "claiming": False, "title": title, "details": details}
        with self._approvals_lock:
            for transport, conv in targets:
                if self._pending_secret_at(transport, conv) is not None:
                    info(f"Secret request refused: another one is still open in {transport.name} [{conv}]: {title}")
                    return DECISION_UNAVAILABLE
            self._pending_secrets[token] = entry
        info(f"Secret requested [{token[:8]}]: {title} → {len(targets)} chat(s), timeout {wait_seconds:.0f}s")

        if not self._show_secret_prompts(entry, details):
            with self._approvals_lock:
                self._pending_secrets.pop(token, None)
            info(f"Secret [{token[:8]}] could not be asked anywhere, answering 'unavailable'")
            return DECISION_UNAVAILABLE
        with self._approvals_lock:
            asked = [(t, c) for t, c, _ in entry["prompts"]]
        self._tell_origin(source, conversation, "secret", asked)

        try:
            result = entry["future"].result(timeout=wait_seconds)
        except concurrent.futures.TimeoutError:
            with self._approvals_lock:
                # An answer landing in this instant wins over the timeout.
                if not entry["future"].done():
                    entry["future"].set_result(DECISION_TIMEOUT)
            result = entry["future"].result()
            if result == DECISION_TIMEOUT:
                info(f"Secret [{token[:8]}] timed out after {wait_seconds:.0f}s")
                self._close_secret_prompts(entry, "⏱ Timed out")
        with self._approvals_lock:
            self._pending_secrets.pop(token, None)
        return result

    def claim_secret(self, transport: ChatTransport, conversation, value: str, who: str = "user") -> bool:
        """A transport offers an incoming text: False when no secret prompt is open in that
        conversation (the text is ordinary chat), True when this call consumed it — the
        transport must then drop the text. The value goes to `accept` and nowhere else."""
        with self._approvals_lock:
            entry = self._pending_secret_at(transport, conversation)
            if entry is None:
                return False
            token = entry["token"]
            if entry["future"].done() or entry["claiming"]:
                info(f"Secret [{token[:8]}]: extra answer from {transport.name} [{conversation}] dropped "
                     f"(already being handled)")
                return True
            entry["claiming"] = True
            entry["attempts"] += 1
            attempt = entry["attempts"]
        try:
            self._take_secret(entry, transport, conversation, value, who, attempt)
        finally:
            with self._approvals_lock:
                entry["claiming"] = False
        return True

    def _take_secret(self, entry: dict, transport: ChatTransport, conversation, value: str, who: str, attempt: int):
        token = entry["token"]
        try:
            outcome = entry["accept"](value)
        except SecretRejected as e:
            reason = str(e)
            if attempt >= self.SECRET_ATTEMPTS:
                info(f"Secret [{token[:8]}] rejected on attempt {attempt}/{self.SECRET_ATTEMPTS} ({reason}), giving up")
                self._finish_secret(entry, DECISION_REJECTED, f"⚠ {escape(reason)} — giving up")
                return
            info(f"Secret [{token[:8]}] rejected ({reason}), asking again ({attempt + 1}/{self.SECRET_ATTEMPTS})")
            # Same token, new details: a transport with that prompt still open updates it in
            # place, so the chat keeps one prompt per request and nothing lands below it.
            details = f"{entry['details']}\n⚠ {reason} — try again ({attempt + 1}/{self.SECRET_ATTEMPTS})".strip()
            if not self._show_secret_prompts(entry, details):
                info(f"Secret [{token[:8]}] could not be asked again anywhere, answering 'unavailable'")
                self._finish_secret(entry, DECISION_UNAVAILABLE, "⚠ Could not ask again")
            return
        except Exception as e:
            # The type only: an exception text could quote the value it choked on.
            error(f"Secret [{token[:8]}]: accept raised {type(e).__name__}, giving up")
            self._finish_secret(entry, DECISION_REJECTED, "⚠ Could not store the value")
            return
        info(f"Secret [{token[:8]}] accepted from {transport.name} [{conversation}] by {who} (attempt {attempt})")
        self._finish_secret(entry, DECISION_ACCEPTED, f"✅ {escape(outcome)}")

    def cancel_secret(self, token: str, who: str = "user") -> bool:
        """The user dismissed the prompt. False when the request is unknown or already over."""
        with self._approvals_lock:
            entry = self._pending_secrets.get(token)
            if entry is None or entry["future"].done():
                info(f"Secret [{token[:8]}] cancel by {who} ignored: not pending")
                return False
        info(f"Secret [{token[:8]}] cancelled by {who}")
        self._finish_secret(entry, DECISION_CANCEL, f"✖ Cancelled by {escape(who)}")
        return True

    def pending_secrets(self) -> int:
        with self._approvals_lock:
            return len(self._pending_secrets)

    def _pending_secret_at(self, transport: ChatTransport, conversation) -> Optional[dict]:
        # Caller holds _approvals_lock.
        for entry in self._pending_secrets.values():
            if (transport, conversation) in entry["targets"] and not entry["future"].done():
                return entry
        return None

    def _finish_secret(self, entry: dict, decision: str, outcome_text: str):
        with self._approvals_lock:
            if entry["future"].done():
                return
            entry["future"].set_result(decision)
        self._close_secret_prompts(entry, outcome_text)

    def _show_secret_prompts(self, entry: dict, details: str) -> bool:
        prompts = []
        for transport, conv in entry["targets"]:
            try:
                ref = transport.send_secret_prompt(conv, entry["token"], entry["title"], details)
            except Exception as e:
                error(f"{transport.name} [{conv}]: secret prompt failed: {e}")
                ref = None
            if ref is not None:
                prompts.append((transport, conv, ref))
        with self._approvals_lock:
            entry["prompts"] = prompts
            result = entry["future"].result() if entry["future"].done() else None
        if result is not None:
            # The request ended (timeout, or a cancel in another chat) while these went out.
            self._close_secret_prompts(entry, "⏱ Timed out" if result == DECISION_TIMEOUT else "✖ Cancelled")
        return bool(prompts)

    def _close_secret_prompts(self, entry: dict, outcome_text: str):
        with self._approvals_lock:
            prompts, entry["prompts"] = entry["prompts"], []
        for transport, conv, ref in prompts:
            try:
                transport.close_secret_prompt(conv, ref, outcome_text)
            except Exception as e:
                trace(f"{transport.name} [{conv}]: closing secret prompt failed: {e}")
