import asyncio
import concurrent.futures
import os
import queue
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

from assistant.integration.Feed import Feed, FeedBatch, FeedEntry
from assistant.integration.Integration import Integration
from assistant import GroupWake
from assistant.ChannelProfile import SOURCE_TELEGRAM
from assistant.GroupWake import BOT_WAKE_CAP, HOW, REFUSED, ROW, TURN
from assistant.Notifier import SecretRejected, notifier
from common.logging import info, trace, error, fine
from assistant.PromptInjectionDetector import is_prompt_injection
from common.ConfigManager import config
from common.DraftPlaceholder import is_draft_placeholder
from common.Paths import AISERVER_DIR
from common.RateLimiter import RateLimiter
from assistant.tools.MediaAnalyzer import media_analyzer, KIND_VIDEO


SESSION_FILE = str(AISERVER_DIR / 'telegram_feed')
CATCH_UP_DIALOGS = 30       # the most recent dialogs a catch-up walks
POLL_INTERVAL = 2           # seconds between drains of the live event queue
LOOP_CALL_TIMEOUT = 15
DIRECTORY_DIALOGS = 300     # the dialogs a recipient lookup walks, with the contact list beside them
DIRECTORY_TTL = 300         # seconds a built directory serves lookups (the approval card and the send share one)
SEND_RATE_LIMIT_WINDOW = 60.0
SEND_RATE_LIMIT_MAX = 5
LOGOUT_TIMEOUT = 30         # seconds for //wipe's log-out, connect included
LOGIN_HINT = "sign the account in with telegram_login(phone) from the console chat, or in the console's Telegram tab"
CODE_REUSE_NOTE = ("Asked again within a few minutes, Telegram repeats the same code without sending a new message — "
                   "wait for that one before requesting another.")


class TelegramFeedIntegration(Integration, Feed):
    """The owner's own Telegram account: a feed of its chats, plus sending.

    Follows the personal Telegram account, so the messages of its chats land in Keep's history,
    as a feed, and sends messages from it (`send_telegram`). Signing in takes the phone number,
    the code Telegram sends and the account's 2FA password.

    The account through Telethon as a Feed (plans/FEED_PLAN.md): a catch-up walks the recent
    dialogs for messages newer than the cursor (a timestamp); live, the NewMessage handler
    queues entries that `fetch` drains. The login runs through the auth methods below: `login`
    drives them from the chat's masked prompt, the TUI tab step by step. The login client lives
    on an in-memory session and the session file is written only once the sign-in completes —
    Telethon writes a file-backed session the moment its client connects, so a login on the file
    itself made `needs_2fa()` false from the first code request on: the tab said "Connected" and
    the feed opened an unauthorized session."""
    sensitive = True
    login_hint = "sign the account in with telegram_login(phone) from the console chat"

    poll_interval = POLL_INTERVAL
    item_noun = "messages"

    def __init__(self):
        super().__init__()
        self._api_id: int = config.get('telegram_feed.api_id', 0)
        self._api_hash: str = config.get('telegram_feed.api_hash', '')
        self._blacklist: list[str] = [kw.lower() for kw in config.get('telegram_feed.blacklist', [])]
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._client = None
        self._login_client = None
        self._me = None
        self._connected_name: Optional[str] = None
        self._auth_phone_hash: Optional[str] = None
        self._open_error: Optional[str] = None
        self._login_lock = threading.Lock()
        self._loop_lock = threading.Lock()
        self._live: "queue.Queue[FeedEntry]" = queue.Queue()
        self._reconnect: Optional[concurrent.futures.Future] = None
        self._rate_limiter = RateLimiter(SEND_RATE_LIMIT_MAX, SEND_RATE_LIMIT_WINDOW)
        self._directory: dict[str, dict[int, tuple]] = {}   # lowercase name → {entity id: (entity, display)}
        self._directory_built: float = 0.0
        # The group wake rule (assistant/GroupWake.py, shared with the iMessage chat): who wrote
        # last in each group the bot is in, and how often other bots woke the assistant there.
        self._wake = GroupWake.GroupWake()
        self._last_sender: dict[int, int] = self._wake.last_sender   # group chat id → sender of its last live message
        self._bot_bubbles: dict[int, tuple[int, str]] = {}   # group chat id → the bot's latest bubble (id, text)

    def _on_reset(self):
        self.shut_down()

    def shut_down(self):
        """Stop the feed thread and let every connection go."""
        from assistant.integration.FeedRunner import Feeds
        Feeds.get_instance().stop(self.feed_name)
        self._shutdown_loop()

    @classmethod
    def wipe_targets(cls) -> list[tuple[Path, str]]:
        session = Path(SESSION_FILE + '.session')
        return [(session, "Telegram feed login (Telethon session, logged out on Telegram first)"),
                (session.with_name(session.name + "-journal"), "Telegram feed session journal")]

    @classmethod
    def log_out(cls) -> str:
        """`//wipe`: terminate the session on Telegram's side before the file goes — deleting
        the file alone leaves the session authorized on the account (six orphaned Floxy
        sessions had piled up by 2026-09-16, and every wipe backup held a live credential).
        The running instance is shut down first so the file is free, then a client of its own
        logs out; Telethon deletes the file on success. Returns what happened, for the wipe's
        listing."""
        if not os.path.exists(SESSION_FILE + '.session'):
            return "no session file"
        api_id, api_hash = config.get('telegram_feed.api_id', 0), config.get('telegram_feed.api_hash', '')
        if not api_id or not api_hash:
            info("TelegramFeed: not logging out on Telegram — api_id / api_hash not configured")
            return "not logged out on Telegram: telegram_feed.api_id / api_hash not configured"
        instance = cls.peek_instance()
        if instance is not None:
            instance.shut_down()

        async def go() -> str:
            from telethon import TelegramClient
            client = TelegramClient(SESSION_FILE, api_id, api_hash)
            await client.connect()
            try:
                if not await client.is_user_authorized():
                    return "the session was not authorized on Telegram, nothing to log out"
                if not await client.log_out():
                    return "not logged out: Telegram refused the log-out"
                return "logged out on Telegram"
            finally:
                if client.is_connected():
                    await client.disconnect()

        try:
            outcome = asyncio.run(asyncio.wait_for(go(), LOGOUT_TIMEOUT))
        except Exception as e:
            error(f"TelegramFeed: log-out failed: {type(e).__name__}: {e}")
            return f"not logged out on Telegram ({type(e).__name__}: {e})"
        info(f"TelegramFeed: {outcome}")
        return outcome

    @property
    def name(self) -> str:
        return "Telegram Feed"

    @property
    def config_keys(self) -> List[str]:
        return ["telegram_feed.api_id", "telegram_feed.api_hash"]

    def check_health(self) -> bool:
        if self._client and self._client.is_connected():
            return True
        return False

    def needs_2fa(self) -> bool:
        return not os.path.exists(SESSION_FILE + '.session')

    # --- Feed ---------------------------------------------------------------------------------

    @property
    def feed_name(self) -> str:
        return "Telegram"

    @property
    def config_slug(self) -> str:
        return "telegram_feed"

    @property
    def open_error(self) -> Optional[str]:
        return self._open_error

    def open_feed(self) -> bool:
        if not self._api_id or not self._api_hash:
            self._open_error = "telegram_feed.api_id / api_hash not configured"
            trace(f"TelegramFeed: {self._open_error}")
            return False
        if self.needs_2fa():
            self._open_error = f"no session file — {LOGIN_HINT}"
            info(f"TelegramFeed: {self._open_error}")
            return False
        self._ensure_client()
        authorized = self._call(self._connect_and_identify())
        if not authorized:
            self._open_error = f"session exists but is not authorized — {LOGIN_HINT}"
            info(f"TelegramFeed: {self._open_error}")
            self._disconnect(self._client)
            self._client = None
            return False
        self._open_error = None
        self._call(self._register_handler())
        info(f"TelegramFeed connected as {self._connected_name}")
        return True

    def close_feed(self):
        self._shutdown_loop()

    def initial_cursor(self, horizon_ts: float) -> Optional[str]:
        return repr(float(horizon_ts))

    def fetch(self, cursor: Optional[str], backfill: bool) -> FeedBatch:
        since = float(cursor) if cursor else 0.0
        self._reconnect_if_dropped()
        if backfill:
            entries = self._call(self._catch_up(since), timeout=None)
        else:
            entries = self._drain_live()
        newest = max([since] + [e.timestamp for e in entries])
        return FeedBatch(entries, cursor=repr(newest), exhausted=True)

    async def _connect_and_identify(self) -> bool:
        if not self._client.is_connected():
            await self._client.connect()
        if not await self._client.is_user_authorized():
            return False
        self._me = await self._client.get_me()
        self._connected_name = self._me.first_name or self._me.username or str(self._me.id)
        return True

    async def _register_handler(self):
        from telethon import events

        @self._client.on(events.NewMessage)
        async def on_new_message(event):
            await self._handle_message(event)

        @self._client.on(events.MessageEdited)
        async def on_message_edited(event):
            self._note_bot_edit(event)

        fine("TelegramFeed: event handlers registered, listening for messages")

    async def _catch_up(self, since: float) -> list[FeedEntry]:
        """Messages newer than `since` across the most recent dialogs, newest dialog first;
        media stays a placeholder (a months-long catch-up must not download it all)."""
        from telethon.tl.types import MessageService
        entries: list[FeedEntry] = []
        dialogs = await self._client.get_dialogs(limit=CATCH_UP_DIALOGS)
        for dialog in dialogs:
            if self._is_own_bot(dialog.entity):
                fine("TelegramFeed: catch-up skips the chat with Floxy's own bot")
                continue
            try:
                async for msg in self._client.iter_messages(dialog.entity):
                    if isinstance(msg, MessageService):
                        continue
                    ts = msg.date.timestamp()
                    if ts <= since:
                        break
                    if not msg.text and not self._media_kind(msg):
                        continue
                    content = await self._format_message(msg, self._me, dialog, analyze_media=False)
                    if content:
                        entries.append(FeedEntry(ts, content))
            except Exception as e:
                fine(f"TelegramFeed: catch-up error for dialog {dialog.name}: {e}")
        if entries:
            info(f"TelegramFeed: catch-up found {len(entries)} messages in {len(dialogs)} dialogs since "
                 f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(since))}")
        else:
            fine(f"TelegramFeed: catch-up found nothing new in {len(dialogs)} dialogs")
        return entries

    def _reconnect_if_dropped(self):
        """A client Telethon gave up on is reconnected on the feed's loop, one attempt at a
        time, and `fetch` goes on with what the queue holds meanwhile. The Oct 29 outage
        outlasted Telethon's reconnect retries: the client stayed disconnected for 4.5 days
        while the feed reported live and every health check failed."""
        client = self._client
        if client is None or client.is_connected():
            return
        if self._reconnect is not None and not self._reconnect.done():
            trace("TelegramFeed: the client is disconnected, a reconnect is still under way")
            return
        info("TelegramFeed: the client is disconnected — reconnecting")

        async def go():
            await client.connect()
            info("TelegramFeed: reconnected")

        self._reconnect = asyncio.run_coroutine_threadsafe(go(), self._get_or_create_loop())
        self._reconnect.add_done_callback(self._reconnect_done)

    @staticmethod
    def _reconnect_done(future):
        e = future.exception() if not future.cancelled() else None
        if e is not None:
            error(f"TelegramFeed: reconnect failed, trying again at the next fetch: {type(e).__name__}: {e}")

    def _drain_live(self) -> list[FeedEntry]:
        entries: list[FeedEntry] = []
        while True:
            try:
                entries.append(self._live.get_nowait())
            except queue.Empty:
                return entries

    async def _handle_message(self, event):
        """Handle a real-time incoming/outgoing message: format it and queue it for `fetch`."""
        from telethon.tl.types import MessageService

        msg = event.message
        if isinstance(msg, MessageService):
            return
        if not msg.text and not self._media_kind(msg):
            return

        try:
            chat = await event.get_chat()

            # Determine dialog-like info
            class DialogProxy:
                def __init__(self, c):
                    self.entity = c
                    self.is_group = hasattr(c, 'megagroup') or hasattr(c, 'participants_count')
                    self.name = getattr(c, 'title', None) or ''

            dialog = DialogProxy(chat)
            if await self._wake_assistant(msg, chat):
                return   # a turn of the dialog now, not a context row as well
            content = await self._format_message(msg, self._me, dialog)

            if content:
                trace(content)
                self._live.put(FeedEntry(msg.date.timestamp(), content))
        except Exception as e:
            trace(f"TelegramFeed: error handling message: {e}")

    # --- waking the assistant from a group ----------------------------------------------------

    @staticmethod
    def _bot_chat_ids() -> list[int]:
        ids = []
        for item in config.get('telegram_bot.allowed_chat_ids', []) or []:
            try:
                ids.append(int(str(item).strip()))
            except ValueError:
                pass
        return ids

    @staticmethod
    def _bot_user_id() -> Optional[int]:
        """The bot's own user id — the prefix of its token, no API call needed."""
        prefix = str(config.get('telegram_bot.bot_token', '') or '').partition(':')[0]
        return int(prefix) if prefix.isdigit() else None

    @staticmethod
    def _names_assistant(text: str) -> bool:
        return GroupWake.names_assistant(text, GroupWake.assistant_names(config))

    def _bot_wake_allowed(self, chat_id: int) -> bool:
        return self._wake.brake(chat_id, BOT_WAKE_CAP)

    @staticmethod
    async def _replies_to_the_bot(msg, bot_id: Optional[int]) -> bool:
        """A reply to one of the bot's own bubbles reaches the bot through the Bot API even in
        privacy mode (TelegramInterface serves it), so the feed must not hand it over too.
        Only a person's reply, though: the Bot API never delivers another bot's messages, so
        another Keep replying to our bubble reaches us through the feed alone."""
        if bot_id is None or not getattr(msg, 'reply_to_msg_id', None):
            return False
        try:
            replied = await msg.get_reply_message()
        except Exception as e:
            trace(f"TelegramFeed: the replied-to message could not be read ({e}), treating it as not the bot's")
            return False
        return replied is not None and getattr(replied, 'sender_id', None) == bot_id

    async def _wake_assistant(self, msg, chat) -> bool:
        """A live message in a group the bot is in (a negative id on telegram_bot.allowed_chat_ids)
        that names the assistant, or that comes right after a message of the bot's own (people
        forget her name and the reply button when they answer her): handed to the chat layer
        as a turn of that group, so the reply goes out as the bot. The bot never receives such a
        message itself — in privacy mode Telegram delivers it only replies to its own bubbles —
        but the owner's account sees the whole group, other bots included. True when handed over.

        The decision is the group wake rule every chat platform shares (`assistant/GroupWake.py`):
        the name, the "right after the bot" rule, and the last sender a refused turn leaves in
        place. The bot's reply itself lands as an edit of her draft bubble, which
        `_note_bot_edit` records."""
        from telethon import utils
        try:
            chat_id = utils.get_peer_id(chat)   # the Bot API id: -<id> for a group, -100<id> for a supergroup
        except Exception as e:
            trace(f"TelegramFeed: no peer id for '{getattr(chat, 'title', chat)}' ({e}), a context row only")
            return False
        ids = self._bot_chat_ids()
        if chat_id >= 0 or chat_id not in ids:
            if self._names_assistant(msg.text or ""):
                trace(f"TelegramFeed: '{getattr(chat, 'title', chat_id)}' names the assistant but the bot is not in it")
            return False
        bot_id = self._bot_user_id()
        outcome = await self._hand_over(msg, chat, chat_id, ids, bot_id)
        self._wake.settle(chat_id, msg.sender_id, outcome)
        return outcome == TURN

    async def _hand_over(self, msg, chat, chat_id: int, ids: list[int], bot_id: Optional[int]) -> str:
        """`_wake_assistant`'s decision: TURN when handed over, ROW when the message stays a
        context row (or nothing at all), REFUSED when the chat layer would not take the turn."""
        text = msg.text or ""
        if bot_id is not None and msg.sender_id == bot_id:
            self._bot_bubbles[chat_id] = (msg.id, text)
            trace("TelegramFeed: the bot's own message in the group — not woken by herself")
            return ROW
        if not text:
            return ROW
        if self._is_draft_placeholder(msg):
            trace(f"TelegramFeed: {text!r} from {msg.sender_id} in '{getattr(chat, 'title', chat_id)}' is a "
                  f"streaming draft's placeholder (another assistant's bubble, its text arrives as an edit the "
                  f"feed never sees) — not woken")
            return ROW
        # A reply to the bot's bubble is no reason here: the bot is handed those itself (below).
        why = self._wake.reason(chat_id, text, bot_id, GroupWake.assistant_names(config))
        if not why:
            return ROW
        how = HOW[why]
        is_from_me = msg.sender_id == self._me.id
        if is_from_me:
            sender_name, is_bot, guest = config.get_owner_name(), False, False
        else:
            sender = await msg.get_sender()
            sender_name = (getattr(sender, 'first_name', None) or getattr(sender, 'username', None)
                           or str(msg.sender_id))
            is_bot = getattr(sender, 'bot', False) is True
            guest = msg.sender_id not in ids
            if is_prompt_injection(text):
                info(f"TelegramFeed: {sender_name} {how} in a group, but the message looks like a prompt "
                     f"injection — not woken")
                return ROW
        if not is_bot and await self._replies_to_the_bot(msg, bot_id):
            trace(f"TelegramFeed: {sender_name} replies to the bot's own bubble — the bot receives it itself, "
                  f"not woken twice")
            return ROW
        if is_bot and not self._bot_wake_allowed(chat_id):
            info(f"TelegramFeed: {sender_name} (a bot) {how}, but bots woke her {BOT_WAKE_CAP} times this "
                 f"hour in [{chat_id}] — not woken (loop brake)")
            return ROW
        info(f"TelegramFeed: {sender_name}{' (bot)' if is_bot else ''}{' (guest)' if guest else ''} {how} "
             f"in '{getattr(chat, 'title', None)}' [{chat_id}], waking her")
        taken = notifier().receive_chat_message(SOURCE_TELEGRAM, chat_id, sender_name, text, guest=guest,
                                                sender_is_bot=is_bot, chat_name=getattr(chat, 'title', None))
        return TURN if taken else REFUSED

    def _note_bot_edit(self, event):
        """The bot's reply is an edit of the draft bubble she posted first, so it never comes
        through `NewMessage`: an edit that changes the text of her latest bubble in a group makes
        her the group's last sender again. A reaction to the bubble arrives as an edit too, with
        the text unchanged, and moves nothing."""
        msg = event.message
        bot_id = self._bot_user_id()
        if bot_id is None or getattr(msg, 'sender_id', None) != bot_id:
            return
        chat_id = event.chat_id
        if chat_id is None or chat_id >= 0:
            return   # the owner's private chat with the bot: her streamed replies there, not a group's
        bubble = self._bot_bubbles.get(chat_id)
        if bubble is None or bubble[0] != msg.id:
            trace(f"TelegramFeed: [{chat_id}] the bot edited a bubble that is not her latest one — ignored")
            return
        text = msg.text or ""
        if text == bubble[1]:
            return
        self._bot_bubbles[chat_id] = (msg.id, text)
        self._wake.spoke(chat_id, bot_id)
        trace(f"TelegramFeed: [{chat_id}] the bot's bubble got its text — she is the last sender again")

    # --- sending ------------------------------------------------------------------------------

    @staticmethod
    def _display(entity) -> str:
        title = getattr(entity, 'title', None)
        if title:
            return title
        name = " ".join(filter(None, (getattr(entity, 'first_name', None), getattr(entity, 'last_name', None))))
        return name or getattr(entity, 'username', None) or str(getattr(entity, 'id', '?'))

    def _index(self, entity):
        """File *entity* under every name it answers to: full name, first name, title, @username."""
        display = self._display(entity)
        keys = {display.lower()}
        first = getattr(entity, 'first_name', None)
        if first:
            keys.add(first.lower())
        username = getattr(entity, 'username', None)
        if username:
            keys.add(username.lower())
            keys.add(f"@{username.lower()}")
        for key in keys:
            self._directory.setdefault(key, {})[entity.id] = (entity, display)

    async def _build_directory(self):
        from telethon.tl.functions.contacts import GetContactsRequest
        self._directory = {}
        dialogs = await self._client.get_dialogs(limit=DIRECTORY_DIALOGS)
        for dialog in dialogs:
            if not self._is_own_bot(dialog.entity):
                self._index(dialog.entity)
        contacts = 0
        try:
            for user in (await self._client(GetContactsRequest(hash=0))).users:
                self._index(user)
                contacts += 1
        except Exception as e:
            error(f"TelegramFeed: reading the contacts failed: {type(e).__name__}: {e}")
        self._directory_built = time.time()
        info(f"TelegramFeed: directory built — {len(dialogs)} dialogs, {contacts} contacts")

    async def _lookup(self, recipient: str):
        """(entity, display, error) for a name, a @username or a phone number (contacts only)."""
        candidate = " ".join(recipient.split())
        if candidate.startswith("+") and candidate[1:].replace(" ", "").isdigit():
            try:
                entity = await self._client.get_entity(candidate.replace(" ", ""))
            except Exception as e:
                return None, None, f"{candidate} is not one of the account's contacts ({type(e).__name__})"
            return entity, self._display(entity), None
        if time.time() - self._directory_built > DIRECTORY_TTL:
            await self._build_directory()
        matches = self._directory.get(candidate.lower(), {})
        if len(matches) == 1:
            entity, display = next(iter(matches.values()))
            info(f"TelegramFeed: resolved '{candidate}' → {display} ({entity.id})")
            return entity, display, None
        if len(matches) > 1:
            # The lookup ignores case, so the "is this the name typed" check must too — else two
            # chats titled Family asked for as "family" collapse to one option without a handle
            options = sorted({d if d.lower() != candidate.lower() else f"{d} (@{getattr(e, 'username', None) or e.id})"
                              for e, d in matches.values()})
            return None, None, f"'{candidate}' is ambiguous — say the full name or @username: {', '.join(options)}"
        return None, None, f"unknown recipient: {candidate} (not among the account's chats or contacts)"

    def resolve_recipient(self, recipient: str):
        """(entity, display name, error) for a contact's or chat's name, a @username or a number."""
        if not (recipient or "").strip():
            return None, None, "empty recipient"
        if not self.check_health():
            return None, None, f"Telegram Feed is not connected — {LOGIN_HINT}"
        return self._call(self._lookup(recipient))

    def send(self, recipient: str, body: str) -> str:
        """Send *body* from the owner's account. "" on success, else why not. The outgoing
        message comes back through the NewMessage handler, so the history gets it like any
        message the owner sends from the phone."""
        if not body or not body.strip():
            info("TelegramFeed send skipped: empty body")
            return "empty message body"
        rate_err = self._rate_limiter.check()
        if rate_err:
            info(f"TelegramFeed send blocked: {rate_err}")
            return rate_err
        entity, display, err = self.resolve_recipient(recipient)
        if err:
            info(f"TelegramFeed send: could not resolve recipient '{recipient}': {err}")
            return err
        try:
            self._call(self._client.send_message(entity, body))
        except Exception as e:
            error(f"TelegramFeed send to {display} failed: {type(e).__name__}: {e}")
            return f"Telegram send failed: {e}"
        self._rate_limiter.record()
        info(f"TelegramFeed: sent to {display}: {body[:80]}")
        return ""

    def _call(self, coroutine, timeout: Optional[float] = LOOP_CALL_TIMEOUT):
        """Run a coroutine on the client's loop from another thread and wait for it."""
        loop = self._get_or_create_loop()
        return asyncio.run_coroutine_threadsafe(coroutine, loop).result(timeout=timeout)

    def _disconnect(self, client):
        """Disconnect *client* on its loop. Telethon's `disconnect()` looks the loop up in the
        calling thread before it returns the coroutine, so called from a feed or worker
        thread it raises and the connection stays open; wrapped, it runs where the loop is."""
        async def go():
            await client.disconnect()

        try:
            self._call(go(), timeout=5)
        except Exception as e:
            trace(f"TelegramFeed: disconnect raised {type(e).__name__}: {e}")

    def _shutdown_loop(self):
        """Disconnect the clients, cancel Telethon's tasks and stop the loop thread."""
        for attr in ("_client", "_login_client"):
            client = getattr(self, attr)
            if client:
                self._disconnect(client)
                setattr(self, attr, None)
        self._connected_name = None
        if self._loop and self._loop.is_running():
            try:
                async def _cancel_all():
                    current = asyncio.current_task()
                    tasks = [t for t in asyncio.all_tasks(self._loop) if not t.done() and t is not current]
                    for t in tasks:
                        t.cancel()
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)

                asyncio.run_coroutine_threadsafe(_cancel_all(), self._loop).result(timeout=5)
            except Exception as e:
                trace(f"TelegramFeed: cancelling tasks raised {type(e).__name__}: {e}")
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop = None

    async def _format_message(self, msg, me, dialog, analyze_media: bool = True) -> Optional[str]:
        if self._is_own_bot(dialog.entity):
            fine("TelegramFeed: skipped a message in the chat with Floxy's own bot")
            return None
        if self._is_own_bot_message(msg):
            # In a group the feed sees the bot's streaming draft the moment it is posted — the
            # first spinner frame — and never the edit that gives it its text, nor the delete
            # of a suppressed reply; the history holds her real reply first hand anyway.
            fine("TelegramFeed: skipped a message of Floxy's own bot (its reply is in the history already)")
            return None
        if self._is_draft_placeholder(msg):
            trace(f"TelegramFeed: skipped {msg.text!r} from {msg.sender_id} — a streaming draft's placeholder "
                  f"(another assistant's bubble, no text yet), not a context row")
            return None

        is_from_me = msg.sender_id == me.id

        if is_from_me:
            sender_name = config.get_owner_name()
        else:
            try:
                sender = await msg.get_sender()
                sender_name = getattr(sender, 'first_name', None) or getattr(sender, 'username', None) or str(msg.sender_id)
            except Exception:
                sender_name = str(msg.sender_id)

        is_group = getattr(dialog, 'is_group', False) or bool(getattr(dialog.entity, 'title', None))
        group_name = getattr(dialog.entity, 'title', None) if is_group else None

        recipient_name = None
        if is_from_me and not is_group:
            recipient_name = getattr(dialog.entity, 'first_name', None) or getattr(dialog.entity, 'username', None)

        if self._is_blacklisted(sender_name, group_name, recipient_name):
            return None

        text = msg.text or ""
        if not is_from_me and text and is_prompt_injection(text):
            info(f"Telegram: blocked prompt injection from {sender_name}")
            return None

        photo_desc = ""
        kind = self._media_kind(msg)
        if kind and analyze_media:
            photo_desc = await self._analyze_media(msg, kind, sender_name)
        elif kind:
            photo_desc = self._placeholder(kind)

        if photo_desc and text:
            content = f"{photo_desc}\nText message: {text}"
        else:
            content = photo_desc or text
        if not content:
            return None

        if is_from_me:
            target = group_name or recipient_name
            if target:
                return f"Telegram sent:\n{sender_name} to {target}: {content}"
            return f"Telegram sent:\n{sender_name}: {content}"
        else:
            target = group_name or config.get_owner_name()
            return f"Telegram received:\nFrom {sender_name} to {target}: {content}"

    @staticmethod
    def _is_own_bot(entity) -> bool:
        """The chat with Floxy's own bot: that conversation is already in the history first
        hand. A bot token is `<bot user id>:<secret>` and a private chat's id is the peer's
        user id, so no lookup is needed; read live, the bot can be configured after the feed opened."""
        bot_id = TelegramFeedIntegration._bot_user_id()
        return bot_id is not None and getattr(entity, 'id', None) == bot_id

    @staticmethod
    def _is_own_bot_message(msg) -> bool:
        bot_id = TelegramFeedIntegration._bot_user_id()
        return bot_id is not None and getattr(msg, 'sender_id', None) == bot_id

    @staticmethod
    def _is_draft_placeholder(msg) -> bool:
        """Another assistant's streaming bubble as the feed sees it: the first spinner frame, posted
        before any text (the edit that fills it never reaches a NewMessage handler)."""
        return not TelegramFeedIntegration._media_kind(msg) and is_draft_placeholder(msg.text or "")

    @staticmethod
    def _media_kind(msg) -> Optional[str]:
        """'image' for a photo, 'video' for a video / GIF (mp4 animation) / round video note, else None."""
        if getattr(msg, 'photo', None):
            return "image"
        if getattr(msg, 'video', None) or getattr(msg, 'gif', None) or getattr(msg, 'video_note', None):
            return KIND_VIDEO
        return None

    @staticmethod
    def _placeholder(kind: str) -> str:
        return "[Video]" if kind == KIND_VIDEO else "[Photo]"

    async def _analyze_media(self, msg, kind: str, sender_name: str) -> str:
        label = "Video" if kind == KIND_VIDEO else "Photo"
        if kind == KIND_VIDEO:
            # Telethon pulls the whole file into memory and a personal account can receive
            # multi-GB clips, so anything the analyzer would refuse anyway is never fetched.
            size = getattr(getattr(msg, 'file', None), 'size', None)
            limit = media_analyzer().video_size_limit()
            if isinstance(size, int) and size > limit:
                info(f"TelegramFeed: video from {sender_name} is {size / 1024 / 1024:.1f} MB, over the "
                     f"{limit / 1024 / 1024:.0f} MB analysis cap, not downloading it")
                return self._placeholder(kind)
        try:
            media_bytes = await msg.download_media(file=bytes)
        except Exception as e:
            error(f"TelegramFeed: {label.lower()} download failed: {e}")
            return self._placeholder(kind)
        if not media_bytes:
            trace(f"TelegramFeed: failed to download {label.lower()}")
            return self._placeholder(kind)
        mime_type = getattr(getattr(msg, 'file', None), 'mime_type', None)
        if kind == KIND_VIDEO and not (mime_type or "").startswith("video/"):
            mime_type = "video/mp4"
        return media_analyzer().describe_bytes(media_bytes, mime_type=mime_type, fallback=self._placeholder(kind),
                                               context=f"{label} sent via Telegram by {sender_name}")

    def _is_blacklisted(self, sender_name: str, group_name: Optional[str],
                        recipient_name: Optional[str] = None) -> bool:
        if not self._blacklist:
            return False
        sender_lower = sender_name.lower()
        group_lower = group_name.lower() if group_name else ""
        recipient_lower = recipient_name.lower() if recipient_name else ""
        for kw in self._blacklist:
            if kw in sender_lower or (group_lower and kw in group_lower) or (recipient_lower and kw in recipient_lower):
                return True
        return False


    # --- The login as one conversation, for the chat (TelegramFeedCommands.cmd_telegram_login) ---

    def login(self, phone: str, ask: Callable[[str, str, Callable[[str], str]], str]) -> Optional[str]:
        """Signs the account in: sends the code to *phone*, asks for it through *ask* (the
        masked prompt: `ask(title, details, accept)` → "accepted" / "cancel" / "timeout" /
        "rejected" / "unavailable"), then for the 2FA password when the account has one.
        Returns None once the session file is written, else why it stopped. The caller
        restarts the integration so a new instance opens the session."""
        if not self._api_id or not self._api_hash:
            return "Telegram Feed login failed: telegram_feed.api_id / api_hash not configured"
        if not self._login_lock.acquire(blocking=False):
            info("TelegramFeed login: refused, another login is in progress")
            return "Telegram Feed login failed: another login is in progress (the console's Telegram tab?)"
        try:
            return self._login(phone, ask)
        finally:
            self.abort_login()
            self._login_lock.release()

    def _login(self, phone: str, ask) -> Optional[str]:
        from telethon import errors
        try:
            sent = self.send_code_request(phone)
        except errors.PhoneNumberInvalidError:
            info(f"TelegramFeed login: Telegram rejected the phone number {phone}")
            return (f"Telegram Feed login failed: Telegram rejected the phone number {phone} — "
                    "it needs the country code, e.g. +46701234567")
        except errors.FloodWaitError as e:
            info(f"TelegramFeed login: flood wait of {e.seconds}s on the code request")
            return f"Telegram Feed login failed: Telegram asks to wait {e.seconds} s before another code is sent"
        except Exception as e:
            error(f"TelegramFeed login: code request failed: {type(e).__name__}: {e}")
            return f"Telegram Feed login failed: could not send the code ({e})"
        info(f"TelegramFeed login: code sent to {phone}, asking for it in the chat")
        where = self.code_destination(sent)

        outcome: dict = {}

        def accept_code(value: str) -> str:
            code = value.strip().replace(" ", "")
            try:
                self.sign_in(phone, code)
            except errors.SessionPasswordNeededError:
                outcome["password"] = True
                return "Code accepted — the account has a two-step verification password"
            except errors.PhoneCodeInvalidError:
                raise SecretRejected("Telegram says that code is wrong")
            except errors.PhoneCodeExpiredError:
                outcome["error"] = "the code expired before it was entered — run telegram_login again for a new one"
                return "That code has expired"
            return "Signed in"

        decision = ask("Telegram login code", f"Telegram sent the code {where}. {CODE_REUSE_NOTE}", accept_code)
        if decision != "accepted":
            return self._login_ended(decision, "code")
        if outcome.get("error"):
            return f"Telegram Feed login failed: {outcome['error']}"

        if outcome.get("password"):
            info("TelegramFeed login: code accepted, the account has a 2FA password — asking for it")

            def accept_password(value: str) -> str:
                try:
                    self.sign_in_2fa(value)
                except errors.PasswordHashInvalidError:
                    raise SecretRejected("Telegram says that password is wrong")
                return "Signed in"

            decision = ask("Telegram two-step verification password",
                           "The account's two-step verification password (not the code).", accept_password)
            if decision != "accepted":
                return self._login_ended(decision, "password")

        if self.needs_2fa():
            error("TelegramFeed login: signed in but no session file was written")
            return "Telegram Feed login failed: signed in but no session file was written"
        info("TelegramFeed login: signed in, session file written")
        return None

    @staticmethod
    def _login_ended(decision: str, what: str) -> str:
        reasons = {
            "cancel": f"the user cancelled the {what} prompt",
            "timeout": f"the user did not enter the {what} in time",
            "rejected": f"the {what} was wrong every time",
            "unavailable": f"no chat could show the {what} prompt — it must be run from the console chat",
        }
        return f"Telegram Feed login failed: {reasons.get(decision, decision)} — nothing was signed in"

    # --- Auth methods, shared by `login` and the TUI tab ---

    def _ensure_client(self):
        """Create and connect the feed's Telethon client on the session file (blocking)."""
        if self._client and self._client.is_connected():
            return

        loop = self._get_or_create_loop()

        async def connect():
            from telethon import TelegramClient
            # Unlimited reconnect retries: with the default five Telethon gives up on an
            # outage of minutes and the feed is dead until a restart.
            self._client = TelegramClient(SESSION_FILE, self._api_id, self._api_hash, connection_retries=None)
            await self._client.connect()

        future = asyncio.run_coroutine_threadsafe(connect(), loop)
        future.result(timeout=15)

    def _ensure_login_client(self):
        """The client a login runs on: an in-memory session, so nothing reaches the disk
        until `_persist_login` writes the session file for a completed sign-in."""
        if self._login_client and self._login_client.is_connected():
            return

        loop = self._get_or_create_loop()

        async def connect():
            from telethon import TelegramClient
            from telethon.sessions import StringSession
            self._login_client = TelegramClient(StringSession(), self._api_id, self._api_hash)
            await self._login_client.connect()

        future = asyncio.run_coroutine_threadsafe(connect(), loop)
        future.result(timeout=15)

    def _get_or_create_loop(self) -> asyncio.AbstractEventLoop:
        with self._loop_lock:
            if self._loop and self._loop.is_running():
                return self._loop

            self._loop = asyncio.new_event_loop()
            t = threading.Thread(target=self._loop.run_forever, name="TelegramFeedLoop", daemon=True)
            t.start()
            return self._loop

    def send_code_request(self, phone: str):
        """Ask Telegram for a login code; returns Telethon's SentCode (`code_destination`
        words where it went)."""
        self._ensure_login_client()
        loop = self._get_or_create_loop()

        async def _send():
            result = await self._login_client.send_code_request(phone)
            self._auth_phone_hash = result.phone_code_hash
            return result

        future = asyncio.run_coroutine_threadsafe(_send(), loop)
        sent = future.result(timeout=30)
        info(f"TelegramFeed: code for {phone} sent {self.code_destination(sent)}"
             f"{self._resend_note(sent)}")
        return sent

    @staticmethod
    def code_destination(sent) -> str:
        """Where Telegram delivered the code, from the SentCode's type — the user looks in the
        wrong place otherwise (an SMS never comes when the app got it)."""
        kind = getattr(sent, "type", None)
        name = type(kind).__name__
        if name == "SentCodeTypeApp":
            return "to your Telegram app — a message from 'Telegram' on a device already signed in, not an SMS"
        if name in ("SentCodeTypeSms", "SentCodeTypeFirebaseSms"):
            return "by SMS"
        if name in ("SentCodeTypeSmsWord", "SentCodeTypeSmsPhrase"):
            return "by SMS, as a word or phrase rather than digits"
        if name == "SentCodeTypeCall":
            return "by a phone call that reads the code out"
        if name == "SentCodeTypeFlashCall":
            return "as a flash call — the code is the last digits of the calling number"
        if name == "SentCodeTypeMissedCall":
            return f"as a missed call from {getattr(kind, 'prefix', '')}… — the code is the last digits of the calling number"
        if name == "SentCodeTypeEmailCode":
            return f"to your login email {getattr(kind, 'email_pattern', '')}"
        if name == "SentCodeTypeFragmentSms":
            return f"by SMS through Fragment ({getattr(kind, 'url', '')})"
        if name == "SentCodeTypeSetUpEmailRequired":
            return "nowhere — Telegram wants a login email set up in the app for this account first"
        return f"({name})"

    @staticmethod
    def _resend_note(sent) -> str:
        parts = []
        timeout = getattr(sent, "timeout", None)
        if timeout:
            parts.append(f"valid for {timeout} s")
        next_type = getattr(sent, "next_type", None)
        if next_type is not None:
            parts.append(f"a resend would go by {type(next_type).__name__.replace('CodeType', '')}")
        return f" ({', '.join(parts)})" if parts else ""

    def sign_in(self, phone: str, code: str):
        """Sign in with the verification code; a completed sign-in writes the session file."""
        self._ensure_login_client()
        loop = self._get_or_create_loop()

        async def _sign_in():
            return await self._login_client.sign_in(phone, code, phone_code_hash=self._auth_phone_hash)

        future = asyncio.run_coroutine_threadsafe(_sign_in(), loop)
        user = future.result(timeout=30)
        self._persist_login()
        return user

    def sign_in_2fa(self, password: str):
        """Sign in with the 2FA password; a completed sign-in writes the session file."""
        self._ensure_login_client()
        loop = self._get_or_create_loop()

        async def _sign_in_2fa():
            return await self._login_client.sign_in(password=password)

        future = asyncio.run_coroutine_threadsafe(_sign_in_2fa(), loop)
        user = future.result(timeout=30)
        self._persist_login()
        return user

    def _persist_login(self):
        """Copy the signed-in login client's DC and auth key into the session file the feed
        opens, then let the login client go."""
        from telethon.sessions import SQLiteSession
        session = self._login_client.session
        stored = SQLiteSession(SESSION_FILE)
        try:
            stored.set_dc(session.dc_id, session.server_address, session.port)
            stored.auth_key = session.auth_key
            stored.save()
        finally:
            stored.close()
        self._open_error = None
        info(f"TelegramFeed: signed in, session file written (DC {session.dc_id})")
        self.abort_login()

    def abort_login(self):
        """Drop the login client, signed in or not: nothing of an unfinished login stays."""
        if self._login_client:
            self._disconnect(self._login_client)
            self._login_client = None
            self._auth_phone_hash = None

    def get_auth_status(self) -> dict:
        """The auth status for the TUI tab, from what the feed and the login already know —
        no connection of its own, which raced the feed thread on the same loop."""
        if not self._api_id or not self._api_hash:
            return {"status": "error", "message": "telegram_feed.api_id / api_hash not configured"}

        if self._login_client:
            return {"status": "needs_auth", "message": "Login in progress — waiting for the code"}

        if self.needs_2fa():
            return {"status": "needs_auth", "message": "No session file — press Ctrl+P to authenticate"}

        if self._connected_name:
            return {"status": "authenticated", "message": f"Connected as {self._connected_name}"}

        if self._open_error and "not authorized" in self._open_error:
            return {"status": "needs_auth", "message": "Session expired or revoked — press Ctrl+P to re-authenticate"}

        if self._open_error:
            return {"status": "error", "message": self._open_error}

        return {"status": "connecting", "message": "Connecting..."}

    def restart_feed(self):
        """Start the feed after a login from the TUI tab: a full restart_integration, so the
        new instance opens the session file the login just wrote."""
        if self.needs_2fa():
            return
        from assistant.integration.IntegrationMonitor import IntegrationMonitor
        IntegrationMonitor.get_instance().restart(self.name)
