"""The Telegram chat transport: a python-telegram-bot application on its own event loop,
reduced to what only Telegram knows — parsing updates into `IncomingMessage`s, and rendering
what the shared chat layer asks for (drafts with the █/░ spinner and throttled edits, the
ephemeral status line, inline approve/deny keyboards, secret prompts, one-shot sends) within
Telegram's flood limits. Everything about *what* to say lives in `interfaces/chat`.

A secret prompt is one message with a Cancel button, edited in place when the value is
refused and asked again, and reduced to its title and outcome when it is over; the next text
the user sends in that chat is the value. It is handed to the hub off the loop thread and
deleted from the chat at once — never logged, never cached for reply quotes, never forwarded
as chat.

Threading: the bot loop runs on the "TelegramBot" thread. The chat layer calls in from
worker threads; `_call()` runs a coroutine on the loop and waits, `_schedule()` fires and
forgets. Handlers on the loop only ever `hub.receive()` (never blocks) and fire-and-forget.
"""
import asyncio
import threading
import time
from html import escape
from typing import Any, List, Optional

import requests
from telegram.error import RetryAfter

from assistant.ChannelProfile import SOURCE_TELEGRAM
from assistant.integration.Integration import Integration
from common import ConfigStore
from common.ConfigManager import config
from common.Mp4Metadata import mp4_display_size, mp4_duration_seconds
from common.logging import error, fine, info, trace
from interfaces.chat import FallbackNotice, RichText
from interfaces.chat.ChatTransport import ChatTransport
from interfaces.chat.ChatTypes import (DECISION_APPROVE, DECISION_CANCEL, DECISION_DENY, KIND_PHOTO, KIND_REACTION,
                                       KIND_TEXT, KIND_VIDEO, KIND_VOICE, PHASE_COMMANDS, PHASE_PREFILL, PHASE_THINKING,
                                       Attachment, IncomingMessage, Progress)

# Telegram's Bot API refuses to serve files above this size to bots.
TELEGRAM_BOT_MAX_DOWNLOAD = 20 * 1024 * 1024
NOT_AUTHORIZED = "Sorry, you are not authorized to use this bot."


class _CommandLine:
    """A command echo posted to a chat: its message id once Telegram has it, and the
    completion detail if the command ended first."""

    def __init__(self, text: str):
        self.text = text
        self.message_id: Optional[int] = None
        self.detail: Optional[str] = None


class TelegramDraft:
    """One reply message: sent on the first visible content, edited as more arrives, within
    Telegram's per-chat edit tolerance; deleted when discarded."""

    EDIT_INTERVAL_SECONDS = 0.5
    # After EDIT_FAST_WINDOW_SECONDS of one response, edits slow to the SLOW interval.
    # Telegram tolerates ~1 edit/s per chat sustained; long agentic tasks at 2 edits/s
    # earn multi-minute flood bans (seen in prod: "Retry in 298 seconds").
    EDIT_INTERVAL_SLOW_SECONDS = 1.5
    EDIT_FAST_WINDOW_SECONDS = 30.0
    PROGRESS_INDICATOR_STYLE = "c64"
    _SPINNER_FRAMES_BRAILLE = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    # off-frame must be a visible non-whitespace char — Telegram rejects whitespace-only
    # edits ("Text must be non-empty"); light-shade gives an authentic C64 fade-blink
    _SPINNER_FRAMES_C64 = ["<code>█</code>", "<code>░</code>"]
    SPINNER_FRAMES = _SPINNER_FRAMES_C64 if PROGRESS_INDICATOR_STYLE == "c64" else _SPINNER_FRAMES_BRAILLE
    THINKING_PREFIX = "Thinking…\n"
    COMMAND_PREFIX = "Running command…\n"
    COMMANDS_PREFIX = "Running commands…\n"

    def __init__(self, transport: "TelegramTransport", chat_id: int):
        self._transport = transport
        self._chat_id = chat_id
        self._message_id: Optional[int] = None
        self._last_delivered = ""
        self._last_edit_time = 0.0
        self._started = time.time()
        self._slow_edits = False
        self._frame = 0
        self._closed = False
        self._typing = transport._start_typing(chat_id)

    @staticmethod
    def _format_token_count(n: int) -> str:
        return f"↓ {n / 1000:.1f}k tokens" if n >= 1000 else f"↓ {n} tokens"

    @classmethod
    def _thinking_label(cls, n: int) -> str:
        """Spinner prefix while thinking — adds a live token count once tokens start flowing."""
        return f"Thinking ({cls._format_token_count(n)})…\n" if n > 0 else cls.THINKING_PREFIX

    def _edit_interval(self) -> float:
        # Fast edits early for snappy feedback; once the response runs long, slow down to
        # stay under Telegram's per-chat flood limits.
        if time.time() - self._started < self.EDIT_FAST_WINDOW_SECONDS:
            return self.EDIT_INTERVAL_SECONDS
        if not self._slow_edits:
            self._slow_edits = True
            trace(f"Telegram [{self._chat_id}]: response running >{self.EDIT_FAST_WINDOW_SECONDS:.0f}s, "
                  f"slowing edits to {self.EDIT_INTERVAL_SLOW_SECONDS}s")
        return self.EDIT_INTERVAL_SLOW_SECONDS

    def _throttled(self) -> bool:
        return self._message_id is not None and (time.time() - self._last_edit_time) < self._edit_interval()

    def show_progress(self, progress: Progress) -> None:
        if self._closed or self._throttled():
            return
        frame = self.SPINNER_FRAMES[self._frame % len(self.SPINNER_FRAMES)]
        self._frame += 1
        if progress.phase == PHASE_THINKING:
            display = f"{self._thinking_label(progress.output_tokens)} {frame}"
        elif progress.phase == PHASE_PREFILL:
            display = f"Prefilling ({progress.prefill:.0f}%)…\n {frame}"
        elif progress.phase == PHASE_COMMANDS:
            display = f"{self.COMMANDS_PREFIX if progress.commands > 1 else self.COMMAND_PREFIX} {frame}"
        else:
            display = frame
        self._deliver(display, is_final=False)

    def update(self, html: str) -> bool:
        if self._closed or not html.strip():
            return False
        html = self._visible(html)
        if self._throttled() or html == self._last_delivered:
            return False
        return self._deliver(html, is_final=False)

    def finalize(self, html: str) -> bool:
        self._close()
        html = self._visible(html)
        if not html.strip():
            self.discard()
            return False
        if self._message_id is not None and html == self._last_delivered:
            trace(f"Telegram [{self._chat_id}]: final text already on screen")
            return True
        return self._deliver(html, is_final=True)

    def _visible(self, html: str) -> str:
        """The fallback notice is shown only while the console's debug toggle is on, as
        the Home tab does; the reply itself is what the chat gets."""
        visible = FallbackNotice.visible(html)
        if visible != html:
            trace(f"Telegram [{self._chat_id}]: fallback notice dropped (console debug off)")
        return visible

    def discard(self) -> None:
        self._close()
        if self._message_id is not None:
            self._transport._delete_message(self._chat_id, self._message_id)
            self._message_id = None

    def _close(self):
        self._closed = True
        self._typing.set()

    def _deliver(self, text: str, is_final: bool) -> bool:
        result = self._transport._send_or_edit(self._chat_id, self._message_id, text, is_final)
        self._last_edit_time = time.time()
        if result is None:
            return False
        self._message_id, delivered = result
        if delivered:
            self._last_delivered = text
        return delivered


class TelegramTransport(Integration, ChatTransport):
    """Keep as a Telegram bot: chat, voice notes, approvals and notifications.

    A chat with Keep in Telegram: write or send a voice message from the phone and get its
    replies, approval prompts and notifications there.

    Runs the bot: updates in, chat-layer rendering out."""

    source = SOURCE_TELEGRAM
    max_message_length = 4096
    lang = "en"

    # Longest we're willing to sleep out a flood ban to deliver a final message.
    FLOOD_WAIT_MAX_SECONDS = 600
    CALL_TIMEOUT_SECONDS = FLOOD_WAIT_MAX_SECONDS + 30
    APPROVAL_PREFIX = "approval"
    SECRET_PREFIX = "secret"
    SECRET_INSTRUCTIONS = ("Send the value as your next message. Only the local server reads it and your message is "
                           "deleted right away; the AI never sees it.")

    # Registered with the IntegrationMonitor by class (TelegramInterface.start), so
    # `restart_integration("Telegram")` rebuilds it with the token config_request just stored:
    # `_on_reset` takes the old bot off the hub and stops its loop, the new instance reads the
    # config again. The one switch is the interface flag, not `enabled_integrations`.
    def __init__(self):
        Integration.__init__(self)
        token = config.get('telegram_bot.bot_token', '') or ''
        # A copied example placeholder (YOUR_TELEGRAM_BOT_TOKEN_HERE) is no token.
        self._bot_token: str = '' if ConfigStore.looks_like_placeholder(token) else token
        # The chat-id lists are read from config at every use (`_allowed_chat_ids`), so a
        # config_set of one applies to the next message and the bot is never restarted from
        # the very chat that wrote it. Only the token and the link previews are baked into the
        # bot at build time and still need restart_integration("Telegram").
        self._seen_chat_ids: dict[str, list[int]] = {}
        self._link_previews: bool = bool(config.get('telegram_bot.link_previews', False))
        self._hub = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._application = None
        self._stop_event: Optional[asyncio.Event] = None
        # start() returns before the bot thread has built and initialised the application, so
        # what is scheduled in that window (the launcher's startup announcement) is held here
        # and fired once polling runs. Guarded by _held_lock.
        self._connecting = False
        self._held: list[tuple] = []  # (coroutine, what)
        self._held_lock = threading.Lock()
        self._message_texts: dict[tuple[int, int], str] = {}  # (chat_id, message_id) → text
        self._max_message_texts = 1000
        # chat_id → message_id of the current ephemeral status line; only touched from
        # coroutines on the bot loop. The asyncio lock serializes update/clear coroutines
        # across their awaits — without it two updates can interleave mid-send and both
        # post a line, orphaning the first.
        self._status_msgs: dict[int, int] = {}
        self._status_lock = asyncio.Lock()
        self._flood_until: dict[int, float] = {}  # chat_id → deadline of the active flood ban
        self._approval_bodies: dict[tuple[int, int], str] = {}  # (chat_id, message_id) → prompt HTML
        # chat_id → (message_id, token, title HTML) of the open secret prompt: while present,
        # the next text in that chat is an answer, not chat.
        self._secret_prompts: dict[int, tuple[int, str, str]] = {}

    # --- Integration ----------------------------------------------------------------------

    @classmethod
    def enabled(cls, config_manager) -> bool:
        """The bot follows `interfaces.telegram.enabled` (on by default), the flag the
        launcher starts the interface on — a rebuild after it was flipped off comes back as
        a disabled placeholder."""
        return bool(config_manager.get('interfaces.telegram.enabled', True))

    @property
    def name(self) -> str:
        return "Telegram"

    @property
    def config_keys(self) -> List[str]:
        return ["telegram_bot.bot_token"]

    def check_health(self) -> bool:
        if not self._bot_token:
            return False
        resp = requests.get(f"https://api.telegram.org/bot{self._bot_token}/getMe", timeout=10)
        return resp.status_code == 200

    @property
    def is_configured(self) -> bool:
        return bool(self._bot_token)

    # --- lifecycle ------------------------------------------------------------------------

    def start(self, hub=None):
        if hub is not None:
            self._hub = hub
        if not self.is_configured:
            trace("Telegram bot not configured (no bot_token). Skipping.")
            return
        if self._hub is None:
            error("Telegram bot started without a chat hub — incoming messages will be apologised to.")

        self._running = True
        self._connecting = True
        self._thread = threading.Thread(target=self._run, name="TelegramBot", daemon=True)
        self._thread.start()
        info("Telegram bot starting...")

    def stop(self):
        self._running = False
        if self._loop and self._stop_event:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
            if self._thread.is_alive() and self._loop:
                info("Telegram bot did not wind down in time; forcing event loop stop")
                self._loop.call_soon_threadsafe(self._loop.stop)

    def _on_reset(self):
        """The monitor's restart drops this instance: leave the hub (its sessions with us are
        cancelled) and stop the bot loop before the rebuilt one registers."""
        info("Telegram bot reset: leaving the chat hub and stopping the bot loop.")
        if self._hub is not None:
            self._hub.unregister(self)
        self.stop()

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._start_bot())
        except RuntimeError as e:
            if self._running:
                error(f"Telegram bot crashed: {e}")
            else:
                trace(f"Telegram bot event loop force-stopped during shutdown: {e}")
        except Exception as e:
            error(f"Telegram bot crashed: {e}")
        finally:
            self._drop_held()
            self._loop.close()

    def _release_held(self):
        """The bot is up: fire what was scheduled while it connected. Runs on the bot loop."""
        with self._held_lock:
            self._connecting = False
            held, self._held = self._held, []
        if not held:
            trace("Telegram bot is up, nothing was held while it connected.")
            return
        info(f"Telegram bot is up, sending {len(held)} held item(s): {', '.join(what for _, what in held)}")
        for coro, _ in held:
            asyncio.ensure_future(coro)

    def _drop_held(self):
        """The bot loop is over (stopped, or it never came up): nothing held can be sent."""
        with self._held_lock:
            self._connecting = False
            held, self._held = self._held, []
        for coro, _ in held:
            coro.close()
        if held:
            info(f"Telegram bot loop ended, {len(held)} held item(s) dropped: {', '.join(what for _, what in held)}")

    async def _start_bot(self):
        from telegram import LinkPreviewOptions, Update
        from telegram.ext import (ApplicationBuilder, CallbackQueryHandler, CommandHandler, Defaults,
                                  MessageHandler, MessageReactionHandler, filters)

        # Bot-wide default for every send and edit (replies, drafts, notifications, fallbacks):
        # a per-call disable_web_page_preview=True still wins, so opt-outs keep working.
        defaults = Defaults(link_preview_options=LinkPreviewOptions(is_disabled=not self._link_previews))
        info(f"Telegram: link previews {'enabled' if self._link_previews else 'disabled'} for all messages.")
        self._application = (ApplicationBuilder().token(self._bot_token).concurrent_updates(True)
                             .defaults(defaults).build())
        self._application.add_handler(CommandHandler("start", self._handle_start))
        self._application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))
        self._application.add_handler(MessageHandler(filters.PHOTO, self._handle_photo))
        self._application.add_handler(MessageHandler(filters.VOICE, self._handle_voice))
        self._application.add_handler(MessageHandler(filters.VIDEO | filters.VIDEO_NOTE | filters.ANIMATION,
                                                     self._handle_video))
        self._application.add_handler(MessageReactionHandler(self._handle_reaction))
        self._application.add_handler(CallbackQueryHandler(self._handle_callback_query))

        info("Telegram bot polling started.")
        await self._application.initialize()
        await self._application.start()
        await self._application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        self._release_held()

        self._stop_event = asyncio.Event()
        if self._running:
            await self._stop_event.wait()

        await self._application.updater.stop()
        await self._application.stop()
        await self._application.shutdown()
        info("Telegram bot stopped.")

    # --- loop bridging --------------------------------------------------------------------

    def _available(self, what: str) -> bool:
        if not self._application or not self._loop or not self._running:
            trace(f"Telegram bot not running, skipping {what}.")
            return False
        return True

    def _call(self, coro, what: str, timeout: Optional[float] = None):
        """Run a coroutine on the bot loop from another thread and wait for its result;
        None when the bot is not running or the call failed."""
        if not self._available(what):
            coro.close()
            return None
        if threading.current_thread() is self._thread:
            coro.close()
            error(f"Telegram: blocking {what} attempted from the bot loop thread — dropped")
            return None
        try:
            return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout or self.CALL_TIMEOUT_SECONDS)
        except Exception as e:
            error(f"Telegram: {what} failed: {e}")
            return None

    def _schedule(self, coro, what: str) -> bool:
        """Fire a coroutine on the bot loop without waiting. True when it was queued — or
        held, while the bot is still connecting, to be fired as soon as it is up."""
        with self._held_lock:
            if self._connecting:
                self._held.append((coro, what))
                trace(f"Telegram bot still connecting, {what} held until it is up ({len(self._held)} waiting).")
                return True
        if not self._available(what):
            coro.close()
            return False
        asyncio.run_coroutine_threadsafe(coro, self._loop)
        return True

    # --- ChatTransport: who ---------------------------------------------------------------

    @property
    def _allowed_chat_ids(self) -> list[int]:
        return self._live_chat_ids('telegram_bot.allowed_chat_ids')

    @property
    def _notify_chat_ids(self) -> list[int]:
        return self._live_chat_ids('telegram_bot.notify_chat_ids')

    def _live_chat_ids(self, key: str) -> list[int]:
        """The list under *key* as config holds it now — ConfigStore reloads the config after
        every write, so a config_set is in force at the next message. A change is logged once."""
        ids = self._parse_chat_ids(config.get(key, []), key)
        if ids != self._seen_chat_ids.get(key):
            if key in self._seen_chat_ids:
                info(f"{key} changed to {ids or 'empty'} — in force from now, no bot restart needed")
            self._seen_chat_ids[key] = ids
        return ids

    def owner_conversations(self) -> list:
        """The owners' private chats: the positive ids of the allow list. A group (a negative
        id) is never one — what goes to the owners alone (proactive pushes by default, the
        approval cards and secret prompts, the first-run brief) must not land in front of its
        guests."""
        allowed = self._allowed_chat_ids
        if not allowed:
            trace("No allowed_chat_ids configured, nowhere to notify.")
        owners = [c for c in allowed if c > 0]
        if len(owners) < len(allowed):
            trace(f"Owner conversations: {owners} ({len(allowed) - len(owners)} group(s) on the allow list left out)")
        return owners

    def notification_conversations(self) -> list:
        notify = self._notify_chat_ids
        if not notify:
            return self.owner_conversations()
        allowed = self._allowed_chat_ids
        outside = [c for c in notify if allowed and c not in allowed]
        if outside:
            info(f"telegram_bot.notify_chat_ids: {outside} not in allowed_chat_ids — notified, but cannot talk to the bot")
        return list(notify)

    @staticmethod
    def _parse_chat_ids(raw, key: str = 'telegram_bot.allowed_chat_ids') -> list[int]:
        """A chat id list from config as ints: a config_set writes the id as text, and a hand
        edit may quote it; Telegram hands chat ids over as ints, so the list must hold ints."""
        if raw is None:
            return []
        if not isinstance(raw, list):
            raw = [raw]
        ids: list[int] = []
        for item in raw:
            try:
                ids.append(int(str(item).strip()))
            except ValueError:
                info(f"{key}: ignoring '{item}' — a chat id is a number")
        return ids

    def _is_authorized(self, chat_id: int) -> bool:
        allowed = self._allowed_chat_ids
        if not allowed:
            return True
        return chat_id in allowed

    def _is_owner(self, user_id) -> bool:
        """The people on the allow list are the household's owners: a private chat's id is the
        person's user id, so the positive entries name them (the negative ones are groups).
        An empty list means anyone may talk to the bot, and is then an owner."""
        allowed = self._allowed_chat_ids
        if not allowed:
            return True
        return isinstance(user_id, int) and user_id in allowed

    def _bot_user_id(self) -> Optional[int]:
        """A bot token is `<bot user id>:<secret>`; no API call needed."""
        prefix = str(config.get('telegram_bot.bot_token', '') or '').partition(':')[0]
        return int(prefix) if prefix.isdigit() else None

    def _replies_to_bot(self, update) -> bool:
        replied = getattr(update.message, 'reply_to_message', None)
        sender = getattr(replied, 'from_user', None) if replied else None
        bot_id = self._bot_user_id()
        return bot_id is not None and getattr(sender, 'id', None) == bot_id

    def _guest_fields(self, update) -> dict:
        guest, is_bot = self._sender_facts(update)
        return {"guest": guest, "sender_is_bot": is_bot}

    def _sender_facts(self, update) -> tuple[bool, bool]:
        """(guest, is_bot) of the update's sender."""
        user = getattr(update, 'effective_user', None)
        user_id = getattr(user, 'id', None)
        is_bot = getattr(user, 'is_bot', False) is True
        return not self._is_owner(user_id), is_bot

    def _track_message_text(self, chat_id: int, message_id: int, text: str):
        if len(self._message_texts) >= self._max_message_texts:
            oldest_key = next(iter(self._message_texts))
            del self._message_texts[oldest_key]
        self._message_texts[(chat_id, message_id)] = text

    @staticmethod
    def _sender_name(update) -> str:
        """Display name of whoever sent the update (works for messages and reactions),
        falling back to the configured owner name when Telegram gives us no user."""
        user = getattr(update, 'effective_user', None)
        name = (user.first_name or user.username) if user else None
        if not name:
            name = config.get_owner_name()
            trace(f"Telegram: update carries no sender name, using owner '{name}'")
        return name

    # --- inbound: updates → IncomingMessage ------------------------------------------------

    @staticmethod
    def _chat_name(update) -> Optional[str]:
        """A group's title, carried into the turn header so the assistant knows which chat it is
        in; a private chat has no title."""
        title = getattr(getattr(update, 'effective_chat', None), 'title', None)
        return title if isinstance(title, str) and title else None

    async def _handle_start(self, update, context):
        chat_id = update.effective_chat.id
        name = self._chat_name(update)
        info(f"Telegram /start from chat_id: {chat_id}" + (f" (group '{name}')" if name else ""))
        await update.message.reply_text(
            f"Hello! I'm Keep AI.\n\nYour chat ID is: {chat_id}\n\n"
            f"Send me a message and I'll respond."
        )

    async def _admit(self, update) -> Optional[int]:
        """The chat id when the sender may talk to the bot, else None (after telling them)."""
        chat_id = update.effective_chat.id
        if update.message is None:
            # MessageHandler also fires for edited messages and channel posts, which carry no
            # `message`; edits are not re-answered.
            trace(f"Telegram [{chat_id}]: update without a new message (edit or channel post), ignored")
            return None
        if not self._is_authorized(chat_id):
            fine(f"Telegram: unauthorized chat_id {chat_id}")
            await update.message.reply_text(NOT_AUTHORIZED)
            return None
        if chat_id < 0 and not self._allowed_chat_ids:
            # An empty allow list makes everyone an owner — in a private chat, which is the
            # person's own. In a group that would hand the household to every member.
            info(f"Telegram [{chat_id}]: a group message while allowed_chat_ids is empty — ignored, a group "
                 f"is served only when its id is on the list")
            return None
        if chat_id < 0 and not self._replies_to_bot(update):
            # A group: the bot answers replies to its own bubbles. Privacy mode delivers nothing
            # else anyway; with it off this keeps her from answering every message. Her name
            # said in the group reaches her through the Telegram Feed (the owner's account).
            trace(f"Telegram [{chat_id}]: group message not replying to the bot, ignored")
            return None
        return chat_id

    async def _receive(self, update, msg: IncomingMessage):
        if self._hub is None:
            await self._apologise_unavailable(update)
            return
        self._hub.receive(self, msg)

    @staticmethod
    async def _apologise_unavailable(update):
        error("Telegram: no chat hub attached, cannot answer.")
        await update.message.reply_text("Sorry, the assistant is not available right now.")

    @staticmethod
    def _reply_to_text(update) -> Optional[str]:
        replied = update.message.reply_to_message
        if not replied:
            return None
        return replied.text or replied.caption or None

    async def _handle_message(self, update, context):
        chat_id = await self._admit(update)
        if chat_id is None:
            return
        if chat_id in self._secret_prompts:
            await self._take_secret_reply(update, chat_id)
            return
        user_text = update.message.text
        self._track_message_text(chat_id, update.message.message_id, user_text)
        guest, is_bot = self._sender_facts(update)
        info(f"Telegram [{chat_id}]: {user_text}" + (" (guest)" if guest else "") + (" (bot)" if is_bot else ""))
        await self._receive(update, IncomingMessage(
            conversation=chat_id, sender=self._sender_name(update), message_id=update.message.message_id,
            kind=KIND_TEXT, text=user_text, reply_to_text=self._reply_to_text(update),
            chat_name=self._chat_name(update), guest=guest, sender_is_bot=is_bot))

    async def _take_secret_reply(self, update, chat_id: int):
        """The text is the answer to the open secret prompt: hand it to the hub (off the loop —
        the hub may re-prompt through a blocking `_call`) and delete it from the chat. Neither
        the text nor its length is logged."""
        message_id = update.message.message_id
        if self._hub is None:
            await self._delete_message_async(chat_id, message_id)
            await self._apologise_unavailable(update)
            return
        if not self._is_owner(getattr(getattr(update, 'effective_user', None), 'id', None)):
            # The prompt is the owner's to answer; the text is deleted all the same, it may
            # have been meant as the secret.
            info(f"Telegram [{chat_id}]: text for the open secret prompt from {self._sender_name(update)}, "
                 f"not an owner — deleted, not taken")
            await self._delete_message_async(chat_id, message_id)
            return
        claimed = await asyncio.to_thread(self._hub.claim_secret, self, chat_id, update.message.text or "",
                                          self._sender_name(update))
        await self._delete_message_async(chat_id, message_id)
        if claimed:
            info(f"Telegram [{chat_id}]: secret reply handed to the hub and deleted")
            return
        # The prompt closed in the instant the text was sent: not an answer, not chat either.
        info(f"Telegram [{chat_id}]: text arrived for a secret prompt that had just closed — deleted, not answered")
        await self._send_text_async(chat_id, "That prompt had already closed, so nothing was stored — "
                                             "your message was deleted.", silent=True)

    async def _handle_photo(self, update, context):
        chat_id = await self._admit(update)
        if chat_id is None:
            return
        caption = update.message.caption or ""
        self._track_message_text(chat_id, update.message.message_id, caption or "[photo]")
        hub = self._hub
        if hub is None:
            await self._apologise_unavailable(update)
            return
        group_id = getattr(update.message, 'media_group_id', None)
        sender = self._sender_name(update)
        # Announced before the download: album members are handled concurrently, and the
        # batch must not go out while a sibling is still downloading.
        hub.expect_photo(self, chat_id, sender, group_id)
        try:
            photo = update.message.photo[-1]
            file = await context.bot.get_file(photo.file_id)
            photo_bytes = bytes(await file.download_as_bytearray())
        except Exception as e:
            error(f"Telegram [{chat_id}]: photo download failed: {e}")
            hub.photo_dropped(self, chat_id)
            # A caption sent as its own message has been folded into this album; without a
            # photo the batch is skipped, so the user must hear why nothing comes back.
            await update.message.reply_text("Sorry, I couldn't download that photo.")
            return
        media_type = "image/png" if photo_bytes[:4] == b'\x89PNG' else "image/jpeg"
        info(f"Telegram [{chat_id}]: photo received ({len(photo_bytes)} bytes, group {group_id}): {caption or '[no caption]'}")
        hub.receive(self, IncomingMessage(
            conversation=chat_id, sender=sender, message_id=update.message.message_id,
            kind=KIND_PHOTO, text=caption, attachments=[Attachment(photo_bytes, media_type, "photo")],
            group_id=group_id, chat_name=self._chat_name(update), **self._guest_fields(update)))

    async def _handle_voice(self, update, context):
        chat_id = await self._admit(update)
        if chat_id is None:
            return
        info(f"Telegram [{chat_id}]: voice message received")
        self._track_message_text(chat_id, update.message.message_id, "[voice message]")
        try:
            file = await context.bot.get_file(update.message.voice.file_id)
            voice_bytes = bytes(await file.download_as_bytearray())
        except Exception as e:
            error(f"Telegram [{chat_id}]: voice download failed: {e}")
            await update.message.reply_text("Sorry, I couldn't download that voice message.")
            return
        mime_type = getattr(update.message.voice, 'mime_type', None) or "audio/ogg"
        await self._receive(update, IncomingMessage(
            conversation=chat_id, sender=self._sender_name(update), message_id=update.message.message_id,
            kind=KIND_VOICE, attachments=[Attachment(voice_bytes, mime_type, "voice")],
            reply_to_text=self._reply_to_text(update), chat_name=self._chat_name(update),
            **self._guest_fields(update)))

    async def _handle_video(self, update, context):
        """A video, round video note or GIF (mp4 animation). The chat layer has it described
        by the video tier; only the download and Telegram's size cap are handled here."""
        chat_id = await self._admit(update)
        if chat_id is None:
            return
        message = update.message
        if message.video:
            media, kind = message.video, "video"
        elif message.video_note:
            media, kind = message.video_note, "video_note"
        else:
            media, kind = message.animation, "animation"
        label = Attachment(b"", "", kind).label
        caption = (message.caption or "").strip()
        file_size = getattr(media, 'file_size', None) or 0
        info(f"Telegram [{chat_id}]: {label} received ({file_size / 1024 / 1024:.1f} MB): {caption or '[no caption]'}")
        self._track_message_text(chat_id, message.message_id, caption or f"[{label}]")

        if file_size > TELEGRAM_BOT_MAX_DOWNLOAD:
            info(f"Telegram [{chat_id}]: {label} exceeds the {TELEGRAM_BOT_MAX_DOWNLOAD // (1024 * 1024)} MB bot download limit, skipping")
            await message.reply_text("That video is too large for me — Telegram only lets bots download files up to 20 MB.")
            return
        try:
            file = await context.bot.get_file(media.file_id)
            video_bytes = bytes(await file.download_as_bytearray())
        except Exception as e:
            error(f"Telegram [{chat_id}]: {label} download failed: {e}")
            await message.reply_text("Sorry, I couldn't download that video.")
            return
        mime_type = getattr(media, 'mime_type', None) or "video/mp4"
        await self._receive(update, IncomingMessage(
            conversation=chat_id, sender=self._sender_name(update), message_id=message.message_id,
            kind=KIND_VIDEO, text=caption, attachments=[Attachment(video_bytes, mime_type, kind)],
            reply_to_text=self._reply_to_text(update), chat_name=self._chat_name(update),
            **self._guest_fields(update)))

    async def _handle_reaction(self, update, context):
        reaction = update.message_reaction
        if not reaction:
            return
        chat_id = reaction.chat.id
        if not self._is_authorized(chat_id):
            fine(f"Telegram: reaction from unauthorized chat_id {chat_id}")
            return
        if chat_id < 0 and not self._allowed_chat_ids:
            # The same rule as _admit: with an empty allow list everyone is an owner, which in
            # a group would let any member start an owner's turn with one tap.
            info(f"Telegram [{chat_id}]: a group reaction while allowed_chat_ids is empty — ignored, a group "
                 f"is served only when its id is on the list")
            return

        old_emojis = {r.emoji for r in (reaction.old_reaction or []) if hasattr(r, 'emoji')}
        new_emojis = [r.emoji for r in (reaction.new_reaction or []) if hasattr(r, 'emoji')]
        added = [e for e in new_emojis if e not in old_emojis]
        if not added:
            trace(f"Telegram [{chat_id}]: reaction removed or custom emoji, ignoring")
            return

        emoji_str = " ".join(added)
        target_text = self._message_texts.get((chat_id, reaction.message_id))
        info(f"Telegram [{chat_id}]: reaction {emoji_str} to message {reaction.message_id} ({target_text!r})")
        if self._hub is None:
            error("Telegram: no chat hub attached, reaction dropped.")
            return
        self._hub.receive(self, IncomingMessage(
            conversation=chat_id, sender=self._sender_name(update), message_id=reaction.message_id,
            kind=KIND_REACTION, reaction=emoji_str, reaction_target_text=target_text,
            chat_name=self._chat_name(update), **self._guest_fields(update)))

    # --- ChatTransport: the draft -------------------------------------------------------------

    def begin_draft(self, conversation) -> TelegramDraft:
        return TelegramDraft(self, conversation)

    def _start_typing(self, chat_id: int) -> threading.Event:
        """Keep Telegram's "typing…" indicator alive until the returned event is set."""
        stop = threading.Event()
        if not self._schedule(self._keep_typing(chat_id, stop), "typing indicator"):
            stop.set()
        return stop

    async def _keep_typing(self, chat_id: int, stop: threading.Event):
        trace(f"Telegram [{chat_id}]: typing indicator started")
        while not stop.is_set():
            try:
                await self._application.bot.send_chat_action(chat_id=chat_id, action="typing")
            except Exception:
                pass
            for _ in range(40):
                if stop.is_set():
                    break
                await asyncio.sleep(0.1)
        trace(f"Telegram [{chat_id}]: typing indicator cancelled")

    def _send_or_edit(self, chat_id: int, message_id: Optional[int], text: str, is_final: bool):
        """(message_id, delivered) from the loop, or None when the bot is unavailable."""
        return self._call(self._send_or_edit_async(chat_id, message_id, text, is_final), "draft update")

    def _delete_message(self, chat_id: int, message_id: int):
        self._schedule(self._delete_message_async(chat_id, message_id), "message delete")

    async def _delete_message_async(self, chat_id: int, message_id: int):
        try:
            await self._application.bot.delete_message(chat_id=chat_id, message_id=message_id)
        except Exception as e:
            fine(f"Telegram: delete draft message failed: {e}")

    def _flood_wait_seconds(self, e: RetryAfter) -> float:
        """How long to sleep out a flood ban before retrying a final delivery."""
        return min(e.retry_after + 1, self.FLOOD_WAIT_MAX_SECONDS)

    def _note_flood_ban(self, chat_id, e: RetryAfter):
        self._flood_until[chat_id] = time.time() + e.retry_after

    async def _send_new(self, chat_id: int, text: str, parse_mode: Optional[str] = "HTML"):
        return await self._application.bot.send_message(chat_id=chat_id, text=text, parse_mode=parse_mode)

    async def _edit(self, chat_id: int, message_id: int, text: str, parse_mode: Optional[str] = "HTML"):
        kwargs = {"parse_mode": parse_mode} if parse_mode else {}
        await self._application.bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text, **kwargs)

    async def _send_or_edit_async(self, chat_id: int, message_id: Optional[int], text: str, is_final: bool = False):
        """Send a new message or edit an existing one. Returns (message_id, delivered).

        Intermediate updates (is_final=False) are best-effort: during an active flood ban
        they are dropped instead of retried, so the ban is never fed with more calls and
        the stream loop is never blocked. Final content waits the ban out. delivered=False
        means the text never reached the chat — the draft must not record it as shown, or
        the finalization dedup would skip the real delivery and the answer would be lost."""
        if not is_final:
            ban_left = self._flood_until.get(chat_id, 0.0) - time.time()
            if ban_left > 0:
                trace(f"Telegram [{chat_id}]: flood cooldown ({ban_left:.0f}s left), dropping intermediate update")
                return message_id, False
        sanitized = RichText.sanitize(text)
        if message_id is None:
            try:
                msg = await self._send_new(chat_id, sanitized)
            except RetryAfter as e:
                self._note_flood_ban(chat_id, e)
                if not is_final:
                    error(f"Telegram: flood control hit on send, dropping intermediate message (retry in {e.retry_after}s)")
                    return None, False
                error(f"Telegram: flood control hit on final send, waiting {e.retry_after}s to deliver")
                await asyncio.sleep(self._flood_wait_seconds(e))
                try:
                    msg = await self._send_new(chat_id, sanitized)
                except Exception as e2:
                    error(f"Telegram: final send failed even after flood wait, dropping message: {e2}")
                    return None, False
            except Exception as e:
                error(f"Telegram: HTML send failed, retrying once with HTML: {e}")
                try:
                    msg = await self._send_new(chat_id, sanitized)
                except Exception as e2:
                    error(f"Telegram: HTML retry also failed, falling back to plain text: {e2}")
                    plain = RichText.strip_tags(sanitized)
                    try:
                        msg = await self._send_new(chat_id, plain, parse_mode=None)
                    except RetryAfter as e3:
                        self._note_flood_ban(chat_id, e3)
                        error(f"Telegram: plain-text send also hit flood control, dropping message (retry in {e3.retry_after}s)")
                        return None, False
                    except Exception as e3:
                        error(f"Telegram: plain-text send also failed, dropping message: {e3}")
                        return None, False
            if msg:
                self._track_message_text(chat_id, msg.message_id, text)
                return msg.message_id, True
            return None, False

        delivered = True
        self._track_message_text(chat_id, message_id, text)
        try:
            await self._edit(chat_id, message_id, sanitized)
        except RetryAfter as e:
            self._note_flood_ban(chat_id, e)
            if not is_final:
                error(f"Telegram: flood control hit, dropping intermediate edit (retry in {e.retry_after}s)")
                return message_id, False
            error(f"Telegram: flood control hit on final edit, waiting {e.retry_after}s to deliver")
            await asyncio.sleep(self._flood_wait_seconds(e))
            try:
                await self._edit(chat_id, message_id, sanitized)
            except Exception as e2:
                error(f"Telegram: edit failed even after flood wait: {e2}")
                delivered = await self._fallback_send_new(chat_id, sanitized)
        except Exception as html_err:
            if "message is not modified" in str(html_err).lower():
                # Telegram already shows this text (seen when two spinner frames land in the
                # same instant): nothing to retry, and a retry would only fail the same way.
                trace(f"Telegram [{chat_id}]: edit skipped, message already shows this text")
                return message_id, True
            error(f"Telegram: HTML edit failed, retrying once with HTML: {html_err}")
            try:
                await self._edit(chat_id, message_id, sanitized)
            except Exception as retry_err:
                error(f"Telegram: HTML edit retry also failed, falling back to plain text: {retry_err}")
                plain = RichText.strip_tags(sanitized)
                try:
                    await self._edit(chat_id, message_id, plain, parse_mode=None)
                except RetryAfter as e:
                    self._note_flood_ban(chat_id, e)
                    if not is_final:
                        error(f"Telegram: flood control hit (plain), dropping intermediate edit (retry in {e.retry_after}s)")
                        return message_id, False
                    error(f"Telegram: flood control hit on final plain edit, waiting {e.retry_after}s to deliver")
                    await asyncio.sleep(self._flood_wait_seconds(e))
                    try:
                        await self._edit(chat_id, message_id, plain, parse_mode=None)
                    except Exception as e2:
                        error(f"Telegram: edit failed even after flood wait (plain): {e2}")
                        delivered = await self._fallback_send_new(chat_id, plain)
                except Exception as e:
                    error(f"Telegram: edit failed (plain): {e}")
                    if is_final:
                        delivered = await self._fallback_send_new(chat_id, plain)
                    else:
                        delivered = False
        return message_id, delivered

    async def _fallback_send_new(self, chat_id: int, text: str) -> bool:
        """Send a new message when editing the final response failed. True when it reached the chat."""
        try:
            info("Telegram: falling back to send_message after edit failure")
            await self._send_new(chat_id, text)
            return True
        except Exception as e:
            error(f"Telegram: fallback HTML send failed, stripping tags: {e}")
            try:
                await self._send_new(chat_id, RichText.strip_tags(text), parse_mode=None)
                return True
            except Exception as e2:
                error(f"Telegram: fallback send_message also failed: {e2}")
                return False

    # --- ChatTransport: one-shot sends --------------------------------------------------------

    def send_text(self, conversation, html: str, *, silent: bool = False, preview: bool = True) -> bool:
        if not html or not html.strip():
            trace("Empty message text, skipping Telegram send.")
            return False
        return self._schedule(self._send_text_async(conversation, html, silent, preview), "send")

    async def _send_text_async(self, chat_id: int, text: str, silent: bool = False, preview: bool = True):
        text = RichText.sanitize(text)
        kwargs = dict(chat_id=chat_id, disable_notification=silent)
        if not preview:
            # Only ever force previews off: passing False would override the bot-wide default.
            kwargs["disable_web_page_preview"] = True
        try:
            info(f"Sending telegram message: {text} to {chat_id}.")
            msg = await self._application.bot.send_message(text=text, parse_mode="HTML", **kwargs)
            self._track_message_text(chat_id, msg.message_id, text)
        except RetryAfter as e:
            # Notifications run as background tasks — wait the ban out instead of
            # hammering the API and losing the message.
            self._note_flood_ban(chat_id, e)
            error(f"Telegram: flood control hit sending to {chat_id}, waiting {e.retry_after}s to deliver")
            await asyncio.sleep(self._flood_wait_seconds(e))
            try:
                msg = await self._application.bot.send_message(text=text, parse_mode="HTML", **kwargs)
                self._track_message_text(chat_id, msg.message_id, text)
            except Exception as e2:
                error(f"Failed to send Telegram message to {chat_id} even after flood wait: {e2}")
        except Exception as e:
            error(f"Telegram: HTML send_message failed for {chat_id}, stripping tags: {e}")
            try:
                plain = RichText.strip_tags(text)
                msg = await self._application.bot.send_message(text=plain, **kwargs)
                self._track_message_text(chat_id, msg.message_id, plain)
            except Exception as e2:
                error(f"Failed to send Telegram message to {chat_id}: {e2}")

    def send_photo(self, conversation, data: bytes, caption: str = "") -> bool:
        return self._schedule(self._send_photo_async(conversation, data, caption), "photo send")

    async def _send_photo_async(self, chat_id: int, photo_bytes: bytes, caption: str):
        try:
            await self._application.bot.send_photo(chat_id=chat_id, photo=photo_bytes, caption=caption[:1024])
        except Exception as e:
            error(f"Failed to send Telegram photo to {chat_id}: {e}")

    def send_photo_album(self, conversation, photos: list[tuple[bytes, str]]) -> bool:
        return self._schedule(self._send_photo_album_async(conversation, photos), "photo album send")

    async def _send_photo_album_async(self, chat_id: int, photos: list[tuple[bytes, str]]):
        from telegram import InputMediaPhoto
        media = [InputMediaPhoto(media=photo_bytes, caption=caption[:1024] if i == 0 else None)
                 for i, (photo_bytes, caption) in enumerate(photos)]
        try:
            await self._application.bot.send_media_group(chat_id=chat_id, media=media)
        except Exception as e:
            error(f"Failed to send Telegram photo album to {chat_id}: {e}")

    def send_video(self, conversation, data: bytes, caption: str = "", thumbnail: Optional[bytes] = None) -> bool:
        return self._schedule(self._send_video_async(conversation, data, caption, thumbnail), "video send")

    async def _send_video_async(self, chat_id: int, video_bytes: bytes, caption: str, thumbnail: Optional[bytes] = None):
        # Telegram derives nothing from a bot upload: without explicit dimensions it lays the player
        # out from the coded frame size and ignores the rotation matrix phone cameras use for
        # portrait clips (squashed playback), and without a duration/thumbnail it shows 00:00 on black.
        try:
            size = mp4_display_size(video_bytes)
            duration = mp4_duration_seconds(video_bytes)
        except Exception as e:
            error(f"Telegram: could not read the clip's MP4 header ({e}), sending it without dimensions or duration")
            size = duration = None
        extra = {}
        if size:
            extra["width"], extra["height"] = size
        else:
            trace("Telegram: no video track header found, sending clip without dimensions")
        if duration:
            extra["duration"] = duration
        if thumbnail:
            extra["thumbnail"] = thumbnail
        shown = f"{size[0]}x{size[1]}" if size else "unknown size"
        shown += f", {duration}s" if duration else ", unknown duration"
        shown += ", thumbnail" if thumbnail else ", no thumbnail"
        try:
            await self._application.bot.send_video(
                chat_id=chat_id, video=video_bytes, caption=caption[:1024], supports_streaming=True, **extra
            )
            info(f"Telegram [{chat_id}]: sent video ({len(video_bytes) / 1024 / 1024:.1f} MB, {shown})")
        except Exception as e:
            error(f"Failed to send Telegram video to {chat_id}: {e}")

    # --- ChatTransport: the status line ---------------------------------------------------------

    def send_command_line(self, conversation, text: str) -> Any:
        ref = _CommandLine(text)
        if not self._schedule(self._send_command_line_async(conversation, ref), "command line"):
            return None
        return ref

    def finish_command_line(self, conversation, ref, text: str, detail: str) -> None:
        if ref is None:
            return
        self._schedule(self._finish_command_line_async(conversation, ref, detail), "command line detail")

    async def _send_command_line_async(self, chat_id: int, ref: "_CommandLine"):
        try:
            msg = await self._application.bot.send_message(
                chat_id=chat_id, text=RichText.command_line(ref.text), parse_mode="HTML",
                disable_notification=True, disable_web_page_preview=True)
        except Exception as e:
            error(f"Telegram: command line {ref.text!r} not sent to {chat_id}: {e}")
            return
        ref.message_id = msg.message_id
        self._track_message_text(chat_id, msg.message_id, ref.text)
        if ref.detail:
            # The command ended before its line was even posted: annotate it now.
            await self._finish_command_line_async(chat_id, ref, ref.detail)

    async def _finish_command_line_async(self, chat_id: int, ref: "_CommandLine", detail: str):
        ref.detail = detail
        if ref.message_id is None:
            trace(f"Telegram: command line {ref.text!r} still on its way, keeping the detail for it")
            return
        try:
            await self._edit(chat_id, ref.message_id, RichText.command_line(ref.text, detail))
        except Exception as e:
            # Best-effort: the line stays as it was rather than feeding a flood ban.
            fine(f"Telegram: command line detail for {ref.text!r} not shown: {e}")

    def set_status_line(self, conversation, html: Optional[str]) -> None:
        if html is None:
            self._schedule(self._clear_status_line(conversation), "status line clear")
        else:
            self._schedule(self._update_status_line(conversation, html), "status line")

    async def _update_status_line(self, chat_id: int, body: str):
        """Send or edit the single ephemeral status line of one chat."""
        async with self._status_lock:
            message_id = self._status_msgs.get(chat_id)
            if message_id is not None:
                try:
                    await self._application.bot.edit_message_text(
                        chat_id=chat_id, message_id=message_id, text=body, parse_mode="HTML",
                    )
                    trace(f"Telegram [{chat_id}]: edited status line {message_id}")
                    return
                except Exception as e:
                    if "message is not modified" in str(e).lower():
                        # Same text twice in a row — the line already shows it. Sending a
                        # new message here would orphan the old one (it stays in the chat).
                        trace(f"Telegram [{chat_id}]: status line {message_id} already shows this text")
                        return
                    trace(f"Telegram [{chat_id}]: status line edit failed, sending new: {e}")
            try:
                msg = await self._application.bot.send_message(
                    chat_id=chat_id, text=body, parse_mode="HTML",
                    disable_notification=True, disable_web_page_preview=True,
                )
                self._status_msgs[chat_id] = msg.message_id
                self._track_message_text(chat_id, msg.message_id, body)
                info(f"Telegram [{chat_id}]: sent status line {msg.message_id}")
            except Exception as e:
                error(f"Failed to send Telegram status line to {chat_id}: {e}")

    async def _clear_status_line(self, chat_id: int):
        async with self._status_lock:
            message_id = self._status_msgs.pop(chat_id, None)
            if message_id is None:
                trace(f"Telegram [{chat_id}]: no status line to clear")
                return
            try:
                await self._application.bot.delete_message(chat_id=chat_id, message_id=message_id)
                info(f"Telegram [{chat_id}]: deleted status line {message_id}")
            except Exception as e:
                trace(f"Telegram [{chat_id}]: status line delete failed: {e}")

    # --- ChatTransport: approvals ---------------------------------------------------------------

    def send_approval_prompt(self, conversation, token: str, title: str, details: str) -> Any:
        body = f"<b>Approval needed</b>\n{escape(title)}"
        if details:
            body += f"\n\n<blockquote>{escape(details)}</blockquote>"
        return self._call(self._send_approval_async(conversation, token, body), "approval prompt", timeout=30)

    async def _send_approval_async(self, chat_id: int, token: str, body: str) -> Optional[int]:
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Approve", callback_data=f"{self.APPROVAL_PREFIX}:{DECISION_APPROVE}:{token}"),
            InlineKeyboardButton("❌ Deny", callback_data=f"{self.APPROVAL_PREFIX}:{DECISION_DENY}:{token}"),
        ]])
        try:
            msg = await self._application.bot.send_message(
                chat_id=chat_id, text=body, parse_mode="HTML", reply_markup=keyboard,
            )
        except Exception as e:
            error(f"Failed to send approval message to {chat_id}: {e}")
            return None
        self._approval_bodies[(chat_id, msg.message_id)] = body
        return msg.message_id

    def close_approval_prompt(self, conversation, prompt_ref, outcome_text: str) -> None:
        self._schedule(self._close_approval_async(conversation, prompt_ref, outcome_text), "approval close")

    async def _close_approval_async(self, chat_id: int, message_id: int, outcome_text: str):
        body = self._approval_bodies.pop((chat_id, message_id), None)
        try:
            if body is not None:
                await self._application.bot.edit_message_text(
                    chat_id=chat_id, message_id=message_id, text=f"{body}\n\n<i>{outcome_text}</i>",
                    parse_mode="HTML", reply_markup=None,
                )
            else:
                await self._application.bot.edit_message_reply_markup(
                    chat_id=chat_id, message_id=message_id, reply_markup=None,
                )
        except Exception as e:
            trace(f"Approval message close failed: {e}")

    # --- secret prompts -------------------------------------------------------------------

    def send_secret_prompt(self, conversation, token: str, title: str, details: str) -> Any:
        header = f"🔐 <b>{escape(title)}</b>"
        body = f"{header}\n{self.SECRET_INSTRUCTIONS}"
        if details:
            body += f"\n\n<blockquote>{escape(details)}</blockquote>"
        return self._call(self._send_secret_async(conversation, token, header, body), "secret prompt", timeout=30)

    async def _send_secret_async(self, chat_id: int, token: str, header: str, body: str) -> Optional[int]:
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("✖ Cancel", callback_data=f"{self.SECRET_PREFIX}:{DECISION_CANCEL}:{token}"),
        ]])
        open_prompt = self._secret_prompts.get(chat_id)
        if open_prompt is not None and open_prompt[1] == token:
            # Asked again after a refused value: the same message, updated — a second one
            # would land below the reply bubble that is already waiting for the outcome.
            message_id = open_prompt[0]
            try:
                await self._application.bot.edit_message_text(
                    chat_id=chat_id, message_id=message_id, text=body, parse_mode="HTML", reply_markup=keyboard,
                )
            except Exception as e:
                error(f"Failed to update secret prompt in {chat_id}: {e}")
                return None
            info(f"Telegram [{chat_id}]: secret prompt [{token[:8]}] updated in place, next text is the answer")
            return message_id
        try:
            msg = await self._application.bot.send_message(
                chat_id=chat_id, text=body, parse_mode="HTML", reply_markup=keyboard,
            )
        except Exception as e:
            error(f"Failed to send secret prompt to {chat_id}: {e}")
            return None
        self._secret_prompts[chat_id] = (msg.message_id, token, header)
        info(f"Telegram [{chat_id}]: secret prompt [{token[:8]}] shown, next text is the answer")
        return msg.message_id

    def close_secret_prompt(self, conversation, prompt_ref, outcome_text: str) -> None:
        # Stop claiming texts right away, on the caller's thread: the edit that shows the
        # outcome can wait, the very next message must not be swallowed.
        open_prompt = self._secret_prompts.get(conversation)
        header = None
        if open_prompt is not None and open_prompt[0] == prompt_ref:
            del self._secret_prompts[conversation]
            header = open_prompt[2]
        self._schedule(self._close_secret_async(conversation, prompt_ref, header, outcome_text), "secret prompt close")

    async def _close_secret_async(self, chat_id: int, message_id: int, header: Optional[str], outcome_text: str):
        """The prompt shrinks to its title and the outcome: the instructions have done their job."""
        try:
            if header is not None:
                await self._application.bot.edit_message_text(
                    chat_id=chat_id, message_id=message_id, text=f"{header}\n<i>{outcome_text}</i>",
                    parse_mode="HTML", reply_markup=None,
                )
            else:
                await self._application.bot.edit_message_reply_markup(
                    chat_id=chat_id, message_id=message_id, reply_markup=None,
                )
        except Exception as e:
            trace(f"Secret prompt close failed: {e}")

    # --- callbacks ------------------------------------------------------------------------

    async def _handle_callback_query(self, update, context):
        query = update.callback_query
        if query is None:
            return
        chat_id = query.message.chat.id if query.message else None
        if chat_id is not None and not self._is_authorized(chat_id):
            await query.answer("Not authorized.", show_alert=True)
            return

        parts = (query.data or "").split(":", 2)
        if len(parts) != 3:
            await query.answer()
            return
        action, token = parts[1], parts[2]
        user = query.from_user
        who = (user.username or user.first_name or str(user.id)) if user else "user"
        if not self._is_owner(getattr(user, 'id', None)):
            # The card shows in the chat the request came from — in a group, to everyone.
            info(f"Telegram [{chat_id}]: {who} tapped an approval card but is not an owner, ignored")
            await query.answer("Only an owner of the household can decide this.", show_alert=True)
            return

        if parts[0] == self.SECRET_PREFIX and action == DECISION_CANCEL:
            if self._hub is None or not self._hub.cancel_secret(token, who):
                await query.answer("Already closed.")
                try:
                    await query.edit_message_reply_markup(reply_markup=None)
                except Exception:
                    pass
                return
            await query.answer("Cancelled.")
            return
        if parts[0] != self.APPROVAL_PREFIX or action not in (DECISION_APPROVE, DECISION_DENY):
            await query.answer()
            return

        if self._hub is None or not self._hub.resolve_approval(token, action, who):
            await query.answer("Already resolved.")
            try:
                await query.edit_message_reply_markup(reply_markup=None)
            except Exception:
                pass
            return
        await query.answer("Approved." if action == DECISION_APPROVE else "Denied.")
