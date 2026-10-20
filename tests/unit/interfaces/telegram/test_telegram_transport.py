"""TelegramTransport: only what Telegram itself is responsible for — update parsing into
IncomingMessage, the draft's send/edit/flood/fallback behaviour, the status line, approval
keyboards, one-shot sends and the loop bridge. The reply state machine and intake rules are
covered in tests/unit/interfaces/chat."""
import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from interfaces.chat.ChatTypes import (KIND_PHOTO, KIND_REACTION, KIND_TEXT, KIND_VIDEO, KIND_VOICE,
                                       PHASE_COMMANDS, PHASE_PREFILL, PHASE_THINKING, PHASE_WAITING, Progress)

MODULE = 'interfaces.telegram.TelegramTransport'


_config_patchers: list = []


def _make_transport(allowed=(), token='test-token-123', running=True, link_previews=False, notify=()):
    # The chat-id lists are read from config at every use, so the patch outlives construction;
    # the autouse fixture stops it. `t.config_values` is the dict the tests edit to simulate a
    # config_set + reload.
    values = {
        'telegram_bot.bot_token': token,
        'telegram_bot.allowed_chat_ids': list(allowed),
        'telegram_bot.notify_chat_ids': list(notify),
        'telegram_bot.link_previews': link_previews,
    }
    patcher = patch(f'{MODULE}.config')
    mock_config = patcher.start()
    _config_patchers.append(patcher)
    mock_config.get = lambda key, default='': values.get(key, default)
    from interfaces.telegram.TelegramTransport import TelegramTransport
    t = TelegramTransport()
    t.config_values = values
    if running:
        t._application = MagicMock()
        t._application.bot.send_message = AsyncMock(return_value=MagicMock(message_id=42))
        t._application.bot.edit_message_text = AsyncMock()
        t._application.bot.delete_message = AsyncMock()
        t._application.bot.send_chat_action = AsyncMock()
        t._loop = MagicMock()
        t._running = True
    return t


@pytest.fixture(autouse=True)
def reset_singleton():
    from interfaces.telegram.TelegramTransport import TelegramTransport
    TelegramTransport.reset_instance()
    yield
    TelegramTransport.reset_instance()
    while _config_patchers:
        _config_patchers.pop().stop()


@pytest.fixture
def transport():
    return _make_transport()


@pytest.fixture
def hub(transport):
    hub = MagicMock()
    transport._hub = hub
    return hub


def _update(chat_id=123, message_id=10, text="hello", first_name="Anna", reply=None, title=None, user_id=7,
            is_bot=False):
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.effective_chat.title = title  # a group has one, a private chat has none
    update.effective_user.id = user_id
    update.effective_user.is_bot = is_bot
    update.message.message_id = message_id
    update.message.text = text
    update.message.reply_text = AsyncMock()
    update.message.reply_to_message = reply
    update.effective_user.first_name = first_name
    update.effective_user.username = "anna"
    update.message.media_group_id = None
    return update


def _file_context(data=b'\xff\xd8\xff\xe0test'):
    ctx = MagicMock()
    file_mock = AsyncMock()
    file_mock.download_as_bytearray = AsyncMock(return_value=bytearray(data))
    ctx.bot.get_file = AsyncMock(return_value=file_mock)
    return ctx


def received(hub):
    hub.receive.assert_called_once()
    return hub.receive.call_args[0][1]


def _bot_in_group(allowed=(42, -5001)):
    """A bot (user id 777) that was added to group -5001; 42 is the owner's private chat."""
    transport = _make_transport(allowed=allowed, token='777:secret')
    hub = MagicMock()
    transport._hub = hub
    return transport, hub


def _membership(chat_id, title, old_status, new_status, first_name="Anna"):
    """A my_chat_member update: the bot's own status in `chat_id` went old → new."""
    update = MagicMock()
    update.my_chat_member.chat.id = chat_id
    update.my_chat_member.chat.title = title
    update.my_chat_member.old_chat_member.status = old_status
    update.my_chat_member.new_chat_member.status = new_status
    update.effective_user.first_name = first_name
    return update


def _bubble_of(user_id, text="earlier"):
    replied = MagicMock()
    replied.text = text
    replied.caption = None
    replied.from_user.id = user_id
    return replied


@pytest.mark.unit
class TestConfigAndAuth:
    def test_is_configured(self):
        assert _make_transport().is_configured is True
        assert _make_transport(token='').is_configured is False

    def test_authorization(self):
        assert _make_transport()._is_authorized(12345) is True
        t = _make_transport(allowed=[111, 222])
        assert t._is_authorized(111) and not t._is_authorized(333)
        assert t.owner_conversations() == [111, 222]

    def test_sender_name_prefers_first_name_then_username_then_owner(self, transport):
        update = MagicMock()
        update.effective_user.first_name = "Anna"
        assert transport._sender_name(update) == "Anna"
        update.effective_user.first_name = None
        update.effective_user.username = "anna_h"
        assert transport._sender_name(update) == "anna_h"
        update.effective_user = None
        with patch(f'{MODULE}.config') as cfg:
            cfg.get_owner_name.return_value = "Owner"
            assert transport._sender_name(update) == "Owner"

    def test_start_skips_when_not_configured(self):
        t = _make_transport(token='', running=False)
        t.start(MagicMock())
        assert t._thread is None

    def test_message_text_tracking_is_bounded(self, transport):
        transport._max_message_texts = 5
        for i in range(7):
            transport._track_message_text(100, i, f"msg {i}")
        assert len(transport._message_texts) == 5
        assert (100, 0) not in transport._message_texts and (100, 6) in transport._message_texts


@pytest.mark.unit
class TestGroupDestinations:
    """One post_message destination per group on the allow list, by title with the id as alias."""

    @pytest.fixture(autouse=True)
    def registry(self):
        from assistant import Destinations
        Destinations.clear()
        yield
        Destinations.clear()

    @staticmethod
    def _get_chat(titles: dict):
        async def get_chat(chat_id):
            if chat_id not in titles:
                raise RuntimeError("chat not found")
            return MagicMock(title=titles[chat_id])
        return get_chat

    def test_every_group_is_registered_by_title_and_a_private_chat_is_not(self):
        from assistant import Destinations
        t = _make_transport(allowed=(111, -5, -7))
        t._application.bot.get_chat = self._get_chat({-5: "Marcus & Alice - board", -7: "Family"})
        asyncio.run(t._register_group_destinations())
        assert Destinations.names() == ["Marcus & Alice - board", "Family"]
        assert Destinations.find("-7").name == "Family"
        assert Destinations.find("111") is None

    def test_a_group_telegram_cannot_describe_is_reached_by_its_id(self):
        from assistant import Destinations
        t = _make_transport(allowed=(-5,))
        t._application.bot.get_chat = self._get_chat({})
        asyncio.run(t._register_group_destinations())
        assert Destinations.names() == ["-5"]

    def test_a_post_goes_to_the_group_as_html_split_at_telegrams_limit(self):
        from assistant import Destinations
        t = _make_transport(allowed=(-5,))
        t._application.bot.get_chat = self._get_chat({-5: "Family"})
        asyncio.run(t._register_group_destinations())
        sent = []
        with patch.object(t, "send_text", side_effect=lambda cid, html, **kw: sent.append((cid, html)) or True):
            assert Destinations.send("family", "a" * 5000) == ""
        assert [cid for cid, _ in sent] == [-5, -5]
        assert "".join(html for _, html in sent) == "a" * 5000

    def test_a_changed_allow_list_registers_the_groups_again_and_stop_drops_them(self):
        from assistant import Destinations
        t = _make_transport(allowed=(-5,))
        t._application.bot.get_chat = self._get_chat({-5: "Family", -9: "Board"})
        asyncio.run(t._register_group_destinations())
        scheduled = []
        with patch.object(t, "_schedule", side_effect=lambda coro, what: scheduled.append(what) or coro.close() or True):
            t.config_values['telegram_bot.allowed_chat_ids'] = [-5, -9]
            t._allowed_chat_ids
            t._allowed_chat_ids
        assert scheduled == ["group destinations"]
        asyncio.run(t._register_group_destinations())
        assert Destinations.names() == ["Family", "Board"]
        t.stop()
        assert Destinations.names() == []

    def test_a_group_taken_off_the_allow_list_loses_its_destination(self):
        from assistant import Destinations
        t = _make_transport(allowed=(-5, -9))
        t._application.bot.get_chat = self._get_chat({-5: "Family", -9: "Board"})
        asyncio.run(t._register_group_destinations())
        t.config_values['telegram_bot.allowed_chat_ids'] = [-5]
        asyncio.run(t._register_group_destinations())
        assert Destinations.names() == ["Family"]
        assert Destinations.find("-9") is None

    def test_two_groups_with_one_title_both_stay_reachable_by_their_ids(self):
        from assistant import Destinations
        t = _make_transport(allowed=(-5, -7))
        t._application.bot.get_chat = self._get_chat({-5: "Family", -7: "Family"})
        asyncio.run(t._register_group_destinations())
        posted = []
        with patch.object(t, "send_text", side_effect=lambda cid, html, **kw: posted.append(cid) or True):
            assert Destinations.send("-5", "dinner at seven") == ""
            assert Destinations.send("-7", "dinner at seven") == ""
        assert posted == [-5, -7]


@pytest.mark.unit
class TestLifecycle:
    @staticmethod
    def _mock_application():
        app = MagicMock()
        app.initialize = AsyncMock()
        app.start = AsyncMock()
        app.stop = AsyncMock()
        app.shutdown = AsyncMock()
        app.updater.start_polling = AsyncMock()
        app.updater.stop = AsyncMock()
        builder = MagicMock()
        chain = builder.token.return_value.concurrent_updates.return_value
        chain.defaults.return_value.build.return_value = app
        return app, builder, chain

    def _start_and_stop(self, t, builder):
        with patch('telegram.ext.ApplicationBuilder', return_value=builder):
            t.start(MagicMock())
            deadline = time.monotonic() + 5
            while t._stop_event is None and time.monotonic() < deadline:
                time.sleep(0.01)
            assert t._stop_event is not None, "bot never reached its idle wait"
            t.stop()

    def test_link_previews_disabled_bot_wide_by_default(self):
        t = _make_transport(running=False)
        app, builder, chain = self._mock_application()
        self._start_and_stop(t, builder)
        defaults = chain.defaults.call_args[0][0]
        assert defaults.link_preview_options.is_disabled is True

    def test_link_previews_config_enables_them(self):
        t = _make_transport(running=False, link_previews=True)
        app, builder, chain = self._mock_application()
        self._start_and_stop(t, builder)
        defaults = chain.defaults.call_args[0][0]
        assert defaults.link_preview_options.is_disabled is False

    def test_stop_completes_the_bots_own_teardown(self):
        """stop() must wake the idle coroutine and let it finish updater.stop /
        application.stop / shutdown without an error log."""
        t = _make_transport(running=False)
        app, builder, _ = self._mock_application()
        errors = []
        with patch('telegram.ext.ApplicationBuilder', return_value=builder), \
             patch(f'{MODULE}.error', lambda msg: errors.append(msg)):
            t.start(MagicMock())
            deadline = time.monotonic() + 5
            while t._stop_event is None and time.monotonic() < deadline:
                time.sleep(0.01)
            assert t._stop_event is not None, "bot never reached its idle wait"
            t.stop()
        assert not t._thread.is_alive()
        app.updater.stop.assert_awaited_once()
        app.stop.assert_awaited_once()
        app.shutdown.assert_awaited_once()
        assert errors == []
        assert t._loop.is_closed()


    def test_a_send_made_while_the_bot_connects_goes_out_once_it_is_up(self):
        """start() returns before the bot thread has initialised the application; the
        launcher's startup announcement is sent in that window and must not be lost."""
        t = _make_transport(running=False)
        app, builder, _ = self._mock_application()
        app.bot.send_message = AsyncMock(return_value=MagicMock(message_id=42))
        connected = threading.Event()

        async def initialize():
            while not connected.is_set():
                await asyncio.sleep(0.005)

        app.initialize = AsyncMock(side_effect=initialize)
        with patch('telegram.ext.ApplicationBuilder', return_value=builder):
            t.start(MagicMock())
            assert t.send_text(100, "Server is up again") is True
            app.bot.send_message.assert_not_awaited()
            connected.set()
            deadline = time.monotonic() + 5
            while not app.bot.send_message.await_count and time.monotonic() < deadline:
                time.sleep(0.005)
            t.stop()
        assert app.bot.send_message.await_args.kwargs["chat_id"] == 100
        assert app.bot.send_message.await_args.kwargs["text"] == "Server is up again"

    def test_a_startup_error_is_retried_and_the_held_send_goes_out_once_the_bot_is_up(self):
        """A 429 on the first getMe after a quick restart must not end the bot for good: the
        thread tries again after the wait Telegram names, and the announcement held while
        it connected is still delivered."""
        from telegram.error import RetryAfter
        t = _make_transport(running=False)
        t.START_RETRY_MAX_SECONDS = 0.01
        app, builder, _ = self._mock_application()
        app.initialize = AsyncMock(side_effect=[RetryAfter(5), None])
        app.bot.send_message = AsyncMock(return_value=MagicMock(message_id=42))
        errors = []
        with patch('telegram.ext.ApplicationBuilder', return_value=builder), \
             patch(f'{MODULE}.error', lambda msg: errors.append(msg)):
            t.start(MagicMock())
            assert t.send_text(100, "Server is up again") is True
            deadline = time.monotonic() + 5
            while t._stop_event is None and time.monotonic() < deadline:
                time.sleep(0.005)
            assert t._stop_event is not None, "bot never came up"
            t.stop()
        assert app.initialize.await_count == 2
        assert app.bot.send_message.await_args.kwargs["text"] == "Server is up again"
        assert len(errors) == 1 and "retrying" in errors[0]
        assert not t._thread.is_alive()

    def test_stop_during_the_retry_wait_ends_the_thread_and_drops_what_it_held(self):
        t = _make_transport(running=False)
        app, builder, _ = self._mock_application()
        app.initialize = AsyncMock(side_effect=OSError("no route to host"))
        ran = []

        async def coro():
            ran.append(True)

        # The real delay is 5 s, slept in half-second chunks that stop() cannot wake: a short one
        # keeps the test from sitting out a chunk after the stop.
        with patch('telegram.ext.ApplicationBuilder', return_value=builder), patch(f'{MODULE}.error'), \
             patch.object(t, '_start_retry_seconds', return_value=0.02):
            t.start(MagicMock())
            assert t._schedule(coro(), "send") is True
            deadline = time.monotonic() + 5
            while not app.initialize.await_count and time.monotonic() < deadline:
                time.sleep(0.005)
            t.stop()
        assert not t._thread.is_alive()
        assert ran == [] and t._held == []
        assert t._connecting is False

    def test_the_retry_delay_follows_telegram_then_grows(self):
        from telegram.error import RetryAfter
        t = _make_transport(running=False)
        assert t._start_retry_seconds(RetryAfter(5), 1) == 6
        assert t._start_retry_seconds(RetryAfter(10_000), 1) == t.START_RETRY_MAX_SECONDS
        assert t._start_retry_seconds(OSError("x"), 1) == t.START_RETRY_BASE_SECONDS
        assert t._start_retry_seconds(OSError("x"), 3) == t.START_RETRY_BASE_SECONDS * 4
        assert t._start_retry_seconds(OSError("x"), 50) == t.START_RETRY_MAX_SECONDS

    def test_a_crash_once_the_bot_is_up_is_not_retried(self):
        t = _make_transport(running=False)
        app, builder, _ = self._mock_application()
        app.updater.stop = AsyncMock(side_effect=RuntimeError("boom"))
        errors = []
        with patch('telegram.ext.ApplicationBuilder', return_value=builder), \
             patch(f'{MODULE}.error', lambda msg: errors.append(msg)), patch(f'{MODULE}.trace') as trace:
            self._start_and_stop(t, builder)
        assert not t._thread.is_alive()
        assert app.initialize.await_count == 1
        assert errors == [] and any("boom" in str(c) for c in trace.call_args_list)

    def test_health_fails_when_the_bot_thread_is_dead(self):
        t = _make_transport(running=False)
        t._running = True
        t._thread = MagicMock(is_alive=MagicMock(return_value=False))
        with patch(f'{MODULE}.requests.get') as get, patch(f'{MODULE}.error'):
            assert t.check_health() is False
            get.assert_not_called()
        t._thread.is_alive.return_value = True
        with patch(f'{MODULE}.requests.get', return_value=MagicMock(status_code=200)):
            assert t.check_health() is True


@pytest.mark.unit
class TestLoopBridge:
    def test_call_and_schedule_are_noops_when_not_running(self):
        t = _make_transport(running=False)

        async def coro():
            return 1

        assert t._call(coro(), "x") is None
        assert t._schedule(coro(), "x") is False

    def test_call_refuses_the_loop_thread(self, transport):
        transport._thread = threading.current_thread()

        async def coro():
            return 1

        with patch(f'{MODULE}.error') as err:
            assert transport._call(coro(), "draft update") is None
        assert "bot loop thread" in err.call_args[0][0]

    def test_call_returns_none_and_logs_when_the_loop_call_fails(self, transport):
        # A hung or failed loop call becomes a False delivery, never an exception in the chat thread
        import concurrent.futures

        async def coro():
            return "never"

        with patch(f'{MODULE}.asyncio.run_coroutine_threadsafe') as rcts, patch(f'{MODULE}.error') as err:
            rcts.return_value.result.side_effect = concurrent.futures.TimeoutError()
            assert transport._call(coro(), "draft update") is None
            rcts.return_value.result.assert_called_once_with(transport.CALL_TIMEOUT_SECONDS)
            rcts.call_args[0][0].close()
            assert "draft update failed" in err.call_args[0][0]

            rcts.return_value.result.reset_mock()
            transport._call(coro(), "approval prompt", timeout=30)
            rcts.return_value.result.assert_called_once_with(30)
            rcts.call_args[0][0].close()

    def test_start_typing_without_loop_is_already_stopped(self):
        t = _make_transport(running=False)
        assert t._start_typing(1).is_set()


@pytest.mark.unit
class TestInboundText:
    @pytest.mark.asyncio
    async def test_text_becomes_an_incoming_message(self, transport, hub):
        await transport._handle_message(_update(), MagicMock())
        msg = received(hub)
        assert hub.receive.call_args[0][0] is transport
        assert msg.conversation == 123 and msg.sender == "Anna" and msg.message_id == 10
        assert msg.kind == KIND_TEXT and msg.text == "hello" and msg.reply_to_text is None
        assert transport._message_texts[(123, 10)] == "hello"
        assert msg.chat_name is None  # a private chat: nothing for the turn header's location

    @pytest.mark.asyncio
    async def test_a_group_message_carries_the_group_title(self):
        transport, hub = _bot_in_group()
        await transport._handle_message(_update(chat_id=-5001, title="Family", reply=_bubble_of(777)), MagicMock())
        msg = received(hub)
        assert msg.conversation == -5001 and msg.chat_name == "Family"

    @pytest.mark.asyncio
    async def test_start_in_a_group_logs_its_title(self, transport):
        with patch(f'{MODULE}.info') as log:
            await transport._handle_start(_update(chat_id=-5001, title="Family"), MagicMock())
        assert "-5001" in log.call_args[0][0] and "Family" in log.call_args[0][0]

    @pytest.mark.asyncio
    async def test_start_in_a_private_chat_leaves_the_chat_id_and_how_to_list_it_in_the_history(self):
        # Outside the setup a stranger's Start is a row, never a turn.
        transport, _ = _bot_in_group(allowed=[42])
        with patch(f'{MODULE}.system_note') as note, patch(f'{MODULE}.FirstRun.in_ftue', return_value=False), \
                patch(f'{MODULE}.notifier') as notifier:
            await transport._handle_start(_update(chat_id=99, first_name="Bob"), MagicMock())
        source, text = note.call_args[0]
        assert source == "telegram" and "99" in text and "Bob" in text
        assert "not on" in text and 'allowed_chat_ids[1]' in text
        notifier.return_value.open_conversation.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_during_the_setup_opens_the_conversation_instead_of_leaving_a_row(self):
        # The first-time setup's way to learn the owner's chat id without them copying it over,
        # and to go on without them typing "done": the assistant gets the fact as a turn.
        transport, _ = _bot_in_group(allowed=[42])
        with patch(f'{MODULE}.system_note') as note, patch(f'{MODULE}.FirstRun.in_ftue', return_value=True), \
                patch(f'{MODULE}.notifier') as notifier:
            notifier.return_value.open_conversation.return_value = True
            await transport._handle_start(_update(chat_id=99, first_name="Bob"), MagicMock())
        turn = notifier.return_value.open_conversation.call_args[0][0]
        assert "99" in turn and "Bob" in turn and 'allowed_chat_ids[1]' in turn
        note.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_during_the_setup_falls_back_to_a_row_when_no_chat_takes_the_turn(self):
        transport, _ = _bot_in_group(allowed=[42])
        with patch(f'{MODULE}.system_note') as note, patch(f'{MODULE}.FirstRun.in_ftue', return_value=True), \
                patch(f'{MODULE}.notifier') as notifier:
            notifier.return_value.open_conversation.return_value = False
            await transport._handle_start(_update(chat_id=99, first_name="Bob"), MagicMock())
        assert "99" in note.call_args[0][1]

    @pytest.mark.asyncio
    async def test_start_from_a_listed_chat_says_they_are_an_owner(self):
        transport, _ = _bot_in_group(allowed=[42])
        with patch(f'{MODULE}.system_note') as note:
            await transport._handle_start(_update(chat_id=42), MagicMock())
        text = note.call_args[0][1]
        assert "owner" in text and "not on" not in text

    @pytest.mark.asyncio
    async def test_start_in_a_group_leaves_no_row_of_its_own(self):
        # A group's id arrives when the bot joins it (my_chat_member).
        transport, _ = _bot_in_group()
        with patch(f'{MODULE}.system_note') as note:
            await transport._handle_start(_update(chat_id=-5001, title="Family"), MagicMock())
        note.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_bot_joining_a_group_leaves_the_groups_id_in_the_history(self):
        # In privacy mode nothing else from an unlisted group ever reaches the bot, so this
        # row is what "add the group I just created to the allow list" resolves against.
        transport, _ = _bot_in_group(allowed=[42])
        with patch(f'{MODULE}.system_note') as note:
            await transport._handle_my_chat_member(_membership(-5002, "Garden", "left", "member"), MagicMock())
        source, text = note.call_args[0]
        assert source == "telegram" and "-5002" in text and "Garden" in text and "Anna" in text
        assert "not on" in text and "allowed_chat_ids" in text

    @pytest.mark.asyncio
    async def test_joining_a_listed_group_and_leaving_one_are_told_apart(self):
        transport, _ = _bot_in_group()
        with patch(f'{MODULE}.system_note') as note:
            await transport._handle_my_chat_member(_membership(-5001, "Family", "left", "administrator"), MagicMock())
            joined = note.call_args[0][1]
            await transport._handle_my_chat_member(_membership(-5001, "Family", "member", "kicked"), MagicMock())
            left = note.call_args[0][1]
        assert "-5001" in joined and "not on" not in joined
        assert "removed" in left and "-5001" in left

    @pytest.mark.asyncio
    async def test_membership_changes_that_are_not_a_join_or_leave_leave_no_row(self, transport):
        with patch(f'{MODULE}.system_note') as note:
            # promoted to admin, a private chat blocking the bot, and a leave it already made
            for change in (_membership(-5001, "Family", "member", "administrator"),
                           _membership(123, None, "member", "kicked"),
                           _membership(-5001, "Family", "left", "kicked")):
                await transport._handle_my_chat_member(change, MagicMock())
        note.assert_not_called()

    @pytest.mark.asyncio
    async def test_reply_target_text_or_caption_is_carried(self, transport, hub):
        replied = MagicMock()
        replied.text = None
        replied.caption = "Photo caption"
        await transport._handle_message(_update(reply=replied), MagicMock())
        assert received(hub).reply_to_text == "Photo caption"

        hub.receive.reset_mock()
        replied.text = ""
        replied.caption = ""
        await transport._handle_message(_update(reply=replied), MagicMock())
        assert received(hub).reply_to_text is None

    @pytest.mark.asyncio
    async def test_unauthorized_chat_is_rejected(self, hub):
        t = _make_transport(allowed=[999])
        t._hub = hub
        update = _update(chat_id=123)
        await t._handle_message(update, MagicMock())
        update.message.reply_text.assert_called_once_with("Sorry, you are not authorized to use this bot.")
        hub.receive.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_hub_apologises(self, transport):
        update = _update()
        await transport._handle_message(update, MagicMock())
        assert "not available" in update.message.reply_text.call_args[0][0]

    @pytest.mark.asyncio
    async def test_edited_message_is_ignored_not_crashed_on(self, transport, hub):
        # PTB's MessageHandler fires for edited_message updates too, where `message` is None
        update = _update()
        update.message = None
        update.edited_message.text = "hello (edited)"
        ctx = _file_context()
        for handler in (transport._handle_message, transport._handle_photo, transport._handle_voice,
                        transport._handle_video):
            await handler(update, ctx)
        hub.receive.assert_not_called()
        ctx.bot.get_file.assert_not_called()

        t = _make_transport(allowed=[999])  # the unauthorized branch must not reply to a None message either
        t._hub = hub
        await t._handle_message(update, MagicMock())
        hub.receive.assert_not_called()


@pytest.mark.unit
class TestInboundMedia:
    @pytest.mark.asyncio
    async def test_photo_is_downloaded_with_caption_and_group(self, transport, hub):
        update = _update(text=None)
        update.message.caption = "Describe this"
        update.message.media_group_id = "g1"
        photo = MagicMock()
        photo.file_id = "file_abc"
        update.message.photo = [MagicMock(), photo]
        ctx = _file_context(b'\x89PNG\r\n')
        file_mock = ctx.bot.get_file.return_value
        order = []
        hub.expect_photo.side_effect = lambda *args: order.append("announce")

        async def get_file(file_id):
            order.append("download")
            return file_mock

        ctx.bot.get_file = AsyncMock(side_effect=get_file)
        await transport._handle_photo(update, ctx)
        ctx.bot.get_file.assert_called_once_with("file_abc")
        # The album is told before the download starts, so it waits for this member.
        hub.expect_photo.assert_called_once_with(transport, 123, "Anna", "g1")
        assert order == ["announce", "download"]
        msg = received(hub)
        assert msg.kind == KIND_PHOTO and msg.text == "Describe this" and msg.group_id == "g1"
        assert msg.attachments[0].data == b'\x89PNG\r\n' and msg.attachments[0].mime_type == "image/png"
        assert transport._message_texts[(123, 10)] == "Describe this"

    @pytest.mark.asyncio
    async def test_photo_without_caption_defaults_to_jpeg(self, transport, hub):
        update = _update()
        update.message.caption = None
        update.message.photo = [MagicMock(file_id="f")]
        await transport._handle_photo(update, _file_context(b'\xff\xd8data'))
        msg = received(hub)
        assert msg.text == "" and msg.attachments[0].mime_type == "image/jpeg" and msg.group_id is None
        assert transport._message_texts[(123, 10)] == "[photo]"

    @pytest.mark.asyncio
    async def test_photo_download_failure_is_dropped(self, transport, hub):
        update = _update()
        update.message.photo = [MagicMock(file_id="f")]
        ctx = MagicMock()
        ctx.bot.get_file = AsyncMock(side_effect=RuntimeError("network"))
        await transport._handle_photo(update, ctx)
        hub.receive.assert_not_called()
        hub.expect_photo.assert_called_once_with(transport, 123, "Anna", None)
        hub.photo_dropped.assert_called_once_with(transport, 123)  # else the album waits forever
        # A caption folded into the album is dropped with it, so the user must be told.
        update.message.reply_text.assert_awaited_once_with("Sorry, I couldn't download that photo.")

    @pytest.mark.asyncio
    async def test_photo_without_a_hub_apologises_before_downloading(self, transport):
        transport._hub = None
        update = _update()
        update.message.photo = [MagicMock(file_id="f")]
        ctx = _file_context()
        await transport._handle_photo(update, ctx)
        ctx.bot.get_file.assert_not_called()
        update.message.reply_text.assert_awaited_once_with("Sorry, the assistant is not available right now.")

    @pytest.mark.asyncio
    async def test_voice_is_downloaded(self, transport, hub):
        update = _update()
        update.message.voice.file_id = "voice_file_123"
        update.message.voice.mime_type = "audio/ogg"
        ctx = _file_context(b'\x00\x01\x02')
        await transport._handle_voice(update, ctx)
        ctx.bot.get_file.assert_called_once_with("voice_file_123")
        msg = received(hub)
        assert msg.kind == KIND_VOICE
        assert msg.attachments[0].data == b'\x00\x01\x02' and msg.attachments[0].kind == "voice"

    @pytest.mark.asyncio
    async def test_voice_download_failure_tells_the_user(self, transport, hub):
        update = _update()
        ctx = MagicMock()
        ctx.bot.get_file = AsyncMock(side_effect=RuntimeError("network"))
        await transport._handle_voice(update, ctx)
        assert "couldn't download" in update.message.reply_text.call_args[0][0]
        hub.receive.assert_not_called()

    @staticmethod
    def _video_update(kind="video", caption="What happens here?", file_size=1024, mime_type="video/mp4"):
        update = _update(text=None)
        update.message.caption = caption
        media = MagicMock()
        media.file_id = "file_vid"
        media.file_size = file_size
        media.mime_type = mime_type
        update.message.video = update.message.video_note = update.message.animation = None
        setattr(update.message, kind, media)
        return update

    @pytest.mark.asyncio
    async def test_video_is_downloaded_with_caption(self, transport, hub):
        update = self._video_update()
        ctx = _file_context(b"mp4-bytes")
        await transport._handle_video(update, ctx)
        ctx.bot.get_file.assert_called_once_with("file_vid")
        msg = received(hub)
        assert msg.kind == KIND_VIDEO and msg.text == "What happens here?"
        att = msg.attachments[0]
        assert att.data == b"mp4-bytes" and att.mime_type == "video/mp4" and att.kind == "video"
        assert transport._message_texts[(123, 10)] == "What happens here?"
        update.message.reply_text.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("attr,label", [("video_note", "video note"), ("animation", "GIF")])
    async def test_video_note_and_gif_kinds(self, transport, hub, attr, label):
        update = self._video_update(kind=attr, caption=None, mime_type=None)
        await transport._handle_video(update, _file_context(b"mp4"))
        att = received(hub).attachments[0]
        assert att.kind == attr and att.label == label and att.mime_type == "video/mp4"
        assert transport._message_texts[(123, 10)] == f"[{label}]"

    @pytest.mark.asyncio
    async def test_too_large_for_bot_api_is_not_downloaded(self, transport, hub):
        update = self._video_update(file_size=25 * 1024 * 1024)
        ctx = _file_context()
        await transport._handle_video(update, ctx)
        ctx.bot.get_file.assert_not_called()
        assert "too large" in update.message.reply_text.call_args[0][0]
        hub.receive.assert_not_called()

    @pytest.mark.asyncio
    async def test_unauthorized_media_is_rejected(self, hub):
        t = _make_transport(allowed=[999])
        t._hub = hub
        for handler in (t._handle_photo, t._handle_voice, t._handle_video):
            update = _update(chat_id=123)
            await handler(update, _file_context())
            update.message.reply_text.assert_called_once_with("Sorry, you are not authorized to use this bot.")
        hub.receive.assert_not_called()


@pytest.mark.unit
class TestInboundReaction:
    @staticmethod
    def _reaction_update(chat_id=123, message_id=10, old=(), new=()):
        update = MagicMock()
        update.message = None
        reaction = MagicMock()
        reaction.chat.id = chat_id
        reaction.message_id = message_id

        def obj(emoji):
            r = MagicMock()
            r.emoji = emoji
            return r

        reaction.old_reaction = [obj(e) for e in old]
        reaction.new_reaction = [obj(e) for e in new]
        update.message_reaction = reaction
        update.effective_user.first_name = "Anna"
        return update

    @pytest.mark.asyncio
    async def test_reaction_carries_emoji_and_target_text(self, transport, hub):
        transport._track_message_text(123, 10, "Hello world")
        await transport._handle_reaction(self._reaction_update(new=["👍"]), MagicMock())
        msg = received(hub)
        assert msg.kind == KIND_REACTION and msg.reaction == "👍" and msg.reaction_target_text == "Hello world"
        assert msg.message_id == 10 and msg.sender == "Anna"

    @pytest.mark.asyncio
    async def test_unknown_target_and_multiple_added(self, transport, hub):
        await transport._handle_reaction(self._reaction_update(message_id=999, old=["❤"], new=["❤", "👎", "🔥"]), MagicMock())
        msg = received(hub)
        assert msg.reaction == "👎 🔥" and msg.reaction_target_text is None

    @pytest.mark.asyncio
    async def test_removed_or_unchanged_reactions_are_ignored(self, transport, hub):
        await transport._handle_reaction(self._reaction_update(old=["👍"], new=[]), MagicMock())
        await transport._handle_reaction(self._reaction_update(old=["👍"], new=["👍"]), MagicMock())
        hub.receive.assert_not_called()

    @pytest.mark.asyncio
    async def test_unauthorized_reaction_is_ignored(self, hub):
        t = _make_transport(allowed=[999])
        t._hub = hub
        await t._handle_reaction(self._reaction_update(chat_id=123, new=["👍"]), MagicMock())
        hub.receive.assert_not_called()


@pytest.mark.unit
class TestSendOrEdit:
    """The draft's delivery coroutine: flood bans, HTML fallbacks, final vs intermediate."""

    @pytest.fixture
    def t(self, transport):
        transport._application.bot.send_message = AsyncMock(return_value=MagicMock(message_id=42))
        return transport

    @pytest.mark.asyncio
    async def test_new_message(self, t):
        assert await t._send_or_edit_async(123, None, "Hello") == (42, True)
        t._application.bot.send_message.assert_called_once_with(chat_id=123, text="Hello", parse_mode="HTML")
        assert t._message_texts[(123, 42)] == "Hello"

    @pytest.mark.asyncio
    async def test_edit_existing(self, t):
        assert await t._send_or_edit_async(123, 42, "Updated text") == (42, True)
        t._application.bot.edit_message_text.assert_called_once_with(
            chat_id=123, message_id=42, text="Updated text", parse_mode="HTML")

    @pytest.mark.asyncio
    async def test_send_retries_html_once_then_falls_back_to_plain(self, t):
        t._application.bot.send_message = AsyncMock(
            side_effect=[Exception("parse error"), Exception("retry failed"), MagicMock(message_id=7)])
        with patch(f'{MODULE}.error') as err:
            assert await t._send_or_edit_async(123, None, "some <b>text</b>") == (7, True)
        calls = t._application.bot.send_message.call_args_list
        assert calls[0].kwargs == dict(chat_id=123, text="some <b>text</b>", parse_mode="HTML")
        assert calls[2].kwargs == dict(chat_id=123, text="some text", parse_mode=None)
        err.assert_any_call("Telegram: HTML send failed, retrying once with HTML: parse error")
        err.assert_any_call("Telegram: HTML retry also failed, falling back to plain text: retry failed")

    @pytest.mark.asyncio
    async def test_send_html_retry_succeeds(self, t):
        t._application.bot.send_message = AsyncMock(side_effect=[Exception("transient"), MagicMock(message_id=7)])
        assert await t._send_or_edit_async(123, None, "some <b>text</b>") == (7, True)
        assert t._application.bot.send_message.call_count == 2

    @pytest.mark.asyncio
    async def test_intermediate_edit_hitting_flood_control_is_dropped(self, t):
        from telegram.error import RetryAfter
        t._application.bot.edit_message_text = AsyncMock(side_effect=RetryAfter(60))
        with patch(f'{MODULE}.asyncio.sleep', new_callable=AsyncMock) as sleep:
            assert await t._send_or_edit_async(123, 42, "Updated") == (42, False)
        sleep.assert_not_called()
        assert t._flood_until[123] > time.time()

    @pytest.mark.parametrize("retry_after,expected", [(5, 6), (299, 300), (3600, 600)])
    def test_flood_wait_is_clamped_to_the_maximum(self, t, retry_after, expected):
        # A day-long ban must not park the bot loop past the call timeout; the cap keeps it bounded
        from telegram.error import RetryAfter
        assert t._flood_wait_seconds(RetryAfter(retry_after)) == expected
        assert expected < t.CALL_TIMEOUT_SECONDS

    @pytest.mark.asyncio
    async def test_final_edit_waits_out_flood_control(self, t):
        from telegram.error import RetryAfter
        t._application.bot.edit_message_text = AsyncMock(side_effect=[RetryAfter(5), None])
        with patch(f'{MODULE}.asyncio.sleep', new_callable=AsyncMock) as sleep:
            assert await t._send_or_edit_async(123, 42, "Final", is_final=True) == (42, True)
        sleep.assert_called_once_with(6)
        assert t._application.bot.edit_message_text.call_count == 2
        t._application.bot.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_cooldown_drops_intermediate_without_api_call_but_not_final(self, t):
        t._flood_until[123] = time.time() + 100
        assert await t._send_or_edit_async(123, 42, "Spinner") == (42, False)
        t._application.bot.edit_message_text.assert_not_called()
        assert await t._send_or_edit_async(123, 42, "Final", is_final=True) == (42, True)
        t._application.bot.edit_message_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_intermediate_send_hitting_flood_control_is_dropped(self, t):
        from telegram.error import RetryAfter
        t._application.bot.send_message = AsyncMock(side_effect=RetryAfter(60))
        with patch(f'{MODULE}.asyncio.sleep', new_callable=AsyncMock) as sleep:
            assert await t._send_or_edit_async(123, None, "Hello") == (None, False)
        sleep.assert_not_called()
        assert t._flood_until[123] > time.time()

    @pytest.mark.asyncio
    async def test_final_send_waits_out_flood_control(self, t):
        from telegram.error import RetryAfter
        t._application.bot.send_message = AsyncMock(side_effect=[RetryAfter(5), MagicMock(message_id=42)])
        with patch(f'{MODULE}.asyncio.sleep', new_callable=AsyncMock) as sleep:
            assert await t._send_or_edit_async(123, None, "Final answer", is_final=True) == (42, True)
        sleep.assert_called_once_with(6)

    @pytest.mark.asyncio
    async def test_final_edit_failure_after_flood_wait_falls_back_to_send(self, t):
        from telegram.error import RetryAfter
        t._application.bot.edit_message_text = AsyncMock(side_effect=RetryAfter(1))
        with patch(f'{MODULE}.asyncio.sleep', new_callable=AsyncMock):
            assert await t._send_or_edit_async(123, 42, "Final text", is_final=True) == (42, True)
        t._application.bot.send_message.assert_called_once_with(chat_id=123, text="Final text", parse_mode="HTML")

    @pytest.mark.asyncio
    async def test_no_fallback_when_not_final(self, t):
        from telegram.error import RetryAfter
        t._application.bot.edit_message_text = AsyncMock(side_effect=RetryAfter(1))
        with patch(f'{MODULE}.asyncio.sleep', new_callable=AsyncMock):
            assert await t._send_or_edit_async(123, 42, "Some text", is_final=False) == (42, False)
        t._application.bot.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_edit_plain_branch_drops_intermediate_on_flood_control(self, t):
        from telegram.error import RetryAfter
        t._application.bot.edit_message_text = AsyncMock(
            side_effect=[ValueError("bad HTML"), ValueError("retry bad"), RetryAfter(2)])
        with patch(f'{MODULE}.asyncio.sleep', new_callable=AsyncMock) as sleep:
            assert await t._send_or_edit_async(123, 42, "text") == (42, False)
        sleep.assert_not_called()
        assert t._application.bot.edit_message_text.call_count == 3
        assert t._flood_until[123] > time.time()

    @pytest.mark.asyncio
    async def test_edit_plain_generic_failure_falls_back_on_final(self, t):
        t._application.bot.edit_message_text = AsyncMock(
            side_effect=[ValueError("bad HTML"), ValueError("retry bad"), RuntimeError("plain also failed")])
        assert await t._send_or_edit_async(123, 42, "<b>Final</b>", is_final=True) == (42, True)
        t._application.bot.send_message.assert_called_once_with(chat_id=123, text="Final", parse_mode="HTML")

    @pytest.mark.asyncio
    async def test_edit_plain_flood_retry_failure_falls_back_on_final(self, t):
        from telegram.error import RetryAfter
        t._application.bot.edit_message_text = AsyncMock(
            side_effect=[ValueError("bad HTML"), ValueError("retry bad"), RetryAfter(2), RuntimeError("still broken")])
        with patch(f'{MODULE}.asyncio.sleep', new_callable=AsyncMock) as sleep:
            assert await t._send_or_edit_async(123, 42, "Final", is_final=True) == (42, True)
        sleep.assert_called_once_with(3)
        assert t._application.bot.edit_message_text.call_count == 4
        t._application.bot.send_message.assert_called_once_with(chat_id=123, text="Final", parse_mode="HTML")

    @pytest.mark.asyncio
    async def test_edit_fallback_strips_html_and_logs(self, t):
        t._application.bot.edit_message_text = AsyncMock(side_effect=[ValueError("bad HTML"), ValueError("retry bad"), None])
        with patch(f'{MODULE}.error') as err:
            await t._send_or_edit_async(123, 42, "Check <b>this</b> and <code>that</code>")
        plain_call = t._application.bot.edit_message_text.call_args_list[2]
        assert plain_call.kwargs["text"] == "Check this and that" and "parse_mode" not in plain_call.kwargs
        err.assert_any_call("Telegram: HTML edit failed, retrying once with HTML: bad HTML")
        err.assert_any_call("Telegram: HTML edit retry also failed, falling back to plain text: retry bad")

    @pytest.mark.asyncio
    async def test_fallback_send_retries_plain_and_reports_total_failure(self, t):
        t._application.bot.send_message = AsyncMock(side_effect=[ValueError("bad HTML"), MagicMock()])
        assert await t._fallback_send_new(123, "<b>Response</b>") is True
        assert t._application.bot.send_message.call_args_list[1].kwargs["text"] == "Response"

        t._application.bot.send_message = AsyncMock(side_effect=[Exception("HTML fail"), Exception("plain fail")])
        with patch(f'{MODULE}.error') as err:
            assert await t._fallback_send_new(123, "<b>text</b>") is False
        err.assert_any_call("Telegram: fallback HTML send failed, stripping tags: HTML fail")
        err.assert_any_call("Telegram: fallback send_message also failed: plain fail")

    @pytest.mark.asyncio
    async def test_disallowed_tags_are_stripped_before_sending(self, t):
        await t._send_or_edit_async(123, None, "<div><b>x</b></div>")
        assert t._application.bot.send_message.call_args.kwargs["text"] == "<b>x</b>"


@pytest.mark.unit
class TestCommandLine:
    @pytest.mark.asyncio
    async def test_the_line_is_posted_silently_and_edited_with_the_detail(self, transport):
        from interfaces.telegram.TelegramTransport import _CommandLine
        ref = _CommandLine("Reading x")
        await transport._send_command_line_async(123, ref)
        kwargs = transport._application.bot.send_message.call_args.kwargs
        assert kwargs["text"] == " • <i>Reading x</i>" and kwargs["disable_notification"] is True
        assert ref.message_id == 42
        await transport._finish_command_line_async(123, ref, "done after 4s")
        transport._application.bot.edit_message_text.assert_awaited_once_with(
            chat_id=123, message_id=42, text=" • <i>Reading x</i> <i>(done after 4s)</i>", parse_mode="HTML")

    @pytest.mark.asyncio
    async def test_a_detail_that_arrives_before_the_line_is_applied_once_it_is_posted(self, transport):
        from interfaces.telegram.TelegramTransport import _CommandLine
        ref = _CommandLine("Reading x")
        await transport._finish_command_line_async(123, ref, "done after 0.1s")
        transport._application.bot.edit_message_text.assert_not_awaited()
        await transport._send_command_line_async(123, ref)
        assert transport._application.bot.edit_message_text.call_args.kwargs["text"].endswith("<i>(done after 0.1s)</i>")

    @pytest.mark.asyncio
    async def test_a_failed_edit_is_not_an_error(self, transport):
        from interfaces.telegram.TelegramTransport import _CommandLine
        ref = _CommandLine("Reading x")
        await transport._send_command_line_async(123, ref)
        transport._application.bot.edit_message_text = AsyncMock(side_effect=Exception("flood"))
        with patch(f'{MODULE}.error') as err:
            await transport._finish_command_line_async(123, ref, "done after 4s")
        err.assert_not_called()

    def test_sync_entry_points_schedule_on_the_bot_loop(self, transport):
        scheduled = []
        transport._schedule = lambda coro, what: (scheduled.append(what), coro.close(), True)[2]
        ref = transport.send_command_line(123, "Reading x")
        assert ref is not None and ref.text == "Reading x"
        transport.finish_command_line(123, ref, "Reading x", "done after 4s")
        transport.finish_command_line(123, None, "Reading x", "done after 4s")   # never posted: nothing to do
        assert scheduled == ["command line", "command line detail"]

    def test_nothing_is_kept_when_the_bot_is_down(self):
        t = _make_transport(running=False)
        assert t.send_command_line(123, "Reading x") is None


class _RecordingTransport:
    """A stand-in for TelegramTransport under TelegramDraft: records every delivery attempt
    and answers with scripted (message_id, delivered) results."""

    def __init__(self, results=None):
        self.results = list(results or [])
        self.deliveries = []
        self.deleted = []
        self.typing_events = []

    def _start_typing(self, chat_id):
        ev = threading.Event()
        self.typing_events.append(ev)
        return ev

    def _send_or_edit(self, chat_id, message_id, text, is_final):
        self.deliveries.append((message_id, text, is_final))
        if self.results:
            return self.results.pop(0)
        return (message_id if message_id is not None else 42), True

    def _delete_message(self, chat_id, message_id):
        self.deleted.append(message_id)

    @property
    def texts(self):
        return [d[1] for d in self.deliveries]


@pytest.mark.unit
class TestNotModifiedEdit:
    @pytest.mark.asyncio
    async def test_an_unchanged_edit_counts_as_delivered_without_retries(self, transport):
        transport._application.bot.edit_message_text = AsyncMock(side_effect=Exception(
            "Message is not modified: specified new message content and reply markup are exactly the same"))
        with patch(f'{MODULE}.error') as err:
            assert await transport._send_or_edit_async(111, 42, "<code>█</code>") == (42, True)
        transport._application.bot.edit_message_text.assert_awaited_once()
        transport._application.bot.send_message.assert_not_called()
        err.assert_not_called()


class TestTelegramDraft:
    def _draft(self, results=None):
        from interfaces.telegram.TelegramTransport import TelegramDraft
        t = _RecordingTransport(results)
        d = TelegramDraft(t, 123)
        d.EDIT_INTERVAL_SECONDS = 0.0
        return t, d

    def test_progress_frames_alternate_and_reply_replaces_them(self):
        from interfaces.telegram.TelegramTransport import TelegramDraft
        t, d = self._draft()
        d.show_progress(Progress(PHASE_WAITING))
        d.show_progress(Progress(PHASE_WAITING))
        assert t.texts == TelegramDraft.SPINNER_FRAMES[:2]
        assert d.update("Hello") is True
        assert d.finalize("Hello world") is True
        assert t.deliveries[-1] == (42, "Hello world", True)
        assert t.deliveries[2] == (42, "Hello", False)

    def test_thinking_label_with_token_count(self):
        t, d = self._draft()
        d.show_progress(Progress(PHASE_THINKING, output_tokens=123))
        d.show_progress(Progress(PHASE_THINKING, output_tokens=20700))
        d.show_progress(Progress(PHASE_THINKING, output_tokens=0))
        d.show_progress(Progress(PHASE_WAITING))
        assert t.texts[0].startswith("Thinking (↓ 123 tokens)…\n")
        assert t.texts[1].startswith("Thinking (↓ 20.7k tokens)…\n")
        assert t.texts[2].startswith("Thinking…\n")
        assert t.texts[3] in d.SPINNER_FRAMES

    def test_prefill_label_carries_the_percentage(self):
        t, d = self._draft()
        d.show_progress(Progress(PHASE_PREFILL, prefill=42.6))
        assert t.texts[0].startswith("Prefilling (43%)…\n")

    def test_running_command_label(self):
        t, d = self._draft()
        d.show_progress(Progress(PHASE_COMMANDS, commands=1))
        d.show_progress(Progress(PHASE_COMMANDS, commands=3))
        assert t.texts[0].startswith("Running command…\n")
        assert t.texts[1].startswith("Running commands…\n")

    def test_updates_are_throttled_within_the_edit_interval(self):
        t, d = self._draft()
        d.EDIT_INTERVAL_SECONDS = 10.0
        d.show_progress(Progress(PHASE_THINKING))
        d.show_progress(Progress(PHASE_THINKING))
        assert d.update("Hello") is False
        assert len(t.deliveries) == 1  # the first send goes out immediately, the rest waits
        assert d.finalize("Hello") is True
        assert t.deliveries[-1] == (42, "Hello", True)

    def test_slow_interval_after_the_fast_window(self):
        t, d = self._draft()
        d.EDIT_INTERVAL_SECONDS = 0.5
        d._started = time.time() - 31
        assert d._edit_interval() == d.EDIT_INTERVAL_SLOW_SECONDS

    def test_finalize_dedups_text_already_on_screen(self):
        t, d = self._draft()
        d.update("Same")
        assert d.finalize("Same") is True
        assert len(t.deliveries) == 1

    def test_dropped_intermediate_does_not_skip_final_delivery(self):
        """Regression: an update dropped by flood control must not count as shown, or the
        final dedup would skip the real delivery and the answer would be lost."""
        t, d = self._draft(results=[(None, False), (7, True)])
        assert d.update("Hello") is False
        assert d.finalize("Hello") is True
        assert t.deliveries == [(None, "Hello", False), (None, "Hello", True)]

    def test_identical_update_is_not_resent(self):
        t, d = self._draft()
        d.update("Hi")
        assert d.update("Hi") is False
        assert len(t.deliveries) == 1

    def test_discard_deletes_the_message_and_stops_typing(self):
        t, d = self._draft()
        d.show_progress(Progress(PHASE_WAITING))
        d.discard()
        assert t.deleted == [42]
        assert t.typing_events[0].is_set()
        assert d.update("late") is False

    def test_discard_without_a_message_deletes_nothing(self):
        t, d = self._draft()
        d.discard()
        assert t.deleted == []

    def test_finalize_stops_typing_and_closes(self):
        t, d = self._draft()
        d.finalize("Done")
        assert t.typing_events[0].is_set()
        d.show_progress(Progress(PHASE_WAITING))
        assert len(t.deliveries) == 1

    def test_unavailable_loop_returns_false(self):
        t, d = self._draft(results=[None])
        assert d.update("x") is False

    NOTICE = "\n\n(Fable refused this request, had to use Opus fallback)"

    def test_fallback_notice_is_dropped_while_console_debug_is_off(self):
        t, d = self._draft()
        with patch('interfaces.chat.FallbackNotice.console_debug', return_value=False):
            assert d.update("Sure." + self.NOTICE) is True
            assert d.finalize("Sure." + self.NOTICE) is True
        assert t.deliveries == [(None, "Sure.", False)]   # the final is the text already on screen

    def test_fallback_notice_shows_with_console_debug_on(self):
        t, d = self._draft()
        with patch('interfaces.chat.FallbackNotice.console_debug', return_value=True):
            assert d.finalize("Sure." + self.NOTICE) is True
        assert t.deliveries == [(None, "Sure." + self.NOTICE, True)]

    def test_a_reply_that_is_only_the_notice_is_discarded(self):
        t, d = self._draft()
        d.show_progress(Progress(PHASE_WAITING))
        with patch('interfaces.chat.FallbackNotice.console_debug', return_value=False):
            assert d.finalize(self.NOTICE) is False
        assert t.deleted == [42]

@pytest.mark.unit
class TestOneShotSends:
    def test_send_text_schedules_and_guards(self, transport):
        with patch(f'{MODULE}.asyncio.run_coroutine_threadsafe') as rcts:
            assert transport.send_text(111, "Hello") is True
            rcts.assert_called_once()
            assert rcts.call_args[0][1] is transport._loop
            rcts.call_args[0][0].close()
            assert transport.send_text(111, "") is False
            assert transport.send_text(111, "  \n") is False
        t = _make_transport(running=False)
        assert t.send_text(111, "Hello") is False

    @pytest.mark.asyncio
    async def test_send_text_async_html_flags_and_fallback(self, transport):
        await transport._send_text_async(111, "<b>Hello</b>", silent=True, preview=False)
        transport._application.bot.send_message.assert_called_once_with(
            chat_id=111, text="<b>Hello</b>", parse_mode="HTML", disable_notification=True, disable_web_page_preview=True)

        # preview=True defers to the bot-wide default instead of forcing previews back on
        transport._application.bot.send_message = AsyncMock(return_value=MagicMock(message_id=3))
        await transport._send_text_async(111, "<b>Hello</b>")
        transport._application.bot.send_message.assert_called_once_with(
            chat_id=111, text="<b>Hello</b>", parse_mode="HTML", disable_notification=False)

        transport._application.bot.send_message = AsyncMock(side_effect=[Exception("bad HTML"), MagicMock(message_id=2)])
        await transport._send_text_async(111, "<b>Bold</b> and <code>code</code>")
        plain_call = transport._application.bot.send_message.call_args_list[1]
        assert plain_call.kwargs["text"] == "Bold and code" and "parse_mode" not in plain_call.kwargs

    @pytest.mark.asyncio
    async def test_send_text_async_waits_out_flood_ban(self, transport):
        from telegram.error import RetryAfter
        transport._application.bot.send_message = AsyncMock(side_effect=[RetryAfter(3), MagicMock(message_id=2)])
        with patch(f'{MODULE}.asyncio.sleep', new_callable=AsyncMock) as sleep:
            await transport._send_text_async(111, "hi")
        sleep.assert_called_once_with(4)
        assert transport._application.bot.send_message.call_count == 2

    @pytest.mark.asyncio
    async def test_photo_caption_truncated(self, transport):
        transport._application.bot.send_photo = AsyncMock()
        await transport._send_photo_async(111, b"\xff\xd8", "x" * 2000)
        kwargs = transport._application.bot.send_photo.call_args.kwargs
        assert kwargs["chat_id"] == 111 and kwargs["photo"] == b"\xff\xd8" and len(kwargs["caption"]) == 1024

    @pytest.mark.asyncio
    async def test_album_only_first_photo_gets_caption(self, transport):
        transport._application.bot.send_media_group = AsyncMock()
        await transport._send_photo_album_async(111, [(b"1", "x" * 2000), (b"2", "Cap2"), (b"3", "Cap3")])
        media = transport._application.bot.send_media_group.call_args.kwargs["media"]
        assert len(media) == 3 and len(media[0].caption) == 1024 and media[1].caption is None and media[2].caption is None

    @pytest.mark.asyncio
    async def test_video_passes_display_size_duration_and_thumbnail(self, transport):
        transport._application.bot.send_video = AsyncMock()
        with patch(f'{MODULE}.mp4_display_size', return_value=(720, 1280)), \
             patch(f'{MODULE}.mp4_duration_seconds', return_value=58):
            await transport._send_video_async(111, b"mp4", "A" * 2000, thumbnail=b"jpg")
        kwargs = transport._application.bot.send_video.call_args.kwargs
        assert kwargs["chat_id"] == 111 and kwargs["video"] == b"mp4" and len(kwargs["caption"]) == 1024
        assert kwargs["supports_streaming"] is True
        assert (kwargs["width"], kwargs["height"], kwargs["duration"], kwargs["thumbnail"]) == (720, 1280, 58, b"jpg")

    @pytest.mark.asyncio
    async def test_video_still_sends_when_header_parse_fails(self, transport):
        transport._application.bot.send_video = AsyncMock()
        with patch(f'{MODULE}.mp4_display_size', side_effect=ValueError("bad atom")):
            await transport._send_video_async(111, b"mp4", "clip", thumbnail=b"jpg")
        kwargs = transport._application.bot.send_video.call_args.kwargs
        assert kwargs["thumbnail"] == b"jpg" and "width" not in kwargs and "duration" not in kwargs

    @pytest.mark.asyncio
    async def test_a_turned_clip_is_re_encoded_upright_and_sent_with_the_new_coded_size(self, transport):
        # Telegram for iOS drops the track matrix at playback, so the turn goes into the pixels
        # before the upload; the dimensions are read from the clip that is actually sent.
        transport._application.bot.send_video = AsyncMock()
        sizes = {b"turned": (720, 1280), b"upright": (720, 1280)}
        with patch(f'{MODULE}.mp4_is_rotated', side_effect=lambda data: data == b"turned"), \
             patch(f'{MODULE}.bake_rotation', return_value=b"upright") as bake, \
             patch(f'{MODULE}.mp4_display_size', side_effect=sizes.get) as display_size, \
             patch(f'{MODULE}.mp4_duration_seconds', return_value=21):
            await transport._send_video_async(111, b"turned", "clip", thumbnail=b"jpg")
        bake.assert_called_once_with(b"turned")
        display_size.assert_called_once_with(b"upright")
        kwargs = transport._application.bot.send_video.call_args.kwargs
        assert kwargs["video"] == b"upright"
        assert (kwargs["width"], kwargs["height"], kwargs["duration"], kwargs["thumbnail"]) == (720, 1280, 21, b"jpg")

    @pytest.mark.asyncio
    async def test_a_clip_stored_as_shown_is_sent_without_re_encoding(self, transport):
        transport._application.bot.send_video = AsyncMock()
        with patch(f'{MODULE}.mp4_is_rotated', return_value=False), \
             patch(f'{MODULE}.bake_rotation') as bake, \
             patch(f'{MODULE}.mp4_display_size', return_value=(1280, 720)):
            await transport._send_video_async(111, b"landscape", "clip")
        bake.assert_not_called()
        assert transport._application.bot.send_video.call_args.kwargs["video"] == b"landscape"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("outcome", [{"return_value": None}, {"side_effect": RuntimeError("no encoder")}])
    async def test_a_failed_re_encode_sends_the_clip_as_it_is(self, transport, outcome):
        transport._application.bot.send_video = AsyncMock()
        with patch(f'{MODULE}.mp4_is_rotated', return_value=True), \
             patch(f'{MODULE}.bake_rotation', **outcome), \
             patch(f'{MODULE}.mp4_display_size', return_value=(720, 1280)):
            await transport._send_video_async(111, b"turned", "clip")
        kwargs = transport._application.bot.send_video.call_args.kwargs
        assert kwargs["video"] == b"turned" and (kwargs["width"], kwargs["height"]) == (720, 1280)

    @pytest.mark.asyncio
    async def test_video_without_a_track_header_still_carries_its_duration(self, transport):
        # width=None or a dropped duration puts the 00:00-on-black player back; absent keys are the contract
        transport._application.bot.send_video = AsyncMock()
        with patch(f'{MODULE}.mp4_display_size', return_value=None), \
             patch(f'{MODULE}.mp4_duration_seconds', return_value=12):
            await transport._send_video_async(111, b"mp4", "clip")
        kwargs = transport._application.bot.send_video.call_args.kwargs
        assert kwargs["duration"] == 12 and kwargs["supports_streaming"] is True
        assert not {"width", "height", "thumbnail"} & kwargs.keys()


@pytest.mark.unit
class TestStatusLine:
    def test_set_status_line_schedules_update_or_clear(self, transport):
        with patch(f'{MODULE}.asyncio.run_coroutine_threadsafe') as rcts:
            transport.set_status_line(111, " • <i>x</i>")
            transport.set_status_line(111, None)
            assert [c[0][0].__name__ for c in rcts.call_args_list] == ["_update_status_line", "_clear_status_line"]
            for c in rcts.call_args_list:
                c[0][0].close()
        t = _make_transport(running=False)
        with patch(f'{MODULE}.asyncio.run_coroutine_threadsafe') as rcts:
            t.set_status_line(111, "x")
            rcts.assert_not_called()

    def test_a_group_never_gets_a_status_line(self, transport):
        """The line names what the assistant does (a remember is "Saving to memory: …") in
        front of the group's guests; it is the owners' like the approval cards."""
        with patch(f'{MODULE}.asyncio.run_coroutine_threadsafe') as rcts:
            transport.set_status_line(-100500, " • <i>Saving to memory…</i>")
            transport.set_status_line(-100500, None)
            rcts.assert_not_called()

    @pytest.mark.asyncio
    async def test_first_status_sends_new_message(self, transport):
        await transport._update_status_line(111, " • <i>Turning on kitchen…</i>")
        transport._application.bot.send_message.assert_called_once_with(
            chat_id=111, text=" • <i>Turning on kitchen…</i>", parse_mode="HTML",
            disable_notification=True, disable_web_page_preview=True)
        assert transport._status_msgs[111] == 42

    @pytest.mark.asyncio
    async def test_second_status_edits_the_same_message(self, transport):
        transport._status_msgs[111] = 42
        await transport._update_status_line(111, " • <i>Setting temperature…</i>")
        transport._application.bot.edit_message_text.assert_called_once_with(
            chat_id=111, message_id=42, text=" • <i>Setting temperature…</i>", parse_mode="HTML")
        transport._application.bot.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_edit_failure_falls_back_to_new_message(self, transport):
        transport._status_msgs[111] = 42
        transport._application.bot.edit_message_text = AsyncMock(side_effect=Exception("gone"))
        transport._application.bot.send_message = AsyncMock(return_value=MagicMock(message_id=43))
        await transport._update_status_line(111, " • <i>x</i>")
        assert transport._status_msgs[111] == 43

    @pytest.mark.asyncio
    async def test_not_modified_keeps_the_message(self, transport):
        transport._status_msgs[111] = 42
        transport._application.bot.edit_message_text = AsyncMock(side_effect=Exception(
            "Message is not modified: specified new message content and reply markup are exactly the same"))
        await transport._update_status_line(111, " • <i>Clicking on KÖP</i>")
        transport._application.bot.send_message.assert_not_called()
        assert transport._status_msgs[111] == 42

    @pytest.mark.asyncio
    async def test_concurrent_updates_dont_orphan_a_message(self, transport):
        send_calls = []

        async def slow_send(**kwargs):
            await asyncio.sleep(0)
            msg = MagicMock()
            msg.message_id = 42 + len(send_calls)
            send_calls.append(kwargs)
            return msg

        transport._application.bot.send_message = AsyncMock(side_effect=slow_send)
        await asyncio.gather(transport._update_status_line(111, " • <i>First…</i>"),
                             transport._update_status_line(111, " • <i>Second…</i>"))
        assert transport._application.bot.send_message.call_count == 1
        transport._application.bot.edit_message_text.assert_called_once()
        assert transport._status_msgs[111] == 42

    @pytest.mark.asyncio
    async def test_clear_racing_update_does_not_orphan_a_message(self, transport):
        async def slow_send(**kwargs):
            await asyncio.sleep(0)
            return MagicMock(message_id=42)

        transport._application.bot.send_message = AsyncMock(side_effect=slow_send)
        await asyncio.gather(transport._update_status_line(111, " • <i>Working…</i>"),
                             transport._clear_status_line(111))
        transport._application.bot.delete_message.assert_called_once_with(chat_id=111, message_id=42)
        assert 111 not in transport._status_msgs

    @pytest.mark.asyncio
    async def test_clear(self, transport):
        transport._status_msgs[111] = 42
        await transport._clear_status_line(111)
        transport._application.bot.delete_message.assert_called_once_with(chat_id=111, message_id=42)
        assert 111 not in transport._status_msgs
        await transport._clear_status_line(111)
        assert transport._application.bot.delete_message.call_count == 1
        transport._status_msgs[111] = 43
        transport._application.bot.delete_message = AsyncMock(side_effect=Exception("too old"))
        await transport._clear_status_line(111)
        assert 111 not in transport._status_msgs


@pytest.mark.unit
class TestApprovals:
    @pytest.mark.asyncio
    async def test_prompt_has_keyboard_and_escaped_body(self, transport):
        transport._application.bot.send_message = AsyncMock(return_value=MagicMock(message_id=99))
        body = "<b>Approval needed</b>\nSend &lt;hi&gt;\n\n<blockquote>details</blockquote>"
        assert await transport._send_approval_async(555, "tok", body) == 99
        kwargs = transport._application.bot.send_message.call_args.kwargs
        assert kwargs["chat_id"] == 555 and kwargs["text"] == body and kwargs["parse_mode"] == "HTML"
        buttons = kwargs["reply_markup"].inline_keyboard[0]
        assert [b.callback_data for b in buttons] == ["approval:approve:tok", "approval:deny:tok"]
        assert transport._approval_bodies[(555, 99)] == body

    def test_approval_body_escapes_title_and_quotes_details(self, transport):
        # An unescaped '<' in a tool title fails the HTML send and the approval gate shows nothing
        with patch.object(transport, '_send_approval_async', return_value=MagicMock()) as send, \
             patch.object(transport, '_call', side_effect=lambda coro, *a, **kw: coro):
            transport.send_approval_prompt(555, "tok", "Run rm <all> & wait", "df & du")
            assert send.call_args[0] == (555, "tok", "<b>Approval needed</b>\nRun rm &lt;all&gt; &amp; wait"
                                                     "\n\n<blockquote>df &amp; du</blockquote>")
            transport.send_approval_prompt(555, "tok", "Plain", "")
            assert send.call_args[0][2] == "<b>Approval needed</b>\nPlain"

    @pytest.mark.asyncio
    async def test_prompt_send_failure_returns_none(self, transport):
        transport._application.bot.send_message = AsyncMock(side_effect=Exception("boom"))
        assert await transport._send_approval_async(555, "tok", "body") is None

    @pytest.mark.asyncio
    async def test_close_edits_prompt_with_outcome(self, transport):
        transport._approval_bodies[(555, 99)] = "<b>Approval needed</b>"
        await transport._close_approval_async(555, 99, "✅ Approved by alice")
        transport._application.bot.edit_message_text.assert_called_once_with(
            chat_id=555, message_id=99, text="<b>Approval needed</b>\n\n<i>✅ Approved by alice</i>",
            parse_mode="HTML", reply_markup=None)
        assert (555, 99) not in transport._approval_bodies

    @pytest.mark.asyncio
    async def test_close_unknown_prompt_only_removes_keyboard(self, transport):
        transport._application.bot.edit_message_reply_markup = AsyncMock()
        await transport._close_approval_async(555, 98, "⏱ Timed out")
        transport._application.bot.edit_message_reply_markup.assert_called_once_with(chat_id=555, message_id=98, reply_markup=None)

    @staticmethod
    def _callback(data, chat_id=555):
        update = MagicMock()
        update.callback_query.data = data
        update.callback_query.message.chat.id = chat_id
        update.callback_query.from_user = MagicMock(username="alice", first_name="Alice", id=1)
        update.callback_query.answer = AsyncMock()
        update.callback_query.edit_message_reply_markup = AsyncMock()
        return update

    @pytest.mark.asyncio
    async def test_callback_resolves_through_the_hub(self, transport, hub):
        hub.resolve_approval.return_value = True
        await transport._handle_callback_query(self._callback("approval:approve:abc123"), MagicMock())
        hub.resolve_approval.assert_called_once_with("abc123", "approve", "alice")
        update = self._callback("approval:deny:abc123")
        await transport._handle_callback_query(update, MagicMock())
        update.callback_query.answer.assert_awaited_with("Denied.")

    @pytest.mark.asyncio
    async def test_callback_already_resolved_strips_keyboard(self, transport, hub):
        hub.resolve_approval.return_value = False
        update = self._callback("approval:approve:unknown")
        await transport._handle_callback_query(update, MagicMock())
        update.callback_query.answer.assert_awaited_with("Already resolved.")
        update.callback_query.edit_message_reply_markup.assert_awaited_once_with(reply_markup=None)

    @pytest.mark.asyncio
    async def test_callback_rejects_unauthorized_chat_and_malformed_data(self, hub):
        t = _make_transport(allowed=[555])
        t._hub = hub
        update = self._callback("approval:approve:any", chat_id=9999)
        await t._handle_callback_query(update, MagicMock())
        update.callback_query.answer.assert_awaited_with("Not authorized.", show_alert=True)
        update = self._callback("garbage")
        await t._handle_callback_query(update, MagicMock())
        update.callback_query.answer.assert_awaited_once_with()
        hub.resolve_approval.assert_not_called()


@pytest.mark.unit
class TestSecretPrompts:
    """A secret prompt makes the next text in that chat an answer: handed to the hub off the
    loop, deleted from the chat, and never logged, cached or forwarded."""

    @pytest.fixture
    def log_lines(self, monkeypatch):
        import common.logging as logging
        lines = []
        monkeypatch.setattr(logging, "log", lambda message, *a, **k: lines.append(message))
        return lines

    @pytest.mark.asyncio
    async def test_prompt_has_a_cancel_button_and_arms_the_chat(self, transport):
        transport._application.bot.send_message = AsyncMock(return_value=MagicMock(message_id=99))
        assert await transport._send_secret_async(555, "tok", "header", "body") == 99
        kwargs = transport._application.bot.send_message.call_args.kwargs
        assert kwargs["chat_id"] == 555 and kwargs["text"] == "body" and kwargs["parse_mode"] == "HTML"
        buttons = kwargs["reply_markup"].inline_keyboard[0]
        assert [b.callback_data for b in buttons] == ["secret:cancel:tok"]
        assert transport._secret_prompts[555] == (99, "tok", "header")

    @pytest.mark.asyncio
    async def test_re_asking_with_the_same_token_edits_the_prompt_in_place(self, transport):
        # A second message would land below the reply bubble already waiting for the outcome.
        transport._secret_prompts[555] = (99, "tok", "header")
        assert await transport._send_secret_async(555, "tok", "header", "body 2") == 99
        transport._application.bot.send_message.assert_not_called()
        kwargs = transport._application.bot.edit_message_text.call_args.kwargs
        assert kwargs["chat_id"] == 555 and kwargs["message_id"] == 99 and kwargs["text"] == "body 2"
        assert kwargs["reply_markup"].inline_keyboard[0][0].callback_data == "secret:cancel:tok"
        assert transport._secret_prompts[555] == (99, "tok", "header")
        # Another request's token in that chat is a new message.
        transport._application.bot.send_message = AsyncMock(return_value=MagicMock(message_id=100))
        assert await transport._send_secret_async(555, "tok2", "h2", "b2") == 100
        assert transport._secret_prompts[555] == (100, "tok2", "h2")

    def test_send_secret_prompt_builds_the_body_and_returns_ref(self, transport):
        with patch.object(transport, '_send_secret_async', return_value=MagicMock()) as send, \
             patch.object(transport, '_call', side_effect=lambda coro, *a, **kw: coro):
            transport.send_secret_prompt(555, "tok", "Amex <card>", "")
            assert send.call_args[0] == (555, "tok", "🔐 <b>Amex &lt;card&gt;</b>",
                                         "🔐 <b>Amex &lt;card&gt;</b>\n" + transport.SECRET_INSTRUCTIONS)
            transport.send_secret_prompt(555, "tok", "Visa (card)", "⚠ too short & try again")
            assert send.call_args[0][3].endswith("\n\n<blockquote>⚠ too short &amp; try again</blockquote>")
        t = _make_transport(running=False)
        assert t.send_secret_prompt(555, "tok", "x", "") is None

    @pytest.mark.asyncio
    async def test_prompt_send_failure_returns_none_and_arms_nothing(self, transport):
        transport._application.bot.send_message = AsyncMock(side_effect=Exception("boom"))
        assert await transport._send_secret_async(555, "tok", "header", "body") is None
        assert 555 not in transport._secret_prompts

    @pytest.mark.asyncio
    async def test_answer_is_claimed_deleted_and_never_logged_cached_or_forwarded(self, transport, hub, log_lines):
        transport._secret_prompts[123] = (99, "tok", "body")
        hub.claim_secret.return_value = True
        update = _update(chat_id=123, message_id=10, text="4111 1111 1111 1111")
        await transport._handle_message(update, MagicMock())
        hub.claim_secret.assert_called_once_with(transport, 123, "4111 1111 1111 1111", "Anna")
        transport._application.bot.delete_message.assert_awaited_once_with(chat_id=123, message_id=10)
        hub.receive.assert_not_called()
        assert (123, 10) not in transport._message_texts
        assert not any("4111" in line for line in log_lines), log_lines

    @pytest.mark.asyncio
    async def test_text_for_a_prompt_that_just_closed_is_deleted_and_explained(self, transport, hub):
        transport._secret_prompts[123] = (99, "tok", "body")
        hub.claim_secret.return_value = False
        transport._application.bot.send_message = AsyncMock(return_value=MagicMock(message_id=5))
        await transport._handle_message(_update(chat_id=123, message_id=10, text="secret"), MagicMock())
        transport._application.bot.delete_message.assert_awaited_once_with(chat_id=123, message_id=10)
        hub.receive.assert_not_called()
        assert "already closed" in transport._application.bot.send_message.call_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_a_chat_without_a_prompt_is_ordinary_chat(self, transport, hub):
        transport._secret_prompts[999] = (99, "tok", "body")
        await transport._handle_message(_update(chat_id=123), MagicMock())
        hub.claim_secret.assert_not_called()
        received(hub)

    @pytest.mark.asyncio
    async def test_no_hub_deletes_the_answer_and_apologises(self, transport):
        transport._secret_prompts[123] = (99, "tok", "body")
        update = _update(chat_id=123, message_id=10, text="secret")
        await transport._handle_message(update, MagicMock())
        transport._application.bot.delete_message.assert_awaited_once_with(chat_id=123, message_id=10)
        assert "not available" in update.message.reply_text.call_args[0][0]

    @pytest.mark.asyncio
    async def test_close_disarms_the_chat_at_once_and_edits_the_prompt(self, transport):
        transport._secret_prompts[555] = (99, "tok", "🔐 <b>Amex</b>")
        with patch.object(transport, '_schedule', side_effect=lambda coro, what: (coro.close(), True)[1]) as sched:
            transport.close_secret_prompt(555, 99, "✅ Stored as amex")
        assert 555 not in transport._secret_prompts
        assert sched.call_args[0][1] == "secret prompt close"
        # The instructions are gone once the prompt is over: title and outcome only.
        await transport._close_secret_async(555, 99, "🔐 <b>Amex</b>", "✅ Stored as amex")
        transport._application.bot.edit_message_text.assert_called_once_with(
            chat_id=555, message_id=99, text="🔐 <b>Amex</b>\n<i>✅ Stored as amex</i>", parse_mode="HTML",
            reply_markup=None)

    @pytest.mark.asyncio
    async def test_close_of_a_superseded_prompt_keeps_the_new_one_armed(self, transport):
        transport._secret_prompts[555] = (100, "tok", "body")
        transport._application.bot.edit_message_reply_markup = AsyncMock()
        with patch.object(transport, '_schedule', side_effect=lambda coro, what: (coro.close(), True)[1]):
            transport.close_secret_prompt(555, 99, "⚠ too short")
        assert transport._secret_prompts[555] == (100, "tok", "body")
        await transport._close_secret_async(555, 99, None, "⚠ too short")
        transport._application.bot.edit_message_reply_markup.assert_called_once_with(
            chat_id=555, message_id=99, reply_markup=None)

    @pytest.mark.asyncio
    async def test_cancel_button_goes_through_the_hub(self, transport, hub):
        hub.cancel_secret.return_value = True
        update = TestApprovals._callback("secret:cancel:abc123")
        await transport._handle_callback_query(update, MagicMock())
        hub.cancel_secret.assert_called_once_with("abc123", "alice")
        update.callback_query.answer.assert_awaited_with("Cancelled.")
        hub.resolve_approval.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_of_a_closed_prompt_strips_the_button(self, transport, hub):
        hub.cancel_secret.return_value = False
        update = TestApprovals._callback("secret:cancel:old")
        await transport._handle_callback_query(update, MagicMock())
        update.callback_query.answer.assert_awaited_with("Already closed.")
        update.callback_query.edit_message_reply_markup.assert_awaited_once_with(reply_markup=None)

    @pytest.mark.asyncio
    async def test_other_secret_actions_are_ignored(self, transport, hub):
        update = TestApprovals._callback("secret:approve:abc")
        await transport._handle_callback_query(update, MagicMock())
        hub.cancel_secret.assert_not_called()
        hub.resolve_approval.assert_not_called()
        update.callback_query.answer.assert_awaited_once_with()


class TestAllowedChatIds:
    def test_ids_written_as_text_still_authorise(self):
        # config_set stores the id as text, and a hand edit may quote it; Telegram sends ints
        t = _make_transport(allowed=("123456", 789, " 42 "), running=False)
        assert t._allowed_chat_ids == [123456, 789, 42]
        assert t._is_authorized(123456) and t._is_authorized(42)
        assert not t._is_authorized(1)

    def test_a_non_numeric_entry_is_ignored_not_fatal(self):
        t = _make_transport(allowed=("abc", 5), running=False)
        assert t._allowed_chat_ids == [5]

    def test_notifications_go_to_every_allowed_chat_by_default(self):
        t = _make_transport(allowed=(111, 222), running=False)
        assert t.notification_conversations() == [111, 222]

    def test_a_group_on_the_allow_list_is_not_an_owner_conversation(self):
        # What is the owners' alone — the pushes by default, the cards, the brief — must not
        # land in front of a group's guests; a group gets pushes only when notify_chat_ids names it.
        t = _make_transport(allowed=(111, -5001), running=False)
        assert t.owner_conversations() == [111]
        assert t.notification_conversations() == [111]
        assert t._is_authorized(-5001)
        t.config_values['telegram_bot.notify_chat_ids'] = [-5001]
        assert t.notification_conversations() == [-5001]

    def test_notify_chat_ids_narrow_the_pushes_but_not_who_may_talk(self):
        t = _make_transport(allowed=(111, 222), notify=("111",), running=False)
        assert t.notification_conversations() == [111]
        assert t.owner_conversations() == [111, 222]
        assert t._is_authorized(222)

    def test_a_config_change_is_in_force_without_a_restart(self):
        # The FTUE assistant once restarted the bot from the Telegram chat that had just sent
        # its chat id — that killed the conversation. The lists are read live instead.
        t = _make_transport(allowed=(), running=False)
        assert t._is_authorized(999)
        t.config_values['telegram_bot.allowed_chat_ids'] = ["50017001"]
        assert t._is_authorized(50017001)
        assert not t._is_authorized(999)
        assert t.owner_conversations() == [50017001]
        t.config_values['telegram_bot.notify_chat_ids'] = [7]
        assert t.notification_conversations() == [7]

    def test_a_single_id_and_no_ids(self):
        from interfaces.telegram.TelegramTransport import TelegramTransport
        assert TelegramTransport._parse_chat_ids(123) == [123]
        assert TelegramTransport._parse_chat_ids(None) == []
        assert TelegramTransport._parse_chat_ids([]) == []


@pytest.mark.unit
class TestConfigured:
    def test_a_copied_example_placeholder_is_no_token(self):
        t = _make_transport(token='YOUR_TELEGRAM_BOT_TOKEN_HERE', running=False)
        assert t.is_configured is False
        assert t.check_health() is False

    def test_reset_leaves_the_hub_and_stops(self, transport, hub):
        with patch.object(transport, 'stop') as stop:
            transport._on_reset()
        hub.unregister.assert_called_once_with(transport)
        stop.assert_called_once()

    def test_reset_without_a_hub_just_stops(self):
        t = _make_transport(running=False)
        with patch.object(t, 'stop') as stop:
            t._on_reset()
        stop.assert_called_once()


@pytest.mark.unit
class TestGroupsAndGuests:
    """In a group the bot answers replies to its own bubbles only (her name said there reaches her
    through the Telegram Feed); the positive ids of the allow list are the owners, anyone else is
    a guest whose turn is marked as such."""

    @pytest.mark.asyncio
    async def test_a_group_message_not_replying_to_the_bot_is_ignored(self):
        transport, hub = _bot_in_group()
        update = _update(chat_id=-5001, title="Family", text="what's for dinner?")
        await transport._handle_message(update, MagicMock())
        hub.receive.assert_not_called()
        update.message.reply_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_reply_to_someone_else_in_the_group_is_ignored(self):
        transport, hub = _bot_in_group()
        await transport._handle_message(_update(chat_id=-5001, reply=_bubble_of(8)), MagicMock())
        hub.receive.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_reply_to_the_bot_in_the_group_is_answered_with_the_quote(self):
        transport, hub = _bot_in_group()
        await transport._handle_message(_update(chat_id=-5001, reply=_bubble_of(777, "the forecast")), MagicMock())
        msg = received(hub)
        assert msg.reply_to_text == "the forecast" and msg.conversation == -5001
        assert msg.reply_to_bot is True  # the chat layer answers it instead of allowing a NO

    @pytest.mark.asyncio
    async def test_a_photo_in_the_group_follows_the_same_rule(self):
        transport, hub = _bot_in_group()
        update = _update(chat_id=-5001)
        update.message.photo = [MagicMock(file_id="f1")]
        await transport._handle_photo(update, _file_context())
        hub.receive.assert_not_called()
        hub.expect_photo.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_owner_and_a_guest_are_told_apart_by_user_id(self):
        transport, hub = _bot_in_group()
        await transport._handle_message(_update(chat_id=-5001, reply=_bubble_of(777), user_id=42), MagicMock())
        assert received(hub).guest is False
        hub.receive.reset_mock()
        await transport._handle_message(_update(chat_id=-5001, reply=_bubble_of(777), user_id=8), MagicMock())
        msg = received(hub)
        assert msg.guest is True and msg.sender_is_bot is False

    @pytest.mark.asyncio
    async def test_another_bot_is_marked_as_one(self):
        transport, hub = _bot_in_group()
        update = _update(chat_id=-5001, reply=_bubble_of(777), user_id=888, is_bot=True, first_name="Erik's Keep")
        await transport._handle_message(update, MagicMock())
        msg = received(hub)
        assert msg.sender_is_bot is True and msg.guest is True and msg.sender == "Erik's Keep"

    @pytest.mark.asyncio
    async def test_an_empty_allow_list_makes_everyone_an_owner(self, transport, hub):
        await transport._handle_message(_update(user_id=8), MagicMock())
        assert received(hub).guest is False

    @pytest.mark.asyncio
    async def test_an_empty_allow_list_serves_no_group(self):
        # Everyone an owner is fine in a private chat, the person's own; in a group it would
        # hand the household to every member, so a group is served only when its id is listed.
        transport, hub = _bot_in_group(allowed=())
        update = _update(chat_id=-5001, reply=_bubble_of(777), user_id=8)
        await transport._handle_message(update, MagicMock())
        hub.receive.assert_not_called()
        update.message.reply_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_reaction_in_a_group_is_ignored_while_the_allow_list_is_empty(self):
        # With no allow list everyone is an owner: a reaction in a group the bot was dragged
        # into would start an owner's turn with one tap, as a text there used to.
        transport, hub = _bot_in_group(allowed=())
        update = TestInboundReaction._reaction_update(chat_id=-5001, new=["👍"])
        update.effective_user.id = 8
        await transport._handle_reaction(update, MagicMock())
        hub.receive.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_guests_reaction_in_the_group_is_a_guest_turn_of_that_group(self):
        transport, hub = _bot_in_group()
        transport._track_message_text(-5001, 10, "dinner is pasta")
        update = TestInboundReaction._reaction_update(chat_id=-5001, new=["👍"])
        update.effective_user.id = 8
        update.effective_user.is_bot = False
        update.effective_chat.title = "Family"
        await transport._handle_reaction(update, MagicMock())
        msg = received(hub)
        assert msg.kind == KIND_REACTION and msg.guest is True and msg.chat_name == "Family"
        assert msg.reaction_target_text == "dinner is pasta"

    @pytest.mark.asyncio
    async def test_a_guests_voice_reply_to_the_bot_is_marked_a_guest_turn_too(self):
        # A guest's turn runs no commands; the mark must survive every kind of message, not
        # only text (a captioned photo once lost it).
        transport, hub = _bot_in_group()
        update = _update(chat_id=-5001, reply=_bubble_of(777), user_id=8, title="Family")
        update.message.voice = MagicMock(file_id="v1", mime_type="audio/ogg")
        await transport._handle_voice(update, _file_context(b"OggS"))
        msg = received(hub)
        assert msg.kind == KIND_VOICE and msg.guest is True and msg.sender_is_bot is False
        assert msg.chat_name == "Family" and msg.conversation == -5001

    @pytest.mark.asyncio
    async def test_a_secret_reply_from_a_guest_is_deleted_not_taken(self):
        transport, hub = _bot_in_group()
        transport._secret_prompts[-5001] = (99, "tok", "body")
        update = _update(chat_id=-5001, reply=_bubble_of(777), user_id=8, text="hunter2")
        await transport._handle_message(update, MagicMock())
        hub.claim_secret.assert_not_called()
        hub.receive.assert_not_called()
        transport._application.bot.delete_message.assert_awaited_once_with(chat_id=-5001, message_id=10)

    @staticmethod
    def _tap(chat_id, user_id):
        update = MagicMock()
        update.callback_query.data = "approval:approve:abc123"
        update.callback_query.message.chat.id = chat_id
        update.callback_query.from_user = MagicMock(username="erik", first_name="Erik", id=user_id)
        update.callback_query.answer = AsyncMock()
        update.callback_query.edit_message_reply_markup = AsyncMock()
        return update

    @pytest.mark.asyncio
    async def test_an_approval_tap_by_a_guest_is_refused(self):
        transport, hub = _bot_in_group()
        update = self._tap(-5001, user_id=8)
        await transport._handle_callback_query(update, MagicMock())
        hub.resolve_approval.assert_not_called()
        assert update.callback_query.answer.await_args.kwargs.get("show_alert") is True

    @pytest.mark.asyncio
    async def test_an_approval_tap_by_the_owner_in_the_group_counts(self):
        transport, hub = _bot_in_group()
        hub.resolve_approval.return_value = True
        await transport._handle_callback_query(self._tap(-5001, user_id=42), MagicMock())
        hub.resolve_approval.assert_called_once_with("abc123", "approve", "erik")
