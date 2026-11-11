"""The feed and the assistant's own Messages account: a chat that account is in — the 1:1 chat
with the assistant, a group it is a member of — is the iMessage chat interface's to read, and
the feed of the owner's account leaves its rows out instead of storing every bubble twice."""
from unittest.mock import patch

import pytest

from common.ConfigManager import config
from common.imessage.MessagesHost import HostHealth, Message, UNREACHABLE
from integrations.imessage import IMessageIntegration as module
from integrations.imessage.IMessageIntegration import CHAT_ADDRESSES_RETRY, CHAT_ENABLED_KEY, IMessageIntegration

from .host_fakes import FUTURE, FakeHost, RECENT, received, send_later, sent

ASSISTANT = "keep@example.com"
ALICE = "+46701740605"
SATELLITE = "http://192.0.2.7:8898"


def in_chat(message: Message, *participants) -> Message:
    message.participants = sorted(participants)
    return message


@pytest.fixture(autouse=True)
def reset_singleton():
    IMessageIntegration.reset_instance()
    yield
    IMessageIntegration.reset_instance()


@pytest.fixture(autouse=True)
def no_injection_screening():
    with patch.object(module, "is_prompt_injection", return_value=False):
        yield


@pytest.fixture
def settings():
    """The config keys the feed decides by; a test changes them in place."""
    keys = {CHAT_ENABLED_KEY: True, "imessage_chat.host": SATELLITE, "imessage.host": "local"}
    real = config.get
    with patch.object(config, "get", side_effect=lambda key, default=None: keys[key] if key in keys
                      else real(key, default)):
        yield keys


@pytest.fixture
def chat_host():
    """The host of the assistant's own account, as the feed asks it who the account is."""
    host = FakeHost()
    host.health_answer = HostHealth(True, addresses=[ASSISTANT])
    return host


@pytest.fixture
def feed(settings, chat_host):
    integration = IMessageIntegration()
    integration.host = FakeHost()
    with patch.object(module.Hosts, "build", return_value=chat_host):
        yield integration


def stored(feed, *rows, backfill=False) -> list[str]:
    feed.host.rows = list(rows)
    return [entry.content for entry in feed._poll_new_messages(backfill)]


@pytest.mark.unit
class TestChatsTheAssistantIsIn:
    def test_what_the_owner_writes_to_the_assistant_is_not_a_feed_row(self, feed):
        assert stored(feed, in_chat(sent(1, "Post to the family chat when the sofa ships", ["Keep"]), ASSISTANT)) == []

    def test_what_the_assistant_answers_is_not_a_feed_row(self, feed):
        assert stored(feed, in_chat(received(1, "Will do", "Keep", ASSISTANT), ASSISTANT)) == []

    def test_a_group_the_assistant_is_a_member_of_is_left_out_whoever_writes(self, feed):
        rows = [in_chat(received(1, "Dinner?", "Alice", ALICE, chat_name="Family"), ALICE, ASSISTANT),
                in_chat(sent(2, "At seven", chat_name="Family"), ALICE, ASSISTANT)]
        assert stored(feed, *rows) == []

    def test_every_other_chat_is_stored_as_before(self, feed):
        rows = [in_chat(received(1, "Hello", "Alice", ALICE), ALICE),
                in_chat(sent(2, "Post to the family chat", ["Keep"]), ASSISTANT),
                in_chat(sent(3, "Hi Alice", ["Alice"]), ALICE)]
        contents = stored(feed, *rows)
        assert len(contents) == 2 and "Hello" in contents[0] and "Hi Alice" in contents[1]

    def test_the_cursor_passes_the_rows_left_out(self, feed):
        stored(feed, in_chat(sent(7, "Hello", ["Keep"]), ASSISTANT))
        assert feed._last_seen_rowid == 7

    def test_the_catch_up_leaves_them_out_too(self, feed):
        assert stored(feed, in_chat(sent(1, "Hello", ["Keep"]), ASSISTANT), backfill=True) == []

    def test_an_address_is_the_same_however_it_is_cased_or_spaced(self, feed, chat_host):
        chat_host.health_answer = HostHealth(True, addresses=["Keep@Example.com", "+46 70-174 06 07"])
        rows = [in_chat(sent(1, "By mail", ["Keep"]), "keep@example.com"),
                in_chat(sent(2, "By phone", ["Keep"]), "+46701740607")]
        assert stored(feed, *rows) == []

    def test_a_participant_is_the_same_however_the_owners_chat_db_spells_it(self, feed):
        """The owner's chat.db keeps a handle as it was first typed; the assistant's host names it normalised."""
        rows = [in_chat(sent(1, "By mail", ["Keep"]), " Keep@Example.COM"),
                in_chat(received(2, "Dinner?", "Alice", ALICE, chat_name="Family"), ALICE, "KEEP@example.com")]
        assert stored(feed, *rows) == []

    def test_a_message_from_the_assistant_is_known_by_its_sender_when_the_host_names_no_participants(self, feed):
        """The owner's host a version behind: `Message.participants` is empty."""
        assert stored(feed, received(1, "Will do", "Keep", ASSISTANT)) == []

    def test_a_message_scheduled_for_the_assistant_is_not_followed_until_it_sends(self, feed):
        stored(feed, in_chat(send_later(1, FUTURE, recipients=["Keep"]), ASSISTANT))
        assert feed._scheduled_rowids == set()


@pytest.mark.unit
class TestWhenNothingIsLeftOut:
    def test_without_the_chat_interface_the_assistants_host_is_never_asked(self, feed, settings, chat_host):
        settings[CHAT_ENABLED_KEY] = False
        assert len(stored(feed, in_chat(sent(1, "Hello", ["Keep"]), ASSISTANT))) == 1
        assert chat_host.calls == []

    def test_a_chat_given_the_owners_own_account_names_nobody_to_leave_out(self, feed, settings, chat_host):
        """The chat interface refuses to run there; the account's addresses are the owner's."""
        settings["imessage_chat.host"] = "local"
        assert len(stored(feed, in_chat(sent(1, "Hello", ["Keep"]), ASSISTANT))) == 1
        assert chat_host.calls == []

    def test_until_the_assistants_host_has_named_an_address_everything_is_stored(self, feed, chat_host):
        chat_host.health_answer = HostHealth(False, UNREACHABLE)
        assert len(stored(feed, in_chat(sent(1, "Hello", ["Keep"]), ASSISTANT))) == 1

    def test_a_host_that_names_no_address_leaves_everything_in(self, feed, chat_host):
        chat_host.health_answer = HostHealth(True)
        assert len(stored(feed, in_chat(sent(1, "Hello", ["Keep"]), ASSISTANT))) == 1

    def test_the_chat_interface_turned_off_gives_the_chats_back_to_the_feed(self, feed, settings):
        assert stored(feed, in_chat(sent(1, "Hello", ["Keep"]), ASSISTANT)) == []
        settings[CHAT_ENABLED_KEY] = False
        assert len(stored(feed, in_chat(sent(2, "Hello again", ["Keep"]), ASSISTANT))) == 1


@pytest.mark.unit
class TestReadingTheAddresses:
    def test_the_assistants_host_is_not_asked_at_every_poll(self, feed, chat_host):
        for rowid in (1, 2, 3):
            stored(feed, in_chat(sent(rowid, "Hello", ["Keep"]), ASSISTANT))
        assert chat_host.calls == [("health", False)]

    def test_a_host_that_went_away_keeps_the_addresses_it_named(self, feed, chat_host):
        stored(feed, in_chat(sent(1, "Hello", ["Keep"]), ASSISTANT))
        chat_host.health_answer = HostHealth(False, UNREACHABLE)
        feed._chat_addresses_due = 0
        assert stored(feed, in_chat(sent(2, "Still there?", ["Keep"]), ASSISTANT)) == []

    def test_an_account_that_changed_its_address_is_followed(self, feed, chat_host):
        stored(feed, in_chat(sent(1, "Hello", ["Keep"]), ASSISTANT))
        chat_host.health_answer = HostHealth(True, addresses=["new@example.com"])
        feed._chat_addresses_due = 0
        rows = [in_chat(sent(2, "To the old one", ["Somebody"]), ASSISTANT),
                in_chat(sent(3, "To the new one", ["Keep"]), "new@example.com")]
        contents = stored(feed, *rows)
        assert len(contents) == 1 and "To the old one" in contents[0]

    def test_a_host_away_at_the_first_read_is_asked_again_soon_and_then_left_alone(self, feed, chat_host):
        """Until the account has named an address every bubble with the assistant is stored twice:
        that is not waited out for as long as addresses already read are trusted."""
        chat_host.health_answer = HostHealth(False, UNREACHABLE)
        clock = [5000.0]
        with patch.object(module.time, "monotonic", side_effect=lambda: clock[0]):
            assert len(stored(feed, in_chat(sent(1, "Hello", ["Keep"]), ASSISTANT))) == 1
            chat_host.health_answer = HostHealth(True, addresses=[ASSISTANT])
            clock[0] += CHAT_ADDRESSES_RETRY - 1
            assert len(stored(feed, in_chat(sent(2, "Hello?", ["Keep"]), ASSISTANT))) == 1
            assert len(chat_host.calls) == 1, "not at every poll either"
            clock[0] += 2
            assert stored(feed, in_chat(sent(3, "There you are", ["Keep"]), ASSISTANT)) == []
            clock[0] += CHAT_ADDRESSES_RETRY + 1
            stored(feed, in_chat(sent(4, "Still there", ["Keep"]), ASSISTANT))
            assert len(chat_host.calls) == 2, "once it has answered, the next read is the refresh's"

    def test_a_chat_moved_to_another_account_is_asked_who_it_is_at_the_next_poll(self, feed, settings):
        """`imessage_chat.host` now names another Mac: the chats of the account before are the
        feed's again, and the new account's are left out without waiting for the refresh."""
        stored(feed, in_chat(sent(1, "Hello", ["Keep"]), ASSISTANT))
        moved = FakeHost()
        moved.health_answer = HostHealth(True, addresses=["new@example.com"])
        settings["imessage_chat.host"] = "http://192.0.2.8:8898"
        with patch.object(module.Hosts, "build", return_value=moved):
            contents = stored(feed, in_chat(sent(2, "To the old one", ["Somebody"]), ASSISTANT),
                              in_chat(sent(3, "To the new one", ["Keep"]), "new@example.com"))
        assert len(contents) == 1 and "To the old one" in contents[0]
