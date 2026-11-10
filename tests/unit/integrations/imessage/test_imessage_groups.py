"""send_imessage to a group chat, end to end over a host on a chat.db of Apple's shape: the lookup
by name or members, the AppleScript that addresses the chat, the approval card and the history
echo. The lookup rules themselves are tested with the host layer (tests/unit/common/imessage)."""
import sqlite3
from unittest.mock import Mock, patch

import pytest

from common.imessage import MessagesSender as sender_module
from common.imessage.Contacts import Contacts
from common.imessage.LocalMessagesHost import LocalMessagesHost
from common.imessage.MessagesDatabase import MessagesDatabase
from integrations.imessage.IMessageIntegration import IMessageIntegration

FAMILY_GUID, PAIR_GUID = "any;+;chat100", "any;+;chat200"
NAMES = {"+46701740605": "Alice Smith", "+46701740606": "Bob Jones", "+46701740607": "Charlotte Smith",
         "+46701740608": "Frances Smith"}


def make_chat_db(path):
    """The group "Family" (Charlotte, Frances) and an unnamed group of Alice and Bob."""
    conn = sqlite3.connect(path)
    conn.executescript(f"""
        CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, guid TEXT, style INTEGER, display_name TEXT);
        CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT, service TEXT);
        CREATE TABLE chat_handle_join (chat_id INTEGER, handle_id INTEGER);
        CREATE TABLE message (ROWID INTEGER PRIMARY KEY, date INTEGER);
        CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER);
        INSERT INTO chat VALUES (1, '{FAMILY_GUID}', 43, 'Family'), (2, '{PAIR_GUID}', 43, NULL);
        INSERT INTO handle VALUES (10, '+46701740605', 'iMessage'), (11, '+46701740606', 'iMessage'),
                                  (12, '+46701740607', 'iMessage'), (13, '+46701740608', 'iMessage');
        INSERT INTO chat_handle_join VALUES (1, 12), (1, 13), (2, 10), (2, 11);
        INSERT INTO message VALUES (100, 5), (101, 9);
        INSERT INTO chat_message_join VALUES (1, 100), (2, 101);
    """)
    conn.commit()
    conn.close()


@pytest.fixture(autouse=True)
def reset_singleton():
    IMessageIntegration.reset_instance()
    yield
    IMessageIntegration.reset_instance()


@pytest.fixture
def integration(tmp_path):
    """The instance under test, and the one `IMessageIntegration.get_instance()` hands the
    command set, over a host whose chat.db and contacts are the fixture's — never the Mac's."""
    make_chat_db(tmp_path / "chat.db")
    database = MessagesDatabase(tmp_path / "chat.db")
    contacts = Contacts(database, tmp_path / "AddressBook")
    contacts._contact_cache = dict(NAMES)
    contacts._name_to_handle = {"alice": "+46701740605"}
    contacts._contact_cache_time = 1e18        # the contact cache is fresh, never refreshed from disk
    m = IMessageIntegration()
    m.host = LocalMessagesHost(database, contacts)
    with patch.object(IMessageIntegration, "get_instance", return_value=m):
        yield m


def _osascript(stdout="sent:group"):
    return patch.object(sender_module.subprocess, "run", return_value=Mock(returncode=0, stdout=stdout, stderr=""))


@pytest.mark.unit
class TestGroupSend:
    def test_a_group_send_addresses_the_chat_by_id_and_echoes_the_group(self, integration):
        with _osascript() as run:
            assert integration.send("Family", "dinner at 7") == ""
        script = run.call_args[0][0][2]
        assert f'chat id "{FAMILY_GUID}"' in script and "buddy" not in script
        assert '"dinner at 7"' in script
        echo = integration._pending_entries[0].content
        assert "to Family:" in echo and "dinner at 7" in echo

    def test_an_unnamed_group_is_echoed_by_its_members(self, integration):
        with _osascript():
            integration.send("Alice, Bob", "hi both")
        assert "to Alice Smith, Bob Jones:" in integration._pending_entries[0].content

    def test_the_owner_in_the_member_list_is_ignored(self, integration):
        assert integration.resolve_recipient("Marcus, Alice & Bob") == (PAIR_GUID, None)

    def test_a_contact_send_still_addresses_a_buddy(self, integration):
        with _osascript("sent:imessage") as run:
            integration.send("Alice", "hi")
        assert "buddy" in run.call_args[0][0][2] and "chat id" not in run.call_args[0][0][2]
        assert "to Alice Smith: hi" in integration._pending_entries[0].content

    def test_the_resolved_group_is_known_afterwards(self, integration):
        handle, _ = integration.resolve_recipient("Alice & Bob")
        assert integration.is_group_guid(handle)
        assert integration.group_for_handle(handle).members == ["Alice Smith", "Bob Jones"]


@pytest.mark.unit
class TestGroupApprovalCard:
    def test_the_card_names_the_group_and_its_members(self, integration, commands_instance):
        text = commands_instance.describe("send_imessage", ["Alice, Bob", "hello"])
        assert "group" in text and "Alice Smith, Bob Jones" in text
        assert "any;+;" not in text

    def test_the_card_puts_the_members_behind_a_named_group(self, integration, commands_instance):
        text = commands_instance.describe("send_imessage", "Family, hello")
        assert "Family (Charlotte Smith, Frances Smith)" in text
