"""send_imessage to a group chat: the lookup by name or members, the AppleScript that addresses
the chat, the approval card and the history echo."""
import sqlite3
from unittest.mock import Mock, patch

import pytest

from integrations.imessage.IMessageIntegration import IMessageIntegration, GroupChat


FAMILY = GroupChat("any;+;chat100", "Family", ("Charlotte Smith", "Frances Smith", "Ingeborg Smith"), 900)
PAIR = GroupChat("any;+;chat200", None, ("Alice Smith", "Bob Jones"), 800)
PAIR_OLD = GroupChat("any;+;chat201", None, ("Alice Smith", "Bob Jones"), 100)   # a recreated group
TRIO = GroupChat("any;+;chat300", None, ("Alice Smith", "Bob Jones", "Carol King"), 700)
GROUPS = [FAMILY, PAIR, PAIR_OLD, TRIO]


@pytest.fixture(autouse=True)
def reset_singleton():
    IMessageIntegration.reset_instance()
    yield
    IMessageIntegration.reset_instance()


@pytest.fixture
def integration():
    """The instance under test, and the one `IMessageIntegration.get_instance()` hands the
    command set — a second instance would start with a stale cache and read the machine's
    AddressBook and chat.db."""
    m = IMessageIntegration()
    m._contact_cache_time = 1e18        # the contact cache is fresh, never refreshed from disk
    m._name_to_handle = {"alice": "+12345678901"}
    with patch.object(IMessageIntegration, "_load_group_chats", return_value=list(GROUPS)), \
            patch.object(IMessageIntegration, "get_instance", return_value=m):
        yield m


def _osascript(stdout="sent:group"):
    return Mock(returncode=0, stdout=stdout, stderr="")


@pytest.mark.unit
class TestGroupResolution:
    def test_a_named_group_resolves_to_its_guid(self, integration):
        assert integration.resolve_recipient("family") == (FAMILY.guid, None)

    def test_a_contact_name_still_wins_over_the_group_lookup(self, integration):
        assert integration.resolve_recipient("Alice") == ("+12345678901", None)

    @pytest.mark.parametrize("recipient", ["Alice, Bob", "Bob & Alice", "alice and bob", "Alice Smith, Bob Jones"])
    def test_an_unnamed_group_resolves_by_its_members_in_any_order_and_spelling(self, integration, recipient):
        handle, err = integration.resolve_recipient(recipient)
        assert err is None
        assert handle == PAIR.guid, "the most recently active of the two look-alike groups"

    def test_the_owner_in_the_member_list_is_ignored(self, integration):
        assert integration.resolve_recipient("Marcus, Alice & Bob")[0] == PAIR.guid

    def test_a_member_list_no_group_has_exactly_names_the_groups_that_contain_them(self, integration):
        handle, err = integration.resolve_recipient("Alice, Carol")
        assert handle is None
        assert "Alice Smith, Bob Jones, Carol King" in err
        assert "Charlotte" not in err

    def test_a_single_unknown_name_is_still_an_unknown_contact(self, integration):
        handle, err = integration.resolve_recipient("Zelda")
        assert handle is None and "unknown contact" in err

    def test_the_resolved_group_is_known_afterwards(self, integration):
        handle, _ = integration.resolve_recipient("Alice & Bob")
        assert integration.is_group_guid(handle)
        assert integration.group_for_handle(handle) is PAIR
        assert not integration.is_group_guid("+12345678901")


@pytest.mark.unit
class TestGroupSend:
    def test_a_group_send_addresses_the_chat_by_id_and_echoes_the_group(self, integration):
        with patch("integrations.imessage.IMessageIntegration.subprocess.run", return_value=_osascript()) as run:
            assert integration.send("Family", "dinner at 7") == ""
        script = run.call_args[0][0][2]
        assert f'chat id "{FAMILY.guid}"' in script and "buddy" not in script
        assert '"dinner at 7"' in script
        echo = integration._pending_entries[0].content
        assert "to Family:" in echo and "dinner at 7" in echo

    def test_an_unnamed_group_is_echoed_by_its_members(self, integration):
        with patch("integrations.imessage.IMessageIntegration.subprocess.run", return_value=_osascript()):
            integration.send("Alice, Bob", "hi both")
        assert "to Alice Smith, Bob Jones:" in integration._pending_entries[0].content

    def test_a_contact_send_still_addresses_a_buddy(self, integration):
        with patch("integrations.imessage.IMessageIntegration.subprocess.run",
                   return_value=_osascript("sent:imessage")) as run:
            integration.send("Alice", "hi")
        assert "buddy" in run.call_args[0][0][2] and "chat id" not in run.call_args[0][0][2]


@pytest.mark.unit
class TestGroupApprovalCard:
    def test_the_card_names_the_group_and_its_members(self, integration, commands_instance):
        text = commands_instance.describe("send_imessage", ["Alice, Bob", "hello"])
        assert "group" in text and "Alice Smith, Bob Jones" in text
        assert "any;+;" not in text

    def test_the_card_puts_the_members_behind_a_named_group(self, integration, commands_instance):
        text = commands_instance.describe("send_imessage", "Family, hello")
        assert "Family (Charlotte Smith, Frances Smith, Ingeborg Smith)" in text


@pytest.mark.unit
class TestGroupChatQuery:
    """`_load_group_chats` against a chat.db of Apple's shape: members per chat, the newest message
    as the activity, 1:1 chats left out."""

    @staticmethod
    def _chat_db(path):
        conn = sqlite3.connect(path)
        conn.executescript("""
            CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, guid TEXT, style INTEGER, display_name TEXT);
            CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT);
            CREATE TABLE chat_handle_join (chat_id INTEGER, handle_id INTEGER);
            CREATE TABLE message (ROWID INTEGER PRIMARY KEY, date INTEGER);
            CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER);
            INSERT INTO chat VALUES (1, 'any;+;chat1', 43, 'Family'), (2, 'any;+;chat2', 43, ''),
                                    (3, 'any;-;+46701740605', 45, NULL);
            INSERT INTO handle VALUES (10, '+46701740605'), (11, '+46701740606'), (12, 'carol@example.com');
            INSERT INTO chat_handle_join VALUES (1, 10), (1, 11), (2, 11), (2, 12), (3, 10);
            INSERT INTO message VALUES (100, 5), (101, 9), (102, 7);
            INSERT INTO chat_message_join VALUES (1, 100), (1, 101), (2, 102);
        """)
        conn.commit()
        conn.close()

    def test_groups_come_with_named_members_and_their_latest_message(self, tmp_path):
        db = tmp_path / "chat.db"
        self._chat_db(db)
        m = IMessageIntegration()
        m._contact_cache = {"+46701740605": "Alice", "+46701740606": "Bob"}
        with patch("integrations.imessage.IMessageIntegration.CHAT_DB_PATH", str(db)):
            groups = {g.guid: g for g in m._load_group_chats()}
        assert set(groups) == {"any;+;chat1", "any;+;chat2"}
        assert groups["any;+;chat1"] == GroupChat("any;+;chat1", "Family", ("Alice", "Bob"), 9)
        unnamed = groups["any;+;chat2"]
        assert unnamed.name is None and unnamed.members == ("Bob", "carol@example.com")
        assert unnamed.label == "Bob, carol@example.com"

    def test_an_unreadable_db_yields_no_groups(self, tmp_path):
        with patch("integrations.imessage.IMessageIntegration.CHAT_DB_PATH", str(tmp_path / "missing.db")):
            assert IMessageIntegration()._load_group_chats() == []
