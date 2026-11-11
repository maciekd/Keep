"""The iMessage chat's allowlist: who an entry names, who is an owner, and what is refused."""
from unittest.mock import patch

import pytest

from interfaces.imessage import Allowlist as allowlist
from interfaces.imessage.Allowlist import Allowlist, normalize, parse_entry

from .chat_fakes import CHARLOTTE, BOBS_KEEP, MARCUS, STRANGER, configured


@pytest.mark.unit
class TestAnEntry:
    @pytest.mark.parametrize("entry, expected", [
        ("+46701740605", ("", "+46701740605")),
        ("Marcus +46701740605", ("Marcus", "+46701740605")),
        ("Marcus  +46 70-174 06 05", ("Marcus", "+46701740605")),
        ("Ingeborg von Essen +46 (70) 174 06 07", ("Ingeborg von Essen", "+46701740607")),
        ("Charlotte: +46701740606", ("Charlotte", "+46701740606")),
        ("alice@example.com", ("", "alice@example.com")),
        ("Alice Alice@Example.com", ("Alice", "alice@example.com")),
        ("Alice <alice@example.com>", ("Alice", "alice@example.com")),
    ], ids=["a bare phone", "a name and a phone", "a phone written with spaces", "a name of several words",
            "a colon after the name", "a bare email", "an email in capitals", "an email in brackets"])
    def test_names_a_person_by_a_handle_with_the_name_in_front(self, entry, expected):
        assert parse_entry(entry) == expected

    @pytest.mark.parametrize("entry", ["0701740605", "Marcus 070-174 06 05", "Marcus", "", None, "+46", "marcus@",
                                       "46701740605"],
                             ids=["a national number", "a named national number", "a name alone", "empty", "nothing",
                                  "a country code alone", "half an email", "international without the plus"])
    def test_without_a_whole_handle_names_nobody(self, entry):
        """A number is matched as a whole: one that could be anybody's in another country is refused."""
        assert parse_entry(entry) == ("", "")

    def test_handles_are_compared_as_messages_stores_them(self):
        assert normalize(" +46 70-174 06 05 ") == "+46701740605" and normalize("Alice@Example.COM") == "alice@example.com"


@pytest.mark.unit
class TestTheLists:
    def test_an_owner_a_guest_and_another_assistant_are_told_apart(self):
        with configured():
            allowed = Allowlist()
            marcus, charlotte, keep = allowed.find(MARCUS), allowed.find(CHARLOTTE), allowed.find(BOBS_KEEP)
        assert (marcus.owner, marcus.guest, marcus.bot, marcus.name) == (True, False, False, "Marcus")
        assert (charlotte.owner, charlotte.guest, charlotte.bot) == (False, True, False)
        assert (keep.owner, keep.guest, keep.bot, keep.name) == (False, True, True, "Bob's Keep")

    def test_somebody_on_no_list_is_nobody(self):
        with configured():
            assert Allowlist().find(STRANGER) is None and Allowlist().find(None) is None

    def test_a_handle_is_found_however_the_database_spells_it(self):
        with configured():
            assert Allowlist().find("Alice@Example.com").name == "Alice"

    def test_the_last_digits_of_a_number_make_nobody_an_owner(self):
        """Marcus's number under another country code is another person's phone."""
        with configured():
            assert Allowlist().find("+1" + MARCUS[3:]) is None

    def test_a_guest_entry_never_takes_an_owners_rights_away_nor_a_bot_entry(self):
        with configured(imessage_chat__guests=[f"Marc {MARCUS}"], imessage_chat__bots=[MARCUS]):
            marcus = Allowlist().find(MARCUS)
        assert marcus.owner and not marcus.bot and marcus.name == "Marcus"

    def test_a_guest_marked_as_an_assistant_keeps_the_name_it_had(self):
        with configured(imessage_chat__bots=[CHARLOTTE]):
            charlotte = Allowlist().find(CHARLOTTE)
        assert charlotte.bot and charlotte.name == "Charlotte"

    def test_an_entry_that_names_nobody_is_reported_by_its_place_not_by_its_text(self):
        """What the user typed for an owner is theirs alone: the report says where, not what."""
        with configured(imessage_chat__owners=[f"Marcus {MARCUS}", "Frances 0701740608"]):
            allowed = Allowlist()
            assert len(allowed) == 4
        assert len(allowed.problems) == 1
        assert "imessage_chat.owners[1]" in allowed.problems[0] and "0701740608" not in allowed.problems[0]

    def test_a_single_value_written_by_hand_is_a_list_of_one(self):
        with configured(imessage_chat__owners=f"Marcus {MARCUS}"):
            assert Allowlist().find(MARCUS).owner

    def test_a_change_of_the_lists_is_in_force_at_the_next_look(self):
        with configured() as keys:
            allowed = Allowlist()
            assert allowed.find(STRANGER) is None and allowed.refresh() is False
            keys["imessage_chat.guests"] = [*keys["imessage_chat.guests"], f"Dave {STRANGER}"]
            assert allowed.find(STRANGER).name == "Dave"
            keys["imessage_chat.owners"] = []
            assert allowed.refresh() is True and allowed.find(MARCUS) is None

    def test_a_handle_on_no_list_is_said_once_until_the_lists_change(self):
        """The transport looks every member of a group up at each poll: a line per miss buries the log."""
        with configured() as keys, patch.object(allowlist, "trace") as said:
            allowed = Allowlist()
            misses = lambda: sum(STRANGER in call.args[0] for call in said.call_args_list)
            for _ in range(3):
                allowed.find(STRANGER)
            assert misses() == 1
            keys["imessage_chat.guests"] = []
            allowed.find(STRANGER)
            assert misses() == 2, "after an edit a handle that is still on no list says so again"
