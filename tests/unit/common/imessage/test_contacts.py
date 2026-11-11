"""The AddressBook cache: who a handle is, which of a contact's handles a name means, and the
group chat a name or a member list means."""
from unittest.mock import Mock

import pytest

from common.imessage.Contacts import Contacts
from common.imessage.MessagesDatabase import MessagesDatabase
from common.imessage.MessagesHost import GroupChat

from .messages_fakes import address_book


def contacts(cache=None, names=None, activity=None, groups=()):
    """Contacts with a cache that is fresh (never read from disk), over a database that knows
    `activity` (handle → (imessage, last_date, count)) and `groups`."""
    database = Mock()
    database.chat_activity.return_value = activity or {}
    database.group_chats.return_value = list(groups)
    c = Contacts(database)
    c._contact_cache = dict(cache or {})
    c._name_to_handle = dict(names or {})
    c._contact_cache_time = 1e18
    return c


@pytest.mark.unit
class TestNameFor:
    def test_known_contact_resolved(self):
        assert contacts({"+1234567890": "Alice"}).name_for("+1234567890") == "Alice"

    def test_unknown_contact_returns_handle(self):
        assert contacts({"+1234567890": "Alice"}).name_for("+9999999999") == "+9999999999"

    def test_local_number_matches_international_via_suffix(self):
        c = contacts({"0701740620": "Bob", "701740620": "Bob"})   # the suffix key is added by the cache builder
        assert c.name_for("+46701740620") == "Bob"

    def test_a_row_without_a_handle_has_an_unknown_sender(self):
        assert contacts().name_for(None) == "Unknown"


@pytest.mark.unit
class TestNameKeys:
    def test_first_only(self):
        assert Contacts._name_keys("Alice", None) == ["Alice"]

    def test_first_and_last(self):
        assert Contacts._name_keys("Alice", "Smith") == ["Alice", "Alice Smith"]

    def test_blank_last_treated_as_missing(self):
        assert Contacts._name_keys("Alice", "   ") == ["Alice"]

    def test_no_first_yields_nothing(self):
        assert Contacts._name_keys(None, "Smith") == []
        assert Contacts._name_keys("", "Smith") == []

    def test_strips_surrounding_whitespace(self):
        assert Contacts._name_keys("  Alice  ", "  Smith  ") == ["Alice", "Alice Smith"]

    def test_middle_name_answers_to_either_surname_half_and_the_whole(self):
        # a card filed as first, middle and last was unknown by first + middle and by all three alike
        assert Contacts._name_keys("Dave", "Holm", "Berg") == \
            ["Dave", "Dave Berg", "Dave Holm", "Dave Berg Holm"]

    def test_middle_name_without_last(self):
        assert Contacts._name_keys("Dave", None, "Berg") == ["Dave", "Dave Berg"]


@pytest.mark.unit
class TestRecordFullName:
    def test_first_and_last_recorded(self):
        m = {}
        Contacts._record_full_name(m, "Alice", "Smith", "+1 (234) 567-8901")
        assert m == {"+12345678901": "Alice Smith"}

    def test_first_only_recorded(self):
        m = {}
        Contacts._record_full_name(m, "Alice", None, "+12345678901")
        assert m == {"+12345678901": "Alice"}

    def test_middle_name_is_part_of_the_full_name(self):
        m = {}
        Contacts._record_full_name(m, "Dave", "Holm", "+12345678901", "Berg")
        assert m == {"+12345678901": "Dave Berg Holm"}

    def test_first_writer_wins(self):
        """If two contacts share a handle, the first one written sticks (last writer would be ambiguous anyway)."""
        m = {}
        Contacts._record_full_name(m, "Alice", "Smith", "+12345678901")
        Contacts._record_full_name(m, "Bob", "Jones", "+12345678901")
        assert m["+12345678901"] == "Alice Smith"


@pytest.mark.unit
class TestAddCandidate:
    def test_handles_normalized_and_keyed_by_collapsed_name(self):
        candidates = {}
        Contacts._add_candidate(candidates, "  Alice   Smith ", "+1 (234) 567-8901", "_$!<Mobile>!$_")
        assert candidates == {"alice smith": [("+12345678901", "_$!<Mobile>!$_")]}

    def test_same_handle_listed_once(self):
        candidates = {}
        Contacts._add_candidate(candidates, "Alice", "+46 70-174 06 05", "_$!<Mobile>!$_")
        Contacts._add_candidate(candidates, "Alice", "+46701740605", "_$!<iPhone>!$_")
        assert [h for h, _ in candidates["alice"]] == ["+46701740605"]

    def test_blank_name_or_handle_ignored(self):
        candidates = {}
        Contacts._add_candidate(candidates, "   ", "+46701740605")
        Contacts._add_candidate(candidates, "Alice", "")
        assert candidates == {}


def _rank(candidates, activity):
    """Rank with chat.db replaced by `activity` (handle → (imessage, last_date, count))."""
    reverse, reasons = contacts(activity=activity)._rank_candidates(candidates)
    return reverse


@pytest.mark.unit
class TestRankCandidates:
    def test_mobile_beats_landline_listed_first_on_card(self):
        """The bug: the card lists a 031 landline before the mobile and Messages knows neither."""
        candidates = {"bob": [("+4631123456", "_$!<Home>!$_"), ("+46701740605", "_$!<Mobile>!$_")]}
        assert _rank(candidates, {})["bob"] == "+46701740605"

    def test_swedish_07x_number_counts_as_mobile_without_label(self):
        candidates = {"bob": [("+4631123456", None), ("0701740606", None)]}
        assert _rank(candidates, {})["bob"] == "0701740606"

    def test_imessage_registered_email_beats_landline_and_mobile(self):
        """Only the email is an iMessage handle: it wins over the mobile and the landline."""
        candidates = {"carol": [("+4631123456", "_$!<Home>!$_"), ("+46701740607", "_$!<Mobile>!$_"),
                                ("carol@example.com", None)]}
        activity = {"carol@example.com": (True, 600_000_000_000_000_000, 12)}
        assert _rank(candidates, activity)["carol"] == "carol@example.com"

    def test_email_unknown_to_messages_ranks_below_phones(self):
        candidates = {"dave": [("dave@example.com", None), ("+4631123456", "_$!<Work>!$_")]}
        assert _rank(candidates, {})["dave"] == "+4631123456"

    def test_landline_known_to_imessage_beats_unknown_mobile(self):
        candidates = {"alice": [("+4631123456", "_$!<Home>!$_"), ("+46701740608", "_$!<Mobile>!$_")]}
        activity = {"+4631123456": (True, 500_000_000_000_000_000, 3)}
        assert _rank(candidates, activity)["alice"] == "+4631123456"

    def test_sms_only_history_does_not_make_a_handle_imessage(self):
        candidates = {"alice": [("+4631123456", "_$!<Home>!$_"), ("+46701740608", "_$!<Mobile>!$_")]}
        activity = {"+4631123456": (False, 500_000_000_000_000_000, 3)}
        assert _rank(candidates, activity)["alice"] == "+46701740608"

    def test_several_handles_with_history_most_recent_wins(self):
        candidates = {"charlotte": [("+12025550176", "_$!<Mobile>!$_"), ("+46701740605", "_$!<iPhone>!$_"),
                                    ("charlotte@example.com", None)]}
        activity = {
            "+12025550176": (True, 500_000_000_000_000_000, 900),
            "+46701740605": (True, 700_000_000_000_000_000, 40),
            "charlotte@example.com": (True, 600_000_000_000_000_000, 5),
        }
        assert _rank(candidates, activity)["charlotte"] == "+46701740605"

    def test_equal_recency_breaks_on_message_count(self):
        candidates = {"carol": [("+11111111111", None), ("+22222222222", None)]}
        activity = {"+11111111111": (True, 100_000_000, 10), "+22222222222": (True, 100_000_000, 500)}
        assert _rank(candidates, activity)["carol"] == "+22222222222"

    def test_local_format_card_number_matches_e164_chat_handle(self):
        candidates = {"frances": [("+4631123456", "_$!<Home>!$_"), ("0701740609", "_$!<Home>!$_")]}
        activity = {"+46701740609": (True, 1, 1), "701740609": (True, 1, 1)}
        assert _rank(candidates, activity)["frances"] == "0701740609"

    def test_verbatim_local_sms_row_does_not_hide_the_e164_imessage_history(self):
        """chat.db keeps an old SMS row under the local-format id and the iMessage history under
        the E.164 id; the card lists the local format. The phone is still the iMessage handle."""
        candidates = {"carol": [("0701740605", "_$!<Mobile>!$_"), ("carol@example.com", None)]}
        activity = {"0701740605": (False, 100, 1), "+46701740605": (True, 900, 300),
                    "701740605": (True, 900, 301), "carol@example.com": (True, 500, 1)}
        assert _rank(candidates, activity)["carol"] == "0701740605"

    def test_chat_db_asked_about_the_contested_handles_only(self):
        candidates = {"alice": [("+12345678901", None)],
                      "bob": [("+4631123456", "_$!<Home>!$_"), ("bob@example.com", None)]}
        c = contacts()
        c._rank_candidates(candidates)
        assert set(c._database.chat_activity.call_args.args[0]) == {"+4631123456", "bob@example.com"}

    def test_fax_ranks_last(self):
        candidates = {"bob": [("+4631123457", "_$!<HomeFAX>!$_"), ("+4631123456", "_$!<Home>!$_")]}
        assert _rank(candidates, {})["bob"] == "+4631123456"

    def test_same_tier_without_history_keeps_card_order(self):
        candidates = {"bob": [("+11111111111", None), ("+22222222222", None)]}
        assert _rank(candidates, {})["bob"] == "+11111111111"

    def test_single_handle_contacts_do_not_open_chat_db(self):
        candidates = {"alice": [("+12345678901", None)], "bob": [("bob@example.com", None)]}
        c = contacts()
        reverse, _ = c._rank_candidates(candidates)
        c._database.chat_activity.assert_not_called()
        assert reverse == {"alice": "+12345678901", "bob": "bob@example.com"}

    def test_an_unreadable_chat_db_ranks_by_card_alone(self, tmp_path):
        candidates = {"bob": [("+4631123456", "_$!<Home>!$_"), ("+46701740605", "_$!<Mobile>!$_")]}
        reverse, _ = Contacts(MessagesDatabase(tmp_path / "missing.db"))._rank_candidates(candidates)
        assert reverse["bob"] == "+46701740605"


@pytest.mark.unit
class TestRefreshRanking:
    """The whole path: an AddressBook database on disk → resolve."""

    def _resolve(self, tmp_path, people, activity, name):
        database = Mock()
        database.chat_activity.return_value = activity
        database.group_chats.return_value = []
        c = Contacts(database, address_book(tmp_path, people))
        c.refresh()
        resolved = c.resolve(name)
        assert not resolved.error
        return resolved, c

    def test_card_with_a_middle_name_resolves_by_first_and_middle(self, tmp_path):
        people = [("Dave", "Berg", "Holm", [("070-174 06 08", "_$!<Mobile>!$_")], [])]
        resolved, c = self._resolve(tmp_path, people, {}, "Dave Berg")
        assert resolved.handle == "0701740608"
        assert c.resolve("Dave Berg Holm").handle == "0701740608"
        assert resolved.full_name == "Dave Berg Holm"
        assert resolved.name == "Dave", "the feed shows the first name for this handle"

    @pytest.mark.parametrize("child_source", ["AAA", "ZZZ"])
    def test_owners_address_book_names_a_handle_a_childs_list_also_holds(self, tmp_path, child_source):
        """The child's source sorts before the owner's in one case and after it in the other."""
        mobile = [("070-174 06 05", "_$!<Mobile>!$_")]
        own = [("Marcus", "Berg", mobile, ["m@mlab.test"]), ("Alice", "Ek", [], ["alice@example.com"]),
               ("Bob", "Jones", [("070-174 06 06", "_$!<Mobile>!$_")], [])]
        child = [("Pappa", None, mobile, ["m@mlab.test"]), ("Farfar", None, [("070-174 06 09", "_$!<Mobile>!$_")], [])]
        address_book(tmp_path, own, source="MMM")
        folder = address_book(tmp_path, child, source=child_source)
        database = Mock()
        database.chat_activity.return_value = {}
        c = Contacts(database, folder)
        c.refresh()
        assert c.name_for("+46701740605") == "Marcus"
        assert c.name_for("m@mlab.test") == "Marcus"
        assert c.full_name("0701740605") == "Marcus Berg"
        assert c.name_for("0701740609") == "Farfar"

    @pytest.mark.parametrize("child_source", ["AAA", "ZZZ"])
    def test_a_name_on_cards_of_both_lists_is_sent_to_the_owners_contact(self, tmp_path, child_source):
        """The child has an Alice of their own: a message "to Alice" must not go to her."""
        own = [("Alice", "Ek", [("070-174 06 07", "_$!<Mobile>!$_")], []), ("Bob", "Jones", [], ["bob@example.com"]),
               ("Carol", "King", [], ["carol@example.com"])]
        child = [("Alice", None, [("070-174 06 10", "_$!<Mobile>!$_")], [])]
        address_book(tmp_path, own, source="MMM")
        folder = address_book(tmp_path, child, source=child_source)
        database = Mock()
        database.chat_activity.return_value = {}
        database.group_chats.return_value = []
        c = Contacts(database, folder)
        c.refresh()
        resolved = c.resolve("Alice")
        assert (resolved.handle, resolved.full_name) == ("0701740607", "Alice Ek")

    def test_a_source_that_cannot_be_read_costs_only_its_own_cards(self, tmp_path):
        folder = address_book(tmp_path, [("Bob", "Jones", [("070-174 06 06", "_$!<Mobile>!$_")], [])])
        broken = folder / "Sources" / "AAA" / "AddressBook-v22.abcddb"
        broken.parent.mkdir()
        broken.write_bytes(b"not a database at all")
        c = Contacts(Mock(chat_activity=Mock(return_value={})), folder)
        c.refresh()
        assert c.name_for("+46701740606") == "Bob" and c.resolve("Bob Jones").handle == "0701740606"

    def test_landline_first_on_card_still_resolves_to_mobile(self, tmp_path):
        people = [("Bob", "Jones", [("031-12 34 56", "_$!<Home>!$_"), ("070-174 06 05", "_$!<Mobile>!$_")], [])]
        resolved, c = self._resolve(tmp_path, people, {}, "Bob")
        assert resolved.handle == "0701740605"
        assert c.resolve("Bob Jones").handle == "0701740605"

    def test_only_email_registered_with_imessage_wins(self, tmp_path):
        people = [("Carol", None, [("031-12 34 56", "_$!<Home>!$_"), ("+46 70 174 06 07", "_$!<Mobile>!$_")],
                   ["Carol@Example.com"])]
        activity = {"carol@example.com": (True, 1_000, 2)}
        resolved, _ = self._resolve(tmp_path, people, activity, "carol")
        assert resolved.handle == "carol@example.com"

    def test_reason_for_the_choice_is_kept(self, tmp_path):
        people = [("Bob", None, [("031-12 34 56", "_$!<Home>!$_"), ("070-174 06 05", "_$!<Mobile>!$_")], [])]
        _, c = self._resolve(tmp_path, people, {}, "Bob")
        assert "mobile" in c._handle_reasons["bob"]

    def test_the_cache_counts_its_handles_and_a_stale_one_is_read_again(self, tmp_path):
        people = [("Bob", None, [("070-174 06 05", "_$!<Mobile>!$_")], ["bob@example.com"])]
        c = Contacts(Mock(chat_activity=Mock(return_value={})), address_book(tmp_path, people))
        assert len(c) == 0
        c.refresh_if_stale()
        assert len(c) == 3, "the number, its 9-digit suffix and the email"
        c._contact_cache = {}
        c.refresh_if_stale()
        assert len(c) == 0, "read a moment ago: not read again"

    def test_an_address_book_without_sources_leaves_the_cache_empty(self, tmp_path):
        c = Contacts(Mock(), tmp_path / "AddressBook")
        c.refresh()
        assert len(c) == 0


@pytest.mark.unit
class TestResolve:
    def test_resolves_contact_name_case_insensitive(self):
        resolved = contacts(names={"alice": "+12345678901"}).resolve("ALICE")
        assert (resolved.handle, resolved.error) == ("+12345678901", "")

    def test_unknown_name_returns_error(self):
        resolved = contacts(names={"alice": "+1"}).resolve("Zelda")
        assert resolved.handle is None
        assert "Zelda" in resolved.error and "unknown contact" in resolved.error
        assert not resolved.no_contacts

    def test_an_empty_cache_is_told_apart_from_an_unknown_name(self):
        resolved = contacts().resolve("Ingeborg")
        assert resolved.handle is None
        assert resolved.no_contacts and "Ingeborg" in resolved.error

    def test_empty_recipient_is_refused(self):
        assert contacts().resolve("   ").error

    def test_fallback_via_forward_cache(self):
        """If reverse cache misses but forward cache contains the name, recover."""
        c = contacts({"+46701234567": "Ingeborg",
                      "0701234567": "Ingeborg",     # local format
                      "701234567": "Ingeborg"})     # suffix
        resolved = c.resolve("Ingeborg")
        assert resolved.error == ""
        assert resolved.handle == "+46701234567"  # all mobiles, no history: E.164 wins by priority

    def test_fallback_picks_email_when_only_email_available(self):
        resolved = contacts({"ingeborg@example.com": "Ingeborg"}).resolve("Ingeborg")
        assert (resolved.handle, resolved.error) == ("ingeborg@example.com", "")

    def test_raw_phone_passes_through(self):
        resolved = contacts().resolve("+1 (234) 567-8901")
        assert (resolved.handle, resolved.error) == ("+12345678901", "")

    def test_raw_email_passes_through(self):
        resolved = contacts().resolve("Bob@Example.com")
        assert (resolved.handle, resolved.error) == ("bob@example.com", "")

    def test_a_raw_handle_comes_with_the_names_the_card_has(self):
        c = contacts({"+12345678901": "Alice"})
        c._handle_to_full_name = {"+12345678901": "Alice Smith"}
        resolved = c.resolve("+1 (234) 567-8901")
        assert (resolved.name, resolved.full_name) == ("Alice", "Alice Smith")

    def test_a_handle_no_contact_has_comes_without_a_name(self):
        resolved = contacts({"+12345678901": "Alice"}).resolve("+46701740605")
        assert resolved.name is None and resolved.full_name is None

    def test_full_name_disambiguates_first_name_collision(self):
        """Two contacts with the same first name resolve uniquely via full name."""
        c = contacts(names={"alex": "+1",  # last writer wins on bare first name; either is fine
                            "alex smith": "+1", "alex jones": "+2"})
        assert c.resolve("Alex Smith").handle == "+1"
        assert c.resolve("Alex Jones").handle == "+2"

    def test_full_name_extra_whitespace_normalized(self):
        assert contacts(names={"alice smith": "+1"}).resolve("  Alice   Smith  ").handle == "+1"


FAMILY = GroupChat("any;+;chat100", "Family", ["Charlotte Smith", "Frances Smith", "Ingeborg Smith"], 900)
PAIR = GroupChat("any;+;chat200", None, ["Alice Smith", "Bob Jones"], 800)
PAIR_OLD = GroupChat("any;+;chat201", None, ["Alice Smith", "Bob Jones"], 100)   # a recreated group
TRIO = GroupChat("any;+;chat300", None, ["Alice Smith", "Bob Jones", "Carol King"], 700)
GROUPS = [FAMILY, PAIR, PAIR_OLD, TRIO]


@pytest.fixture
def with_groups():
    return contacts(names={"alice": "+12345678901"}, groups=GROUPS)


@pytest.mark.unit
class TestGroupResolution:
    def test_a_named_group_resolves_to_its_guid(self, with_groups):
        resolved = with_groups.resolve("family")
        assert (resolved.handle, resolved.error) == (FAMILY.guid, "")

    def test_a_contact_name_still_wins_over_the_group_lookup(self, with_groups):
        resolved = with_groups.resolve("Alice")
        assert resolved.handle == "+12345678901" and resolved.group is None

    @pytest.mark.parametrize("recipient", ["Alice, Bob", "Bob & Alice", "alice and bob", "Alice Smith, Bob Jones"])
    def test_an_unnamed_group_resolves_by_its_members_in_any_order_and_spelling(self, with_groups, recipient):
        resolved = with_groups.resolve(recipient)
        assert resolved.error == ""
        assert resolved.handle == PAIR.guid, "the most recently active of the two look-alike groups"

    def test_the_callers_own_name_in_the_member_list_is_ignored(self, with_groups):
        assert with_groups.resolve("Marcus, Alice & Bob", owner="Marcus").handle == PAIR.guid

    def test_a_name_the_caller_did_not_claim_counts_as_a_member(self, with_groups):
        """The host knows no owner of its own: who is left out of a member list is the caller's to say."""
        resolved = with_groups.resolve("Marcus, Alice & Bob")
        assert resolved.handle is None and "marcus" in resolved.error

    def test_a_member_list_no_group_has_exactly_names_the_groups_that_contain_them(self, with_groups):
        resolved = with_groups.resolve("Alice, Carol")
        assert resolved.handle is None
        assert "Alice Smith, Bob Jones, Carol King" in resolved.error
        assert "Charlotte" not in resolved.error

    def test_a_member_list_no_group_contains_at_all_is_refused_without_a_hint(self, with_groups):
        resolved = with_groups.resolve("Dave, Carol")
        assert resolved.handle is None
        assert "dave" in resolved.error.lower() and "carol" in resolved.error.lower()
        assert "groups with them" not in resolved.error and "Carol King" not in resolved.error

    def test_a_single_unknown_name_is_still_an_unknown_contact(self, with_groups):
        resolved = with_groups.resolve("Zelda")
        assert resolved.handle is None and "unknown contact" in resolved.error

    def test_the_answer_carries_the_group_it_resolved_to(self, with_groups):
        assert with_groups.resolve("Alice & Bob").group == PAIR

    def test_the_groups_are_read_with_their_members_named_from_the_cache(self, with_groups):
        with_groups._contact_cache = {"+46701740605": "Charlotte"}
        with_groups.group_chats()
        name = with_groups._database.group_chats.call_args.args[0]
        assert name("+46701740605") == "Charlotte"


def sidebar(tmp_path, people, groups=()):
    """Contacts read from an AddressBook on disk, over a chat.db that has these groups."""
    database = Mock()
    database.chat_activity.return_value = {}
    database.group_chats.return_value = list(groups)
    return Contacts(database, address_book(tmp_path, people))


THE_FAMILY = GroupChat("iMessage;+;chat-family", "Family", ["Charlotte", "Marcus"])
CHARLOTTE = ("Charlotte", "Lindqvist", [("070-174 06 06", "_$!<Mobile>!$_")], ["charlotte@example.com"])


@pytest.mark.unit
class TestWhatTheSidebarCallsAChat:
    """The typing dots find a chat's window by this name: a name two chats could answer to is no name."""

    def test_a_person_goes_by_the_full_name_on_their_card_whatever_format_the_number_is_in(self, tmp_path):
        c = sidebar(tmp_path, [CHARLOTTE, ("Ingeborg", "Maria", "Lindqvist", [("+46 70 174 06 07", None)], [])])
        assert c.sidebar_name("iMessage;-;+46701740606") == ("Charlotte Lindqvist", "")
        assert c.sidebar_name("iMessage;-;Charlotte@Example.com") == ("Charlotte Lindqvist", "")
        assert c.sidebar_name("SMS;-;+46701740607") == ("Ingeborg Maria Lindqvist", "")

    def test_a_handle_no_card_has_goes_by_the_handle(self, tmp_path):
        c = sidebar(tmp_path, [CHARLOTTE])
        assert c.sidebar_name("iMessage;-;+46701740699") == ("+46701740699", "")
        assert c.sidebar_name("iMessage;-;Dave@Example.com") == ("dave@example.com", "")

    def test_a_named_group_goes_by_its_name(self, tmp_path):
        assert sidebar(tmp_path, [CHARLOTTE], [THE_FAMILY]).sidebar_name(THE_FAMILY.guid) == ("Family", "")

    @pytest.mark.parametrize("groups, why", [
        ([GroupChat("iMessage;+;chat-family", None, ["Charlotte", "Marcus"])], "an unnamed group"),
        ([THE_FAMILY, GroupChat("SMS;+;chat-family-2", "Family", ["Charlotte"])], "two groups with one name"),
        ([GroupChat("iMessage;+;chat-family", "Charlotte Lindqvist", ["Marcus"])], "a group named like a contact"),
        ([], "a group chat.db does not have"),
    ], ids=lambda value: value if isinstance(value, str) else "")
    def test_a_group_whose_name_is_not_its_alone_has_none(self, tmp_path, groups, why):
        name, reason = sidebar(tmp_path, [CHARLOTTE], groups).sidebar_name("iMessage;+;chat-family")
        assert name == "" and reason, why

    def test_two_different_people_with_one_name_are_not_told_apart_by_it(self, tmp_path):
        twin = ("Charlotte", "Lindqvist", [("070-174 06 08", None)], [])
        name, reason = sidebar(tmp_path, [CHARLOTTE, twin]).sidebar_name("iMessage;-;+46701740606")
        assert name == "" and reason

    def test_the_same_person_on_two_cards_is_one_person(self, tmp_path):
        """A contact kept in two accounts: the cards share a handle."""
        again = ("Charlotte", "Lindqvist", [("+46 70 174 06 06", None)], [])
        assert sidebar(tmp_path, [CHARLOTTE, again]).sidebar_name("iMessage;-;+46701740606")[0] == "Charlotte Lindqvist"

    def test_a_number_two_people_share_names_neither(self, tmp_path):
        frances = ("Frances", "Lindqvist", [("070-174 06 06", "_$!<Home>!$_")], [])
        name, reason = sidebar(tmp_path, [CHARLOTTE, frances]).sidebar_name("iMessage;-;+46701740606")
        assert name == "" and reason

    @pytest.mark.parametrize("child_source", ["AAA", "ZZZ"])
    def test_a_childs_name_for_the_owners_contact_is_not_a_second_name(self, tmp_path, child_source):
        """The child's list is the smaller source, sorting before the owner's or after it."""
        own = [CHARLOTTE, ("Alice", "Ek", [], ["alice@example.com"]), ("Bob", "Jones", [], ["bob@example.com"])]
        child = [("Mamma", None, [("070-174 06 06", None)], []), ("Farfar", None, [("070-174 06 09", None)], [])]
        address_book(tmp_path, child, source=child_source)
        c = sidebar(tmp_path, own)
        assert c.sidebar_name("iMessage;-;+46701740606") == ("Charlotte Lindqvist", "")
        assert c.sidebar_name("iMessage;-;+46701740609") == ("Farfar", "")

    def test_a_person_named_like_a_group_is_not_found_by_that_name(self, tmp_path):
        group = GroupChat("iMessage;+;chat-x", "Charlotte Lindqvist", ["Marcus"])
        name, reason = sidebar(tmp_path, [CHARLOTTE], [group]).sidebar_name("iMessage;-;+46701740606")
        assert name == "" and reason
