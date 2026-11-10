"""Destinations: the registry post_message delivers through — lookup, rendering by kind,
the failure reasons the model corrects itself on, and the listeners the prompt follows."""
import gc

import pytest

from assistant import Destinations
from assistant.Destinations import Destination, KIND_CHAT, KIND_GROUP, KIND_VOICE


@pytest.fixture(autouse=True)
def clean_registry():
    Destinations.clear()
    yield
    Destinations.clear()


def recording(name, kind, aliases=(), accept=True):
    sent = []

    def send(text):
        sent.append(text)
        return accept
    Destinations.register(Destination(name, kind, send, tuple(aliases)))
    return sent


@pytest.mark.unit
class TestANameThatIsNotTheRegistrantsToChoose:
    """`claim` / `release`: a chat's title is set by the people in it, so a registrant that
    goes by titles gets a name only while it is free and gives back only its own registration."""

    def test_a_name_somebody_answers_to_is_not_claimed(self):
        theirs = recording("Family", KIND_GROUP, aliases=("-100123",))
        mine = []
        for name in ("family", "-100123"):
            assert Destinations.claim(Destination(name, KIND_GROUP, lambda text: mine.append(text) or True)) is None
        Destinations.send("Family", "hi")
        Destinations.send("-100123", "again")
        assert (theirs, mine) == (["hi", "again"], [])

    def test_a_claim_leaves_out_the_alias_somebody_else_answers_to(self):
        theirs = recording("Charlotte", KIND_CHAT)
        claimed = Destinations.claim(Destination("Charlotte (work)", KIND_CHAT, lambda text: True,
                                                 aliases=("charlotte", "c@example.com")))
        assert claimed.aliases == ("c@example.com",)
        Destinations.send("charlotte", "hi")
        assert theirs == ["hi"]

    def test_releasing_a_registration_never_drops_the_one_that_holds_its_name_now(self):
        """The registrant lost its name to a `register` of the same name and cleans up later."""
        mine = Destinations.claim(Destination("Family", KIND_GROUP, lambda text: True))
        theirs = recording("Family", KIND_GROUP)
        assert Destinations.holds(mine) is False
        assert Destinations.release(mine) is False
        assert Destinations.send("Family", "hi") == "" and theirs == ["hi"]

    def test_a_released_registration_is_gone_and_its_name_free_again(self):
        mine = Destinations.claim(Destination("Family", KIND_GROUP, lambda text: True))
        assert Destinations.holds(mine) and Destinations.release(mine) is True
        assert Destinations.names() == [] and Destinations.claim(Destination("Family", KIND_GROUP, lambda text: True))


@pytest.mark.unit
class TestLookup:
    def test_a_group_is_found_by_its_title_or_its_chat_id_whatever_the_case(self):
        sent = recording("Marcus & Alice - board", KIND_GROUP, aliases=("-4700174060",))
        assert Destinations.send("marcus & alice - BOARD", "hi") == ""
        assert Destinations.send("-4700174060", "again") == ""
        assert sent == ["hi", "again"]

    def test_the_sender_reaches_a_sender_aware_destination_only(self):
        seen = []
        Destinations.register(Destination("kitchen", KIND_VOICE, lambda text, sender: seen.append((text, sender)) or True,
                                          sender_aware=True))
        plain = recording("owner", KIND_CHAT)
        turn = object()
        assert Destinations.send("kitchen", "hi", sender=turn) == ""
        assert Destinations.send("owner", "hi", sender=turn) == ""
        assert seen == [("hi", turn)] and plain == ["hi"]

    def test_registering_the_same_name_again_replaces_the_old_sender(self):
        old = recording("owner", KIND_CHAT)
        new = recording("Owner", KIND_CHAT)
        Destinations.send("owner", "hi")
        assert old == [] and new == ["hi"]
        assert Destinations.names() == ["Owner"]

    def test_an_unregistered_destination_is_gone_and_dropping_it_twice_is_harmless(self):
        recording("kitchen", KIND_VOICE)
        Destinations.unregister("Kitchen")
        Destinations.unregister("kitchen")
        assert Destinations.names() == []

    @pytest.mark.parametrize("spelling", ["Family  Chat", "Family\u00a0Chat", "family chat"],
                             ids=["two spaces", "a no-break space", "another case"])
    def test_a_name_registered_in_another_spelling_is_the_same_name(self, spelling):
        """`find` reads the two spellings as one name, so `register` must too: a claimed
        registration beside it would keep answering for both while its owner thinks it holds
        a name of its own."""
        claimed = Destinations.claim(Destination("Family Chat", KIND_GROUP, lambda text: "the claimed one"))
        theirs = recording(spelling, KIND_GROUP)
        assert Destinations.holds(claimed) is False, "the claimant cannot tell that it lost the name"
        assert Destinations.send("Family Chat", "hi") == "" and theirs == ["hi"]
        assert len(Destinations.names()) == 1
        Destinations.unregister("FAMILY   chat")
        assert Destinations.names() == []


@pytest.mark.unit
class TestADestinationThatSaysWhyNot:
    def test_a_reason_instead_of_a_yes_is_a_failure_with_that_reason(self):
        """A send that answers with a text did not deliver: a truthy value must not read as taken."""
        Destinations.register(Destination("Charlotte", KIND_CHAT,
                                          lambda text: "the daily limit of 200 messages to Charlotte is reached"))
        reason = Destinations.send("Charlotte", "hi")
        assert reason.startswith("post_message failed") and "daily limit of 200" in reason

    def test_a_label_of_its_own_is_what_the_list_calls_it(self):
        recording("Family", KIND_GROUP)
        Destinations.register(Destination("Chess club", KIND_GROUP, lambda text: True, label="iMessage group"))
        described = Destinations.describe()
        assert '"Chess club" (iMessage group)' in described and '"Family" (Telegram group)' in described


@pytest.mark.unit
class TestSend:
    def test_an_unknown_destination_fails_naming_the_ones_that_exist(self):
        recording("owner", KIND_CHAT)
        recording("Marcus & Alice - board", KIND_GROUP)
        reason = Destinations.send("garage", "hi")
        assert reason.startswith("post_message failed")
        assert "garage" in reason and "owner" in reason and '"Marcus & Alice - board"' in reason

    def test_a_voice_destination_gets_the_words_without_the_chat_markup(self):
        sent = recording("kitchen", KIND_VOICE)
        assert Destinations.send("kitchen", "<b>The pasta</b> is done &amp; drained.") == ""
        assert sent == ["The pasta is done & drained."]

    def test_a_chat_destination_gets_the_markup_as_written(self):
        sent = recording("owner", KIND_CHAT)
        Destinations.send("owner", "<b>Car</b> at 80%")
        assert sent == ["<b>Car</b> at 80%"]

    def test_a_text_that_is_only_markup_is_refused_before_it_is_sent(self):
        sent = recording("kitchen", KIND_VOICE)
        assert "empty" in Destinations.send("kitchen", "<i></i>")
        assert sent == []

    def test_a_destination_that_does_not_take_the_message_is_a_failure_the_model_sees(self):
        recording("owner", KIND_CHAT, accept=False)
        assert "did not take" in Destinations.send("owner", "hi")

    def test_a_sender_that_raises_is_a_failure_not_an_exception(self):
        def boom(text):
            raise RuntimeError("socket closed")
        Destinations.register(Destination("owner", KIND_CHAT, boom))
        reason = Destinations.send("owner", "hi")
        assert "RuntimeError" in reason


@pytest.mark.unit
class TestDescribe:
    def test_nothing_registered_says_so_instead_of_an_empty_line(self):
        assert Destinations.describe() == Destinations.NONE_YET

    @pytest.mark.parametrize("name", ['Team "A" (Telegram group), owner (your chats), "x', "Team\nowner (your chats)",
                                      "Team “A”\u2028owner"],
                             ids=["quotes", "a line break", "typographic quotes and a line separator"])
    def test_a_name_cannot_end_its_own_quotes_or_its_line_whoever_registered_it(self, name):
        """A group's title is its members' to set, and the list stands in every prompt."""
        sent = recording(name, KIND_GROUP)
        described = Destinations.describe()
        assert described.count('"') == 2 and described.startswith('"') and described.endswith('" (Telegram group)')
        assert len(described.splitlines()) == 1
        assert Destinations.send(described[1:described.rindex('"')], "hi") == "" and sent == ["hi"], \
            "the name as the list shows it does not find the destination"

    def test_a_groups_name_is_always_between_quotes_and_a_plain_name_of_the_registrants_own_is_not(self):
        """A group is named by its members: "Ingeborg" as a group must not read like the
        person, so it is shown as a title whatever it is made of."""
        recording("Ingeborg", KIND_GROUP)
        recording("Charlotte", KIND_CHAT)
        described = Destinations.describe()
        assert '"Ingeborg" (' in described and "Charlotte (" in described and '"Charlotte"' not in described

    @pytest.mark.parametrize("name", ["Zoë", "ownerǀyour", "ᐸsystemᐳ", "a.b"],
                             ids=["an accent", "a letter shaped like a bar", "letters shaped like brackets", "a mark"])
    def test_a_name_of_more_than_plain_letters_is_between_quotes(self, name):
        recording(name, KIND_CHAT)
        assert Destinations.describe().startswith(f'"{name}" (')

    def test_an_unlisted_destination_is_found_and_not_shown(self):
        sent = []
        Destinations.register(Destination("owner", KIND_CHAT, lambda text: True))
        Destinations.register(Destination("c@example.com", KIND_CHAT, lambda text: sent.append(text) or True, listed=False))
        assert "example.com" not in Destinations.describe()
        assert Destinations.send("c@example.com", "hi") == "" and sent == ["hi"]

    def test_names_with_spaces_are_quoted_and_every_kind_is_labelled(self):
        recording("owner", KIND_CHAT)
        recording("Marcus & Alice - board", KIND_GROUP)
        recording("kitchen", KIND_VOICE)
        line = Destinations.describe()
        assert line.index("owner (") < line.index('"Marcus & Alice - board" (') < line.index("kitchen (")
        assert line.count("(") == 3


@pytest.mark.unit
class TestListeners:
    def test_a_listener_is_told_of_every_registration_and_removal(self):
        class Follower:
            def __init__(self):
                self.calls = 0

            def changed(self):
                self.calls += 1
        follower = Follower()
        Destinations.add_listener(follower.changed)
        recording("owner", KIND_CHAT)
        Destinations.unregister("owner")
        Destinations.unregister("owner")   # nothing changed: no call
        assert follower.calls == 2

    def test_a_listener_whose_owner_is_gone_is_dropped_not_called(self):
        class Follower:
            def changed(self):
                raise AssertionError("a dead manager was called back")
        follower = Follower()
        Destinations.add_listener(follower.changed)
        del follower
        gc.collect()
        recording("owner", KIND_CHAT)   # must not raise

    def test_a_listener_that_raises_does_not_stop_the_registration(self):
        def broken():
            raise RuntimeError("prompt lock")
        Destinations.add_listener(broken)
        recording("owner", KIND_CHAT)
        assert Destinations.names() == ["owner"]
