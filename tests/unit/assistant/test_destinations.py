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
class TestLookup:
    def test_a_group_is_found_by_its_title_or_its_chat_id_whatever_the_case(self):
        sent = recording("Marcus & Alice - board", KIND_GROUP, aliases=("-5342761890",))
        assert Destinations.send("marcus & alice - BOARD", "hi") == ""
        assert Destinations.send("-5342761890", "again") == ""
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
