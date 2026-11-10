"""assistant/GroupWake: the rule that decides which message of a group chat is said to the
assistant — the one piece the Telegram feed and the iMessage chat transport both decide by."""
from unittest.mock import Mock

import pytest

from assistant import GroupWake
from assistant.GroupWake import FOLLOWS, NAMED, REFUSED, REPLY, ROW, TURN

NAMES = ["Michelle", "michel"]
OWN, ALICE, BOB = "own", "alice", "bob"
GROUP, OTHER = "family", "chess"


@pytest.fixture
def wake():
    return GroupWake.GroupWake()


@pytest.mark.unit
class TestTheName:
    def test_a_name_is_a_whole_word_in_any_case(self):
        assert GroupWake.names_assistant("MICHELLE come here", NAMES)
        assert GroupWake.names_assistant("ask michel", NAMES) and GroupWake.names_assistant("Michelle's turn", NAMES)
        assert not GroupWake.names_assistant("Michelles", NAMES) and not GroupWake.names_assistant("hello", NAMES)

    def test_a_name_with_a_regex_character_is_read_as_written(self):
        assert GroupWake.names_assistant("hey k.e.e.p, lights", ["k.e.e.p"])
        assert not GroupWake.names_assistant("hey kaeaeap, lights", ["k.e.e.p"])

    def test_without_a_name_nothing_names_the_assistant(self):
        assert not GroupWake.names_assistant("Michelle?", ["", None])

    def test_the_names_are_the_wake_word_and_its_aliases_of_the_config_asked(self):
        source = Mock()
        source.get.side_effect = lambda key, default=None: {"wake_word": "Hal", "wake_word_aliases": ["hall", ""]}.get(key, default)
        assert GroupWake.assistant_names(source) == ["Hal", "hall"]


@pytest.mark.unit
class TestWhyAMessageWakes:
    def test_a_message_like_any_other_does_not(self, wake):
        assert wake.reason(GROUP, "what's for dinner?", OWN, NAMES) == ""

    def test_the_name_does(self, wake):
        assert wake.reason(GROUP, "Michelle, what's for dinner?", OWN, NAMES) == NAMED

    def test_a_reply_to_the_assistants_own_bubble_does_without_the_name(self, wake):
        assert wake.reason(GROUP, "and dessert?", OWN, NAMES, replies_to_own=True) == REPLY

    def test_the_message_right_after_the_assistants_own_does(self, wake):
        wake.spoke(GROUP, OWN)
        assert wake.reason(GROUP, "thanks, and the salad?", OWN, NAMES) == FOLLOWS

    def test_only_the_message_directly_after_it(self, wake):
        wake.spoke(GROUP, OWN)
        wake.settle(GROUP, ALICE, TURN)
        assert wake.reason(GROUP, "I'll be late", OWN, NAMES) == ""

    def test_the_rule_is_per_group(self, wake):
        wake.spoke(GROUP, OWN)
        assert wake.reason(OTHER, "anyone up for a game?", OWN, NAMES) == ""

    def test_a_message_without_words_wakes_nobody(self, wake):
        wake.spoke(GROUP, OWN)
        assert wake.reason(GROUP, "  ", OWN, NAMES, replies_to_own=True) == ""

    def test_a_consumer_without_an_id_of_its_own_is_followed_by_nothing(self, wake):
        """The Telegram feed without a bot token: nothing was ever settled under None on purpose."""
        wake.settle(GROUP, None, ROW)
        assert wake.reason(GROUP, "hello", None, NAMES) == ""


@pytest.mark.unit
class TestWhoWroteLast:
    def test_a_message_that_stayed_context_still_ends_the_follow_up(self, wake):
        wake.spoke(GROUP, OWN)
        wake.settle(GROUP, ALICE, ROW)
        assert not wake.follows(GROUP, OWN)

    def test_a_turn_the_chat_layer_refused_leaves_the_follow_up_armed(self, wake):
        """The group never saw that message answered: the next one still follows the assistant."""
        wake.spoke(GROUP, OWN)
        wake.settle(GROUP, BOB, REFUSED)
        assert wake.reason(GROUP, "hello?", OWN, NAMES) == FOLLOWS

    def test_the_groups_the_assistant_wrote_last_in_can_be_listed(self, wake):
        wake.spoke(GROUP, OWN)
        wake.spoke(OTHER, OWN)
        wake.settle(OTHER, ALICE, TURN)
        assert wake.following(OWN) == [GROUP]


@pytest.mark.unit
class TestTheBrake:
    def test_a_cap_lets_that_many_through_within_the_window(self):
        now = [1000.0]
        wake = GroupWake.GroupWake(clock=lambda: now[0])
        assert [wake.brake(GROUP, cap=2, window=60) for _ in range(3)] == [True, True, False]
        now[0] += 61
        assert wake.brake(GROUP, cap=2, window=60), "the hour moved on"

    def test_each_key_has_its_own_count(self, wake):
        assert wake.brake(GROUP, cap=1) and wake.brake(OTHER, cap=1)
        assert not wake.brake(GROUP, cap=1)

    def test_a_refusal_is_not_counted(self):
        now = [1000.0]
        wake = GroupWake.GroupWake(clock=lambda: now[0])
        wake.brake(GROUP, cap=1, window=60)
        for _ in range(5):
            wake.brake(GROUP, cap=1, window=60)
        now[0] += 61
        assert wake.brake(GROUP, cap=1, window=60)
