"""ChatState: what the iMessage chat still knows after a restart."""
import time

import pytest

from interfaces.imessage.ChatState import NAMES_PER_GROUP, ChatState

DAY = 86400


@pytest.fixture
def path(tmp_path):
    return tmp_path / "cache" / "imessage_chat.json"


@pytest.mark.unit
class TestAcrossARestart:
    def test_a_state_never_written_has_no_cursor(self, path):
        assert ChatState("local", path).cursor() is None

    def test_the_cursor_the_groups_and_the_counts_come_back(self, path):
        state = ChatState("local", path)
        state.save_cursor(4711)
        state.save_following(["iMessage;+;chat-b", "iMessage;+;chat-a"])
        state.count_sent("+46701740606")
        state.count_sent("+46701740606")
        again = ChatState("local", path)
        assert again.cursor() == 4711
        assert sorted(again.following()) == ["iMessage;+;chat-a", "iMessage;+;chat-b"]
        assert again.sent_today("+46701740606") == 2 and again.sent_today("+46701740605") == 0

    def test_a_state_about_another_account_is_not_this_ones(self, path):
        """A ROWID of another Mac's chat.db means nothing here, and neither do its groups."""
        other = ChatState("http://studio.local:8898", path)
        other.save_cursor(4711)
        other.save_following(["iMessage;+;chat-a"])
        mine = ChatState("local", path)
        assert mine.cursor() is None and mine.following() == []
        mine.save_cursor(12)
        assert ChatState("http://studio.local:8898", path).cursor() is None, "the file is the new account's now"

    @pytest.mark.parametrize("content", ["", "not json", "[1, 2]", '{"account": "local", "cursor": "12"}',
                                         '{"account": "local", "cursor": true}'],
                             ids=["empty", "garbage", "not an object", "a cursor as text", "a cursor as a flag"])
    def test_a_file_that_cannot_be_used_is_no_state_not_an_exception(self, path, content):
        path.parent.mkdir(parents=True)
        path.write_text(content)
        assert ChatState("local", path).cursor() is None

    @pytest.mark.parametrize("vouched", ['["iMessage;+;chat-family"]', '{"iMessage;+;chat-family": "+46701740605"}'],
                             ids=["a list of groups", "a handle where the list of them belongs"])
    def test_a_say_about_a_group_that_is_not_written_as_one_vouches_for_nothing(self, path, vouched):
        """What makes a group a place to post to is never read out of a damaged entry."""
        path.parent.mkdir(parents=True)
        path.write_text('{"account": "local", "vouched": %s}' % vouched)
        assert ChatState("local", path).vouched() == {}

    @pytest.mark.xfail(strict=True, reason="ChatState checks the type of the cursor, the vouched groups and the names, but "
                                           "not of 'sent' and 'following': a damaged one raises — from 'following' when the "
                                           "transport is built, from 'sent' at every send")
    @pytest.mark.parametrize("content", ['{"account": "local", "sent": "many"}',
                                         '{"account": "local", "sent": {"day": "%(today)s", "counts": {"+46701740606": "many"}}}',
                                         '{"account": "local", "following": 5}'],
                             ids=["the day's counts as text", "a count that is no number", "the groups as a number"])
    def test_a_damaged_count_or_group_list_is_no_state_not_an_exception(self, path, content):
        path.parent.mkdir(parents=True)
        path.write_text(content % {"today": time.strftime("%Y-%m-%d")})
        state = ChatState("local", path)
        assert state.following() == [] and state.sent_today("+46701740606") == 0
        assert state.count_sent("+46701740606") == 1

    def test_a_folder_that_cannot_be_written_costs_the_state_not_the_chat(self, tmp_path):
        blocked = tmp_path / "a-file"
        blocked.write_text("in the way")
        state = ChatState("local", blocked / "imessage_chat.json")
        state.save_cursor(5)
        assert state.cursor() == 5, "the running process still knows where it is"


@pytest.mark.unit
class TestTheDaysCount:
    def test_a_new_day_starts_from_nothing(self, path):
        state = ChatState("local", path)
        now = time.time()
        assert [state.count_sent("+46701740606", now) for _ in range(3)] == [1, 2, 3]
        assert state.sent_today("+46701740606", now) == 3
        assert state.sent_today("+46701740606", now + DAY) == 0
        assert state.count_sent("+46701740606", now + DAY) == 1

    def test_each_recipient_is_counted_alone(self, path):
        state = ChatState("local", path)
        state.count_sent("+46701740606")
        assert state.count_sent("iMessage;+;chat-family") == 1


FAMILY, CHESS = "iMessage;+;chat-family", "iMessage;+;chat-chess"
MARCUS, FRANCES = "+46701740605", "+46701740607"


@pytest.mark.unit
class TestAnOwnersSayAboutAGroup:
    def test_what_is_forgotten_stays_forgotten_after_a_restart(self, path):
        state = ChatState("local", path)
        state.vouch(FAMILY, MARCUS)
        state.vouch(FAMILY, FRANCES)
        state.vouch(CHESS, MARCUS)
        assert state.unvouch(FAMILY, [MARCUS]) is True
        assert state.unvouch(CHESS) is True
        assert ChatState("local", path).vouched() == {FAMILY: [FRANCES]}

    def test_forgetting_what_was_never_said_changes_nothing(self, path):
        state = ChatState("local", path)
        state.vouch(FAMILY, MARCUS)
        assert state.unvouch(FAMILY, [FRANCES]) is False and state.unvouch(CHESS) is False
        assert state.vouched() == {FAMILY: [MARCUS]}


@pytest.mark.unit
class TestTheNamesGroupsWerePostedToUnder:
    def test_a_name_stays_its_groups_across_a_restart_until_the_group_gives_its_names_up(self, path):
        state = ChatState("local", path)
        state.keep_name("family", FAMILY)
        state.keep_name("chess club", CHESS)
        again = ChatState("local", path)
        assert again.names() == {"family": FAMILY, "chess club": CHESS}
        assert again.free_names({FAMILY}) == ["chess club"]
        assert ChatState("local", path).names() == {"family": FAMILY}

    def test_a_group_renamed_over_and_over_keeps_only_its_latest_names(self, path):
        state = ChatState("local", path)
        state.keep_name("family", FAMILY)
        renames = 3 * NAMES_PER_GROUP
        for n in range(renames):
            state.keep_name(f"trip {n}", CHESS)
        names = state.names()
        assert names["family"] == FAMILY, "one group's renaming cost another its name"
        assert sum(guid == CHESS for guid in names.values()) < renames and f"trip {renames - 1}" in names
