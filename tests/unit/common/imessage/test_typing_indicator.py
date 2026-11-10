"""TypingIndicator over an in-memory Messages: when the dots show, in which window, how long —
and every way a step can be other than expected, each of which must end in no dots, nothing
typed anywhere else and no window left behind."""
import threading
from unittest.mock import patch

import pytest

from common.imessage import TypingIndicator as module
from common.imessage.Accessibility import Session
from common.imessage.MessagesHost import (TYPING_BUSY, TYPING_FAILED, TYPING_NO_ACCESSIBILITY, TYPING_NO_MESSAGES,
                                          TYPING_NO_ROW, TYPING_NO_WINDOW, TYPING_NOT_ON_CONSOLE, TYPING_SCREEN_LOCKED,
                                          TYPING_UNAVAILABLE)
from common.imessage.TypingIndicator import (KEEPALIVE_SECONDS, MAX_NODES, MAX_WINDOWS, RETRY_SECONDS, TIMEOUT_SECONDS,
                                             WINDOW_WAIT, TypingIndicator, names_row, same_name)

from .ax_fakes import Clock, FakeMessages, Node, Timers

FAMILY, CHARLOTTE = "iMessage;+;chat-family", "iMessage;-;+46701740606"
FAMILY_NAME, CHARLOTTE_NAME = "Family", "Charlotte Lindqvist"


class Dots:
    """An indicator over a fake Messages whose sidebar shows the family group and Charlotte."""

    def __init__(self):
        self.mac = FakeMessages()
        self.mac.chat(FAMILY_NAME)
        self.mac.chat(CHARLOTTE_NAME, latest="Running late, sorry")
        self.clock = Clock()
        self.timers = Timers(self.clock)
        self.indicator = TypingIndicator(self.mac, self.clock, self.clock.sleep, self.timers)

    def on(self, chat=FAMILY, name=FAMILY_NAME):
        return self.indicator.on(chat, name)

    def off(self, chat=FAMILY):
        return self.indicator.off(chat)

    def untouched(self) -> bool:
        """Nothing was typed anywhere and no window was opened or closed."""
        return (self.mac.typed, self.mac.opened, self.mac.closed) == ([], [], [])

    def nothing_left(self) -> bool:
        """Only the main window is on screen, its own compose field as the user left it."""
        return self.mac.windows == [self.mac.main] and self.mac.draft("Messages") == ""


@pytest.fixture
def dots():
    return Dots()


@pytest.mark.unit
class TestShowingTheDots:
    def test_the_chats_own_window_is_opened_and_typed_into(self, dots):
        assert dots.on().showing
        assert dots.mac.opened == [FAMILY_NAME]
        assert dots.mac.draft(FAMILY_NAME) != "", "the dots show while the compose field holds text"
        assert {title for title, _ in dots.mac.typed} == {FAMILY_NAME}
        assert dots.mac.draft("Messages") == "", "the main window's compose field is the user's"

    def test_the_field_is_rewritten_while_the_reply_is_worked_on_and_each_write_is_a_change(self, dots):
        dots.on()
        for _ in range(3):
            assert dots.timers.tick()
        values = [value for _, value in dots.mac.typed]
        assert len(values) == 4 and all(value for value in values)
        assert all(now != before for before, now in zip(values, values[1:]))
        assert dots.clock.now - 1000.0 == 3 * KEEPALIVE_SECONDS

    def test_off_clears_the_field_before_it_closes_the_window(self, dots):
        dots.on()
        seen = []
        close = dots.mac._close
        dots.mac._close = lambda window: (seen.append(dots.mac.draft(FAMILY_NAME)), close(window))
        assert not dots.off().showing
        assert seen == [""], "a window closed over a draft leaves the draft in the conversation"
        assert dots.mac.closed == [FAMILY_NAME] and dots.nothing_left()
        assert not dots.timers.armed and dots.indicator.showing() == []

    def test_asked_again_while_they_show_no_second_window_opens(self, dots):
        dots.on()
        assert dots.on().showing
        assert dots.mac.opened == [FAMILY_NAME] and len(dots.mac.typed) == 1

    def test_off_for_a_chat_without_dots_touches_nothing(self, dots):
        dots.off()
        assert dots.mac.calls == 0

    def test_a_window_that_takes_a_moment_to_open_is_waited_for(self, dots):
        dots.mac.open_after_polls = 3
        assert dots.on().showing
        assert 0 < dots.clock.now - 1000.0 < WINDOW_WAIT

    def test_a_row_with_its_name_once_more_as_a_label_inside_is_one_row(self, dots):
        dots.mac.chat("Chess Club", labelled=True)
        assert dots.on("iMessage;+;chat-chess", "Chess Club").showing


@pytest.mark.unit
class TestTheGates:
    GATES = {
        TYPING_NO_ACCESSIBILITY: lambda mac: setattr(mac, "granted", False),
        TYPING_NOT_ON_CONSOLE: lambda mac: setattr(mac, "session_state", Session(on_console=False, locked=False)),
        TYPING_SCREEN_LOCKED: lambda mac: setattr(mac, "session_state", Session(on_console=True, locked=True)),
        TYPING_NO_MESSAGES: lambda mac: setattr(mac, "running", False),
        TYPING_NO_WINDOW: lambda mac: mac.windows.clear(),
    }

    @pytest.mark.parametrize("gate", list(GATES))
    def test_a_closed_gate_means_no_dots_and_nothing_touched(self, dots, gate):
        self.GATES[gate](dots.mac)
        result = dots.on()
        assert (result.showing, result.reason) == (False, gate) and result.detail
        assert dots.untouched()

    def test_a_process_without_a_login_session_is_not_on_the_console(self, dots):
        dots.mac.session_state = None
        assert dots.on().reason == TYPING_NOT_ON_CONSOLE

    def test_without_the_grant_not_one_call_reaches_messages(self, dots):
        dots.mac.granted = False
        dots.on()
        assert dots.mac.calls == 1, "the grant is asked about, and nothing after it"

    def test_a_locked_screen_is_seen_before_any_window_is_looked_at(self, dots):
        """A locked session's one window contains itself: the gate is what keeps the walk out of it."""
        dots.mac.session_state = Session(on_console=True, locked=True)
        dots.on()
        assert dots.mac.calls == 2

    def test_messages_with_only_another_window_open_has_no_sidebar_to_find_the_chat_in(self, dots):
        dots.mac.windows[:] = [dots.mac.chat_window("Settings")]
        assert dots.on().reason == TYPING_NO_WINDOW and dots.mac.typed == []

    def test_frameworks_that_cannot_be_loaded_are_a_gate_not_an_exception(self, dots):
        dots.mac.trusted = lambda: (_ for _ in ()).throw(ImportError("No module named 'ApplicationServices'"))
        assert dots.on().reason == TYPING_UNAVAILABLE
        assert dots.indicator.blocked() == TYPING_UNAVAILABLE

    def test_a_grant_given_later_works_at_the_next_reply_without_a_restart(self, dots):
        dots.mac.granted = False
        dots.on()
        assert dots.indicator.blocked() == TYPING_NO_ACCESSIBILITY
        dots.mac.granted = True
        assert dots.indicator.blocked() == "", "the health answer follows the Mac, not the last reply"
        assert dots.on().showing

    def test_nobody_asked_for_dots_so_the_health_look_loads_nothing(self, dots):
        assert dots.indicator.blocked() == "" and dots.mac.calls == 0

    def test_a_gate_is_logged_once_however_many_replies_meet_it(self, dots):
        dots.mac.granted = False
        with patch.object(module, "info") as said:
            for _ in range(4):
                dots.on()
                dots.on(CHARLOTTE, CHARLOTTE_NAME)
            assert said.call_count == 1
            dots.mac.granted = True
            dots.on()
            assert said.call_count == 2, "... and once more when it opens"


@pytest.mark.unit
class TestFindingTheRightChat:
    @pytest.mark.parametrize("shown, name, same", [
        ("Family", "Family", True),
        ("family", "Family", False),
        ("Family Dinner", "Family", False),
        ("Famil", "Family", False),
        ("", "Family", False),
        (None, "Family", False),
        ("Family", "", False),
        ("Charlotte Lindqvist", "Charlotte Lindqvist", True),
        ("‪+46 70‑174 06 05‬", "+46701740605", True),
        ("+46 (70) 174-06-05", "+46701740605", True),
        ("+46 70 174 06 06", "+46701740605", False),
        ("070-174 06 05", "+46701740605", False),
        ("+46 70 174 06 05 & Bob", "+46701740605", False),
        ("Marcus", "Märcus", False),
        ("Märcus", "Märcus", True),
    ])
    def test_a_name_is_the_same_only_when_it_is_exactly_that_chat(self, shown, name, same):
        assert same_name(shown, name) is same

    @pytest.mark.parametrize("description, name, leads", [
        ("Family, See you at seven", "Family", True),
        ("Family", "Family", True),
        ("Family Dinner, See you at seven", "Family", False),
        ("Charlotte Lindqvist, Family, see you", "Family", False),
        ("Alice, Bob & Carol, Sure, fine", "Alice, Bob & Carol", True),
        ("+46 70 174 06 05, On my way", "+46701740605", True),
        ("+46 70 174 06 05 6, On my way", "+46701740605", False),
        ("Maybe: Family, see you", "Family", False),
    ])
    def test_a_row_is_a_chats_only_by_its_leading_name(self, description, name, leads):
        assert names_row(description, name) is leads

    def test_one_description_cannot_tell_a_chat_from_a_longer_named_one_so_both_visible_means_no_dots(self, dots):
        """"Family, Trip, hello" is the group "Family, Trip" — or "Family" saying "Trip, hello"."""
        dots.mac.chat("Family, Trip", latest="hello")
        assert dots.on().reason == TYPING_NO_ROW and dots.untouched()
        assert dots.on("iMessage;+;chat-trip", "Family, Trip").showing

    def test_a_chat_that_is_not_among_the_visible_rows_gets_no_dots(self, dots):
        result = dots.on("iMessage;+;chat-chess", "Chess Club")
        assert (result.showing, result.reason) == (False, TYPING_NO_ROW) and dots.untouched()

    def test_two_rows_with_the_chats_name_are_nobodys_row(self, dots):
        dots.mac.chat(FAMILY_NAME, latest="The other Family group")
        assert dots.on().reason == TYPING_NO_ROW and dots.untouched()

    def test_a_longer_name_that_starts_with_the_chats_is_another_chat(self, dots):
        dots.mac.sidebar.children.clear()
        dots.mac.chat("Family Dinner")
        dots.mac.chat("Charlotte Lindqvist-Berg")
        assert dots.on().reason == TYPING_NO_ROW
        assert dots.on(CHARLOTTE, CHARLOTTE_NAME).reason == TYPING_NO_ROW and dots.untouched()

    def test_a_chat_named_by_a_number_is_found_however_messages_groups_the_digits(self, dots):
        dots.mac.chat("+46 70 174 06 05")
        assert dots.on("iMessage;-;+46701740605", "+46701740605").showing

    def test_a_row_that_opens_another_chats_window_is_closed_again_unwritten(self, dots):
        """An unnamed group shown by its members leads with a member's name: the row looks like
        that person's, the window it opens says otherwise."""
        dots.mac.sidebar.children.clear()
        dots.mac.chat("Alice, Bob", latest="Sounds good")
        result = dots.on("iMessage;-;alice@example.com", "Alice")
        assert (result.showing, result.reason) == (False, TYPING_FAILED)
        assert dots.mac.typed == [] and dots.mac.closed == ["Alice, Bob"] and dots.nothing_left()

    def test_such_a_row_is_not_pressed_for_that_chat_again(self, dots):
        """The press showed whose row it is: it is the longer-named chat's from then on."""
        dots.mac.sidebar.children.clear()
        dots.mac.chat("Alice, Bob")
        for _ in range(3):
            dots.on("iMessage;-;alice@example.com", "Alice")
        dots.clock.now += RETRY_SECONDS + 1
        assert dots.on("iMessage;-;alice@example.com", "Alice").reason == TYPING_NO_ROW
        assert dots.mac.opened == ["Alice, Bob"], "the group's window was opened and closed again"
        assert dots.on("iMessage;+;chat-ab", "Alice, Bob").showing, "the group itself lost its dots"

    def test_once_her_own_row_is_visible_she_gets_her_dots(self, dots):
        dots.mac.sidebar.children.clear()
        dots.mac.chat("Alice, Bob")
        dots.on("iMessage;-;alice@example.com", "Alice")
        dots.clock.now += RETRY_SECONDS + 1
        dots.mac.chat("Alice", latest="On my way")
        assert dots.on("iMessage;-;alice@example.com", "Alice").showing and dots.mac.typed[-1][0] == "Alice"

    def test_a_window_that_appears_at_the_press_and_is_not_provably_the_rows_is_left_alone(self, dots):
        """The user opened another chat in the very two seconds the row's window was waited
        for: nothing says it is Keep's, so its close button is not pressed."""
        dots.mac.sidebar.children.clear()
        dots.mac.chat(FAMILY_NAME, opens=CHARLOTTE_NAME)
        result = dots.on()
        assert (result.showing, result.reason) == (False, TYPING_FAILED)
        assert dots.mac.closed == [] and dots.mac.typed == [] and dots.mac.window(CHARLOTTE_NAME) is not None
        dots.on()
        assert len(dots.mac.opened) == 1, "the row was pressed again at the next reply"

    def test_an_interrupt_while_the_window_is_waited_for_does_not_leave_it_open(self, dots):
        """Ctrl-C in the live-check tool, during the up to two seconds of the wait."""
        dots.mac.open_after_polls = 1

        def interrupted(seconds):
            raise KeyboardInterrupt

        indicator = TypingIndicator(dots.mac, dots.clock, interrupted, dots.timers)
        with pytest.raises(KeyboardInterrupt):
            indicator.on(FAMILY, FAMILY_NAME)
        assert dots.mac.closed == [FAMILY_NAME] and dots.nothing_left() and indicator.showing() == []

    def test_a_window_the_user_opened_for_the_chat_is_never_typed_into_or_closed(self, dots):
        theirs = dots.mac.chat_window(FAMILY_NAME)
        result = dots.on()
        assert (result.showing, result.reason) == (False, TYPING_FAILED)
        assert dots.untouched() and theirs in dots.mac.windows

    def test_a_window_the_user_has_open_for_another_chat_is_left_alone(self, dots):
        theirs = dots.mac.chat_window(CHARLOTTE_NAME)
        dots.on()
        dots.timers.tick()
        dots.off()
        assert {title for title, _ in dots.mac.typed} == {FAMILY_NAME}
        assert dots.mac.closed == [FAMILY_NAME] and theirs in dots.mac.windows

    def test_a_chat_without_a_name_is_not_looked_for(self, dots):
        assert dots.on(FAMILY, "  ").reason == TYPING_FAILED and dots.untouched()


@pytest.mark.unit
class TestAWindowThatIsNotAsExpected:
    def test_no_window_opens(self, dots):
        dots.mac.sidebar.children[0].actions = {key: lambda: None for key in dots.mac.sidebar.children[0].actions}
        result = dots.on()
        assert (result.showing, result.reason) == (False, TYPING_FAILED) and dots.mac.typed == []
        assert WINDOW_WAIT <= dots.clock.now - 1000.0 < WINDOW_WAIT + 1, "the wait for the window is bounded"

    def test_a_window_without_a_compose_field_is_closed_again(self, dots):
        open_window = dots.mac._open

        def without_field(title):
            open_window(title)
            dots.mac.pending[-1].children[0].children.pop()
        dots.mac._open = without_field
        assert dots.on().reason == TYPING_FAILED
        assert dots.mac.typed == [] and dots.mac.closed == [FAMILY_NAME] and dots.nothing_left()

    def test_a_field_that_takes_no_value_ends_in_a_closed_window(self, dots):
        open_window = dots.mac._open

        def frozen(title):
            open_window(title)
            dots.mac.pending[-1].children[0].children[-1].frozen = True
        dots.mac._open = frozen
        assert dots.on().reason == TYPING_FAILED
        assert dots.mac.closed == [FAMILY_NAME] and dots.nothing_left()

    def test_a_title_that_changes_while_the_field_is_looked_for_means_nothing_is_written(self, dots):
        """The title is read again right before the first write: finding the field takes a moment."""
        def retitle_once_open():
            window = dots.mac.window(FAMILY_NAME)
            if window is not None:
                seen.append(1)
                if len(seen) == 3:      # past the look that found the window, inside the search for its field
                    window.attributes[module.TITLE] = "Chess Club"
        seen: list[int] = []
        dots.mac.on_call = retitle_once_open
        assert dots.on().reason == TYPING_FAILED
        assert dots.mac.typed == [] and dots.mac.closed == ["Chess Club"] and dots.nothing_left()

    def test_a_title_that_changes_while_the_dots_show_ends_them_at_the_next_rewrite(self, dots):
        dots.on()
        dots.mac.window(FAMILY_NAME).attributes[module.TITLE] = "Chess Club"
        dots.timers.tick()
        assert [value for _, value in dots.mac.typed[1:]] == [""], "nothing but the clearing was written there"
        assert dots.mac.closed == ["Chess Club"], "the window is this code's, whatever it shows now"
        assert dots.indicator.showing() == [] and not dots.timers.armed

    def test_a_field_that_is_gone_at_a_rewrite_ends_them(self, dots):
        dots.on()
        dots.mac.field_of(FAMILY_NAME).attributes[module.IDENTIFIER] = None
        dots.timers.tick()
        assert all(value in ("", module.VALUES[0]) for _, value in dots.mac.typed)
        assert dots.mac.closed == [FAMILY_NAME] and dots.indicator.showing() == []

    def test_a_window_the_user_closed_meanwhile_is_simply_gone(self, dots):
        dots.on()
        dots.mac._close(dots.mac.window(FAMILY_NAME))
        with patch.object(module, "error") as complained:
            dots.timers.tick()
        complained.assert_not_called()
        assert dots.indicator.showing() == []
        assert dots.on().showing, "the next reply opens the chat's window anew"

    def test_asked_again_after_the_user_closed_the_window_it_is_opened_anew(self, dots):
        dots.on()
        dots.mac._close(dots.mac.window(FAMILY_NAME))
        assert dots.on().showing and dots.mac.opened == [FAMILY_NAME, FAMILY_NAME]

    def test_a_seam_that_starts_raising_under_the_keep_alive_is_said_once_and_ends_it(self, dots):
        dots.on()
        dots.mac.session = lambda: (_ for _ in ()).throw(ValueError("depythonifying 'pointer', got 'NoneType'"))
        with patch.object(module, "error") as complained:
            dots.timers.tick()
            assert not dots.timers.tick(), "no timer is left running to say it again every few seconds"
        assert complained.call_count == 1 and dots.indicator.showing() == []

    def test_a_seam_that_raises_is_an_answer_too(self, dots):
        dots.mac.session = lambda: (_ for _ in ()).throw(ValueError("depythonifying 'pointer', got 'NoneType'"))
        result = dots.on()
        assert (result.showing, result.reason) == (False, TYPING_FAILED)
        assert not dots.off().showing


@pytest.mark.unit
class TestEveryWalkIsCapped:
    def test_a_window_that_contains_itself_ends_the_walk(self, dots):
        """What Messages shows of a locked session, should the lock flag ever not say so."""
        loop = Node("AXApplication", title="Messages")
        loop.children.append(loop)
        dots.mac.windows[:] = [loop]
        result = dots.on()
        assert (result.showing, result.reason) == (False, TYPING_NO_WINDOW)
        assert dots.mac.calls < 200 and dots.mac.typed == []

    def test_a_sidebar_larger_than_messages_ever_shows_is_not_read_to_its_end(self, dots):
        for number in range(MAX_NODES):
            dots.mac.sidebar.children.append(Node(module.ROW_ROLE, description=f"Chat {number}, Hello"))
        result = dots.on()
        assert (result.showing, result.reason) == (False, TYPING_FAILED) and dots.untouched()

    def test_a_tree_nested_past_the_depth_cap_is_not_followed(self, dots):
        deep = dots.mac.sidebar
        for _ in range(module.MAX_DEPTH + 2):
            deep.children.append(Node())
            deep = deep.children[-1]
        assert dots.on().reason == TYPING_FAILED and dots.untouched()

    def test_a_messages_that_answers_slowly_is_given_up_on(self, dots):
        for number in range(200):
            dots.mac.chat(f"Chat {number}")
        dots.mac.on_call = lambda: dots.clock.sleep(0.02)
        result = dots.on()
        assert (result.showing, result.reason) == (False, TYPING_FAILED) and dots.untouched()
        assert dots.clock.now - 1000.0 < 3 * module.WALK_SECONDS


@pytest.mark.unit
class TestDotsNobodyTurnedOff:
    def test_they_end_by_themselves_and_the_window_is_closed(self, dots):
        dots.on()
        ticks = 0
        while dots.timers.tick():
            ticks += 1
            assert ticks < 100
        lasted = dots.clock.now - 1000.0
        assert TIMEOUT_SECONDS <= lasted < TIMEOUT_SECONDS + KEEPALIVE_SECONDS
        assert dots.mac.typed[-1] == (FAMILY_NAME, "") and dots.mac.closed == [FAMILY_NAME]
        assert dots.nothing_left() and dots.indicator.showing() == []

    def test_each_round_of_a_long_turn_starts_the_bound_anew(self, dots):
        dots.on()
        for _ in range(int(TIMEOUT_SECONDS / KEEPALIVE_SECONDS) - 2):
            dots.timers.tick()
        dots.on()
        for _ in range(5):
            dots.timers.tick()
        assert dots.clock.now - 1000.0 > TIMEOUT_SECONDS and dots.indicator.showing() == [FAMILY]

    def test_the_bound_is_each_chats_own(self, dots):
        dots.on()
        for _ in range(12):
            dots.timers.tick()
        dots.on(CHARLOTTE, CHARLOTTE_NAME)
        for _ in range(12):
            dots.timers.tick()
        assert dots.indicator.showing() == [CHARLOTTE] and dots.mac.closed == [FAMILY_NAME]


@pytest.mark.unit
class TestSeveralChatsAtOnce:
    def test_each_chat_has_its_own_window_and_its_own_writes(self, dots):
        dots.on()
        dots.on(CHARLOTTE, CHARLOTTE_NAME)
        dots.timers.tick()
        assert dots.mac.opened == [FAMILY_NAME, CHARLOTTE_NAME]
        assert sorted(title for title, _ in dots.mac.typed) == sorted([FAMILY_NAME, CHARLOTTE_NAME] * 2)

    def test_one_chats_off_leaves_the_others_dots_alone(self, dots):
        dots.on()
        dots.on(CHARLOTTE, CHARLOTTE_NAME)
        dots.off()
        before = len(dots.mac.typed)
        dots.timers.tick()
        assert dots.mac.closed == [FAMILY_NAME] and dots.mac.draft(CHARLOTTE_NAME) != ""
        assert dots.mac.typed[before:] and {title for title, _ in dots.mac.typed[before:]} == {CHARLOTTE_NAME}

    def test_no_more_windows_are_kept_open_than_the_bound(self, dots):
        chats = [(f"iMessage;+;chat-{number}", f"Group {number}") for number in range(MAX_WINDOWS + 1)]
        for _, name in chats:
            dots.mac.chat(name)
        results = [dots.on(chat, name) for chat, name in chats]
        assert [result.showing for result in results] == [True] * MAX_WINDOWS + [False]
        assert results[-1].reason == TYPING_BUSY and len(dots.mac.windows) == 1 + MAX_WINDOWS
        dots.off(chats[0][0])
        assert dots.on(*chats[-1]).showing, "a window that closed makes room"

    def test_two_threads_asking_for_two_chats_both_get_their_dots(self, dots):
        results = {}
        threads = [threading.Thread(target=lambda c=chat, n=name: results.__setitem__(c, dots.on(c, n)))
                   for chat, name in ((FAMILY, FAMILY_NAME), (CHARLOTTE, CHARLOTTE_NAME))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)
        assert results[FAMILY].showing and results[CHARLOTTE].showing
        assert sorted(dots.mac.opened) == sorted([FAMILY_NAME, CHARLOTTE_NAME])


@pytest.mark.unit
class TestNeverLeavingAMess:
    def test_shutdown_clears_and_closes_every_window_it_opened(self, dots):
        dots.on()
        dots.on(CHARLOTTE, CHARLOTTE_NAME)
        theirs = dots.mac.chat_window("Chess Club")
        dots.indicator.shutdown()
        assert sorted(dots.mac.closed) == sorted([FAMILY_NAME, CHARLOTTE_NAME])
        assert dots.mac.windows == [dots.mac.main, theirs] and not dots.timers.armed
        assert all(value == "" for _, value in dots.mac.typed[-2:])

    def test_a_screen_locked_mid_reply_is_not_written_through_and_the_window_goes_once_it_unlocks(self, dots):
        dots.on()
        dots.mac.session_state = Session(on_console=True, locked=True)
        dots.timers.tick()
        assert len(dots.mac.typed) == 1 and dots.mac.closed == [], "a locked session is not reached into"
        assert dots.indicator.showing() == [] and dots.timers.armed, "... and the window is not forgotten"
        dots.mac.session_state = Session(on_console=True, locked=False)
        dots.timers.tick()
        assert dots.mac.typed[-1] == (FAMILY_NAME, "") and dots.mac.closed == [FAMILY_NAME]
        assert dots.nothing_left() and not dots.timers.armed

    def test_an_off_that_meets_a_locked_screen_is_made_good_the_same_way(self, dots):
        dots.on()
        dots.mac.session_state = Session(on_console=True, locked=True)
        assert dots.off().reason == TYPING_SCREEN_LOCKED and dots.mac.closed == []
        dots.mac.session_state = Session(on_console=True, locked=False)
        assert dots.on().showing, "the stale window is cleared and closed before the chat's is opened anew"
        assert dots.mac.closed == [FAMILY_NAME] and len(dots.mac.windows) == 2

    def test_a_reason_is_logged_once_per_chat_not_at_every_reply(self, dots):
        with patch.object(module, "info") as said:
            for _ in range(5):
                dots.on("iMessage;+;chat-chess", "Chess Club")
            assert said.call_count == 1
