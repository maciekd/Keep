"""tools/imessage_live_check: what the live check refuses to do, whom it shows dots to, when it
sends, and what it never prints — over a fixture chat.db and an in-memory Messages. Its real run
is a person's, at an unlocked Mac with a phone in hand."""
import importlib.util
import sqlite3
import subprocess
import sys
from unittest.mock import Mock, patch

import pytest

from common.Paths import AISERVER_DIR
from common.imessage import MessagesSender as sender_module
from common.imessage.Contacts import Contacts
from common.imessage.LocalMessagesHost import LocalMessagesHost
from common.imessage.MessagesDatabase import MessagesDatabase
from common.imessage.TypingIndicator import TypingIndicator
from tests.unit.common.imessage.ax_fakes import Clock, FakeMessages, Timers
from tests.unit.common.imessage.messages_fakes import ChatDb, GROUP, NOW, address_book

MARCUS, STRANGER = "+46701740605", "+46701740699"
SECRET = "The door code is 4711"

_spec = importlib.util.spec_from_file_location(
    "imessage_live_check", AISERVER_DIR / "tools" / "imessage_live_check" / "imessage_live_check.py")
live_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(live_check)


class Run:
    """The check over a Mac whose Messages has a chat with Marcus (the one who runs it), a chat
    with somebody else and a group; the phone's owner answers every question with `answer`."""

    def __init__(self, tmp_path, answer="y", indicator_sleep=None):
        db = ChatDb(tmp_path / "chat.db")
        db.handle(1, MARCUS).handle(2, STRANGER)
        db.chat(1, f"iMessage;-;{MARCUS}", [1]).chat(2, f"iMessage;-;{STRANGER}", [2])
        db.chat(3, "iMessage;+;chat-family", [1, 2], name="Family", style=GROUP)
        db.message(1, "Hello", ts=NOW - 300, chat=1, sender=1, guid="G-1", service="iMessage")
        db.message(2, SECRET, ts=NOW - 200, chat=2, sender=2, guid="G-2", service="SMS")
        db.message(3, "ok", ts=NOW - 100, chat=1, guid="G-3", service="iMessage", thread_originator_guid="G-1")
        self.mac = FakeMessages()
        self.mac.chat("Marcus Lindqvist", latest="Hello")
        self.mac.chat("+46 70 174 06 99", latest=SECRET)
        self.clock = Clock()
        database = MessagesDatabase(db.path)
        people = [("Marcus", "Lindqvist", [("070-174 06 05", "_$!<Mobile>!$_")], [])]
        self.host = LocalMessagesHost(database, Contacts(database, address_book(tmp_path, people)),
                                      typing_indicator=TypingIndicator(self.mac, self.clock,
                                                                       indicator_sleep or self.clock.sleep,
                                                                       Timers(self.clock)))
        self.said: list[str] = []
        self.asked: list[str] = []
        self.answer = answer
        self.sleep = self.clock.sleep

    def __call__(self, *argv) -> int:
        def ask(question):
            self.asked.append(question)
            return self.answer
        sent = Mock(returncode=0, stdout="sent:imessage", stderr="")
        with patch.object(sender_module.subprocess, "run", return_value=sent) as self.osascript:
            return live_check.main(list(argv), host=self.host, ax=self.mac, say=self.said.append, ask=ask,
                                   sleep=self.sleep)

    @property
    def output(self) -> str:
        return "\n".join(self.said)

    def messages_untouched(self) -> bool:
        return (self.mac.typed, self.mac.opened, self.mac.closed) == ([], [], [])


@pytest.fixture
def run(tmp_path):
    return Run(tmp_path)


@pytest.mark.unit
class TestWhatItRefuses:
    def test_without_a_handle_it_does_nothing_at_all(self, run):
        assert run() == 2
        assert run.mac.calls == 0 and run.messages_untouched() and not run.osascript.called

    @pytest.mark.parametrize("argument", ["Charlotte", "0701740605", "+4670", "iMessage;+;chat-family",
                                          f"{MARCUS},{STRANGER}", "marcus@", " "])
    def test_anything_but_one_persons_handle_is_refused(self, run, argument):
        assert run(argument) == 2
        assert run.mac.calls == 0 and run.messages_untouched() and not run.osascript.called

    def test_a_flag_alone_is_no_handle(self, run):
        assert run("--send-test-message") == 2 and not run.osascript.called

    def test_started_as_the_script_it_is_without_a_handle_it_refuses_and_loads_nothing(self, tmp_path):
        """From any folder, as a person would start it: the refusal comes before Keep's modules
        are even imported, so nothing of the Mac can have been looked at."""
        script = AISERVER_DIR / "tools" / "imessage_live_check" / "imessage_live_check.py"
        result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True, timeout=30)
        assert result.returncode == 2 and "handle" in result.stdout
        assert "common" not in result.stdout + result.stderr, "no log line of Keep's: nothing of it ran"

    def test_a_number_is_taken_as_people_write_it(self):
        assert live_check.own_handle("+46 70-174 06 05") == MARCUS
        assert live_check.own_handle("Marcus@MLab.test") == "marcus@mlab.test"


@pytest.mark.unit
class TestTheDots:
    def test_they_go_to_the_chat_of_the_handle_given_and_nowhere_else(self, run):
        assert run(MARCUS) == 0
        assert run.mac.opened == ["Marcus Lindqvist"]
        assert {title for title, _ in run.mac.typed} == {"Marcus Lindqvist"}

    def test_they_are_cleared_and_the_window_closed_when_it_is_done(self, run):
        run(MARCUS)
        assert run.mac.typed[-1] == ("Marcus Lindqvist", "") and run.mac.windows == [run.mac.main]

    def test_they_stay_on_for_the_time_asked(self, run):
        run(MARCUS, "--seconds", "20")
        assert run.clock.now - 1000.0 >= 20

    def test_no_message_is_sent_unless_asked(self, run):
        run(MARCUS)
        assert not run.osascript.called

    def test_what_the_phone_showed_is_the_verdict(self, tmp_path):
        run = Run(tmp_path, answer="n")
        assert run(MARCUS) == 1 and run.asked

    def test_an_interrupted_run_still_takes_the_dots_away(self, run):
        def interrupted(seconds):
            raise KeyboardInterrupt
        run.sleep = interrupted
        with pytest.raises(KeyboardInterrupt):
            run(MARCUS)
        assert run.mac.typed[-1][1] == "" and run.mac.windows == [run.mac.main]

    def test_a_ctrl_c_while_the_chats_window_is_opening_leaves_no_window_behind(self, tmp_path):
        """The row is pressed and the typing code waits for the window: it has not taken it for
        its own yet, so neither its `off` nor its shutdown would close it."""
        def interrupted(seconds):
            raise KeyboardInterrupt
        run = Run(tmp_path, indicator_sleep=interrupted)
        run.mac.open_after_polls = 1
        own = run.mac.chat_window("Charlotte Lindqvist")        # a window the user has open themselves
        with pytest.raises(KeyboardInterrupt):
            run(MARCUS)
        assert run.mac.opened == ["Marcus Lindqvist"], "the interrupt came after the press"
        assert run.mac.closed == ["Marcus Lindqvist"] and run.mac.pending == []
        assert run.mac.windows == [run.mac.main, own] and run.mac.typed == []

    def test_a_closed_gate_is_reported_and_no_window_is_touched(self, run):
        run.mac.granted = False
        assert run(MARCUS, "--send-test-message") == 1
        assert run.messages_untouched() and not run.osascript.called

    def test_report_only_touches_no_window_and_sends_nothing(self, run):
        assert run(MARCUS, "--report-only") == 0
        assert run.messages_untouched() and not run.osascript.called

    def test_report_only_with_a_test_message_is_refused_rather_than_announced_and_not_sent(self, run):
        assert run(MARCUS, "--report-only", "--send-test-message") == 2
        assert run.mac.calls == 0 and run.messages_untouched() and not run.osascript.called

    def test_a_chat_the_sidebar_does_not_show_fails_the_check_without_typing_anywhere(self, run):
        run.mac.sidebar.children.pop(0)
        assert run(MARCUS) == 1 and run.mac.typed == []


@pytest.mark.unit
class TestTheTestMessage:
    def test_with_the_flag_one_message_goes_to_that_same_handle_after_the_dots_are_gone(self, run):
        on_screen = []
        code = None

        def send(*args, **kwargs):
            on_screen.append(list(run.mac.windows))
            return Mock(returncode=0, stdout="sent:imessage", stderr="")
        with patch.object(sender_module.subprocess, "run", side_effect=send) as osascript:
            code = live_check.main([MARCUS, "--send-test-message"], host=run.host, ax=run.mac, say=run.said.append,
                                   ask=lambda question: "y", sleep=run.clock.sleep)
        assert code == 0 and osascript.call_count == 1
        script = osascript.call_args.args[0][-1]
        assert MARCUS in script and STRANGER not in script and live_check.TEST_MESSAGE in script
        assert on_screen == [[run.mac.main]], "the dots' window was closed before the send"

    def test_dots_that_did_not_show_send_nothing(self, tmp_path):
        run = Run(tmp_path, answer="n")
        assert run(MARCUS, "--send-test-message") == 1 and not run.osascript.called


@pytest.mark.unit
class TestWhatItNeverPrints:
    def test_no_message_text_and_nobody_elses_handle(self, run):
        run(MARCUS)
        assert SECRET not in run.output and "4711" not in run.output
        assert STRANGER not in run.output and "174 06 99" not in run.output

    def test_not_even_when_the_own_chat_is_not_found_and_the_rows_are_looked_through(self, run):
        run.mac.sidebar.children.pop(0)
        run.mac.chat("+46 70 174 06 05 (work)", latest="Hello")
        run(MARCUS)
        assert "(work)" in run.output, "how the sidebar spells the row of the own handle is the finding"
        assert SECRET not in run.output and "174 06 99" not in run.output

    def test_a_contacts_row_spelled_in_another_order_is_shown_and_nobody_elses(self, run):
        """With a contact card the row carries the person's name, not the handle: the finding is
        how the sidebar spells that name."""
        run.mac.sidebar.children.pop(0)
        run.mac.chat("Lindqvist Marcus", latest="Hello")
        run.mac.chat("Charlotte Lindqvist", latest="Hello")
        assert run(MARCUS) == 1
        assert "Lindqvist Marcus" in run.output
        assert "Charlotte" not in run.output and SECRET not in run.output and "174 06 99" not in run.output

    @pytest.mark.parametrize("rows, shown", [
        (["Lindqvist Marcus, hi", f"Charlotte Lindqvist, {SECRET}", "Marcus, hi"], ["Lindqvist Marcus"]),
        (["MARCUS  lindqvist, hi", "Marcus Lindqvist Senior, hi"], ["MARCUS  lindqvist"]),
        (["Marcus, Lindqvist", "Lindqvist, Marcus Lindqvist"], []),
    ], ids=["another order", "another case", "a part of the name, or the name in a message"])
    def test_a_row_named_with_the_words_of_the_expected_name_is_the_contacts_own(self, rows, shown):
        assert live_check.rows_spelling(MARCUS, rows, "Marcus Lindqvist") == shown

    def test_without_an_expected_name_no_row_is_shown_for_its_words(self):
        assert live_check.rows_spelling(MARCUS, ["Lindqvist Marcus, hi", ", hi", "   "], "") == []

    @pytest.mark.parametrize("handle, rows, shown", [
        (MARCUS, ["+46 70 174 06 05, hi", f"Charlotte Lindqvist, {SECRET}", "+46 70 174 06 99, hi"], ["+46 70 174 06 05"]),
        ("m@mlab.test", ["m@mlab.test, hi", "Charlotte Lindqvist, write to m@mlab.test"], ["m@mlab.test"]),
        ("m@mlab.test", ["Charlotte Lindqvist, hi", "Family, hello"], []),
    ])
    def test_only_a_row_that_carries_the_own_handle_in_its_name_is_ever_shown(self, handle, rows, shown):
        assert live_check.rows_spelling(handle, rows) == shown


@pytest.mark.unit
class TestTheDatabaseReport:
    def test_the_columns_it_asks_for_are_the_ones_the_suites_chat_db_has(self, run):
        """The fixture schema is what the host layer's tests read: a column the check lists and
        the fixture lacks is a column nothing here is known to read."""
        assert live_check.report_database(run.host.database.path, run.said.append) is True

    def test_a_missing_column_is_named_and_fails_the_report(self, tmp_path):
        path = tmp_path / "old.db"
        ChatDb(path)
        with sqlite3.connect(path) as conn:
            conn.execute("ALTER TABLE message DROP COLUMN thread_originator_guid")
        said: list[str] = []
        assert live_check.report_database(str(path), said.append) is False
        assert any("thread_originator_guid" in line for line in said)

    def test_a_mac_before_send_later_is_noted_not_failed(self, tmp_path):
        ChatDb(tmp_path / "older.db", send_later_columns=False)
        said: list[str] = []
        assert live_check.report_database(str(tmp_path / "older.db"), said.append) is True
        assert any("schedule_type" in line for line in said)

    def test_a_database_that_cannot_be_read_fails_the_report(self, tmp_path):
        assert live_check.report_database(str(tmp_path / "no-such.db"), [].append) is False

    def test_it_counts_what_the_columns_are_assumed_to_hold(self, run):
        live_check.report_database(run.host.database.path, run.said.append)
        counted = {line.split(None, 1)[1]: int(line.split()[0]) for line in run.said if line.strip()[:1].isdigit()}
        assert counted["messages"] == 3
        assert counted["1:1 chats (chat.style 45)"] == 2 and counted["group chats (chat.style 43)"] == 1
        assert counted["replies in a thread (thread_originator_guid set)"] == 1
        assert counted["... whose thread_originator_guid is the guid of a message that is there"] == 1
