"""The messages to an owner who has gone quiet: when a step of the schedule is due, what
spends it, what starts it over, and what is reported of it."""
from datetime import datetime
from unittest.mock import Mock

import pytest

import common.ConfigManager as config_module
from assistant import Destinations, Reactivation
from assistant.Notifier import set_notifier
from assistant.ChannelProfile import SOURCE_AUTONOMOUS
from assistant.agents.AutonomousActionManager import PERIODIC_CHECK_INTERVAL_SECONDS
from common.FloxyDatabase import FloxyDatabase

HOUR = 3600
DAY = 24 * HOUR
# The owner's last turn: a Monday evening, 20:00.
LAST_TURN = datetime(2026, 3, 2, 20, 0).timestamp()


@pytest.fixture(autouse=True)
def history(tmp_path, monkeypatch):
    """A history of the test's own: the worker's shared database holds whatever rows the tests
    before this one left, and a clock would be read out of them."""
    monkeypatch.setattr(FloxyDatabase, "DB_PATH", str(tmp_path / "floxy.db"))
    db = FloxyDatabase()
    monkeypatch.setattr(FloxyDatabase, "get_instance", classmethod(lambda cls: db))
    return db


@pytest.fixture
def settings(monkeypatch):
    values = {Reactivation.ENABLED_KEY: True, "owner_name": "Marcus"}
    monkeypatch.setattr(config_module.config, "get", lambda key, default=None: values.get(key, default))
    return values


@pytest.fixture
def telemetry(monkeypatch):
    sink = Mock()
    monkeypatch.setattr(Reactivation.Telemetry, "get_instance", lambda: sink)
    return sink


@pytest.fixture(autouse=True)
def chats():
    """What `owner` reaches: Telegram on a phone, and a console nobody has open."""
    hub = Mock()
    hub.owner_channels.return_value = [("Telegram", True), ("Console", False)]
    set_notifier(hub)
    Destinations.clear()
    Destinations.register(Destinations.Destination("owner", Destinations.KIND_CHAT, lambda text: True))
    Destinations.register(Destinations.Destination("kitchen", Destinations.KIND_VOICE, lambda text: True))
    yield hub
    set_notifier(None)
    Destinations.clear()


@pytest.fixture
def quiet_owner(settings, telemetry):
    """An owner whose last turn was at LAST_TURN, with the messages switched on."""
    Reactivation.owner_spoke("Telegram", now=LAST_TURN)
    return telemetry


def spend(step, at, posted=True):
    return Reactivation.record(step, ["post_message(owner, Hello)"] if posted else [], now=at)


class TestWhenAStepIsDue:
    def test_nothing_is_due_on_the_day_the_owner_wrote(self, quiet_owner):
        assert Reactivation.due(LAST_TURN + 22 * HOUR) is None

    def test_the_first_message_is_due_an_hour_before_the_owners_time_of_day_on_the_next_day(self, quiet_owner):
        step = Reactivation.due(LAST_TURN + 23 * HOUR)

        assert (step.number, step.kind) == (1, Reactivation.KIND_DAILY)

    def test_a_check_that_notices_a_step_a_whole_interval_late_still_writes_before_the_owners_time(self):
        # the periodic check notices a step up to its own interval after it fell due
        for number in range(1, Reactivation.DAILY_STEPS + 1):
            noticed_at_the_latest = Reactivation.due_at(LAST_TURN, number) + PERIODIC_CHECK_INTERVAL_SECONDS
            assert noticed_at_the_latest <= LAST_TURN + number * DAY

    def test_one_message_a_day_for_a_week_and_the_seventh_is_the_back_off(self, quiet_owner):
        kinds = []
        for day in range(1, 8):
            at = LAST_TURN + day * DAY
            step = Reactivation.due(at)
            kinds.append((step.number, step.kind))
            spend(step, at)

        assert kinds == [(n, Reactivation.KIND_DAILY) for n in range(1, 7)] + [(7, Reactivation.KIND_BACKOFF)]

    def test_after_the_week_a_message_comes_every_fourteen_days(self, quiet_owner):
        spend(Reactivation.due(LAST_TURN + 7 * DAY), LAST_TURN + 7 * DAY)

        assert Reactivation.due(LAST_TURN + 8 * DAY) is None
        assert Reactivation.due(LAST_TURN + 20 * DAY) is None
        first = Reactivation.due(LAST_TURN + 21 * DAY)
        spend(first, LAST_TURN + 21 * DAY)
        assert Reactivation.due(LAST_TURN + 34 * DAY) is None
        second = Reactivation.due(LAST_TURN + 35 * DAY)
        assert [(s.number, s.kind) for s in (first, second)] == [(8, Reactivation.KIND_FORTNIGHTLY),
                                                                 (9, Reactivation.KIND_FORTNIGHTLY)]

    def test_a_step_that_was_spent_is_not_asked_again_until_the_next_one(self, quiet_owner):
        at = LAST_TURN + DAY
        spend(Reactivation.due(at), at)

        assert Reactivation.due(at + HOUR) is None
        assert Reactivation.due(at + DAY).number == 2

    def test_a_declined_step_is_spent_like_a_sent_one(self, quiet_owner):
        at = LAST_TURN + DAY

        outcome = spend(Reactivation.due(at), at, posted=False)

        assert outcome == Reactivation.OUTCOME_DECLINED
        assert Reactivation.due(at + HOUR) is None

    def test_the_steps_a_server_that_was_down_missed_are_one_message_the_newest(self, quiet_owner):
        at = LAST_TURN + 4 * DAY          # nothing went out on days 1 to 3

        step = Reactivation.due(at)
        spend(step, at)

        assert step.number == 4
        assert Reactivation.due(at + 60) is None, "the three before it are not sent after it"

    def test_the_back_off_comes_on_the_seventh_day_however_few_went_out_before_it(self, quiet_owner):
        spend(Reactivation.due(LAST_TURN + 2 * DAY), LAST_TURN + 2 * DAY)      # the only one of the six

        assert Reactivation.due(LAST_TURN + 9 * DAY).kind == Reactivation.KIND_BACKOFF

    def test_the_seventh_calls_nothing_off_for_an_owner_who_was_never_written_to(self, quiet_owner):
        # the key switched on a week into the silence, or a server that was down for it
        step = Reactivation.due(LAST_TURN + 9 * DAY)

        assert (step.number, step.kind) == (Reactivation.DAILY_STEPS, Reactivation.KIND_DAILY)
        assert Reactivation.BACKOFF not in Reactivation.note(step)

    def test_steps_the_model_declined_are_no_messages_to_call_off(self, quiet_owner):
        for day in range(1, Reactivation.DAILY_STEPS):
            spend(Reactivation.due(LAST_TURN + day * DAY), LAST_TURN + day * DAY, posted=False)

        assert Reactivation.due(LAST_TURN + Reactivation.DAILY_STEPS * DAY).kind == Reactivation.KIND_DAILY

    def test_the_message_after_an_unannounced_seventh_is_still_fourteen_days_on(self, quiet_owner):
        at = LAST_TURN + Reactivation.DAILY_STEPS * DAY
        spend(Reactivation.due(at), at)

        assert Reactivation.due(at + 13 * DAY) is None
        assert Reactivation.due(at + 14 * DAY).kind == Reactivation.KIND_FORTNIGHTLY

    def test_a_step_waits_while_the_one_before_it_was_spent_within_the_gap(self, quiet_owner):
        late = LAST_TURN + DAY + 20 * HOUR        # step 1 went out 20 hours late, after a long night
        spend(Reactivation.due(late, after_night=True), late)
        step_two_due = Reactivation.due_at(LAST_TURN, 2)

        assert step_two_due - late < Reactivation.MIN_GAP_SECONDS
        assert Reactivation.due(step_two_due + 60) is None
        assert Reactivation.due(late + Reactivation.MIN_GAP_SECONDS, after_night=True).number == 2

    def test_a_step_found_hours_past_its_hour_waits_for_that_hour_of_the_next_day(self, quiet_owner):
        restart = LAST_TURN + DAY + 7 * HOUR      # 03:00, the nightly update: step 1 was due at 19:00

        assert Reactivation.due(restart) is None
        assert Reactivation.due(LAST_TURN + 2 * DAY - 2 * HOUR) is None, "still the night after"
        assert Reactivation.due(Reactivation.due_at(LAST_TURN, 2) + 60).number == 2

    def test_a_step_is_still_sent_a_postponed_check_after_its_hour(self, quiet_owner):
        at = Reactivation.due_at(LAST_TURN, 1) + PERIODIC_CHECK_INTERVAL_SECONDS + 20 * 60

        assert Reactivation.due(at).number == 1

    def test_the_last_daily_step_missed_on_its_day_goes_out_at_its_hour_of_a_later_day(self, quiet_owner):
        back_up = LAST_TURN + 9 * DAY + 5 * HOUR   # down since day 6, up again at 01:00 on day 9

        assert Reactivation.due(back_up) is None
        assert Reactivation.due(LAST_TURN + 10 * DAY - HOUR + 60).number == 7

    def test_the_step_the_night_held_back_goes_out_with_the_morning(self, quiet_owner):
        morning = LAST_TURN + DAY + 11 * HOUR     # 07:00: Night mode was on from before 19:00

        assert Reactivation.due(morning) is None
        assert Reactivation.due(morning, after_night=True).number == 1

    def test_nothing_is_due_while_the_key_is_off(self, quiet_owner, settings):
        settings[Reactivation.ENABLED_KEY] = False

        assert Reactivation.due(LAST_TURN + 3 * DAY) is None

    def test_the_key_is_off_unless_the_config_says_otherwise(self, quiet_owner, settings):
        del settings[Reactivation.ENABLED_KEY]

        assert Reactivation.due(LAST_TURN + 3 * DAY) is None

    def test_an_install_whose_owner_has_no_turn_on_record_is_never_written_to(self, settings, telemetry):
        assert Reactivation.due(LAST_TURN + 30 * DAY) is None

    def test_an_install_without_a_clock_counts_from_the_last_thing_somebody_wrote(self, settings, telemetry, history):
        history.add_history_entry("Marcus", "what is on today?", LAST_TURN - DAY, "Telegram")
        history.add_history_entry("Marcus", "good night", LAST_TURN, "Console")
        history.add_history_entry("assistant", "Sleep well.", LAST_TURN + 1)

        assert Reactivation.due(LAST_TURN + 5 * HOUR) is None
        step = Reactivation.due(Reactivation.due_at(LAST_TURN, 3) + 60)
        assert (step.number, step.since) == (3, LAST_TURN)

    def test_what_the_server_and_the_feeds_wrote_since_is_not_the_owner_speaking(self, settings, telemetry, history):
        history.add_history_entry("Marcus", "good night", LAST_TURN, "Console")
        history.add_history_entry("system", "Charlotte: are you home?", LAST_TURN + 2 * DAY, "iMessage Feed")
        history.add_history_entry("system", "Server is up again", LAST_TURN + 2 * DAY + 1, "first_run")
        history.add_history_entry("command", "post_message: sent", LAST_TURN + 2 * DAY + 2, SOURCE_AUTONOMOUS)
        history.add_history_entry("note", "It is 20:00.", LAST_TURN + 2 * DAY + 3)
        history.add_history_entry("User", "periodic check", LAST_TURN + 2 * DAY + 4, SOURCE_AUTONOMOUS)

        assert Reactivation.due(Reactivation.due_at(LAST_TURN, 3) + 60).since == LAST_TURN

    def test_a_clock_read_from_the_history_is_kept(self, settings, telemetry, history):
        history.add_history_entry("Marcus", "good night", LAST_TURN, "Console")
        at = Reactivation.due_at(LAST_TURN, 1) + 60
        spend(Reactivation.due(at), at)
        history.add_history_entry("Alice", "is anybody there?", at + 60, "Telegram")

        assert Reactivation.due(Reactivation.due_at(LAST_TURN, 2) + 60).number == 2, "a guest's row moves no clock that exists"

    def test_no_clock_is_read_from_the_history_while_the_key_is_off(self, settings, telemetry, history):
        history.add_history_entry("Marcus", "good night", LAST_TURN, "Console")
        settings[Reactivation.ENABLED_KEY] = False

        assert Reactivation.due(Reactivation.due_at(LAST_TURN, 1) + 60) is None
        assert not Reactivation.STATE_FILE.exists()

    def test_a_state_file_that_cannot_be_read_is_no_clock(self, quiet_owner):
        Reactivation.STATE_FILE.write_text("{not json")

        assert Reactivation.due(LAST_TURN + 3 * DAY) is None


class TestTheAssistantWasNotQuietEither:
    def test_a_message_of_the_assistant_s_own_that_day_stands_in_for_the_step(self, quiet_owner):
        at = Reactivation.due_at(LAST_TURN, 1) + 60
        Reactivation.assistant_wrote(now=at - 5 * HOUR)

        assert Reactivation.due(at) is None
        assert quiet_owner.reactivation_nudge.call_args.kwargs["outcome"] == Reactivation.OUTCOME_COVERED
        assert quiet_owner.reactivation_nudge.call_args.kwargs["via"] == []
        Reactivation.due(at + HOUR)
        assert quiet_owner.reactivation_nudge.call_count == 1, "the step is spent, not covered again at the next check"

    def test_an_install_that_writes_every_day_never_sends_one(self, quiet_owner):
        for day in range(1, 10):
            Reactivation.assistant_wrote(now=LAST_TURN + day * DAY - 30 * 60)     # 19:30, after the step's 19:00
            assert Reactivation.due(Reactivation.due_at(LAST_TURN, 1) + day * DAY + 60) is None

    def test_a_message_from_more_than_a_day_ago_holds_nothing_back(self, quiet_owner):
        Reactivation.assistant_wrote(now=LAST_TURN + 2 * HOUR)

        assert Reactivation.due(Reactivation.due_at(LAST_TURN, 2) + 60).number == 2
        quiet_owner.reactivation_nudge.assert_not_called()

    def test_the_message_a_step_sent_does_not_stand_in_for_the_next_step(self, quiet_owner):
        at = Reactivation.due_at(LAST_TURN, 1) + 55 * 60
        step = Reactivation.due(at)
        Reactivation.assistant_wrote(now=at + 20)      # the check's own post_message
        spend(step, at + 30)

        assert Reactivation.due(Reactivation.due_at(LAST_TURN, 2) + 60).number == 2

    def test_a_covered_step_is_no_message_for_the_seventh_to_call_off(self, quiet_owner):
        at = Reactivation.due_at(LAST_TURN, 6) + 60
        Reactivation.assistant_wrote(now=at - HOUR)
        Reactivation.due(at)

        assert Reactivation.due(Reactivation.due_at(LAST_TURN, 7) + 60).kind == Reactivation.KIND_DAILY

    def test_the_owner_s_turn_does_not_forget_that_the_assistant_wrote(self, settings, telemetry):
        Reactivation.assistant_wrote(now=LAST_TURN - DAY)       # before any turn of the owner's
        Reactivation.owner_spoke("Telegram", now=LAST_TURN)
        Reactivation.assistant_wrote(now=LAST_TURN + 10.5 * HOUR)
        Reactivation.owner_spoke("Telegram", now=LAST_TURN + 11 * HOUR)     # half an hour after the message

        assert Reactivation.due(LAST_TURN + 11 * HOUR + DAY - HOUR + 60) is None
        assert telemetry.reactivation_nudge.call_args.kwargs["outcome"] == Reactivation.OUTCOME_COVERED

    def test_a_message_noted_before_there_was_a_clock_counts_once_the_history_gives_one(self, settings, telemetry, history):
        history.add_history_entry("Marcus", "good night", LAST_TURN, "Console")
        at = Reactivation.due_at(LAST_TURN, 2) + 60
        Reactivation.assistant_wrote(now=at - 3 * HOUR)

        assert Reactivation.due(at) is None
        assert Reactivation.due(Reactivation.due_at(LAST_TURN, 4) + 60).number == 4

    def test_a_failure_to_note_the_message_does_not_reach_the_command(self, quiet_owner, monkeypatch):
        monkeypatch.setattr(Reactivation, "_load", Mock(side_effect=RuntimeError("disk full")))

        Reactivation.assistant_wrote()


class TestTheOwnersTurn:
    def test_a_turn_starts_the_schedule_over_from_itself(self, quiet_owner):
        at = LAST_TURN + 2 * DAY
        spend(Reactivation.due(at), at)
        back = LAST_TURN + 3 * DAY + 5 * HOUR

        Reactivation.owner_spoke("Console", now=back)

        assert Reactivation.due(back + 22 * HOUR) is None
        assert Reactivation.due(back + DAY).number == 1

    def test_the_clock_runs_while_the_key_is_off(self, settings, telemetry):
        settings[Reactivation.ENABLED_KEY] = False
        Reactivation.owner_spoke("Telegram", now=LAST_TURN)
        settings[Reactivation.ENABLED_KEY] = True

        step = Reactivation.due(LAST_TURN + 2 * DAY)

        assert step.number == 2 and step.quiet_seconds == 2 * DAY

    def test_a_step_found_before_the_owner_spoke_does_not_spend_a_step_of_the_new_silence(self, quiet_owner):
        at = LAST_TURN + DAY
        step = Reactivation.due(at)
        Reactivation.owner_spoke("Telegram", now=at + 30)      # while the check was thinking

        spend(step, at + 60)

        assert Reactivation.due(at + 30 + DAY).number == 1

    def test_a_failure_to_note_the_turn_does_not_reach_the_request(self, settings, telemetry, monkeypatch):
        monkeypatch.setattr(Reactivation, "_save", Mock(side_effect=RuntimeError("disk full")))

        Reactivation.owner_spoke("Telegram", now=LAST_TURN)     # does not raise


class TestWhatIsReported:
    def test_a_sent_step_reports_the_kinds_of_the_destinations_and_never_their_names(self, quiet_owner):
        at = LAST_TURN + 3 * DAY
        step = Reactivation.due(at)

        outcome = Reactivation.record(step, ["load_toolkit(weather)", "post_message(kitchen, Rain at four)"], now=at)

        reported = quiet_owner.reactivation_nudge.call_args.kwargs
        assert outcome == Reactivation.OUTCOME_SENT
        assert reported["via"] == [Destinations.KIND_VOICE]
        assert (reported["step"], reported["kind"], reported["outcome"]) == (3, Reactivation.KIND_DAILY, "sent")
        assert reported["inactive_hours"] == 72
        assert reported["owner_channels"] == ["Telegram", "Console"]
        assert "kitchen" not in str(reported)

    def test_a_check_that_posted_nothing_reports_a_declined_step(self, quiet_owner):
        at = LAST_TURN + DAY

        Reactivation.record(Reactivation.due(at), ["load_toolkit(weather)"], now=at)

        reported = quiet_owner.reactivation_nudge.call_args.kwargs
        assert (reported["outcome"], reported["via"]) == (Reactivation.OUTCOME_DECLINED, [])

    def test_a_message_to_a_destination_that_is_gone_still_counts_as_sent(self, quiet_owner):
        kinds = Reactivation.posted_kinds(["post_message(the old group, Hello)"])

        assert kinds == [Reactivation.UNKNOWN_DESTINATION]

    def test_the_owners_return_reports_the_silence_and_the_messages_sent_in_it(self, quiet_owner):
        for day, posted in ((1, True), (2, False), (3, True)):
            at = LAST_TURN + day * DAY
            spend(Reactivation.due(at), at, posted=posted)
        back = LAST_TURN + 3 * DAY + 6 * HOUR

        Reactivation.owner_spoke("Telegram", now=back)

        reported = quiet_owner.reactivation_return.call_args.kwargs
        assert reported["nudges_sent"] == 2, "the declined step is no message"
        assert reported["last_step"] == 3
        assert reported["hours_since_nudge"] == 6
        assert reported["inactive_hours"] == 78
        assert reported["source"] == "Telegram"

    def test_a_return_nobody_was_written_to_for_is_reported_too(self, quiet_owner):
        Reactivation.owner_spoke("Console", now=LAST_TURN + 2 * DAY)

        reported = quiet_owner.reactivation_return.call_args.kwargs
        assert (reported["nudges_sent"], reported["last_step"], reported["hours_since_nudge"]) == (0, None, None)

    def test_a_turn_the_same_day_is_no_return(self, quiet_owner):
        Reactivation.owner_spoke("Telegram", now=LAST_TURN + 5 * HOUR)

        quiet_owner.reactivation_return.assert_not_called()

    def test_the_first_turn_on_record_is_no_return(self, settings, telemetry):
        Reactivation.owner_spoke("Telegram", now=LAST_TURN)

        telemetry.reactivation_return.assert_not_called()


class TestTheNote:
    def test_it_says_who_has_been_quiet_for_how_long_and_how_to_write_to_them(self, quiet_owner):
        note = Reactivation.note(Reactivation.due(LAST_TURN + 3 * DAY))

        assert "Marcus" in note
        assert "3 d" in note
        assert "post_message" in note

    def test_an_install_without_an_owner_name_still_says_who(self, quiet_owner, settings):
        del settings["owner_name"]

        assert "{" not in Reactivation.note(Reactivation.due(LAST_TURN + DAY))

    def test_the_seventh_says_that_the_daily_messages_stop_and_what_follows(self, quiet_owner):
        sixth = Reactivation.due(LAST_TURN + 6 * DAY)
        daily = Reactivation.note(sixth)
        spend(sixth, LAST_TURN + 6 * DAY)
        back_off = Reactivation.note(Reactivation.due(LAST_TURN + 7 * DAY))

        assert Reactivation.BACKOFF in back_off and Reactivation.BACKOFF not in daily
        assert f"every {Reactivation.FORTNIGHT_DAYS} days" in back_off

    def test_the_history_row_names_the_key_that_turns_the_messages_off(self, quiet_owner):
        # what a later chat turn has to go by when the owner says "stop writing to me"
        assert Reactivation.ENABLED_KEY in Reactivation.opening(Reactivation.due(LAST_TURN + DAY))


class TestWhatTheCheckLeftBehind:
    def test_a_step_is_spent_though_the_notifier_cannot_say_what_the_owner_s_chats_are(self, quiet_owner, chats):
        chats.owner_channels.side_effect = RuntimeError("the hub is stopping")
        at = LAST_TURN + DAY

        outcome = spend(Reactivation.due(at), at)

        assert outcome == Reactivation.OUTCOME_SENT
        assert Reactivation.due(at + HOUR) is None, "the message went out: it must not be asked for again"
        assert quiet_owner.reactivation_nudge.call_args.kwargs["owner_channels"] == []

    def test_only_the_messages_are_counted_among_the_commands_of_a_check(self, quiet_owner):
        kinds = Reactivation.posted_kinds(["load_toolkit(post_message)", "post_message(owner, Hello)",
                                           "remember(post_message(kitchen, x) worked)", "post_message(Kitchen, Tea)"])

        assert kinds == [Destinations.KIND_CHAT, Destinations.KIND_VOICE]

    @pytest.mark.xfail(strict=True, reason="posted_kinds cuts the destination at the first comma of the processor's "
                                           "record, so a group whose name holds one is reported as unknown — or as "
                                           "the kind of whatever answers to the part before the comma")
    def test_a_message_to_a_group_with_a_comma_in_its_name_is_reported_as_one_to_a_group(self, quiet_owner):
        group = "Marcus, Charlotte & Frances"
        Destinations.register(Destinations.Destination(group, Destinations.KIND_GROUP, lambda text: True))
        Destinations.register(Destinations.Destination("Marcus", Destinations.KIND_CHAT, lambda text: True))
        # the processor's record of a call: its name and the cleaned parameters joined with ", "
        executed = [f"post_message({', '.join([group, 'Dinner is at six'])})"]

        assert Reactivation.posted_kinds(executed) == [Destinations.KIND_GROUP]
