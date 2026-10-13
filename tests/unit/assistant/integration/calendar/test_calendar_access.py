"""CalendarAccess: the accounts of every enabled and configured CalendarSource as one calendar —
the week's header and events, loading named calendars, creating an event, one account failing,
and no account at all."""
from datetime import date, datetime, timedelta
from typing import List
from unittest.mock import MagicMock, patch

import pytest
import pytz

from assistant.integration.Integration import Integration, IntegrationState
from assistant.integration.IntegrationMonitor import IntegrationMonitor
from assistant.integration.calendar.CalendarAccess import CalendarAccess, EVENTS_UNAVAILABLE, NOT_CONFIGURED, NO_ACCOUNT_HINT
from assistant.integration.calendar.CalendarAccount import CalendarAccount, CalendarInfo, split_names
from assistant.integration.calendar.CalendarSource import CalendarSource
from assistant.integration.calendar.Event import Event, NO_EVENTS

UTC = pytz.UTC
WEEK = (date(2025, 6, 15), date(2025, 6, 20))


@pytest.fixture(autouse=True)
def stockholm():
    config = MagicMock()
    config.get_home_timezone.return_value = "Europe/Stockholm"
    with patch("assistant.integration.calendar.Event.config", new=config), \
            patch("assistant.integration.calendar.CalendarAccess.config", new=config):
        yield


class _Account(CalendarAccount):
    """A calendar account over dicts: `calendars` maps a name to its events (None for a
    Reminders list), `aliases` maps the names the account takes besides the listed ones to the
    listed one they mean (the way Google takes "primary"), `created` records what create_event
    was handed."""

    service_name = "dicts"

    def __init__(self, name="iCloud", calendars=None, shown=(), default="Home", failing=False, aliases=(), refuses=""):
        self._name = name
        self._calendars = calendars or {}
        self._shown = list(shown)
        self._default = default
        self._failing = failing
        self._aliases = dict(aliases)
        self.created: List[tuple] = []
        self.deleted: List[tuple] = []
        self.moved: List[tuple] = []
        self.asked: List[list] = []
        self.refuses = refuses

    @property
    def name(self) -> str:
        return self._name

    @property
    def service(self) -> str:
        return self.service_name

    @property
    def shown_calendars(self) -> list[str]:
        return self._shown

    @property
    def default_calendar(self) -> str:
        return self._default

    def calendars(self) -> list[CalendarInfo]:
        if self._failing:
            raise ConnectionError("failed")
        return [CalendarInfo(name, events is not None) for name, events in self._calendars.items()]

    def listed_name(self, alias):
        return self._aliases.get(alias.lower())

    def events(self, calendar_names, start, end):
        self.asked.append(list(calendar_names))
        return [e for name in calendar_names for e in (self._calendars.get(name) or [])]

    def create_event(self, calendar_name, event):
        if calendar_name not in self._calendars and calendar_name not in self._aliases:
            raise ValueError(f"Calendar '{calendar_name}' not found. Available: {list(self._calendars)}")
        self.created.append((calendar_name, event))

    def delete_event(self, calendar_name, event):
        if self.refuses:
            raise ValueError(self.refuses)
        self.deleted.append((calendar_name, event))

    def move_event(self, calendar_name, event, start, end, all_day):
        if self.refuses:
            raise ValueError(self.refuses)
        self.moved.append((calendar_name, event, start, end, all_day))

    def check(self):
        return True


def _event(title, day, hour, calendar="Home", id=""):
    return Event(datetime(day.year, day.month, day.day, hour, 0, tzinfo=UTC),
                 datetime(day.year, day.month, day.day, hour + 1, 0, tzinfo=UTC), title, calendar=calendar,
                 id=id or title.lower().replace(" ", "-"))


def _household(**kwargs):
    day = date(2025, 6, 15)
    return _Account(calendars={"Home": [_event("Dentist Charlotte", day, 9)],
                               "Family": [_event("Dinner at Ingeborg's", day, 17, "Family")],
                               "Work": [_event("Budget review", day, 13, "Work")],
                               "Shopping": None}, **kwargs)


@pytest.mark.unit
class TestFetch:
    def test_only_the_shown_calendars_have_their_events_listed(self):
        text = CalendarAccess([_household(shown=["Home", "family"])]).fetch(*WEEK)
        assert "Dentist Charlotte" in text and "Dinner at Ingeborg's" in text
        assert "Budget review" not in text

    def test_the_header_names_the_account_the_calendars_shown_and_the_ones_left_out(self):
        header = CalendarAccess([_household(shown=["Home", "Family"])]).fetch(*WEEK).split("\n\n")[0]
        shown, left_out = header.split(" — ")
        assert shown.startswith("iCloud (dicts): ") and "Home" in shown and "Family" in shown and "Work" not in shown
        assert left_out.startswith("not shown") and "Work" in left_out
        assert "Home" not in left_out and "Family" not in left_out

    def test_an_account_without_a_service_is_headed_by_its_name_alone_and_nothing_hidden_adds_nothing(self):
        account = _Account("Plain", {"Home": [_event("Dentist Charlotte", date(2025, 6, 15), 9)]})
        account.service_name = ""
        header = CalendarAccess([account]).fetch(*WEEK).split("\n\n")[0]
        assert header == "Plain: Home"

    def test_a_reminders_list_is_searched_but_never_named(self):
        account = _household(shown=["Home"])
        text = CalendarAccess([account]).fetch(*WEEK)
        assert "Shopping" not in text
        account_without_shown = _household()
        CalendarAccess([account_without_shown]).fetch(*WEEK)
        assert "Shopping" in account_without_shown.asked[0] and "Shopping" not in CalendarAccess([account_without_shown]).fetch(*WEEK)

    def test_without_the_setting_every_calendar_is_shown_and_none_is_listed(self):
        text = CalendarAccess([_household()]).fetch(*WEEK)
        assert "Dentist Charlotte" in text and "Budget review" in text
        assert "load_calendar" not in text

    def test_a_shown_name_the_account_resolves_as_an_alias_shows_that_calendar(self):
        """The documented `shown_calendars: [primary]` of a Google account: "primary" is never a
        listed name, the account says which one it means."""
        day = date(2025, 6, 15)
        account = _Account("Google", {"marcus@example.com": [_event("Standup", day, 9, "marcus@example.com")],
                                      "Family": [_event("Dinner", day, 18, "Family")]},
                           shown=["Primary", "marcus@example.com"], aliases={"primary": "marcus@example.com"})
        text = CalendarAccess([account]).fetch(*WEEK)
        assert "Standup" in text and "Dinner" not in text
        assert account.asked == [["marcus@example.com"]], "the alias is resolved once and asked by its listed name"
        assert text.startswith("Google (dicts): marcus@example.com — not shown: Family")

    def test_a_shown_name_the_account_lacks_shows_nothing_but_names_the_rest(self):
        text = CalendarAccess([_household(shown=["Nope"])]).fetch(*WEEK)
        assert text.startswith("iCloud (dicts): none shown — not shown: ")
        assert "Work" in text and text.endswith(NO_EVENTS)

    def test_lines_carry_their_calendar_only_when_more_than_one_is_shown(self):
        one = CalendarAccess([_household(shown=["Home"])]).fetch(*WEEK)
        assert "2025-06-15 11:00 - 12:00: Dentist Charlotte" in one and "[Home]" not in one
        two = CalendarAccess([_household(shown=["Home", "Work"])]).fetch(*WEEK)
        assert "2025-06-15 11:00 - 12:00 [Home]: Dentist Charlotte" in two and "[Work]: Budget review" in two

    def test_one_calendar_in_each_of_two_accounts_still_tags_the_lines(self):
        day = date(2025, 6, 16)
        first = _Account("Family", {"Home": [_event("Late", day, 15)]})
        second = _Account("Google", {"Work": [_event("Early", day, 8, "Work")]})
        text = CalendarAccess([first, second]).fetch(*WEEK)
        assert "[Work]: Early" in text and "[Home]: Late" in text

    def test_events_of_several_accounts_are_one_list_in_time_order(self):
        day = date(2025, 6, 16)
        first = _Account("Family", {"Home": [_event("Late", day, 15)]})
        second = _Account("Google", {"Work": [_event("Early", day, 8, "Work")]})

        text = CalendarAccess([first, second]).fetch(*WEEK)

        assert text.index("Early") < text.index("Late")
        header = text.split("\n\n")[0].split("\n")
        assert header[0].startswith("Family") and header[1].startswith("Google")

    def test_one_failing_account_is_named_while_the_others_still_show(self):
        broken = _Account("Broken", failing=True)
        text = CalendarAccess([_household(), broken]).fetch(*WEEK)
        assert "(Broken: calendar could not be loaded)" in text
        assert "Dentist Charlotte" in text

    def test_every_account_failing_is_unavailable_not_an_empty_week(self):
        text = CalendarAccess([_Account(failing=True)]).fetch(*WEEK)
        assert text == EVENTS_UNAVAILABLE and text != NO_EVENTS

    def test_no_account_is_not_configured(self):
        assert CalendarAccess([]).fetch(*WEEK) == NOT_CONFIGURED

    def test_the_coming_week_starts_today_and_covers_eight_days(self):
        account = _Account()
        with patch.object(CalendarAccess, "fetch", return_value="") as fetch:
            CalendarAccess([account]).coming_week()
        start, end = fetch.call_args.args
        assert start == date.today() and (end - start).days == 8


@pytest.mark.unit
class TestLoad:
    def test_loads_only_the_named_calendar_whatever_its_case(self):
        tomorrow = date.today() + timedelta(days=1)
        account = _Account(calendars={"Home": [_event("Dentist Charlotte", tomorrow, 9)],
                                      "Work": [_event("Budget review", tomorrow, 13, "Work")]})

        text = CalendarAccess([account]).load(["work"])

        assert "Budget review" in text and "Dentist Charlotte" not in text
        assert text.startswith("Calendar 'Work', ")

    def test_an_alias_the_account_resolves_loads_its_calendar(self):
        tomorrow = date.today() + timedelta(days=1)
        account = _Account("Google", {"marcus@example.com": [_event("Standup", tomorrow, 9, "marcus@example.com")]},
                           aliases={"primary": "marcus@example.com"})
        text = CalendarAccess([account]).load(["primary"])
        assert "Standup" in text and account.asked == [["marcus@example.com"]]

    def test_an_unknown_name_fails_and_says_which_calendars_exist(self):
        tomorrow = date.today() + timedelta(days=1)
        account = _Account(calendars={"Home": [], "Work": [_event("Budget review", tomorrow, 13, "Work")], "Shopping": None})

        text = CalendarAccess([account]).load(["School", "'Work'"])

        assert "failed" in text and "Home" in text and "Shopping" not in text.split("\n\n")[0]
        assert "Budget review" in text

    def test_a_calendar_of_a_second_account_is_found_too_and_headed_with_its_account(self):
        tomorrow = date.today() + timedelta(days=1)
        second = _Account("Google", {"Kids": [_event("Football", tomorrow, 16, "Kids")]})

        text = CalendarAccess([_Account(), second]).load(["kids"])

        assert "Football" in text and text.startswith("Calendar 'Google/Kids'")

    def test_a_name_two_accounts_share_must_be_qualified_with_the_account(self):
        tomorrow = date.today() + timedelta(days=1)
        family = _Account("Family", {"Kids": [_event("Football", tomorrow, 16, "Kids")]})
        google = _Account("Google", {"Kids": [_event("Piano", tomorrow, 17, "Kids")]})
        access = CalendarAccess([family, google])

        bare = access.load(["Kids"])
        assert "failed" in bare and "Family/Kids" in bare and "Google/Kids" in bare and "Football" not in bare

        qualified = access.load(["google/kids"])
        assert "Piano" in qualified and "Football" not in qualified

    def test_an_unknown_name_across_accounts_lists_the_calendars_by_account(self):
        tomorrow = date.today() + timedelta(days=1)
        access = CalendarAccess([_Account("Family", {"Home": []}), _Account("Google", {"Work": [_event("x", tomorrow, 9)]})])
        text = access.load(["School"])
        assert "Family/Home" in text and "Google/Work" in text

    def test_without_an_account_loading_raises_the_hint(self):
        with pytest.raises(ValueError, match="list_integrations"):
            CalendarAccess([]).load(["Home"])


@pytest.mark.unit
class TestCreate:
    def test_the_event_lands_in_the_first_accounts_default_calendar_in_local_time(self):
        account = _household(default="Family")

        CalendarAccess([account]).create("Dentist", "2025-07-15", "10:00", "11:00")

        (calendar_name, event), = account.created
        assert calendar_name == "Family" and event.title == "Dentist"
        assert event.start.utcoffset() == timedelta(hours=2)   # CEST in July, not CET
        assert (event.start.hour, event.end.hour) == (10, 11)

    def test_a_missing_default_calendar_raises_and_names_the_available_ones(self):
        with pytest.raises(ValueError, match="Calendar 'Nope' not found") as raised:
            CalendarAccess([_household(default="Nope")]).create("Dentist", "2025-10-28", "10:00", "11:00")
        assert "Home" in str(raised.value)

    def test_the_default_calendar_is_matched_to_the_listed_name_whatever_its_case(self):
        account = _household(default="family")
        CalendarAccess([account]).create("Dentist", "2025-10-28", "10:00", "11:00")
        (calendar_name, _), = account.created
        assert calendar_name == "Family"

    def test_a_default_the_account_resolves_itself_is_handed_over_unchanged(self):
        """Google's default is "primary", an alias the account understands but never lists: the
        composite must not refuse it on the listed names alone."""
        account = _Account("Google", {"marcus@example.com": []}, default="primary", aliases={"primary": "marcus@example.com"})
        where = CalendarAccess([account]).create("Dentist", "2025-10-28", "10:00", "11:00")
        (calendar_name, _), = account.created
        assert calendar_name == "primary" and where == "Google/primary"

    def test_without_a_default_calendar_the_accounts_first_event_calendar_takes_it(self):
        account = _household(default="")
        CalendarAccess([account]).create("Dentist", "2025-10-28", "10:00", "11:00")
        (calendar_name, _), = account.created
        assert calendar_name == "Home"

    def test_the_default_is_the_first_accounts_even_when_a_later_one_has_its_own(self):
        first, second = _household(default="Family"), _Account("Google", {"Work": []}, default="Work")
        CalendarAccess([first, second]).create("Dentist", "2025-10-28", "10:00", "11:00")
        assert [c for c, _ in first.created] == ["Family"] and second.created == []

    def test_a_named_calendar_of_any_account_takes_the_event(self):
        first, second = _household(), _Account("Google", {"Kids": []})
        where = CalendarAccess([first, second]).create("Football", "2025-10-28", "16:00", "17:00", "kids")
        (calendar_name, event), = second.created
        assert calendar_name == "Kids" and event.account == "Google" and where == "Google/Kids"
        assert first.created == []

    def test_a_shared_name_needs_the_account_and_the_qualified_form_resolves_it(self):
        family, google = _Account("Family", {"Kids": []}), _Account("Google", {"Kids": []})
        access = CalendarAccess([family, google])
        with pytest.raises(ValueError, match="Family/Kids"):
            access.create("Football", "2025-10-28", "16:00", "17:00", "Kids")
        access.create("Football", "2025-10-28", "16:00", "17:00", "Google/Kids")
        assert len(google.created) == 1 and family.created == []

    def test_an_unknown_calendar_name_raises_with_the_available_ones(self):
        with pytest.raises(ValueError, match="no calendar named 'Nope'") as raised:
            CalendarAccess([_household()]).create("Dentist", "2025-10-28", "10:00", "11:00", "Nope")
        assert "Home" in str(raised.value)

    def test_without_an_account_creating_raises_the_hint(self):
        with pytest.raises(ValueError, match=NO_ACCOUNT_HINT[:20]):
            CalendarAccess([]).create("Dentist", "2025-10-28", "10:00", "11:00")


def _day_account(**kwargs):
    """A household with two events on the 15th, one on the 16th, and an all-day one on the 15th
    and 16th; the shown calendars are Home and Family, Work is hidden."""
    day, next_day = date(2025, 6, 15), date(2025, 6, 16)
    sports = Event(datetime(2025, 6, 15, tzinfo=UTC), datetime(2025, 6, 17, tzinfo=UTC), "Sports camp", calendar="Family",
                   id="camp", all_day=True)
    return _Account(calendars={"Home": [_event("Dentist Charlotte", day, 9), _event("Dentist Frances", day, 14),
                                        _event("Dentist Charlotte", next_day, 9, id="dentist-2")],
                               "Family": [sports, _event("Dinner at Ingeborg's", day, 17, "Family")],
                               "Work": [_event("Budget review", day, 13, "Work")]},
                    shown=("Home", "Family"), **kwargs)


@pytest.mark.unit
class TestDelete:
    def test_the_event_titled_so_on_that_day_is_deleted_from_its_own_account_and_calendar(self):
        account = _day_account()
        other = _Account("Google", {"Kids": [_event("Dentist Bob", date(2025, 6, 15), 14, "Kids", id="g-1")]})

        said = CalendarAccess([other, account]).delete("dentist frances", "2025-06-15")

        (calendar_name, event), = account.deleted
        assert calendar_name == "Home" and event.id == "dentist-frances" and other.deleted == []
        assert "Dentist Frances" in said and "iCloud/Home" in said and "16:00" in said   # CEST

    def test_the_search_reads_only_the_shown_calendars_with_a_day_of_margin(self):
        account = _day_account()
        with pytest.raises(ValueError, match="no event titled 'Budget review'"):
            CalendarAccess([account]).delete("Budget review", "2025-06-15")
        asked, = account.asked
        assert asked == ["Home", "Family"]

    def test_a_named_calendar_finds_the_event_in_a_hidden_one(self):
        account = _day_account()
        CalendarAccess([account]).delete("Budget review", "2025-06-15", "work")
        (calendar_name, event), = account.deleted
        assert calendar_name == "Work" and event.title == "Budget review" and account.asked == [["Work"]]

    def test_an_unknown_calendar_name_raises_before_any_search(self):
        account = _day_account()
        with pytest.raises(ValueError, match="no calendar named 'Nope'"):
            CalendarAccess([account]).delete("Dentist Charlotte", "2025-06-15", "Nope")
        assert account.asked == []

    def test_the_date_is_the_home_timezone_one_so_the_same_title_on_another_day_is_not_it(self):
        account = _day_account()
        CalendarAccess([account]).delete("Dentist Charlotte", "2025-06-16")
        (_, event), = account.deleted
        assert event.id == "dentist-2"

    def test_a_part_of_the_title_is_enough_when_it_singles_out_one_event(self):
        account = _day_account()
        CalendarAccess([account]).delete("Ingeborg", "2025-06-15")
        (_, event), = account.deleted
        assert event.title == "Dinner at Ingeborg's"

    def test_a_part_matching_several_titles_is_refused_naming_them_with_their_times(self):
        account = _day_account()
        with pytest.raises(ValueError, match="several") as raised:
            CalendarAccess([account]).delete("Dentist", "2025-06-15")
        message = str(raised.value)
        assert "Dentist Charlotte" in message and "Dentist Frances" in message and "11:00" in message and "16:00" in message
        assert account.deleted == []

    def test_a_whole_title_wins_over_the_longer_titles_that_contain_it(self):
        """'Dentist' is also a part of 'Dentist Frances': the exact match is the one meant, not an
        ambiguity — and the quotes an LLM wraps the title in are not part of it."""
        day = date(2025, 6, 15)
        account = _Account(calendars={"Home": [_event("Dentist", day, 9), _event("Dentist Frances", day, 14)]},
                           shown=("Home",))
        CalendarAccess([account]).delete("'dentist'", "2025-06-15")
        (_, event), = account.deleted
        assert event.id == "dentist"

    def test_an_account_that_only_creates_refuses_a_delete_and_a_move_naming_itself(self):
        class _CreateOnly(_Account):
            delete_event = CalendarAccount.delete_event
            move_event = CalendarAccount.move_event

        account = _CreateOnly(calendars={"Home": [_event("Dentist Frances", date(2025, 6, 15), 14)]}, shown=("Home",))
        access = CalendarAccess([account])
        with pytest.raises(ValueError, match="not supported") as raised:
            access.delete("Dentist Frances", "2025-06-15")
        assert account.name in str(raised.value) and account.service in str(raised.value)
        with pytest.raises(ValueError, match="not supported"):
            access.move("Dentist Frances", "2025-06-15", "2025-06-16")
        assert account.deleted == [] and account.moved == []

    def test_an_all_day_event_is_found_on_each_of_its_days(self):
        account = _day_account()
        CalendarAccess([account]).delete("Sports camp", "2025-06-16")
        (calendar_name, event), = account.deleted
        assert calendar_name == "Family" and event.all_day

    def test_no_match_names_what_the_day_holds(self):
        account = _day_account()
        with pytest.raises(ValueError, match="no event titled 'Piano'") as raised:
            CalendarAccess([account]).delete("Piano", "2025-06-15")
        assert "Dentist Charlotte" in str(raised.value) and "Sports camp" in str(raised.value)

    def test_the_accounts_refusal_is_the_error(self):
        account = _day_account(refuses="recurring event")
        with pytest.raises(ValueError, match="recurring event"):
            CalendarAccess([account]).delete("Dentist Frances", "2025-06-15")

    def test_a_bad_date_or_no_account_raises(self):
        with pytest.raises(ValueError):
            CalendarAccess([_day_account()]).delete("Dentist Frances", "tomorrow")
        with pytest.raises(ValueError, match=NO_ACCOUNT_HINT[:20]):
            CalendarAccess([]).delete("Dentist Frances", "2025-06-15")


@pytest.mark.unit
class TestMove:
    def test_with_times_the_event_gets_them_on_the_new_date_in_local_time(self):
        account = _day_account()

        said = CalendarAccess([account]).move("Dentist Frances", "2025-06-15", "2025-06-18", "10:00", "11:30")

        (calendar_name, event, start, end, all_day), = account.moved
        assert calendar_name == "Home" and event.id == "dentist-frances" and not all_day
        assert start == datetime(2025, 6, 18, 8, 0, tzinfo=UTC) and end == datetime(2025, 6, 18, 9, 30, tzinfo=UTC)
        assert "2025-06-15 16:00 - 17:00" in said and "2025-06-18 10:00 - 11:30" in said

    def test_without_times_the_event_keeps_its_time_of_day_and_length(self):
        account = _day_account()
        CalendarAccess([account]).move("Dentist Frances", "2025-06-15", "2025-06-18")
        (_, _, start, end, all_day), = account.moved
        assert (start, end, all_day) == (datetime(2025, 6, 18, 14, 0, tzinfo=UTC), datetime(2025, 6, 18, 15, 0, tzinfo=UTC), False)

    def test_without_times_the_wall_clock_survives_a_change_of_daylight_saving(self):
        account = _Account(calendars={"Home": [_event("Lunch", date(2025, 10, 20), 10)]})   # 12:00 CEST
        CalendarAccess([account]).move("Lunch", "2025-10-20", "2025-11-03")               # CET by then
        (_, _, start, _, _), = account.moved
        assert start == datetime(2025, 11, 3, 11, 0, tzinfo=UTC)

    def test_an_all_day_event_without_times_stays_all_day_and_keeps_its_length(self):
        account = _day_account()
        CalendarAccess([account]).move("Sports camp", "2025-06-15", "2025-06-20")
        (_, event, start, end, all_day), = account.moved
        assert event.id == "camp" and all_day
        assert (start, end) == (datetime(2025, 6, 20, tzinfo=UTC), datetime(2025, 6, 22, tzinfo=UTC))

    def test_an_all_day_event_given_times_becomes_a_timed_one(self):
        account = _day_account()
        CalendarAccess([account]).move("Sports camp", "2025-06-15", "2025-06-20", "09:00", "12:00")
        (_, _, start, _, all_day), = account.moved
        assert not all_day and start == datetime(2025, 6, 20, 7, 0, tzinfo=UTC)

    def test_the_same_date_with_new_times_changes_only_the_time(self):
        account = _day_account()
        said = CalendarAccess([account]).move("Dentist Frances", "2025-06-15", "2025-06-15", "08:00", "09:00")
        (_, _, start, _, _), = account.moved
        assert start == datetime(2025, 6, 15, 6, 0, tzinfo=UTC) and "2025-06-15 08:00 - 09:00" in said

    def test_one_time_alone_a_bad_new_date_or_an_ambiguous_title_raises_without_moving(self):
        account = _day_account()
        access = CalendarAccess([account])
        with pytest.raises(ValueError, match="both"):
            access.move("Dentist Frances", "2025-06-15", "2025-06-18", "10:00", "")
        with pytest.raises(ValueError):
            access.move("Dentist Frances", "2025-06-15", "next week")
        with pytest.raises(ValueError, match="several"):
            access.move("Dentist", "2025-06-15", "2025-06-18")
        assert account.moved == []

    def test_a_named_calendar_of_another_account_is_where_the_event_is_looked_for(self):
        first, second = _day_account(), _Account("Google", {"Kids": [_event("Football", date(2025, 6, 15), 16, "Kids", id="g-1")]})
        said = CalendarAccess([first, second]).move("Football", "2025-06-15", "2025-06-16", calendar_name="Google/Kids")
        (calendar_name, event, _, _, _), = second.moved
        assert calendar_name == "Kids" and event.id == "g-1" and first.moved == [] and "Google/Kids" in said


@pytest.mark.unit
class TestSplitNames:
    def test_a_list_a_string_and_a_comma_joined_item_all_name_each_calendar(self):
        assert split_names(["Home", "Family"]) == ["Home", "Family"]
        assert split_names("Home, Work") == ["Home", "Work"]
        assert split_names(["Home, Family", "Work"]) == ["Home", "Family", "Work"]

    def test_anything_else_is_no_names(self):
        assert split_names(None) == [] and split_names(42) == [] and split_names([]) == []


class _Source(Integration, CalendarSource):
    def __init__(self, name: str, keys: List[str], accounts):
        super().__init__()
        self._name, self._keys, self._accounts = name, keys, accounts

    @property
    def name(self) -> str:
        return self._name

    @property
    def config_keys(self) -> List[str]:
        return self._keys

    def check_health(self) -> bool:
        return True

    def calendar_accounts(self):
        return self._accounts


@pytest.mark.unit
class TestGatheringThroughTheMonitor:
    @pytest.fixture(autouse=True)
    def monitor(self):
        IntegrationMonitor.reset_instance()
        yield IntegrationMonitor.get_instance()
        IntegrationMonitor.reset_instance()

    def test_configured_sources_contribute_and_unconfigured_ones_do_not(self, monitor):
        icloud, google = _Account("iCloud"), _Account("Google")
        unconfigured = _Source("Google", ["fictional_calendar_section_nobody_has"], [google])
        monitor.register(_Source("CalDAV", [], [icloud]))
        monitor.register(unconfigured)

        assert unconfigured.state is IntegrationState.NOT_CONFIGURED
        assert CalendarAccess.current().accounts == [icloud]

    def test_nothing_registered_is_no_account(self, monitor):
        assert CalendarAccess.current().accounts == []
        assert CalendarAccess.current().coming_week() == NOT_CONFIGURED

    def test_no_calendar_integration_registered_yet_means_still_starting(self, monitor):
        """The prompt sections start before the launcher registers the integrations: an empty
        monitor is "starting", a disabled calendar integration is "not configured"."""
        from assistant.integration.IntegrationMonitor import DisabledIntegration
        assert CalendarAccess.current().starting is True

        monitor.register(DisabledIntegration(_Source))
        access = CalendarAccess.current()
        assert access.starting is False and access.accounts == []
