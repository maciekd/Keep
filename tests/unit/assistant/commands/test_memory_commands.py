"""The watch list and procedure commands: thin delegates to the stores, returning the reason
as the command's error when the store refuses."""
from unittest.mock import Mock, patch

import pytest

from assistant import WatchList

from assistant.Procedures import Procedures
from assistant.commands.CommandParser import CommandParser
from assistant.commands.Commands import Commands


@pytest.fixture(autouse=True)
def stores(tmp_path, monkeypatch):
    monkeypatch.setattr(WatchList, "WATCHLIST_DIR", tmp_path / "watchlist")
    monkeypatch.setattr(WatchList, "LOG_DIR", tmp_path / "log" / "watchlist")
    monkeypatch.setattr(Procedures, "DIR", tmp_path / "procedures")
    monkeypatch.setattr(Procedures, "LOG_DIR", tmp_path / "log" / "procedures")
    return tmp_path


@pytest.fixture
def commands():
    return Commands(Mock(command_reply=None, command_reply_contains_error=False))


def run(commands: Commands, text: str):
    """Parse `text` the way the dispatcher does and run it."""
    (name, params), _ = CommandParser(commands.list_commands()).extract_command_and_params(text)
    return commands.run(name, params)


@pytest.mark.unit
class TestWatchCommands:
    def test_save_update_delete_through_the_dispatcher(self, commands):
        assert not run(commands, 'save_watch("Parcel from shop.example", "order 4711, courier", "2026-03-12")')
        item, = WatchList.items()
        assert (item.name, item.note, item.due) == ("parcel-from-shop-example", "order 4711, courier", "2026-03-12")
        assert not run(commands, 'save_watch("parcel-from-shop-example", "shipped, arrives (probably) Friday")')
        item, = WatchList.items()
        assert (item.note, item.due) == ("shipped, arrives (probably) Friday", "2026-03-12")
        assert not run(commands, 'checked_watch("parcel-from-shop-example")')
        item, = WatchList.items()
        assert item.note == "shipped, arrives (probably) Friday"
        assert not run(commands, 'delete_watch("parcel-from-shop-example")')
        assert WatchList.items() == []

    def test_an_unknown_item_comes_back_as_the_commands_error(self, commands):
        assert "dentist" in run(commands, 'delete_watch("dentist")')

    def test_loading_a_watch_item_is_a_background_command(self, commands):
        commands.background_command = Mock()
        run(commands, 'load_watch("parcel", "dentist")')
        function, params = commands.background_command.call_args.args[:2]
        assert params == ["parcel", "dentist"] and callable(function)

    def test_loading_returns_the_named_items_whole_and_says_what_is_open_otherwise(self, commands):
        WatchList.save("Parcel", "order 4711\nleave it with Alice next door", "2026-03-12")
        WatchList.save("Dentist", "12:30")
        loaded = commands._run_load_watch("parcel", "dentist")
        assert "Alice next door" in loaded and "12:30" in loaded
        assert "parcel" in commands._run_load_watch() and "Alice" not in commands._run_load_watch()
        assert "parcel" in commands._run_load_watch("sofa")

    def test_loading_an_item_brings_the_watch_toolkit_unless_the_window_holds_it(self, commands):
        """The prompt documents `load_watch` alone: the commands that update an item come with
        the item. A second load points nowhere — the reference is in the window, and
        `load_toolkit("watch")` would point back at it."""
        from assistant.commands import Toolkits
        WatchList.save("Parcel", "order 4711", "2026-03-12")
        reference = Toolkits.render(Toolkits.by_name("watch"))
        history = Mock()
        history.loaded_toolkits.return_value = {}
        with patch.object(Commands, "_prompt_manager", return_value=Mock(message_history=history, render_template=lambda t: t)):
            loaded = commands._run_load_watch("parcel")
            assert loaded.startswith("# ") and "order 4711" in loaded and loaded.endswith(reference)
            assert loaded.index("order 4711") < loaded.index(reference)
            history.loaded_toolkits.return_value = {"watch": 1.0}
            again = commands._run_load_watch("parcel")
        assert "order 4711" in again and Toolkits.HEADER_PREFIX not in again
        assert reference in commands._run_load_watch("parcel"), "without a PromptManager (tests) the reference comes along"

    def test_loading_a_calendar_is_a_background_command_and_runs_in_test_suite_mode(self, commands):
        commands.background_command = Mock()
        run(commands, 'load_calendar("Work")')
        function, params = commands.background_command.call_args.args[:2]
        assert params == ["Work"] and callable(function)
        assert commands.is_pure("load_calendar")

    def test_loading_a_calendar_without_a_name_or_with_a_broken_account_is_a_failure_message(self, commands):
        assert "failed" in commands._owner("load_calendar")._run_load_calendar()
        with patch("assistant.commands.Commands.CalendarAccess.current") as current:
            current.return_value.load.side_effect = ValueError("no login")
            assert "no login" in commands._owner("load_calendar")._run_load_calendar("Work")

    def test_a_watch_without_a_name_is_an_error(self, commands):
        assert run(commands, "save_watch()")
        assert WatchList.items() == []

    def test_the_retired_commands_are_gone(self, commands):
        available = commands.list_commands()
        assert not {"watch", "watch_update", "watch_close", "procedure_note"} & set(available)


@pytest.mark.unit
class TestProcedureCommands:
    BODY = '1) open shop.example (the cart)\n2) click "Pay", never press Back'

    def test_save_and_delete_through_the_dispatcher(self, commands):
        assert not run(commands, f'save_procedure("Shop checkout", "paying at shop.example", """{self.BODY}""")')
        assert self.BODY in Procedures.read("shop-checkout")
        assert not run(commands, 'delete_procedure("shop-checkout")')
        assert Procedures.names() == []

    def test_the_assistant_may_replace_a_procedure(self, commands):
        run(commands, 'save_procedure("shop", "paying", """old steps""")')
        assert not run(commands, 'save_procedure("shop", "paying", """new steps""")')
        assert "new steps" in Procedures.read("shop")

    def test_store_refusals_come_back_as_the_commands_error(self, commands):
        assert run(commands, 'save_procedure("shop", "paying")')
        assert "bank-login" in run(commands, 'delete_procedure("bank-login")')

    def test_loading_returns_each_named_procedure(self, commands):
        Procedures.save("shop-checkout", "paying", "cart steps", None, None)
        Procedures.save("school-portal-login", "school", "login steps", None, None)
        reply = commands._run_load_procedure("shop checkout", "school-portal-login")
        assert "cart steps" in reply and "login steps" in reply

    def test_loading_nothing_or_an_unknown_name_says_what_exists(self, commands):
        Procedures.save("shop-checkout", "paying", "cart steps", None, None)
        assert "shop-checkout" in commands._run_load_procedure()
        unknown = commands._run_load_procedure("bank-login")
        assert "shop-checkout" in unknown and "cart steps" not in unknown

    def test_loading_a_procedure_is_a_background_command(self, commands):
        commands.background_command = Mock()
        run(commands, 'load_procedure("shop-checkout")')
        function, params = commands.background_command.call_args.args[:2]
        assert params == ["shop-checkout"] and callable(function)

    def test_loading_runs_in_test_suite_mode_but_writing_does_not(self, commands):
        assert commands.is_pure("load_procedure") and commands.is_pure("load_watch")
        assert not any(commands.is_pure(c) for c in ("save_procedure", "delete_procedure", "save_watch", "checked_watch", "delete_watch"))


@pytest.mark.unit
class TestEcho:
    @pytest.mark.parametrize("command, params, needle", [
        ("save_watch", ["Parcel from shop.example", "order 4711", "2026-03-12"], "Parcel from shop.example"),
        ("checked_watch", ["festival-tickets"], "festival-tickets"),
        ("delete_watch", ["dentist-charlotte", "visit done"], "dentist-charlotte"),
        ("load_procedure", ["shop-checkout"], "shop-checkout"),
        ("save_procedure", ["shop-checkout", "paying", "a long body, with commas"], "shop-checkout"),
    ])
    def test_the_chat_echo_names_the_thing_and_not_the_whole_payload(self, commands, command, params, needle):
        text = commands.describe(command, params)
        assert needle in text
        assert "a long body" not in text and "order 4711" not in text
