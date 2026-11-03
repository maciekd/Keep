"""The mechanism that lets an integration folder bring its own commands (CommandSet)."""
import importlib
import sys
from unittest.mock import Mock, patch

import pytest

from assistant.ChannelProfile import profile_for
from assistant.commands import CommandSet as command_set_module
from assistant.commands.CommandSet import CommandSet
from assistant.commands.Commands import Commands


class LampCommands(CommandSet):
    MULTI_PARAM_COMMANDS = frozenset({"lamp_scene"})
    SENSITIVE_COMMANDS = frozenset({"lamp_scene"})
    PURE_COMMANDS = frozenset({"lamp_state"})
    SELF_EVIDENT_COMMANDS = frozenset({"lamp_on"})
    APPROVAL_NOTES = {"lamp_scene": "Changes every lamp in the room."}

    def cmd_lamp_on(self, parameter_string):
        return f"on:{parameter_string}"

    def cmd_lamp_scene(self, *args):
        return "scene:" + "|".join(args)

    def cmd_lamp_state(self, parameter_string):
        return f"asked over {self.profile.name}, request {self.context}"

    def describe_lamp_on(self, p):
        return f"Lighting {p}"

    def helper(self):
        return "not a command"


class KettleCommands(CommandSet):
    def cmd_lamp_on(self, parameter_string):
        return "the kettle answers to a lamp's name"


class DimmerCommands(CommandSet):
    def cmd_dim(self, *args):
        return "a folder taking a core command"


@pytest.fixture
def commands():
    return Commands(Mock(), command_sets=[LampCommands])


def test_a_single_parameter_command_of_a_set_gets_the_first_parameter(commands):
    assert commands.run("lamp_on", ["hall", "ignored"]) == "on:hall"
    assert commands.run("lamp_on", []) == "on:"


def test_a_multi_parameter_command_of_a_set_gets_every_parameter(commands):
    assert commands.run("lamp_scene", ["hall", "dusk"]) == "scene:hall|dusk"


def test_the_flags_of_a_command_come_from_the_set_that_owns_it(commands):
    assert commands.is_sensitive("lamp_scene") and not commands.is_sensitive("lamp_on")
    assert commands.is_pure("lamp_state") and not commands.is_pure("lamp_scene")
    assert commands.is_multi_param("lamp_scene") and not commands.is_multi_param("lamp_on")
    assert "every lamp" in commands.approval_note("lamp_scene") and commands.approval_note("lamp_on") == ""


def test_core_commands_keep_their_own_flags_beside_a_set(commands):
    assert commands.is_multi_param("dim") and commands.is_sensitive("send_email") and commands.is_pure("load_toolkit")
    assert not commands.is_sensitive("turn_on")


def test_whether_a_commands_effect_is_its_answer_is_the_owners_verdict(commands):
    # RequestProcessor asks the host, per command, whether a request may end silently on it;
    # the host must find the set's declaration, and a core action keeps its own.
    assert commands.is_self_evident("lamp_on") and not commands.is_self_evident("lamp_state")
    assert commands.is_self_evident("turn_off") and not commands.is_self_evident("remember")
    assert not commands.is_self_evident("no_such_command")


def test_a_set_describes_its_commands_and_one_without_a_describer_echoes_the_call(commands):
    assert "hall" in commands.describe("lamp_on", ["hall"])
    assert commands.describe("lamp_scene", ["hall", "dusk"]) is None
    echo = commands.echo_text("lamp_scene", ["hall", "dusk"])
    assert echo.startswith("lamp_scene(") and "hall" in echo and "dusk" in echo


def test_the_command_list_holds_the_sets_commands_among_the_core_ones_in_order(commands):
    names = commands.list_commands()
    assert {"lamp_on", "lamp_scene", "lamp_state", "turn_on", "load_toolkit"} <= set(names)
    assert "helper" not in names and names == sorted(names)


def test_a_set_reads_the_channel_and_the_request_of_the_run_not_of_its_construction(commands):
    commands._current_profile = profile_for("telegram")
    commands._current_context = "request-7"
    first = commands.run("lamp_state", [])
    commands._current_profile = profile_for("console")
    commands._current_context = "request-8"
    second = commands.run("lamp_state", [])
    assert "request-7" in first and "request-8" in second
    assert profile_for("telegram").name in first and profile_for("console").name in second


@pytest.mark.parametrize("sets, owners", [
    ([LampCommands, KettleCommands], ("LampCommands", "KettleCommands")),
    ([DimmerCommands], ("Commands", "DimmerCommands")),
], ids=["two sets", "a set and core"])
def test_a_command_name_with_two_owners_is_refused_naming_both(sets, owners):
    with pytest.raises(ValueError) as refused:
        Commands(Mock(), command_sets=sets)
    assert all(owner in str(refused.value) for owner in owners)


def test_an_unknown_command_is_still_unknown_with_sets_hosted(commands):
    assert "lamp_off" in commands.run("lamp_off", ["hall"])
    assert commands.describe("lamp_off", ["hall"]) is None


@pytest.fixture
def workshop(tmp_path, monkeypatch):
    """A package laid out like `integrations`: a folder with a working command set, one whose
    command module raises on import, and a module that is not a command module though it
    holds a command set."""
    root = tmp_path / "workshop"
    files = {
        "__init__.py": "",
        "broken/__init__.py": "raise RuntimeError('a folder whose package itself fails')\n",
        "broken/BrokenCommands.py": "",
        "lamp/__init__.py": "",
        "lamp/LampCommands.py": "from assistant.commands.CommandSet import CommandSet\n"
                                "class WorkshopLamp(CommandSet):\n    def cmd_lamp_on(self, p): return p\n",
        "lamp/LampIntegration.py": "from assistant.commands.CommandSet import CommandSet\n"
                                   "class Stray(CommandSet):\n    def cmd_stray(self, p): return p\n",
        "kettle/__init__.py": "",
        "kettle/KettleCommands.py": "import a_package_nobody_installed\n",
        "radio/__init__.py": "",
        "radio/RadioCommands.py": "from assistant.commands.CommandSet import CommandSet\n"
                                  "class WorkshopRadio(CommandSet):\n    def cmd_radio_on(self, p): return p\n",
    }
    for name, source in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield importlib.import_module("workshop")
    for name in [n for n in sys.modules if n == "workshop" or n.startswith("workshop.")]:
        del sys.modules[name]


def test_discovery_takes_the_command_modules_and_leaves_a_broken_one_out(workshop):
    found = command_set_module.find_in(workshop)
    assert [cls.__name__ for cls in found] == ["WorkshopLamp", "WorkshopRadio"]
    assert "workshop.lamp.LampIntegration" not in sys.modules


def test_a_folder_whose_package_raises_costs_only_itself(workshop):
    """pkgutil imports a folder's package to walk it; `broken` raises something other than an
    ImportError, which without the error hook ends the walk — and the boot — before the
    folders after it. It is logged and the walk goes on."""
    with patch.object(command_set_module, "error") as error:
        found = command_set_module.find_in(workshop)
    assert [cls.__name__ for cls in found] == ["WorkshopLamp", "WorkshopRadio"]
    assert any("workshop.broken" in call.args[0] for call in error.call_args_list)


def test_every_command_module_of_this_repository_is_discovered():
    from common.Paths import INTEGRATIONS_DIR
    modules = {f"integrations.{path.parent.name}.{path.stem}" for path in INTEGRATIONS_DIR.glob("*/*Commands.py")}
    assert modules and {cls.__module__ for cls in command_set_module.discover()} == modules


@pytest.mark.parametrize("owner", [Commands, *command_set_module.discover()], ids=lambda owner: owner.__name__)
def test_a_flag_names_only_a_command_its_owner_implements(owner):
    # The flags are asked of the command's owner, so one left behind when a command moves
    # to another owner is never read: a sensitive command would run without approval.
    flagged = (owner.MULTI_PARAM_COMMANDS | owner.SENSITIVE_COMMANDS | owner.PURE_COMMANDS
               | owner.SELF_EVIDENT_COMMANDS | set(owner.APPROVAL_NOTES))
    assert flagged - set(command_set_module.command_names(owner)) == set()
