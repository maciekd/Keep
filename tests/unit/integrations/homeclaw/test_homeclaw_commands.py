"""The commands parse what the LLM writes — refs, open/close, positions, the dry-run flag —
and hand it to the integration; a refusal reaches the LLM as text."""
from unittest.mock import Mock, patch

import pytest

from assistant.integration.Feed import FeedEntry
from integrations.homeclaw.HomeClawClient import HomeClawUnavailable
from integrations.homeclaw.HomeClawCommands import HomeClawCommands
from integrations.homeclaw.HomeClawIntegration import HomeClawIntegration
from integrations.homeclaw.HomeClawModel import Unresolved


@pytest.fixture
def homeclaw():
    with patch.object(HomeClawIntegration, "get_instance", return_value=Mock()) as get_instance:
        for setter in ("set_cover", "set_garage", "set_valve", "set_light", "set_thermostat", "trigger_scene"):
            getattr(get_instance.return_value, setter).return_value = "done"
        yield get_instance.return_value


def _run(commands_instance, command, *params):
    commands_instance.run(command, list(params))
    commands_instance.wait_for_background(timeout=5)
    return commands_instance.command_processor.set_command_reply.call_args.args[0]


def test_open_and_close_are_the_end_positions_of_a_cover(commands_instance, homeclaw):
    _run(commands_instance, "homekit_cover", "B2000001", "open")
    homeclaw.set_cover.assert_called_with("B2000001", 100, False)
    _run(commands_instance, "homekit_cover", "B2000001", "40 %")
    homeclaw.set_cover.assert_called_with("B2000001", 40, False)


def test_the_dry_run_flag_is_taken_off_the_arguments(commands_instance, homeclaw):
    _run(commands_instance, "homekit_cover", "B2000001", "close", "dry_run")
    homeclaw.set_cover.assert_called_with("B2000001", 0, True)


def test_the_garage_ref_may_be_left_out(commands_instance, homeclaw):
    _run(commands_instance, "homekit_garage", "open")
    homeclaw.set_garage.assert_called_with("", True, False)


def test_valve_light_thermostat_and_scene_reach_their_setter(commands_instance, homeclaw):
    _run(commands_instance, "homekit_valve", "D4000001", "off")
    homeclaw.set_valve.assert_called_with("D4000001", False, False)
    _run(commands_instance, "homekit_light", "B6000001", "on", "dry_run")
    homeclaw.set_light.assert_called_with("B6000001", True, True)
    _run(commands_instance, "homekit_thermostat", "21,5 °C")
    homeclaw.set_thermostat.assert_called_with("", 21.5, False)
    _run(commands_instance, "homekit_scene", "Good Night")
    homeclaw.trigger_scene.assert_called_with("Good Night", False)


def test_a_bad_position_never_reaches_the_integration(commands_instance, homeclaw):
    reply = _run(commands_instance, "homekit_cover", "B2000001", "150")
    homeclaw.set_cover.assert_not_called()
    assert "150" in reply


def test_a_refusal_from_the_integration_is_the_reply(commands_instance, homeclaw):
    homeclaw.set_light.side_effect = Unresolved("Several lights are called 'Lamp'")
    assert "Several" in _run(commands_instance, "homekit_light", "Lamp", "on")


def test_the_event_log_is_one_line_per_change_and_says_when_there_is_none(commands_instance, homeclaw):
    homeclaw.events.return_value = [FeedEntry(1772359205, "HomeKit: Garage Door [Garage] current door state closed → open")]
    reply = _run(commands_instance, "homekit_events", "6h")
    homeclaw.events.assert_called_with("6h")
    assert "Garage Door [Garage]" in reply and reply.count("\n") == 0
    homeclaw.events.return_value = []
    assert "No HomeKit changes" in _run(commands_instance, "homekit_events", "")
    homeclaw.events.assert_called_with("1h")


def test_the_app_not_running_is_the_reply(commands_instance, homeclaw):
    homeclaw.events.side_effect = HomeClawUnavailable("HomeClaw is not running (no socket)")
    assert "not running" in _run(commands_instance, "homekit_events", "1h")


def test_the_garage_and_the_thermostat_need_approval():
    assert HomeClawCommands.SENSITIVE_COMMANDS == {"homekit_garage", "homekit_thermostat"}
    assert set(HomeClawCommands.APPROVAL_NOTES) == HomeClawCommands.SENSITIVE_COMMANDS


def test_automations_take_an_optional_name_and_a_refusal_reaches_the_llm(commands_instance, homeclaw):
    homeclaw.describe_automations.return_value = "Weekday Morning: at 06:30"
    assert "06:30" in _run(commands_instance, "homekit_automations", "morning")
    homeclaw.describe_automations.assert_called_with("morning")
    homeclaw.describe_automations.side_effect = HomeClawUnavailable("HomeClaw is not running (no socket)")
    assert "not running" in _run(commands_instance, "homekit_automations")
