from unittest.mock import ANY, Mock, patch

import pytest

from integrations.ferry.FerryIntegration import FerryError, FerryIntegration


@pytest.fixture
def ferry():
    with patch.object(FerryIntegration, "get_instance", return_value=Mock()) as get_instance:
        yield get_instance.return_value


def test_the_named_harbor_is_passed_unquoted(commands_instance, ferry):
    ferry.departures.return_value = "From Ekö to Strandvik: 14:10"
    commands_instance.run("ferry_departures", ['"Ekö"'])
    commands_instance.wait_for_background(timeout=5)
    ferry.departures.assert_called_once_with("Ekö")
    commands_instance.command_processor.set_command_reply.assert_called_once_with(
        "From Ekö to Strandvik: 14:10", contains_error=False, parts=ANY)


def test_no_argument_asks_for_both_directions(commands_instance, ferry):
    commands_instance.run("ferry_departures", [])
    commands_instance.wait_for_background(timeout=5)
    ferry.departures.assert_called_once_with("")


def test_a_failure_reaches_the_llm_as_text(commands_instance, ferry):
    ferry.departures.side_effect = FerryError("Trafikverket refused the API key")
    commands_instance.run("ferry_departures", ["Ekö"])
    commands_instance.wait_for_background(timeout=5)
    assert "refused" in commands_instance.command_processor.set_command_reply.call_args.args[0]
