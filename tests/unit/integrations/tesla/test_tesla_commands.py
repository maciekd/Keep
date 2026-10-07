from unittest.mock import Mock, patch

import pytest

from integrations.tesla.TeslaIntegration import TeslaCommandError, TeslaIntegration


@pytest.fixture
def tesla_api():
    with patch.object(TeslaIntegration, "get_instance", return_value=Mock()) as get_instance:
        yield get_instance.return_value


class TestTeslaBackgroundCommands:
    def test_defrost_success_is_silent(self, commands_instance, tesla_api):
        mock_api = tesla_api
        mock_api.set_max_defrost.return_value = "Max defrost enabled for Model Y."

        commands_instance.run("tesla_max_defrost", ["Model Y"])
        commands_instance.wait_for_background(timeout=5)

        mock_api.set_max_defrost.assert_called_once_with("Model Y")
        commands_instance.command_processor.set_command_reply.assert_not_called()
        commands_instance.command_processor.background_command_done.assert_called_once()

    def test_error_caught_and_returned(self, commands_instance, tesla_api):
        mock_api = tesla_api
        mock_api.set_max_defrost.side_effect = TeslaCommandError("vehicle offline")

        commands_instance.run("tesla_max_defrost", [""])
        commands_instance.wait_for_background(timeout=5)

        commands_instance.command_processor.set_command_reply.assert_called_once_with(
            "Tesla command failed: vehicle offline", contains_error=False
        )

    def test_empty_param_passes_none(self, commands_instance, tesla_api):
        mock_api = tesla_api
        mock_api.start_air_conditioning.return_value = "Air conditioning started for car."

        commands_instance.run("tesla_air_con_on", [])
        commands_instance.wait_for_background(timeout=5)

        mock_api.start_air_conditioning.assert_called_once_with(None)


class TestTeslaNavigate:
    def test_vehicle_leads_and_the_address_keeps_its_commas(self, commands_instance, tesla_api):
        tesla_api.navigate_to.return_value = "Navigation to Storgatan 1, Stockholm sent to Model Y."

        commands_instance.run("tesla_navigate", ["Model Y", "Storgatan 1", "Stockholm"])
        commands_instance.wait_for_background(timeout=5)

        tesla_api.navigate_to.assert_called_once_with("Model Y", "Storgatan 1, Stockholm")
        commands_instance.command_processor.set_command_reply.assert_called_once_with(
            "Navigation to Storgatan 1, Stockholm sent to Model Y.", contains_error=False
        )

    def test_a_lone_argument_is_a_destination_for_the_default_car(self, commands_instance, tesla_api):
        tesla_api.navigate_to.return_value = "Navigation to IKEA Kungens Kurva sent to car."

        commands_instance.run("tesla_navigate", ["IKEA Kungens Kurva"])
        commands_instance.wait_for_background(timeout=5)

        tesla_api.navigate_to.assert_called_once_with(None, "IKEA Kungens Kurva")

    def test_no_destination_is_refused_without_calling_the_api(self, commands_instance, tesla_api):
        result = commands_instance.run("tesla_navigate", [])
        commands_instance.wait_for_background(timeout=5)

        assert result.startswith("tesla_navigate failed")
        tesla_api.navigate_to.assert_not_called()

    def test_api_error_is_returned_as_the_reply(self, commands_instance, tesla_api):
        tesla_api.navigate_to.side_effect = TeslaCommandError("vehicle asleep")

        commands_instance.run("tesla_navigate", ["Model Y", "Storgatan 1"])
        commands_instance.wait_for_background(timeout=5)

        commands_instance.command_processor.set_command_reply.assert_called_once_with(
            "Tesla command failed: vehicle asleep", contains_error=False
        )

    def test_describe_names_the_destination(self, commands_instance):
        assert "Storgatan 1, Stockholm" in commands_instance.describe(
            "tesla_navigate", ["Model Y", "Storgatan 1", "Stockholm"])
