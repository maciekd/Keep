"""The car commands: resolved through the fleet to whichever brand owns the car, silent on
success for the switches, the provider's line for navigation, the hint without a provider."""
from unittest.mock import Mock, patch

import pytest

from assistant.commands.Commands import Commands
from assistant.integration.VehicleProvider import Fleet, Unsupported, VehicleError, VehicleSpec

MODULE = "assistant.commands.Commands.vehicles"


def _provider(name, *specs):
    provider = Mock()
    provider.name = name
    provider.vehicles.return_value = list(specs)
    return provider


@pytest.fixture
def commands_instance():
    cp = Mock()
    cp.command_reply = None
    cp.command_reply_contains_error = False
    return Commands(cp)


@pytest.fixture
def tesla():
    return _provider("Tesla", VehicleSpec("family", "Family Model Y", "VIN1", ("Family Model Y", "VIN1"), is_default=True))


@pytest.fixture
def fleet(tesla):
    volvo = _provider("Volvo", VehicleSpec("wagon", "The Volvo", "V1"))
    with patch(f"{MODULE}.fleet", return_value=Fleet([tesla, volvo])):
        yield Fleet([tesla, volvo])


def _run(commands_instance, name, params):
    result = commands_instance.run(name, params)
    commands_instance.wait_for_background(timeout=5)
    return result


@pytest.mark.unit
class TestSwitches:
    def test_a_success_says_nothing_and_reaches_the_car_of_the_id(self, commands_instance, fleet, tesla):
        tesla.max_defrost.return_value = "Max defrost enabled for Family Model Y."
        _run(commands_instance, "car_max_defrost", ["Family Model Y"])
        assert tesla.max_defrost.call_args[0][0].id == "family"
        commands_instance.command_processor.set_command_reply.assert_not_called()
        commands_instance.command_processor.background_command_done.assert_called_once()

    def test_an_empty_id_is_the_default_car(self, commands_instance, fleet, tesla):
        _run(commands_instance, "car_climate_on", [])
        assert tesla.climate_on.call_args[0][0].id == "family"

    def test_the_id_picks_the_brand(self, commands_instance, fleet, tesla):
        _run(commands_instance, "car_open_trunk", ["wagon"])
        volvo = fleet.providers[1]
        assert volvo.open_trunk.call_args[0][0].id == "wagon"
        tesla.open_trunk.assert_not_called()

    def test_a_brands_error_is_the_reply(self, commands_instance, fleet, tesla):
        tesla.climate_off.side_effect = VehicleError("vehicle offline")
        _run(commands_instance, "car_climate_off", ["family"])
        commands_instance.command_processor.set_command_reply.assert_called_once_with(
            "Car command failed: vehicle offline", contains_error=False)

    def test_an_action_the_brand_lacks_is_reported_not_raised(self, commands_instance, fleet):
        fleet.providers[1].max_defrost.side_effect = Unsupported("Volvo cannot start max defrost on The Volvo.")
        _run(commands_instance, "car_max_defrost", ["wagon"])
        reply = commands_instance.command_processor.set_command_reply.call_args[0][0]
        assert "cannot start max defrost" in reply

    def test_an_unknown_id_names_the_cars(self, commands_instance, fleet, tesla):
        _run(commands_instance, "car_open_trunk", ["Saab"])
        reply = commands_instance.command_processor.set_command_reply.call_args[0][0]
        assert "No car matches 'Saab'" in reply and "family" in reply and "wagon" in reply
        tesla.open_trunk.assert_not_called()

    def test_without_a_provider_the_reply_is_the_setup_hint(self, commands_instance):
        with patch(f"{MODULE}.fleet", return_value=Fleet([])), \
                patch(f"{MODULE}.no_provider_hint", return_value='Tesla is disabled: enable_integration("Tesla").'):
            _run(commands_instance, "car_max_defrost", ["family"])
        commands_instance.command_processor.set_command_reply.assert_called_once_with(
            'Tesla is disabled: enable_integration("Tesla").', contains_error=False)


@pytest.mark.unit
class TestNavigate:
    def test_the_vehicle_leads_and_the_address_keeps_its_commas(self, commands_instance, fleet, tesla):
        tesla.navigate_to.return_value = "Navigation to Storgatan 1, Stockholm sent to Family Model Y."
        _run(commands_instance, "car_navigate", ["family", "Storgatan 1", "Stockholm"])
        spec, destination = tesla.navigate_to.call_args[0]
        assert (spec.id, destination) == ("family", "Storgatan 1, Stockholm")
        commands_instance.command_processor.set_command_reply.assert_called_once_with(
            "Navigation to Storgatan 1, Stockholm sent to Family Model Y.", contains_error=False)

    def test_a_lone_argument_is_a_destination_for_the_default_car(self, commands_instance, fleet, tesla):
        _run(commands_instance, "car_navigate", ["IKEA Kungens Kurva"])
        spec, destination = tesla.navigate_to.call_args[0]
        assert (spec.id, destination) == ("family", "IKEA Kungens Kurva")

    def test_no_destination_is_refused_without_calling_the_provider(self, commands_instance, fleet, tesla):
        result = _run(commands_instance, "car_navigate", [])
        assert result.startswith("car_navigate failed")
        tesla.navigate_to.assert_not_called()

    def test_a_brands_error_is_the_reply(self, commands_instance, fleet, tesla):
        tesla.navigate_to.side_effect = VehicleError("vehicle asleep")
        _run(commands_instance, "car_navigate", ["family", "Storgatan 1"])
        commands_instance.command_processor.set_command_reply.assert_called_once_with(
            "Car command failed: vehicle asleep", contains_error=False)

    def test_describe_names_the_destination(self, commands_instance):
        assert "Storgatan 1, Stockholm" in commands_instance.describe("car_navigate", ["family", "Storgatan 1", "Stockholm"])

    def test_an_empty_vehicle_argument_leaves_the_destination_to_the_default_car(self, commands_instance, fleet, tesla):
        _run(commands_instance, "car_navigate", ["", "IKEA Kungens Kurva"])
        spec, destination = tesla.navigate_to.call_args[0]
        assert (spec.id, destination) == ("family", "IKEA Kungens Kurva")
