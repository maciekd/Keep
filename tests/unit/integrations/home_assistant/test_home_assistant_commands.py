"""The home_assistant toolkit's commands reach the integration's rule methods and hand their
text back as the reply."""
from unittest.mock import Mock, patch

import pytest

from integrations.home_assistant.HomeAssistantCommands import HomeAssistantCommands
from integrations.home_assistant.HomeAssistantIntegration import HomeAssistantIntegration
from tests.unit.waiting import FAILURE_BOUND


@pytest.fixture
def house():
    with patch.object(HomeAssistantIntegration, "get_instance", return_value=Mock()) as get_instance:
        yield get_instance.return_value


def _reply(commands_instance):
    return commands_instance.command_processor.set_command_reply.call_args.args[0]


@pytest.mark.unit
class TestRuleCommands:
    def test_exclude_hands_the_pattern_over_and_replies_with_the_integrations_text(self, commands_instance, house):
        house.exclude.return_value = "Excluded binary_sensor.hp_*: 12 entities (…) leave the house data."
        commands_instance.run("exclude_entities", ["binary_sensor.hp_*"])
        commands_instance.wait_for_background(timeout=FAILURE_BOUND)
        house.exclude.assert_called_once_with("binary_sensor.hp_*")
        assert "12 entities" in _reply(commands_instance)

    def test_include_and_clear(self, commands_instance, house):
        house.include.return_value = "Included binary_sensor.plejd_hall_is_gateway: 1 entity."
        house.clear_rule.return_value = "Cleared the include rule for binary_sensor.plejd_hall_is_gateway."
        commands_instance.run("include_entities", ["binary_sensor.plejd_hall_is_gateway"])
        commands_instance.wait_for_background(timeout=FAILURE_BOUND)
        house.include.assert_called_once_with("binary_sensor.plejd_hall_is_gateway")
        commands_instance.run("clear_entity_rule", ["binary_sensor.plejd_hall_is_gateway"])
        commands_instance.wait_for_background(timeout=FAILURE_BOUND)
        house.clear_rule.assert_called_once_with("binary_sensor.plejd_hall_is_gateway")
        assert "Cleared" in _reply(commands_instance)

    def test_the_rules_list_is_the_reply(self, commands_instance, house):
        house.rules_text.return_value = "exclude binary_sensor.hp_* — 12 entities"
        commands_instance.run("entity_rules", [])
        commands_instance.wait_for_background(timeout=FAILURE_BOUND)
        assert "binary_sensor.hp_*" in _reply(commands_instance)

    def test_a_refusal_reaches_the_llm_as_text(self, commands_instance, house):
        house.exclude.return_value = "'hall light' is not an entity id pattern"
        commands_instance.run("exclude_entities", ["hall light"])
        commands_instance.wait_for_background(timeout=FAILURE_BOUND)
        assert "not an entity id pattern" in _reply(commands_instance)


@pytest.mark.unit
class TestToolkit:
    def test_listing_is_pure_and_the_toolkit_offers_all_four(self):
        from assistant.commands import Toolkits
        assert "entity_rules" in HomeAssistantCommands.PURE_COMMANDS
        toolkit = Toolkits.by_name("home_assistant")
        assert toolkit is not None and toolkit.folder == "home_assistant"
        assert set(toolkit.commands) == {"exclude_entities", "include_entities", "clear_entity_rule", "entity_rules"}
