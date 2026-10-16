"""The automations: when each fires and on which days, what its hidden action set does per
accessory, a named scene by its name, a line each in the section, and a reply that still
comes when a detail cannot be read."""
from integrations.homeclaw.HomeClawAutomations import actions_text, condition_text, trigger_text
from .conftest import BUTTON, HIDDEN_SET, MORNING, ok


def test_the_trigger_drops_the_condition_count_and_the_press_type_reads_as_words():
    assert trigger_text({"event_summary": "at sunset-45m (+1 condition)", "trigger_type": "significant_time"}) == "at sunset-45m"
    assert "single press" in trigger_text({"event_summary": "Bedside Switch single_press", "trigger_type": "button"})


def test_a_trigger_without_summary_falls_back_to_its_time_and_an_orphan_says_it_never_fires():
    assert trigger_text({"fire_time": "06:30", "trigger_type": "timer"}) == "at 06:30"
    assert "never fires" in trigger_text({"event_summary": "", "trigger_type": "orphaned"})


def test_weekday_sets_read_as_daily_workdays_and_weekend():
    assert condition_text({"type": "weekdays", "weekdays": [7, 1, 2, 3, 4, 5, 6]}) == "daily"
    assert condition_text({"type": "weekdays", "weekdays": [2, 3, 4, 5, 6]}) == "Mon–Fri"
    assert condition_text({"type": "weekdays", "weekdays": [1, 7]}) == "Sat–Sun"
    assert condition_text({"type": "weekdays", "weekdays": [2, 4], "summary": "Mon, Wed"}) == "Mon, Wed"


def test_accessories_with_the_same_setting_are_grouped_and_one_accessory_is_one_setting():
    text = actions_text([
        {"accessory": "Lamp A", "accessory_id": "1", "characteristic": "brightness", "value": "30"},
        {"accessory": "Lamp A", "accessory_id": "1", "characteristic": "power", "value": "1"},
        {"accessory": "Lamp B", "accessory_id": "2", "characteristic": "power", "value": "0"},
        {"accessory": "Lamp C", "accessory_id": "3", "characteristic": "power", "value": "false"},
        {"accessory": "Heater", "accessory_id": "4", "characteristic": "target_heating_cooling", "value": "1"},
        {"accessory": "Heater", "accessory_id": "4", "characteristic": "target_temperature", "value": "22"},
    ])
    assert "off: Lamp B, Lamp C" in text
    assert "on 30 %: Lamp A" in text
    assert "heat 22 °C: Heater" in text


class TestAgainstTheApp:
    def test_the_command_reply_has_trigger_days_state_and_the_hidden_actions(self, homeclaw):
        lines = homeclaw.describe_automations().splitlines()
        morning = lines.index(next(line for line in lines if line.startswith("Weekday Morning")))
        button = lines.index(next(line for line in lines if line.startswith("Bedside Button")))
        assert "06:30" in lines[morning] and "Mon–Fri" in lines[morning] and "disabled" not in lines[morning]
        assert "on 30 %: Floor Lamp" in lines[morning + 1] and "closed: Awning Roof" in lines[morning + 1]
        assert "21 °C: Radiator" in lines[morning + 1]
        assert "disabled" in lines[button] and "scene Good Night" in lines[button + 1]

    def test_a_named_scene_is_only_read_when_asked_for_by_name(self, app, homeclaw):
        homeclaw.describe_automations()
        assert app.of("get_scene") == [{"id": HIDDEN_SET}]
        text = homeclaw.describe_automations("bedside")
        assert "Weekday Morning" not in text
        assert "scene Good Night: off: Floor Lamp" in text

    def test_an_unknown_name_says_so(self, homeclaw):
        assert "No HomeKit automation" in homeclaw.describe_automations("garden party")

    def test_an_automation_whose_detail_fails_is_kept_without_conditions(self, app, homeclaw):
        app.replies["get_automation"] = {"success": False, "error": "Automation not found"}
        automations = homeclaw.automations()
        assert {a["id"] for a in automations} == {MORNING, BUTTON}
        assert all(not a.get("conditions") for a in automations)

    def test_the_section_ends_with_a_line_per_automation(self, homeclaw):
        section = homeclaw.snapshot()
        block = section[section.index("## Automations"):].splitlines()[1:]
        assert len(block) == 2
        assert any("Weekday Morning" in line and "4 actions" in line for line in block)

    def test_an_app_without_automations_keeps_the_rest_of_the_section(self, app, homeclaw):
        app.replies["list_automations"] = {"success": False, "error": "Unknown command: list_automations"}
        section = homeclaw.snapshot()
        assert "## Automations" not in section and "## Game room" in section

    def test_no_automations_adds_no_block(self, app, homeclaw):
        app.replies["list_automations"] = ok([])
        assert "## Automations" not in homeclaw.snapshot()
        assert "no HomeKit automations" in homeclaw.describe_automations()
