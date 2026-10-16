"""`homeclaw_set`: the three parameters, their normalisation, and the background run;
`homeclaw_events`: the hours, the rendering, the app's absence."""
from unittest.mock import patch

import pytest

from assistant.integration.Feed import FeedEntry
from integrations.homeclaw.HomeClawClient import HomeClawUnavailable
from integrations.homeclaw.HomeClawIntegration import HomeClawIntegration

MODULE = "integrations.homeclaw.HomeClawCommands"


@pytest.fixture
def owner(commands_instance):
    return commands_instance._owner("homeclaw_set")


@pytest.mark.unit
class TestCmdHomeclawSet:
    def test_runs_the_write_in_the_background_with_the_raw_name_and_a_bare_value(self, commands_instance, owner):
        with patch.object(commands_instance, "background_command") as bg:
            assert commands_instance.run("homeclaw_set", ['"Child Radiator [Child Room]"', "Target Temperature", '"21°C"']) == ""
        function, params = bg.call_args.args
        assert params == ("Child Radiator (Child Room)", "target_temperature", "21")
        integration = HomeClawIntegration.__new__(HomeClawIntegration)
        with patch.object(HomeClawIntegration, "get_instance", return_value=integration), \
                patch.object(HomeClawIntegration, "set_characteristic", return_value="done") as setter:
            assert function(params) == "done"
        setter.assert_called_once_with("Child Radiator (Child Room)", "target_temperature", "21")

    @pytest.mark.parametrize("params", [[], ['"Lamp"'], ['"Lamp"', "power"], ['"Lamp"', "", "true"]])
    def test_fewer_than_three_parameters_is_refused_before_any_work(self, commands_instance, params):
        with patch.object(commands_instance, "background_command") as bg:
            assert "expected" in commands_instance.run("homeclaw_set", params)
        bg.assert_not_called()

    def test_is_a_sensitive_multi_parameter_command(self, commands_instance):
        assert commands_instance.is_multi_param("homeclaw_set")
        assert commands_instance.is_sensitive("homeclaw_set")

    def test_describe_names_what_is_set(self, owner):
        assert "target_temperature" in owner.describe_homeclaw_set('"Lamp", target_temperature, 21')


@pytest.mark.unit
class TestCmdHomeclawEvents:
    @pytest.fixture
    def run(self, commands_instance):
        owner = commands_instance._owner("homeclaw_events")
        integration = HomeClawIntegration.__new__(HomeClawIntegration)
        with patch.object(HomeClawIntegration, "get_instance", return_value=integration), \
                patch.object(HomeClawIntegration, "recent_events") as recent:
            yield owner._run_homeclaw_events, recent

    @pytest.mark.parametrize("params, hours", [([], 24.0), (["6"], 6.0), (['"1.5h"'], 1.5), (["9999"], 336.0)])
    def test_the_hours_default_to_a_day_and_are_capped(self, commands_instance, params, hours):
        with patch.object(commands_instance, "background_command") as bg:
            assert commands_instance.run("homeclaw_events", params) == ""
        assert bg.call_args.args[1] == hours

    @pytest.mark.parametrize("params", [["soon"], ["0"], ["-2"]])
    def test_a_window_that_is_not_hours_is_refused(self, commands_instance, params):
        with patch.object(commands_instance, "background_command") as bg:
            assert "failed" in commands_instance.run("homeclaw_events", params)
        bg.assert_not_called()

    def test_lists_each_change_with_its_time(self, run):
        run_events, recent = run
        recent.return_value = [FeedEntry(1792170600.0, "Garage Door [Garage] is open")]
        text = run_events(6.0)
        assert "Garage Door [Garage] is open" in text and "2026-10-16" in text
        recent.assert_called_once_with(6.0)

    def test_nothing_changed(self, run):
        run_events, recent = run
        recent.return_value = []
        assert "Nothing changed" in run_events(24.0)

    def test_the_apps_absence_is_the_answer(self, run):
        run_events, recent = run
        recent.side_effect = HomeClawUnavailable("the HomeClaw app is not running (no socket)")
        assert "not running" in run_events(24.0)
        recent.side_effect = TimeoutError()
        assert run_events(24.0) == "Request timed out"
