"""`hue_pair`: the background run, the disabled answer, the link button, discovery, and the
config writes plus restart on success."""
from unittest.mock import Mock, patch

import pytest
import requests

from assistant.commands.Commands import Commands
from assistant.integration.Integration import IntegrationState
from integrations.hue.HueClient import HueError, LinkButtonNotPressed

MODULE = "integrations.hue.HueCommands"


def _monitor(state=IntegrationState.NOT_CONFIGURED):
    integration = Mock()
    integration.state = state
    monitor = Mock()
    monitor.find.return_value = integration
    monitor.restart.return_value = {"restartable": True, "status": {"name": "Philips Hue", "state": "ok", "config_keys": [], "last_error": None}}
    return monitor


@pytest.fixture
def run(commands_instance):
    """Runs _run_hue_pair with the bridge calls and the config store patched; yields (run, mocks)."""
    owner = commands_instance._owner("hue_pair")
    with patch(f"{MODULE}.pair") as pair, patch(f"{MODULE}.discover") as discover, \
            patch(f"{MODULE}.ConfigStore") as store, patch(f"{MODULE}.config") as cfg:
        cfg.get.return_value = ""
        mocks = Mock(pair=pair, discover=discover, store=store, cfg=cfg, monitor=_monitor())
        with patch.object(commands_instance, "integration_monitor", return_value=mocks.monitor):
            yield owner._run_hue_pair, mocks


@pytest.mark.unit
class TestCmdHuePair:
    def test_runs_in_the_background_on_the_config_lane(self, commands_instance):
        with patch.object(commands_instance, "background_command") as bg:
            assert commands_instance.run("hue_pair", ['"192.0.2.10"']) == ""
        bg.assert_called_once_with(commands_instance._owner("hue_pair")._run_hue_pair, ("192.0.2.10",), lane=Commands.CONFIG_LANE)

    def test_an_unregistered_integration_asks_for_a_restart(self, run):
        run_pair, mocks = run
        mocks.monitor.find.return_value = None
        assert "restart" in run_pair("192.0.2.10").lower()
        mocks.pair.assert_not_called()

    def test_a_disabled_integration_is_pointed_at_enable_integration(self, run):
        run_pair, mocks = run
        mocks.monitor.find.return_value.state = IntegrationState.DISABLED
        assert "enable_integration" in run_pair("192.0.2.10")
        mocks.pair.assert_not_called()

    def test_success_writes_both_keys_and_restarts(self, run):
        run_pair, mocks = run
        mocks.pair.return_value = "abc123"
        text = run_pair("192.0.2.10")
        assert "192.0.2.10" in text
        mocks.store.set_value.assert_any_call("hue.app_key", "abc123")
        mocks.store.set_value.assert_any_call("hue.bridge", "192.0.2.10")
        mocks.monitor.restart.assert_called_once_with("Philips Hue")

    def test_a_bridge_already_configured_is_not_written_again(self, run):
        run_pair, mocks = run
        mocks.cfg.get.return_value = "192.0.2.10"
        mocks.pair.return_value = "abc123"
        run_pair("")
        mocks.pair.assert_called_once_with("192.0.2.10")
        assert [c.args[0] for c in mocks.store.set_value.call_args_list] == ["hue.app_key"]
        mocks.discover.assert_not_called()

    def test_the_link_button_answer_asks_for_the_press(self, run):
        run_pair, mocks = run
        mocks.pair.side_effect = LinkButtonNotPressed("link button not pressed")
        text = run_pair("192.0.2.10")
        assert "button" in text and "hue_pair()" in text
        mocks.store.set_value.assert_not_called()

    @pytest.mark.parametrize("failure", [HueError("invalid value"), requests.ConnectionError("refused")])
    def test_other_failures_are_reported(self, run, failure):
        run_pair, mocks = run
        mocks.pair.side_effect = failure
        assert run_pair("192.0.2.10").startswith("hue_pair failed")

    def test_without_an_address_the_bridge_is_discovered(self, run):
        run_pair, mocks = run
        mocks.discover.return_value = ["192.0.2.20"]
        mocks.pair.return_value = "abc123"
        run_pair("")
        mocks.pair.assert_called_once_with("192.0.2.20")
        mocks.store.set_value.assert_any_call("hue.bridge", "192.0.2.20")

    def test_no_bridge_found_asks_for_the_address(self, run):
        run_pair, mocks = run
        mocks.discover.return_value = []
        assert 'hue_pair("<address>")' in run_pair("")
        mocks.pair.assert_not_called()

    def test_several_bridges_are_listed(self, run):
        run_pair, mocks = run
        mocks.discover.return_value = ["192.0.2.20", "192.0.2.21"]
        text = run_pair("")
        assert "192.0.2.20" in text and "192.0.2.21" in text
        mocks.pair.assert_not_called()

    def test_discovery_is_asked_without_a_host_and_only_then(self, run):
        run_pair, mocks = run
        mocks.discover.return_value = ["192.0.2.20"]
        mocks.pair.return_value = "abc123"
        run_pair("")
        mocks.discover.assert_called_once_with()
        mocks.pair.assert_called_once_with("192.0.2.20")
