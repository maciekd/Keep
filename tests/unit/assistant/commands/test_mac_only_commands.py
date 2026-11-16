"""What the setup commands answer on a server that is not a Mac: a command that acts on the Mac
is refused, a Mac-only integration is listed as such and not enabled, and the config keys of the
server's own audio and of the launchd watchdog are not written."""
from unittest.mock import Mock, patch

import pytest

from assistant.commands import Toolkits
from assistant.commands.Commands import Commands
from common import Platform, Watchdog


@pytest.fixture
def commands():
    processor = Mock()
    processor.command_reply = None
    processor.command_reply_contains_error = False
    return Commands(processor)


@pytest.fixture
def linux(monkeypatch):
    monkeypatch.setattr(Platform, "current", lambda: Platform.LINUX)


@pytest.fixture
def mac(monkeypatch):
    monkeypatch.setattr(Platform, "current", lambda: Platform.MAC)


def _statuses(**extra):
    photos = {"name": "Photos", "state": "disabled", "needs_mac": True, "last_check_time": None, "keys": [],
              "restartable": True, "last_error": Platform.mac_only("Photos")}
    photos.update(extra)
    return [photos]


@pytest.mark.unit
class TestMacOnlyCommand:
    def test_every_mac_only_command_is_refused_on_linux_before_it_runs(self, commands, linux):
        for command in Commands.MAC_ONLY_COMMANDS:
            with patch.object(Commands, f"cmd_{command}") as handler:
                answer = commands.run(command, [])
            handler.assert_not_called()
            assert command in answer and "Linux" in answer

    def test_on_a_mac_it_runs(self, commands, mac):
        for command in Commands.MAC_ONLY_COMMANDS:
            with patch.object(Commands, f"cmd_{command}", return_value="done") as handler:
                assert commands.run(command, []) == "done"
            handler.assert_called_once()

    def test_the_toolkit_offers_it_on_a_mac_alone(self, linux, monkeypatch):
        setup = Toolkits.owner_of("show_full_disk_access_setting")
        assert "show_full_disk_access_setting" not in setup.offered_commands()
        assert set(setup.commands) - set(setup.offered_commands()) == Toolkits.MAC_ONLY_COMMANDS
        monkeypatch.setattr(Platform, "current", lambda: Platform.MAC)
        assert setup.offered_commands() == setup.commands


@pytest.mark.unit
class TestMacOnlyIntegrationInTheSetupCommands:
    def test_list_integrations_says_it_needs_macos_and_offers_no_enable(self, commands):
        monitor = Mock()
        monitor.describe_all.return_value = _statuses()
        with patch.object(Commands, "_integration_monitor", return_value=monitor):
            line = commands._run_list_integrations()
        assert "needs macOS" in line and "enable_integration" not in line

    def test_enable_integration_refuses_it_and_leaves_the_config_alone(self, commands):
        monitor = Mock()
        monitor.get_all_statuses.return_value = _statuses()
        with patch.object(Commands, "_integration_monitor", return_value=monitor), \
                patch("assistant.commands.Commands.ConfigStore.enable_integration") as write:
            answer = commands._run_enable_integration("photos")
        write.assert_not_called()
        monitor.restart.assert_not_called()
        assert "only on a Mac" in answer

    def test_disabling_it_still_takes_it_off_the_list(self, commands):
        """A config copied from a Mac lists it: taking it out must work on the server that cannot run it."""
        monitor = Mock()
        monitor.get_all_statuses.return_value = _statuses()
        monitor.restart.return_value = {"restartable": True, "status": _statuses()[0]}
        with patch.object(Commands, "_integration_monitor", return_value=monitor), \
                patch("assistant.commands.Commands.ConfigStore.disable_integration", return_value="Photos disabled") as write:
            commands._run_disable_integration("Photos")
        write.assert_called_once_with("Photos")


@pytest.mark.unit
class TestMacOnlyConfig:
    @pytest.mark.parametrize("key", ["local_outposts.enabled", "local_outposts.location",
                                     "local_outposts.exclude_mics[0]"])
    def test_the_servers_own_audio_is_not_configured_on_linux(self, commands, linux, key):
        with patch.object(commands, "background_command") as write:
            answer = commands.cmd_config_set(key, "kitchen")
        write.assert_not_called()
        assert "only on a Mac" in answer

    def test_the_watchdog_key_answers_that_systemd_restarts_the_server(self, commands, linux):
        with patch.object(commands, "background_command") as write:
            answer = commands.cmd_config_set(Watchdog.KEY, "true")
        write.assert_not_called()
        assert "systemd" in answer

    def test_a_watchdog_key_taken_out_on_linux_is_not_answered_with_the_macs_launchd_agent(self, linux):
        """config_delete takes the key out of a config copied from a Mac, and its answer ends
        with the key's hint."""
        from assistant.commands.Commands import Commands
        hint = Commands._restart_hint(Watchdog.KEY)
        assert "systemd" in hint and "launchd agent" not in hint.split("systemd")[1]

    @pytest.mark.parametrize("key", ["local_outposts.location", Watchdog.KEY, "owner_name"])
    def test_on_a_mac_the_same_keys_are_written(self, commands, mac, key):
        with patch.object(commands, "background_command") as write:
            assert commands.cmd_config_set(key, "true") is None
        write.assert_called_once()

    def test_a_key_that_means_the_same_everywhere_is_written_on_linux(self, commands, linux):
        with patch.object(commands, "background_command") as write:
            commands.cmd_config_set("owner_name", "Marcus")
        write.assert_called_once()
