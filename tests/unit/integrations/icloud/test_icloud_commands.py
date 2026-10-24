from unittest.mock import Mock, patch

import pytest

from integrations.icloud.ICloudIntegration import iCloudAPI


@pytest.fixture
def icloud_api():
    with patch.object(iCloudAPI, "get_instance", return_value=Mock()) as get_instance:
        yield get_instance.return_value


class TestCmdIcloudDevices:
    def test_lists_ids_slots_and_the_next_index(self, commands_instance, icloud_api):
        icloud_api.get_icloud_devices.return_value = [
            {"id": "uuid-dad", "name": "Dad's iPhone", "model": "iPhone 15", "class": "iPhone", "configured_as": "dad", "slot": 0},
            {"id": "uuid-anna", "name": "Anna's iPad", "model": "", "class": "iPad", "configured_as": None, "slot": None},
        ]
        icloud_api.configured_device_count.return_value = 1
        text = commands_instance._owner("icloud_devices")._run_icloud_devices()
        lines = text.splitlines()
        assert lines[0] == "Dad's iPhone (iPhone 15) — id uuid-dad — configured as 'dad' (icloud.devices[0])"
        assert lines[1] == "Anna's iPad — id uuid-anna — not configured"
        assert lines[-1] == "icloud.devices has 1 entry; the next one is icloud.devices[1] (keys id and display_name)."

    def test_empty_listing_points_at_the_login(self, commands_instance, icloud_api):
        icloud_api.get_icloud_devices.return_value = []
        assert "login may be incomplete" in commands_instance._owner("icloud_devices")._run_icloud_devices()

    def test_the_command_lets_the_listing_sign_in_for_find_my(self, commands_instance, icloud_api):
        """The TUI's table never signs in; the command, asked for by the user, may."""
        icloud_api.get_icloud_devices.return_value = []
        commands_instance._owner("icloud_devices")._run_icloud_devices()
        assert icloud_api.get_icloud_devices.call_args.kwargs.get("sign_in") is True


class TestCmdIcloudUnlocked:
    def test_lifts_the_park_as_the_user_and_reports_the_status(self, commands_instance, icloud_api):
        icloud_api.name = "iCloud"
        icloud_api.get_auth_status.return_value = {"status": "authenticated", "message": "Connected to iCloud"}
        owner = commands_instance._owner("icloud_unlocked")
        with patch.object(type(owner.commands), "integration_monitor", return_value=Mock()):
            text = owner._run_icloud_unlocked()
        icloud_api.refresh_connection.assert_called_once_with(by_user=True)
        assert "authenticated" in text and "Connected to iCloud" in text


class TestCmdIcloudLoginCode:
    def test_the_code_goes_to_the_background_unquoted(self, commands_instance, icloud_api):
        from assistant.commands.Commands import Commands
        with patch.object(commands_instance, "background_command") as bg:
            assert commands_instance.run("icloud_login_code", ['"123456"']) == ""
        bg.assert_called_once_with(commands_instance._owner("icloud_login_code")._run_icloud_login_code, "123456",
                                   lane=Commands.CONFIG_LANE)

    def test_without_a_code_it_refuses(self, commands_instance, icloud_api):
        with patch.object(commands_instance, "background_command") as bg:
            assert commands_instance.run("icloud_login_code", [""]).startswith("icloud_login_code failed")
        bg.assert_not_called()

    def test_the_background_run_submits_the_code_and_relays_the_outcome(self, commands_instance, icloud_api):
        icloud_api.submit_user_code.return_value = "iCloud signed in — Apple accepted the code."
        text = commands_instance._owner("icloud_login_code")._run_icloud_login_code("123456")
        icloud_api.submit_user_code.assert_called_once_with("123456")
        assert "accepted" in text


class TestCmdFindPhone:
    """`find_phone` answers at once and rings in the background, where a sign-in iCloud wants
    first is waited out; the LLM is told the ping is on its way, and a ring that did not happen
    reaches the user as a system note (2026-10-24: the command returned nothing, the ring was
    refused in its thread, and the assistant said "I've pinged it")."""

    def test_answers_at_once_and_hands_the_ring_to_the_background(self, commands_instance, icloud_api):
        icloud_api.device_label.return_value = "Charlotte's iPhone"
        with patch.object(commands_instance, "fire_and_forget") as fire:
            text = commands_instance.run("find_phone", ["charlotte"])
        assert "Charlotte's iPhone" in text and "on its way" in text
        icloud_api.play_sound.assert_not_called()
        ring, name = fire.call_args.args
        ring(name)
        icloud_api.play_sound.assert_called_once_with("charlotte")

    def test_an_unknown_device_is_refused_before_anything_runs(self, commands_instance, icloud_api):
        from integrations.icloud.ICloudIntegration import iCloudConfigurationError
        icloud_api.device_label.side_effect = iCloudConfigurationError("No iCloud device matches 'bob'")
        with patch.object(commands_instance, "fire_and_forget") as fire:
            text = commands_instance.run("find_phone", ["bob"])
        assert "No iCloud device matches" in text
        fire.assert_not_called()

    def test_a_ring_that_fails_is_a_system_note(self, commands_instance, icloud_api):
        from integrations.icloud.ICloudIntegration import iCloudAuthenticationError
        icloud_api.device_label.return_value = "Charlotte's iPhone"
        icloud_api.play_sound.side_effect = iCloudAuthenticationError("still waiting for its 2FA code")
        with patch.object(commands_instance, "fire_and_forget", side_effect=lambda fn, p: fn(p)), \
             patch("integrations.icloud.ICloudCommands.system_note") as note:
            commands_instance.run("find_phone", ["charlotte"])
        source, text = note.call_args.args
        assert source == "iCloud" and "Charlotte's iPhone" in text and "2FA" in text
