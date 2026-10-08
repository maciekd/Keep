"""FirstRun: a young install, or one whose user is running setup commands, is in the FTUE —
the setup toolkit preloaded and the integrations' state in the prompt; the brief greets an
install with fewer than three integrations."""
import json
from unittest.mock import patch

import pytest

from assistant import FirstRun
from common.ConfigManager import config


def _config(overrides: dict):
    real_get = config.get
    return patch.object(config, 'get', side_effect=lambda key, default=None: overrides.get(key, real_get(key, default)))


@pytest.mark.unit
class TestBriefDue:
    def test_no_enabled_integration_opens_the_brief(self):
        with _config({'enabled_integrations': []}):
            assert FirstRun.brief_due() is True

    def test_two_enabled_integrations_still_do(self):
        with _config({'enabled_integrations': ['iMessage', 'iCloud']}):
            assert FirstRun.brief_due() is True

    def test_three_enabled_integrations_retire_it(self):
        with _config({'enabled_integrations': ['iMessage', 'iCloud', 'Photos']}):
            assert FirstRun.brief_due() is False


DAY = 86400
NOW = 1_800_000_000.0


@pytest.mark.unit
class TestInFtue:
    """Setup is in progress while fewer than FTUE_INTEGRATION_THRESHOLD integrations are
    enabled, or while the install is younger than FTUE_INSTALL_DAYS and a setup command ran in
    the last FTUE_SETUP_GRACE_HOURS."""

    ENOUGH = ['a', 'b', 'c']

    def _paths(self, tmp_path, installed_days_ago=None, setup_hours_ago=None):
        state, activity = tmp_path / "first_run.json", tmp_path / "setup_activity.json"
        if installed_days_ago is not None:
            FirstRun.State.fresh(now=NOW - installed_days_ago * DAY).save(state)
        if setup_hours_ago is not None:
            FirstRun.note_setup_activity(now=NOW - setup_hours_ago * 3600, path=activity)
        return dict(now=NOW, state_path=state, activity_path=activity)

    def test_too_few_integrations_is_setup_whatever_the_age_and_activity(self, tmp_path):
        with _config({'enabled_integrations': ['a', 'b']}):
            assert FirstRun.in_ftue(**self._paths(tmp_path, installed_days_ago=400)) is True
            assert FirstRun.in_ftue(**self._paths(tmp_path)) is True

    def test_a_young_install_with_recent_setup_activity_is_in_setup_whatever_is_enabled(self, tmp_path):
        with _config({'enabled_integrations': ['a', 'b', 'c', 'd', 'e', 'f']}):
            assert FirstRun.in_ftue(**self._paths(tmp_path, installed_days_ago=0.1, setup_hours_ago=1)) is True

    def test_an_install_a_few_days_old_still_is_while_the_user_keeps_at_it(self, tmp_path):
        with _config({'enabled_integrations': self.ENOUGH}):
            days, hours = FirstRun.FTUE_INSTALL_DAYS - 1, FirstRun.FTUE_SETUP_GRACE_HOURS - 1
            assert FirstRun.in_ftue(**self._paths(tmp_path, installed_days_ago=days, setup_hours_ago=hours)) is True

    def test_a_young_install_that_has_gone_quiet_is_not(self, tmp_path):
        with _config({'enabled_integrations': self.ENOUGH}):
            hours = FirstRun.FTUE_SETUP_GRACE_HOURS + 1
            assert FirstRun.in_ftue(**self._paths(tmp_path, installed_days_ago=3, setup_hours_ago=hours)) is False
            assert FirstRun.in_ftue(**self._paths(tmp_path, installed_days_ago=3)) is False

    def test_an_old_install_is_not_whatever_the_setup_activity(self, tmp_path):
        with _config({'enabled_integrations': self.ENOUGH}):
            days = FirstRun.FTUE_INSTALL_DAYS + 1
            assert FirstRun.in_ftue(**self._paths(tmp_path, installed_days_ago=days, setup_hours_ago=1)) is False

    def test_an_established_install_without_a_state_file_counts_as_old(self, tmp_path):
        with _config({'enabled_integrations': self.ENOUGH}):
            assert FirstRun.in_ftue(**self._paths(tmp_path, setup_hours_ago=1)) is False

    @pytest.mark.parametrize("enabled, installed_days_ago, setup_hours_ago", [
        (['a', 'b'], None, None),                                              # too few integrations
        (ENOUGH, None, 1),                                                     # established, no state file
        (ENOUGH, 1, 1),                                                        # young and active
        (ENOUGH, 1, FirstRun.FTUE_SETUP_GRACE_HOURS + 1),                      # young but quiet
    ])
    def test_the_pollers_call_traces_nothing_on_any_branch_and_keeps_the_verdict(
            self, tmp_path, enabled, installed_days_ago, setup_hours_ago):
        """The integrations section polls in_ftue every few seconds and logs the verdict itself
        when it changes; the quiet call must reach the same verdict without a trace line."""
        paths = self._paths(tmp_path, installed_days_ago=installed_days_ago, setup_hours_ago=setup_hours_ago)
        with _config({'enabled_integrations': enabled}):
            loud = FirstRun.in_ftue(**paths)
            with patch.object(FirstRun, 'trace') as trace:
                quiet = FirstRun.in_ftue(**paths, verbose=False)
        assert quiet is loud
        trace.assert_not_called()

    def test_the_activity_stamp_is_the_time_the_command_ran(self, tmp_path):
        path = tmp_path / "setup_activity.json"
        FirstRun.note_setup_activity(now=NOW, path=path)
        assert FirstRun.last_setup_activity(path) == NOW

    def test_an_unreadable_stamp_counts_as_no_activity(self, tmp_path):
        path = tmp_path / "setup_activity.json"
        path.write_text("not json")
        assert FirstRun.last_setup_activity(path) is None
        path.write_text(json.dumps({"something_else": 1}))
        assert FirstRun.last_setup_activity(path) is None

    def test_the_setup_toolkit_is_preloaded_only_while_setup_is_in_progress(self):
        from assistant.commands import Toolkits
        with patch.object(FirstRun, "in_ftue", return_value=True):
            assert FirstRun.preload_toolkits() == ("setup",)
        with patch.object(FirstRun, "in_ftue", return_value=False):
            assert FirstRun.preload_toolkits() == ()
        assert Toolkits.by_name("setup") is not None


@pytest.mark.unit
class TestBrief:
    def test_names_the_owner_when_known(self):
        with _config({'owner_name': 'Anna'}):
            text = FirstRun.brief()
        assert "You are talking to Anna." in text
        assert "owner_name is not set" not in text

    def test_asks_for_the_owner_name_when_not_set(self):
        with _config({'owner_name': 'user'}):
            text = FirstRun.brief()
        assert 'config_set("owner_name", ...)' in text
        assert "You are talking to" not in text
