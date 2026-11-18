import json
import subprocess
from unittest.mock import patch, Mock

import pytest

from assistant.agents import CodeQualityMaintainer as cqm
from assistant.agents.CodeQualityMaintainer import (
    claude_status, claude_path, ClaudeStatus, CodeQualityMaintainer, ClaudeNotInstalledError, GithubAccess,
    NOT_INSTALLED_SHORT, NOT_LOGGED_IN_SHORT, DISABLED_MESSAGE, NOT_INSTALLED_MESSAGE,
)
from assistant.agents.BuildPreflight import BuildPreflight, Unmet, CREDITS_AGE_FOR_HEALTH
from assistant.integration.Integration import IntegrationState
from assistant.integration.IntegrationMonitor import IntegrationMonitor
from integrations.self_improvement.SelfImprovementIntegration import SelfImprovementIntegration
from common import ConfigSchema

LOGGED_IN = json.dumps({"loggedIn": True, "authMethod": "claude.ai", "email": "me@example.com",
                        "subscriptionType": "max"})
LOGGED_OUT = json.dumps({"loggedIn": False})


def _completed(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


@pytest.fixture
def cli_at():
    with patch.object(cqm, "claude_path", return_value="/opt/bin/claude"):
        yield


@pytest.mark.unit
class TestClaudePath:
    @patch.object(cqm.os, "access", return_value=True)
    @patch.object(cqm.os.path, "isfile", return_value=True)
    @patch.object(cqm.shutil, "which", return_value=None)
    def test_installer_location_when_not_on_path(self, _which, _isfile, _access):
        assert claude_path() == cqm.CLAUDE_FALLBACK_PATH

    @patch.object(cqm.os.path, "isfile", return_value=False)
    @patch.object(cqm.shutil, "which", return_value=None)
    def test_none_when_missing(self, _which, _isfile):
        assert claude_path() is None


@pytest.mark.unit
class TestClaudeStatus:
    @patch.object(cqm, "claude_path", return_value=None)
    def test_not_installed(self, _path):
        status = claude_status()
        assert status == ClaudeStatus(None, False, NOT_INSTALLED_SHORT)
        assert not status.ok

    @patch.object(cqm.subprocess, "run", return_value=_completed(LOGGED_IN))
    def test_logged_in(self, run, cli_at):
        status = claude_status()
        assert status.ok
        assert status.path == "/opt/bin/claude"
        assert status.detail == "logged in as me@example.com (max)"
        cmd = run.call_args[0][0]
        assert cmd == ["/opt/bin/claude", "auth", "status", "--json"]
        assert run.call_args[1]["timeout"] == cqm.AUTH_STATUS_TIMEOUT
        assert run.call_args[1]["stdin"] is subprocess.DEVNULL

    @patch.object(cqm, "claude_api_key", return_value=None)
    @patch.object(cqm.subprocess, "run", return_value=_completed(LOGGED_OUT))
    def test_logged_out_without_a_key(self, _run, _key, cli_at):
        status = claude_status()
        assert not status.ok
        assert status.path == "/opt/bin/claude"
        assert status.detail == NOT_LOGGED_IN_SHORT
        assert not status.api_key

    @patch.object(cqm, "claude_api_key", return_value="sk-ant-test-key")
    @patch.object(cqm.subprocess, "run", return_value=_completed(LOGGED_OUT))
    def test_logged_out_with_a_key_stands_in(self, run, _key, cli_at):
        status = claude_status()
        assert status.ok and status.api_key
        assert "api_keys.anthropic" in status.detail
        assert run.call_args[1].get("env") is None   # the check runs without the key: it reports the login
        assert status.env()[cqm.API_KEY_ENV] == "sk-ant-test-key"

    @patch.object(cqm, "claude_api_key", return_value="sk-ant-test-key")
    @patch.object(cqm.subprocess, "run", return_value=_completed(LOGGED_IN))
    def test_login_wins_over_the_key(self, _run, _key, cli_at):
        status = claude_status()
        assert status.ok and not status.api_key
        assert status.env() is None

    def test_placeholder_key_does_not_count(self):
        config = Mock()
        config.missing_required.return_value = ["api_keys.anthropic"]
        with patch.object(cqm, "ConfigManager", return_value=config):
            assert cqm.claude_api_key() is None
        config.missing_required.return_value = []
        config.get.return_value = "sk-ant-test-key"
        with patch.object(cqm, "ConfigManager", return_value=config):
            assert cqm.claude_api_key() == "sk-ant-test-key"

    @patch.object(cqm.subprocess, "run", return_value=_completed("", "error: unknown command 'auth'", 1))
    def test_no_json_reports_the_output(self, _run, cli_at):
        status = claude_status()
        assert not status.ok
        assert "exit 1" in status.detail and "unknown command" in status.detail

    @patch.object(cqm.subprocess, "run", side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=30))
    def test_timeout(self, _run, cli_at):
        status = claude_status()
        assert not status.ok
        assert "timed out" in status.detail

    @patch.object(cqm.subprocess, "run", side_effect=OSError("exec format error"))
    def test_cannot_run(self, _run, cli_at):
        status = claude_status()
        assert not status.ok
        assert "exec format error" in status.detail


@pytest.mark.unit
class TestSelfImprovementIntegration:
    def setup_method(self):
        SelfImprovementIntegration.reset_instance()
        IntegrationMonitor.reset_instance()

    def teardown_method(self):
        SelfImprovementIntegration.reset_instance()
        IntegrationMonitor.reset_instance()

    NO_CLI = Unmet("claude", NOT_INSTALLED_SHORT)
    NO_LOGIN = Unmet("claude", NOT_LOGGED_IN_SHORT)
    NO_USAGE = Unmet("credits", "Claude Code cannot run the model: out of usage credits.")
    NO_GH = Unmet("github", "The GitHub CLI (gh) is not logged in.")
    NO_AUTHOR = Unmet("git_author", "git has no author on the server machine.")

    def _checked(self, *unmet):
        """On a machine that lacks *unmet*: the integration's state after each of as many checks
        as turn a failure into an error, its health text, and the preflight as it was asked."""
        monitor = IntegrationMonitor.get_instance()
        integration = SelfImprovementIntegration.get_instance()
        monitor.register(integration)
        with patch.object(BuildPreflight, "check", return_value=BuildPreflight(tuple(unmet))) as check:
            states = [monitor.check_now("Self Improvement") for _ in range(IntegrationMonitor.FAILURE_THRESHOLD)]
        return states, integration.last_error or "", check

    def test_missing_cli_is_the_error(self):
        with patch.object(BuildPreflight, "check", return_value=BuildPreflight((self.NO_CLI,))):
            with pytest.raises(RuntimeError, match="CLI not found"):
                SelfImprovementIntegration().check_health()

    def test_monitor_shows_the_login_hint(self):
        states, said, _ = self._checked(self.NO_LOGIN)
        assert states[0] == IntegrationState.WARNING
        assert said == NOT_LOGGED_IN_SHORT

    def test_everything_a_build_lacks_is_in_the_one_health_text(self):
        """Enabling it gave "ok" on a machine that then lost builds to five gaps, one at a time."""
        states, said, _ = self._checked(self.NO_USAGE, self.NO_GH, self.NO_AUTHOR)
        assert all(item.text in said for item in (self.NO_USAGE, self.NO_GH, self.NO_AUTHOR))
        assert states[-1] == IntegrationState.ERROR      # without usage nothing runs, an analysis included

    def test_what_only_a_build_needs_is_a_warning_that_never_becomes_an_error(self):
        """Without gh, or an author for the commits, the analyses and the issue drafts still work."""
        states, said, _ = self._checked(self.NO_GH, self.NO_AUTHOR)
        assert set(states) == {IntegrationState.WARNING}
        assert self.NO_GH.text in said and self.NO_AUTHOR.text in said

    def test_the_health_check_asks_for_usage_once_a_day_and_not_at_every_check(self):
        """The check runs every 15 minutes around the clock and a test request costs about a cent."""
        _, _, check = self._checked()
        ages = {c.kwargs["credits_max_age"] for c in check.call_args_list}
        assert ages == {CREDITS_AGE_FOR_HEALTH}
        assert CREDITS_AGE_FOR_HEALTH >= 24 * 60 * 60

    def test_coming_up_asks_for_usage_anew(self):
        """Setting up is when a missing credit matters most: a pass from before the integration
        was switched on, or restarted, is not believed."""
        from assistant.agents import BuildPreflight as preflight
        preflight._credits_passed[("/opt/bin/claude", False, "claude-fable-5-2")] = 1000.0
        SelfImprovementIntegration.reset_instance()
        SelfImprovementIntegration.get_instance()
        assert not preflight._credits_passed

    def test_the_health_state_carries_the_few_words_and_not_the_remedy(self):
        """The state is in list_integrations and can be in the prompt; the remedy is the refused build's to give."""
        gh = Unmet("github", "The GitHub CLI (gh) is not logged in. Run `gh auth login` on the server machine.",
                   brief="there is no GitHub login")
        _, said, _ = self._checked(gh, self.NO_AUTHOR)
        assert "there is no GitHub login" in said and "gh auth login" not in said

    def test_example_config_documents_the_setup(self):
        assert "self_improvement" in ConfigSchema.sections()
        assert ConfigSchema.leaves("self_improvement") == ["self_improvement.nightly_runs"]   # the CLI setup is by hand; the one key is the nightly switch
        text = ConfigSchema.section_text("self_improvement")
        assert "claude.ai/install.sh" in text and "/login" in text and "api_keys.anthropic" in text
        assert "Self Improvement" in ConfigSchema._lines()[0:12].__str__()


@pytest.mark.unit
class TestGating:
    def test_disabled_reason(self):
        with patch.object(SelfImprovementIntegration, "enabled", return_value=False):
            assert CodeQualityMaintainer.unavailable_reason() == DISABLED_MESSAGE
        with patch.object(SelfImprovementIntegration, "enabled", return_value=True):
            assert CodeQualityMaintainer.unavailable_reason() is None

    @patch.object(SelfImprovementIntegration, "enabled", return_value=False)
    def test_nightly_run_skipped_while_disabled(self, _enabled):
        maintainer = CodeQualityMaintainer()
        with patch.object(CodeQualityMaintainer, "_run_subprocess") as run, \
                patch.object(CodeQualityMaintainer, "_run_claude") as claude:
            maintainer._execute_nightly_run()
        run.assert_not_called()
        claude.assert_not_called()

    @patch.object(cqm, "claude_path", return_value=None)
    def test_run_claude_without_a_binary(self, _path):
        with pytest.raises(ClaudeNotInstalledError):
            CodeQualityMaintainer._run_claude("prompt", "label")

    @patch.object(cqm, "claude_path", return_value=None)
    def test_analyze_code_reports_the_missing_cli(self, _path):
        assert CodeQualityMaintainer.analyze_code("why?") == NOT_INSTALLED_MESSAGE

    @patch.object(cqm, "claude_path", return_value=None)
    def test_improve_code_reports_the_missing_cli(self, _path):
        pusher = GithubAccess(login="marcus", repo="marcus/keep", permission="WRITE")
        # The branch is a name: read through the mocked git it was a Mock, the job's record of it
        # never matched on the way back and was left for the next test as a job cut short.
        with patch.object(CodeQualityMaintainer, "_run_subprocess"), \
                patch.object(CodeQualityMaintainer, "_current_branch", return_value="release"), \
                patch.object(CodeQualityMaintainer, "_github_access", return_value=(pusher, None)), \
                patch.object(CodeQualityMaintainer, "_version", return_value="v2026.10.1"), \
                patch.object(CodeQualityMaintainer, "_trial_instructions", return_value=""):
            assert CodeQualityMaintainer.improve_code("fix it") == (NOT_INSTALLED_MESSAGE, None)


GB = 1024 ** 3


@pytest.mark.unit
class TestBuildMemory:
    """Too little memory for a build is said, never refused and never a health state."""

    @staticmethod
    def _on(ram_gb: float, swap_gb: float = 0):
        return patch("assistant.agents.CodeQualityMaintainer.machine_memory",
                     return_value=(int(ram_gb * GB), int(swap_gb * GB)))

    @pytest.mark.parametrize("ram_gb, swap_gb, said", [
        (1.9, 0, ("2 GB", "no swap")),          # the container the first build was killed in
        (3.8, 0, ("4 GB", "no swap")),          # a machine sold as 4 GB reports less
        (5.8, 1, ("6 GB", "1 GB of swap")),
    ])
    def test_a_small_machine_is_told_what_it_has(self, ram_gb, swap_gb, said):
        with self._on(ram_gb, swap_gb):
            warning = cqm.build_memory_warning()
        assert all(fact in warning for fact in said)

    @pytest.mark.parametrize("ram_gb, swap_gb", [(7.7, 0), (3.8, 4), (64, 0)])
    def test_enough_memory_or_swap_to_make_it_up_is_no_warning(self, ram_gb, swap_gb):
        with self._on(ram_gb, swap_gb):
            assert cqm.build_memory_warning() is None

    def test_memory_that_cannot_be_read_is_no_warning(self):
        with patch("assistant.agents.CodeQualityMaintainer.machine_memory", side_effect=OSError("no /proc")):
            assert cqm.build_memory_warning() is None

    def test_a_small_machine_is_a_note_on_enabling_and_a_healthy_integration(self):
        """A warning state would be shown at every check and ask for what no restart gives."""
        SelfImprovementIntegration.reset_instance()
        try:
            integration = SelfImprovementIntegration()
            small = BuildPreflight((Unmet("memory", "This machine has 4 GB of memory.", blocks=False),))
            with self._on(3.8), patch.object(BuildPreflight, "check", return_value=small):
                assert "4 GB" in integration.enable_note()
                assert integration.check_health() is True
            assert integration.enable_note() is None      # the conftest's roomy machine
        finally:
            SelfImprovementIntegration.reset_instance()

    def test_the_installer_warns_under_the_same_figure(self):
        """Contract between install.sh (BUILD_RAM_GB) and the server: one figure, told twice."""
        import re
        from common.Paths import AISERVER_DIR
        figure = re.search(r"^BUILD_RAM_GB=(\d+)", (AISERVER_DIR / "install.sh").read_text(), re.M)
        assert figure and int(figure.group(1)) == cqm.BUILD_MEMORY_GB
