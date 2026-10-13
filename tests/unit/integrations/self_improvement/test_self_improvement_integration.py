import json
import subprocess
from unittest.mock import patch, Mock

import pytest

from assistant.agents import CodeQualityMaintainer as cqm
from assistant.agents.CodeQualityMaintainer import (
    claude_status, claude_path, ClaudeStatus, CodeQualityMaintainer, ClaudeNotInstalledError,
    NOT_INSTALLED_SHORT, NOT_LOGGED_IN_SHORT, DISABLED_MESSAGE, NOT_INSTALLED_MESSAGE,
)
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

    @patch("integrations.self_improvement.SelfImprovementIntegration.claude_status", return_value=ClaudeStatus(None, False, NOT_INSTALLED_SHORT))
    def test_missing_cli_is_the_error(self, _status):
        with pytest.raises(RuntimeError, match="CLI not found"):
            SelfImprovementIntegration().check_health()

    @patch("integrations.self_improvement.SelfImprovementIntegration.claude_status", return_value=ClaudeStatus("/opt/bin/claude", False, NOT_LOGGED_IN_SHORT))
    def test_monitor_shows_the_login_hint(self, _status):
        monitor = IntegrationMonitor.get_instance()
        integration = SelfImprovementIntegration.get_instance()
        monitor.register(integration)
        assert monitor.check_now("Self Improvement") == IntegrationState.WARNING
        assert integration.last_error == NOT_LOGGED_IN_SHORT

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
        with patch.object(CodeQualityMaintainer, "_run_subprocess"), \
                patch.object(CodeQualityMaintainer, "_github_unavailable", return_value=None), \
                patch.object(CodeQualityMaintainer, "_version", return_value="v2026.10.1"), \
                patch.object(CodeQualityMaintainer, "_trial_instructions", return_value=""):
            assert CodeQualityMaintainer.improve_code("fix it") == NOT_INSTALLED_MESSAGE
