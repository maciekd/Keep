from datetime import date, datetime
from unittest.mock import patch, Mock, MagicMock, call

import pytest

from common.ConfigManager import ConfigManager

import json
import os
import subprocess

from assistant.agents.CodeQualityMaintainer import CodeQualityMaintainer, WORKING_DIR, AISERVER_DIR, CLAUDE_PATH, GIT_TIMEOUT, GENERATE_TESTS_PROMPT, SLOW_TEST_MS, TEST_HEALTH_RUNS, ClaudeNotLoggedInError, NOT_LOGGED_IN_MESSAGE, NIGHTLY_OFF_MESSAGE, DISABLED_MESSAGE

_REAL_NIGHTLY_SKIP_REASON = CodeQualityMaintainer.nightly_skip_reason   # the autouse fixture below patches it


@pytest.fixture(autouse=True)
def self_improvement_enabled():
    """The jobs run only while the Self Improvement integration is enabled and the scheduled ones
    only with self_improvement.nightly_runs on; these tests are about the jobs themselves."""
    with patch.object(CodeQualityMaintainer, "unavailable_reason", return_value=None), \
            patch.object(CodeQualityMaintainer, "nightly_skip_reason", return_value=None):
        yield


@pytest.fixture(autouse=True)
def claude_installed():
    with patch("assistant.agents.CodeQualityMaintainer.claude_path", return_value=CLAUDE_PATH):
        yield


@pytest.fixture
def maintainer():
    return CodeQualityMaintainer()


@pytest.mark.unit
class TestScheduling:
    @patch('assistant.agents.CodeQualityMaintainer.time.sleep')
    @patch('assistant.agents.CodeQualityMaintainer.datetime')
    def test_triggers_at_run_hour(self, mock_dt, mock_sleep, maintainer):
        mock_dt.now.return_value = datetime(2026, 2, 19, 3, 0, 0)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        call_count = 0

        def stop_after_trigger(*args):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                maintainer._running = False

        mock_sleep.side_effect = stop_after_trigger

        with patch.object(maintainer, '_execute_nightly_run') as mock_run:
            maintainer._running = True
            maintainer._run()
            mock_run.assert_called_once()

    @patch('assistant.agents.CodeQualityMaintainer.time.sleep')
    @patch('assistant.agents.CodeQualityMaintainer.datetime')
    def test_skips_if_already_ran_today(self, mock_dt, mock_sleep, maintainer):
        today = date(2026, 2, 19)
        mock_dt.now.return_value = datetime(2026, 2, 19, 3, 0, 0)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        maintainer._last_run_date = today
        call_count = 0

        def stop_soon(*args):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                maintainer._running = False

        mock_sleep.side_effect = stop_soon

        with patch.object(maintainer, '_execute_nightly_run') as mock_run:
            maintainer._running = True
            maintainer._run()
            mock_run.assert_not_called()

    @patch('assistant.agents.CodeQualityMaintainer.time.sleep')
    @patch('assistant.agents.CodeQualityMaintainer.datetime')
    def test_skips_wrong_hour(self, mock_dt, mock_sleep, maintainer):
        mock_dt.now.return_value = datetime(2026, 2, 19, 10, 0, 0)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        call_count = 0

        def stop_soon(*args):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                maintainer._running = False

        mock_sleep.side_effect = stop_soon

        with patch.object(maintainer, '_execute_nightly_run') as mock_run:
            maintainer._running = True
            maintainer._run()
            mock_run.assert_not_called()

    @patch('assistant.agents.CodeQualityMaintainer.time.sleep')
    @patch('assistant.agents.CodeQualityMaintainer.datetime')
    def test_sets_last_run_date(self, mock_dt, mock_sleep, maintainer):
        mock_dt.now.return_value = datetime(2026, 2, 19, 3, 0, 0)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        def stop_after_one(*args):
            maintainer._running = False

        mock_sleep.side_effect = stop_after_one

        with patch.object(maintainer, '_execute_nightly_run'):
            maintainer._running = True
            maintainer._run()
            assert maintainer._last_run_date == date(2026, 2, 19)


def _pr(number, login, fork=False):
    return {"number": number, "author": {"login": login}, "isCrossRepository": fork}


@pytest.mark.unit
class TestAutoMergeCandidates:
    """On a public repository anyone can open a test-only PR, and a merged test file is code the
    next unit run executes: only trusted authors' PRs from this repository's own branches count."""

    @staticmethod
    def _gh(prs, login="marcus-bot"):
        def run(cmd, **kwargs):
            if cmd[:3] == ["gh", "api", "user"]:
                return Mock(returncode=0, stdout=login + "\n", stderr="")
            return Mock(returncode=0, stdout=json.dumps(prs), stderr="")
        return run

    @patch('assistant.agents.CodeQualityMaintainer.ConfigManager')
    @patch('assistant.agents.CodeQualityMaintainer.subprocess.run')
    def test_keeps_the_gh_accounts_own_prs_by_default(self, mock_run, mock_cm):
        mock_cm.return_value.get.side_effect = lambda k, default=None: default
        mock_run.side_effect = self._gh([_pr(1, "Marcus-Bot"), _pr(2, "stranger"), _pr(3, "marcus-bot")])
        assert CodeQualityMaintainer._auto_merge_candidates() == [1, 3]

    @patch('assistant.agents.CodeQualityMaintainer.ConfigManager')
    @patch('assistant.agents.CodeQualityMaintainer.subprocess.run')
    def test_a_fork_pr_never_counts_even_from_a_trusted_author(self, mock_run, mock_cm):
        mock_cm.return_value.get.side_effect = lambda k, default=None: default
        mock_run.side_effect = self._gh([_pr(1, "marcus-bot", fork=True), _pr(2, "marcus-bot")])
        assert CodeQualityMaintainer._auto_merge_candidates() == [2]

    @patch('assistant.agents.CodeQualityMaintainer.ConfigManager')
    @patch('assistant.agents.CodeQualityMaintainer.subprocess.run')
    def test_the_configured_authors_replace_the_gh_account(self, mock_run, mock_cm):
        mock_cm.return_value.get.side_effect = \
            lambda k, default=None: {"self_improvement.auto_merge_authors": ["Marcus", "nightly"]}.get(k, default)
        mock_run.side_effect = self._gh([_pr(1, "marcus"), _pr(2, "marcus-bot"), _pr(3, "nightly")])
        assert CodeQualityMaintainer._auto_merge_candidates() == [1, 3]
        assert not [c for c in mock_run.call_args_list if c[0][0][:3] == ["gh", "api", "user"]]

    @patch('assistant.agents.CodeQualityMaintainer.ConfigManager')
    @patch('assistant.agents.CodeQualityMaintainer.subprocess.run')
    def test_nothing_counts_when_the_gh_account_is_unknown(self, mock_run, mock_cm):
        mock_cm.return_value.get.side_effect = lambda k, default=None: default
        mock_run.return_value = Mock(returncode=1, stdout="", stderr="gh: not logged in")
        assert CodeQualityMaintainer._auto_merge_candidates() == []

    @patch.object(CodeQualityMaintainer, '_trusted_pr_authors', return_value={"marcus"})
    @patch('assistant.agents.CodeQualityMaintainer.subprocess.run')
    def test_returns_nothing_when_gh_fails(self, mock_run, mock_trusted):
        mock_run.return_value = Mock(returncode=1, stdout='', stderr="gh: not logged in")
        assert CodeQualityMaintainer._auto_merge_candidates() == []
        mock_run.side_effect = Exception("timeout")
        assert CodeQualityMaintainer._auto_merge_candidates() == []
        mock_run.side_effect = None
        mock_run.return_value = Mock(returncode=0, stdout='', stderr="")
        assert CodeQualityMaintainer._auto_merge_candidates() == []


@pytest.mark.unit
class TestStepAutoMerge:
    """The merge is plain `gh` calls: a PR is merged only when every path it changes is under tests/."""

    PRS = {7: ("test: cover the parser", ["tests/unit/test_parser.py"]),
           8: ("fix: the parser", ["tests/unit/test_parser.py", "src/common/Parser.py"]),
           9: ("test: cover the lexer", ["tests/unit/test_lexer.py", "tests/unit/data/lexer.txt"])}

    @pytest.fixture
    def gh(self):
        """_run_subprocess as a fake git + gh; `fail` holds the PR numbers GitHub refuses to merge."""
        fail = set()

        def run(cmd, label, timeout):
            if cmd[:3] == ["gh", "pr", "merge"] and int(cmd[3]) in fail:
                return Mock(returncode=1, stdout="", stderr="not mergeable: conflicts")
            return Mock(returncode=0, stdout="master\n", stderr="")
        with patch.object(CodeQualityMaintainer, '_run_subprocess', side_effect=run) as mock_sub, \
                patch.object(CodeQualityMaintainer, '_tests_path', return_value="tests/"), \
                patch.object(CodeQualityMaintainer, '_pr_title_and_files', side_effect=lambda n: self.PRS[n]), \
                patch('assistant.agents.CodeQualityMaintainer.ConfigManager') as mock_cm:
            mock_cm.return_value.get.side_effect = lambda k, default=None: default
            mock_sub.fail, mock_sub.config = fail, mock_cm.return_value.get
            yield mock_sub

    @staticmethod
    def _merges(mock_sub):
        return [c[0][0] for c in mock_sub.call_args_list if c[0][0][:3] == ["gh", "pr", "merge"]]

    @patch.object(CodeQualityMaintainer, '_auto_merge_candidates', return_value=[])
    def test_skips_when_no_open_prs(self, mock_candidates, gh, maintainer):
        assert maintainer._step_auto_merge() == ""
        gh.assert_not_called()

    @patch.object(CodeQualityMaintainer, '_auto_merge_candidates', return_value=[7, 8, 9])
    def test_merges_only_the_prs_confined_to_the_tests_directory(self, mock_candidates, gh, maintainer):
        report = maintainer._step_auto_merge()
        assert self._merges(gh) == [["gh", "pr", "merge", "7", "--rebase", "--delete-branch"],
                                    ["gh", "pr", "merge", "9", "--rebase", "--delete-branch"]]
        merged, skipped = report.split("Skipped PRs")
        assert "#7" in merged and "#9" in merged and "#8" not in merged
        assert "#8" in skipped and "src/common/Parser.py" in skipped

    @patch.object(CodeQualityMaintainer, '_auto_merge_candidates', return_value=[7])
    def test_merges_with_the_configured_strategy(self, mock_candidates, gh, maintainer):
        gh.config.side_effect = lambda k, default=None: {"self_improvement.merge_strategy": "squash"}.get(k, default)
        maintainer._step_auto_merge()
        assert self._merges(gh) == [["gh", "pr", "merge", "7", "--squash", "--delete-branch"]]

    @patch.object(CodeQualityMaintainer, '_auto_merge_candidates', return_value=[7])
    def test_a_file_moved_into_tests_from_outside_is_not_merged(self, mock_candidates, gh, maintainer):
        moved = ("test: move the parser", ["tests/unit/Parser.py", "src/common/Parser.py"])
        with patch.object(CodeQualityMaintainer, '_pr_title_and_files', return_value=moved):
            report = maintainer._step_auto_merge()
        assert self._merges(gh) == []
        assert "src/common/Parser.py" in report.split("Skipped PRs")[1]

    @patch.object(CodeQualityMaintainer, '_auto_merge_candidates', return_value=[7, 9])
    def test_a_refused_merge_is_reported_and_the_next_pr_still_merges(self, mock_candidates, gh, maintainer):
        gh.fail.add(7)
        merged, skipped = maintainer._step_auto_merge().split("Skipped PRs")
        assert "#9" in merged and "#7" not in merged
        assert "#7" in skipped and "conflicts" in skipped

    @patch.object(CodeQualityMaintainer, '_auto_merge_candidates', return_value=[7, 9])
    def test_a_pr_that_cannot_be_read_is_not_merged(self, mock_candidates, gh, maintainer):
        with patch.object(CodeQualityMaintainer, '_pr_title_and_files', side_effect=RuntimeError("gh: timeout")):
            report = maintainer._step_auto_merge()
        assert self._merges(gh) == []
        assert "#7" in report and "#9" in report and "gh: timeout" in report

    @patch.object(CodeQualityMaintainer, '_auto_merge_candidates', return_value=[7])
    def test_a_pr_without_changed_files_is_not_merged(self, mock_candidates, gh, maintainer):
        with patch.object(CodeQualityMaintainer, '_pr_title_and_files', return_value=("test: nothing", [])):
            maintainer._step_auto_merge()
        assert self._merges(gh) == []

    @patch.object(CodeQualityMaintainer, '_auto_merge_candidates', return_value=[7])
    def test_restores_the_starting_branch(self, mock_candidates, gh, maintainer):
        maintainer._step_auto_merge()
        assert gh.call_args_list[-1] == call(["git", "checkout", "master"], "git checkout", GIT_TIMEOUT)


@pytest.mark.unit
class TestPrTitleAndFiles:
    @staticmethod
    def _gh(files, changed):
        """subprocess.run as a fake gh: `files` are (filename, previous_filename) pairs."""
        def run(cmd, **kwargs):
            if cmd[:3] == ["gh", "pr", "view"]:
                return Mock(returncode=0, stderr="", stdout=json.dumps(
                    {"title": "test: cover the parser", "changedFiles": changed}))
            return Mock(returncode=0, stderr="", stdout="".join(
                json.dumps({"filename": name, "previous_filename": previous}) + "\n" for name, previous in files))
        return run

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.run')
    def test_returns_the_title_and_every_changed_path(self, mock_run):
        mock_run.side_effect = self._gh([("tests/unit/a.py", None), ("tests/unit/b.py", None)], 2)
        assert CodeQualityMaintainer._pr_title_and_files(7) == \
            ("test: cover the parser", ["tests/unit/a.py", "tests/unit/b.py"])

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.run')
    def test_a_renamed_file_counts_under_its_previous_path_too(self, mock_run):
        """Only the new path is under tests/; the PR still takes src/common/Parser.py away."""
        mock_run.side_effect = self._gh([("tests/unit/Parser.py", "src/common/Parser.py")], 1)
        assert set(CodeQualityMaintainer._pr_title_and_files(7)[1]) == \
            {"tests/unit/Parser.py", "src/common/Parser.py"}

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.run')
    def test_refuses_a_file_list_shorter_than_the_pr(self, mock_run):
        """The paths gh left out could be outside tests/."""
        mock_run.side_effect = self._gh([("tests/unit/a.py", None)], 140)
        with pytest.raises(RuntimeError):
            CodeQualityMaintainer._pr_title_and_files(7)

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.run')
    def test_raises_when_gh_fails(self, mock_run):
        mock_run.return_value = Mock(returncode=1, stdout="", stderr="gh: not found")
        with pytest.raises(RuntimeError, match="gh: not found"):
            CodeQualityMaintainer._pr_title_and_files(7)


@pytest.mark.unit
class TestStepTestHealth:
    @patch.object(CodeQualityMaintainer, '_run_claude', return_value="2 flaky tests fixed")
    @patch('assistant.agents.CodeQualityMaintainer.subprocess.run')
    def test_reads_the_branch_then_claude_then_checks_it_out_again(self, mock_run, mock_claude, maintainer):
        mock_run.return_value = Mock(stdout="release\n", returncode=0)
        assert maintainer._step_test_health() == "2 flaky tests fixed"
        assert [c[0][0] for c in mock_run.call_args_list] == [
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], ["git", "checkout", "release"]]

    @patch.object(CodeQualityMaintainer, '_run_claude', return_value="")
    @patch('assistant.agents.CodeQualityMaintainer.subprocess.run')
    def test_the_prompt_carries_the_limit_and_the_run_count(self, mock_run, mock_claude, maintainer):
        mock_run.return_value = Mock(stdout="master\n", returncode=0)
        maintainer._step_test_health()
        prompt = mock_claude.call_args[0][0]
        assert f"{SLOW_TEST_MS} ms" in prompt and f"{TEST_HEALTH_RUNS} times" in prompt
        assert "{" not in prompt

    @patch.object(CodeQualityMaintainer, '_run_claude', side_effect=Exception("claude timed out"))
    @patch('assistant.agents.CodeQualityMaintainer.subprocess.run')
    def test_a_failed_run_reports_nothing_and_still_restores_the_branch(self, mock_run, mock_claude, maintainer):
        mock_run.return_value = Mock(stdout="master\n", returncode=0)
        assert maintainer._step_test_health() == ""
        assert mock_run.call_args[0][0] == ["git", "checkout", "master"]


@pytest.mark.unit
class TestStepGenerateTests:
    @patch.object(CodeQualityMaintainer, '_run_claude', return_value="")
    @patch('assistant.agents.CodeQualityMaintainer.subprocess.run')
    def test_reads_the_branch_then_claude_then_checks_it_out_again(self, mock_run, mock_claude, maintainer):
        mock_run.return_value = Mock(stdout="release\n", returncode=0)
        maintainer._step_generate_tests()

        assert [c[0][0] for c in mock_run.call_args_list] == [
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], ["git", "checkout", "release"]]
        assert mock_run.call_args[1]["cwd"] == WORKING_DIR
        mock_claude.assert_called_once_with(GENERATE_TESTS_PROMPT, "generate-tests")

    @patch.object(CodeQualityMaintainer, '_run_claude', side_effect=Exception("claude failed"))
    @patch('assistant.agents.CodeQualityMaintainer.subprocess.run')
    def test_returns_empty_on_error(self, mock_run, mock_claude, maintainer):
        mock_run.return_value = Mock(stdout="", returncode=0)
        result = maintainer._step_generate_tests()
        assert result == ""

    @patch.object(CodeQualityMaintainer, '_run_claude', side_effect=Exception("claude timed out"))
    @patch('assistant.agents.CodeQualityMaintainer.subprocess.run')
    def test_checkout_master_runs_on_claude_error(self, mock_run, mock_claude, maintainer):
        mock_run.return_value = Mock(stdout="", returncode=0)
        maintainer._step_generate_tests()
        checkout_calls = [c for c in mock_run.call_args_list if c[0][0] == ["git", "checkout", "master"]]
        assert len(checkout_calls) == 1


@pytest.mark.unit
class TestStepBugAnalysis:
    @patch.object(CodeQualityMaintainer, '_run_claude', side_effect=Exception("claude failed"))
    def test_returns_empty_on_error(self, mock_claude, maintainer):
        result = maintainer._step_bug_analysis()
        assert result == ""


@pytest.mark.unit
class TestAnalyzeCode:
    @patch.object(CodeQualityMaintainer, '_send_email')
    @patch.object(CodeQualityMaintainer, '_run_claude', return_value="")
    def test_returns_fallback_on_no_output(self, mock_claude, mock_email):
        result = CodeQualityMaintainer.analyze_code("some bug")
        assert result == "Self-analysis produced no output."

    @patch.object(CodeQualityMaintainer, '_send_email')
    @patch.object(CodeQualityMaintainer, '_run_claude', side_effect=ClaudeNotLoggedInError())
    def test_explains_missing_login_instead_of_raw_cli_text(self, mock_claude, mock_email):
        result = CodeQualityMaintainer.analyze_code("some bug")
        assert result == NOT_LOGGED_IN_MESSAGE
        mock_email.assert_not_called()

    @patch.object(CodeQualityMaintainer, '_send_email')
    @patch.object(CodeQualityMaintainer, '_run_claude', return_value="## Found a bug\nDetails here")
    def test_emails_report_when_claude_returns_output(self, mock_claude, mock_email):
        CodeQualityMaintainer.analyze_code("some bug")
        mock_email.assert_called_once()
        subject, html = mock_email.call_args[0]
        assert subject == "Keep Self-analysis Report"
        assert "Self-analysis" in html
        assert "Found a bug" in html
        assert "Details here" in html

    @patch.object(CodeQualityMaintainer, '_send_email')
    @patch.object(CodeQualityMaintainer, '_run_claude', return_value="")
    def test_skips_email_when_no_output(self, mock_claude, mock_email):
        CodeQualityMaintainer.analyze_code("some bug")
        mock_email.assert_not_called()

    @patch.object(CodeQualityMaintainer, '_send_email', side_effect=Exception("SMTP boom"))
    @patch.object(CodeQualityMaintainer, '_run_claude', return_value="Root cause: X")
    def test_email_failure_does_not_break_return_value(self, mock_claude, mock_email):
        result = CodeQualityMaintainer.analyze_code("some bug")
        assert result == "Root cause: X"


@pytest.mark.unit
class TestImproveCode:
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    @patch.object(CodeQualityMaintainer, '_run_claude', side_effect=ClaudeNotLoggedInError())
    def test_explains_missing_login(self, mock_claude, mock_sub):
        mock_sub.return_value = Mock(stdout="", returncode=0)
        assert CodeQualityMaintainer.improve_code("some bug") == NOT_LOGGED_IN_MESSAGE
        mock_sub.assert_called_with(["git", "checkout", "master"], "git checkout", GIT_TIMEOUT)

    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    @patch.object(CodeQualityMaintainer, '_run_claude', return_value="")
    def test_returns_fallback_on_no_output(self, mock_claude, mock_sub):
        mock_sub.return_value = Mock(stdout="", returncode=0)
        result = CodeQualityMaintainer.improve_code("some bug")
        assert result == "Self-improvement produced no output."

    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    @patch.object(CodeQualityMaintainer, '_run_claude', return_value="done")
    def test_git_pull_then_claude_then_checkout_of_the_starting_branch(self, mock_claude, mock_sub):
        mock_sub.return_value = Mock(stdout="release\n", returncode=0)
        CodeQualityMaintainer.improve_code("bug")
        assert mock_sub.call_args_list == [
            call(["git", "rev-parse", "--abbrev-ref", "HEAD"], "git rev-parse", GIT_TIMEOUT),
            call(["git", "pull"], "git pull", GIT_TIMEOUT),
            call(["git", "checkout", "release"], "git checkout", GIT_TIMEOUT)]

    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_detached_head_falls_back_to_master(self, mock_sub):
        mock_sub.return_value = Mock(stdout="HEAD\n", returncode=0)
        assert CodeQualityMaintainer._current_branch() == "master"
        mock_sub.side_effect = Exception("no git")
        assert CodeQualityMaintainer._current_branch() == "master"

    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    @patch.object(CodeQualityMaintainer, '_run_claude', side_effect=Exception("claude crashed"))
    def test_checkout_master_runs_on_claude_error(self, mock_claude, mock_sub):
        mock_sub.return_value = Mock(stdout="", returncode=0)
        result = CodeQualityMaintainer.improve_code("bug")
        assert "claude crashed" in result
        checkout_calls = [c for c in mock_sub.call_args_list if c[0][0] == ["git", "checkout", "master"]]
        assert len(checkout_calls) == 1

    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    @patch.object(CodeQualityMaintainer, '_run_claude')
    def test_checkout_master_runs_on_git_pull_error(self, mock_claude, mock_sub):
        def fail_pull_allow_checkout(cmd, label, timeout):
            if cmd == ["git", "pull"]:
                raise Exception("git pull failed")
            return Mock(stdout="", returncode=0)
        mock_sub.side_effect = fail_pull_allow_checkout
        result = CodeQualityMaintainer.improve_code("bug")
        assert "git pull failed" in result
        mock_claude.assert_not_called()
        checkout_calls = [c for c in mock_sub.call_args_list if c[0][0] == ["git", "checkout", "master"]]
        assert len(checkout_calls) == 1


@pytest.mark.unit
class TestRunSubprocess:
    @patch('assistant.agents.CodeQualityMaintainer.subprocess.run')
    def test_skips_none_stdout(self, mock_run):
        mock_run.return_value = Mock(stdout=None, stderr=None)
        with patch('assistant.agents.CodeQualityMaintainer.trace') as mock_trace:
            CodeQualityMaintainer._run_subprocess(["git", "pull"], "git pull", 60)
            mock_trace.assert_not_called()

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.run')
    def test_passes_cwd_and_timeout(self, mock_run):
        mock_run.return_value = Mock(stdout="", stderr="")
        CodeQualityMaintainer._run_subprocess(["git", "pull"], "git pull", 120)
        mock_run.assert_called_once_with(
            ["git", "pull"], cwd=WORKING_DIR, timeout=120,
            capture_output=True, text=True, stdin=subprocess.DEVNULL
        )


def _mock_popen(stdout_lines=None, stderr=""):
    mock_proc = Mock()
    mock_proc.stdout = iter(stdout_lines or [])
    mock_proc.stderr.read.return_value = stderr
    mock_proc.wait.return_value = 0
    return mock_proc


@pytest.mark.unit
class TestRunClaude:
    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    def test_builds_correct_command(self, mock_popen):
        mock_popen.return_value = _mock_popen()
        CodeQualityMaintainer._run_claude("test prompt", "label")
        cmd = mock_popen.call_args[0][0]
        assert cmd == [CLAUDE_PATH, "--permission-mode", "auto", "-p", "test prompt",
                       "--output-format", "stream-json", "--verbose",
                       "--model", "claude-fable-5-1"]
        assert mock_popen.call_args[1]["cwd"] == WORKING_DIR

    @patch('assistant.agents.CodeQualityMaintainer.parse_llm_config', return_value=("ClaudeLLM", "claude-sonnet-5"))
    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    def test_model_configurable_via_config(self, mock_popen, mock_parse):
        mock_popen.return_value = _mock_popen()
        CodeQualityMaintainer._run_claude("test prompt", "label")
        cmd = mock_popen.call_args[0][0]
        assert cmd[-2:] == ["--model", "claude-sonnet-5"]
        mock_parse.assert_called_once_with("code_quality")

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    def test_extracts_result_text(self, mock_popen):
        lines = ['{"type": "assistant", "message": {"content": [{"type": "text", "text": "thinking..."}]}}\n',
                 '{"type": "result", "result": "Final answer"}\n']
        mock_popen.return_value = _mock_popen(lines)
        result = CodeQualityMaintainer._run_claude("prompt", "label")
        assert result == "Final answer"

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    def test_falls_back_to_assistant_text_when_no_result(self, mock_popen):
        lines = ['{"type": "assistant", "message": {"content": [{"type": "text", "text": "analysis output"}]}}\n']
        mock_popen.return_value = _mock_popen(lines)
        result = CodeQualityMaintainer._run_claude("prompt", "label")
        assert result == "analysis output"

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    def test_returns_empty_when_no_result_and_no_assistant_text(self, mock_popen):
        lines = ['{"type": "system", "data": "init"}\n']
        mock_popen.return_value = _mock_popen(lines)
        result = CodeQualityMaintainer._run_claude("prompt", "label")
        assert result == ""

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    def test_raises_when_cli_reports_not_logged_in(self, mock_popen):
        lines = ['{"type": "result", "is_error": true, "result": "Not logged in \u00b7 Please run /login"}\n']
        mock_popen.return_value = _mock_popen(lines)
        with pytest.raises(ClaudeNotLoggedInError) as exc:
            CodeQualityMaintainer._run_claude("prompt", "label")
        assert "not logged in" in str(exc.value).lower()
        assert "SSH" in str(exc.value)

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    def test_raises_when_stderr_reports_invalid_api_key(self, mock_popen):
        proc = _mock_popen([])
        proc.stderr.read.return_value = "Invalid API key · Please run /login"
        mock_popen.return_value = proc
        with pytest.raises(ClaudeNotLoggedInError):
            CodeQualityMaintainer._run_claude("prompt", "label")

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    def test_normal_report_mentioning_login_bug_is_not_treated_as_auth_failure(self, mock_popen):
        lines = ['{"type": "result", "is_error": false, "result": '
                 '"Bug: ConfigManager raises \'Invalid API key\' instead of \'not logged in\' when the key is empty."}\n']
        mock_popen.return_value = _mock_popen(lines)
        assert CodeQualityMaintainer._run_claude("prompt", "label").startswith("Bug: ConfigManager")

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    def test_assistant_text_mentioning_login_without_result_is_not_auth_failure(self, mock_popen):
        lines = ['{"type": "assistant", "message": {"content": [{"type": "text", "text": "Please run /login is the fix."}]}}\n']
        mock_popen.return_value = _mock_popen(lines)
        assert CodeQualityMaintainer._run_claude("prompt", "label") == "Please run /login is the fix."

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    def test_failed_run_without_login_markers_is_returned_for_the_caller_to_judge(self, mock_popen):
        lines = ['{"type": "result", "is_error": true, "result": "Rate limit reached, try again later"}\n']
        mock_popen.return_value = _mock_popen(lines)
        assert CodeQualityMaintainer._run_claude("prompt", "label") == "Rate limit reached, try again later"

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    def test_handles_invalid_json(self, mock_popen):
        lines = ['not json at all\n', '{"type": "result", "result": "ok"}\n']
        mock_popen.return_value = _mock_popen(lines)
        result = CodeQualityMaintainer._run_claude("prompt", "label")
        assert result == "ok"

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    def test_falls_back_to_assistant_text_when_result_empty(self, mock_popen):
        lines = [
            '{"type": "assistant", "message": {"content": [{"type": "text", "text": "## Full analysis here"}]}}\n',
            '{"type": "result", "result": ""}\n',
        ]
        mock_popen.return_value = _mock_popen(lines)
        result = CodeQualityMaintainer._run_claude("prompt", "label")
        assert result == "## Full analysis here"

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    def test_uses_last_assistant_text_not_earlier_ones(self, mock_popen):
        lines = [
            '{"type": "assistant", "message": {"content": [{"type": "text", "text": "early message"}]}}\n',
            '{"type": "assistant", "message": {"content": [{"type": "text", "text": "final analysis"}]}}\n',
            '{"type": "result", "result": ""}\n',
        ]
        mock_popen.return_value = _mock_popen(lines)
        result = CodeQualityMaintainer._run_claude("prompt", "label")
        assert result == "final analysis"

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    def test_assistant_fallback_ignores_thinking_blocks(self, mock_popen):
        lines = [
            '{"type": "assistant", "message": {"content": [{"type": "thinking", "thinking": "Let me think..."}, {"type": "text", "text": "visible output"}]}}\n',
            '{"type": "result", "result": ""}\n',
        ]
        mock_popen.return_value = _mock_popen(lines)
        result = CodeQualityMaintainer._run_claude("prompt", "label")
        assert result == "visible output"

@pytest.mark.unit
class TestFormatStreamLine:
    def test_assistant_text(self):
        obj = {"type": "assistant", "message": {"content": [{"type": "text", "text": "Hello world"}]}}
        assert CodeQualityMaintainer._format_stream_line(obj) == "Hello world"

    def test_assistant_tool_use_skipped(self):
        obj = {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Bash", "input": {"command": "git log"}}
        ]}}
        assert CodeQualityMaintainer._format_stream_line(obj) is None

    def test_assistant_thinking(self):
        obj = {"type": "assistant", "message": {"content": [{"type": "thinking", "thinking": "Let me analyze this"}]}}
        result = CodeQualityMaintainer._format_stream_line(obj)
        assert "(thinking:" in result
        assert "Let me analyze" in result

    def test_user_type_returns_none(self):
        obj = {"type": "user", "message": {"content": [
            {"type": "tool_result", "content": "some output"}
        ]}}
        assert CodeQualityMaintainer._format_stream_line(obj) is None

    def test_result_with_cost(self):
        obj = {"type": "result", "result": "Final report", "cost_usd": 0.1234}
        result = CodeQualityMaintainer._format_stream_line(obj)
        assert "RESULT" in result
        assert "$0.1234" in result
        assert "Final report" in result

    def test_result_empty(self):
        obj = {"type": "result", "result": ""}
        result = CodeQualityMaintainer._format_stream_line(obj)
        assert "(empty)" in result

    def test_assistant_empty_content_returns_none(self):
        obj = {"type": "assistant", "message": {"content": []}}
        assert CodeQualityMaintainer._format_stream_line(obj) is None


@pytest.mark.unit
class TestMarkdownToHtml:
    def test_h1(self):
        assert CodeQualityMaintainer._markdown_to_html("# Title") == "<h1>Title</h1>"

    def test_h2(self):
        assert CodeQualityMaintainer._markdown_to_html("## Section") == "<h2>Section</h2>"

    def test_h3(self):
        assert CodeQualityMaintainer._markdown_to_html("### Subsection") == "<h3>Subsection</h3>"

    def test_bold(self):
        result = CodeQualityMaintainer._markdown_to_html("This is **bold** text")
        assert "<strong>bold</strong>" in result

    def test_inline_code(self):
        result = CodeQualityMaintainer._markdown_to_html("Use `foo()` here")
        assert ">foo()</code>" in result
        assert "color:#6dbf6d" in result

    def test_bold_in_list_item(self):
        result = CodeQualityMaintainer._markdown_to_html("- This is **bold** in a list")
        assert "<strong>bold</strong>" in result

    def test_inline_markup_in_heading(self):
        result = CodeQualityMaintainer._markdown_to_html("## The `foo` function")
        assert ">foo</code>" in result
        assert "<h2>" in result

    def test_long_span_between_backticks_not_matched(self):
        long_text = "a " * 50
        result = CodeQualityMaintainer._inline_markup(f"the `{long_text}` word")
        assert "<code" not in result
        assert f"`{long_text}`" in result

    def test_short_inline_code_matched(self):
        result = CodeQualityMaintainer._inline_markup("`short` and `ok`")
        assert ">short</code>" in result
        assert ">ok</code>" in result

    def test_horizontal_rule(self):
        result = CodeQualityMaintainer._markdown_to_html("above\n---\nbelow")
        assert "<hr " in result

    def test_unordered_list_dash(self):
        result = CodeQualityMaintainer._markdown_to_html("- item one\n- item two")
        assert "<ul>" in result
        assert "item one</li>" in result
        assert "item two</li>" in result
        assert "</ul>" in result

    def test_unordered_list_asterisk(self):
        result = CodeQualityMaintainer._markdown_to_html("* item one\n* item two")
        assert "<ul>" in result
        assert "item one</li>" in result

    def test_code_block(self):
        md = "```\nprint('hello')\n```"
        result = CodeQualityMaintainer._markdown_to_html(md)
        assert "<pre " in result
        assert "<code>" in result
        assert "print(&#x27;hello&#x27;)" in result
        assert "</code></pre>" in result

    def test_paragraph(self):
        result = CodeQualityMaintainer._markdown_to_html("Just some text")
        assert "<p>Just some text</p>" in result

    def test_list_closed_before_heading(self):
        md = "- item\n## Next"
        result = CodeQualityMaintainer._markdown_to_html(md)
        list_end = result.index("</ul>")
        heading_start = result.index("<h2>")
        assert list_end < heading_start

    def test_html_tags_in_text_are_escaped(self):
        result = CodeQualityMaintainer._markdown_to_html("poor support for `<style>` blocks")
        assert "&lt;style&gt;" in result
        assert "<style>" not in result.replace("&lt;style&gt;", "")

    def test_html_tags_in_inline_code_are_escaped(self):
        result = CodeQualityMaintainer._inline_markup("use `<div>` for layout")
        assert "&lt;div&gt;" in result
        assert "<div>" not in result

    def test_code_block_content_is_escaped(self):
        md = "```\n<style>body{color:red}</style>\n```"
        result = CodeQualityMaintainer._markdown_to_html(md)
        assert "&lt;style&gt;" in result
        assert "<style>body" not in result

    def test_table_renders_as_html_table(self):
        md = "| Feature | Commit | Gap |\n|---------|--------|-----|\n| foo | abc123 | none |\n| bar | def456 | one |"
        result = CodeQualityMaintainer._markdown_to_html(md)
        assert "<table " in result
        assert "<thead>" in result
        assert "<th " in result
        assert ">Feature</th>" in result
        assert ">Commit</th>" in result
        assert ">Gap</th>" in result
        assert "<tbody>" in result
        assert ">foo</td>" in result
        assert ">abc123</td>" in result
        assert ">bar</td>" in result
        assert "</table>" in result
        assert "<p>| Feature" not in result

    def test_table_separator_alone_is_not_a_table(self):
        result = CodeQualityMaintainer._markdown_to_html("|---|---|")
        assert "<table" not in result

    def test_table_with_inline_code_in_cell(self):
        md = "| Func | Note |\n|------|------|\n| `foo()` | bar |"
        result = CodeQualityMaintainer._markdown_to_html(md)
        assert ">foo()</code>" in result

    def test_table_closes_open_list(self):
        md = "- item one\n| H1 | H2 |\n|----|----|\n| a | b |"
        result = CodeQualityMaintainer._markdown_to_html(md)
        list_end = result.index("</ul>")
        table_start = result.index("<table")
        assert list_end < table_start

    def test_table_with_alignment_separator(self):
        md = "| L | R |\n|:---|---:|\n| a | b |"
        result = CodeQualityMaintainer._markdown_to_html(md)
        assert "<table " in result
        assert ">a</td>" in result

@pytest.mark.unit
class TestSendEmail:
    @patch('assistant.agents.CodeQualityMaintainer.smtplib.SMTP')
    @patch('assistant.agents.CodeQualityMaintainer.ConfigManager')
    def test_sends_via_the_configured_smtp_server(self, mock_cm_class, mock_smtp_class, maintainer):
        configured = {"email.address": "me@example.com", "email.password": "pw",
                      "email.smtp_host": "smtp.example.com", "email.smtp_port": "2525"}
        mock_cm_class.return_value.get.side_effect = lambda k, default=None: configured.get(k, default)

        CodeQualityMaintainer._send_email("Subject", "<p>Body</p>")

        mock_smtp_class.assert_called_once_with("smtp.example.com", 2525)

    @patch('assistant.agents.CodeQualityMaintainer.smtplib.SMTP')
    @patch('assistant.agents.CodeQualityMaintainer.ConfigManager')
    def test_sends_via_gmail_smtp(self, mock_cm_class, mock_smtp_class, maintainer):
        mock_cfg = mock_cm_class.return_value
        mock_cfg.get.side_effect = lambda k, default=None: {"email.address": "me@gmail.com", "email.password": "pw123"}.get(k, default)
        mock_server = MagicMock()
        mock_smtp_class.return_value.__enter__ = Mock(return_value=mock_server)
        mock_smtp_class.return_value.__exit__ = Mock(return_value=False)

        CodeQualityMaintainer._send_email("Subject", "<p>Body</p>")

        mock_smtp_class.assert_called_once_with("smtp.gmail.com", 587)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("me@gmail.com", "pw123")
        mock_server.sendmail.assert_called_once()
        send_args = mock_server.sendmail.call_args[0]
        assert send_args[0] == "me@gmail.com"
        assert send_args[1] == "me@gmail.com"
        assert "Subject" in send_args[2]
        assert "<p>Body</p>" in send_args[2]

@pytest.mark.unit
class TestNightlySkipReason:
    """The scheduled runs need the integration on and self_improvement.nightly_runs true; the
    ad-hoc commands only the former."""

    def _reason(self, nightly_runs, unavailable=None):
        with patch.object(CodeQualityMaintainer, "unavailable_reason", return_value=unavailable), \
                patch("assistant.agents.CodeQualityMaintainer.ConfigManager") as cm:
            cm.return_value.get.side_effect = lambda key, default=None: default if nightly_runs is None else nightly_runs
            return _REAL_NIGHTLY_SKIP_REASON()

    def test_off_by_default(self):
        assert self._reason(nightly_runs=None) == NIGHTLY_OFF_MESSAGE

    def test_runs_when_true(self):
        assert self._reason(nightly_runs=True) is None

    def test_integration_switch_wins(self):
        assert self._reason(nightly_runs=True, unavailable=DISABLED_MESSAGE) == DISABLED_MESSAGE

    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_nightly_run_skips_while_off(self, mock_sub, maintainer):
        with patch.object(CodeQualityMaintainer, "nightly_skip_reason", return_value=NIGHTLY_OFF_MESSAGE), \
                patch.object(CodeQualityMaintainer, '_step_bug_analysis') as bugs, \
                patch.object(CodeQualityMaintainer, '_step_auto_merge') as merge:
            maintainer._execute_nightly_run()
        bugs.assert_not_called()
        merge.assert_not_called()
        mock_sub.assert_not_called()


@pytest.mark.unit
class TestExecuteNightlyRun:
    @pytest.fixture(autouse=True)
    def later_steps(self):
        with patch.object(CodeQualityMaintainer, '_step_test_health', return_value="") as health, \
                patch.object(CodeQualityMaintainer, '_step_auto_merge', return_value="") as merge:
            self.health, self.merge = health, merge
            yield

    @patch.object(CodeQualityMaintainer, '_has_recent_commits', return_value=True)
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_runs_the_four_steps_in_order_with_the_auto_merge_last(self, mock_sub, mock_commits, maintainer):
        call_order = []
        self.health.side_effect = lambda: call_order.append('health') or ''
        self.merge.side_effect = lambda: call_order.append('merge') or ''
        with patch.object(CodeQualityMaintainer, '_step_bug_analysis', side_effect=lambda: call_order.append('bugs') or ''):
            with patch.object(CodeQualityMaintainer, '_step_generate_tests', side_effect=lambda: call_order.append('tests') or ''):
                with patch.object(CodeQualityMaintainer, '_send_email'):
                    maintainer._execute_nightly_run()
        assert call_order == ['bugs', 'tests', 'health', 'merge']

    @patch.object(CodeQualityMaintainer, '_has_recent_commits', return_value=True)
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_one_email_carries_every_step_that_reported(self, mock_sub, mock_commits, maintainer):
        self.health.return_value = "test_sync_waits removed"
        self.merge.return_value = "Merged PR #42"
        with patch.object(CodeQualityMaintainer, '_step_bug_analysis', return_value=""), \
                patch.object(CodeQualityMaintainer, '_step_generate_tests', return_value=""), \
                patch.object(CodeQualityMaintainer, '_send_email') as mock_email:
            maintainer._execute_nightly_run()
        mock_email.assert_called_once()
        html = mock_email.call_args[0][1]
        assert "test_sync_waits removed" in html and "Merged PR #42" in html

    @patch.object(CodeQualityMaintainer, '_has_recent_commits', return_value=True)
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_step2_runs_even_if_step1_fails(self, mock_sub, mock_commits, maintainer):
        with patch.object(CodeQualityMaintainer, '_step_bug_analysis', side_effect=Exception("step 1 boom")):
            with patch.object(CodeQualityMaintainer, '_step_generate_tests', return_value='') as mock_step2:
                with patch.object(CodeQualityMaintainer, '_send_email'):
                    maintainer._execute_nightly_run()
                    mock_step2.assert_called_once()

    @patch.object(CodeQualityMaintainer, '_has_recent_commits', return_value=True)
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_sends_combined_email_with_both_sections(self, mock_sub, mock_commits, maintainer):
        with patch.object(CodeQualityMaintainer, '_step_bug_analysis', return_value="Bug in foo.py"):
            with patch.object(CodeQualityMaintainer, '_step_generate_tests', return_value="Added test_foo.py"):
                with patch.object(CodeQualityMaintainer, '_send_email') as mock_email:
                    maintainer._execute_nightly_run()
                    mock_email.assert_called_once()
                    subject, html = mock_email.call_args[0]
                    assert subject == "Keep Nightly Code Quality Report"
                    assert "Bug Analysis" in html
                    assert "Test Generation" in html

    @patch.object(CodeQualityMaintainer, '_has_recent_commits', return_value=True)
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_sends_email_with_only_bug_section(self, mock_sub, mock_commits, maintainer):
        with patch.object(CodeQualityMaintainer, '_step_bug_analysis', return_value="Bug found"):
            with patch.object(CodeQualityMaintainer, '_step_generate_tests', return_value=""):
                with patch.object(CodeQualityMaintainer, '_send_email') as mock_email:
                    maintainer._execute_nightly_run()
                    mock_email.assert_called_once()
                    _, html = mock_email.call_args[0]
                    assert "Bug Analysis" in html
                    assert "Test Generation" not in html

    @patch.object(CodeQualityMaintainer, '_has_recent_commits', return_value=True)
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_sends_email_with_only_test_section(self, mock_sub, mock_commits, maintainer):
        with patch.object(CodeQualityMaintainer, '_step_bug_analysis', return_value=""):
            with patch.object(CodeQualityMaintainer, '_step_generate_tests', return_value="Created tests"):
                with patch.object(CodeQualityMaintainer, '_send_email') as mock_email:
                    maintainer._execute_nightly_run()
                    mock_email.assert_called_once()
                    _, html = mock_email.call_args[0]
                    assert "Test Generation" in html
                    assert "Bug Analysis" not in html

    @patch.object(CodeQualityMaintainer, '_has_recent_commits', return_value=True)
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_skips_email_when_both_empty(self, mock_sub, mock_commits, maintainer):
        with patch.object(CodeQualityMaintainer, '_step_bug_analysis', return_value=""):
            with patch.object(CodeQualityMaintainer, '_step_generate_tests', return_value=""):
                with patch.object(CodeQualityMaintainer, '_send_email') as mock_email:
                    maintainer._execute_nightly_run()
                    mock_email.assert_not_called()

    @patch.object(CodeQualityMaintainer, '_has_recent_commits', return_value=False)
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_a_night_without_recent_commits_only_auto_merges(self, mock_sub, mock_commits, maintainer):
        with patch.object(CodeQualityMaintainer, '_step_bug_analysis') as mock_bugs:
            with patch.object(CodeQualityMaintainer, '_step_generate_tests') as mock_tests:
                maintainer._execute_nightly_run()
                mock_bugs.assert_not_called()
                mock_tests.assert_not_called()
        self.health.assert_not_called()
        self.merge.assert_called_once()

    @patch.object(CodeQualityMaintainer, '_has_recent_commits', return_value=False)
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_a_single_step_runs_alone_whatever_the_commits(self, mock_sub, mock_commits, maintainer):
        with patch.object(CodeQualityMaintainer, '_step_bug_analysis') as mock_bugs:
            maintainer._execute_nightly_run(only="health")
        self.health.assert_called_once()
        mock_bugs.assert_not_called()
        self.merge.assert_not_called()

    @patch.object(CodeQualityMaintainer, '_has_recent_commits', return_value=True)
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_force_ignores_the_schedule_switch_but_not_the_integration(self, mock_sub, mock_commits, maintainer):
        with patch.object(CodeQualityMaintainer, 'nightly_skip_reason', return_value=NIGHTLY_OFF_MESSAGE), \
                patch.object(CodeQualityMaintainer, '_step_bug_analysis', return_value=""), \
                patch.object(CodeQualityMaintainer, '_step_generate_tests', return_value=""):
            maintainer._execute_nightly_run(force=True)
            assert self.merge.call_count == 1
            with patch.object(CodeQualityMaintainer, 'unavailable_reason', return_value=DISABLED_MESSAGE):
                maintainer._execute_nightly_run(force=True)
            assert self.merge.call_count == 1

    @patch.object(CodeQualityMaintainer, '_has_recent_commits', return_value=True)
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_a_second_run_does_not_start_while_one_is_going(self, mock_sub, mock_commits, maintainer):
        from assistant.agents.CodeQualityMaintainer import _nightly_lock
        with _nightly_lock:
            maintainer._execute_nightly_run()
        mock_sub.assert_not_called()
        self.merge.assert_not_called()

    @patch.object(CodeQualityMaintainer, '_has_recent_commits', return_value=True)
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_pulls_before_checking_commits(self, mock_sub, mock_commits, maintainer):
        with patch.object(CodeQualityMaintainer, '_step_bug_analysis', return_value=""):
            with patch.object(CodeQualityMaintainer, '_step_generate_tests', return_value=""):
                maintainer._execute_nightly_run()
                mock_sub.assert_called_once_with(["git", "pull"], "git pull", GIT_TIMEOUT)


def developer_release(value: bool):
    """Pins the `developer_release` key: the developer's own config.yaml has it true, a test
    must not inherit that."""
    return patch.object(ConfigManager, 'get',
                        side_effect=lambda key, default=None: value if key == "developer_release" else default)


@pytest.mark.unit
class TestPullAndRestart:
    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_success_with_changes(self, mock_sub, mock_popen):
        mock_sub.return_value = Mock(stdout="Updating abc123..def456\n1 file changed", stderr="", returncode=0)
        result = CodeQualityMaintainer.pull_and_restart()
        assert "Update initiated" in result
        assert "pulled updates" in result
        mock_popen.assert_called_once()
        popen_kwargs = mock_popen.call_args[1]
        assert popen_kwargs["cwd"] == AISERVER_DIR
        assert popen_kwargs["start_new_session"] is True

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_already_up_to_date(self, mock_sub, mock_popen):
        mock_sub.return_value = Mock(stdout="Already up to date.", stderr="", returncode=0)
        result = CodeQualityMaintainer.pull_and_restart()
        assert "no new changes" in result
        mock_popen.assert_called_once()

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    @patch.object(CodeQualityMaintainer, '_current_branch', return_value="release")
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_git_pull_failure_does_not_restart(self, mock_sub, mock_branch, mock_popen):
        mock_sub.return_value = Mock(stdout="", stderr="fatal: not a git repository", returncode=128)
        with developer_release(False):
            result = CodeQualityMaintainer.pull_and_restart()
        assert "Git pull failed" in result
        mock_popen.assert_not_called()

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    @patch.object(CodeQualityMaintainer, '_current_branch', return_value="release")
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_pulls_in_place_when_already_on_the_target_branch(self, mock_sub, mock_branch, mock_popen):
        mock_sub.return_value = Mock(stdout="Already up to date.", stderr="", returncode=0)
        with developer_release(False):
            CodeQualityMaintainer.pull_and_restart()
        assert [c[0][0] for c in mock_sub.call_args_list] == [["git", "pull"]]
        mock_popen.assert_called_once()

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    @patch.object(CodeQualityMaintainer, '_current_branch', return_value="release")
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_developer_release_switches_to_master_before_pulling(self, mock_sub, mock_branch, mock_popen):
        mock_sub.return_value = Mock(stdout="Already up to date.", stderr="", returncode=0)
        with developer_release(True):
            CodeQualityMaintainer.pull_and_restart()
        assert [c[0][0] for c in mock_sub.call_args_list] == [
            ["git", "fetch", "origin", "master"], ["git", "checkout", "master"], ["git", "pull"]]
        mock_popen.assert_called_once()

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    @patch.object(CodeQualityMaintainer, '_current_branch', return_value="master")
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_failed_switch_leaves_the_branch_and_does_not_restart(self, mock_sub, mock_branch, mock_popen):
        def git(cmd, label, timeout):
            if cmd[:2] == ["git", "checkout"]:
                return Mock(stdout="", stderr="error: your local changes would be overwritten", returncode=1)
            return Mock(stdout="", stderr="", returncode=0)
        mock_sub.side_effect = git
        with developer_release(False):
            result = CodeQualityMaintainer.pull_and_restart()
        assert "Could not switch to release" in result and "still on master" in result
        assert ["git", "pull"] not in [c[0][0] for c in mock_sub.call_args_list]
        mock_popen.assert_not_called()

    def test_target_branch_follows_the_config_key(self):
        with developer_release(False):
            assert CodeQualityMaintainer.target_branch() == "release"
        with developer_release(True):
            assert CodeQualityMaintainer.target_branch() == "master"

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_spawns_start_sh_with_correct_args(self, mock_sub, mock_popen):
        mock_sub.return_value = Mock(stdout="Already up to date.", stderr="", returncode=0)
        CodeQualityMaintainer.pull_and_restart()
        args = mock_popen.call_args[0][0]
        assert args == ['bash', 'start.sh', str(os.getpid())]
        kwargs = mock_popen.call_args[1]
        assert kwargs["stdout"] == subprocess.DEVNULL
        assert kwargs["stderr"] == subprocess.DEVNULL
        assert kwargs["stdin"] == subprocess.DEVNULL
