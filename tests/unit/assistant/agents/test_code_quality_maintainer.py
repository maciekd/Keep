from datetime import date, datetime
from unittest.mock import patch, Mock, MagicMock, call

import pytest
from pathlib import Path

from common.ConfigManager import ConfigManager
from common.Paths import AISERVER_DIR
from common.Updater import Updater

import json
import os
import subprocess
import threading

from assistant.agents.CodeQualityMaintainer import CodeQualityMaintainer, JobPullRequest, WORKING_DIR, CLAUDE_PATH, GIT_TIMEOUT, GENERATE_TESTS_PROMPT, SLOW_TEST_MS, TEST_HEALTH_RUNS, ClaudeNotLoggedInError, ClaudeRunFailedError, ClaudeUsageLimitError, ClaudeStatus, API_KEY_ENV, CUSTOM_HEADERS_ENV, NOT_LOGGED_IN_MESSAGE, NIGHTLY_OFF_MESSAGE, DISABLED_MESSAGE, api_key_cost_note, NIGHTLY_FIX_LABEL, GH_NOT_LOGGED_IN_MESSAGE, NIGHTLY_STEPS, LOG_TRIAGE_HOURS, GH_NOT_INSTALLED_MESSAGE, GithubAccess, IssueDraft, PREFILL_URL_LIMIT, ISSUE_LABEL

_REAL_NIGHTLY_SKIP_REASON = CodeQualityMaintainer.nightly_skip_reason   # the autouse fixture below patches it


@pytest.fixture(autouse=True)
def self_improvement_enabled():
    """The jobs run only while the Self Improvement integration is enabled and the scheduled ones
    only with self_improvement.nightly_runs on; these tests are about the jobs themselves."""
    with patch.object(CodeQualityMaintainer, "unavailable_reason", return_value=None), \
            patch.object(CodeQualityMaintainer, "nightly_skip_reason", return_value=None):
        yield


@pytest.fixture(autouse=True)
def on_a_branch_installs_follow():
    """The suite runs on whatever branch its checkout is on — a developer's feature branch too,
    which the update paths would take for a branch of the install's own and leave alone
    (tests/unit/assistant/agents/test_own_branch.py is about that)."""
    with patch.object(CodeQualityMaintainer, "own_branch", return_value=None):
        yield


@pytest.fixture(autouse=True)
def claude_installed():
    with patch("assistant.agents.CodeQualityMaintainer.claude_path", return_value=CLAUDE_PATH):
        yield


@pytest.fixture(autouse=True)
def no_real_processes():
    """A test that reaches Popen unpatched would start Claude Code (or start.sh) on this machine
    for real — one did, when a new step was added without patching it here. A test that wants
    Popen patches it itself; that inner patch wins over this one."""
    real_popen = subprocess.Popen

    def guard(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args")
        if cmd and os.path.basename(str(cmd[0])) in ("claude", "bash"):
            raise AssertionError(f"a unit test tried to start a real process: {cmd}")
        return real_popen(*args, **kwargs)   # git and gh in a subprocess.run are harmless and some tests run them
    with patch("assistant.agents.CodeQualityMaintainer.subprocess.Popen", side_effect=guard):
        yield


@pytest.fixture(autouse=True)
def signed_release():
    """Every pull of `release` goes through SignedRelease, which fetches and verifies for real —
    on this repository, from its origin. Here it is a mock that finds the release current; a
    test that is about it sets its return value. (SignedRelease's own tests run it on throwaway
    repositories.)"""
    with patch("common.SignedRelease.SignedRelease.update",
               return_value=("", "Already up to date.")) as update:
        yield update


@pytest.fixture
def maintainer():
    return CodeQualityMaintainer()


PUSHER = GithubAccess(login="marcus", repo="marcus/keep", permission="WRITE")
READER = GithubAccess(login="alice", repo="marcus/keep", permission="READ")


def _claude_prompt(job, trial=""):
    """Runs *job* through the real _pr_job with git, gh and Claude Code stubbed: (the prompt
    Claude Code got — None when it never ran —, the job's result)."""
    prompts = []
    with patch.object(CodeQualityMaintainer, "_run_subprocess", return_value=Mock(returncode=0, stdout="master\n", stderr="")), \
            patch.object(CodeQualityMaintainer, "_open_pull_requests", return_value={}), \
            patch.object(CodeQualityMaintainer, "_trial_instructions", return_value=trial), \
            patch.object(CodeQualityMaintainer, "_run_claude", side_effect=lambda prompt, label: prompts.append(prompt) or "done"), \
            patch("assistant.agents.PullRequestTrial.PullRequestTrial.current", return_value=None):
        result = job()
    return (prompts[0] if prompts else None), result


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


    @patch('assistant.agents.CodeQualityMaintainer.time.sleep')
    @patch('assistant.agents.CodeQualityMaintainer.datetime')
    def test_the_nightly_update_comes_before_the_run(self, mock_dt, mock_sleep, maintainer):
        """The run takes an hour or more; an update after it restarted the server inside the
        memory curation's hour (2026-10-14). The update goes first, and when it does not restart
        the run follows in the same process."""
        mock_dt.now.return_value = datetime(2026, 2, 19, 3, 0, 0)
        order = []

        def stop_after_one(*args):
            maintainer._running = False

        mock_sleep.side_effect = stop_after_one

        def update():
            order.append("update")
            return False

        with patch.object(maintainer, '_execute_nightly_run', side_effect=lambda: order.append("run")), \
                patch.object(maintainer, '_nightly_update', side_effect=update):
            maintainer._running = True
            maintainer._run()
        assert order == ["update", "run"]

    @patch('assistant.agents.CodeQualityMaintainer.time.sleep')
    @patch('assistant.agents.CodeQualityMaintainer.datetime')
    def test_a_restarting_update_leaves_the_run_to_the_new_process(self, mock_dt, mock_sleep, maintainer):
        """restart() returns while start.sh kills this process; a Claude Code job started now would
        die with it, so the loop ends instead."""
        mock_dt.now.return_value = datetime(2026, 2, 19, 3, 0, 0)
        mock_sleep.side_effect = lambda *a: setattr(maintainer, '_running', False)
        with patch.object(maintainer, '_execute_nightly_run') as run, \
                patch.object(maintainer, '_nightly_update', return_value=True):
            maintainer._running = True
            maintainer._run()
        run.assert_not_called()

    @patch('assistant.agents.CodeQualityMaintainer.threading.Thread')
    @patch.object(CodeQualityMaintainer, '_head_commit', return_value="a" * 40)
    @patch('assistant.agents.CodeQualityMaintainer.datetime')
    def test_a_start_inside_the_run_hour_takes_the_night_over(self, mock_dt, mock_head, mock_thread, maintainer):
        """The nightly update restarts the server at the top of RUN_HOUR; the new process is the
        one that runs the night, so booting inside the hour must not count it as done."""
        mock_dt.now.return_value = datetime(2026, 2, 19, 3, 0, 40)
        maintainer.start()
        assert maintainer._last_run_date is None


BOOT, NEWER = "b" * 40, "c" * 40


@pytest.mark.unit
class TestNightlyUpdate:
    @pytest.fixture(autouse=True)
    def wiring(self, maintainer):
        """A maintainer booted on BOOT whose pull goes through; git's HEAD afterwards and the
        config are each test's to set. Nothing here reaches git or start.sh."""
        maintainer._boot_commit = BOOT
        self.config = {"nightly_update": True, "developer_release": True}
        with patch.object(ConfigManager, 'get', side_effect=lambda key, default=None: self.config.get(key, default)), \
                patch.object(Updater, 'pull', return_value=("", "")) as pull, \
                patch.object(Updater, 'head_commit', return_value=NEWER) as head, \
                patch.object(Updater, 'commits_between', return_value=2), \
                patch.object(Updater, 'sync', return_value=("", "")) as sync, \
                patch.object(CodeQualityMaintainer, 'restart') as restart, \
                patch('assistant.SystemNote.system_note') as note_:
            self.pull, self.head, self.sync, self.restart, self.note = pull, head, sync, restart, note_
            yield

    def test_new_commits_restart_the_server_with_a_history_only_note(self, maintainer):
        maintainer._nightly_update()
        self.pull.assert_called_once_with("master")
        self.restart.assert_called_once()
        (source, text), kwargs = self.note.call_args
        assert source == "server" and "2 new commits" in text and "master" in text
        assert kwargs == {"push": False}   # the chats get the new process's startup announcement instead

    def test_no_new_commits_means_no_restart(self, maintainer):
        self.head.return_value = BOOT
        maintainer._nightly_update()
        self.pull.assert_called_once_with("master")
        self.restart.assert_not_called()
        self.note.assert_not_called()
        self.sync.assert_not_called()   # pip only runs ahead of a restart

    def test_a_pin_bump_is_installed_before_the_restart_and_named_in_the_note(self, maintainer):
        self.sync.return_value = ("", "installed 1 package: pkg-a 1.0.0 → 2.0.0")
        maintainer._nightly_update()
        self.sync.assert_called_once_with()
        self.restart.assert_called_once()
        assert "pkg-a" in self.note.call_args[0][1]

    def test_a_failed_install_keeps_the_old_code_running_and_says_why(self, maintainer):
        self.sync.return_value = ("pip exited 1 (no matching distribution)", "")
        maintainer._nightly_update()
        self.restart.assert_not_called()
        text = self.note.call_args[0][1]
        assert "no matching distribution" in text and "old code" in text

    def test_a_failed_pull_keeps_the_old_code_running_and_says_why(self, maintainer):
        self.pull.return_value = ("Git pull failed: merge conflict in config", "")
        maintainer._nightly_update()
        self.restart.assert_not_called()
        assert "merge conflict in config" in self.note.call_args[0][1]

    def test_an_unreadable_head_does_not_restart(self, maintainer):
        self.head.return_value = ""
        maintainer._nightly_update()
        self.restart.assert_not_called()

    def _trial(self, **night):
        from assistant.agents.PullRequestTrial import PullRequestTrial, Trial, Overnight
        trial = Trial(number=204, title="Tesla: close the trunk once", head_ref="feat/trunk",
                      since="14:02", return_branch="master")
        return patch.multiple(PullRequestTrial, current=Mock(return_value=trial),
                              overnight=Mock(return_value=Overnight(**night)))

    def test_a_trial_that_took_new_code_in_restarts_on_its_own_branch(self, maintainer):
        with self._trial():
            assert maintainer._nightly_update() is True
        self.pull.assert_not_called()   # the checkout never leaves the trial branch
        self.sync.assert_called_once_with()
        assert self.restart.call_args.kwargs == {"fallback_branch": "master"}, \
            "start.sh goes back to the normal branch if the merged code does not boot"
        (_, text), kwargs = self.note.call_args
        assert "PR #204" in text and kwargs == {"push": False}

    def test_a_trial_with_nothing_new_does_not_restart(self, maintainer):
        self.head.return_value = BOOT
        with self._trial():
            assert maintainer._nightly_update() is False
        self.pull.assert_not_called()
        self.restart.assert_not_called()
        self.sync.assert_not_called()

    def test_a_trial_that_could_not_take_the_night_in_keeps_running_and_says_why(self, maintainer):
        with self._trial(failure="Git fetch failed: no route to host"):
            assert maintainer._nightly_update() is False
        self.pull.assert_not_called()
        self.restart.assert_not_called()
        assert "no route to host" in self.note.call_args[0][1]

    def test_a_trial_whose_merged_pins_do_not_install_keeps_the_old_process(self, maintainer):
        self.sync.return_value = ("pip exited 1 (no matching distribution)", "")
        with self._trial():
            assert maintainer._nightly_update() is False
        self.restart.assert_not_called()
        self.pull.assert_not_called()
        (_, text), kwargs = self.note.call_args
        assert "no matching distribution" in text and "PR #204" in text and kwargs == {"push": False}

    def test_a_trial_night_names_the_packages_it_installed(self, maintainer):
        self.sync.return_value = ("", "installed 1 package: pkg-a 1.0.0 → 2.0.0")
        with self._trial():
            assert maintainer._nightly_update() is True
        assert "pkg-a" in self.note.call_args[0][1]

    def test_a_trial_whose_head_cannot_be_read_does_not_restart(self, maintainer):
        self.head.return_value = ""
        with self._trial():
            assert maintainer._nightly_update() is False
        self.restart.assert_not_called()

    def test_a_trial_that_is_over_returns_to_the_normal_branch_and_the_user_is_told_why(self, maintainer):
        with self._trial(ended="PR #204 is merged and part of master now"):
            assert maintainer._nightly_update() is True
        self.pull.assert_called_once_with("master")
        self.restart.assert_called_once()
        assert "fallback_branch" not in self.restart.call_args.kwargs
        (_, text), kwargs = self.note.call_args
        assert "part of master now" in text and kwargs == {"push": True}

    def test_a_trial_that_ends_on_the_commit_it_stood_on_still_restarts(self, maintainer):
        """The process reads its branch once, at boot: without a restart it would go on
        believing it runs a trial."""
        self.head.return_value = BOOT
        with self._trial(ended="PR #204 is merged and part of master now"):
            assert maintainer._nightly_update() is True
        self.restart.assert_called_once()

    def test_switched_off_it_touches_nothing(self, maintainer):
        self.config["nightly_update"] = False
        maintainer._nightly_update()
        self.pull.assert_not_called()
        self.restart.assert_not_called()

    def test_it_is_on_when_the_key_is_absent(self, maintainer):
        del self.config["nightly_update"]
        maintainer._nightly_update()
        self.restart.assert_called_once()

    def test_no_update_in_local_development_mode(self, maintainer):
        with patch('common.DevelopmentConfig.LOCAL_DEVELOPMENT_MODE', True):
            maintainer._nightly_update()
        self.pull.assert_not_called()
        self.restart.assert_not_called()

    def test_no_update_when_the_boot_commit_is_unknown(self, maintainer):
        maintainer._boot_commit = ""
        maintainer._nightly_update()
        self.pull.assert_not_called()
        self.restart.assert_not_called()

    def test_no_update_under_a_running_nightly_run(self, maintainer):
        from assistant.agents import CodeQualityMaintainer as module
        assert module._nightly_lock.acquire(blocking=False)
        try:
            maintainer._nightly_update()
        finally:
            module._nightly_lock.release()
        self.pull.assert_not_called()
        self.restart.assert_not_called()

    def test_a_switch_to_an_older_release_still_restarts(self, maintainer):
        """release may sit behind the commit a master process booted on: nothing is "new", the
        code is still different."""
        self.config["developer_release"] = False
        with patch.object(Updater, 'commits_between', return_value=0):
            maintainer._nightly_update()
        self.pull.assert_called_once_with("release")
        self.restart.assert_called_once()
        assert "release" in self.note.call_args[0][1]


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
        assert self._merges(gh) == [["gh", "pr", "merge", "7", "--rebase", "--delete-branch", "--admin"],
                                    ["gh", "pr", "merge", "9", "--rebase", "--delete-branch", "--admin"]]
        merged, skipped = report.split("Skipped PRs")
        assert "#7" in merged and "#9" in merged and "#8" not in merged
        assert "#8" in skipped and "src/common/Parser.py" in skipped

    @patch.object(CodeQualityMaintainer, '_auto_merge_candidates', return_value=[7])
    def test_merges_with_the_configured_strategy(self, mock_candidates, gh, maintainer):
        gh.config.side_effect = lambda k, default=None: {"self_improvement.merge_strategy": "squash"}.get(k, default)
        maintainer._step_auto_merge()
        assert self._merges(gh) == [["gh", "pr", "merge", "7", "--squash", "--delete-branch", "--admin"]]

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

    def test_the_report_lists_what_the_test_generation_merged_earlier_in_the_night(self, gh, maintainer):
        """The test PR is merged right after it is opened, long before the auto-merge step runs; the
        night's end still lists it, so one section holds every merge of the night."""
        with patch.object(CodeQualityMaintainer, '_auto_merge_candidates', return_value=[7]):
            maintainer._merge_test_prs()
        with patch.object(CodeQualityMaintainer, '_auto_merge_candidates', return_value=[9]):
            merged, skipped = maintainer._step_auto_merge().split("Skipped PRs")
        assert "#7" in merged and "#9" in merged
        assert len(self._merges(gh)) == 2, "the earlier merge is listed, not merged again"

    def test_a_night_whose_only_merge_was_the_test_pr_still_reports_it(self, gh, maintainer):
        with patch.object(CodeQualityMaintainer, '_auto_merge_candidates', return_value=[7]):
            maintainer._merge_test_prs()
        with patch.object(CodeQualityMaintainer, '_auto_merge_candidates', return_value=[]):
            report = maintainer._step_auto_merge()
        assert "#7" in report.split("Skipped PRs")[0]

    def test_last_nights_merge_is_not_listed_again_tonight(self, gh, maintainer):
        with patch.object(CodeQualityMaintainer, '_auto_merge_candidates', return_value=[7]):
            maintainer._merge_test_prs()          # last night, in the same process
        with patch.object(CodeQualityMaintainer, '_auto_merge_candidates', return_value=[9]), \
                patch.object(CodeQualityMaintainer, '_has_recent_commits', return_value=False), \
                patch.object(CodeQualityMaintainer, '_step_log_triage', return_value=""), \
                patch.object(CodeQualityMaintainer, '_step_fix_bugs', return_value=""), \
                patch.object(CodeQualityMaintainer, '_step_open_pins', return_value=""), \
                patch.object(CodeQualityMaintainer, '_markdown_to_html', side_effect=lambda md: md), \
                patch.object(CodeQualityMaintainer, '_send_email') as send:
            maintainer._nightly_run(only=None)
        merged = send.call_args[0][1].split("Skipped PRs")[0]
        assert "#9" in merged and "#7" not in merged

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
    def test_pr_files_asks_for_the_status_and_the_deletions(self, mock_run):
        """The scope check reads them, so the jq filter must request them."""
        mock_run.side_effect = self._gh([("src/integrations/ring/RingIntegration.py", None)], 1)
        _, files = CodeQualityMaintainer._pr_files(7)
        assert files[0]["filename"] == "src/integrations/ring/RingIntegration.py"
        jq = [a for a in mock_run.call_args.args[0] if "filename" in a][0]
        assert "status" in jq and "deletions" in jq

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
    @pytest.fixture(autouse=True)
    def merge(self):
        with patch.object(CodeQualityMaintainer, '_merge_test_prs', return_value=(None, [])) as merge:
            yield merge

    @patch.object(CodeQualityMaintainer, '_run_claude', return_value="")
    @patch('assistant.agents.CodeQualityMaintainer.subprocess.run')
    def test_reads_the_branch_then_claude_then_checks_it_out_again(self, mock_run, mock_claude, maintainer):
        mock_run.return_value = Mock(stdout="release\n", returncode=0)
        maintainer._step_generate_tests()

        assert [c[0][0] for c in mock_run.call_args_list] == [
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], ["git", "checkout", "release"]]
        assert mock_run.call_args[1]["cwd"] == WORKING_DIR
        mock_claude.assert_called_once_with(GENERATE_TESTS_PROMPT, "generate-tests")

    @patch.object(CodeQualityMaintainer, '_run_claude', return_value="Eight tests added. PR #70.")
    @patch('assistant.agents.CodeQualityMaintainer.subprocess.run')
    def test_the_test_pr_is_merged_once_the_checkout_is_back_and_the_report_says_so(
            self, mock_run, mock_claude, merge, maintainer):
        """The fix step after it branches from master, so the tests must be there by then."""
        events = []
        mock_run.side_effect = lambda cmd, **_: events.append(cmd[:2]) or Mock(stdout="master\n", returncode=0)
        merge.side_effect = lambda: events.append("merge") or (["- #70 test: eight tests"], ["- #65 feat — outside"])
        report = maintainer._step_generate_tests()
        assert events.index(["git", "checkout"]) < events.index("merge")
        assert "- #70 test: eight tests" in report and "#65" not in report, "the skipped PRs are the night's end's to list"
        assert report.startswith("Eight tests added")

    @patch.object(CodeQualityMaintainer, '_run_claude', return_value="Nothing to add.")
    @patch('assistant.agents.CodeQualityMaintainer.subprocess.run')
    def test_no_test_pr_to_merge_leaves_the_report_as_claude_wrote_it(self, mock_run, mock_claude, merge, maintainer):
        mock_run.return_value = Mock(stdout="master\n", returncode=0)
        assert maintainer._step_generate_tests() == "Nothing to add."

    def test_the_prompt_forbids_fixing_and_asks_for_the_fix_list_the_fix_step_reads(self):
        lowered = GENERATE_TESTS_PROMPT.lower()
        assert "do not fix" in lowered and "xfail" in lowered and "strict=true" in lowered
        assert "fix list" in lowered and "critical and major rows" in lowered

    @patch.object(CodeQualityMaintainer, '_run_claude', side_effect=Exception("claude failed"))
    @patch('assistant.agents.CodeQualityMaintainer.subprocess.run')
    def test_returns_empty_on_error_and_merges_nothing(self, mock_run, mock_claude, merge, maintainer):
        mock_run.return_value = Mock(stdout="", returncode=0)
        result = maintainer._step_generate_tests()
        assert result == ""
        merge.assert_not_called()

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
class TestStepLogTriage:
    def test_the_prompt_asks_for_the_configured_window_and_the_logs_and_the_fix_list(self):
        with patch.object(CodeQualityMaintainer, '_run_claude', return_value="report") as run:
            assert CodeQualityMaintainer._step_log_triage() == "report"
        prompt, label = run.call_args[0]
        assert f"last {LOG_TRIAGE_HOURS} hours" in prompt
        assert "log/floxy.log" in prompt and "log/raw-floxy.log" in prompt   # the files the server writes
        assert "runs commit" in prompt   # the start marker that tells which code was running
        assert "fix list" in prompt and "critical and major rows" in prompt   # what the fix step reads
        assert label == "log-triage"

    @patch.object(CodeQualityMaintainer, '_run_claude', side_effect=Exception("claude failed"))
    def test_returns_empty_on_error(self, mock_claude):
        assert CodeQualityMaintainer._step_log_triage() == ""

    @patch.object(CodeQualityMaintainer, '_run_claude', side_effect=ClaudeUsageLimitError("log-triage", "out of usage credits"))
    def test_a_failed_claude_run_reaches_the_night(self, mock_claude):
        with pytest.raises(ClaudeUsageLimitError):
            CodeQualityMaintainer._step_log_triage()


@pytest.mark.unit
class TestFixInput:
    def test_the_three_reports_stand_under_their_headings_with_the_tests_last(self):
        text = CodeQualityMaintainer.fix_input({"tests": "pinned", "bugs": "off by one", "logs": "a stall",
                                                "health": "ignored"})
        assert (text.index("# Bug Analysis") < text.index("off by one") < text.index("# Log Triage")
                < text.index("a stall") < text.index("# Test Generation") < text.index("pinned"))
        assert "ignored" not in text

    def test_a_missing_report_leaves_only_the_other_and_none_leaves_nothing(self):
        only_logs = CodeQualityMaintainer.fix_input({"bugs": "", "logs": "L"})
        assert "# Log Triage" in only_logs and "Bug Analysis" not in only_logs
        assert CodeQualityMaintainer.fix_input({}) == ""
        assert CodeQualityMaintainer.fix_input({"bugs": "", "logs": ""}) == ""


@pytest.mark.unit
class TestImproveLocalIntegration:
    """The job on an install's own folders outside the checkout (#156): the run stays in the
    checkout (CLAUDE.md applies) with the local directories added as writable, no PR."""

    ROOTS = (Path("/home/marcus/KeepLocal"), Path("/home/marcus/KeepExtra"))

    def _run(self, **kwargs):
        calls = []

        def fake(prompt, label, add_dirs=()):
            calls.append((prompt, label, add_dirs))
            return "Moved the folder.\nCommitted and pushed to forge.local."
        with patch.object(CodeQualityMaintainer, "_run_claude", side_effect=fake):
            report = CodeQualityMaintainer.improve_local_integration("push it to forge.local", self.ROOTS, **kwargs)
        return report, calls[0]

    def test_the_local_directories_are_writable_and_the_checkout_is_the_working_directory(self):
        report, (prompt, _, add_dirs) = self._run()
        assert add_dirs == self.ROOTS
        assert all(str(root) in prompt for root in self.ROOTS)
        assert "push it to forge.local" in prompt and report.startswith("Moved the folder.")

    def test_the_prompt_names_the_folder_of_an_existing_local_integration(self):
        _, (prompt, _, _) = self._run(name="Podcasts", folder="/home/marcus/KeepLocal/podcasts")
        assert "/home/marcus/KeepLocal/podcasts" in prompt and "Podcasts" in prompt
        assert "new folder" not in prompt

    def test_a_new_local_integration_goes_into_the_first_directory(self):
        _, (prompt, _, _) = self._run(name="Ring")
        assert "Ring" in prompt and "new folder in /home/marcus/KeepLocal" in prompt

    @patch.object(CodeQualityMaintainer, "_run_claude", side_effect=ClaudeNotLoggedInError())
    def test_explains_missing_login_instead_of_raw_cli_text(self, mock_claude):
        assert CodeQualityMaintainer.improve_local_integration("fix it", self.ROOTS) == NOT_LOGGED_IN_MESSAGE

    @patch.object(CodeQualityMaintainer, "_run_claude", return_value="")
    def test_no_output_is_said_so(self, mock_claude):
        assert "no output" in CodeQualityMaintainer.improve_local_integration("fix it", self.ROOTS)


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
class TestAnalyzeCost:
    """The cost_analysis command's job: the keep-cost-analysis skill run by Claude Code. Its
    report is delivered from a background thread that only logs an exception, so the job
    itself must answer with text whatever happens to the run."""

    @patch.object(CodeQualityMaintainer, '_run_claude', return_value="## Where it goes\n- checks $3.10")
    def test_the_window_the_llm_gave_reaches_the_skill_contract_with_its_argument(self, run):
        """The skill parses its window from its arguments (`24h`, `3d`, `7d`); the command's
        window must stand in the prompt as the skill's argument, and no window means one day."""
        assert CodeQualityMaintainer.analyze_cost("7d") == "## Where it goes\n- checks $3.10"
        assert "7d" in run.call_args.args[0]
        CodeQualityMaintainer.analyze_cost("")
        assert "24h" in run.call_args.args[0] and "7d" not in run.call_args.args[0]

    @patch.object(CodeQualityMaintainer, '_run_claude', side_effect=ClaudeNotLoggedInError())
    def test_an_unavailable_cli_is_the_report_not_an_exception(self, run):
        assert CodeQualityMaintainer.analyze_cost("") == NOT_LOGGED_IN_MESSAGE

    @patch.object(CodeQualityMaintainer, '_run_claude', side_effect=ClaudeRunFailedError("cost-analysis", "timed out"))
    def test_a_run_that_did_not_finish_says_so_in_the_report(self, run):
        report = CodeQualityMaintainer.analyze_cost("3d")
        assert "timed out" in report

    @patch.object(CodeQualityMaintainer, '_run_claude', return_value="")
    def test_a_run_with_no_output_still_delivers_a_line(self, run):
        assert CodeQualityMaintainer.analyze_cost("").strip()


@pytest.mark.unit
class TestImproveCode:
    @pytest.fixture(autouse=True)
    def gh_logged_in(self):
        """improve_code first asks `gh auth status` and reads the version and the trial state
        from git — real subprocesses on the developer's machine, ~250 ms each test and a
        different answer on a machine without gh. These tests are about the job itself."""
        with patch.object(CodeQualityMaintainer, "_github_access", return_value=(PUSHER, None)), \
                patch.object(CodeQualityMaintainer, "_version", return_value="v2026.10.1"), \
                patch.object(CodeQualityMaintainer, "_trial_instructions", return_value=""):
            yield

    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    @patch.object(CodeQualityMaintainer, '_run_claude', side_effect=ClaudeNotLoggedInError())
    def test_explains_missing_login(self, mock_claude, mock_sub):
        mock_sub.return_value = Mock(stdout="", returncode=0)
        assert CodeQualityMaintainer.improve_code("some bug") == (NOT_LOGGED_IN_MESSAGE, None)
        mock_sub.assert_called_with(["git", "checkout", "master"], "git checkout", GIT_TIMEOUT)

    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    @patch.object(CodeQualityMaintainer, '_run_claude', side_effect=RuntimeError("claude exited 137"))
    def test_a_crashed_run_is_the_result_and_the_starting_branch_comes_back(self, mock_claude, mock_sub):
        mock_sub.return_value = Mock(stdout="release\n", returncode=0)
        result, pr = CodeQualityMaintainer._pr_job("prompt", "self-improvement")
        assert result.startswith("Self-improvement failed") and "exited 137" in result and pr is None
        assert mock_sub.call_args_list[-1] == call(["git", "checkout", "release"], "git checkout", GIT_TIMEOUT)

    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    @patch.object(CodeQualityMaintainer, '_run_claude', return_value="")
    def test_returns_fallback_on_no_output(self, mock_claude, mock_sub):
        mock_sub.return_value = Mock(stdout="", returncode=0)
        result = CodeQualityMaintainer.improve_code("some bug")
        assert result == ("Self-improvement produced no output.", None)

    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    @patch.object(CodeQualityMaintainer, '_run_claude', return_value="done")
    def test_on_release_the_signed_update_then_claude_then_checkout_of_the_starting_branch(
            self, mock_claude, mock_sub, signed_release):
        """An install's checkout is never pulled plainly: an unsigned push to `release` would land
        on disk before the run and be the code of the next restart."""
        order = []
        mock_sub.side_effect = lambda cmd, *a: (order.append(cmd), Mock(stdout="release\n", returncode=0))[1]
        signed_release.side_effect = lambda branch: (order.append(["signed update", branch]), ("", ""))[1]
        mock_claude.side_effect = lambda *a: (order.append(["claude"]), "done")[1]
        CodeQualityMaintainer.improve_code("bug")
        steps = [c for c in order if c[0] != "gh" and c[:2] != ["git", "rev-parse"]]
        assert steps == [["signed update", "release"], ["claude"], ["git", "checkout", "release"]]

    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    @patch.object(CodeQualityMaintainer, '_run_claude', return_value="done")
    def test_on_master_git_pull_then_claude_then_checkout_of_the_starting_branch(
            self, mock_claude, mock_sub, signed_release):
        mock_sub.return_value = Mock(stdout="master\n", returncode=0)
        CodeQualityMaintainer.improve_code("bug")
        commands = [c.args[0] for c in mock_sub.call_args_list if c.args[0][:2] != ["git", "rev-parse"]]
        assert [c for c in commands if c[0] == "git"] == [["git", "pull"], ["git", "checkout", "master"]]
        signed_release.assert_not_called()
        # The open PRs are listed before and after the run — what was open before is never the job's.
        listings = [i for i, c in enumerate(commands) if c[:3] == ["gh", "pr", "list"]]
        assert len(listings) == 2 and listings[0] < commands.index(["git", "pull"]) < listings[1]

    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    @patch.object(CodeQualityMaintainer, '_run_claude', side_effect=Exception("claude crashed"))
    def test_checkout_master_runs_on_claude_error(self, mock_claude, mock_sub):
        mock_sub.return_value = Mock(stdout="", returncode=0)
        result, pr = CodeQualityMaintainer.improve_code("bug")
        assert "claude crashed" in result and pr is None
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
        result, pr = CodeQualityMaintainer.improve_code("bug")
        assert "git pull failed" in result and pr is None
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
    @pytest.fixture(autouse=True)
    def logged_in(self):
        """_run_claude asks claude_status() first (a `claude auth status` call); these tests are
        about the run itself, with a login in place."""
        with patch('assistant.agents.CodeQualityMaintainer.claude_status',
                   return_value=ClaudeStatus(CLAUDE_PATH, True, "logged in as me@example.com (max)")):
            yield

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    def test_builds_correct_command(self, mock_popen):
        mock_popen.return_value = _mock_popen()
        CodeQualityMaintainer._run_claude("test prompt", "label")
        cmd = mock_popen.call_args[0][0]
        assert cmd == [CLAUDE_PATH, "--permission-mode", "auto", "-p", "test prompt",
                       "--output-format", "stream-json", "--verbose",
                       "--model", "claude-fable-5-2"]
        assert mock_popen.call_args[1]["cwd"] == WORKING_DIR
        assert mock_popen.call_args[1]["env"] is None   # a login keeps its precedence: nothing added

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    def test_local_directories_are_added_as_writable_with_the_checkout_still_the_working_directory(self, mock_popen):
        """Contract with the Claude Code CLI: `--add-dir` takes the extra directories as the
        arguments after it (#156). The run stays in the checkout, so CLAUDE.md applies."""
        mock_popen.return_value = _mock_popen()
        roots = (Path("/home/marcus/KeepLocal"), Path("/home/marcus/KeepExtra"))
        CodeQualityMaintainer._run_claude("test prompt", "label", add_dirs=roots)
        cmd = mock_popen.call_args[0][0]
        assert cmd[cmd.index("--add-dir") + 1:] == ["/home/marcus/KeepLocal", "/home/marcus/KeepExtra"]
        assert cmd[:cmd.index("--add-dir")] == [CLAUDE_PATH, "--permission-mode", "auto", "-p", "test prompt",
                                                "--output-format", "stream-json", "--verbose",
                                                "--model", "claude-fable-5-2"]
        assert mock_popen.call_args[1]["cwd"] == WORKING_DIR

    @patch('assistant.agents.CodeQualityMaintainer.claude_api_key', return_value="sk-ant-test-key")
    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    def test_api_key_stands_in_for_a_login_through_the_environment(self, mock_popen, _key):
        mock_popen.return_value = _mock_popen()
        with patch('assistant.agents.CodeQualityMaintainer.claude_status',
                   return_value=ClaudeStatus(CLAUDE_PATH, True, "no login, using api_keys.anthropic", api_key=True)):
            CodeQualityMaintainer._run_claude("test prompt", "label")
        env = mock_popen.call_args[1]["env"]
        assert env[API_KEY_ENV] == "sk-ant-test-key"
        assert "PATH" in env                                   # the server's own environment, plus the key
        assert "sk-ant-test-key" not in " ".join(mock_popen.call_args[0][0])   # never on the command line

    @patch('assistant.agents.CodeQualityMaintainer.claude_workspace_id', return_value="wrkspc_01TEST")
    @patch('assistant.agents.CodeQualityMaintainer.claude_api_key', return_value="sk-ant-test-key")
    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    def test_a_key_not_tied_to_a_workspace_names_it_to_the_cli_as_a_header(self, mock_popen, _key, _workspace):
        """Contract with the Claude Code CLI: ANTHROPIC_CUSTOM_HEADERS holds "Name: Value" lines."""
        mock_popen.return_value = _mock_popen()
        with patch('assistant.agents.CodeQualityMaintainer.claude_status',
                   return_value=ClaudeStatus(CLAUDE_PATH, True, "no login, using api_keys.anthropic", api_key=True)):
            CodeQualityMaintainer._run_claude("test prompt", "label")
        assert mock_popen.call_args[1]["env"][CUSTOM_HEADERS_ENV] == "anthropic-workspace-id: wrkspc_01TEST"

    @patch('assistant.agents.CodeQualityMaintainer.claude_api_key', return_value="sk-ant-test-key")
    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    def test_a_run_the_api_key_paid_for_says_so_in_its_report(self, mock_popen, _key):
        lines = ['{"type": "result", "result": "Opened PR #12", "total_cost_usd": 3.14159}\n']
        mock_popen.return_value = _mock_popen(lines)
        with patch('assistant.agents.CodeQualityMaintainer.claude_status',
                   return_value=ClaudeStatus(CLAUDE_PATH, True, "no login, using api_keys.anthropic", api_key=True)):
            result = CodeQualityMaintainer._run_claude("prompt", "label")
        assert result.startswith("Opened PR #12\n\n")           # the report itself comes first, untouched
        note = result[len("Opened PR #12"):]
        assert note == api_key_cost_note(3.14159)
        assert "not logged in" in note and "API credits" in note and "expensive" in note
        assert "$3.14" in note                                     # this run's cost, to the cent
        assert "not over SSH" in note and "/login" in note         # the way out

    @patch('assistant.agents.CodeQualityMaintainer.claude_api_key', return_value="sk-ant-test-key")
    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    def test_the_note_leaves_the_cost_out_when_the_cli_reported_none(self, mock_popen, _key):
        lines = ['{"type": "assistant", "message": {"content": [{"type": "text", "text": "partial"}]}}\n']
        mock_popen.return_value = _mock_popen(lines)
        with patch('assistant.agents.CodeQualityMaintainer.claude_status',
                   return_value=ClaudeStatus(CLAUDE_PATH, True, "no login, using api_keys.anthropic", api_key=True)):
            result = CodeQualityMaintainer._run_claude("prompt", "label")
        assert "API credits" in result and "cost $" not in result

    @patch('assistant.agents.CodeQualityMaintainer.claude_api_key', return_value="sk-ant-test-key")
    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    def test_an_empty_run_stays_empty_on_the_api_key(self, mock_popen, _key):
        mock_popen.return_value = _mock_popen(['{"type": "system", "data": "init"}\n'])
        with patch('assistant.agents.CodeQualityMaintainer.claude_status',
                   return_value=ClaudeStatus(CLAUDE_PATH, True, "no login, using api_keys.anthropic", api_key=True)):
            assert CodeQualityMaintainer._run_claude("prompt", "label") == ""   # callers key "no output" off ""

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    def test_a_login_gets_no_cost_note(self, mock_popen):
        lines = ['{"type": "result", "result": "done", "total_cost_usd": 3.14159}\n']
        mock_popen.return_value = _mock_popen(lines)
        assert CodeQualityMaintainer._run_claude("prompt", "label") == "done"   # the figure is notional on a plan

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    def test_no_credentials_raises_before_starting_the_cli(self, mock_popen):
        with patch('assistant.agents.CodeQualityMaintainer.claude_status',
                   return_value=ClaudeStatus(CLAUDE_PATH, False, "no credentials")):
            with pytest.raises(ClaudeNotLoggedInError):
                CodeQualityMaintainer._run_claude("test prompt", "label")
        mock_popen.assert_not_called()

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    def test_manual_permission_mode_in_the_init_event_is_an_error(self, mock_popen):
        lines = ['{"type": "system", "subtype": "init", "permissionMode": "default", "model": "claude-haiku-4-5"}\n',
                 '{"type": "result", "result": "done"}\n']
        mock_popen.return_value = _mock_popen(lines)
        with patch('assistant.agents.CodeQualityMaintainer.error') as mock_error:
            assert CodeQualityMaintainer._run_claude("prompt", "label") == "done"
        assert any("auto mode" in str(c) and "default" in str(c) for c in mock_error.call_args_list)

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    def test_auto_permission_mode_is_not_an_error(self, mock_popen):
        lines = ['{"type": "system", "subtype": "init", "permissionMode": "auto", "model": "claude-fable-5-2"}\n',
                 '{"type": "result", "result": "done"}\n']
        mock_popen.return_value = _mock_popen(lines)
        with patch('assistant.agents.CodeQualityMaintainer.error') as mock_error:
            CodeQualityMaintainer._run_claude("prompt", "label")
        mock_error.assert_not_called()

    def test_result_line_carries_the_cost_the_cli_reports(self):
        line = CodeQualityMaintainer._format_stream_line(
            {"type": "result", "result": "ok", "total_cost_usd": 1.23456, "num_turns": 7, "duration_ms": 65000})
        assert "$1.2346" in line and "7 turns" in line and "65 s" in line
        legacy = CodeQualityMaintainer._format_stream_line({"type": "result", "result": "ok", "cost_usd": 0.5})
        assert "$0.5000" in legacy

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
        message = str(exc.value)
        assert "api_keys.anthropic" in message and "/login" in message   # both ways to give it credentials
        assert "SSH" in message

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
    def test_a_flagged_result_is_a_failed_run_not_a_report(self, mock_popen):
        """Taken as a report, the CLI's reason stood as the night's bug analysis."""
        lines = ['{"type": "result", "is_error": true, "result": "Rate limit reached, try again later"}\n']
        mock_popen.return_value = _mock_popen(lines)
        with pytest.raises(ClaudeRunFailedError) as exc:
            CodeQualityMaintainer._run_claude("prompt", "bug-analysis")
        assert "Rate limit reached" in str(exc.value) and "bug-analysis" in str(exc.value)
        assert not exc.value.stops_the_night   # the next step may still run

    @pytest.mark.parametrize("flagged", [True, False])
    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    def test_the_out_of_credits_answer_stops_the_night_whether_or_not_the_cli_flags_it(self, mock_popen, flagged):
        text = ("You're out of usage credits. Switch to another model, or manage usage credits at "
                "https://claude.ai/settings/usage, to continue.")
        lines = [json.dumps({"type": "result", "is_error": flagged, "result": text}) + "\n"]
        mock_popen.return_value = _mock_popen(lines)
        with pytest.raises(ClaudeUsageLimitError) as exc:
            CodeQualityMaintainer._run_claude("prompt", "log-triage")
        assert exc.value.stops_the_night
        assert "out of usage credits" in str(exc.value)

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    def test_a_report_that_discusses_usage_limits_is_a_report(self, mock_popen):
        """The unflagged marker check is for the CLI's one-line message, not a report's length."""
        report = ("Bug: the retry loop never gives up on a spend limit error from the API. "
                  + "The loop at Retry.py:12 reads the status and tries again. " * 8)
        lines = [json.dumps({"type": "result", "is_error": False, "result": report}) + "\n"]
        mock_popen.return_value = _mock_popen(lines)
        assert CodeQualityMaintainer._run_claude("prompt", "label").startswith("Bug: the retry loop")

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    def test_a_model_that_requires_usage_credits_stops_the_night(self, mock_popen):
        lines = [json.dumps({"type": "result", "is_error": True,
                             "result": "Fable 5.1 requires usage credits."}) + "\n"]
        mock_popen.return_value = _mock_popen(lines)
        with pytest.raises(ClaudeUsageLimitError) as exc:
            CodeQualityMaintainer._run_claude("prompt", "bug-analysis")
        assert exc.value.stops_the_night

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    def test_stderr_is_drained_while_stdout_is_still_open(self, mock_popen):
        """A CLI filling the stderr pipe blocks until it is read: stdout must not have to end first."""
        stderr_read = threading.Event()

        def stdout_lines():
            assert stderr_read.wait(2), "stderr is only read after stdout ends — a full pipe deadlocks"
            yield '{"type": "result", "result": "done"}\n'

        proc = _mock_popen()
        proc.stdout = stdout_lines()
        proc.stderr.read.side_effect = lambda: (stderr_read.set(), "x" * 100_000)[1]
        mock_popen.return_value = proc
        assert CodeQualityMaintainer._run_claude("prompt", "label") == "done"

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    def test_a_run_killed_at_the_timeout_fails_instead_of_returning_its_last_text(self, mock_popen):
        lines = ['{"type": "assistant", "message": {"content": [{"type": "text", "text": "Looking at"}]}}\n']
        proc = _mock_popen(lines)
        mock_popen.return_value = proc
        fire_at_once = lambda _interval, fn: Mock(start=fn)
        with patch('assistant.agents.CodeQualityMaintainer.threading.Timer', side_effect=fire_at_once):
            with pytest.raises(ClaudeRunFailedError) as exc:
                CodeQualityMaintainer._run_claude("prompt", "bug-analysis")
        proc.kill.assert_called_once()
        assert "timed out" in str(exc.value) and not exc.value.stops_the_night

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
    """The report goes out through the account send_email sends from (`mail.send_from`, else the
    first configured one), to that account's own address, so it gets the same SMTP handling and
    Sent-folder treatment as every other mail the server sends."""

    MAIL_ACCESS = 'assistant.integration.mail.MailAccess.MailAccess.current'

    @staticmethod
    def _sender(current, address="m@mlab.test", name="Personal", result=""):
        account = current.return_value.sender.return_value
        account.address = address
        account.name = name
        account.send_message.return_value = result
        return account

    def test_report_is_mailed_from_the_sending_account_to_its_own_address(self, maintainer):
        with patch(self.MAIL_ACCESS) as current:
            account = self._sender(current)
            CodeQualityMaintainer._send_email("Subject", "<p>Body</p>")
            current.return_value.sender.assert_called_once_with()
            msg, recipients = account.send_message.call_args[0]
        assert recipients == ["m@mlab.test"]
        assert msg["To"] == "m@mlab.test"
        assert msg["Subject"] == "Subject"
        assert "<p>Body</p>" in msg.get_payload()[0].get_payload()

    def test_attachments_ride_along_as_files_after_the_html_body(self, maintainer):
        with patch(self.MAIL_ACCESS) as current:
            account = self._sender(current)
            address = CodeQualityMaintainer._send_email("Subject", "<p>Body</p>", [("prompt.md", "# [SYSTEM]\nhi")])
            msg, _ = account.send_message.call_args[0]
        assert address == "m@mlab.test"
        html_part, attachment = msg.get_payload()
        assert "<p>Body</p>" in html_part.get_payload()
        assert attachment.get_filename() == "prompt.md"
        assert attachment.get_payload(decode=True).decode() == "# [SYSTEM]\nhi"

    def test_a_refused_send_raises(self, maintainer):
        with patch(self.MAIL_ACCESS) as current:
            self._sender(current, result="SMTP error: boom")
            with pytest.raises(RuntimeError, match="boom"):
                CodeQualityMaintainer._send_email("Subject", "<p>Body</p>")

    def test_without_a_mail_account_nothing_is_sent(self, maintainer):
        with patch(self.MAIL_ACCESS) as current:
            current.return_value.sender.return_value = None
            CodeQualityMaintainer._send_email("Subject", "<p>Body</p>")

    def test_an_account_without_an_address_sends_nothing(self, maintainer):
        with patch(self.MAIL_ACCESS) as current:
            account = self._sender(current, address="")
            CodeQualityMaintainer._send_email("Subject", "<p>Body</p>")
            account.send_message.assert_not_called()

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
        with patch.object(CodeQualityMaintainer, '_step_log_triage', return_value="") as logs, \
                patch.object(CodeQualityMaintainer, '_step_fix_bugs', return_value="") as fix, \
                patch.object(CodeQualityMaintainer, '_step_test_health', return_value="") as health, \
                patch.object(CodeQualityMaintainer, '_step_auto_merge', return_value="") as merge, \
                patch.object(CodeQualityMaintainer, '_step_open_pins', return_value="") as pins:
            self.logs, self.fix, self.health, self.merge, self.pins = logs, fix, health, merge, pins
            yield

    @patch.object(CodeQualityMaintainer, '_has_recent_commits', return_value=True)
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_runs_the_seven_steps_in_order_with_the_tests_first_the_fix_after_the_reviews_and_the_pins_last(
            self, mock_sub, mock_commits, maintainer):
        call_order = []
        self.logs.side_effect = lambda: call_order.append('logs') or ''
        self.fix.side_effect = lambda analysis: call_order.append('fix') or ''
        self.health.side_effect = lambda: call_order.append('health') or ''
        self.merge.side_effect = lambda: call_order.append('merge') or ''
        self.pins.side_effect = lambda: call_order.append('pins') or ''
        with patch.object(CodeQualityMaintainer, '_step_bug_analysis', side_effect=lambda: call_order.append('bugs') or ''):
            with patch.object(CodeQualityMaintainer, '_step_generate_tests', side_effect=lambda: call_order.append('tests') or ''):
                with patch.object(CodeQualityMaintainer, '_send_email'):
                    maintainer._execute_nightly_run()
        assert call_order == ['tests', 'bugs', 'logs', 'fix', 'health', 'merge', 'pins']
        assert list(NIGHTLY_STEPS) == call_order   # //nightly names them the same way

    @patch.object(CodeQualityMaintainer, '_has_recent_commits', return_value=True)
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_the_email_puts_the_findings_first_the_fixes_after_them_and_the_tests_last(
            self, mock_sub, mock_commits, maintainer):
        self.logs.return_value = "a stall"
        self.fix.return_value = "PR #12"
        self.health.return_value = "test_slow"
        self.merge.return_value = "merged #11"
        self.pins.return_value = "| 1 | test_x | a word is lost | trivial |"
        with patch.object(CodeQualityMaintainer, '_step_bug_analysis', return_value="off by one"), \
                patch.object(CodeQualityMaintainer, '_step_generate_tests', return_value="pinned"), \
                patch.object(CodeQualityMaintainer, '_markdown_to_html', side_effect=lambda md: md), \
                patch.object(CodeQualityMaintainer, '_send_email') as send:
            maintainer._execute_nightly_run()
        text = send.call_args[0][1]
        headings = [line[2:] for line in text.splitlines() if line.startswith("# ")]
        assert headings == ["Bug Analysis", "Log Triage", "Bug Fixes", "Test Generation", "Test Health",
                            "Auto-Merge Test PRs", "Open Pins"]
        assert text.index("off by one") < text.index("a stall") < text.index("PR #12") < text.index("pinned")
        assert text.index("merged #11") < text.index("| 1 | test_x |")

    @patch.object(CodeQualityMaintainer, '_has_recent_commits', return_value=True)
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_the_fix_step_reads_the_bugs_the_new_tests_pinned(self, mock_sub, mock_commits, maintainer):
        with patch.object(CodeQualityMaintainer, '_step_bug_analysis', return_value="| Major | foo.py:3 | off by one |"), \
                patch.object(CodeQualityMaintainer, '_step_generate_tests',
                             return_value="| Major | baz.py:7 | test_x pins it |"), \
                patch.object(CodeQualityMaintainer, '_send_email'):
            maintainer._execute_nightly_run()
        text = self.fix.call_args[0][0]
        assert text.index("| Major | foo.py:3 |") < text.index("# Test Generation") < text.index("| Major | baz.py:7 |")

    @patch.object(CodeQualityMaintainer, '_has_recent_commits', return_value=True)
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_the_fix_step_reads_the_nights_bug_analysis_and_log_triage_under_their_headings(
            self, mock_sub, mock_commits, maintainer):
        self.logs.return_value = "| Critical | bar.py:9 | traceback at startup |"
        with patch.object(CodeQualityMaintainer, '_step_bug_analysis', return_value="| Major | foo.py:3 | off by one |"), \
                patch.object(CodeQualityMaintainer, '_step_generate_tests', return_value=""), \
                patch.object(CodeQualityMaintainer, '_send_email'):
            maintainer._execute_nightly_run()
        self.fix.assert_called_once()
        text = self.fix.call_args[0][0]
        bugs, logs = text.index("| Major | foo.py:3 |"), text.index("| Critical | bar.py:9 |")
        assert text.index("# Bug Analysis") < bugs < text.index("# Log Triage") < logs

    @patch.object(CodeQualityMaintainer, '_has_recent_commits', return_value=True)
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_out_of_credits_at_the_bug_analysis_skips_the_later_claude_steps_and_the_report_says_so(
            self, mock_sub, mock_commits, maintainer):
        """2026-10-22: the four later steps each came back with the credits line in a second, the
        fix step read "nothing Critical or Major" in them and the night mailed "nothing to fix"."""
        self.logs.return_value = "| Major | bar.py:9 | a feed stalled |"
        self.merge.return_value = "merged #11"
        limit = ClaudeUsageLimitError("bug-analysis", "You're out of usage credits.")
        with patch.object(CodeQualityMaintainer, '_step_bug_analysis', side_effect=limit), \
                patch.object(CodeQualityMaintainer, '_step_generate_tests', return_value="pinned"), \
                patch.object(CodeQualityMaintainer, '_markdown_to_html', side_effect=lambda md: md), \
                patch.object(CodeQualityMaintainer, '_send_email') as send:
            maintainer._execute_nightly_run()
        for step in (self.logs, self.fix, self.health, self.pins):
            step.assert_not_called()
        self.merge.assert_called_once()   # no Claude in it
        text = send.call_args[0][1]
        assert "out of usage credits" in text and "bug analysis" in text
        assert text.index("# Log Triage") < text.index("Not run") < text.index("merged #11")
        assert "a feed stalled" not in text

    @patch.object(CodeQualityMaintainer, '_has_recent_commits', return_value=True)
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_a_failed_run_that_is_not_the_credits_leaves_the_night_going_and_names_itself_to_the_fix_step(
            self, mock_sub, mock_commits, maintainer):
        self.logs.return_value = "| Major | bar.py:9 | a feed stalled |"
        failed = ClaudeRunFailedError("bug-analysis", "Reached max turns (50)")
        with patch.object(CodeQualityMaintainer, '_step_bug_analysis', side_effect=failed), \
                patch.object(CodeQualityMaintainer, '_step_generate_tests', return_value=""), \
                patch.object(CodeQualityMaintainer, '_send_email'):
            maintainer._execute_nightly_run()
        self.fix.assert_called_once()
        text = self.fix.call_args[0][0]
        assert text.index("# Bug Analysis") < text.index("max turns") < text.index("# Log Triage") < text.index("a feed stalled")
        self.health.assert_called_once()
        self.pins.assert_called_once()

    @patch.object(CodeQualityMaintainer, '_has_recent_commits', return_value=True)
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_a_crashed_bug_analysis_leaves_the_fix_step_only_the_log_triage(self, mock_sub, mock_commits, maintainer):
        self.logs.return_value = "| Major | bar.py:9 | a feed stalled |"
        with patch.object(CodeQualityMaintainer, '_step_bug_analysis', side_effect=Exception("boom")), \
                patch.object(CodeQualityMaintainer, '_step_generate_tests', return_value=""), \
                patch.object(CodeQualityMaintainer, '_send_email'):
            maintainer._execute_nightly_run()
        text = self.fix.call_args[0][0]
        assert "| Major | bar.py:9 |" in text and "Bug Analysis" not in text

    @patch.object(CodeQualityMaintainer, '_has_recent_commits', return_value=True)
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_two_reports_without_content_leave_the_fix_step_nothing_to_read(self, mock_sub, mock_commits, maintainer):
        with patch.object(CodeQualityMaintainer, '_step_bug_analysis', side_effect=Exception("boom")), \
                patch.object(CodeQualityMaintainer, '_step_generate_tests', return_value=""), \
                patch.object(CodeQualityMaintainer, '_send_email'):
            maintainer._execute_nightly_run()
        self.fix.assert_called_once_with("")

    @patch.object(CodeQualityMaintainer, '_has_recent_commits', return_value=False)
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_the_fix_step_alone_runs_the_bug_analysis_and_the_log_triage_first(self, mock_sub, mock_commits, maintainer):
        with patch.object(CodeQualityMaintainer, '_step_bug_analysis', return_value="Critical: x") as bugs, \
                patch.object(CodeQualityMaintainer, '_step_generate_tests') as tests, \
                patch.object(CodeQualityMaintainer, '_send_email'):
            maintainer._execute_nightly_run(only="fix")
        bugs.assert_called_once()
        self.logs.assert_called_once()
        self.fix.assert_called_once()
        assert "Critical: x" in self.fix.call_args[0][0]
        tests.assert_not_called()
        self.merge.assert_not_called()

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
    def test_a_night_without_recent_commits_skips_the_steps_about_the_commits(self, mock_sub, mock_commits, maintainer):
        """The logs are about the running system, so the triage, the fix it feeds, the auto-merge
        and the open pins still run; the review, the tests and the test health of the commits do not."""
        self.logs.return_value = "| Major | bar.py:9 | a feed stalled |"
        with patch.object(CodeQualityMaintainer, '_step_bug_analysis') as mock_bugs, \
                patch.object(CodeQualityMaintainer, '_step_generate_tests') as mock_tests, \
                patch.object(CodeQualityMaintainer, '_send_email'):
            maintainer._execute_nightly_run()
        mock_bugs.assert_not_called()
        mock_tests.assert_not_called()
        self.health.assert_not_called()
        self.logs.assert_called_once()
        assert "| Major | bar.py:9 |" in self.fix.call_args[0][0]
        self.merge.assert_called_once()
        self.pins.assert_called_once()

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
    @patch.object(Updater, 'current_branch', return_value="master")
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_pulls_before_checking_commits(self, mock_sub, mock_branch, mock_commits, maintainer):
        with patch.object(CodeQualityMaintainer, '_step_bug_analysis', return_value=""):
            with patch.object(CodeQualityMaintainer, '_step_generate_tests', return_value=""):
                maintainer._execute_nightly_run()
                mock_sub.assert_called_once_with(["git", "pull"], "git pull", GIT_TIMEOUT)

    @patch.object(CodeQualityMaintainer, '_has_recent_commits', return_value=True)
    @patch.object(Updater, 'current_branch', return_value="release")
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_on_release_the_night_starts_from_the_signed_release_not_a_plain_pull(
            self, mock_sub, mock_branch, mock_commits, maintainer, signed_release):
        signed_release.return_value = ("The newest release carries no tag signed by a key in release_signers", "")
        with patch.object(CodeQualityMaintainer, '_step_bug_analysis', return_value=""):
            with patch.object(CodeQualityMaintainer, '_step_generate_tests', return_value=""):
                maintainer._execute_nightly_run()
        signed_release.assert_called_once_with("release")
        assert ["git", "pull"] not in [c.args[0] for c in mock_sub.call_args_list]


def developer_release(value: bool):
    """Pins the `developer_release` key: the developer's own config.yaml has it true, a test
    must not inherit that."""
    return patch.object(ConfigManager, 'get',
                        side_effect=lambda key, default=None: value if key == "developer_release" else default)


@pytest.mark.unit
class TestUpdaterDelegates:
    """The update machinery is common/Updater's (tests/unit/common/test_updater.py); the maintainer
    keeps the names its callers use and adds the branch `developer_release` names."""

    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_the_delegates_run_git_through_the_maintainer_s_own_runner(self, mock_sub):
        """The jobs, PullRequestTrial and their tests fake git at `_run_subprocess`: a delegate
        running git any other way would reach the real checkout past every one of those fakes."""
        mock_sub.return_value = Mock(stdout="release\n", stderr="", returncode=0)
        assert CodeQualityMaintainer._current_branch() == "release"
        mock_sub.return_value = Mock(stdout=NEWER + "\n", stderr="", returncode=0)
        assert CodeQualityMaintainer._head_commit() == NEWER
        mock_sub.return_value = Mock(stdout="Already up to date.\n", stderr="", returncode=0)
        with patch.object(Updater, 'current_branch', return_value="master"):
            CodeQualityMaintainer._pull_current()
        assert mock_sub.call_args.args[0] == ["git", "pull"]

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    @patch.object(Updater, 'sync', return_value=("", ""))
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_the_pull_goes_to_the_branch_the_config_key_names(self, mock_sub, mock_sync, mock_popen, signed_release):
        mock_sub.return_value = Mock(stdout="Already up to date.\n", stderr="", returncode=0)
        with developer_release(True), patch.object(Updater, 'current_branch', return_value="release"):
            CodeQualityMaintainer.pull_and_restart()
        assert [c.args[0] for c in mock_sub.call_args_list] == [
            ["git", "fetch", "origin", "master"], ["git", "checkout", "master"], ["git", "pull"]]
        signed_release.assert_not_called()
        with developer_release(False), patch.object(Updater, 'current_branch', return_value="master"):
            CodeQualityMaintainer.pull_and_restart()
        signed_release.assert_called_once_with("master")


@pytest.mark.unit
class TestPullAndRestart:
    @pytest.fixture(autouse=True)
    def venv_in_sync(self):
        """The venv matches the pins unless a test says otherwise — nothing here runs pip."""
        with patch.object(Updater, 'sync', return_value=("", "")) as sync:
            self.sync = sync
            yield

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_a_pin_bump_is_installed_after_the_pull_and_before_the_restart(self, mock_sub, mock_popen):
        mock_sub.return_value = Mock(stdout="Updating abc123..def456\n1 file changed", stderr="", returncode=0)
        order = []
        self.sync.side_effect = lambda: (order.append("pip"), ("", "installed 1 package: pkg-a (missing)"))[1]
        mock_popen.side_effect = lambda *a, **k: order.append("restart")
        with developer_release(True):
            result = CodeQualityMaintainer.pull_and_restart()
        assert order == ["pip", "restart"]
        assert "pkg-a" in result

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_a_failed_install_does_not_restart_and_says_why(self, mock_sub, mock_popen):
        mock_sub.return_value = Mock(stdout="Updating abc123..def456\n1 file changed", stderr="", returncode=0)
        self.sync.return_value = ("pip exited 1 (no matching distribution)", "")
        with developer_release(True):
            result = CodeQualityMaintainer.pull_and_restart()
        assert "not restarted" in result and "no matching distribution" in result
        mock_popen.assert_not_called()

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_a_failed_pull_never_reaches_pip(self, mock_sub, mock_popen):
        mock_sub.return_value = Mock(stdout="", stderr="fatal: not a git repository", returncode=128)
        with developer_release(True):
            CodeQualityMaintainer.pull_and_restart()
        self.sync.assert_not_called()
        mock_popen.assert_not_called()

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_success_with_changes(self, mock_sub, mock_popen):
        mock_sub.return_value = Mock(stdout="Updating abc123..def456\n1 file changed", stderr="", returncode=0)
        with developer_release(True):
            result = CodeQualityMaintainer.pull_and_restart()
        assert "Update initiated" in result
        assert "pulled updates" in result
        mock_popen.assert_called_once()
        popen_kwargs = mock_popen.call_args[1]
        assert popen_kwargs["cwd"] == str(AISERVER_DIR)
        assert popen_kwargs["start_new_session"] is True

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_already_up_to_date(self, mock_sub, mock_popen):
        mock_sub.return_value = Mock(stdout="Already up to date.", stderr="", returncode=0)
        with developer_release(True):
            result = CodeQualityMaintainer.pull_and_restart()
        assert "no new changes" in result
        mock_popen.assert_called_once()

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    @patch.object(Updater, 'current_branch', return_value="master")
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_git_pull_failure_does_not_restart(self, mock_sub, mock_branch, mock_popen):
        mock_sub.return_value = Mock(stdout="", stderr="fatal: not a git repository", returncode=128)
        with developer_release(True):
            result = CodeQualityMaintainer.pull_and_restart()
        assert "not a git repository" in result
        mock_popen.assert_not_called()

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    @patch.object(Updater, 'current_branch', return_value="release")
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_a_release_that_fails_verification_does_not_restart(self, mock_sub, mock_branch, mock_popen,
                                                                 signed_release):
        refusal = "The newest release (1b6c9ce7) carries no tag signed by a key in release_signers"
        signed_release.return_value = (refusal, "")
        with developer_release(False):
            result = CodeQualityMaintainer.pull_and_restart()
        assert "1b6c9ce7" in result
        self.sync.assert_not_called()
        mock_popen.assert_not_called()

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    @patch.object(Updater, 'current_branch', return_value="release")
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_release_is_updated_only_through_the_signed_release(self, mock_sub, mock_branch, mock_popen,
                                                                 signed_release):
        with developer_release(False):
            CodeQualityMaintainer.pull_and_restart()
        signed_release.assert_called_once_with("release")
        mock_sub.assert_not_called()
        mock_popen.assert_called_once()

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    @patch.object(Updater, 'current_branch', return_value="release")
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_failed_switch_leaves_the_branch_and_does_not_restart(self, mock_sub, mock_branch, mock_popen):
        def git(cmd, label, timeout):
            if cmd[:2] == ["git", "checkout"]:
                return Mock(stdout="", stderr="error: your local changes would be overwritten", returncode=1)
            return Mock(stdout="", stderr="", returncode=0)
        mock_sub.side_effect = git
        with developer_release(True):
            result = CodeQualityMaintainer.pull_and_restart()
        assert "master" in result and "still on release" in result
        assert ["git", "pull"] not in [c[0][0] for c in mock_sub.call_args_list]
        mock_popen.assert_not_called()

    def test_target_branch_follows_the_config_key(self):
        with developer_release(False):
            assert CodeQualityMaintainer.target_branch() == "release"
        with developer_release(True):
            assert CodeQualityMaintainer.target_branch() == "master"

    @patch('common.StartupTracker.StartupTracker.write_resume_marker')
    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_a_restart_asked_for_in_a_chat_leaves_the_resume_marker_before_spawning(self, mock_sub, mock_popen, marker):
        mock_sub.return_value = Mock(stdout="Updating abc123..def456\n1 file changed", stderr="", returncode=0)
        order = []
        marker.side_effect = lambda reason: order.append(("marker", reason))
        mock_popen.side_effect = lambda *a, **k: order.append(("spawn", None))
        with developer_release(True):
            CodeQualityMaintainer.pull_and_restart(resumes_conversation=True)
        assert [step for step, _ in order] == ["marker", "spawn"]
        assert "abc123..def456" in order[0][1], "the reason tells the new process what the pull brought"

    @patch('common.StartupTracker.StartupTracker.write_resume_marker')
    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_a_restart_nobody_waits_on_leaves_no_marker(self, mock_sub, mock_popen, marker):
        mock_sub.return_value = Mock(stdout="Already up to date.", stderr="", returncode=0)
        CodeQualityMaintainer.pull_and_restart()
        CodeQualityMaintainer.restart("nightly update abc → def")
        assert mock_popen.call_count == 2
        marker.assert_not_called()

    @patch('common.StartupTracker.StartupTracker.write_resume_marker')
    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    @patch.object(CodeQualityMaintainer, '_run_subprocess')
    def test_a_failed_pull_leaves_no_stale_marker(self, mock_sub, mock_popen, marker):
        mock_sub.return_value = Mock(stdout="", stderr="fatal: not a git repository", returncode=128)
        with developer_release(True):
            CodeQualityMaintainer.pull_and_restart(resumes_conversation=True)
        mock_popen.assert_not_called()
        marker.assert_not_called()

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


class TestReportOutcome:
    """What cmd_self_improvement reads off a Claude Code report: the title line. The PR is
    not read off the text (`TestJobPullRequest`)."""

    def test_the_first_line_is_undressed_markdown(self):
        assert CodeQualityMaintainer.first_line("\n\n## **Timers keep their reason**\nbody") == "Timers keep their reason"
        assert CodeQualityMaintainer.first_line("- *done*: fixed `it`\n") == "done: fixed it"
        assert CodeQualityMaintainer.first_line("   \n") == "(empty report)"


def _file(name, status="modified", deletions=0, previous=None):
    return {"filename": name, "previous_filename": previous, "status": status, "deletions": deletions}


@pytest.mark.unit
class TestScopeViolations:
    """What improve_integration may deliver: one integration folder's files, or a new folder
    plus additions to the registry files — checked on the PR after the run."""

    def test_a_change_inside_the_named_folder_and_its_tests_passes(self):
        files = [_file("src/integrations/tesla/TeslaCommands.py", deletions=4),
                 _file("tests/unit/integrations/tesla/test_tesla_commands.py", "added")]
        assert CodeQualityMaintainer.scope_violations(files, "tesla") == []

    def test_a_core_file_fails_even_beside_a_folder_change(self):
        files = [_file("src/integrations/tesla/TeslaCommands.py"), _file("src/assistant/Assistant.py")]
        violations = CodeQualityMaintainer.scope_violations(files, "tesla")
        assert len(violations) == 1 and "src/assistant/Assistant.py" in violations[0]

    def test_another_folder_fails_for_a_fix(self):
        violations = CodeQualityMaintainer.scope_violations([_file("src/integrations/hue/HueCommands.py")], "tesla")
        assert any("hue" in v and "tesla" in v for v in violations)

    def test_a_file_moved_in_from_outside_counts_under_its_old_path(self):
        files = [_file("src/integrations/tesla/Parser.py", "renamed", previous="src/common/Parser.py")]
        assert any("src/common/Parser.py" in v for v in CodeQualityMaintainer.scope_violations(files, "tesla"))

    def test_a_registry_file_fails_for_a_fix(self):
        files = [_file("src/integrations/tesla/TeslaCommands.py"), _file("readme.md")]
        assert any("readme.md" in v for v in CodeQualityMaintainer.scope_violations(files, "tesla"))

    def test_a_new_folder_with_registry_additions_passes(self):
        files = [_file("src/integrations/ring/RingIntegration.py", "added"),
                 _file("src/integrations/ring/toolkit.md", "added"),
                 _file("tests/unit/integrations/ring/test_ring.py", "added"),
                 _file("docs/atlas/descriptions.py"), _file("requirements.txt"), _file("readme.md")]
        assert CodeQualityMaintainer.scope_violations(files, None) == []

    def test_a_new_integration_may_not_change_an_existing_folder_or_remove_registry_lines(self):
        files = [_file("src/integrations/hue/HueCommands.py"), _file("config.example.yaml", deletions=2)]
        violations = CodeQualityMaintainer.scope_violations(files, None)
        assert any("existing integration" in v for v in violations)
        assert any("config.example.yaml" in v for v in violations)

    def test_two_folders_fail_for_a_new_integration(self):
        files = [_file("src/integrations/ring/RingIntegration.py", "added"),
                 _file("src/integrations/arlo/ArloIntegration.py", "added")]
        assert any("several" in v for v in CodeQualityMaintainer.scope_violations(files, None))

    def test_an_empty_pr_fails(self):
        assert CodeQualityMaintainer.scope_violations([], "tesla")


JOB_PR = JobPullRequest(31, "https://github.com/example/keep/pull/31", "tesla-trunk-once")


@pytest.mark.unit
class TestImproveIntegration:
    REPORT = "Done: https://github.com/example/keep/pull/31\n\nTesla: close the trunk once"

    @staticmethod
    def _run(folder, files, report=None, pr=JOB_PR, existing=("tesla", "music", "microsoft")):
        with patch.object(CodeQualityMaintainer, "_github_access", return_value=(PUSHER, None)), \
                patch.object(CodeQualityMaintainer, "_version", return_value="v2026.10.1"), \
                patch.object(CodeQualityMaintainer, "_pr_job",
                             return_value=(report or TestImproveIntegration.REPORT, pr)) as job, \
                patch.object(CodeQualityMaintainer, "_pr_files", return_value=("Tesla: trunk", files)) as pr_files, \
                patch.object(CodeQualityMaintainer, "_existing_integration_folders", return_value=set(existing)), \
                patch.object(CodeQualityMaintainer, "_close_pr") as close:
            result, left_open = CodeQualityMaintainer.improve_integration("Tesla", folder, "close the trunk once")
        job.pr_files = pr_files
        return result, job, close, left_open

    def test_a_pr_inside_the_folder_is_left_open_and_reported_as_is(self):
        result, job, close, left_open = self._run("tesla", [_file("src/integrations/tesla/TeslaCommands.py")])
        assert result == self.REPORT and left_open == JOB_PR
        close.assert_not_called()
        prompt = job.call_args.args[0]
        assert "src/integrations/tesla/" in prompt and "close the trunk once" in prompt

    def test_a_straying_pr_is_closed_and_the_user_sent_to_report_issue(self):
        result, _, close, left_open = self._run("tesla", [_file("src/assistant/Assistant.py")])
        close.assert_called_once()
        assert close.call_args.args[0] == 31 and "src/assistant/Assistant.py" in close.call_args.args[1]
        assert "closed" in result and "report_issue" in result and self.REPORT in result
        assert left_open is None, "a closed PR is not one to try"
        assert JOB_PR.head in result, "the user is told the branch survives the closing"

    def test_a_report_that_mentions_another_pr_closes_nothing_when_the_job_opened_none(self):
        """2026-10-26 08:11: the job declined and echoed "PR #120" from the request; the
        developers' PR #120 was closed and its branch deleted. The PR is the one the job
        opened — none here — and the report's text does not count."""
        report = ("I didn't build anything or open a PR. Keep already has a Spotify integration, the music folder. "
                  "The same pattern as the Microsoft 365 retry (PR #120, https://github.com/example/keep/pull/120).")
        result, job, close, left_open = self._run(None, [_file("src/integrations/microsoft/GraphClient.py")],
                                                  report=report, pr=None)
        close.assert_not_called()
        job.pr_files.assert_not_called()
        assert result == report and left_open is None

    def test_a_fix_confined_to_an_existing_folder_passes_even_without_a_resolved_folder(self):
        """2026-10-26 22:00: "music" resolved to no integration (the name is "Sonos & Spotify"),
        the job ran in new-integration mode, changed the music folder alone and was closed for
        "changing an existing integration". A PR inside one existing folder is that folder's."""
        files = [_file("src/integrations/music/MusicIntegration.py"),
                 _file("tests/unit/integrations/music/test_music_manager.py")]
        result, _, close, left_open = self._run(None, files, pr=JobPullRequest(132, "u", "music-retry"))
        close.assert_not_called()
        assert result == self.REPORT and left_open.number == 132

    def test_a_new_integration_in_a_folder_the_checkout_lacks_is_still_judged_as_new(self):
        files = [_file("src/integrations/ring/RingIntegration.py"), _file("src/integrations/ring/toolkit.md", "added")]
        _, _, close, _ = self._run(None, files, existing=("tesla",))
        close.assert_called_once()
        assert "existing integration" in close.call_args.args[1]

    def test_a_change_to_an_existing_folder_and_a_registry_file_stays_a_straying_one(self):
        files = [_file("src/integrations/music/MusicIntegration.py"), _file("requirements.txt")]
        _, _, close, _ = self._run(None, files)
        close.assert_called_once()

    def test_a_pr_whose_files_cannot_be_read_is_closed_too(self):
        with patch.object(CodeQualityMaintainer, "_github_access", return_value=(PUSHER, None)), \
                patch.object(CodeQualityMaintainer, "_pr_job", return_value=(self.REPORT, JOB_PR)), \
                patch.object(CodeQualityMaintainer, "_pr_files", side_effect=RuntimeError("gh: rate limited")), \
                patch.object(CodeQualityMaintainer, "_close_pr") as close:
            result, _ = CodeQualityMaintainer.improve_integration("Tesla", "tesla", "close the trunk once")
        close.assert_called_once()
        assert "rate limited" in result

    def test_the_tried_pr_is_never_closed_for_straying(self):
        # During a trial the job pushes onto the tried PR, whose files are the developers'
        # whole change; closing it would end the trial the server runs.
        from assistant.agents.PullRequestTrial import PullRequestTrial, Trial
        trial = Trial(number=31, title="core refactor", head_ref="feat/core", since="14:02", return_branch="release")
        with patch.object(PullRequestTrial, "current", return_value=trial):
            result, _, close, _ = self._run("tesla", [_file("src/assistant/Assistant.py")])
        close.assert_not_called()
        assert result == self.REPORT
        other = Trial(number=30, title="t", head_ref="feat/x", since="14:02", return_branch="release")
        with patch.object(PullRequestTrial, "current", return_value=other):
            _, _, close, _ = self._run("tesla", [_file("src/assistant/Assistant.py")])
        close.assert_called_once()

    def test_a_declined_request_opens_nothing_to_check(self):
        result, job, close, _ = self._run("tesla", [], report="Declined: the request names no symptom.", pr=None)
        assert result.startswith("Declined") and not close.called
        job.pr_files.assert_not_called()

    def test_a_new_integration_gets_the_recipe_and_the_registry_files(self):
        _, job, _, _ = self._run(None, [_file("src/integrations/ring/RingIntegration.py", "added")], existing=())
        prompt = job.call_args.args[0]
        assert "Adding an integration" in prompt and "docs/atlas/descriptions.py" in prompt

    def test_without_gh_nothing_runs(self):
        with patch.object(CodeQualityMaintainer, "_github_access", return_value=(None, "gh is out")), \
                patch.object(CodeQualityMaintainer, "_pr_job") as job:
            assert CodeQualityMaintainer.improve_integration("Tesla", "tesla", "x") == ("gh is out", None)
            assert CodeQualityMaintainer.improve_code("x") == ("gh is out", None)
        job.assert_not_called()

    def test_the_jobs_hand_the_account_to_the_pr_job(self):
        """The PR job needs the account to tell whether the PR comes from a fork."""
        _, job, _, _ = self._run("tesla", [_file("src/integrations/tesla/TeslaCommands.py")])
        assert PUSHER in job.call_args.args


def _prs(*entries):
    return {number: JobPullRequest(number, f"https://github.com/example/keep/pull/{number}", head)
            for number, head in entries}


@pytest.mark.unit
class TestJobPullRequest:
    """Which PR a job is held to have opened: one open after the run that was not open before
    it, matched by the branch the run left the tree on. Never one the report mentions."""

    @staticmethod
    def _find(before, after, work_branch=None, trial=None, login=None):
        from assistant.agents.PullRequestTrial import PullRequestTrial
        with patch.object(CodeQualityMaintainer, "_open_pull_requests", return_value=after), \
                patch.object(PullRequestTrial, "current", return_value=trial):
            return CodeQualityMaintainer._job_pull_request("improve-integration Tesla", before, work_branch, login)

    def test_a_new_pr_by_another_account_is_never_the_jobs(self):
        """On a public repository anyone's PR can open during the hour a job runs."""
        theirs = {133: JobPullRequest(133, "u", "tesla-trunk", "dave")}
        assert self._find({}, theirs, login="Alice") is None
        assert self._find({}, theirs, "tesla-trunk", login="alice") is None
        both = {**theirs, 134: JobPullRequest(134, "u", "hue-dim", "alice")}
        assert self._find({}, both, login="Alice") == both[134]

    def test_the_new_pr_from_the_jobs_branch_is_the_jobs(self):
        before, after = _prs((120, "microsoft-retry")), _prs((120, "microsoft-retry"), (133, "tesla-trunk"))
        assert self._find(before, after, "tesla-trunk") == after[133]

    def test_a_pr_open_before_the_run_is_never_the_jobs(self):
        before = _prs((120, "microsoft-retry"))
        assert self._find(before, dict(before), "microsoft-retry") is None
        assert self._find(before, dict(before)) is None

    def test_a_new_pr_from_another_branch_is_not_the_jobs_while_the_tree_sits_on_the_jobs_branch(self):
        after = _prs((133, "someone-elses-branch"))
        assert self._find({}, after, "tesla-trunk") is None

    def test_the_only_new_pr_is_the_jobs_when_the_tree_is_back_on_its_branch(self):
        after = _prs((133, "tesla-trunk"))
        assert self._find({}, after) == after[133]
        two = _prs((133, "tesla-trunk"), (134, "hue-dim"))
        assert self._find({}, two) is None, "two new PRs and no branch to tell them apart: none is guessed"

    def test_an_unlistable_state_means_no_pr(self):
        assert self._find(None, _prs((133, "tesla-trunk")), "tesla-trunk") is None
        assert self._find({}, None, "tesla-trunk") is None

    def test_during_a_trial_the_tried_pr_is_the_jobs(self):
        from assistant.agents.PullRequestTrial import Trial
        trial = Trial(number=31, title="t", head_ref="feat/core", since="14:02", return_branch="release")
        before = _prs((31, "feat/core"))
        assert self._find(before, dict(before), trial=trial) == before[31]
        assert self._find(before, {}, trial=trial) is None

    @patch.object(CodeQualityMaintainer, "_run_subprocess")
    def test_the_open_prs_are_read_from_gh(self, mock_sub):
        mock_sub.return_value = Mock(returncode=0, stdout=json.dumps(
            [{"number": 133, "url": "https://github.com/example/keep/pull/133", "headRefName": "tesla-trunk",
              "author": {"login": "alice"}}]))
        assert CodeQualityMaintainer._open_pull_requests() == {
            133: JobPullRequest(133, "https://github.com/example/keep/pull/133", "tesla-trunk", "alice")}
        mock_sub.return_value = Mock(returncode=1, stdout="", stderr="gh: HTTP 502")
        assert CodeQualityMaintainer._open_pull_requests() is None

    def test_the_pr_job_hands_the_pr_the_run_left_the_tree_on_back(self):
        after = _prs((133, "tesla-trunk"))
        with patch.object(CodeQualityMaintainer, "_run_subprocess", return_value=Mock(returncode=0, stdout="")), \
                patch.object(Updater, "current_branch", side_effect=["release", "release", "tesla-trunk"]), \
                patch.object(CodeQualityMaintainer, "_open_pull_requests", side_effect=[{}, after]), \
                patch.object(CodeQualityMaintainer, "_run_claude", return_value="Opened the PR."), \
                patch("assistant.agents.PullRequestTrial.PullRequestTrial.current", return_value=None):
            report, pr = CodeQualityMaintainer._pr_job("prompt", "improve-integration Tesla")
        assert (report, pr) == ("Opened the PR.", after[133])

    @patch.object(CodeQualityMaintainer, "_run_subprocess")
    def test_closing_a_pr_keeps_its_branch(self, mock_sub):
        """A contract with gh: without --delete-branch the branch survives, so a closed PR is
        reopened in one step instead of restored by hand (2026-10-26, twice)."""
        mock_sub.return_value = Mock(returncode=0, stdout="", stderr="")
        CodeQualityMaintainer._close_pr(132, "touches two folders")
        command = mock_sub.call_args.args[0]
        assert command[:4] == ["gh", "pr", "close", "132"] and "--delete-branch" not in command
        assert "touches two folders" in command[command.index("--comment") + 1]


def _gh_answers(status=0, user="marcus", view=None, view_rc=0):
    """subprocess.run answering gh auth status, gh api user and gh repo view."""
    view = view if view is not None else {"nameWithOwner": "marcus/keep", "viewerPermission": "WRITE"}

    def run(cmd, **kwargs):
        if cmd[1:3] == ["auth", "status"]:
            return Mock(returncode=status, stdout="", stderr="" if status == 0 else "You are not logged into any GitHub hosts")
        if cmd[1:3] == ["api", "user"]:
            return Mock(returncode=0, stdout=f"{user}\n", stderr="")
        if cmd[1:3] == ["repo", "view"]:
            return Mock(returncode=view_rc, stdout=json.dumps(view), stderr="" if view_rc == 0 else "HTTP 404")
        raise AssertionError(f"unexpected command {cmd}")
    return run


@pytest.mark.unit
class TestGithubAccess:
    @patch('assistant.agents.CodeQualityMaintainer.subprocess.run')
    def test_a_logged_in_gh_is_the_account_and_its_permission(self, mock_run):
        mock_run.side_effect = _gh_answers(user="alice", view={"nameWithOwner": "marcus/keep", "viewerPermission": "read"})
        access, reason = CodeQualityMaintainer._github_access()
        assert reason is None and access == GithubAccess("alice", "marcus/keep", "READ")
        assert CodeQualityMaintainer._github_unavailable() is None

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.run')
    def test_a_logged_out_gh_names_the_login(self, mock_run):
        mock_run.side_effect = _gh_answers(status=1)
        access, reason = CodeQualityMaintainer._github_access()
        assert access is None and reason == GH_NOT_LOGGED_IN_MESSAGE and "gh auth login" in reason
        assert CodeQualityMaintainer._github_unavailable() == reason

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.run', side_effect=FileNotFoundError("gh"))
    def test_a_missing_gh_names_the_install(self, mock_run):
        access, reason = CodeQualityMaintainer._github_access()
        assert access is None and reason == GH_NOT_INSTALLED_MESSAGE and "brew install gh" in reason

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.run')
    def test_an_unreadable_repository_is_a_reason(self, mock_run):
        mock_run.side_effect = _gh_answers(view_rc=1)
        access, reason = CodeQualityMaintainer._github_access()
        assert access is None and "HTTP 404" in reason

    @pytest.mark.parametrize("permission, push, label", [
        ("ADMIN", True, True), ("MAINTAIN", True, True), ("WRITE", True, True),
        ("TRIAGE", False, True), ("READ", False, False)])
    def test_what_each_permission_may_do(self, permission, push, label):
        access = GithubAccess("alice", "marcus/keep", permission)
        assert (access.can_push, access.can_label) == (push, label)


def _fork_answers(forks, fork_rc=0):
    """_run_subprocess answering gh repo fork, gh auth setup-git and gh repo list."""
    def run(cmd, label, timeout):
        if cmd[1:3] == ["repo", "fork"]:
            return Mock(returncode=fork_rc, stdout="", stderr="" if fork_rc == 0 else "HTTP 403: forking is disabled")
        if cmd[1:3] == ["auth", "setup-git"]:
            return Mock(returncode=0, stdout="", stderr="")
        if cmd[1:3] == ["repo", "list"]:
            return Mock(returncode=0, stdout=json.dumps(forks), stderr="")
        raise AssertionError(f"unexpected command {cmd}")
    return run


def _fork(name, parent):
    owner, repo = parent.split("/")
    return {"nameWithOwner": name, "parent": {"name": repo, "owner": {"login": owner}}}


@pytest.mark.unit
class TestEnsureFork:
    def test_the_fork_of_the_repository_is_found_among_the_accounts_forks(self):
        forks = [_fork("alice/tensorflow", "tensorflow/tensorflow"), _fork("alice/keep-1", "Marcus/Keep")]
        with patch.object(CodeQualityMaintainer, "_run_subprocess", side_effect=_fork_answers(forks)) as run:
            assert CodeQualityMaintainer._ensure_fork(READER) == "alice/keep-1"
        commands = [c.args[0] for c in run.call_args_list]
        assert commands[0][:4] == ["gh", "repo", "fork", "marcus/keep"] and "--clone=false" in commands[0]
        assert ["gh", "auth", "setup-git"] in commands, "the push over HTTPS goes through gh's login"

    def test_no_matching_fork_raises(self):
        forks = [_fork("alice/tensorflow", "tensorflow/tensorflow")]
        with patch.object(CodeQualityMaintainer, "_run_subprocess", side_effect=_fork_answers(forks)):
            with pytest.raises(RuntimeError):
                CodeQualityMaintainer._ensure_fork(READER)

    def test_a_failing_fork_raises_with_githubs_words(self):
        with patch.object(CodeQualityMaintainer, "_run_subprocess", side_effect=_fork_answers([], fork_rc=1)):
            with pytest.raises(RuntimeError) as raised:
                CodeQualityMaintainer._ensure_fork(READER)
        assert "forking is disabled" in str(raised.value)


@pytest.mark.unit
class TestPrJobDelivery:
    """What a PR job is told on top of its prompt about where the PR goes."""

    def test_an_account_without_push_opens_the_pr_from_its_fork(self):
        with patch.object(CodeQualityMaintainer, "_ensure_fork", return_value="alice/keep") as fork:
            prompt, _ = _claude_prompt(lambda: CodeQualityMaintainer._pr_job("Fix the trunk.", "self-improvement", READER))
        fork.assert_called_once_with(READER)
        assert prompt.startswith("Fix the trunk.")
        assert "https://github.com/alice/keep.git" in prompt
        assert "--repo marcus/keep" in prompt and "--head alice:" in prompt

    def test_no_fork_means_no_run(self):
        with patch.object(CodeQualityMaintainer, "_ensure_fork", side_effect=RuntimeError("forking is disabled")):
            prompt, (report, pr) = _claude_prompt(lambda: CodeQualityMaintainer._pr_job("Fix it.", "self-improvement", READER))
        assert prompt is None and pr is None
        assert "forking is disabled" in report

    def test_during_a_trial_the_trial_paragraph_wins_and_no_fork_is_made(self):
        with patch.object(CodeQualityMaintainer, "_ensure_fork") as fork:
            prompt, _ = _claude_prompt(lambda: CodeQualityMaintainer._pr_job("Fix it.", "self-improvement", READER),
                                       trial="\nPUSH TO PR #204\n")
        fork.assert_not_called()
        assert prompt.endswith("PUSH TO PR #204\n") and "alice" not in prompt

    @pytest.mark.parametrize("access", [PUSHER, None])
    def test_an_account_that_can_push_gets_the_prompt_as_is(self, access):
        with patch.object(CodeQualityMaintainer, "_ensure_fork") as fork:
            prompt, _ = _claude_prompt(lambda: CodeQualityMaintainer._pr_job("Fix it.", "self-improvement", access))
        fork.assert_not_called()
        assert prompt == "Fix it."


@pytest.mark.unit
class TestIssueDraft:
    def test_a_short_draft_is_prefilled_whole(self):
        from urllib.parse import urlparse, parse_qs
        draft = IssueDraft("marcus/keep", "Hallway lights stay on", "The scene ended at 23:10.\n\n- step one")
        link, prefilled = draft.link()
        url = urlparse(link)
        query = parse_qs(url.query)
        assert prefilled and url.netloc == "github.com" and url.path == "/marcus/keep/issues/new"
        assert query == {"title": [draft.title], "body": [draft.body], "labels": [ISSUE_LABEL]}

    def test_a_long_draft_is_linked_with_its_title_alone(self):
        from urllib.parse import urlparse, parse_qs
        draft = IssueDraft("marcus/keep", "Hallway lights stay on", "log excerpt " * 400)
        link, prefilled = draft.link()
        assert not prefilled and len(link) <= PREFILL_URL_LIMIT
        query = parse_qs(urlparse(link).query)
        assert query["title"] == [draft.title] and "body" not in query

    def test_a_comment_links_the_issue_itself(self):
        link, prefilled = IssueDraft("marcus/keep", "", "Seen again tonight.", comment_on=7).link()
        assert link == "https://github.com/marcus/keep/issues/7" and not prefilled


@pytest.mark.unit
class TestReadIssueDraft:
    def test_a_new_issue_draft_reads(self, tmp_path):
        path = tmp_path / "d.md"
        path.write_text("Title: Hallway lights stay on\n\nThe scene ended at 23:10.\nThen nothing.\n")
        draft = CodeQualityMaintainer.read_issue_draft(path, "marcus/keep")
        assert draft == IssueDraft("marcus/keep", "Hallway lights stay on", "The scene ended at 23:10.\nThen nothing.")

    def test_a_comment_draft_reads(self, tmp_path):
        path = tmp_path / "d.md"
        path.write_text("Comment on: #12\n\nSeen again tonight at 23:10.")
        draft = CodeQualityMaintainer.read_issue_draft(path, "marcus/keep")
        assert draft.comment_on == 12 and draft.body == "Seen again tonight at 23:10."

    @pytest.mark.parametrize("text", [None, "Title: Lights\n\n   \n", "Lights stay on\n\nbody", ""])
    def test_no_or_a_malformed_draft_is_none(self, tmp_path, text):
        path = tmp_path / "d.md"
        if text is not None:
            path.write_text(text)
        assert CodeQualityMaintainer.read_issue_draft(path, "marcus/keep") is None


@pytest.mark.unit
class TestReportIssue:
    @pytest.fixture(autouse=True)
    def version(self):
        with patch.object(CodeQualityMaintainer, "_version", return_value="v2026.10.1"):
            yield

    def _report(self, access, reason=None, write_draft=None, origin="marcus/keep", tmp_path=None):
        prompts = []

        def claude(prompt, label):
            prompts.append(prompt)
            if write_draft is not None:
                import re
                path = re.search(r"Write the draft to (\S+):", prompt).group(1)
                with open(path, "w") as f:
                    f.write(write_draft)
            return "Drafted: the hallway lights."
        with patch.object(CodeQualityMaintainer, "_github_access", return_value=(access, reason)), \
                patch.object(CodeQualityMaintainer, "_origin_repo", return_value=origin), \
                patch("assistant.agents.CodeQualityMaintainer.ISSUE_DRAFT_DIR", tmp_path), \
                patch.object(CodeQualityMaintainer, "_run_claude", side_effect=claude):
            result = CodeQualityMaintainer.report_issue("the hallway lights stayed on at 23:10")
        return (prompts[0] if prompts else None), result

    def test_without_gh_the_issue_is_drafted_and_comes_back(self, tmp_path):
        prompt, (report, draft) = self._report(None, GH_NOT_INSTALLED_MESSAGE,
                                               write_draft="Title: Hallway lights stay on\n\nThe scene ended.",
                                               tmp_path=tmp_path)
        assert "gh issue create" not in prompt and str(tmp_path) in prompt
        assert "api.github.com/search/issues?q=repo:marcus/keep" in prompt
        assert report == "Drafted: the hallway lights."
        assert draft == IssueDraft("marcus/keep", "Hallway lights stay on", "The scene ended.",
                                   why="the GitHub CLI is not installed on this machine")

    def test_the_draft_says_why_gh_could_not_file_it(self, tmp_path):
        """Three refusals, three wordings: the user hears the one that applies, not "no login"
        when gh is logged in but the repository cannot be read."""
        for reason, words in ((GH_NOT_LOGGED_IN_MESSAGE, "no GitHub login"),
                              ("The GitHub CLI is logged in, but the repository cannot be read through it: boom",
                               "cannot read the repository")):
            _, (_, draft) = self._report(None, reason, write_draft="Title: Lights\n\nBody.", tmp_path=tmp_path)
            assert words in draft.why, reason

    def test_a_run_that_drafted_nothing_returns_no_draft(self, tmp_path):
        _, (report, draft) = self._report(None, GH_NOT_INSTALLED_MESSAGE, tmp_path=tmp_path)
        assert draft is None and report

    def test_without_gh_and_without_a_github_origin_nothing_runs(self, tmp_path):
        prompt, result = self._report(None, GH_NOT_LOGGED_IN_MESSAGE, origin=None, tmp_path=tmp_path)
        assert prompt is None and result == (GH_NOT_LOGGED_IN_MESSAGE, None)

    def test_with_gh_the_issue_is_filed_and_labelled_only_where_the_account_may(self, tmp_path):
        prompt, (_, draft) = self._report(GithubAccess("alice", "marcus/keep", "READ"), tmp_path=tmp_path)
        assert "gh issue create" in prompt and ISSUE_LABEL not in prompt and draft is None
        prompt, _ = self._report(GithubAccess("alice", "marcus/keep", "TRIAGE"), tmp_path=tmp_path)
        assert ISSUE_LABEL in prompt


@pytest.mark.unit
class TestOriginRepo:
    """The repository a drafted issue is filed on is read from the checkout's origin, so an
    install without gh still links the right repository whatever form its remote URL takes."""

    @staticmethod
    def _origin(url, returncode=0):
        with patch.object(CodeQualityMaintainer, "_run_subprocess",
                          return_value=Mock(returncode=returncode, stdout=f"{url}\n", stderr="")):
            return CodeQualityMaintainer._origin_repo()

    @pytest.mark.parametrize("url", [
        "git@github.com:marcus/keep.git",
        "ssh://git@github.com/marcus/keep.git",
        "https://github.com/marcus/keep.git",
        "https://github.com/marcus/keep",
        "https://github.com/marcus/keep/",
    ])
    def test_every_form_of_a_github_remote_gives_owner_and_name(self, url):
        assert self._origin(url) == "marcus/keep"

    def test_a_remote_elsewhere_or_none_at_all_is_no_repository(self):
        assert self._origin("https://gitlab.example/marcus/keep.git") is None
        assert self._origin("", returncode=128) is None


@pytest.mark.unit
class TestIssueIn:
    def test_a_new_issue_and_a_comment_both_name_the_issue(self):
        assert CodeQualityMaintainer.issue_in("Filed https://github.com/example/keep/issues/7\nsummary") == \
            (7, "https://github.com/example/keep/issues/7")
        number, url = CodeQualityMaintainer.issue_in("Commented on https://github.com/example/keep/issues/7#issuecomment-5")
        assert number == 7 and url.endswith("#issuecomment-5")

    def test_a_pull_request_link_is_not_an_issue(self):
        assert CodeQualityMaintainer.issue_in("https://github.com/example/keep/pull/7") == (None, None)
        assert CodeQualityMaintainer.issue_in("") == (None, None)


class TestPullRequestTrialHooks:
    """What the maintainer does differently while the server runs a PR as a trial."""

    def test_the_nightly_run_waits_while_a_pr_is_tried(self, maintainer):
        with patch.object(CodeQualityMaintainer, "trial_skip_reason", return_value="the server is trying PR #204"), \
                patch.object(maintainer, "_nightly_run") as run:
            maintainer._execute_nightly_run(force=True)
        run.assert_not_called()

    def test_the_nightly_run_goes_ahead_on_a_normal_branch(self, maintainer):
        with patch.object(CodeQualityMaintainer, "trial_skip_reason", return_value=None), \
                patch.object(maintainer, "_nightly_run") as run:
            maintainer._execute_nightly_run()
        run.assert_called_once()

    def test_the_skip_reason_names_the_tried_pr_and_is_none_otherwise(self):
        from assistant.agents.PullRequestTrial import PullRequestTrial, Trial
        trial = Trial(number=204, title="t", head_ref="feat/trunk", since="14:02", return_branch="release")
        with patch.object(PullRequestTrial, "current", return_value=trial):
            assert "PR #204" in CodeQualityMaintainer.trial_skip_reason()
        with patch.object(PullRequestTrial, "current", return_value=None):
            assert CodeQualityMaintainer.trial_skip_reason() is None

    @pytest.mark.parametrize("job", ["improve_code", "improve_integration"])
    def test_a_pr_job_during_a_trial_is_told_to_push_to_the_pr(self, job):
        with patch.object(CodeQualityMaintainer, "_github_access", return_value=(PUSHER, None)), \
                patch.object(CodeQualityMaintainer, "_version", return_value="v2026.10.1"):
            if job == "improve_code":
                prompt, _ = _claude_prompt(lambda: CodeQualityMaintainer.improve_code("fix the trunk"),
                                           trial="\nPUSH TO PR #204\n")
            else:
                prompt, _ = _claude_prompt(lambda: CodeQualityMaintainer.improve_integration("Tesla", "tesla", "fix the trunk"),
                                           trial="\nPUSH TO PR #204\n")
        assert prompt.endswith("PUSH TO PR #204\n") and "fix the trunk" in prompt

    @patch('assistant.agents.CodeQualityMaintainer.subprocess.Popen')
    def test_a_trial_restart_hands_start_sh_the_branch_to_fall_back_to(self, mock_popen):
        CodeQualityMaintainer.restart("trying PR #204", fallback_branch="release")
        assert mock_popen.call_args[0][0] == ['bash', 'start.sh', str(os.getpid()), 'release']
        CodeQualityMaintainer.restart("audio recovery")
        assert mock_popen.call_args[0][0] == ['bash', 'start.sh', str(os.getpid())]


@pytest.mark.unit
class TestFixBugsStep:
    """The step after the bug analysis: one PR fixing the Critical and Major items of its fix
    list, and no Claude Code run on a night that has none."""

    ANALYSIS = "| Severity | Where | What |\n| Major | foo.py:3 | off by one |\n| Minor | bar.py:9 | typo |"

    def test_without_an_analysis_nothing_runs(self):
        with patch.object(CodeQualityMaintainer, '_pr_job') as job, \
                patch.object(CodeQualityMaintainer, '_github_unavailable') as gh:
            assert CodeQualityMaintainer._step_fix_bugs("") == ""
        job.assert_not_called()
        gh.assert_not_called()

    @pytest.mark.parametrize("analysis", ["| Minor | bar.py:9 | typo |\n| Nit | baz.py:1 | naming |",
                                          "The majority of the changes are fine. No findings."])
    def test_an_analysis_without_a_critical_or_major_finding_costs_no_run_but_is_reported(self, analysis):
        with patch.object(CodeQualityMaintainer, '_pr_job') as job:
            report = CodeQualityMaintainer._step_fix_bugs(analysis)
        job.assert_not_called()
        assert report   # the email says why nothing was fixed

    @pytest.mark.parametrize("severity", ["Major", "CRITICAL"])
    def test_a_critical_or_major_finding_starts_the_pr_job_with_the_analysis(self, severity):
        analysis = f"| {severity} | foo.py:3 | off by one |"
        with patch.object(CodeQualityMaintainer, '_pr_job', return_value=("PR #7 opened", JOB_PR)) as job, \
                patch.object(CodeQualityMaintainer, '_github_unavailable', return_value=None), \
                patch.object(CodeQualityMaintainer, '_trial_instructions', return_value=""), \
                patch.object(CodeQualityMaintainer, '_version', return_value="v2026.10.1"):
            assert CodeQualityMaintainer._step_fix_bugs(analysis) == "PR #7 opened"
        prompt = job.call_args[0][0]
        assert analysis in prompt
        assert NIGHTLY_FIX_LABEL in prompt and "v2026.10.1" in prompt

    def test_the_trial_instructions_follow_the_prompt(self):
        with patch.object(CodeQualityMaintainer, '_github_unavailable', return_value=None):
            prompt, _ = _claude_prompt(lambda: CodeQualityMaintainer._step_fix_bugs(self.ANALYSIS),
                                       trial="\nPush to the tried PR.")
        assert prompt.endswith("Push to the tried PR.") and "off by one" in prompt

    def test_the_step_hands_the_report_on_without_the_pr(self):
        """The step feeds the nightly report; _pr_job's tuple there rendered as its repr."""
        with patch.object(CodeQualityMaintainer, '_pr_job', return_value=("Fixed the trunk.", JOB_PR)), \
                patch.object(CodeQualityMaintainer, '_github_unavailable', return_value=None), \
                patch.object(CodeQualityMaintainer, '_version', return_value="v2026.10.1"):
            report = CodeQualityMaintainer._step_fix_bugs(self.ANALYSIS)
        assert isinstance(report, str) and report == "Fixed the trunk."

    def test_without_gh_the_step_says_so_instead_of_running(self):
        with patch.object(CodeQualityMaintainer, '_pr_job') as job, \
                patch.object(CodeQualityMaintainer, '_github_unavailable', return_value=GH_NOT_LOGGED_IN_MESSAGE):
            assert CodeQualityMaintainer._step_fix_bugs(self.ANALYSIS) == GH_NOT_LOGGED_IN_MESSAGE
        job.assert_not_called()

    def test_the_severity_screen_matches_whole_words_in_any_case(self):
        assert CodeQualityMaintainer.fixable_severities("MAJOR: x\nminor: y") == ["major"]
        assert CodeQualityMaintainer.fixable_severities("a critical path; a majority vote") == ["critical"]
        assert CodeQualityMaintainer.fixable_severities("critically slow") == []


@pytest.mark.unit
class TestNightlyUpdateFailures:
    """What a git process that hangs past GIT_TIMEOUT at 03:00 costs: the update handles a
    failing exit code (the night runs on the old code), a raise must not cost more."""

    @patch('assistant.agents.CodeQualityMaintainer.time.sleep')
    @patch('assistant.agents.CodeQualityMaintainer.datetime')
    def test_an_update_that_raises_still_leaves_the_night_to_run_on_the_old_code(self, mock_dt, mock_sleep, maintainer):
        mock_dt.now.return_value = datetime(2026, 2, 19, 3, 0, 0)
        mock_sleep.side_effect = lambda *_: setattr(maintainer, '_running', False)
        hung = subprocess.TimeoutExpired(["git", "pull"], GIT_TIMEOUT)
        with patch.object(maintainer, '_execute_nightly_run') as run, \
                patch.object(maintainer, '_nightly_update', side_effect=hung):
            maintainer._running = True
            maintainer._run()
        run.assert_called_once()

    @patch.object(CodeQualityMaintainer, '_run_claude', return_value="| Major | baz.py:7 | test_x pins it | guard None |")
    @patch('assistant.agents.CodeQualityMaintainer.subprocess.run')
    def test_a_merge_that_raises_after_the_test_generation_keeps_the_fix_list(self, mock_run, mock_claude, maintainer):
        mock_run.return_value = Mock(stdout="master\n", returncode=0)
        hung = subprocess.TimeoutExpired(["git", "checkout", "master"], GIT_TIMEOUT)
        with patch.object(CodeQualityMaintainer, '_merge_test_prs', side_effect=hung):
            report = maintainer._step_generate_tests()
        assert "baz.py:7" in report


@pytest.mark.unit
class TestOpenPins:
    """The last step of the night lists every strict xfail of the unit suite — the pinned bugs
    waiting for the maintainer's verdict — rated by Claude Code; a `wontfix` is a verdict given."""

    @pytest.fixture
    def tests_root(self, tmp_path):
        unit = tmp_path / "tests" / "unit" / "assistant"
        unit.mkdir(parents=True)
        with patch('assistant.agents.CodeQualityMaintainer._AISERVER_PATH', tmp_path):
            yield unit

    def test_counts_the_strict_xfails_and_leaves_the_wontfix_and_the_plain_xfail_out(self, tests_root):
        (tests_root / "test_a.py").write_text(
            '@pytest.mark.xfail(strict=True, reason="a word ending at the limit is dropped")\n'
            'def test_one(): ...\n'
            '@pytest.mark.xfail(\n    strict=True,\n    reason="two groups with one title")\n'
            'def test_two(): ...\n'
            '@pytest.mark.wontfix(reason="harmless, the fix would cost more than the bug")\n'
            'def test_three(): ...\n'
            '@pytest.mark.xfail(reason="flaky upstream")\n'
            'def test_four(): ...\n')
        (tests_root / "helper.py").write_text('@pytest.mark.xfail(strict=True, reason="not a test file")\n')
        assert CodeQualityMaintainer.pinned_tests() == 2

    def test_a_suite_without_a_pin_costs_no_claude_run_and_still_reports(self, tests_root):
        (tests_root / "test_a.py").write_text('def test_one(): ...\n')
        with patch.object(CodeQualityMaintainer, '_run_claude') as run:
            report = CodeQualityMaintainer._step_open_pins()
        run.assert_not_called()
        assert report   # the email still says there is nothing waiting

    def test_a_pinned_suite_is_rated_by_claude_and_the_table_is_the_report(self, tests_root):
        (tests_root / "test_a.py").write_text(
            '@pytest.mark.xfail(strict=True, reason="a word ending at the limit is dropped")\ndef test_one(): ...\n')
        table = "| 1 | tests/unit/assistant/test_a.py::test_one | a word is dropped | trivial |"
        with patch.object(CodeQualityMaintainer, '_run_claude', return_value=table) as run:
            assert CodeQualityMaintainer._step_open_pins() == table
        run.assert_called_once()
        assert "wontfix" in run.call_args[0][0]   # the prompt tells the run which pins are verdicts already

    def test_a_crashed_rating_leaves_the_night_its_other_sections(self, tests_root):
        (tests_root / "test_a.py").write_text(
            '@pytest.mark.xfail(strict=True, reason="a bug")\ndef test_one(): ...\n')
        with patch.object(CodeQualityMaintainer, '_run_claude', side_effect=RuntimeError("claude down")):
            assert CodeQualityMaintainer._step_open_pins() == ""

