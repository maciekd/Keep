"""A branch of the install's own — somebody develops on this server's checkout — is left where
it is: the nightly update skips the night, a restart relaunches in place, and only
`leave_own_branch` goes back to the branch the install follows."""
from contextlib import contextmanager
from unittest.mock import Mock, patch

import pytest

from assistant.agents import PullRequestTrial as trial_module
from assistant.agents.CodeQualityMaintainer import CodeQualityMaintainer
from assistant.agents.PullRequestTrial import PullRequestTrial
from common.Updater import Updater

BOOT, NEWER = "b" * 40, "c" * 40


@contextmanager
def _server(booted, checked_out=None):
    """A process booted on *booted* whose checkout is on *checked_out* now (the same by default)."""
    with patch("common.Version.branch", return_value=booted), \
            patch.object(CodeQualityMaintainer, "_current_branch", return_value=checked_out or booted):
        yield


class TestWhichBranchIsTheInstallsOwn:
    @pytest.mark.parametrize("booted", ["release", "master", "trial/pr-204", None])
    def test_a_line_an_install_follows_a_trial_and_a_detached_head_are_not(self, booted):
        with _server(booted):
            assert CodeQualityMaintainer.own_branch() is None

    def test_a_branch_somebody_made_is(self):
        with _server("feat/garage-door"):
            assert CodeQualityMaintainer.own_branch() == "feat/garage-door"

    def test_a_checkout_that_moved_since_the_boot_is_not(self):
        """A Claude Code job has the tree on its work branch for the length of a run, and a
        process booted on a branch somebody has since left is not on it either."""
        with _server("feat/garage-door", checked_out="release"):
            assert CodeQualityMaintainer.own_branch() is None


@pytest.fixture
def job_record():
    """The record a job keeps of the branch it started on, gone before and after the test."""
    from assistant.agents import CodeQualityMaintainer as module
    module.JOB_HOME_FILE.unlink(missing_ok=True)
    yield module.JOB_HOME_FILE
    module.JOB_HOME_FILE.unlink(missing_ok=True)


class TestAJobCutShortLeavesNoOwnBranch:
    """A relaunch that pulls nothing (the audio recovery, a crash, the watchdog) while a Claude
    Code job has the tree on its work branch boots on that branch: it is the job's, and the
    updates must go on bringing the install back."""

    def _job(self, started_on, left_on, during):
        """A PR job from *started_on* whose run leaves the tree on *left_on*; *during* is called mid-run."""
        def run(*_):
            current.return_value = left_on
            during()
            return "report"
        with patch.object(CodeQualityMaintainer, "_current_branch", return_value=started_on) as current, \
                patch.object(CodeQualityMaintainer, "_trial_instructions", return_value=""), \
                patch.object(CodeQualityMaintainer, "_open_pull_requests", return_value={}), \
                patch.object(CodeQualityMaintainer, "_pull_current"), \
                patch.object(CodeQualityMaintainer, "_job_pull_request", return_value=None), \
                patch.object(CodeQualityMaintainer, "_run_subprocess") as git, \
                patch.object(CodeQualityMaintainer, "_run_claude", side_effect=run):
            CodeQualityMaintainer._pr_job("prompt", "self-improvement")
        return git

    def test_a_process_booted_on_a_running_jobs_work_branch_is_not_on_a_branch_of_its_own(self, job_record):
        seen = []

        def relaunched_mid_job():
            with patch("common.Version.branch", return_value="fix/nightly"):
                seen.append(CodeQualityMaintainer.own_branch())

        git = self._job("release", "fix/nightly", relaunched_mid_job)
        assert seen == [None]
        assert git.call_args.args[0] == ["git", "checkout", "release"]

    def test_a_job_that_came_back_leaves_no_record(self, job_record):
        self._job("release", "fix/nightly", lambda: None)
        with _server("feat/garage-door"):
            assert CodeQualityMaintainer.own_branch() == "feat/garage-door"

    def test_a_job_started_on_the_own_branch_does_not_disown_it(self, job_record):
        job_record.write_text("feat/garage-door\n")
        with _server("feat/garage-door"):
            assert CodeQualityMaintainer.own_branch() == "feat/garage-door"

    @pytest.mark.parametrize("booted, kept", [("release", False), ("master", False), ("fix/nightly", True)])
    def test_the_record_of_a_cut_short_job_lasts_until_the_server_is_back_on_its_line(self, job_record, booted, kept):
        """Dropped then, so a branch somebody checks out later is not taken for a job's leftover."""
        job_record.write_text("release\n")
        with patch("common.Version.branch", return_value=booted):
            CodeQualityMaintainer._forget_finished_interruption()
        assert job_record.exists() is kept


class TestTheNightlyRunStaysOffAnOwnBranch:
    @pytest.mark.parametrize("force", [False, True])
    def test_no_run_and_so_no_pr_is_branched_off_somebodys_work(self, force):
        maintainer = CodeQualityMaintainer()
        with _server("feat/garage-door"), \
                patch.object(CodeQualityMaintainer, "unavailable_reason", return_value=None), \
                patch.object(CodeQualityMaintainer, "nightly_skip_reason", return_value=None), \
                patch.object(CodeQualityMaintainer, "trial_skip_reason", return_value=None), \
                patch.object(maintainer, "_nightly_run") as run:
            maintainer._execute_nightly_run(force=force)
        run.assert_not_called()

    def test_on_the_line_the_install_follows_it_runs(self):
        maintainer = CodeQualityMaintainer()
        with _server("release"), \
                patch.object(CodeQualityMaintainer, "unavailable_reason", return_value=None), \
                patch.object(CodeQualityMaintainer, "nightly_skip_reason", return_value=None), \
                patch.object(CodeQualityMaintainer, "trial_skip_reason", return_value=None), \
                patch.object(maintainer, "_nightly_run") as run:
            maintainer._execute_nightly_run()
        run.assert_called_once()


@pytest.fixture
def machinery():
    """Everything an update would touch, faked: the pull, pip, the relaunch."""
    with patch.object(Updater, "pull", return_value=("", "Already up to date.")) as pull, \
            patch.object(Updater, "head_commit", return_value=NEWER), \
            patch.object(Updater, "commits_between", return_value=2), \
            patch.object(Updater, "sync", return_value=("", "")) as sync, \
            patch.object(CodeQualityMaintainer, "restart") as restart, \
            patch.object(CodeQualityMaintainer, "target_branch", return_value="release"), \
            patch("assistant.SystemNote.system_note"):
        yield Mock(pull=pull, sync=sync, restart=restart)


class TestNothingMovesItByItself:
    def test_the_nightly_update_skips_the_night(self, machinery):
        maintainer = CodeQualityMaintainer.__new__(CodeQualityMaintainer)
        maintainer._boot_commit = BOOT
        with _server("feat/garage-door"), \
                patch("assistant.agents.CodeQualityMaintainer.DevelopmentConfig.LOCAL_DEVELOPMENT_MODE", False), \
                patch("assistant.agents.CodeQualityMaintainer.ConfigManager") as config:
            config.return_value.get.side_effect = lambda key, default=None: default
            assert maintainer._nightly_update() is False
        machinery.pull.assert_not_called()
        machinery.restart.assert_not_called()

    def test_a_restart_relaunches_in_place_and_pulls_nothing(self, machinery):
        with _server("feat/garage-door"):
            result = CodeQualityMaintainer.pull_and_restart(resumes_conversation=True)
        machinery.pull.assert_not_called()
        machinery.sync.assert_called_once()   # the branch may have bumped a pin
        machinery.restart.assert_called_once()
        assert machinery.restart.call_args.kwargs == {"resumes_conversation": True}
        assert "feat/garage-door" in result and "release" in result

    def test_packages_that_do_not_install_keep_the_old_process(self, machinery):
        machinery.sync.return_value = ("pip exited 1 (no matching distribution)", "")
        with _server("feat/garage-door"):
            result = CodeQualityMaintainer.pull_and_restart()
        machinery.restart.assert_not_called()
        assert "no matching distribution" in result

    def test_leaving_it_pulls_the_branch_the_install_follows(self, machinery):
        with _server("feat/garage-door"):
            CodeQualityMaintainer.pull_and_restart(leave_own_branch=True)
        machinery.pull.assert_called_once_with("release")
        machinery.restart.assert_called_once()

    def test_on_a_normal_branch_a_restart_pulls_as_ever(self, machinery):
        with _server("release"):
            CodeQualityMaintainer.pull_and_restart()
        machinery.pull.assert_called_once_with("release")


class TestTheAssistantAndTheUserAreTold:
    @pytest.fixture(autouse=True)
    def on_it(self):
        with _server("feat/garage-<door>"), patch.object(trial_module.Version, "branch", return_value="feat/garage-<door>"), \
                patch.object(CodeQualityMaintainer, "target_branch", return_value="release"):
            yield

    def test_the_prompt_names_the_branch_and_the_way_back(self):
        line = PullRequestTrial.prompt_line()
        assert "feat/garage-<door>" in line and "release" in line and "leave_own_branch" in line

    def test_the_startup_announcement_says_no_updates_arrive(self):
        note = PullRequestTrial.announcement_note()
        assert "feat/garage-&lt;door&gt;" in note and "release" in note and "update" in note

    def test_a_pr_job_gets_no_trial_instructions(self):
        assert PullRequestTrial.pr_job_instructions() == ""
