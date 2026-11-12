"""PullRequestTrial: the branch is the state, the vetting is the auto-merge's trust, a
trial's restart hands start.sh the branch to fall back to, and the nightly update keeps a
trial — with the night's release merged in — until the PR is released."""
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch, call

import pytest

from assistant.agents import PullRequestTrial as module
from assistant.agents.PullRequestTrial import PullRequestTrial, TrialRefused, parse_number, trial_number, GIT_TIMEOUT
from assistant.agents.PullRequestTrial import Trial
from assistant.agents.CodeQualityMaintainer import CodeQualityMaintainer
from common.Updater import Updater

from ...git_sandbox import History, empty_repo, git as sandbox_git, spawn


TODAY = module.datetime.now().strftime("%Y-%m-%d")


def _pr(**overrides) -> dict:
    pr = {"number": 204, "title": "Tesla: close the trunk once", "state": "OPEN",
          "author": {"login": "Marcus"}, "isCrossRepository": False, "headRefName": "feat/trunk"}
    pr.update(overrides)
    return pr


@pytest.fixture(autouse=True)
def fresh_cache():
    PullRequestTrial._pr.cache_clear()
    yield
    PullRequestTrial._pr.cache_clear()


@pytest.fixture
def github():
    """`gh pr view` answers with the PR the test sets, through subprocess.run."""
    with patch.object(module.subprocess, "run") as run:
        def answer(pr):
            run.return_value = Mock(returncode=0, stdout=json.dumps(pr), stderr="")
        answer(_pr())
        run.answer = answer
        yield run


@pytest.fixture
def trusted():
    with patch.object(CodeQualityMaintainer, "_trusted_pr_authors", return_value={"marcus"}):
        yield


@pytest.fixture
def git():
    """CodeQualityMaintainer._run_subprocess, which every git call goes through; clean tree."""
    with patch.object(CodeQualityMaintainer, "_run_subprocess") as run:
        run.return_value = Mock(returncode=0, stdout="", stderr="")
        yield run


class TestTheBranchIsTheState:
    @pytest.mark.parametrize("branch, number", [("trial/pr-204", 204), ("trial/pr-7", 7), ("master", None),
                                                ("release", None), ("trial/pr-", None), ("trial/pr-x", None),
                                                (None, None)])
    def test_a_trial_branch_names_its_pr(self, branch, number):
        assert trial_number(branch) == number

    def test_no_trial_on_a_normal_branch_and_nothing_asked_of_github(self, github):
        with patch.object(module.Version, "branch", return_value="release"):
            assert PullRequestTrial.current() is None
            assert PullRequestTrial.prompt_line() == ""
            assert PullRequestTrial.announcement_note() == ""
            assert PullRequestTrial.pr_job_instructions() == ""
        github.assert_not_called()

    def test_the_trial_is_read_from_the_boot_branch_and_github(self, github, git):
        git.return_value = Mock(returncode=0, stdout=f"{TODAY} 14:02:11 +0200\n", stderr="")
        with patch.object(module.Version, "branch", return_value="trial/pr-204"), \
                patch.object(CodeQualityMaintainer, "target_branch", return_value="release"):
            trial = PullRequestTrial.current()
        assert (trial.number, trial.title, trial.head_ref, trial.since, trial.return_branch) == \
               (204, "Tesla: close the trunk once", "feat/trunk", "14:02", "release")
        assert trial.branch == "trial/pr-204"

    def test_a_trial_begun_on_another_day_shows_its_date(self, github, git):
        git.return_value = Mock(returncode=0, stdout="2020-01-01 23:50:00 +0200\n", stderr="")
        with patch.object(module.Version, "branch", return_value="trial/pr-204"), \
                patch.object(CodeQualityMaintainer, "target_branch", return_value="release"):
            assert PullRequestTrial.current().since == "2020-01-01 23:50"

    def test_a_trial_began_when_the_pr_was_checked_out_not_at_last_nights_merge(self, github, git):
        """The branch's reflog as git prints it, newest first: the nightly merges are entries too."""
        git.return_value = Mock(returncode=0, stderr="", stdout=(
            f"{TODAY} 03:00:07 +0100 merge 5d2c1a9: Merge made by the 'ort' strategy.\n"
            "2020-01-01 23:50:00 +0100 branch: Created from FETCH_HEAD\n"
            "2019-12-24 09:00:00 +0100 branch: Created from FETCH_HEAD\n"))
        with patch.object(module.Version, "branch", return_value="trial/pr-204"), \
                patch.object(CodeQualityMaintainer, "target_branch", return_value="release"):
            assert PullRequestTrial.current().since == "2020-01-01 23:50"

    def test_github_down_still_knows_the_number(self, github, git):
        github.return_value = Mock(returncode=1, stdout="", stderr="gh: could not resolve host")
        with patch.object(module.Version, "branch", return_value="trial/pr-204"), \
                patch.object(CodeQualityMaintainer, "target_branch", return_value="release"):
            trial = PullRequestTrial.current()
            assert trial.number == 204 and trial.title == "" and trial.head_ref == ""
            assert "PR #204" in PullRequestTrial.prompt_line()
            assert PullRequestTrial.pr_job_instructions() == "", "without the PR's branch a job cannot be told where to push"

    def test_the_prompt_line_carries_the_pr_the_time_and_the_way_back(self, github, git):
        git.return_value = Mock(returncode=0, stdout=f"{TODAY} 14:02:11 +0200\n", stderr="")
        with patch.object(module.Version, "branch", return_value="trial/pr-204"), \
                patch.object(CodeQualityMaintainer, "target_branch", return_value="master"):
            line = PullRequestTrial.prompt_line()
        assert "PR #204" in line and "Tesla: close the trunk once" in line and "14:02" in line
        assert "master" in line and "restart_server" in line and "03:00" in line

    def test_the_startup_announcement_tells_the_user_which_pr_is_on_trial(self, github, git):
        github.answer(_pr(title="Tesla: <trunk> & frunk"))
        with patch.object(module.Version, "branch", return_value="trial/pr-204"), \
                patch.object(CodeQualityMaintainer, "target_branch", return_value="master"):
            note = PullRequestTrial.announcement_note()
        assert "PR #204" in note and "Tesla: &lt;trunk&gt; &amp; frunk" in note and "master" in note

    def test_a_pr_job_is_told_to_push_to_the_tried_pr_and_open_none(self, github, git):
        with patch.object(module.Version, "branch", return_value="trial/pr-204"), \
                patch.object(CodeQualityMaintainer, "target_branch", return_value="release"):
            assert PullRequestTrial.current().head_repo == "", "a branch of origin has no fork to name"
            text = PullRequestTrial.pr_job_instructions()
        assert "refs/heads/feat/trunk" in text and "PR #204" in text and "trial/pr-204" in text
        assert "do NOT create a branch" in text
        assert "git push origin HEAD:refs/heads/feat/trunk" in text

    def test_a_pr_job_on_a_tried_fork_pr_pushes_to_the_fork(self, github, git):
        github.answer(_pr(isCrossRepository=True, headRepositoryOwner={"login": "alice"},
                          headRepository={"nameWithOwner": "alice/keep"}))
        with patch.object(module.Version, "branch", return_value="trial/pr-204"), \
                patch.object(CodeQualityMaintainer, "target_branch", return_value="release"):
            assert PullRequestTrial.current().head_repo == "alice/keep"
            text = PullRequestTrial.pr_job_instructions()
        assert "git push https://github.com/alice/keep.git HEAD:refs/heads/feat/trunk" in text
        assert "git push origin" not in text


class TestVetting:
    def test_an_open_pr_of_this_repository_by_an_own_author_passes(self, github, trusted, git):
        assert PullRequestTrial.vet(204)["title"] == "Tesla: close the trunk once"

    def test_a_merged_pr_the_install_does_not_have_yet_passes(self, github, trusted, git):
        """The days between the merge and the release: the PR is closed to GitHub, and its
        code is still nowhere the install's branch reaches."""
        github.answer(_pr(state="MERGED", mergeCommit={"oid": "d" * 40}))
        git.side_effect = lambda cmd, label, timeout: Mock(
            returncode=1 if cmd[1] == "merge-base" else 0, stderr="",
            stdout="abcdef12\n" if cmd[1] == "rev-parse" else "")
        assert PullRequestTrial.vet(204)["number"] == 204

    def test_a_merged_pr_the_install_already_runs_is_refused(self, github, trusted, git):
        github.answer(_pr(state="MERGED", mergeCommit={"oid": "d" * 40}))
        git.side_effect = lambda cmd, label, timeout: Mock(
            returncode=0, stderr="", stdout="abcdef12\n" if cmd[1] == "rev-parse" else "")
        with patch.object(CodeQualityMaintainer, "target_branch", return_value="release"), \
                pytest.raises(TrialRefused) as refused:
            PullRequestTrial.vet(204)
        assert "release" in str(refused.value)

    @pytest.mark.parametrize("pr, words", [
        (_pr(state="CLOSED"), "closed"),
        (_pr(isCrossRepository=True), "fork"),
        (_pr(author={"login": "stranger"}), "stranger"),
    ])
    def test_a_pr_that_is_not_the_installs_own_open_one_is_refused(self, github, trusted, git, pr, words):
        github.answer(pr)
        with pytest.raises(TrialRefused) as refused:
            PullRequestTrial.vet(204)
        assert words in str(refused.value)

    def test_a_pr_from_the_authors_own_fork_passes(self, github, trusted, git):
        """An account without push access opens its PRs from its own fork."""
        github.answer(_pr(isCrossRepository=True, headRepositoryOwner={"login": "marcus"},
                          headRepository={"nameWithOwner": "marcus/keep-fork"}))
        assert PullRequestTrial.vet(204)["number"] == 204

    def test_a_pr_from_someone_elses_fork_is_refused(self, github, trusted, git):
        github.answer(_pr(isCrossRepository=True, headRepositoryOwner={"login": "dave"},
                          headRepository={"nameWithOwner": "dave/keep"}))
        with pytest.raises(TrialRefused) as refused:
            PullRequestTrial.vet(204)
        assert "dave" in str(refused.value)

    def test_a_dirty_checkout_is_refused(self, github, trusted, git):
        git.return_value = Mock(returncode=0, stdout=" M src/x.py\n", stderr="")
        with pytest.raises(TrialRefused) as refused:
            PullRequestTrial.vet(204)
        assert "uncommitted" in str(refused.value)

    def test_an_unreadable_pr_is_refused_with_githubs_words(self, github, trusted, git):
        github.return_value = Mock(returncode=1, stdout="", stderr="no pull requests found for 999")
        with pytest.raises(TrialRefused) as refused:
            PullRequestTrial.vet(999)
        assert "no pull requests found" in str(refused.value)

    def test_the_vetting_reads_github_fresh_not_the_cache(self, github, trusted, git):
        PullRequestTrial._pr(204)
        github.answer(_pr(state="CLOSED"))
        with pytest.raises(TrialRefused):
            PullRequestTrial.vet(204)


class TestStart:
    @pytest.fixture(autouse=True)
    def machinery(self, github, trusted, git):
        self.git = git
        self.git.side_effect = lambda cmd, label, timeout: Mock(
            returncode=0, stderr="",
            stdout={"rev-parse": "release\n" if "--abbrev-ref" in cmd else "abcdef12\n"}.get(cmd[1], ""))
        with patch.object(CodeQualityMaintainer, "_sync_requirements", return_value=("", "")) as sync, \
                patch.object(CodeQualityMaintainer, "restart") as restart, \
                patch.object(CodeQualityMaintainer, "target_branch", return_value="release"):
            self.sync, self.restart = sync, restart
            yield

    def _git_calls(self):
        return [c.args[0] for c in self.git.call_args_list if c.args[0][0] == "git"]

    def test_fetches_the_head_checks_it_out_syncs_and_restarts_with_the_fallback(self):
        result = PullRequestTrial.start(204)
        calls = self._git_calls()
        fetch, checkout = ["git", "fetch", "origin", "pull/204/head"], ["git", "checkout", "-B", "trial/pr-204", "FETCH_HEAD"]
        assert fetch in calls and checkout in calls and calls.index(checkout) > calls.index(fetch), \
            "fetched into FETCH_HEAD so a second try of the tried PR can move the checked-out branch"
        self.sync.assert_called_once()
        self.restart.assert_called_once()
        assert self.restart.call_args.kwargs == {"fallback_branch": "release", "resumes_conversation": True}, \
            "the user asked for the trial in a chat and hears there whether its server came up"
        assert "PR #204" in result and "abcdef12" in result and "release" in result

    def test_the_branch_the_install_follows_is_merged_into_the_checked_out_pr(self):
        PullRequestTrial.start(204)
        calls = self._git_calls()
        merge = next(c for c in calls if "merge" in c and "--abort" not in c)
        assert merge[-1] == "abcdef12", "the commit `release` resolves to on this machine"
        assert calls.index(merge) > calls.index(["git", "checkout", "-B", "trial/pr-204", "FETCH_HEAD"])

    def test_a_pr_that_does_not_merge_with_the_release_is_not_tried(self):
        self.git.side_effect = lambda cmd, label, timeout: Mock(
            returncode=1 if "merge" in cmd and "--abort" not in cmd else 0, stderr="",
            stdout={"rev-parse": "release\n" if "--abbrev-ref" in cmd else "abcdef12\n"}.get(cmd[1], ""))
        result = PullRequestTrial.start(204)
        calls = self._git_calls()
        assert ["git", "merge", "--abort"] in calls and calls[-1] == ["git", "checkout", "release"]
        assert "not tried" in result and "release" in result
        self.sync.assert_not_called()
        self.restart.assert_not_called()

    def test_a_second_try_that_fails_puts_the_running_trial_back_where_it_stood(self):
        self.sync.return_value = ("pip exited 1", "")
        self.git.side_effect = lambda cmd, label, timeout: Mock(
            returncode=0, stderr="",
            stdout={"rev-parse": "trial/pr-204\n" if "--abbrev-ref" in cmd else "abcdef12\n"}.get(cmd[1], ""))
        PullRequestTrial.start(204)
        assert self._git_calls()[-1] == ["git", "checkout", "-B", "trial/pr-204", "abcdef12"]

    def test_a_refusal_fetches_nothing(self, github):
        github.answer(_pr(state="CLOSED"))
        result = PullRequestTrial.start(204)
        assert result.startswith("Cannot try PR #204")
        assert not any(c[1] == "fetch" for c in self._git_calls())
        self.restart.assert_not_called()

    def test_a_failed_fetch_does_not_restart(self):
        self.git.side_effect = lambda cmd, label, timeout: Mock(
            returncode=1 if cmd[1] == "fetch" else 0, stdout="release\n" if cmd[1] == "rev-parse" else "",
            stderr="fatal: couldn't find remote ref")
        result = PullRequestTrial.start(204)
        assert "couldn't find remote ref" in result
        self.restart.assert_not_called()

    def test_a_failed_checkout_does_not_sync_or_restart(self):
        self.git.side_effect = lambda cmd, label, timeout: Mock(
            returncode=1 if cmd[1] == "checkout" else 0, stdout="release\n" if cmd[1] == "rev-parse" else "",
            stderr="error: your local changes would be overwritten")
        result = PullRequestTrial.start(204)
        assert "would be overwritten" in result and result.startswith("Cannot try PR #204")
        self.sync.assert_not_called()
        self.restart.assert_not_called()

    def test_packages_that_do_not_install_put_the_previous_branch_back(self):
        self.sync.return_value = ("pip exited 1 (no matching distribution)", "")
        result = PullRequestTrial.start(204)
        assert self._git_calls()[-1] == ["git", "checkout", "release"]
        assert "no matching distribution" in result and "release" in result
        self.restart.assert_not_called()

    def test_installed_packages_are_named(self):
        self.sync.return_value = ("", "installed 1 package: pkg-a (missing)")
        assert "pkg-a" in PullRequestTrial.start(204)


class TestOvernight:
    """The nightly update of a running trial, on a throwaway repository: `release` and the
    trial branch part at one commit, the PR changed b.txt, and each test says what the
    release did since. GitHub answers with `self.pr` (an exception is raised), read fresh."""

    WHEN = datetime(2026, 3, 2, 12, tzinfo=timezone.utc)

    BASE = {"a.txt": "one\n", "b.txt": "one\n"}

    @pytest.fixture(autouse=True)
    def install(self, tmp_path):
        self.repo = repo = empty_repo(tmp_path / "install")
        self.pr = _pr()
        released, self.pr_head = self._write(("the PR", {"b.txt": "the PR's\n"}), release=1, trial=2)
        (repo / ".git" / "HEAD").write_text("ref: refs/heads/trial/pr-204\n")
        sandbox_git(repo, "reset", "-q", "--hard")
        self.trial = Trial(number=204, title="Tesla: close the trunk once", head_ref="feat/trunk",
                           since="14:02", return_branch="release")
        run = lambda cmd, label, timeout: spawn([cmd[0], "-C", str(repo), *cmd[1:]], timeout=timeout)
        with patch.object(CodeQualityMaintainer, "_run_subprocess", side_effect=run), \
                patch.object(PullRequestTrial, "_pr", SimpleNamespace(__wrapped__=self._read_pr)), \
                patch.object(Updater, "tip", side_effect=lambda target: self.tip) as tip:
            self.tip_asked = tip
            self.tip = ("", released)
            yield

    def _read_pr(self, number: int) -> dict:
        if isinstance(self.pr, Exception):
            raise self.pr
        return self.pr

    def _write(self, change: tuple, release: int, trial: int = 0) -> list[str]:
        """The commit both branches start from and one `change` (message, files) on top of it,
        as objects and refs alone — a git process is a tenth of what a test here may take.
        The first commit is the same object every time, so a second call forks the history."""
        history = History()
        history.commit(self.WHEN, "the release the trial began on", self.BASE)
        history.commit(self.WHEN, change[0], {**self.BASE, **change[1]})
        history.branch("release", release)
        if trial:
            history.branch("trial/pr-204", trial)
        hashes = history.write(self.repo)
        (self.repo / ".git" / "refs" / "heads" / "master").unlink()
        return hashes

    def _release(self, files: dict) -> str:
        """A release made since the trial began, fetched and verified: what `Updater.tip` hands over."""
        commit = self._write(("the next release", files), release=2)[-1]
        self.tip = ("", commit)
        return commit

    def _head(self, ref: str) -> str:
        return sandbox_git(self.repo, "rev-parse", ref)

    def _holds(self, commit: str) -> bool:
        return spawn(["git", "-C", str(self.repo), "merge-base", "--is-ancestor", commit, "HEAD"]).returncode == 0

    def _on_trial_and_clean(self) -> bool:
        return sandbox_git(self.repo, "rev-parse", "--abbrev-ref", "HEAD") == "trial/pr-204" \
            and not sandbox_git(self.repo, "status", "--porcelain")

    def test_a_new_release_is_merged_in_and_the_trial_lasts(self):
        released = self._release({"a.txt": "two\n"})
        night = PullRequestTrial.overnight(self.trial)
        assert night == module.Overnight()
        assert self._holds(released) and self._holds(self.pr_head) and self._on_trial_and_clean()
        assert (self.repo / "a.txt").read_text() == "two\n" and (self.repo / "b.txt").read_text() == "the PR's\n"

    def test_a_night_without_a_release_moves_nothing(self):
        assert PullRequestTrial.overnight(self.trial) == module.Overnight()
        assert self._head("HEAD") == self.pr_head, "an unmoved HEAD is what tells the caller not to restart"

    def test_a_merged_pr_stays_on_trial_until_the_release_holds_it(self):
        self.pr = _pr(state="MERGED", mergeCommit={"oid": "d" * 40})
        released = self._release({"a.txt": "two\n"})
        night = PullRequestTrial.overnight(self.trial)
        assert not night.ended and not night.failure
        assert self._holds(released) and self._holds(self.pr_head)

    def test_a_merged_pr_the_release_holds_ends_the_trial(self):
        """Squash-merged: the release holds the merge commit, never the PR's own."""
        released = self._release({"b.txt": "the PR's\n"})
        self.pr = _pr(state="MERGED", mergeCommit={"oid": released})
        night = PullRequestTrial.overnight(self.trial)
        assert "release" in night.ended and "merged" in night.ended
        assert self._head("HEAD") == self.pr_head, "the way back to release is the caller's normal update"

    def test_a_release_the_pr_conflicts_with_ends_the_trial_and_leaves_no_merge_behind(self):
        self._release({"b.txt": "the release's\n"})
        night = PullRequestTrial.overnight(self.trial)
        assert "release" in night.ended and not night.failure
        assert self._head("HEAD") == self.pr_head and self._on_trial_and_clean()

    def test_a_closed_pr_ends_the_trial_without_a_fetch(self):
        self.pr = _pr(state="CLOSED")
        assert "closed" in PullRequestTrial.overnight(self.trial).ended
        self.tip_asked.assert_not_called()

    def test_github_out_of_reach_keeps_the_trial_as_it_is(self):
        self.pr = RuntimeError("gh: could not resolve host")
        self._release({"a.txt": "two\n"})
        night = PullRequestTrial.overnight(self.trial)
        assert not night.ended and "could not resolve host" in night.failure
        assert self._head("HEAD") == self.pr_head

    def test_a_release_that_does_not_verify_keeps_the_trial_as_it_is(self):
        self._release({"a.txt": "two\n"})
        self.tip = ("The newest release carries no tag signed by a key in release_signers", "")
        night = PullRequestTrial.overnight(self.trial)
        assert not night.ended and "no tag signed" in night.failure
        assert self._head("HEAD") == self.pr_head

    def test_uncommitted_work_in_the_way_of_the_merge_is_a_failure_not_the_end(self):
        self._release({"a.txt": "two\n"})
        (self.repo / "a.txt").write_text("somebody's edit\n")
        night = PullRequestTrial.overnight(self.trial)
        assert night.failure and not night.ended
        assert (self.repo / "a.txt").read_text() == "somebody's edit\n"

    def test_a_checkout_moved_off_the_trial_branch_is_the_end_of_the_trial(self):
        sandbox_git(self.repo, "checkout", "-q", "release")
        assert "release" in PullRequestTrial.overnight(self.trial).ended


class TestFeedback:
    def test_the_comment_carries_the_verdict_and_the_version(self, git):
        git.return_value = Mock(returncode=0, stdout="https://github.com/example/keep/pull/204#issuecomment-1\n", stderr="")
        with patch.object(module.Version, "branch", return_value="trial/pr-204"), \
                patch.object(module.Version, "detail", return_value="v2026.10.3 (abcdef12, trial/pr-204)"), \
                patch.object(PullRequestTrial, "_pr", return_value=_pr()), \
                patch.object(PullRequestTrial, "_since", return_value="14:02"), \
                patch.object(CodeQualityMaintainer, "target_branch", return_value="release"):
            result = PullRequestTrial.feedback(204, "tested, the trunk closes once now")
        cmd = git.call_args.args[0]
        assert cmd[:4] == ["gh", "pr", "comment", "204"]
        body = cmd[cmd.index("--body") + 1]
        assert "tested, the trunk closes once now" in body and "abcdef12" in body and "trial" in body
        assert "issuecomment-1" in result

    def test_feedback_on_a_pr_that_is_not_the_trial_says_so(self, git):
        with patch.object(module.Version, "branch", return_value="release"), \
                patch.object(module.Version, "detail", return_value="v2026.10.3 (abcdef12, release)"):
            PullRequestTrial.feedback(31, "works")
        body = git.call_args.args[0][-1]
        assert "trial" not in body and "abcdef12" in body

    def test_a_failed_comment_is_reported(self, git):
        git.return_value = Mock(returncode=1, stdout="", stderr="gh: not logged in")
        with patch.object(module.Version, "branch", return_value="release"), \
                patch.object(module.Version, "detail", return_value="v1"):
            assert "not logged in" in PullRequestTrial.feedback(31, "works")

    def test_nothing_is_posted_without_words(self, git):
        assert "nothing" in PullRequestTrial.feedback(31, "  ")
        git.assert_not_called()


class TestRollbackNote:
    def test_the_note_start_sh_left_is_told_once(self, tmp_path):
        path = tmp_path / "trial_rollback.txt"
        path.write_text("The server did not come up on the pull request's code within 120s, so it is back on release.\nTraceback...")
        with patch.object(module, "ROLLBACK_NOTE_PATH", path):
            note = PullRequestTrial.take_rollback_note()
            assert "back on release" in note and "Traceback" in note
            assert not path.exists()
            assert PullRequestTrial.take_rollback_note() is None

    def test_no_note_without_a_rollback(self, tmp_path):
        with patch.object(module, "ROLLBACK_NOTE_PATH", tmp_path / "missing.txt"):
            assert PullRequestTrial.take_rollback_note() is None

    def test_a_long_note_keeps_its_tail_where_the_error_is(self, tmp_path):
        path = tmp_path / "trial_rollback.txt"
        path.write_text("HEAD " + "x" * (3 * module.ROLLBACK_NOTE_CHARS) + " TAIL")
        with patch.object(module, "ROLLBACK_NOTE_PATH", path):
            note = PullRequestTrial.take_rollback_note()
        assert note.endswith("TAIL") and "HEAD" not in note
        assert len(note) < module.ROLLBACK_NOTE_CHARS + 50

    def test_a_blank_note_is_nothing_to_tell_and_is_removed_all_the_same(self, tmp_path):
        path = tmp_path / "trial_rollback.txt"
        path.write_text("  \n\n")
        with patch.object(module, "ROLLBACK_NOTE_PATH", path):
            assert PullRequestTrial.take_rollback_note() is None
        assert not path.exists()


@pytest.mark.parametrize("text, number", [("204", 204), ("#204", 204), (" PR 204 ", 204), ("pr#204", 204),
                                          ("https://github.com/example/keep/pull/204", 204), ("204, fix", None),
                                          ("", None), ("the trunk one", None)])
def test_the_number_is_read_from_what_the_llm_passes(text, number):
    assert parse_number(text) == number
