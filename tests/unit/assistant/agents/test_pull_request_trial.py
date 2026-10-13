"""PullRequestTrial: the branch is the state, the vetting is the auto-merge's trust, and a
trial's restart hands start.sh the branch to fall back to."""
import json
from unittest.mock import Mock, patch, call

import pytest

from assistant.agents import PullRequestTrial as module
from assistant.agents.PullRequestTrial import PullRequestTrial, TrialRefused, parse_number, trial_number, GIT_TIMEOUT
from assistant.agents.CodeQualityMaintainer import CodeQualityMaintainer


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
            text = PullRequestTrial.pr_job_instructions()
        assert "refs/heads/feat/trunk" in text and "PR #204" in text and "trial/pr-204" in text
        assert "do NOT create a branch" in text


class TestVetting:
    def test_an_open_pr_of_this_repository_by_an_own_author_passes(self, github, trusted, git):
        assert PullRequestTrial.vet(204)["title"] == "Tesla: close the trunk once"

    @pytest.mark.parametrize("pr, words", [
        (_pr(state="MERGED"), "merged"),
        (_pr(state="CLOSED"), "closed"),
        (_pr(isCrossRepository=True), "fork"),
        (_pr(author={"login": "stranger"}), "stranger"),
    ])
    def test_a_pr_that_is_not_the_installs_own_open_one_is_refused(self, github, trusted, git, pr, words):
        github.answer(pr)
        with pytest.raises(TrialRefused) as refused:
            PullRequestTrial.vet(204)
        assert words in str(refused.value)

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
        github.answer(_pr(state="MERGED"))
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
        assert self.restart.call_args.kwargs == {"fallback_branch": "release"}
        assert "PR #204" in result and "abcdef12" in result and "release" in result

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
