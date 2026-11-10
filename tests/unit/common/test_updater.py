"""common/Updater: the branch, the pull, the requirements sync, the restart verdict and the relaunch —
with the checkout, the signed release, pip and the start script all faked. Nothing here runs git."""
import ast
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

import common.Updater as updater_module
from common.ConfigManager import ConfigManager
from common.Paths import AISERVER_DIR, REPO_DIR
from common.Updater import GIT_TIMEOUT, Updater, _pull_summary

BOOT, NEWER = "b" * 40, "c" * 40
PULLED = ("Updating bbbbbbb..ccccccc\nFast-forward\n src/a.py | 12 ++--\n tests/b.py | 40 +++\n"
          " 2 files changed, 44 insertions(+), 8 deletions(-)\n")


class FakeCheckout:
    """Answers the git commands Updater runs the way a checkout would and keeps what it was asked."""

    def __init__(self, branch="master", head=NEWER, ahead=2, pulled=PULLED, failing=None, events=None):
        self.branch, self.head, self.ahead, self.pulled = branch, head, ahead, pulled
        self.failing = failing or {}          # git verb → stderr of a failing exit
        self.events = events if events is not None else []

    def __call__(self, cmd, label, timeout):
        self.events.append(cmd)
        verb = cmd[1]
        if verb in self.failing:
            return subprocess.CompletedProcess(cmd, 1, "", self.failing[verb])
        if verb == "checkout":
            self.branch = cmd[2]
        out = {"rev-parse": self.branch if "--abbrev-ref" in cmd else self.head,
               "rev-list": str(self.ahead), "pull": self.pulled}.get(verb, "")
        return subprocess.CompletedProcess(cmd, 0, f"{out}\n" if out else "", "")

    @property
    def moves(self) -> list:
        """The commands that change the checkout, in order — the reads left out."""
        return [cmd for cmd in self.events if isinstance(cmd, list) and cmd[1] not in ("rev-parse", "rev-list")]


@pytest.fixture(autouse=True)
def signed_release():
    """SignedRelease fetches and verifies for real; here it finds the release current."""
    with patch("common.SignedRelease.SignedRelease.update", return_value=("", "Already up to date.")) as update:
        yield update


@pytest.fixture(autouse=True)
def sync():
    """The venv matches the pins unless a test says otherwise — nothing here runs pip."""
    with patch("common.RequirementsChecker.RequirementsChecker.sync", return_value=("", "")) as sync_:
        yield sync_


@pytest.fixture(autouse=True)
def popen():
    with patch("common.Updater.subprocess.Popen") as popen_:
        yield popen_


@pytest.mark.unit
class TestBranchAndCommit:
    def test_the_branch_is_the_one_git_names(self):
        assert Updater(run=FakeCheckout(branch="release")).current_branch() == "release"

    def test_a_detached_head_or_a_failing_git_counts_as_master(self):
        assert Updater(run=FakeCheckout(branch="HEAD")).current_branch() == "master"

        def no_git(cmd, label, timeout):
            raise OSError("no git")
        assert Updater(run=no_git).current_branch() == "master"

    def test_the_head_commit_is_the_trimmed_hash(self):
        assert Updater(run=FakeCheckout(head=NEWER)).head_commit() == NEWER

    def test_the_head_commit_is_empty_when_git_fails(self):
        assert Updater(run=FakeCheckout(failing={"rev-parse": "fatal: not a git repository"})).head_commit() == ""

        def no_git(cmd, label, timeout):
            raise OSError("no git")
        assert Updater(run=no_git).head_commit() == ""

    def test_commits_between_counts_the_range_and_is_zero_on_failure(self):
        checkout = FakeCheckout(ahead=3)
        assert Updater(run=checkout).commits_between(BOOT, NEWER) == 3
        assert checkout.events[-1][-1] == f"{BOOT}..{NEWER}"
        assert Updater(run=FakeCheckout(failing={"rev-list": "bad revision"})).commits_between(BOOT, NEWER) == 0


@pytest.mark.unit
class TestPull:
    def test_it_pulls_in_place_when_the_checkout_is_on_the_target(self, signed_release):
        checkout = FakeCheckout(branch="master")
        failure, output = Updater(run=checkout).pull("master")
        assert failure == "" and checkout.moves == [["git", "pull"]]
        assert "2 files changed" in output and "src/a.py" not in output   # the diffstat is not carried on
        signed_release.assert_not_called()

    def test_a_target_the_checkout_is_not_on_is_fetched_and_checked_out_before_the_pull(self):
        """The target is whatever the caller names — no branch but `release` is special here."""
        checkout = FakeCheckout(branch="release")
        failure, _ = Updater(run=checkout).pull("next")
        assert failure == ""
        assert checkout.moves == [["git", "fetch", "origin", "next"], ["git", "checkout", "next"], ["git", "pull"]]

    def test_a_failed_switch_says_where_the_checkout_stayed_and_does_not_pull(self):
        checkout = FakeCheckout(branch="release", failing={"checkout": "error: your local changes would be overwritten"})
        failure, output = Updater(run=checkout).pull("master")
        assert "master" in failure and "still on release" in failure and "local changes" in failure
        assert output == "" and ["git", "pull"] not in checkout.moves

    def test_a_failed_pull_is_the_failure(self):
        failure, output = Updater(run=FakeCheckout(failing={"pull": "fatal: unable to access the remote"})).pull("master")
        assert "unable to access the remote" in failure and output == ""

    @pytest.mark.parametrize("current", ["release", "master"])
    def test_release_moves_only_through_the_signed_release(self, signed_release, current):
        checkout = FakeCheckout(branch=current)
        assert Updater(run=checkout).pull("release") == ("", "Already up to date.")
        signed_release.assert_called_once_with(current)   # which also does the switch, after the tag verified
        assert checkout.moves == []

    def test_a_release_that_fails_verification_is_the_failure(self, signed_release):
        signed_release.return_value = ("The newest release (1b6c9ce7) carries no tag signed by a key in release_signers", "")
        checkout = FakeCheckout(branch="release")
        failure, _ = Updater(run=checkout).pull("release")
        assert "1b6c9ce7" in failure and checkout.moves == []


@pytest.mark.unit
class TestPullCurrent:
    def test_any_other_branch_is_pulled_where_it_is(self, signed_release):
        checkout = FakeCheckout(branch="tesla-trunk")
        Updater(run=checkout).pull_current()
        assert checkout.moves == [["git", "pull"]]
        signed_release.assert_not_called()

    def test_on_release_it_is_the_signed_release_not_a_plain_pull(self, signed_release):
        signed_release.return_value = ("The newest release carries no tag signed by a key in release_signers", "")
        checkout = FakeCheckout(branch="release")
        Updater(run=checkout).pull_current()   # the refusal is logged, the caller goes on
        signed_release.assert_called_once_with("release")
        assert checkout.moves == []


@pytest.mark.unit
class TestUpdate:
    """The unattended update: restart only when the code left the commit the process booted on."""

    def test_new_commits_are_synced_and_ask_for_a_restart(self, sync):
        events = []
        sync.side_effect = lambda: (events.append("pip"), ("", ""))[1]
        result = Updater(run=FakeCheckout(branch="master", head=NEWER, ahead=2, events=events)).update("master", BOOT)
        assert result.restart and not result.failure
        assert (result.target, result.head, result.commits) == ("master", NEWER, 2)
        assert "2 new commits" in result.change
        assert events.index(["git", "pull"]) < events.index("pip")

    def test_one_commit_is_not_spoken_of_in_the_plural(self):
        change = Updater(run=FakeCheckout(ahead=1)).update("master", BOOT).change
        assert "1 new commit" in change and "commits" not in change

    def test_no_new_commit_means_no_pip_and_no_restart(self, sync):
        result = Updater(run=FakeCheckout(head=BOOT)).update("master", BOOT)
        assert not result.restart and not result.failure and result.head == BOOT
        sync.assert_not_called()   # pip only runs ahead of a restart

    def test_a_pin_bump_is_named_in_the_change(self, sync):
        sync.return_value = ("", "installed 1 package: pkg-a 1.0.0 → 2.0.0")
        result = Updater(run=FakeCheckout()).update("master", BOOT)
        assert result.restart and result.installed in result.change and "2 new commits" in result.change

    def test_a_failed_sync_keeps_the_old_process_and_says_what_was_pulled(self, sync):
        sync.return_value = ("pip exited 1 (no matching distribution)", "")
        result = Updater(run=FakeCheckout()).update("master", BOOT)
        assert not result.restart
        assert result.sync_failure == result.failure == "pip exited 1 (no matching distribution)"
        assert result.pull_failure == "" and "2 new commits" in result.change

    def test_a_failed_pull_keeps_the_old_process_and_never_reaches_pip(self, sync):
        checkout = FakeCheckout(failing={"pull": "merge conflict in config"})
        result = Updater(run=checkout).update("master", BOOT)
        assert not result.restart and "merge conflict in config" in result.pull_failure
        assert result.sync_failure == "" and result.failure == result.pull_failure
        sync.assert_not_called()

    def test_an_unreadable_head_after_the_pull_does_not_restart(self, sync):
        class HeadGone(FakeCheckout):
            def __call__(self, cmd, label, timeout):
                if cmd == ["git", "rev-parse", "HEAD"]:
                    return subprocess.CompletedProcess(cmd, 128, "", "fatal: bad object HEAD")
                return super().__call__(cmd, label, timeout)
        result = Updater(run=HeadGone()).update("master", BOOT)
        assert not result.restart and result.head == "" and not result.failure
        sync.assert_not_called()

    def test_an_unknown_boot_commit_is_no_update_at_all(self, sync, signed_release):
        """Every commit would look new: the process would restart at each call, forever."""
        checkout = FakeCheckout()
        for target in ("master", "release"):
            result = Updater(run=checkout).update(target, "")
            assert not result.restart and not result.failure
        assert checkout.events == []
        sync.assert_not_called()
        signed_release.assert_not_called()

    def test_a_switch_to_a_branch_that_is_not_ahead_still_restarts(self, signed_release):
        """release may sit behind the commit a master process booted on: nothing is "new", the
        code is still different."""
        result = Updater(run=FakeCheckout(branch="master", head=NEWER, ahead=0)).update("release", BOOT)
        signed_release.assert_called_once_with("master")
        assert result.restart and result.commits == 0 and NEWER[:8] in result.change


@pytest.mark.unit
class TestRefresh:
    """The update of a restart somebody asked for: it happens whatever the pull brings."""

    def test_nothing_new_still_asks_for_the_restart(self, sync):
        result = Updater(run=FakeCheckout(pulled="Already up to date.")).refresh("master")
        assert result.restart and result.up_to_date
        sync.assert_called_once_with()   # a hand pull before the restart may have bumped a pin

    def test_what_the_pull_brought_and_what_pip_installed_come_back(self, sync):
        events = []
        sync.side_effect = lambda: (events.append("pip"), ("", "installed 1 package: pkg-a (missing)"))[1]
        result = Updater(run=FakeCheckout(events=events)).refresh("master")
        assert result.restart and not result.up_to_date
        assert "bbbbbbb..ccccccc" in result.pulled and "pkg-a" in result.installed
        assert events.index(["git", "pull"]) < events.index("pip")

    def test_a_failed_pull_never_reaches_pip(self, sync):
        result = Updater(run=FakeCheckout(failing={"pull": "fatal: not a git repository"})).refresh("master")
        assert not result.restart and "not a git repository" in result.pull_failure
        sync.assert_not_called()

    def test_a_failed_sync_keeps_the_old_process(self, sync):
        sync.return_value = ("pip exited 1 (no matching distribution)", "")
        result = Updater(run=FakeCheckout()).refresh("master")
        assert not result.restart and "no matching distribution" in result.sync_failure
        assert result.pull_failure == ""

    def test_a_release_that_fails_verification_is_a_pull_failure(self, signed_release, sync):
        signed_release.return_value = ("The newest release (1b6c9ce7) carries no tag signed by a key in release_signers", "")
        result = Updater(run=FakeCheckout(branch="release")).refresh("release")
        assert not result.restart and "1b6c9ce7" in result.pull_failure
        sync.assert_not_called()


@pytest.mark.unit
class TestRelaunch:
    def test_the_named_script_gets_this_process_s_pid_then_the_caller_s_arguments(self, popen):
        """A contract with the start scripts: the first argument is the pid of the process to take
        down (pgrep misses a server that spawned the script itself), the rest is the caller's."""
        Updater.relaunch("satellite_start.sh", "the server moved to v2026.11.4")
        assert popen.call_args.args[0] == ["bash", "satellite_start.sh", str(os.getpid())]
        Updater.relaunch("start.sh", "trying PR #204", "release")
        assert popen.call_args.args[0] == ["bash", "start.sh", str(os.getpid()), "release"]

    def test_the_script_runs_from_the_root_detached_from_the_process_it_kills(self, popen):
        Updater.relaunch("start.sh")
        kwargs = popen.call_args.kwargs
        assert Path(kwargs["cwd"]) == AISERVER_DIR
        assert kwargs["start_new_session"] is True   # or the script dies with the process it takes down
        assert kwargs["stdin"] == kwargs["stdout"] == kwargs["stderr"] == subprocess.DEVNULL

    def test_a_relaunch_pulls_nothing(self, popen, signed_release, sync):
        with patch("common.Updater.subprocess.run") as run:
            Updater.relaunch("start.sh", "audio recovery")
        run.assert_not_called()
        signed_release.assert_not_called()
        sync.assert_not_called()


@pytest.mark.unit
class TestStandsAlone:
    """A process without an assistant updates through this module: it knows no caller and no config."""

    def test_no_step_reads_the_config(self, popen):
        with patch.object(ConfigManager, "get", side_effect=AssertionError("Updater read config.yaml")):
            updater = Updater(run=FakeCheckout(branch="release"))
            assert updater.update("master", BOOT).restart
            assert updater.refresh("release").restart
            updater.pull_current()
            Updater.relaunch("satellite_start.sh", "moved to master")
        popen.assert_called_once()

    def test_it_imports_nothing_of_the_assistant_the_interfaces_or_the_integrations_nor_the_config(self):
        tree = ast.parse(Path(updater_module.__file__).read_text(encoding="utf-8"))
        imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names} | \
                   {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        roots = {name.split(".")[0] for name in imported}
        assert not roots & {"assistant", "interfaces", "integrations"}
        assert not any("Config" in name for name in imported)

    def test_without_a_runner_git_runs_in_the_checkout_and_never_waits_for_input(self):
        """A git that asks for a credential would hang on the terminal the process was started from."""
        with patch("common.Updater.subprocess.run",
                   return_value=subprocess.CompletedProcess([], 0, f"{NEWER}\n", "")) as run:
            assert Updater().head_commit() == NEWER
        kwargs = run.call_args.kwargs
        assert Path(kwargs["cwd"]) == REPO_DIR and kwargs["stdin"] == subprocess.DEVNULL
        assert kwargs["timeout"] == GIT_TIMEOUT


@pytest.mark.unit
class TestPullSummary:
    """The pull output is logged, sent to the chat and carried through the restart as the
    resumed turn's query, so the per-file diffstat was repeated a dozen times."""

    def test_the_diffstat_is_dropped_and_the_first_and_summary_lines_kept(self):
        stdout = ("Updating 5e70670..bcff709\nFast-forward\n src/a.py | 12 ++--\n tests/b.py | 40 +++\n"
                  " 13 files changed, 176 insertions(+), 51 deletions(-)\n")
        assert _pull_summary(stdout) == "Updating 5e70670..bcff709, 13 files changed, 176 insertions(+), 51 deletions(-)"
        assert _pull_summary("Already up to date.\n") == "Already up to date."
        assert _pull_summary("Updating a..b\nFast-forward\n src/a.py | 1 +\n 1 file changed, 1 insertion(+)") == \
            "Updating a..b, 1 file changed, 1 insertion(+)"
        assert _pull_summary("") == ""

