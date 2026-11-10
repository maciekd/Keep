"""ServerFollower: when a satellite updates itself to its server's version, and when it leaves
well alone — over a fake Updater and a fake clock; nothing here runs git, pip or a start script."""
import os
from unittest.mock import Mock, patch

import pytest

import common.DevelopmentConfig as DevelopmentConfig
from common.Updater import Update, Updater
from common.imessage import Wire
from common.imessage.ServerFollower import FOLLOWED_BRANCHES, MIN_INTERVAL_SECONDS, RETRY_SECONDS, ServerFollower
from tests.unit.common.test_updater import FakeCheckout

BOOT, NEWER, NEWEST = "b0" * 20, "c1" * 20, "d2" * 20


class Later:
    """Where an update waits instead of a thread, until the test runs it."""

    def __init__(self):
        self.waiting = []

    def __call__(self, work):
        self.waiting.append(work)

    def run(self):
        while self.waiting:
            self.waiting.pop(0)()


class Clock:
    now = 1000.0

    def __call__(self):
        return self.now


def restart(target, boot):
    return Update(target, pulled="Updating b0b0b0b..c1c1c1c", head=NEWER, commits=3, restart=True)


@pytest.fixture
def parts():
    updater = Mock(spec=Updater)
    updater.update.side_effect = restart
    updater.commits_between.return_value = 0
    return {"updater": updater, "relaunch": Mock(), "spawn": Later(), "clock": Clock()}


@pytest.fixture
def follower(parts):
    return ServerFollower("release", BOOT, **parts)


@pytest.mark.unit
class TestWhenItUpdates:
    def test_a_server_on_the_same_branch_and_commit_moves_nothing(self, follower, parts):
        for _ in range(3):
            follower.notice("release", BOOT)
        assert parts["spawn"].waiting == [] and not follower.status().updating

    def test_another_commit_is_pulled_off_the_request_and_then_the_process_relaunched(self, follower, parts):
        follower.notice("release", NEWER)
        parts["updater"].update.assert_not_called()     # the request that brought the news is answered first
        assert follower.status().updating
        parts["spawn"].run()
        parts["updater"].update.assert_called_once_with("release", BOOT)
        assert parts["relaunch"].call_args.args[0] == "start.sh"
        assert follower.status().update_failure == ""

    def test_another_branch_of_the_two_is_followed_too(self, follower, parts):
        """A server that switched from release to master takes its satellite along."""
        follower.notice("master", BOOT)
        parts["spawn"].run()
        parts["updater"].update.assert_called_once_with("master", BOOT)

    @pytest.mark.parametrize("branch", ["trial/pr-212", "fix/login", "main", "Release", "", "release; rm -rf ~"])
    def test_a_server_on_any_other_branch_moves_nothing(self, follower, parts, branch):
        follower.notice(branch, NEWER)
        assert parts["spawn"].waiting == []
        assert follower.status().update_failure == "" and not follower.status().updating

    def test_the_followed_branches_are_the_two_lines_of_the_code(self):
        """A contract with scripts/release.sh and the `developer_release` key: these are the lines an install tracks."""
        assert set(FOLLOWED_BRANCHES) == {"release", "master"}

    @pytest.mark.parametrize("commit", ["", "HEAD", "c1c1", "$(reboot)", NEWER + "\n--upload-pack=x", NEWER.upper()])
    def test_a_commit_that_is_not_a_hash_is_no_news(self, follower, parts, commit):
        follower.notice("release", commit)
        assert parts["spawn"].waiting == []

    def test_developer_mode_follows_nobody(self, follower, parts):
        with patch.object(DevelopmentConfig, "LOCAL_DEVELOPMENT_MODE", True):
            follower.notice("release", NEWER)
        assert parts["spawn"].waiting == []

    def test_an_install_that_cannot_tell_its_own_commit_never_updates_and_says_so(self, parts):
        """Every commit would look new, and the process would restart at each poll."""
        follower = ServerFollower("release", "", **parts)
        follower.notice("release", NEWER)
        assert parts["spawn"].waiting == []
        assert follower.status().update_failure and follower.status().commit == ""


@pytest.mark.unit
class TestOneAtATimeAndNotTheSameTwice:
    def test_news_that_arrives_during_an_update_starts_no_second_one(self, follower, parts):
        for commit in (NEWER, NEWER, NEWEST):
            follower.notice("release", commit)
        assert len(parts["spawn"].waiting) == 1

    def test_a_target_that_failed_is_not_tried_again_at_every_poll(self, follower, parts):
        parts["updater"].update.side_effect = lambda target, boot: Update(target, pull_failure="Git pull failed: no route to host")
        for _ in range(50):
            follower.notice("release", NEWER)
            parts["spawn"].run()
        assert parts["updater"].update.call_count == 1

    def test_it_is_tried_again_once_the_retry_time_has_passed(self, follower, parts):
        parts["updater"].update.side_effect = lambda target, boot: Update(target, pull_failure="Git pull failed")
        follower.notice("release", NEWER)
        parts["spawn"].run()
        parts["clock"].now += RETRY_SECONDS - 1
        follower.notice("release", NEWER)
        parts["spawn"].run()
        assert parts["updater"].update.call_count == 1
        parts["clock"].now += 2
        follower.notice("release", NEWER)
        parts["spawn"].run()
        assert parts["updater"].update.call_count == 2

    def test_a_server_that_moved_on_is_followed_without_waiting_out_the_hour(self, follower, parts):
        parts["updater"].update.side_effect = lambda target, boot: Update(target, sync_failure="pip failed")
        follower.notice("release", NEWER)
        parts["spawn"].run()
        parts["clock"].now += MIN_INTERVAL_SECONDS
        follower.notice("release", NEWEST)
        parts["spawn"].run()
        assert parts["updater"].update.call_count == 2

    def test_a_caller_that_names_a_new_commit_with_every_request_gets_no_pull_for_each(self, follower, parts):
        """Whoever holds the token could otherwise keep the satellite pulling without a pause."""
        parts["updater"].update.side_effect = lambda target, boot: Update(target, pull_failure="Git pull failed")
        for request in range(40):
            follower.notice("release", f"{request:040x}")
            parts["spawn"].run()
            parts["clock"].now += MIN_INTERVAL_SECONDS / 10
        assert parts["updater"].update.call_count == 4

    def test_a_commit_this_mac_cannot_pull_is_not_pulled_at_every_poll_either(self, follower, parts):
        """The pull goes through and brings nothing — an unpushed commit, a release not signed yet."""
        parts["updater"].update.side_effect = lambda target, boot: Update(target, pulled="Already up to date.", head=BOOT)
        for _ in range(50):
            follower.notice("master", NEWER)
            parts["spawn"].run()
        assert parts["updater"].update.call_count == 1
        parts["relaunch"].assert_not_called()
        failure = follower.status().update_failure
        assert NEWER[:8] in failure and BOOT[:8] in failure


    def test_a_satellite_ahead_of_its_server_waits_for_it_without_calling_that_a_failure(self, follower, parts):
        """A satellite installed today, a server that updates tonight: the pull brings nothing
        because there is nothing newer, and the server's commit is behind this Mac's."""
        parts["updater"].update.side_effect = lambda target, boot: Update(target, pulled="Already up to date.", head=BOOT)
        parts["updater"].commits_between.side_effect = [0, 4]     # not known before the pull fetched, known after
        for _ in range(10):
            follower.notice("release", NEWER)
            parts["spawn"].run()
        assert parts["updater"].update.call_count == 1
        assert follower.status().update_failure == "" and not follower.status().updating
        parts["relaunch"].assert_not_called()

    def test_a_satellite_whose_commit_holds_the_servers_pulls_nothing(self, follower, parts):
        """Each pull would bring commits the server does not run yet and relaunch the satellite
        further from it — for as long as the server stays where it is."""
        parts["updater"].commits_between.side_effect = lambda old, new: 4 if (old, new) == (NEWER, BOOT) else 0
        for _ in range(3):
            follower.notice("release", NEWER)
            parts["spawn"].run()
            parts["clock"].now += RETRY_SECONDS + 1
        parts["updater"].update.assert_not_called()
        parts["relaunch"].assert_not_called()
        assert follower.status().update_failure == "" and not follower.status().updating

    def test_a_satellite_ahead_on_the_other_line_still_moves_to_the_servers(self, parts):
        """The server went from master to release, the last one cut from a commit this satellite
        is past: waiting for the server to catch up would keep it on the unsigned line until
        a release passes its commit."""
        parts["updater"].commits_between.side_effect = lambda old, new: 4 if (old, new) == (NEWER, BOOT) else 0
        follower = ServerFollower("master", BOOT, **parts)
        follower.notice("release", NEWER)
        parts["spawn"].run()
        parts["updater"].update.assert_called_once_with("release", BOOT)
        parts["relaunch"].assert_called_once()
        assert follower.status().branch == "release"

    def test_two_lines_that_went_apart_are_not_taken_for_a_satellite_ahead(self, follower, parts):
        """Each side has commits the other lacks: the server's line is still the one to be on."""
        parts["updater"].commits_between.return_value = 2
        follower.notice("master", NEWER)
        parts["spawn"].run()
        parts["updater"].update.assert_called_once_with("master", BOOT)

    def test_the_servers_branch_at_the_commit_already_run_is_followed_without_a_restart_or_a_failure(self, follower, parts):
        """Right after a release: the satellite on `release`, its server on `master`, both at the released commit."""
        parts["updater"].update.side_effect = lambda target, boot: Update(target, pulled="Already up to date.", head=BOOT)
        follower.notice("master", BOOT)
        parts["spawn"].run()
        status = follower.status()
        assert status.update_failure == "" and not status.updating
        assert (status.branch, status.commit) == ("master", BOOT)
        parts["relaunch"].assert_not_called()
        parts["clock"].now += RETRY_SECONDS + 1
        follower.notice("master", BOOT)
        assert parts["spawn"].waiting == [], "the server's branch and commit are the satellite's own now"

    def test_a_pull_that_switched_the_branch_and_brought_nothing_still_tells_the_branch_it_is_on(self, follower, parts):
        parts["updater"].update.side_effect = lambda target, boot: Update(target, pulled="Already up to date.", head=BOOT)
        follower.notice("master", NEWER)
        parts["spawn"].run()
        status = follower.status()
        assert status.update_failure and status.branch == "master"


@pytest.mark.unit
class TestAFailedUpdate:
    OUTCOMES = [
        (lambda target, boot: Update(target, pull_failure="Git pull failed: could not resolve host github.com"), "resolve host"),
        (lambda target, boot: Update(target, sync_failure="pip could not install torch==9.9", head=NEWER), "torch==9.9"),
        (lambda target, boot: Update(target, pulled="Updating"), "git"),
        (Mock(side_effect=TimeoutError("git fetch timed out after 60 seconds")), "TimeoutError"),
    ]

    @pytest.mark.parametrize("update, said", OUTCOMES, ids=["the pull", "the requirements", "no HEAD", "an exception"])
    def test_it_leaves_the_process_serving_and_is_told_in_the_status(self, follower, parts, update, said):
        parts["updater"].update.side_effect = update
        follower.notice("release", NEWER)
        parts["spawn"].run()
        status = follower.status()
        parts["relaunch"].assert_not_called()
        assert said in status.update_failure and not status.updating
        assert (status.branch, status.commit, status.wire) == ("release", BOOT, Wire.WIRE_VERSION)

    def test_a_start_script_that_cannot_be_spawned_is_a_failure_too(self, follower, parts):
        parts["relaunch"].side_effect = OSError("bash: not found")
        follower.notice("release", NEWER)
        parts["spawn"].run()
        assert "bash: not found" in follower.status().update_failure

    def test_the_failure_is_forgotten_when_the_server_names_another_target(self, follower, parts):
        parts["updater"].update.side_effect = lambda target, boot: Update(target, pull_failure="Git pull failed")
        follower.notice("release", NEWER)
        parts["spawn"].run()
        assert follower.status().update_failure
        follower.notice("release", BOOT)        # the server went back to what this process runs
        assert follower.status().update_failure == ""
        parts["clock"].now += MIN_INTERVAL_SECONDS
        follower.notice("release", NEWER)
        follower.notice("trial/pr-212", NEWEST)   # … and on to a branch nobody follows
        parts["spawn"].run()
        assert follower.status().update_failure, "the try that was under way did fail"
        follower.notice("trial/pr-212", NEWEST)
        assert parts["updater"].update.call_count == 2 and follower.status().update_failure == ""


@pytest.mark.unit
class TestOverTheRealUpdater:
    """The follower's own defaults: `Updater.update` over a fake checkout, and the start script as it is spawned."""

    @pytest.fixture(autouse=True)
    def no_real_release_or_pip(self):
        with patch("common.SignedRelease.SignedRelease.update", return_value=("", "Updating to v2026.11.9")) as release, \
                patch("common.RequirementsChecker.RequirementsChecker.sync", return_value=("", "")):
            yield release

    def test_the_relaunch_is_start_sh_with_this_pid_and_nothing_else(self, no_real_release_or_pip):
        """A contract with start.sh: `bash start.sh <pid of the process to replace>` from the repository root —
        the role is in config.yaml, not on the command line."""
        from common.Paths import AISERVER_DIR
        spawn = Later()
        follower = ServerFollower("release", BOOT, Updater(run=FakeCheckout(branch="release", head=NEWER)), spawn=spawn)
        with patch("common.Updater.subprocess.Popen") as popen:
            follower.notice("release", NEWER)
            spawn.run()
        no_real_release_or_pip.assert_called_once()
        assert popen.call_args.args[0] == ["bash", "start.sh", str(os.getpid())]
        assert popen.call_args.kwargs["cwd"] == str(AISERVER_DIR) and popen.call_args.kwargs["start_new_session"] is True

    def test_a_release_that_cannot_be_verified_starts_nothing(self, no_real_release_or_pip):
        no_real_release_or_pip.return_value = ("the newest release tag is not signed by a key in res/release_signers", "")
        spawn = Later()
        follower = ServerFollower("release", BOOT, Updater(run=FakeCheckout(branch="release", head=BOOT)), spawn=spawn)
        with patch("common.Updater.subprocess.Popen") as popen:
            follower.notice("release", NEWER)
            spawn.run()
        popen.assert_not_called()
        assert "release_signers" in follower.status().update_failure

    def test_left_to_itself_the_update_runs_on_a_thread_of_its_own(self):
        import threading
        ran_on = []
        done = threading.Event()
        updater = Mock(spec=Updater)
        updater.commits_between.return_value = 0
        updater.update.side_effect = lambda target, boot: (ran_on.append(threading.current_thread()), done.set(),
                                                           Update(target, pull_failure="x"))[2]
        ServerFollower("release", BOOT, updater, relaunch=Mock()).notice("release", NEWER)
        assert done.wait(2) and ran_on[0] is not threading.current_thread() and ran_on[0].daemon
