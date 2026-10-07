"""PrTrial: running the server on one of its own PRs, and getting back off it."""
import time
from datetime import datetime
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from assistant.agents import PrTrial as module
from assistant.agents.CodeQualityMaintainer import CodeQualityMaintainer
from assistant.agents.PrTrial import PrTrial, Trial, ACTIVE, ENDED, PENDING, MAX_DAYS

HEAD = "a" * 40
NEW_HEAD = "b" * 40
MERGE = "c" * 40


def open_pr(**overrides) -> dict:
    return {"number": 42, "title": "Fix the hallway light", "state": "OPEN", "author": {"login": "marcus"},
            "isCrossRepository": False, "headRefOid": HEAD, "mergeCommit": None, **overrides}


class FakeGit:
    """Answers PrTrial._git by subcommand; everything succeeds unless told otherwise."""

    def __init__(self):
        self.calls = []
        self.stdout = {"status": "", "rev-parse": HEAD}
        self.failing = set()

    def __call__(self, *args):
        self.calls.append(args)
        return CompletedProcess(args, 1 if args[0] in self.failing else 0, self.stdout.get(args[0], ""), "boom")

    def ran(self, *prefix) -> bool:
        return any(call[:len(prefix)] == prefix for call in self.calls)


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A trial directory of its own, a scripted git and gh, and a restart that only records."""
    trial_dir = tmp_path / "trial"
    for name, path in (("TRIAL_DIR", trial_dir), ("STATE_PATH", trial_dir / "state.json"),
                       ("STARTED_MARKER", trial_dir / "started"), ("ROLLED_BACK_PATH", trial_dir / "rolled_back"),
                       ("WATCHDOG_COPY", trial_dir / "watchdog.sh"), ("WATCHDOG_SOURCE", tmp_path / "source.sh")):
        monkeypatch.setattr(module, name, path)
    (tmp_path / "source.sh").write_text("# watchdog")

    class World:
        git = FakeGit()
        pr = open_pr()
        files = ["src/assistant/Lights.py"]
        branch = "release"
        restarts = []
        watchdogs = []

    w = World()
    monkeypatch.setattr(PrTrial, "_git", staticmethod(w.git))
    monkeypatch.setattr(PrTrial, "_pr", staticmethod(lambda number: w.pr))
    monkeypatch.setattr(CodeQualityMaintainer, "_pr_title_and_files", staticmethod(lambda number: ("t", w.files)))
    monkeypatch.setattr(CodeQualityMaintainer, "_trusted_pr_authors", staticmethod(lambda: {"marcus"}))
    monkeypatch.setattr(CodeQualityMaintainer, "_current_branch", staticmethod(lambda: w.branch))
    monkeypatch.setattr(CodeQualityMaintainer, "target_branch", staticmethod(lambda: "release"))
    monkeypatch.setattr(CodeQualityMaintainer, "restart", staticmethod(lambda reason="": w.restarts.append(reason)))
    monkeypatch.setattr(module.subprocess, "Popen", lambda cmd, **kwargs: w.watchdogs.append(cmd))
    return w


def running_trial(status=ACTIVE, age_days=0.0, **overrides) -> Trial:
    trial = Trial(number=42, title="Fix the hallway light", sha=HEAD, previous_branch="release",
                  started=time.time() - age_days * 86400, status=status, **overrides)
    PrTrial._save(trial)
    return trial


class TestStart:
    def test_checks_the_pr_out_arms_the_watchdog_and_restarts(self, world):
        PrTrial.start(42)
        trial = PrTrial.state()
        assert (trial.status, trial.sha, trial.previous_branch) == (PENDING, HEAD, "release")
        assert world.git.ran("checkout", "trial/pr-42")
        assert len(world.restarts) == 1
        # the watchdog is the copy taken before the checkout, told where the way back is
        assert world.watchdogs[0][1:3] == [str(module.WATCHDOG_COPY), "release"]
        assert module.WATCHDOG_COPY.read_text() == "# watchdog"

    @pytest.mark.parametrize("change", [
        lambda w: setattr(w, "pr", open_pr(isCrossRepository=True)),
        lambda w: setattr(w, "pr", open_pr(author={"login": "dave"})),
        lambda w: setattr(w, "pr", open_pr(state="MERGED")),
        lambda w: setattr(w, "files", ["src/x.py", "requirements-voice.txt"]),
        lambda w: w.git.stdout.update(status=" M src/x.py"),
        lambda w: w.git.stdout.update({"rev-parse": NEW_HEAD}),
        lambda w: w.git.failing.add("fetch"),
        lambda w: w.git.failing.add("checkout"),
    ], ids=["fork", "foreign-author", "not-open", "requirements", "dirty-tree", "head-moved", "fetch-fails",
            "checkout-fails"])
    def test_a_refused_pr_leaves_the_server_as_it_was(self, world, change):
        change(world)
        reply = PrTrial.start(42)
        assert "42" in reply
        assert PrTrial.state() is None
        assert world.restarts == [] and world.watchdogs == []

    def test_a_second_pr_is_refused_while_one_is_tried(self, world):
        running_trial()
        PrTrial.start(43)
        assert PrTrial.state().number == 42
        assert world.restarts == []


class TestOverReason:
    def test_an_open_pr_goes_on(self, world):
        assert PrTrial.over_reason(running_trial(age_days=MAX_DAYS - 1)) is None

    def test_the_trial_is_over_after_max_days_whatever_the_pr_says(self, world):
        assert PrTrial.over_reason(running_trial(age_days=MAX_DAYS + 0.1)) is not None

    def test_a_closed_pr_ends_it(self, world):
        world.pr = open_pr(state="CLOSED")
        assert PrTrial.over_reason(running_trial()) is not None

    def test_a_merged_pr_goes_on_until_the_target_branch_has_its_merge(self, world):
        world.pr = open_pr(state="MERGED", mergeCommit={"oid": MERGE})
        world.git.failing.add("merge-base")
        assert PrTrial.over_reason(running_trial()) is None
        world.git.failing.clear()
        assert PrTrial.over_reason(running_trial()) is not None
        assert world.git.ran("merge-base", "--is-ancestor", MERGE, "origin/release")

    def test_an_unreadable_pr_does_not_end_it(self, world, monkeypatch):
        monkeypatch.setattr(PrTrial, "_pr", staticmethod(lambda number: (_ for _ in ()).throw(RuntimeError("gh down"))))
        assert PrTrial.over_reason(running_trial()) is None


class TestRestartDuringATrial:
    def test_without_a_trial_the_normal_pull_applies(self, world):
        assert PrTrial.on_restart() is None

    def test_stays_on_the_pr_without_rearming_when_it_has_no_new_commits(self, world):
        running_trial()
        world.branch = "trial/pr-42"
        assert PrTrial.on_restart()
        assert len(world.restarts) == 1 and world.watchdogs == []
        assert PrTrial.state().status == ACTIVE

    def test_new_commits_are_taken_under_a_fresh_watchdog(self, world):
        running_trial()
        world.git.stdout["rev-parse"] = NEW_HEAD
        PrTrial.on_restart()
        trial = PrTrial.state()
        assert (trial.sha, trial.status) == (NEW_HEAD, PENDING)
        assert world.git.ran("checkout", "-B", "trial/pr-42", NEW_HEAD)
        assert world.watchdogs[0][2] == "release"

    def test_new_commits_that_touch_requirements_are_not_taken(self, world):
        running_trial()
        world.git.stdout["rev-parse"] = NEW_HEAD
        world.files = ["requirements.txt"]
        PrTrial.on_restart()
        assert PrTrial.state().sha == HEAD
        assert not world.git.ran("checkout")

    def test_a_trial_that_is_over_hands_the_restart_back(self, world):
        running_trial(age_days=MAX_DAYS + 1)
        assert PrTrial.on_restart() is None
        assert PrTrial.state().status == ENDED
        assert world.restarts == []

    def test_pull_and_restart_does_not_pull_the_target_branch_during_a_trial(self, world):
        running_trial()
        with patch.object(CodeQualityMaintainer, "_run_subprocess") as run:
            CodeQualityMaintainer.pull_and_restart()
        run.assert_not_called()
        assert len(world.restarts) == 1

    def test_end_marks_the_trial_over_and_restarts_through_the_normal_pull(self, world):
        running_trial()
        with patch.object(CodeQualityMaintainer, "pull_and_restart", return_value="restarting") as pull:
            assert PrTrial.end() == "restarting"
        pull.assert_called_once_with()
        assert PrTrial.state().status == ENDED


class TestConfirmStarted:
    def test_a_pending_trial_becomes_active_and_stands_the_watchdog_down(self, world):
        running_trial(status=PENDING)
        world.branch = "trial/pr-42"
        assert "42" in PrTrial.confirm_started()
        assert PrTrial.state().status == ACTIVE
        assert module.STARTED_MARKER.exists()

    def test_a_later_start_on_the_trial_says_nothing_again(self, world):
        running_trial()
        world.branch = "trial/pr-42"
        assert PrTrial.confirm_started() is None
        assert module.STARTED_MARKER.exists()

    def test_a_rollback_is_reported_with_the_servers_last_output_and_forgotten(self, world):
        running_trial(status=PENDING)
        module.ROLLED_BACK_PATH.write_text("ImportError: no module named lights")
        message = PrTrial.confirm_started()
        assert "ImportError: no module named lights" in message and "release" in message
        assert PrTrial.state() is None

    def test_an_ended_trial_is_reported_once(self, world):
        running_trial(status=ENDED, end_reason="the PR was closed")
        assert "the PR was closed" in PrTrial.confirm_started()
        assert PrTrial.state() is None
        assert PrTrial.confirm_started() is None

    def test_state_for_a_branch_the_checkout_is_not_on_is_dropped(self, world):
        running_trial()
        world.branch = "master"
        assert PrTrial.confirm_started() is None
        assert PrTrial.state() is None


class TestDailyCheck:
    MORNING = datetime(2026, 3, 4, 9, 5)

    def test_reminds_once_a_day_that_no_updates_are_fetched(self, world):
        running_trial(age_days=2)
        with patch("assistant.SystemNote.system_note") as note:
            PrTrial.daily_check(self.MORNING)
            PrTrial.daily_check(self.MORNING.replace(hour=15))
            assert note.call_count == 1
            PrTrial.daily_check(self.MORNING.replace(day=5))
            assert note.call_count == 2
        assert "42" in note.call_args.args[1]
        assert note.call_args.kwargs["quiet"] is True   # a daily reminder must not ring the phone

    def test_keeps_quiet_before_the_reminder_hour_and_without_a_trial(self, world):
        with patch("assistant.SystemNote.system_note") as note:
            PrTrial.daily_check(self.MORNING)
            running_trial()
            PrTrial.daily_check(self.MORNING.replace(hour=3))
        note.assert_not_called()

    def test_a_trial_that_is_over_is_ended_with_a_restart_instead_of_a_reminder(self, world):
        running_trial(age_days=MAX_DAYS + 1)
        with patch("assistant.SystemNote.system_note") as note, \
                patch.object(CodeQualityMaintainer, "pull_and_restart", return_value="restarting") as pull:
            PrTrial.daily_check(self.MORNING)
        pull.assert_called_once_with()
        note.assert_not_called()
        assert PrTrial.state().status == ENDED
