"""Runs the server on one of its own open pull requests for a while, so a self-improvement PR can be tried before it is merged.

`start()` vets the PR (open, from a branch of this repository, by a trusted author, no requirements change,
clean tree), checks its head out as `trial/pr-<N>` and restarts. A watchdog — a copy of trial_watchdog.sh
taken before the checkout, so the PR cannot break it — waits for the marker the server writes once it is up
and otherwise checks the previous branch out again and restarts. While the trial lasts restart_server stays on
the PR instead of pulling the target branch; it ends when the user asks, when the PR is closed, when its merge
has reached the target branch, or after MAX_DAYS. The state lives in cache/trial/, not in config.yaml: the
watchdog has to read it from shell and a config_set must not get around the vetting.
"""
import json
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

from common.logging import info, error, trace
from common.Paths import AISERVER_DIR, CACHE_DIR
from assistant.agents.CodeQualityMaintainer import CodeQualityMaintainer, GIT_TIMEOUT, WORKING_DIR

TRIAL_DIR = CACHE_DIR / "trial"
STATE_PATH = TRIAL_DIR / "state.json"
STARTED_MARKER = TRIAL_DIR / "started"          # written by the server once it is up; the watchdog waits for it
ROLLED_BACK_PATH = TRIAL_DIR / "rolled_back"    # written by the watchdog: the server's last output
WATCHDOG_SOURCE = AISERVER_DIR / "trial_watchdog.sh"
WATCHDOG_COPY = TRIAL_DIR / "watchdog.sh"
WATCHDOG_TIMEOUT = 80   # seconds for start.sh to stop the old server and the new one to load its models
MAX_DAYS = 7
REMINDER_HOUR = 9
PENDING, ACTIVE, ENDED = "pending", "active", "ended"
ROLLBACK_OUTPUT_CHARS = 1200


@dataclass
class Trial:
    number: int
    title: str
    sha: str
    previous_branch: str
    started: float
    status: str = PENDING
    reminded: str = ""       # the date of the last daily reminder
    end_reason: str = ""

    @property
    def branch(self) -> str:
        return f"trial/pr-{self.number}"

    @property
    def day(self) -> int:
        return int((time.time() - self.started) // 86400) + 1


class TrialRefused(Exception):
    pass


class PrTrial:

    @staticmethod
    def state() -> Optional[Trial]:
        try:
            return Trial(**json.loads(STATE_PATH.read_text()))
        except FileNotFoundError:
            return None
        except (OSError, ValueError, TypeError) as e:
            error(f"PrTrial: unreadable state in {STATE_PATH} ({e}), treating it as no trial")
            return None

    @staticmethod
    def _save(trial: Trial) -> None:
        TRIAL_DIR.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(asdict(trial)))

    @staticmethod
    def _clear() -> None:
        for path in (STATE_PATH, STARTED_MARKER, ROLLED_BACK_PATH):
            path.unlink(missing_ok=True)

    @staticmethod
    def _git(*args: str) -> subprocess.CompletedProcess:
        return CodeQualityMaintainer._run_subprocess(["git", *args], f"git {args[0]}", GIT_TIMEOUT)

    @staticmethod
    def _pr(number: int) -> dict:
        result = subprocess.run(
            ["gh", "pr", "view", str(number), "--json",
             "number,title,state,author,isCrossRepository,headRefOid,mergeCommit"],
            cwd=WORKING_DIR, capture_output=True, text=True, timeout=GIT_TIMEOUT)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"gh pr view exited {result.returncode}")
        return json.loads(result.stdout)

    @staticmethod
    def _vet(number: int) -> dict:
        """The PR when this machine may run it, TrialRefused with the reason otherwise. Running a PR is
        running its code next to the user's mail and messages, hence the same trust as the auto-merge."""
        try:
            pr = PrTrial._pr(number)
            _, paths = CodeQualityMaintainer._pr_title_and_files(number)
        except Exception as e:
            raise TrialRefused(f"PR #{number} could not be read: {e}")
        author = ((pr.get("author") or {}).get("login") or "").lower()
        if pr.get("state") != "OPEN":
            raise TrialRefused(f"PR #{number} is {str(pr.get('state')).lower()}, only an open PR can be tried")
        if pr.get("isCrossRepository"):
            raise TrialRefused(f"PR #{number} comes from a fork")
        if author not in CodeQualityMaintainer._trusted_pr_authors():
            raise TrialRefused(f"PR #{number} is by {author or 'an unknown author'}, not one of this install's own PRs")
        requirements = [p for p in paths if p.rsplit("/", 1)[-1].startswith("requirements") and p.endswith(".txt")]
        if requirements:
            raise TrialRefused(f"PR #{number} changes {', '.join(requirements)}: a PR that needs other packages "
                               f"cannot be tried, it has to be merged and released")
        return pr

    @staticmethod
    def _arm_watchdog(previous_branch: str) -> None:
        STARTED_MARKER.unlink(missing_ok=True)
        ROLLED_BACK_PATH.unlink(missing_ok=True)
        info(f"PrTrial: watchdog armed, rolls back to {previous_branch} unless the server is up in {WATCHDOG_TIMEOUT}s")
        subprocess.Popen(["bash", str(WATCHDOG_COPY), previous_branch, str(WATCHDOG_TIMEOUT)], cwd=AISERVER_DIR,
                         start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         stdin=subprocess.DEVNULL)

    @staticmethod
    def start(number: int) -> str:
        current = PrTrial.state()
        if current and current.status != ENDED:
            info(f"PrTrial: try of PR #{number} refused, already trying PR #{current.number}")
            return f"Already trying PR #{current.number} — exit_pr() first."
        try:
            pr = PrTrial._vet(number)
            if PrTrial._git("status", "--porcelain", "--untracked-files=no").stdout.strip():
                raise TrialRefused("the checkout has uncommitted changes")
            previous = CodeQualityMaintainer._current_branch()
            trial = Trial(number=number, title=pr.get("title", ""), sha=pr["headRefOid"],
                          previous_branch=previous, started=time.time(),
                          reminded=datetime.now().date().isoformat())
            fetch = PrTrial._git("fetch", "origin", f"+pull/{number}/head:{trial.branch}")
            if fetch.returncode != 0:
                raise TrialRefused(f"git fetch failed: {(fetch.stderr or '').strip()}")
            fetched = PrTrial._git("rev-parse", trial.branch).stdout.strip()
            if fetched != trial.sha:
                raise TrialRefused(f"PR #{number} changed while it was being checked ({fetched[:8]} is not the "
                                   f"vetted {trial.sha[:8]}), try again")
            # Before the checkout: the watchdog that guards this trial is this branch's, not the PR's.
            TRIAL_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(WATCHDOG_SOURCE, WATCHDOG_COPY)
            checkout = PrTrial._git("checkout", trial.branch)
            if checkout.returncode != 0:
                raise TrialRefused(f"git checkout failed: {(checkout.stderr or '').strip()}")
        except TrialRefused as e:
            info(f"PrTrial: try of PR #{number} refused — {e}")
            return f"Cannot try PR #{number}: {e}."
        PrTrial._save(trial)
        PrTrial._arm_watchdog(previous)
        info(f"PrTrial: on {trial.branch} at {trial.sha[:8]} (was {previous}), restarting")
        CodeQualityMaintainer.restart(f"trying PR #{number}")
        return f"Switched to PR #{number} ({trial.sha[:8]}). Server is restarting..."

    @staticmethod
    def end(reason: str = "the user asked to go back") -> str:
        trial = PrTrial.state()
        if trial is None:
            info("PrTrial: end asked for, but no PR is being tried")
            return "No PR is being tried."
        PrTrial._mark_ended(trial, reason)
        return CodeQualityMaintainer.pull_and_restart()

    @staticmethod
    def _mark_ended(trial: Trial, reason: str) -> None:
        info(f"PrTrial: trial of PR #{trial.number} ends — {reason}")
        trial.status, trial.end_reason = ENDED, reason
        PrTrial._save(trial)

    @staticmethod
    def over_reason(trial: Trial) -> Optional[str]:
        """Why the trial is over, None while it goes on. A merged PR keeps running until its merge
        commit is on the target branch, so the fix never disappears between the merge and the release."""
        if time.time() - trial.started > MAX_DAYS * 86400:
            return f"its {MAX_DAYS} days are up"
        try:
            pr = PrTrial._pr(trial.number)
        except Exception as e:
            error(f"PrTrial: PR #{trial.number} could not be read ({e}), the trial goes on")
            return None
        state = pr.get("state")
        if state == "CLOSED":
            return "the PR was closed without being merged"
        if state == "MERGED":
            target = CodeQualityMaintainer.target_branch()
            merge = (pr.get("mergeCommit") or {}).get("oid")
            PrTrial._git("fetch", "origin", target)
            if merge and PrTrial._git("merge-base", "--is-ancestor", merge, f"origin/{target}").returncode == 0:
                return f"the PR is merged and part of {target}"
            info(f"PrTrial: PR #{trial.number} is merged but {target} does not have it yet, the trial goes on")
            return None
        trace(f"PrTrial: PR #{trial.number} is {state}, day {trial.day} of {MAX_DAYS}")
        return None

    @staticmethod
    def on_restart() -> Optional[str]:
        """restart_server while a PR is tried: the restart's result when the server stays on the PR
        (its head re-fetched), None when there is no trial or it is over and the normal pull applies."""
        trial = PrTrial.state()
        if trial is None or trial.status == ENDED:
            return None
        reason = PrTrial.over_reason(trial)
        if reason:
            PrTrial._mark_ended(trial, reason)
            return None
        update = PrTrial._update_to_head(trial)
        CodeQualityMaintainer.restart(f"still trying PR #{trial.number}")
        return (f"Still trying PR #{trial.number} ({update}); the target branch is not pulled meanwhile. "
                f"Server is restarting...")

    @staticmethod
    def _update_to_head(trial: Trial) -> str:
        fetch = PrTrial._git("fetch", "origin", f"pull/{trial.number}/head")
        head = PrTrial._git("rev-parse", "FETCH_HEAD").stdout.strip() if fetch.returncode == 0 else ""
        if not head or head == trial.sha:
            info(f"PrTrial: PR #{trial.number} has no new commits ({trial.sha[:8]})")
            return "no new commits"
        try:
            PrTrial._vet(trial.number)
        except TrialRefused as e:
            info(f"PrTrial: PR #{trial.number} moved to {head[:8]} but is not taken — {e}")
            return f"its new commits are not taken: {e}"
        if PrTrial._git("checkout", "-B", trial.branch, head).returncode != 0:
            error(f"PrTrial: could not move {trial.branch} to {head[:8]}, staying on {trial.sha[:8]}")
            return "its new commits could not be checked out"
        info(f"PrTrial: PR #{trial.number} updated {trial.sha[:8]} -> {head[:8]}")
        trial.sha, trial.status = head, PENDING
        PrTrial._save(trial)
        PrTrial._arm_watchdog(trial.previous_branch)
        return f"updated to {head[:8]}"

    @staticmethod
    def confirm_started() -> Optional[str]:
        """Called once the server is up: stands the watchdog down and returns what the user should
        be told about the trial — it began, it was rolled back, it ended — or None."""
        trial = PrTrial.state()
        if trial is None:
            trace("PrTrial: no PR is being tried")
            return None
        branch = CodeQualityMaintainer._current_branch()
        if ROLLED_BACK_PATH.exists():
            output = ROLLED_BACK_PATH.read_text(errors="replace").strip()[-ROLLBACK_OUTPUT_CHARS:]
            PrTrial._clear()
            info(f"PrTrial: PR #{trial.number} was rolled back by the watchdog, now on {branch}")
            return (f"PR #{trial.number} did not start within {WATCHDOG_TIMEOUT} seconds, so the server went back "
                    f"to {branch}. The trial is over. The server's last output:\n{output or '(none)'}")
        if trial.status == ENDED:
            PrTrial._clear()
            return f"The trial of PR #{trial.number} is over ({trial.end_reason}). The server now runs {branch}."
        if branch != trial.branch:
            PrTrial._clear()
            info(f"PrTrial: state says PR #{trial.number} but the checkout is on {branch}, state dropped")
            return None
        TRIAL_DIR.mkdir(parents=True, exist_ok=True)
        STARTED_MARKER.write_text(str(time.time()))
        if trial.status == ACTIVE:
            info(f"PrTrial: still trying PR #{trial.number}, day {trial.day} of {MAX_DAYS}")
            return None
        trial.status = ACTIVE
        PrTrial._save(trial)
        info(f"PrTrial: PR #{trial.number} is up at {trial.sha[:8]}")
        return (f"The server now runs PR #{trial.number} ({trial.title}) at {trial.sha[:8]} as a trial. No updates "
                f"are fetched while it lasts; it ends after {MAX_DAYS} days, when the PR is closed, or once it "
                f"is merged and released. exit_pr() goes back to {trial.previous_branch} — the user can also "
                f"send //exit_pr.")

    @staticmethod
    def daily_check(now: datetime) -> None:
        """Once a day from REMINDER_HOUR: ends a trial that is over (a restart), otherwise reminds
        the user that the server runs a PR and gets no updates."""
        trial = PrTrial.state()
        today = now.date().isoformat()
        if trial is None or trial.status != ACTIVE or now.hour < REMINDER_HOUR or trial.reminded == today:
            return
        from assistant.SystemNote import system_note
        reason = PrTrial.over_reason(trial)
        if reason:
            PrTrial._mark_ended(trial, reason)
            info(f"PrTrial: daily check restarts the server: {CodeQualityMaintainer.pull_and_restart()}")
            return
        trial.reminded = today
        PrTrial._save(trial)
        info(f"PrTrial: daily reminder, PR #{trial.number} day {trial.day} of {MAX_DAYS}")
        system_note("server", f"Reminder: the server is testing PR #{trial.number} ({trial.title}), day {trial.day} "
                              f"of {MAX_DAYS}. No new updates are fetched while the trial lasts — "
                              f"say \"go back\" to end it.", quiet=True)
