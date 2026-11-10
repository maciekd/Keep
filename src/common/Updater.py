"""Moves the checkout to the head of a branch, brings the venv to its pins and relaunches the process on the result.

What any process of the tree does to run newer code: the server (the nightly update and
`restart_server`, through `CodeQualityMaintainer`, which adds what is the server's alone — the
config keys, the nightly lock, the system notes, the resume marker) and a process without an
assistant, such as the iMessage satellite. The order is pull → requirements sync → relaunch;
a failure of the pull or of the sync leaves the old process running the code it has.

Nothing here reads config.yaml or knows its caller: the target branch and the start script are
arguments, and what an update came to is returned as an `Update` for the caller to act on and
to report in its own words."""
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional

from common.Paths import AISERVER_DIR, REPO_DIR
from common.RequirementsChecker import RequirementsChecker
from common.SignedRelease import SignedRelease, RELEASE_BRANCH
from common.logging import info, error, trace

GIT_TIMEOUT = 60
DETACHED_BRANCH = "master"   # what a checkout with no branch is taken to follow
UP_TO_DATE = "Already up to date"   # how git — and SignedRelease after it — says a pull moved nothing


def _run(cmd: list, label: str, timeout: int) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, cwd=str(REPO_DIR), timeout=timeout,
                            capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if result.stdout and result.stdout.strip():
        trace(f"Updater [{label}] stdout: {result.stdout.strip()}")
    if result.stderr and result.stderr.strip():
        trace(f"Updater [{label}] stderr: {result.stderr.strip()}")
    return result


@dataclass(frozen=True)
class Update:
    """What bringing the checkout to a branch came to, for the caller to act on and report.

    At most one of the failures is set, and either means the old process keeps running:
    *pull_failure* — the checkout is where it was; *sync_failure* — the code moved but the venv
    is behind its pins, so a new process would die on its requirements check."""
    target: str              # the branch asked for
    pull_failure: str = ""
    sync_failure: str = ""
    pulled: str = ""         # the pull's one-line summary, "Already up to date." when nothing moved
    head: str = ""           # HEAD after the pull, as `Updater.update` read it; "" when git could not say
    commits: int = 0         # how many commits HEAD has that the boot commit lacks; 0 when it is not ahead
    installed: str = ""      # what the requirements sync installed, "" when the venv already matched
    restart: bool = False    # the code and the venv are ready and the process is to be relaunched

    @property
    def failure(self) -> str:
        return self.pull_failure or self.sync_failure

    @property
    def up_to_date(self) -> bool:
        """The pull moved nothing."""
        return UP_TO_DATE in self.pulled

    @property
    def change(self) -> str:
        """What `Updater.update` brought since the boot commit, in words: the new commits — or
        where HEAD is now, after a switch to a branch that is not ahead — and what was installed."""
        text = f"{self.commits} new commit{'s' if self.commits != 1 else ''}" if self.commits else f"now at {self.head[:8]}"
        return f"{text}, {self.installed}" if self.installed else text


class Updater:
    """Moves the checkout to the head of a branch, brings the venv to its pins and relaunches the process on the result.

    How a git command is run is the argument *run* — `(cmd, label, timeout) -> CompletedProcess`,
    in the checkout by default. `CodeQualityMaintainer` passes its own runner, so its git calls
    stay behind the one seam its jobs already go through."""

    def __init__(self, run: Optional[Callable[[list, str, int], subprocess.CompletedProcess]] = None):
        self._run = run or _run

    def current_branch(self) -> str:
        """The branch this checkout follows — `release` on an install, `master` on the
        development machine. Whoever leaves the tree on another branch for a while (a Claude Code
        run on its work branch) restores this one afterwards; a hard-coded master would move an
        install onto the development branch."""
        try:
            result = self._run(["git", "rev-parse", "--abbrev-ref", "HEAD"], "git rev-parse", GIT_TIMEOUT)
            branch = (result.stdout or "").strip()
        except Exception as e:
            error(f"Updater: could not read the current branch: {e}")
            branch = ""
        if not branch or branch == "HEAD":
            info(f"Updater: no branch checked out ({branch or 'no output'}), assuming {DETACHED_BRANCH}")
            return DETACHED_BRANCH
        return branch

    def head_commit(self) -> str:
        """The commit the checkout is on, "" when git cannot say."""
        try:
            result = self._run(["git", "rev-parse", "HEAD"], "git rev-parse", GIT_TIMEOUT)
        except Exception as e:
            error(f"Updater: could not read HEAD: {e}")
            return ""
        if result.returncode != 0:
            error(f"Updater: git rev-parse HEAD exited {result.returncode}: {(result.stderr or '').strip()}")
            return ""
        return (result.stdout or "").strip()

    def commits_between(self, old: str, new: str) -> int:
        """How many commits *new* has that *old* lacks; 0 when git cannot say or *new* is not ahead
        (a switch back to an older release)."""
        try:
            result = self._run(["git", "rev-list", "--count", f"{old}..{new}"], "git rev-list", GIT_TIMEOUT)
            return int((result.stdout or "").strip()) if result.returncode == 0 else 0
        except Exception as e:
            error(f"Updater: could not count the commits {old[:8]}..{new[:8]}: {e}")
            return 0

    def pull(self, target: str) -> tuple[str, str]:
        """Check out *target* if the checkout is on another branch, then pull it — the release
        only as far as a tag signed by a key in res/release_signers (`SignedRelease`). Returns
        (failure, pull output): *failure* says what went wrong and is "" when the pull went
        through."""
        current = self.current_branch()
        if target == RELEASE_BRANCH:
            info(f"Updater: updating {target} to its newest signed release...")
            return SignedRelease.update(current)
        if current != target:
            info(f"Updater: on {current}, the target is {target} — switching before the pull")
            for cmd, label in ((["git", "fetch", "origin", target], "git fetch"),
                               (["git", "checkout", target], "git checkout")):
                result = self._run(cmd, label, GIT_TIMEOUT)
                if result.returncode != 0:
                    error(f"Updater: {label} {target} failed: {result.stderr}")
                    return (f"Could not switch to {target}: {(result.stderr or '').strip()} — "
                            f"still on {current}, the server was not restarted."), ""
        info(f"Updater: pulling latest code on {target}...")
        result = self._run(["git", "pull"], "git pull", GIT_TIMEOUT)
        if result.returncode != 0:
            error(f"Updater: git pull failed: {result.stderr}")
            return f"Git pull failed: {result.stderr}", ""
        return "", _pull_summary(result.stdout or "")

    def pull_current(self) -> None:
        """Bring the branch the checkout is on up to date, without switching: `release` only to
        its newest signed release, any other branch by a plain pull. A failure is logged and the
        caller goes on with the code it has."""
        branch = self.current_branch()
        if branch == RELEASE_BRANCH:
            failure, _ = SignedRelease.update(branch)
            if failure:
                error(f"Updater: {failure}")
            return
        self._run(["git", "pull"], "git pull", GIT_TIMEOUT)

    @staticmethod
    def sync() -> tuple[str, str]:
        """Between the pull and the relaunch: pip brings the venv to the pulled pins while the old
        process still serves, so the new one never dies on its requirements check. Returns
        (failure, installed) as RequirementsChecker.sync does; a failure means do not relaunch."""
        return RequirementsChecker.sync()

    def update(self, target: str, boot_commit: str) -> Update:
        """The unattended update: check out and pull *target*, and when the code is no longer
        *boot_commit* — the commit this process started on; commits pulled now or made locally
        since — bring the venv to its pins. `restart` is set only then and only when both went
        through; no new commit, no pip run and no restart. An unknown boot commit is no update
        at all: every commit would look new and the process would restart at each call."""
        if not boot_commit:
            info("Updater: the commit this process started on is unknown, no update.")
            return Update(target)
        info(f"Updater: bringing the checkout to the head of {target}...")
        failure, pulled = self.pull(target)
        if failure:
            error(f"Updater: the update to {target} failed, not restarting — {failure}")
            return Update(target, pull_failure=failure)
        head = self.head_commit()
        if not head:
            error("Updater: could not read HEAD after the pull, not restarting.")
            return Update(target, pulled=pulled)
        if head == boot_commit:
            info(f"Updater: no new commits on {target} since the start ({head[:8]}), not restarting.")
            return Update(target, pulled=pulled, head=head)
        commits = self.commits_between(boot_commit, head)
        failure, installed = self.sync()
        result = Update(target, sync_failure=failure, pulled=pulled, head=head, commits=commits,
                        installed=installed, restart=not failure)
        if failure:
            error(f"Updater: pulled {result.change} on {target} but not restarting — {failure}")
        else:
            info(f"Updater: {result.change} on {target} ({boot_commit[:8]} → {head[:8]}), ready to restart.")
        return result

    def refresh(self, target: str) -> Update:
        """The update of a restart that happens whatever the pull brings (somebody asked for
        one): check out and pull *target*, then bring the venv to its pins. `restart` is set
        unless the pull or the sync failed."""
        failure, pulled = self.pull(target)
        if failure:
            return Update(target, pull_failure=failure)
        if UP_TO_DATE in pulled:
            info("Updater: already up to date, restarting anyway...")
        else:
            info(f"Updater: pulled changes: {pulled}")
        failure, installed = self.sync()
        if failure:
            error(f"Updater: not restarting — {failure}")
        return Update(target, sync_failure=failure, pulled=pulled, installed=installed, restart=not failure)

    @staticmethod
    def relaunch(start_script: str, reason: str = "", *script_args: str) -> None:
        """Relaunch this process in place through *start_script* — a script of the repository
        root that takes the old process down and starts a fresh one — WITHOUT pulling new code.
        *script_args* follow the pid on its command line."""
        handed_on = f", script arguments: {' '.join(script_args)}" if script_args else ""
        info(f"Updater: spawning {start_script} as detached process"
             f"{f' ({reason})' if reason else ''} (our pid {os.getpid()}{handed_on})...")
        # Pass our own pid so the start script can kill exactly this process. pgrep -f
        # inside start.sh has proven unreliable when start.sh is spawned from
        # within the server itself — every historical auto-restart missed the
        # old process and the replacement died on a port collision.
        subprocess.Popen(
            ['bash', start_script, str(os.getpid()), *script_args],
            cwd=str(AISERVER_DIR),
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL
        )


def _pull_summary(stdout: str) -> str:
    """`git pull`'s output without the per-file diffstat: its first line ("Updating a..b",
    "Already up to date.") and the "N files changed" line — the text is logged, sent to the chat
    and carried through the restart as the resumed turn's query, so the diffstat was repeated
    a dozen times."""
    lines = [line.strip() for line in stdout.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    summary = next((line for line in lines[1:] if re.search(r"\d+ files? changed", line)), "")
    return f"{lines[0]}, {summary}" if summary else lines[0]
