"""Runs the server on one of its own open pull requests for a while, so a PR a user had built can be tried before it is merged.

The branch is the state: `try_pull_request(N)` fetches the PR's head as `trial/pr-N`, checks it out and
restarts, and the server is on a trial exactly while its checkout follows a `trial/` branch. Nothing else
is remembered. The way back is the normal `restart_server`, which pulls the branch `developer_release`
names, and the nightly update at 03:00, which does the same by itself — so a trial lasts a day at most.
While it lasts the Claude Code PR jobs (`improve_integration`, `self_improvement`) push to the tried PR
instead of opening a new one, `pr_feedback` posts the user's verdict on it, and the prompt carries one
line saying what is running. A PR whose server does not come up is caught by start.sh: the restart hands
it the branch to fall back to, and it checks that branch out again and relaunches (`take_rollback_note`
tells the user afterwards).
"""
import html
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Optional

from common import Version
from common.Paths import LOG_DIR
from common.logging import info, error, trace
from assistant.agents.CodeQualityMaintainer import CodeQualityMaintainer, GIT_TIMEOUT, WORKING_DIR

BRANCH_PREFIX = "trial/pr-"
ROLLBACK_NOTE_PATH = LOG_DIR / "trial_rollback.txt"   # written by start.sh when a trial's server never came up
ROLLBACK_NOTE_CHARS = 1500


@dataclass(frozen=True)
class Trial:
    number: int
    title: str
    head_ref: str        # the PR's own branch on origin, where a follow-up job pushes
    since: str           # when the trial began, HH:MM today or YYYY-MM-DD HH:MM
    return_branch: str   # what restart_server and the nightly update go back to

    @property
    def branch(self) -> str:
        return f"{BRANCH_PREFIX}{self.number}"


class TrialRefused(Exception):
    """Why a PR cannot be tried on this machine, in words for the user."""


def trial_number(branch: Optional[str]) -> Optional[int]:
    """The PR number a trial branch names, None for any other branch."""
    if branch and branch.startswith(BRANCH_PREFIX) and branch[len(BRANCH_PREFIX):].isdigit():
        return int(branch[len(BRANCH_PREFIX):])
    return None


class PullRequestTrial:

    @staticmethod
    def _git(*args: str) -> subprocess.CompletedProcess:
        return CodeQualityMaintainer._run_subprocess(["git", *args], f"git {args[0]}", GIT_TIMEOUT)

    @staticmethod
    @lru_cache(maxsize=8)
    def _pr(number: int) -> dict:
        """What GitHub knows of the PR; cached, since the title and the branch name do not move
        while this process runs and the prompt section asks at every build."""
        result = subprocess.run(
            ["gh", "pr", "view", str(number), "--json", "number,title,state,author,isCrossRepository,headRefName"],
            cwd=WORKING_DIR, capture_output=True, text=True, timeout=GIT_TIMEOUT, stdin=subprocess.DEVNULL)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or "").strip() or f"gh pr view exited {result.returncode}")
        return json.loads(result.stdout)

    @staticmethod
    def current() -> Optional[Trial]:
        """The trial this process runs, None on a normal branch. The branch is the one the
        process booted on (Version caches it), so a PR job's own checkouts never flicker it."""
        number = trial_number(Version.branch())
        if number is None:
            return None
        try:
            pr = PullRequestTrial._pr(number)
            title, head_ref = pr.get("title", ""), pr.get("headRefName", "")
        except Exception as e:
            error(f"PullRequestTrial: PR #{number} could not be read ({e}), the trial is known by its number alone")
            title, head_ref = "", ""
        return Trial(number=number, title=title, head_ref=head_ref, since=PullRequestTrial._since(number),
                     return_branch=CodeQualityMaintainer.target_branch())

    @staticmethod
    def _since(number: int) -> str:
        """When the trial branch was last fetched — the moment the trial began."""
        result = PullRequestTrial._git("reflog", "-1", "--format=%ci", f"{BRANCH_PREFIX}{number}")
        stamp = (result.stdout or "").strip()
        try:
            moment = datetime.strptime(stamp[:19], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            trace(f"PullRequestTrial: no reflog time for {BRANCH_PREFIX}{number} ({stamp!r})")
            return "an unknown time"
        return moment.strftime("%H:%M") if moment.date() == datetime.now().date() else moment.strftime("%Y-%m-%d %H:%M")

    @staticmethod
    def vet(number: int) -> dict:
        """The PR when this machine may run it, TrialRefused otherwise. Running a PR is running
        its code next to the user's mail and messages, so the trust is the auto-merge's: a branch
        of this repository, by one of the install's own authors, and still open."""
        try:
            pr = PullRequestTrial._pr.__wrapped__(number)
        except Exception as e:
            raise TrialRefused(f"PR #{number} could not be read: {e}")
        if pr.get("state") != "OPEN":
            raise TrialRefused(f"PR #{number} is {str(pr.get('state', '')).lower() or 'not open'}; only an open PR can be tried")
        if pr.get("isCrossRepository"):
            raise TrialRefused(f"PR #{number} comes from a fork, not from this repository")
        author = ((pr.get("author") or {}).get("login") or "").lower()
        trusted = CodeQualityMaintainer._trusted_pr_authors()
        if author not in trusted:
            raise TrialRefused(f"PR #{number} is by {author or 'an unknown author'}, not one of this install's own PRs")
        if PullRequestTrial._git("status", "--porcelain", "--untracked-files=no").stdout.strip():
            raise TrialRefused("the checkout has uncommitted changes, so no other code can be checked out")
        trace(f"PullRequestTrial: PR #{number} ({pr.get('title')}) by {author} may be tried")
        return pr

    @staticmethod
    def start(number: int) -> str:
        """Fetch the PR's head, check it out, bring the venv to its pins and restart on it. The
        result says what happened; the restart itself is reported by the new process. Any
        failure after the checkout puts the previous branch back."""
        try:
            pr = PullRequestTrial.vet(number)
        except TrialRefused as e:
            info(f"PullRequestTrial: PR #{number} refused — {e}")
            return f"Cannot try PR #{number}: {e}."
        branch = f"{BRANCH_PREFIX}{number}"
        fallback = CodeQualityMaintainer.target_branch()
        previous = CodeQualityMaintainer._current_branch() or fallback
        # Into FETCH_HEAD, not the branch: a second try of the PR already being tried would
        # otherwise ask git to update the checked-out branch, which it refuses.
        fetch = PullRequestTrial._git("fetch", "origin", f"pull/{number}/head")
        if fetch.returncode != 0:
            error(f"PullRequestTrial: fetching PR #{number} failed: {fetch.stderr}")
            return f"Cannot try PR #{number}: git fetch failed: {(fetch.stderr or '').strip()}."
        checkout = PullRequestTrial._git("checkout", "-B", branch, "FETCH_HEAD")
        if checkout.returncode != 0:
            error(f"PullRequestTrial: checking out {branch} failed: {checkout.stderr}")
            return f"Cannot try PR #{number}: git checkout failed: {(checkout.stderr or '').strip()}."
        failure, installed = CodeQualityMaintainer._sync_requirements()
        if failure:
            error(f"PullRequestTrial: PR #{number} needs packages that did not install, back on {previous} — {failure}")
            PullRequestTrial._git("checkout", previous)
            return f"PR #{number} was not tried, the server keeps running {previous}: {failure}"
        head = (PullRequestTrial._git("rev-parse", "--short=8", "HEAD").stdout or "").strip()
        info(f"PullRequestTrial: on {branch} at {head} (was {previous}), restarting with {fallback} as the fallback")
        CodeQualityMaintainer.restart(f"trying PR #{number}", fallback_branch=fallback)
        detail = f" ({installed})" if installed else ""
        return (f"Switched to PR #{number} ({pr.get('title', '')}) at {head}{detail}. Server is restarting on it; "
                f"restart_server goes back to {fallback}, as the nightly update does by itself.")

    @staticmethod
    def feedback(number: int, text: str) -> str:
        """Post the user's verdict on the PR as a comment, from the install's own gh login."""
        text = text.strip()
        if not text:
            return "pr_feedback failed: nothing to post"
        trial = PullRequestTrial.current()
        tested = f"on this install's trial of it, Keep {Version.detail()}" if trial and trial.number == number \
            else f"by the user of a Keep install ({Version.detail()})"
        body = f"**Tested {tested}:**\n\n{text}"
        result = CodeQualityMaintainer._run_subprocess(["gh", "pr", "comment", str(number), "--body", body],
                                                       f"gh pr comment {number}", GIT_TIMEOUT)
        if result.returncode != 0:
            error(f"PullRequestTrial: commenting on PR #{number} failed: {result.stderr}")
            return f"pr_feedback failed: {(result.stderr or '').strip() or 'gh pr comment exited ' + str(result.returncode)}"
        url = (result.stdout or "").strip()
        info(f"PullRequestTrial: feedback posted on PR #{number}: {url}")
        return f"Posted on PR #{number}: {url}" if url else f"Posted on PR #{number}."

    @staticmethod
    def prompt_line() -> str:
        """The line the prompt carries while a trial runs, "" otherwise."""
        trial = PullRequestTrial.current()
        if trial is None:
            return ""
        title = f" ({trial.title})" if trial.title else ""
        return (f"Running PR #{trial.number}{title} as a trial since {trial.since}, normally on {trial.return_branch}. "
                f"restart_server() ends the trial and returns to {trial.return_branch}; the nightly update at 03:00 does "
                f"the same by itself. try_pull_request({trial.number}) again picks up new commits pushed to the PR. "
                f"A PR job (improve_integration, self_improvement) pushes to this PR instead of opening a new one, "
                f"and pr_feedback({trial.number}, text) posts the user's verdict on it.")

    @staticmethod
    def announcement_note() -> str:
        """What the startup announcement adds while a trial runs, in the Notifier's HTML dialect
        — the user asked for the trial and should see at a glance that the server is on it; ""
        on a normal branch."""
        trial = PullRequestTrial.current()
        if trial is None:
            return ""
        title = f", <i>{html.escape(trial.title, quote=False)}</i>" if trial.title else ""
        info(f"PullRequestTrial: the startup announcement says PR #{trial.number} is on trial")
        return (f"You are trying PR #{trial.number}{title}. Tell me what you think and I post the verdict on the PR; "
                f"a restart ends the trial and returns to {trial.return_branch}.")

    @staticmethod
    def pr_job_instructions() -> str:
        """What a Claude Code PR job is told on top of its prompt during a trial: the checkout is
        the PR's head, so the change goes onto that PR. "" on a normal branch."""
        trial = PullRequestTrial.current()
        if trial is None:
            return ""
        if not trial.head_ref:
            error(f"PullRequestTrial: PR #{trial.number}'s branch is unknown, the job is told nothing of the trial")
            return ""
        return (f"\nIMPORTANT: this server runs PR #{trial.number} as a trial, and the checkout is on that PR's head "
                f"(local branch `{trial.branch}`). The request is feedback on that PR, so it is the PR to change: "
                f"do NOT create a branch or open a new pull request. Commit on the current branch and push it to the "
                f"PR's own branch with `git push origin HEAD:refs/heads/{trial.head_ref}`; the PR updates by itself. "
                f"Name PR #{trial.number} in your report.\n")

    @staticmethod
    def take_rollback_note() -> Optional[str]:
        """What start.sh wrote when a trial's server never came up and it went back, once."""
        try:
            text = ROLLBACK_NOTE_PATH.read_text(errors="replace").strip()
        except FileNotFoundError:
            return None
        except OSError as e:
            error(f"PullRequestTrial: {ROLLBACK_NOTE_PATH} could not be read: {e}")
            return None
        try:
            ROLLBACK_NOTE_PATH.unlink()
        except OSError as e:
            error(f"PullRequestTrial: {ROLLBACK_NOTE_PATH} could not be removed, the note will repeat: {e}")
        if not text:
            trace("PullRequestTrial: an empty rollback note, nothing to tell")
            return None
        info(f"PullRequestTrial: a trial was rolled back by start.sh: {text.splitlines()[0]}")
        return f"PR trial rolled back. {text[-ROLLBACK_NOTE_CHARS:]}"


def parse_number(text: str) -> Optional[int]:
    """The PR number in `204`, `#204`, `PR 204` or a GitHub PR URL; None for anything else."""
    match = re.search(r"/pull/(\d+)", text or "") or re.fullmatch(r"\s*(?:pr\s*)?#?\s*(\d+)\s*", text or "", re.IGNORECASE)
    return int(match.group(1)) if match else None
