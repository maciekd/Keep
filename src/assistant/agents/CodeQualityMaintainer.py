import html
import json
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import common.DevelopmentConfig as DevelopmentConfig
from common.logging import info, error, trace, note
from ..llm import parse_llm_config
from common.ConfigManager import ConfigManager
from common.Paths import REPO_DIR, AISERVER_DIR as _AISERVER_PATH
from common.RequirementsChecker import RequirementsChecker

WORKING_DIR = str(REPO_DIR)
AISERVER_DIR = str(_AISERVER_PATH)
CLAUDE_FALLBACK_PATH = os.path.expanduser("~/.local/bin/claude")   # where claude.ai/install.sh puts it
CLAUDE_PATH = shutil.which("claude") or CLAUDE_FALLBACK_PATH
RUN_HOUR = 3
SLOW_TEST_MS = 100
TEST_HEALTH_RUNS = 10
CLAUDE_TIMEOUT = 60 * 60
AUTH_STATUS_TIMEOUT = 30
GIT_TIMEOUT = 60
CHECK_INTERVAL = 900

API_KEY_CONFIG_KEY = "api_keys.anthropic"   # stands in for a Claude Code login, billed per token
API_KEY_ENV = "ANTHROPIC_API_KEY"            # how the CLI takes a key: its environment, never argv
NOT_LOGGED_IN_MARKERS = ("not logged in", "please run /login", "invalid api key")
CREDENTIALS_HINT = (
    f"Either put an Anthropic API key in config.yaml ({API_KEY_CONFIG_KEY}; the runs are then billed "
    "per token to that key), or log in with a claude.ai subscription: open a terminal on the server "
    "machine itself, run `claude` and complete the `/login` flow — not over SSH, a login made in an "
    "SSH session is not seen by the server."
)
NOT_LOGGED_IN_MESSAGE = (
    "Claude Code has no credentials on the server, so this job could not run.\n\n" + CREDENTIALS_HINT
)
NOT_INSTALLED_MESSAGE = (
    "Claude Code is not installed on the server, so this job could not run.\n\n"
    "To fix it, install the CLI on the server machine (`curl -fsSL https://claude.ai/install.sh | bash`). "
    + CREDENTIALS_HINT
)
# One-liners for the integration's health state, what list_integrations shows.
NOT_INSTALLED_SHORT = ("Claude Code CLI not found: install it on the server machine "
                       "(curl -fsSL https://claude.ai/install.sh | bash); with api_keys.anthropic set nothing "
                       "else is needed")
NOT_LOGGED_IN_SHORT = (f"Claude Code has no credentials: set {API_KEY_CONFIG_KEY} in config.yaml, or run `claude` "
                       "in a terminal on the server machine itself (not over SSH) and complete /login")
DISABLED_MESSAGE = (
    'The "Self Improvement" integration is disabled, so Claude Code is not used. '
    'enable_integration("Self Improvement") turns it on; Claude Code must be installed on the server '
    'machine, and api_keys.anthropic or a claude.ai login gives it credentials — '
    'config_help("self_improvement") has the steps.'
)
NIGHTLY_RUNS_KEY = "self_improvement.nightly_runs"
DEVELOPER_RELEASE_KEY = "developer_release"
NIGHTLY_UPDATE_KEY = "nightly_update"
RELEASE_BRANCH = "release"
DEVELOPER_BRANCH = "master"
NIGHTLY_OFF_MESSAGE = f"{NIGHTLY_RUNS_KEY} is off"
NIGHTLY_STEPS = ("bugs", "tests", "health", "merge")   # what //nightly takes to run one step alone
_nightly_lock = threading.Lock()   # the scheduled run and //nightly work in the same checkout


class ClaudeUnavailableError(RuntimeError):
    """The `claude` CLI on the server cannot run a job; the message says what to do about it."""


class ClaudeNotLoggedInError(ClaudeUnavailableError):
    """The `claude` CLI on the server has neither a login nor an API key to stand in for one;
    nothing it is asked to do can succeed."""
    def __init__(self):
        super().__init__(NOT_LOGGED_IN_MESSAGE)


class ClaudeNotInstalledError(ClaudeUnavailableError):
    """No `claude` binary on the PATH or at the installer's default location."""
    def __init__(self):
        super().__init__(NOT_INSTALLED_MESSAGE)


def claude_path() -> Optional[str]:
    """The `claude` binary, resolved now rather than at import so an install after the server
    started is found: the PATH first, then the installer's default location."""
    found = shutil.which("claude")
    if found:
        return found
    if os.path.isfile(CLAUDE_FALLBACK_PATH) and os.access(CLAUDE_FALLBACK_PATH, os.X_OK):
        return CLAUDE_FALLBACK_PATH
    return None


def claude_api_key() -> Optional[str]:
    """The Anthropic API key of config.yaml, or None when it is absent or still the example
    placeholder — what stands in for a Claude Code login."""
    config = ConfigManager()
    if config.missing_required((API_KEY_CONFIG_KEY,)):
        return None
    return config.get(API_KEY_CONFIG_KEY)


@dataclass
class ClaudeStatus:
    """What `claude auth status` says about the CLI on this machine, and whether the config's
    API key stands in for a missing login."""
    path: Optional[str]      # None when the binary is missing
    logged_in: bool          # a login, or the API key standing in for one
    detail: str              # "logged in as … (max)" or the one-line reason it cannot run
    api_key: bool = False    # True: no login, the runs get api_keys.anthropic in their environment

    @property
    def ok(self) -> bool:
        return self.path is not None and self.logged_in

    def env(self) -> Optional[dict]:
        """The environment for a run: the server's own, plus the API key when it stands in for
        a login. None (inherit) otherwise, so a claude.ai login keeps its precedence and its
        billing — the variable would shadow it."""
        if not self.api_key:
            return None
        return {**os.environ, API_KEY_ENV: claude_api_key() or ""}


def claude_status() -> ClaudeStatus:
    """A quick, free check that the CLI is installed and has credentials — `claude auth status
    --json`, no model call — the Self Improvement integration's health check. The check runs
    without the config's key so it reports the login the CLI would use on its own; only when
    there is none does api_keys.anthropic stand in."""
    path = claude_path()
    if path is None:
        info("Claude Code status: CLI not found on the PATH or at " + CLAUDE_FALLBACK_PATH)
        return ClaudeStatus(None, False, NOT_INSTALLED_SHORT)
    try:
        result = subprocess.run([path, "auth", "status", "--json"], cwd=WORKING_DIR,
                                capture_output=True, text=True, stdin=subprocess.DEVNULL,
                                timeout=AUTH_STATUS_TIMEOUT)
    except subprocess.TimeoutExpired:
        error(f"Claude Code status: `claude auth status` timed out after {AUTH_STATUS_TIMEOUT}s")
        return ClaudeStatus(path, False, f"`claude auth status` timed out after {AUTH_STATUS_TIMEOUT}s")
    except OSError as e:
        error(f"Claude Code status: could not run {path}: {e}")
        return ClaudeStatus(path, False, f"could not run {path}: {e}")
    try:
        data = json.loads(result.stdout or "")
        if not isinstance(data, dict):
            raise ValueError("not a JSON object")
    except ValueError:
        output = (result.stderr or result.stdout or "").strip().replace("\n", " ")[:200]
        error(f"Claude Code status: `claude auth status` gave no JSON (exit {result.returncode}): {output}")
        return ClaudeStatus(path, False, f"`claude auth status` failed (exit {result.returncode}): {output}")
    if not data.get("loggedIn"):
        if claude_api_key():
            info(f"Claude Code status: not logged in, {API_KEY_CONFIG_KEY} stands in ({path})")
            return ClaudeStatus(path, True, f"no login, using {API_KEY_CONFIG_KEY} (billed per token)", api_key=True)
        info(f"Claude Code status: not logged in and no {API_KEY_CONFIG_KEY} ({path})")
        return ClaudeStatus(path, False, NOT_LOGGED_IN_SHORT)
    who = data.get("email") or data.get("authMethod") or "unknown account"
    plan = data.get("subscriptionType") or data.get("apiProvider") or ""
    detail = f"logged in as {who}" + (f" ({plan})" if plan else "")
    trace(f"Claude Code status: {detail}, {path}")
    return ClaudeStatus(path, True, detail)


GENERATE_TESTS_PROMPT = """
Look over the commits from the last two days. 
Has something been added without adding proper unit tests (tests that will actually ADD VALUE and not bloat)?
If you find missing tests that will ADD VALUE, create them. 
Test data is fictional (see "Test data" in CLAUDE.md): use the standing cast, never values from logs, config or the database.
Follow "Tests not to write" in CLAUDE.md: no pinned constants, stored-argument checks, mock-in-mock-out or byte-for-byte
string tests — every assertion must be a derived expectation.
Then create a pull request and push it (check rules for commits in the /commit skill).

IMPORTANT: The user will only see your final message so keep the entire analysis in your last message.
"""

BUG_ANALYSIS_PROMPT = """
Run the /check skill with the target "last 24h" to review the last day's commits for bugs.

IMPORTANT: Your entire analysis must be in your final message. Do not ask follow-up questions or offer to implement fixes.
"""

SELF_ANALYSIS_PROMPT = """
I want you to analyze something for me. It may be about this code base (a bug report, how a feature works or
why it behaved as it did, a review, a feasibility check) or about the machine the server runs on (the home
network, a device on it, disk space, updates, a service):

{request}

For the code: log/floxy.out has the server logs (huge — tail it or grep for the time in question).
For the machine: use the shell, but only observe — change, install, restart or write nothing outside the repo.
Never put config.yaml, vault or ~/.claude contents in the report.

Do a thorough analysis and provide a report with what you found.

IMPORTANT: Your entire analysis must be in your final message. Do not ask follow-up questions or offer to implement changes. Keep the report concise — it will be read over a chat on a phone.
"""

TEST_HEALTH_PROMPT = """
Check the health of the unit test suite: it must be reliable and FAST. Find the flaky tests and the slow
tests, and fix them.

Measure:
1. Run the unit suite {runs} times (the command is in CLAUDE.md, "Always run unit tests"), each time with
   `--durations=0 --durations-min=0.02` added and the whole output kept in a file in your scratch directory
   (the `grep` in that command would drop the durations), so that for every test you have its outcome and
   its duration in each run. A test's duration is its setup + call + teardown together.
2. Flaky: a test that failed in some runs but not in all of them. Also flaky: a test whose slowest run took
   over a second while its fastest was under the limit below.
3. Slow: a test whose FASTEST run took more than {slow_ms} ms. The fastest run is the measure because
   machine noise and one-off costs (a first import, a shared fixture) only ever add time.
4. A test that failed in every run is broken, not flaky: leave it alone and report it.

Fix — change files under tests/ ONLY, never the code under test:
- Flaky: find the cause (ordering, shared state, real time, threads, a race) and remove it. Never hide it
  with a retry, a skip, an xfail or a wider tolerance. When the cause is a real bug in the code under test,
  leave both the test and the code as they are and describe the bug in your report.
- Slow: remove the cost at its source (a real sleep or timeout, real I/O, an expensive fixture or import that
  can be faked or made cheaper). When a test cannot get under {slow_ms} ms and what it checks is not worth
  that time, delete it. No test over {slow_ms} ms stays.
- Follow "Test data" and "Tests not to write" in CLAUDE.md in anything you rewrite.
- Verify: measure every test you touched {runs} more times, then run the whole suite once.

If you changed anything, create a branch, commit (check rules for commits in the /commit skill), push it and
create ONE pull request with all of it. If nothing was flaky or slow, change nothing and say so.

Final report:
- Flaky tests: name, cause, fix
- Slow tests made faster: name, before -> after
- Tests REMOVED: every one by name, with why it was not worth keeping
- Bugs in the code under test that a flaky test exposed: the test, the file and line, what goes wrong
- Anything left undone (broken tests, flaky or slow ones you could not fix), and the pull request

IMPORTANT: The user will only see your final message so keep the entire report in your last message.
"""

SELF_IMPROVEMENT_PROMPT = """
This is a request for an automated change to this code base, delivered as a pull request. It is usually a bug fix,
but it can also be a new feature/improvement/integration/etc.

{request}

Steps:
1. Prepare for PR: create a branch.
2. Implement the change.
3. If applicable create new unit tests.
4. Run all unit tests to make sure nothing is broken.
5. Commit (check rules for commits in the /commit skill) and create a PR and push it. 

Log files from running system are available at:
log/floxy.out (Floxy server logs, be mindful, it's huge, so tail)

NOTE: If the request is too vague to act on or the change is too complex/risky for an automated PR, decline and explain why.

IMPORTANT: Your entire response to the user must be in your final message. Keep the report concise — it will be read over a chat on a phone.
"""


class CodeQualityMaintainer:
    """Runs the nightly Claude Code jobs on the repo, one after another: bug analysis, test generation,
    test health (flaky and slow tests), then the auto-merge of test-only PRs — one email for the night.

    A daemon thread fires the run at RUN_HOUR while `self_improvement.nightly_runs` is on (off by
    default, read at each fire); a night without recent commits only auto-merges. Once the run is
    over — or at RUN_HOUR itself when it is off — the nightly update pulls the branch
    `developer_release` names and restarts the server if the code changed (`nightly_update`, on
    by default). Also exposes the ad-hoc analyze_code / improve_code / restart helpers behind the
    self_analysis / self_improvement commands.
    """
    def __init__(self):
        self._running = False
        self._thread = None
        self._last_run_date = None
        self._boot_commit = ""   # HEAD when this process started; the nightly update restarts once it differs

    def start(self):
        self._boot_commit = self._head_commit()
        info(f"CodeQualityMaintainer: this process runs commit {self._boot_commit[:8] or 'unknown'}")
        now = datetime.now()
        if now.hour == RUN_HOUR:
            # Most often the nightly update's own restart: without this the new process would
            # see "RUN_HOUR, nothing ran today" and start the whole night over.
            self._last_run_date = now.date()
            info(f"CodeQualityMaintainer: started inside the {RUN_HOUR}:00 hour, tonight's run counts as done")
        self._running = True
        self._thread = threading.Thread(target=self._run, name="CodeQualityMaintainer", daemon=True)
        self._thread.start()
        info("Starting CodeQualityMaintainer...")

    def stop(self):
        self._running = False

    def _run(self):
        while self._running:
            try:
                now = datetime.now()
                if now.hour == RUN_HOUR and self._last_run_date != now.date():
                    self._last_run_date = now.date()
                    try:
                        self._execute_nightly_run()
                    finally:
                        self._nightly_update()
            except Exception as e:
                error(f"CodeQualityMaintainer error: {e}")
            for _ in range(CHECK_INTERVAL):
                if not self._running:
                    return
                time.sleep(1)

    @staticmethod
    def _has_recent_commits(days=2) -> bool:
        try:
            result = subprocess.run(
                ["git", "log", f"--since={days} days ago", "--oneline"],
                cwd=WORKING_DIR, capture_output=True, text=True, timeout=GIT_TIMEOUT
            )
            commits = result.stdout.strip()
            if commits:
                count = len(commits.splitlines())
                info(f"CodeQualityMaintainer: {count} commit(s) in the last {days} days.")
                return True
            else:
                info(f"CodeQualityMaintainer: no commits in the last {days} days.")
                return False
        except Exception as e:
            error(f"CodeQualityMaintainer: failed to check recent commits: {e}")
            return True

    @staticmethod
    def unavailable_reason() -> Optional[str]:
        """Why Claude Code jobs cannot run right now, None when they can: the "Self Improvement"
        integration is the switch. A missing or logged-out CLI is reported by the run itself."""
        from integrations.self_improvement.SelfImprovementIntegration import SelfImprovementIntegration
        if not SelfImprovementIntegration.enabled(ConfigManager()):
            return DISABLED_MESSAGE
        return None

    @staticmethod
    def nightly_skip_reason() -> Optional[str]:
        """Why the scheduled nightly run must not fire now, None when it
        may: the integration switch first, then `self_improvement.nightly_runs`, which is read at
        each fire so a config_set takes effect without a restart. The ad-hoc commands ignore it."""
        reason = CodeQualityMaintainer.unavailable_reason()
        if reason:
            return reason
        if not ConfigManager().get(NIGHTLY_RUNS_KEY, False):
            return NIGHTLY_OFF_MESSAGE
        return None

    def _execute_nightly_run(self, force=False, only=None):
        """The whole night, or with `only` (one of NIGHTLY_STEPS) that step alone. `force` is
        //nightly: it runs with `self_improvement.nightly_runs` off, never with the integration off."""
        reason = self.unavailable_reason() if force else self.nightly_skip_reason()
        if reason:
            info(f"CodeQualityMaintainer: skipping nightly run — {reason}")
            return
        if not _nightly_lock.acquire(blocking=False):
            info("CodeQualityMaintainer: a nightly run is already going, not starting another.")
            return
        try:
            self._nightly_run(only)
        finally:
            _nightly_lock.release()

    def _nightly_run(self, only):
        note(f"CodeQualityMaintainer: starting nightly run{f' ({only} only)' if only else ''}...")
        self._run_subprocess(["git", "pull"], "git pull", GIT_TIMEOUT)
        steps = dict(zip(NIGHTLY_STEPS, [("Bug Analysis", self._step_bug_analysis),
                                         ("Test Generation", self._step_generate_tests),
                                         ("Test Health", self._step_test_health),
                                         ("Auto-Merge Test PRs", self._step_auto_merge)]))
        if only:
            steps = [steps[only]]
        elif self._has_recent_commits():
            steps = list(steps.values())
        else:
            info("CodeQualityMaintainer: no recent commits, only the auto-merge runs tonight.")
            steps = [steps["merge"]]
        sections = []
        for title, step in steps:
            try:
                report = step()
                if report:
                    sections.append((title, report))
                else:
                    info(f"CodeQualityMaintainer: {title} produced no report.")
            except Exception as e:
                error(f"CodeQualityMaintainer: {title} exception: {e}")
        if sections:
            combined_md = "\n\n---\n\n".join(
                f"# {title}\n\n{body}" for title, body in sections
            )
            html_report = self._markdown_to_html(combined_md)
            self._send_email("Keep Nightly Code Quality Report", html_report)
        else:
            info("CodeQualityMaintainer: nothing to report, skipping email.")
        info("CodeQualityMaintainer: nightly run complete.")

    def _nightly_update(self):
        """After the night's run (at RUN_HOUR itself when that is off or skipped): check out the
        branch `developer_release` names, pull it, and restart when the code is no longer what
        this process booted on — commits pulled now, pulled by the night's run, or made locally
        since the start. No new commit, no restart. `nightly_update: false` turns it off; it is
        read at each fire."""
        from assistant.SystemNote import system_note
        if not ConfigManager().get(NIGHTLY_UPDATE_KEY, True):
            info(f"CodeQualityMaintainer: {NIGHTLY_UPDATE_KEY} is off, no nightly update.")
            return
        if DevelopmentConfig.LOCAL_DEVELOPMENT_MODE:
            info("CodeQualityMaintainer: LOCAL_DEVELOPMENT_MODE is on, no nightly update.")
            return
        if not self._boot_commit:
            info("CodeQualityMaintainer: the commit this process started on is unknown, no nightly update.")
            return
        if not _nightly_lock.acquire(blocking=False):
            info("CodeQualityMaintainer: a nightly run is still going, no nightly update under it.")
            return
        try:
            target = self.target_branch()
            info(f"CodeQualityMaintainer: nightly update — bringing the checkout to the head of {target}...")
            failure, _ = self._pull_target()
            if failure:
                error(f"CodeQualityMaintainer: nightly update failed, not restarting — {failure}")
                system_note("server", f"Nightly update failed, the server keeps running the old code: {failure}",
                            push=False)
                return
            head = self._head_commit()
            if not head:
                error("CodeQualityMaintainer: nightly update could not read HEAD after the pull, not restarting.")
                return
            if head == self._boot_commit:
                info(f"CodeQualityMaintainer: nightly update — no new commits on {target} since the start "
                     f"({head[:8]}), not restarting.")
                return
            count = self._commits_between(self._boot_commit, head)
            change = f"{count} new commit{'s' if count != 1 else ''}" if count else f"now at {head[:8]}"
            failure, installed = self._sync_requirements()
            if failure:
                error(f"CodeQualityMaintainer: nightly update pulled {change} but not restarting — {failure}")
                system_note("server", f"Nightly update pulled {change} on {target} but the server keeps running "
                                      f"the old code: {failure}", push=False)
                return
            if installed:
                change += f", {installed}"
            note(f"CodeQualityMaintainer: nightly update — {change} on {target} "
                 f"({self._boot_commit[:8]} → {head[:8]}), restarting.")
            # History only: the startup announcement of the new process is what the chats get.
            system_note("server", f"Nightly update: {change} on {target}, the server is restarting.", push=False)
            self.restart(f"nightly update {self._boot_commit[:8]} → {head[:8]}")
        finally:
            _nightly_lock.release()

    @staticmethod
    def _head_commit() -> str:
        """The commit the checkout is on, "" when git cannot say."""
        try:
            result = CodeQualityMaintainer._run_subprocess(["git", "rev-parse", "HEAD"], "git rev-parse", GIT_TIMEOUT)
        except Exception as e:
            error(f"CodeQualityMaintainer: could not read HEAD: {e}")
            return ""
        if result.returncode != 0:
            error(f"CodeQualityMaintainer: git rev-parse HEAD exited {result.returncode}: {(result.stderr or '').strip()}")
            return ""
        return (result.stdout or "").strip()

    @staticmethod
    def _commits_between(old: str, new: str) -> int:
        """How many commits *new* has that *old* lacks; 0 when git cannot say or *new* is not ahead
        (a switch back to an older release)."""
        try:
            result = CodeQualityMaintainer._run_subprocess(["git", "rev-list", "--count", f"{old}..{new}"],
                                                           "git rev-list", GIT_TIMEOUT)
            return int((result.stdout or "").strip()) if result.returncode == 0 else 0
        except Exception as e:
            error(f"CodeQualityMaintainer: could not count the commits {old[:8]}..{new[:8]}: {e}")
            return 0

    @staticmethod
    def _tests_path() -> str:
        """The tests directory as GitHub spells it: relative to the git root, with a trailing slash."""
        tests_dir = _AISERVER_PATH / "tests"
        try:
            toplevel = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=WORKING_DIR,
                                      capture_output=True, text=True, timeout=GIT_TIMEOUT).stdout.strip()
            return tests_dir.relative_to(toplevel).as_posix() + "/"
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            tests_path = tests_dir.relative_to(REPO_DIR).as_posix() + "/"
            error(f"CodeQualityMaintainer: could not resolve the git root ({e}), assuming {tests_path}")
            return tests_path

    @staticmethod
    def _trusted_pr_authors() -> set[str]:
        """Whose PRs the auto-merge may touch: self_improvement.auto_merge_authors, or, when that
        is unset, the account `gh` is logged in as — the one the nightly runs open their PRs with.
        On a public repository anyone can open a PR that only touches tests/, and a merged test
        file is code the next unit run executes on this machine."""
        configured = ConfigManager().get('self_improvement.auto_merge_authors', [])
        if configured:
            return {str(login).lower() for login in configured}
        try:
            result = subprocess.run(["gh", "api", "user", "--jq", ".login"], cwd=WORKING_DIR,
                                    capture_output=True, text=True, timeout=GIT_TIMEOUT)
            login = result.stdout.strip().lower()
            if result.returncode == 0 and login:
                return {login}
            error(f"CodeQualityMaintainer: could not resolve the gh account: {result.stderr.strip()}")
        except Exception as e:
            error(f"CodeQualityMaintainer: could not resolve the gh account: {e}")
        return set()

    @staticmethod
    def _auto_merge_candidates() -> list[int]:
        """The open PRs the auto-merge may evaluate, oldest first: from a trusted author and from a
        branch of this repository, never from a fork."""
        trusted = CodeQualityMaintainer._trusted_pr_authors()
        if not trusted:
            info("CodeQualityMaintainer: no trusted PR author known, the auto-merge evaluates nothing.")
            return []
        try:
            result = subprocess.run(
                ["gh", "pr", "list", "--state", "open", "--json", "number,author,isCrossRepository"],
                cwd=WORKING_DIR, capture_output=True, text=True, timeout=GIT_TIMEOUT
            )
            if result.returncode != 0:
                error(f"CodeQualityMaintainer: gh pr list failed: {result.stderr.strip()}")
                return []
            prs = json.loads(result.stdout or "[]")
        except Exception as e:
            error(f"CodeQualityMaintainer: failed to list open PRs: {e}")
            return []
        candidates = []
        for pr in prs:
            author = ((pr.get("author") or {}).get("login") or "").lower()
            if pr.get("isCrossRepository") or author not in trusted:
                info(f"CodeQualityMaintainer: PR #{pr.get('number')} by {author or 'unknown'} is left alone "
                     f"({'from a fork' if pr.get('isCrossRepository') else 'not a trusted author'}).")
                continue
            candidates.append(pr["number"])
        info(f"CodeQualityMaintainer: {len(candidates)} of {len(prs)} open PR(s) are from trusted authors "
             f"({', '.join(sorted(trusted))}).")
        return sorted(candidates)

    @staticmethod
    def _pr_title_and_files(number: int) -> tuple[str, list[str]]:
        """A PR's title and every path it touches — for a renamed file both the new and the previous
        path, so moving src/x.py under tests/ still shows as a change outside tests/ (`gh pr view`
        only knows the new one, hence the REST file list). Raises when gh fails or hands back fewer
        files than the PR changes, since an unseen path cannot be vouched for."""
        def gh(*args: str) -> str:
            result = subprocess.run(["gh", *args], cwd=WORKING_DIR, capture_output=True, text=True,
                                    timeout=GIT_TIMEOUT)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or f"gh {args[0]} exited {result.returncode}")
            return result.stdout

        pr = json.loads(gh("pr", "view", str(number), "--json", "title,changedFiles"))
        files = [json.loads(line) for line in
                 gh("api", f"repos/{{owner}}/{{repo}}/pulls/{number}/files", "--paginate",
                    "--jq", ".[] | {filename, previous_filename}").splitlines() if line.strip()]
        if len(files) != pr.get("changedFiles"):
            raise RuntimeError(f"gh listed {len(files)} of {pr.get('changedFiles')} changed files")
        paths = []
        for f in files:
            paths.append(f["filename"])
            if f.get("previous_filename"):
                trace(f"CodeQualityMaintainer: PR #{number} moves {f['previous_filename']} to {f['filename']}")
                paths.append(f["previous_filename"])
        return pr.get("title", ""), paths

    def _step_auto_merge(self) -> str:
        """Last step of the night, so the test-only PRs the earlier steps opened are merged with it.
        Plain code, no Claude Code run: a PR qualifies when every path it touches is under tests/."""
        info("CodeQualityMaintainer: auto-merge of test-only PRs...")
        candidates = self._auto_merge_candidates()
        if not candidates:
            info("CodeQualityMaintainer: no open PRs from trusted authors, skipping the auto-merge.")
            return ""
        tests_path = self._tests_path()
        merge_strategy = ConfigManager().get('self_improvement.merge_strategy', 'rebase')
        info(f"CodeQualityMaintainer: auto-merge covers {tests_path} with --{merge_strategy}")
        branch = self._current_branch()
        merged, skipped = [], []
        try:
            for number in candidates:
                title = ""
                try:
                    title, paths = self._pr_title_and_files(number)
                    outside = [p for p in paths if not p.startswith(tests_path)]
                    if outside or not paths:
                        reason = f"changes files outside `{tests_path}`: {', '.join(outside[:5])}" if outside \
                            else "changes no files"
                    else:
                        # --admin: the branch's "restrict updates" rule never passes for a non-bypass
                        # merge (so --auto would wait forever); the checks above keep the bypass narrow.
                        result = self._run_subprocess(
                            ["gh", "pr", "merge", str(number), f"--{merge_strategy}", "--delete-branch",
                             "--admin"],
                            f"gh pr merge {number}", GIT_TIMEOUT)
                        reason = "" if result.returncode == 0 else \
                            f"merge failed: {result.stderr.strip() or f'gh exited {result.returncode}'}"
                except Exception as e:
                    reason = f"could not be evaluated: {e}"
                if reason:
                    info(f"CodeQualityMaintainer: PR #{number} not merged — {reason}")
                    skipped.append(f"- #{number} {title} — {reason}")
                else:
                    info(f"CodeQualityMaintainer: merged PR #{number} ({title})")
                    merged.append(f"- #{number} {title}")
        finally:
            self._run_subprocess(["git", "checkout", branch], "git checkout", GIT_TIMEOUT)
        info(f"CodeQualityMaintainer: auto-merge complete, {len(merged)} merged, {len(skipped)} skipped.")
        return "\n\n".join(f"## {heading}\n\n" + ("\n".join(lines) or "None.")
                           for heading, lines in (("Merged PRs", merged), ("Skipped PRs", skipped)))

    @staticmethod
    def _run_subprocess(cmd, label, timeout):
        result = subprocess.run(cmd, cwd=WORKING_DIR, timeout=timeout,
                                capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if result.stdout and result.stdout.strip():
            trace(f"CodeQualityMaintainer [{label}] stdout: {result.stdout.strip()}")
        if result.stderr and result.stderr.strip():
            trace(f"CodeQualityMaintainer [{label}] stderr: {result.stderr.strip()}")
        return result

    @staticmethod
    def _format_stream_line(obj):
        msg_type = obj.get("type", "")
        if msg_type == "assistant":
            message = obj.get("message", {})
            parts = []
            for block in message.get("content", []):
                block_type = block.get("type", "")
                if block_type == "text":
                    parts.append(block.get("text", ""))
                elif block_type == "thinking":
                    snippet = (block.get("thinking") or "")[:120]
                    if snippet:
                        parts.append(f"(thinking: {snippet}...)")
            return " | ".join(parts) if parts else None
        if msg_type == "system" and obj.get("subtype") == "init":
            return (f"INIT: model {obj.get('model')}, permission mode {obj.get('permissionMode')}, "
                    f"credentials {obj.get('apiKeySource') or 'login'}, CLI {obj.get('claude_code_version')}")
        if msg_type == "result":
            result = obj.get("result", "")
            cost = obj.get("total_cost_usd") or obj.get("cost_usd")   # cost_usd: CLIs before the rename
            preview = result[:200] if result else "(empty)"
            facts = [f"${cost:.4f}" if cost else "",
                     f"{obj['num_turns']} turns" if obj.get("num_turns") else "",
                     f"{obj['duration_ms'] / 1000:.0f} s" if obj.get("duration_ms") else ""]
            facts_str = ", ".join(f for f in facts if f)
            return f"RESULT ({facts_str}): {preview}" if facts_str else f"RESULT: {preview}"
        return None

    @staticmethod
    def _run_claude(prompt, label):
        status = claude_status()
        if status.path is None:
            error(f"CodeQualityMaintainer [{label}] Claude Code CLI not found on this machine")
            raise ClaudeNotInstalledError()
        if not status.ok:
            error(f"CodeQualityMaintainer [{label}] Claude Code has no credentials: {status.detail}")
            raise ClaudeNotLoggedInError()
        _, model = parse_llm_config('code_quality')  # only the model matters; the runner is always the `claude` CLI
        trace(f"CodeQualityMaintainer [{label}] using Claude Code model {model}, {status.detail}")
        cmd = [status.path, "--permission-mode", "auto", "-p", prompt,
               "--output-format", "stream-json", "--verbose", "--model", model]
        process = subprocess.Popen(
            cmd, cwd=WORKING_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL, text=True, env=status.env()
        )
        timer = threading.Timer(CLAUDE_TIMEOUT, process.kill)
        timer.start()
        result_text = ""
        result_is_error = False
        last_assistant_text = ""
        stderr = ""
        try:
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    summary = CodeQualityMaintainer._format_stream_line(obj)
                    if summary:
                        trace(f"CodeQualityMaintainer [{label}] {summary}")
                    if obj.get("type") == "system" and obj.get("subtype") == "init" \
                            and obj.get("permissionMode") != "auto":
                        # The CLI falls back to Manual silently when auto mode is unavailable (an
                        # unsupported model, an old CLI); in -p every prompt is then a denial and
                        # the run ends with a report of refused edits rather than an error.
                        error(f"CodeQualityMaintainer [{label}] auto mode unavailable, Claude Code runs in "
                              f"'{obj.get('permissionMode')}' mode with model {obj.get('model')}: edits and "
                              f"commands will be denied — use a model that supports auto mode and a current CLI")
                    if obj.get("type") == "assistant":
                        text_parts = []
                        for block in obj.get("message", {}).get("content", []):
                            if block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                        if text_parts:
                            last_assistant_text = "\n".join(text_parts)
                    if obj.get("type") == "result":
                        result_text = obj.get("result", "")
                        result_is_error = bool(obj.get("is_error"))
                except json.JSONDecodeError:
                    trace(f"CodeQualityMaintainer [{label}] {line}")
            process.wait()
            stderr = process.stderr.read()
            if stderr and stderr.strip():
                trace(f"CodeQualityMaintainer [{label}] stderr: {stderr.strip()}")
        finally:
            timer.cancel()
        output = result_text or last_assistant_text
        # A successful report may legitimately discuss login errors in the code under review, so
        # the result text only counts when the CLI itself flagged the run as failed.
        if (result_is_error and CodeQualityMaintainer._is_not_logged_in(output)) \
                or CodeQualityMaintainer._is_not_logged_in(stderr):
            error(f"CodeQualityMaintainer [{label}] Claude Code is not logged in on this machine: {output.strip() or stderr.strip()}")
            raise ClaudeNotLoggedInError()
        return output

    @staticmethod
    def _is_not_logged_in(text) -> bool:
        lowered = (text or "").lower()
        return any(marker in lowered for marker in NOT_LOGGED_IN_MARKERS)

    @staticmethod
    def _step_generate_tests() -> str:
        branch = CodeQualityMaintainer._current_branch()
        try:
            info("CodeQualityMaintainer: generating missing tests for recent commits...")
            trace("Starting Claude Code in the background...")
            report = CodeQualityMaintainer._run_claude(GENERATE_TESTS_PROMPT, "generate-tests")
            info("CodeQualityMaintainer: test generation complete.")
            return report
        except Exception as e:
            error(f"CodeQualityMaintainer: test generation failed: {e}")
            return ""
        finally:
            CodeQualityMaintainer._run_subprocess(["git", "checkout", branch], "git checkout", GIT_TIMEOUT)

    @staticmethod
    def _step_test_health() -> str:
        branch = CodeQualityMaintainer._current_branch()
        try:
            info(f"CodeQualityMaintainer: test health — flaky tests and tests over {SLOW_TEST_MS} ms "
                 f"across {TEST_HEALTH_RUNS} runs of the unit suite...")
            report = CodeQualityMaintainer._run_claude(
                TEST_HEALTH_PROMPT.format(runs=TEST_HEALTH_RUNS, slow_ms=SLOW_TEST_MS), "test-health")
            info("CodeQualityMaintainer: test health complete.")
            return report
        except Exception as e:
            error(f"CodeQualityMaintainer: test health failed: {e}")
            return ""
        finally:
            CodeQualityMaintainer._run_subprocess(["git", "checkout", branch], "git checkout", GIT_TIMEOUT)

    @staticmethod
    def _step_bug_analysis() -> str:
        try:
            info("CodeQualityMaintainer: bug analysis...")
            trace("Starting Claude Code in the background...")
            report = CodeQualityMaintainer._run_claude(BUG_ANALYSIS_PROMPT, "bug-analysis")
            info("CodeQualityMaintainer: bug analysis complete.")
            return report
        except Exception as e:
            error(f"CodeQualityMaintainer: bug analysis failed: {e}")
            return ""

    @staticmethod
    def analyze_code(request: str) -> str:
        info("CodeQualityMaintainer: starting self-analysis...")
        prompt = SELF_ANALYSIS_PROMPT.format(request=request)
        trace(f"prompt: {prompt}")
        try:
            result = CodeQualityMaintainer._run_claude(prompt, "self-analysis")
        except ClaudeUnavailableError as e:
            return str(e)
        if result:
            try:
                html_report = CodeQualityMaintainer._markdown_to_html(f"# Self-analysis\n\n{result}")
                CodeQualityMaintainer._send_email("Keep Self-analysis Report", html_report)
            except Exception as e:
                error(f"CodeQualityMaintainer: failed to email self-analysis report: {e}")
        else:
            info("CodeQualityMaintainer: self-analysis produced no output, skipping email.")
        return result if result else "Self-analysis produced no output."

    @staticmethod
    def improve_code(request: str) -> str:
        branch = CodeQualityMaintainer._current_branch()
        try:
            info("CodeQualityMaintainer: starting self-improvement...")
            CodeQualityMaintainer._run_subprocess(["git", "pull"], "git pull", GIT_TIMEOUT)
            prompt = SELF_IMPROVEMENT_PROMPT.format(request=request)
            trace(f"prompt: {prompt}")
            result = CodeQualityMaintainer._run_claude(prompt, "self-improvement")
            return result if result else "Self-improvement produced no output."
        except ClaudeUnavailableError as e:
            return str(e)
        except Exception as e:
            error(f"CodeQualityMaintainer: self-improvement failed: {e}")
            return f"Self-improvement failed: {e}"
        finally:
            CodeQualityMaintainer._run_subprocess(["git", "checkout", branch], "git checkout", GIT_TIMEOUT)

    _PR_URL = re.compile(r"https?://github\.com/[\w.-]+/[\w.-]+/pull/(\d+)")
    _PR_NUMBER = re.compile(r"\bPR\s*#(\d+)|\bpull request\s*#(\d+)", re.IGNORECASE)

    @classmethod
    def pull_request_in(cls, report: str) -> tuple[int | None, str | None]:
        """(number, URL) of the pull request a self-improvement report names — the URL when
        the report has one, else a "PR #N" mention without a URL; (None, None) for a report
        that opened none (a declined request, a failed run)."""
        match = cls._PR_URL.search(report or "")
        if match:
            return int(match.group(1)), match.group(0)
        match = cls._PR_NUMBER.search(report or "")
        if match:
            return int(match.group(1) or match.group(2)), None
        trace("CodeQualityMaintainer: the report names no pull request")
        return None, None

    @staticmethod
    def first_line(report: str) -> str:
        """The report's first line of text, its markdown dressing (heading marks, emphasis,
        list bullets) stripped — the title line of a PR report."""
        for line in (report or "").splitlines():
            text = re.sub(r"[*_`]+", "", line.strip().lstrip("#-*> ")).strip()
            if text:
                return text
        return "(empty report)"

    @staticmethod
    def _current_branch() -> str:
        """The branch this checkout follows — `release` on an install, `master` on the
        development machine. A Claude Code run may leave the tree on its work branch, so the
        runs restore this one afterwards; a hard-coded master would move an install onto the
        development branch."""
        try:
            result = CodeQualityMaintainer._run_subprocess(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                                           "git rev-parse", GIT_TIMEOUT)
            branch = (result.stdout or "").strip()
        except Exception as e:
            error(f"CodeQualityMaintainer: could not read the current branch: {e}")
            branch = ""
        if not branch or branch == "HEAD":
            info(f"CodeQualityMaintainer: no branch checked out ({branch or 'no output'}), assuming master")
            return "master"
        return branch

    @staticmethod
    def target_branch() -> str:
        """The branch restart_server pulls: master with `developer_release: true`, else release."""
        developer = bool(ConfigManager().get(DEVELOPER_RELEASE_KEY, False))
        return DEVELOPER_BRANCH if developer else RELEASE_BRANCH

    @staticmethod
    def _pull_target() -> tuple[str, str]:
        """Check out the target branch if the checkout is on another one, then pull it. Returns
        (failure, pull output): *failure* says what went wrong and is "" when the pull went through."""
        target = CodeQualityMaintainer.target_branch()
        current = CodeQualityMaintainer._current_branch()
        if current != target:
            info(f"CodeQualityMaintainer: on {current}, {DEVELOPER_RELEASE_KEY} asks for {target} — switching before the pull")
            for cmd, label in ((["git", "fetch", "origin", target], "git fetch"),
                               (["git", "checkout", target], "git checkout")):
                result = CodeQualityMaintainer._run_subprocess(cmd, label, GIT_TIMEOUT)
                if result.returncode != 0:
                    error(f"CodeQualityMaintainer: {label} {target} failed: {result.stderr}")
                    return (f"Could not switch to {target}: {(result.stderr or '').strip()} — "
                            f"still on {current}, the server was not restarted."), ""
        info(f"CodeQualityMaintainer: pulling latest code on {target}...")
        result = CodeQualityMaintainer._run_subprocess(["git", "pull"], "git pull", GIT_TIMEOUT)
        if result.returncode != 0:
            error(f"CodeQualityMaintainer: git pull failed: {result.stderr}")
            return f"Git pull failed: {result.stderr}", ""
        return "", (result.stdout or "").strip()

    @staticmethod
    def _sync_requirements() -> tuple[str, str]:
        """Between the pull and the restart: pip brings the venv to the pulled pins while the old
        server still serves, so the new one never dies on its requirements check. Returns
        (failure, installed) as RequirementsChecker.sync does; a failure means do not restart."""
        return RequirementsChecker.sync()

    @staticmethod
    def pull_and_restart() -> str:
        failure, output = CodeQualityMaintainer._pull_target()
        if failure:
            return failure

        already_up_to_date = "Already up to date" in output
        if already_up_to_date:
            info("CodeQualityMaintainer: already up to date, restarting anyway...")
        else:
            info(f"CodeQualityMaintainer: pulled changes: {output}")

        failure, installed = CodeQualityMaintainer._sync_requirements()
        if failure:
            error(f"CodeQualityMaintainer: not restarting — {failure}")
            return f"The server was not restarted: {failure}"

        CodeQualityMaintainer.restart()

        changes_msg = "no new changes" if already_up_to_date else f"pulled updates: {output}"
        if installed:
            changes_msg += f"; {installed}"
        return f"Update initiated ({changes_msg}). Server is restarting..."

    @staticmethod
    def restart(reason: str = "") -> str:
        """Relaunch the server in place via start.sh (graceful kill + fresh start) WITHOUT
        pulling new code. Used by pull_and_restart() after the pull, and by automatic
        recovery paths (e.g. AudioRecovery) that just need a clean process to clear an
        in-process wedge while running the same code."""
        info(f"CodeQualityMaintainer: spawning start.sh as detached process"
             f"{f' ({reason})' if reason else ''} (our pid {os.getpid()})...")
        # Pass our own pid so start.sh can kill exactly this process. pgrep -f
        # inside start.sh has proven unreliable when start.sh is spawned from
        # within the server itself — every historical auto-restart missed the
        # old process and the replacement died on a port collision.
        subprocess.Popen(
            ['bash', 'start.sh', str(os.getpid())],
            cwd=AISERVER_DIR,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL
        )
        return "Server is restarting..."

    @staticmethod
    def _inline_markup(text: str) -> str:
        text = html.escape(text)
        text = re.sub(r'`([^`]{1,80})`',
                       lambda m: f'<code style="background:#1a2e1a;padding:1px 5px;border-radius:3px;color:#6dbf6d">{m.group(1)}</code>',
                       text)
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        return text

    @staticmethod
    def _is_table_separator(line: str) -> bool:
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            return False
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if not cells:
            return False
        return all(re.fullmatch(r":?-{3,}:?", c) for c in cells)

    @staticmethod
    def _split_table_row(line: str) -> list:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    @staticmethod
    def _markdown_to_html(md: str) -> str:
        lines = md.split("\n")
        html_lines = []
        in_code_block = False
        in_list = False

        code_block_style = ('style="background:#1a1a1a;padding:12px 16px;border-radius:6px;'
                            'overflow-x:auto;color:#b5d6b5;font-family:monospace;font-size:13px"')
        table_style = ('style="border-collapse:collapse;margin:12px 0;font-size:13px"')
        th_style = ('style="border:1px solid #444;padding:6px 10px;background:#2a2a2a;text-align:left"')
        td_style = ('style="border:1px solid #444;padding:6px 10px"')

        def close_list():
            nonlocal in_list
            if in_list:
                html_lines.append("</ul>")
                in_list = False

        i = 0
        while i < len(lines):
            line = lines[i]

            if line.startswith("```"):
                if in_code_block:
                    html_lines.append("</code></pre>")
                    in_code_block = False
                else:
                    close_list()
                    html_lines.append(f"<pre {code_block_style}><code>")
                    in_code_block = True
                i += 1
                continue

            if in_code_block:
                html_lines.append(html.escape(line))
                i += 1
                continue

            stripped = line.strip()

            if (stripped.startswith("|") and stripped.endswith("|")
                    and i + 1 < len(lines)
                    and CodeQualityMaintainer._is_table_separator(lines[i + 1])):
                close_list()
                headers = CodeQualityMaintainer._split_table_row(line)
                html_lines.append(f"<table {table_style}>")
                html_lines.append("<thead><tr>")
                for h in headers:
                    html_lines.append(f"<th {th_style}>{CodeQualityMaintainer._inline_markup(h)}</th>")
                html_lines.append("</tr></thead>")
                i += 2
                html_lines.append("<tbody>")
                while i < len(lines):
                    row_line = lines[i]
                    row_stripped = row_line.strip()
                    if not (row_stripped.startswith("|") and row_stripped.endswith("|")):
                        break
                    cells = CodeQualityMaintainer._split_table_row(row_line)
                    html_lines.append("<tr>")
                    for c in cells:
                        html_lines.append(f"<td {td_style}>{CodeQualityMaintainer._inline_markup(c)}</td>")
                    html_lines.append("</tr>")
                    i += 1
                html_lines.append("</tbody></table>")
                continue

            if stripped == "---" or stripped == "***" or stripped == "___":
                close_list()
                html_lines.append('<hr style="border:none;border-top:1px solid #444;margin:20px 0">')
            elif line.startswith("### "):
                close_list()
                html_lines.append(f"<h3>{CodeQualityMaintainer._inline_markup(line[4:])}</h3>")
            elif line.startswith("## "):
                close_list()
                html_lines.append(f"<h2>{CodeQualityMaintainer._inline_markup(line[3:])}</h2>")
            elif line.startswith("# "):
                close_list()
                html_lines.append(f"<h1>{CodeQualityMaintainer._inline_markup(line[2:])}</h1>")
            elif line.startswith("- ") or line.startswith("* "):
                if not in_list:
                    html_lines.append("<ul>")
                    in_list = True
                html_lines.append(f"<li>{CodeQualityMaintainer._inline_markup(line[2:])}</li>")
            else:
                close_list()
                processed = CodeQualityMaintainer._inline_markup(line)
                if processed.strip():
                    html_lines.append(f"<p>{processed}</p>")
                else:
                    html_lines.append("")
            i += 1

        if in_code_block:
            html_lines.append("</code></pre>")
        if in_list:
            html_lines.append("</ul>")

        return "\n".join(html_lines)

    @staticmethod
    def _send_email(subject: str, html_body: str):
        """The report goes to the household's own address: the account `send_email` sends from
        (`mail.send_from`, else the first configured one) mails it to itself."""
        from assistant.integration.mail.MailAccess import MailAccess
        account = MailAccess.current().sender()
        if account is None or not account.address:
            error("CodeQualityMaintainer: no mail account configured, skipping send.")
            return
        address = account.address

        full_html = (
            '<html><head>'
            '<meta name="color-scheme" content="dark">'
            '<meta name="supported-color-schemes" content="dark">'
            '<style>:root{color-scheme:dark}</style>'
            '</head><body style="font-family:monospace">'
            f'{html_body}'
            '</body></html>'
        )

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["To"] = address
        msg.attach(MIMEText(full_html, "html"))

        err = account.send_message(msg, [address])
        if err:
            raise RuntimeError(f"report email not sent: {err}")
        info(f"CodeQualityMaintainer: email sent to {address} from {account.name}")
