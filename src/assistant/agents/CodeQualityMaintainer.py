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
from urllib.parse import urlencode
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import common.DevelopmentConfig as DevelopmentConfig
from common.logging import info, error, trace, note
from ..llm import parse_llm_config
from ..llm.ClaudeLLM import WORKSPACE_HEADER, claude_workspace_id
from common.ConfigManager import ConfigManager
from common.Paths import REPO_DIR, LOG_DIR, AISERVER_DIR as _AISERVER_PATH
from common.SignedRelease import RELEASE_BRANCH
from common.Updater import Updater, GIT_TIMEOUT

WORKING_DIR = str(REPO_DIR)
CLAUDE_FALLBACK_PATH = os.path.expanduser("~/.local/bin/claude")   # where claude.ai/install.sh puts it
CLAUDE_PATH = shutil.which("claude") or CLAUDE_FALLBACK_PATH
RUN_HOUR = 3
SLOW_TEST_MS = 100
TEST_HEALTH_RUNS = 10
CLAUDE_TIMEOUT = 60 * 60
AUTH_STATUS_TIMEOUT = 30
CHECK_INTERVAL = 900

API_KEY_CONFIG_KEY = "api_keys.anthropic"   # stands in for a Claude Code login, billed per token
API_KEY_ENV = "ANTHROPIC_API_KEY"            # how the CLI takes a key: its environment, never argv
CUSTOM_HEADERS_ENV = "ANTHROPIC_CUSTOM_HEADERS"  # "Name: Value" lines the CLI adds to every request
NOT_LOGGED_IN_MARKERS = ("not logged in", "please run /login", "invalid api key")
# What the CLI answers in place of a report when the subscription's credits are gone (2026-10-22:
# "You're out of usage credits…" after a 24-minute bug analysis; "Fable 5.1 requires usage credits"
# when the model needs credits the plan lacks) or its spend limit is hit — every
# later run of the night answers the same within a second, so the night stops its Claude steps.
USAGE_LIMIT_MARKERS = ("usage credits", "spend limit", "usage limit")
# A result the CLI did not flag counts as the limit message only when it is one: a sentence or
# two, never the length of a report that happens to discuss a limit.
USAGE_LIMIT_MESSAGE_MAX_CHARS = 400
CREDENTIALS_HINT = (
    f"Either put an Anthropic API key in config.yaml ({API_KEY_CONFIG_KEY}; the runs are then billed "
    "per token to that key), or log in with a claude.ai subscription: open a terminal on the server "
    "machine itself, run `claude` and complete the `/login` flow — not over SSH, a login made in an "
    "SSH session is not seen by the server."
)
NOT_LOGGED_IN_MESSAGE = (
    "Claude Code has no credentials on the server, so this job could not run.\n\n" + CREDENTIALS_HINT
)
# Under every report of a run the API key paid for: the key is the costly way to run Claude
# Code, and a login on the server machine stops it without a config change.
API_KEY_COST_NOTE = (
    "**You're not logged in on Claude Code and use API credits to run — this is expensive.**{cost}\n"
    "Log into Claude Code on the server machine (locally, not over SSH: run `claude` in a terminal "
    "there and complete /login) to stop using API credits for Claude Code."
)


def api_key_cost_note(cost_usd: Optional[float]) -> str:
    """The paragraph appended to a report when api_keys.anthropic paid for the run: the
    warning, this run's cost when the CLI reported one, and the way out."""
    cost = f"\nThis run cost ${cost_usd:.2f}." if cost_usd else ""
    return "\n\n" + API_KEY_COST_NOTE.format(cost=cost)
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
DEVELOPER_BRANCH = "master"
START_SCRIPT = "start.sh"   # what relaunches the server: it takes the old process down by the pid it is handed
NIGHTLY_OFF_MESSAGE = f"{NIGHTLY_RUNS_KEY} is off"
NIGHTLY_STEPS = ("tests", "bugs", "logs", "fix", "health", "merge", "pins")   # what //nightly takes to run one step alone
FIX_STEP_INPUTS = ("bugs", "logs", "tests")   # the reports the fix step reads, under the headings REPORT_TITLES gives them
FIX_SEVERITIES = ("critical", "major")   # the findings the nightly fix step takes on
LOG_TRIAGE_HOURS = 24                    # how far back the log triage reads
REPORT_TITLES = dict(zip(NIGHTLY_STEPS, ("Test Generation", "Bug Analysis", "Log Triage", "Bug Fixes", "Test Health",
                                         "Auto-Merge Test PRs", "Open Pins")))   # the headings of the night's email
# The order of the email's sections: the findings first, what was done about them, then the tests, and last the
# pins — every strict xfail of the suite, the bugs waiting for the maintainer's verdict (fix, or `wontfix`).
# The steps run in NIGHTLY_STEPS order (the fix step reads the three reports before it), the email reads better this way.
REPORT_ORDER = ("bugs", "logs", "fix", "tests", "health", "merge", "pins")
# The pattern of a pinned bug in tests/unit: the open-pins step runs Claude Code only when the tree holds one.
STRICT_XFAIL_PATTERN = r"pytest\.mark\.xfail\(\s*strict\s*=\s*True"
GH_NOT_INSTALLED_MESSAGE = (
    "The GitHub CLI (gh) is not installed on the server machine, so Keep cannot open a pull request. "
    "Install it with `brew install gh` and run `gh auth login` in a terminal on that machine — any GitHub "
    "account will do — then try again."
)
GH_NOT_LOGGED_IN_MESSAGE = (
    "The GitHub CLI (gh) is not logged in on the server machine, so Keep cannot open a pull request. "
    "Run `gh auth login` in a terminal on that machine — any GitHub account will do — then try again."
)
PUSH_PERMISSIONS = ("ADMIN", "MAINTAIN", "WRITE")   # GitHub's viewerPermission values that may push a branch
ISSUE_DRAFT_DIR = LOG_DIR / "issue_drafts"           # report_issue's drafts when gh cannot file them
# The longest prefilled new-issue link sent to the chat. Telegram documents no limit for a link's
# address, so a link stays the size browsers and chat apps all take; a longer draft is linked with
# its title alone and the description is copied from the message.
PREFILL_URL_LIMIT = 2000
# The labels the filed PRs and issues carry, for triage on GitHub (created once in the repository).
INTEGRATION_PR_LABEL = "integration-pr"
SELF_IMPROVEMENT_LABEL = "self-improvement"
NIGHTLY_FIX_LABEL = "nightly-fix"
ISSUE_LABEL = "reported-by-keep"
# What a new integration adds outside its own folder: the registry lines "Adding an integration" in
# CLAUDE.md names. A PR from improve_integration may add to these files, never remove from them.
INTEGRATION_FOLDERS = "src/integrations/"
INTEGRATION_TESTS = "tests/unit/integrations/"
REGISTRY_FILES = frozenset({
    "docs/atlas/descriptions.py",            # the GROUPS blurb
    "src/assistant/commands/Toolkits.py",    # INDEX_ORDER
    "src/assistant/prompt/PromptManager.py", # SECTION_ORDER
    "requirements.txt",                      # the -r line
    "config.example.yaml",                   # the `# >> section:` pointer line
    "readme.md",                             # the integration's row
})
_nightly_lock = threading.Lock()   # the scheduled run and //nightly work in the same checkout


@dataclass(frozen=True)
class JobPullRequest:
    """The pull request a PR job opened, found among the repository's open PRs after the run
    (`CodeQualityMaintainer._job_pull_request`) — never read off the report's text: on
    2026-10-26 a report that echoed "PR #120" from the request had the scope check close
    the developers' PR #120 and delete its branch, while the job had opened nothing."""
    number: int
    url: str
    head: str
    author: str = ""


@dataclass(frozen=True)
class GithubAccess:
    """The account gh is logged in as on this machine and its permission on the repository the
    checkout's origin is. An account that cannot push opens its PRs from a fork of its own."""
    login: str
    repo: str          # owner/name
    permission: str    # GitHub's viewerPermission: ADMIN, MAINTAIN, WRITE, TRIAGE or READ

    @property
    def can_push(self) -> bool:
        return self.permission in PUSH_PERMISSIONS

    @property
    def can_label(self) -> bool:
        return self.can_push or self.permission == "TRIAGE"


@dataclass(frozen=True)
class IssueDraft:
    """An issue report_issue wrote but could not file — no gh login on this machine — for the
    user to file by hand: a new issue (*comment_on* None) or a comment on an open one."""
    repo: str
    title: str
    body: str
    comment_on: Optional[int] = None
    why: str = "there is no GitHub login on this machine"

    def link(self) -> tuple[str, bool]:
        """Where the user files it, and whether the link carries the description too. A comment
        cannot be prefilled: the link is the issue, its comment box at the bottom."""
        if self.comment_on is not None:
            return f"https://github.com/{self.repo}/issues/{self.comment_on}", False
        base = f"https://github.com/{self.repo}/issues/new?"
        full = base + urlencode({"title": self.title, "body": self.body, "labels": ISSUE_LABEL})
        if len(full) <= PREFILL_URL_LIMIT:
            return full, True
        trace(f"IssueDraft: the prefilled link would be {len(full)} characters, linking the title alone")
        return base + urlencode({"title": self.title, "labels": ISSUE_LABEL}), False


class ClaudeUnavailableError(RuntimeError):
    """The `claude` CLI on the server cannot run a job; the message says what to do about it.
    `stops_the_night`: whether every later Claude Code run of the night would fail the same way."""
    stops_the_night = True


class ClaudeNotLoggedInError(ClaudeUnavailableError):
    """The `claude` CLI on the server has neither a login nor an API key to stand in for one;
    nothing it is asked to do can succeed."""
    def __init__(self):
        super().__init__(NOT_LOGGED_IN_MESSAGE)


class ClaudeNotInstalledError(ClaudeUnavailableError):
    """No `claude` binary on the PATH or at the installer's default location."""
    def __init__(self):
        super().__init__(NOT_INSTALLED_MESSAGE)


class ClaudeRunFailedError(ClaudeUnavailableError):
    """The `claude` CLI flagged a run as failed and answered with the reason (a rate limit, the
    turn cap) in place of a report — the step produced nothing; the next one may still run."""
    stops_the_night = False

    def __init__(self, label: str, reason: str):
        super().__init__(f"Claude Code did not finish the {label} run: {reason}")


class ClaudeUsageLimitError(ClaudeRunFailedError):
    """The subscription's credits or spend limit are used up: no later run tonight can do better."""
    stops_the_night = True


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
        env = {**os.environ, API_KEY_ENV: claude_api_key() or ""}
        workspace = claude_workspace_id()
        if workspace:
            env[CUSTOM_HEADERS_ENV] = f"{WORKSPACE_HEADER}: {workspace}"
        return env


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

Do not fix the code — your pull request changes files under tests/ only; it is merged right after you are done when it
touches nothing else, and a fix step runs after you with your report. If a test you write fails because the code is
wrong, keep the test and mark it `@pytest.mark.xfail(strict=True, reason="<the bug>")`: the suite stays green and the
bug is pinned until the fix removes the marker. A test marked `@pytest.mark.wontfix` is a bug the maintainer has
looked at and decided to leave: do not pin that bug again, and never add that marker yourself.
Then create a pull request and push it (check rules for commits in the /commit skill).

IMPORTANT: The user will only see your final message so keep the entire analysis in your last message. Its last table is
the fix list: the bugs your tests pinned, one row each, with the columns Severity (critical/major/minor/nit), Where
(file:line), What (name the xfail test) and Proposed fix. An automated fix step reads this table and takes on the
critical and major rows. When no test pinned a bug, say so plainly and leave the table empty.
"""

BUG_ANALYSIS_PROMPT = """
Run the /check skill with the target "last 24h" to review the last day's commits for bugs.

IMPORTANT: Your entire analysis must be in your final message. Do not ask follow-up questions or offer to implement fixes.
"""

LOG_TRIAGE_PROMPT = """
Go through the server's logs of the last {hours} hours for problems that actually happened while it ran, and judge
each one: is it serious, and has a commit already fixed it?

The logs are under log/, huge and appended to across runs — never read one whole. Narrow to the window first by
the timestamp that starts every line (`grep -n -m1 '^<date> <hour>'` finds the first line of the window, then
tail from there), then grep inside it:
- log/floxy.log — the plain log, `<timestamp> +<delta> [LEVEL] message`; levels ERROR, NOTE, INFO, FINE, TRACE.
  Start with the [ERROR] lines, then the lines that say something failed, retried, timed out, stalled, was skipped
  or restarted.
- log/verbose-floxy.out — ALL logs including fine(), formatted with colors codes that you'll need to strip.
- log/raw-floxy.log — the process's own stderr: the tracebacks of crashes and of exceptions nothing caught.
- log/restart.log — when and why the server restarted.
A line `CodeQualityMaintainer: this process runs commit <sha>` marks a start of the server on that commit, so you
can tell which code was running when a problem occurred.

For each problem you find:
1. Has a commit already fixed it? List the commits since its first occurrence (`git log --since=... --format='%h
   %ad %s' --date=iso`) and read the diffs of those that touch the code around it. A commit that removes the cause
   is a fix only when the problem did not come back on a server start that carried that commit (the "runs commit"
   markers after it say which starts did). Name the fixing commit and leave the problem out of the fix list.
2. Otherwise find the cause: read the code the log lines come from, cite file:line, separate what is a fact from
   what is a guess, and propose the fix when it is obvious.

Severity: critical — a crash, data loss, a user's request lost or answered wrong, a security or privacy hole;
major — a feature does not work or an integration is down; minor — degraded, noisy or slow but working; nit —
cosmetic.

Privacy: quote log lines trimmed to what matters, with a phone number, an email address, a name, an address, a
token or a key replaced by a placeholder like <name> — the fix list ends up in a pull request on GitHub.

Final report, concise (it is read over a chat on a phone):
- One short section per problem: what, when and how often, the code it comes from, "fixed by <commit>" or the
  cause.
- The last table is the fix list: the unfixed problems only, one row each, with the columns Severity
  (critical/major/minor/nit), Where (file:line), What and Proposed fix. An automated fix step reads this table and
  takes on the critical and major rows. When nothing needs fixing, say so plainly and leave the table empty.

IMPORTANT: Your entire analysis must be in your final message. Do not ask follow-up questions or offer to implement fixes.
"""

FIX_BUGS_PROMPT = """
The nightly reviews of this code base just finished — the bug analysis of the day's commits, the triage of the
server's logs and the generation of the tests the day's commits lacked, whichever ran tonight; their reports are
below, each under its own heading. Your job is to fix the items of each report's fix list (its last table, what the
report itself says needs fixing) whose finding is tagged Critical or Major — and only those. Leave every Minor and
Nit item alone. A bug the test generation pinned has a strict-xfail test already on master: fixing it means making
the code right and removing the marker, so that test runs green. A test marked `@pytest.mark.wontfix` is a bug the
maintainer has decided to leave: do not touch it, and never add that marker yourself — leaving a bug is the
maintainer's call, not yours.

Steps:
1. Prepare for PR: create a branch.
2. Fix each Critical and Major item of the fix list. Read the cited code first; if an item cannot be reproduced,
   or its fix is too risky or too large for an automated PR, leave it and say why in the report.
3. If applicable create new unit tests.
4. Run all unit tests to make sure nothing is broken.
5. Commit (check rules for commits in the /commit skill) and create one PR and push it. Give the PR the label
   `{label}` and end its description with the line `Opened by Keep {version} (nightly fix)`.

If nothing was fixed, open no PR and say so.

IMPORTANT: Your entire response to the user must be in your final message: the PR link, a table of the items you fixed and
a table of the items you left with the reason for each. Keep it concise — it will be read over a chat on a phone.

The reports:

{analysis}
"""

SELF_ANALYSIS_PROMPT = """
I want you to analyze something for me. It may be about this code base (a bug report, how a feature works or
why it behaved as it did, a review, a feasibility check) or about the machine the server runs on (the home
network, a device on it, disk space, updates, a service):

{request}

For the code: log/floxy.out has the server logs (huge — tail it or grep for the time in question).
For the machine: use the shell, but only observe — change, install, restart or write nothing outside the repo,
this install's own integration folders outside the checkout included (a change there is a job for
improve_local_integration, not for this analysis: say so in the report instead of making it).
Never put config.yaml, vault or ~/.claude contents in the report.

Do a thorough analysis and provide a report with what you found.

IMPORTANT: Your entire analysis must be in your final message. Do not ask follow-up questions or offer to implement changes. Keep the report concise — it will be read over a chat on a phone.
"""

LOCAL_INTEGRATION_PROMPT = """
This is a request for work on this install's own local integrations: integration folders that live outside this
checkout, in the directories config.yaml's `extra_integration_dirs` lists. A local folder is laid out like one of
`src/integrations/` (the same pieces: `<Name>Integration.py`, `<Name>Commands.py`, `toolkit.md`,
`config.example.yaml`, `requirements.txt` …) and the server imports it as `integrations.<folder>`, so what CLAUDE.md
says about an integration folder applies to it — but the folders are the user's own, not part of this repository:
no branch and no pull request is made for them, the user reviews the result in place.

The local directories: {roots}
{target}

The request:

{request}

Rules:
- Write only under the local directories above — never in this checkout, never anywhere else on the machine.
- The directories are the user's own, so git on them is part of the job when the request asks for it: init a
  repository, commit, add the remote the user names, push to it. Never push anywhere the user did not name.
- A change that needs anything in this checkout (the core, a contract, one of the repository's integrations) cannot
  be made here: make no change, and say in the report exactly what is needed, so the user can file it with
  report_issue.
- Read this checkout freely: the integration framework under src/assistant/integration/, the repository's folders
  as examples, tests/unit/integrations/ as the pattern for a local folder's own tests (which stay in the folder),
  log/floxy.out for what happened (huge — tail it or grep for the time in question).
- Do not restart the server or install anything: the server keeps running the old code until the user restarts it.
- Never put config.yaml, vault or ~/.claude contents in the report.

IMPORTANT: Your entire report must be in your final message: what you changed (files, commits, pushes) and what you
left undone with the reason. Do not ask follow-up questions. Keep it concise — it will be read over a chat on a phone.
"""

COST_ANALYSIS_PROMPT = """
Run the keep-cost-analysis skill of this repository (the Skill tool) with the arguments: {window}

IMPORTANT: Your entire summary must be in your final message. Do not ask follow-up questions. Keep it concise — it will be read over a chat on a phone.
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
- A test marked `@pytest.mark.wontfix` documents a bug the maintainer has decided to leave; it runs as a
  non-strict xfail. When one passed in every run, the bug it documents is gone: remove the marker so the test
  runs as an ordinary one, and list it in the report. Never add that marker yourself.
- Verify: measure every test you touched {runs} more times, then run the whole suite once.

If you changed anything, create a branch, commit (check rules for commits in the /commit skill), push it and
create ONE pull request with all of it. If nothing was flaky or slow, change nothing and say so.

Final report:
- Flaky tests: name, cause, fix
- Slow tests made faster: name, before -> after
- wontfix markers removed: every test by name
- Tests REMOVED: every one by name, with why it was not worth keeping
- Bugs in the code under test that a flaky test exposed: the test, the file and line, what goes wrong
- Anything left undone (broken tests, flaky or slow ones you could not fix), and the pull request

IMPORTANT: The user will only see your final message so keep the entire report in your last message.
"""

OPEN_PINS_PROMPT = """
List the bugs pinned in the unit suite. A pin is a test under tests/unit marked `@pytest.mark.xfail(strict=True, ...)`:
the test generation wrote it, it fails because the code is wrong, and it stays until the bug is fixed or the maintainer
decides to leave the bug. A test marked `@pytest.mark.wontfix` is such a decision, already taken: leave those out.

For each pin, read the test and the code it pins and rate the fix: trivial (a line), small (one function, no design
change), medium (a few places, or new tests), large (a design change, or a risk to the normal code paths).

Change nothing: no edits, no branch, no PR.

IMPORTANT: The user will only see your final message. Make it the table alone, one row per pin, numbered from 1, with
the columns #, Test (the test function's name alone — no file, no class, no backticks: the path makes the table
too wide to read), Reason (the marker's reason in one line), Fix complexity. The maintainer answers it with
"fix 1, 3; wontfix the rest", so the numbers matter. When there is no pin, say so in one line.
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
5. Commit (check rules for commits in the /commit skill) and create a PR and push it. Give the PR the label
   `{label}` and end its description with the line `Opened by Keep {version} (self_improvement)`.

Log files from running system are available at:
log/floxy.out (Floxy server logs, be mindful, it's huge, so tail)

NOTE: If the request is too vague to act on or the change is too complex/risky for an automated PR, decline and explain why.

IMPORTANT: Your entire response to the user must be in your final message. Keep the report concise — it will be read over a chat on a phone.
"""

# The paragraph both improve_integration prompts share: what the PR may touch, since this is a
# user's install and the developers review what comes in.
INTEGRATION_SCOPE = """
This request comes from a user's install, not a developer's: the pull request is reviewed by the developers, so it
must stay inside its own integration. It may change only `src/integrations/{folder}/` and
`tests/unit/integrations/{folder}/`{registry}. The scope is checked after your run against the files the PR
touches, and a PR that strays is closed unseen — so if the change needs anything else (the core, another
integration, a contract), make NO change: say so in your report and describe exactly what is needed, so the user
can file it with report_issue instead.
"""

IMPROVE_INTEGRATION_PROMPT = """
This is a request for an automated change to the "{name}" integration of this code base, delivered as a pull
request. It is usually a bug fix, but it can also be a small feature or improvement of that integration.

{request}
{scope}
Steps:
1. Prepare for PR: create a branch.
2. Implement the change.
3. If applicable create new unit tests.
4. Run all unit tests to make sure nothing is broken.
5. Commit (check rules for commits in the /commit skill) and create a PR and push it. Title it `{name}: ...`,
   give it the label `{label}` and end its description with the line
   `Opened by Keep {version} (improve_integration, {name})`.

Log files from running system are available at:
log/floxy.out (Floxy server logs, be mindful, it's huge, so tail)

NOTE: If the request is too vague to act on or the change is too complex/risky for an automated PR, decline and explain why.

IMPORTANT: Your entire response to the user must be in your final message. Keep the report concise — it will be read over a chat on a phone.
"""

NEW_INTEGRATION_PROMPT = """
This is a request for a new integration of this code base, "{name}", delivered as a pull request. No integration
of that name exists; if one for the same service already does under src/integrations/ (under another name),
build nothing and say which it is.

{request}
{scope}
Steps:
1. Prepare for PR: create a branch.
2. Build the integration as "Adding an integration" in CLAUDE.md describes: a new folder under src/integrations/
   with its Integration class, commands, prompt section, toolkit.md, config.example.yaml and requirements.txt
   as it needs them, and the registry lines the recipe names, each added without removing anything.
3. Create its unit tests under tests/unit/integrations/<folder>/ (fictional test data, see CLAUDE.md).
4. Run all unit tests to make sure nothing is broken.
5. Commit (check rules for commits in the /commit skill) and create a PR and push it. Title it `{name}: ...`,
   give it the label `{label}` and end its description with the line
   `Opened by Keep {version} (improve_integration, {name})`.

NOTE: If the request is too vague to act on (which service, which API, what it should do) or the integration is too
complex/risky for an automated PR, decline and explain why.

IMPORTANT: Your entire response to the user must be in your final message. Keep the report concise — it will be read over a chat on a phone.
"""

REPORT_ISSUE_PROMPT = """
A user of this system wants to report an issue to its developers. Investigate it thoroughly and {filing}, written so
the developers understand it at once and can fix it without asking back:

{request}

Investigate first. log/floxy.out has the server logs (huge — grep for the time in question, then read around it);
the code is in src/. Reconstruct what happened before writing a word.

{duplicates}

The issue (markdown):
- Title: precise, one line, the symptom not the guess.
- What happened, in the user's words, and when (date and time).
- Timeline: what the system did, step by step, from the log — short quoted excerpts with their timestamps,
  trimmed to the lines that matter. Include the commands that ran and the replies around the problem.
- Suspected cause, with file and line references, and what is a fact versus a guess. A fix proposal when it is
  obvious.
- How to reproduce, when you can tell.
- Environment: Keep {version}, the OS and its version, the integrations enabled, the channel the user spoke
  through (Telegram, voice, console, HTTP), the LLM model in use (its name only).
- Last line: `Reported by Keep {version} (report_issue)`.
{label}

Privacy — the issue leaves this machine: never put private information in it. Quote log lines only after replacing
private values with a placeholder like <phone> or <name>.

{final}

IMPORTANT: Your entire response must be in your final message. Do not ask follow-up questions. Keep it concise — it will be read over a chat on a phone.
"""

# The parts of REPORT_ISSUE_PROMPT that depend on whether gh can file the issue on this machine.
ISSUE_FILED = dict(
    filing="file it as a GitHub issue on this repository with `gh issue create`",
    duplicates="""Then search the open issues (`gh issue list --state open --search "..."`, a few keyword variants) for one about the
same thing. If there is one, add what you found as a comment on it (`gh issue comment`) instead of opening a
duplicate, and say so in your report.""",
    final="""Your final message: the issue's URL on its own line, then one line on what it says — or, when you filed nothing
(the report was too vague to investigate, or filing failed), why.""",
)
ISSUE_DRAFTED = dict(
    filing="draft it as a GitHub issue on {repo} — this machine has no GitHub login, so the user files it by hand",
    duplicates="""Then search the open issues of {repo} through GitHub's public API for one about the same thing, a few keyword
variants: `curl -s "https://api.github.com/search/issues?q=repo:{repo}+is:issue+is:open+<keywords>"`. If there
is one, draft what you found as a comment on it instead of a duplicate issue.""",
    final="""Write the draft to {path}: its first line `Title: <the title>` for a new issue, or `Comment on: #<number>` for
a comment on an open one, then an empty line, then the markdown body. Do not file anything on GitHub.

Your final message: one line on what the draft says — or, when you drafted nothing (the report was too vague to
investigate), why.""",
)
_DRAFT_HEAD = re.compile(r"^\s*(?:Title:\s*(?P<title>.+?)|Comment on:\s*#?(?P<number>\d+))\s*$", re.IGNORECASE)


class CodeQualityMaintainer:
    """Runs the nightly Claude Code jobs on the repo, one after another: bug analysis of the day's
    commits, triage of the server's logs (what actually went wrong, and whether a commit fixed it
    since), a PR fixing the Critical and Major items of both, test generation, test health (flaky
    and slow tests), the auto-merge of test-only PRs, then the open pins — every bug a strict-xfail
    test holds, with its fix complexity, for the maintainer to fix or mark `wontfix` — one email for
    the night.

    A daemon thread fires the night at RUN_HOUR. First the nightly update pulls the branch
    `developer_release` names and restarts the server if the code changed (`nightly_update`, on
    by default); the new process boots inside the same hour and takes the night over — its own
    update finds nothing new and the run follows. Then the run, while `self_improvement.nightly_runs`
    is on (off by default, read at each fire); a night without recent commits skips the steps about
    the commits (bug analysis, test generation, test health) and keeps the log triage, the fix,
    the auto-merge and the open pins. What the run itself merges reaches the server at the next night's update or a
    restart_server. Also
    exposes the ad-hoc analyze_code / improve_code / restart helpers behind the self_analysis /
    self_improvement commands.
    """
    def __init__(self):
        self._running = False
        self._thread = None
        self._last_run_date = None
        self._boot_commit = ""   # HEAD when this process started; the nightly update restarts once it differs
        self._merged_tonight: list[str] = []   # the report lines of every PR the night merged, in order

    def start(self):
        self._boot_commit = self._head_commit()
        info(f"CodeQualityMaintainer: this process runs commit {self._boot_commit[:8] or 'unknown'}")
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
                    if self._try_nightly_update():
                        info("CodeQualityMaintainer: the nightly update is restarting the server, "
                             "the new process runs the night")
                        return
                    self._execute_nightly_run()
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

    @staticmethod
    def trial_skip_reason() -> Optional[str]:
        """Why no nightly code-quality run happens while a PR is tried, //nightly included: the
        checkout holds the PR's code, so the run would analyse that and branch its PRs off it.
        The nightly update is not held back — it is what ends the trial."""
        from assistant.agents.PullRequestTrial import PullRequestTrial
        trial = PullRequestTrial.current()
        if trial is None:
            return None
        return f"the server is trying PR #{trial.number}, the nightly run waits until the trial is over"

    def _execute_nightly_run(self, force=False, only=None):
        """The whole night, or with `only` (one of NIGHTLY_STEPS) that step alone. `force` is
        //nightly: it runs with `self_improvement.nightly_runs` off, never with the integration off."""
        reason = (self.unavailable_reason() if force else self.nightly_skip_reason()) or self.trial_skip_reason()
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
        self._pull_current()
        self._merged_tonight = []
        reports = {}
        steps = dict(zip(NIGHTLY_STEPS, [self._step_generate_tests,
                                         self._step_bug_analysis,
                                         self._step_log_triage,
                                         lambda: self._step_fix_bugs(self.fix_input(reports)),
                                         self._step_test_health,
                                         self._step_auto_merge,
                                         self._step_open_pins]))
        if only == "fix":
            info("CodeQualityMaintainer: the fix step reads the bug analysis and the log triage, so those run first "
                 "(the test generation, whose report it also reads, is a step of its own).")
            keys = ["bugs", "logs", "fix"]
        elif only:
            keys = [only]
        elif self._has_recent_commits():
            keys = list(steps)
        else:
            info("CodeQualityMaintainer: no recent commits — the log triage, the fix step, the auto-merge and the "
                 "open pins run tonight, the steps about the commits are skipped.")
            keys = ["logs", "fix", "merge", "pins"]
        sections = []
        claude_down: ClaudeUnavailableError | None = None
        failed_at = None
        for key in keys:
            title, step = REPORT_TITLES[key], steps[key]
            if claude_down and key != "merge":
                info(f"CodeQualityMaintainer: {title} skipped — Claude Code failed at the {failed_at}.")
                reports[key] = f"Not run: Claude Code failed at the {failed_at}. {claude_down}"
                sections.append((key, reports[key]))
                continue
            try:
                report = step()
                reports[key] = report
                if report:
                    sections.append((key, report))
                else:
                    info(f"CodeQualityMaintainer: {title} produced no report.")
            except ClaudeUnavailableError as e:
                error(f"CodeQualityMaintainer: {title} did not run — {e}")
                reports[key] = f"**The {title.lower()} did not run.** {e}"
                sections.append((key, reports[key]))
                if e.stops_the_night:
                    claude_down, failed_at = e, title.lower()
            except Exception as e:
                error(f"CodeQualityMaintainer: {title} exception: {e}")
        if sections:
            sections.sort(key=lambda section: REPORT_ORDER.index(section[0]))
            combined_md = "\n\n---\n\n".join(
                f"# {REPORT_TITLES[key]}\n\n{body}" for key, body in sections
            )
            html_report = self._markdown_to_html(combined_md)
            self._send_email("Keep Nightly Code Quality Report", html_report)
        else:
            info("CodeQualityMaintainer: nothing to report, skipping email.")
        info("CodeQualityMaintainer: nightly run complete.")

    @staticmethod
    def fix_input(reports: dict) -> str:
        """What the fix step reads: the night's bug analysis, log triage and test generation, each
        under the heading the email gives it, the ones that produced nothing left out — "" when
        none did."""
        present = [(REPORT_TITLES[key], reports.get(key, "")) for key in FIX_STEP_INPUTS if reports.get(key)]
        if not present:
            trace("CodeQualityMaintainer: no report of the night is there for the fix step to read")
        return "\n\n".join(f"# {title}\n\n{body}" for title, body in present)

    def _try_nightly_update(self) -> bool:
        """`_nightly_update`, where a raise (a git call past GIT_TIMEOUT) is a failed update, not
        a lost night: `_run` has marked the date by now, so the night must still run here."""
        try:
            return self._nightly_update()
        except Exception as e:
            error(f"CodeQualityMaintainer: the nightly update raised, the night runs on the old code — {e!r}")
            return False

    def _nightly_update(self) -> bool:
        """The first thing of the night, at RUN_HOUR: check out the branch `developer_release`
        names, pull it, and restart when the code is no longer what this process booted on —
        commits pulled now or made locally since the start. No new commit, no restart. True when
        the restart is under way, so the caller leaves the night to the new process.
        `nightly_update: false` turns it off; it is read at each fire."""
        from assistant.SystemNote import system_note
        if not ConfigManager().get(NIGHTLY_UPDATE_KEY, True):
            info(f"CodeQualityMaintainer: {NIGHTLY_UPDATE_KEY} is off, no nightly update.")
            return False
        if DevelopmentConfig.LOCAL_DEVELOPMENT_MODE:
            info("CodeQualityMaintainer: LOCAL_DEVELOPMENT_MODE is on, no nightly update.")
            return False
        if not self._boot_commit:
            info("CodeQualityMaintainer: the commit this process started on is unknown, no nightly update.")
            return False
        if not _nightly_lock.acquire(blocking=False):
            info("CodeQualityMaintainer: a nightly run is still going, no nightly update under it.")
            return False
        try:
            target = self.target_branch()
            info(f"CodeQualityMaintainer: nightly update — the target is {target}.")
            update = self._updater().update(target, self._boot_commit)
            if update.pull_failure:
                system_note("server", "Nightly update failed, the server keeps running the old code: "
                                      f"{update.pull_failure}", push=False)
                return False
            if update.sync_failure:
                system_note("server", f"Nightly update pulled {update.change} on {target} but the server keeps "
                                      f"running the old code: {update.sync_failure}", push=False)
                return False
            if not update.restart:
                return False
            span = f"{self._boot_commit[:8]} → {update.head[:8]}"
            note(f"CodeQualityMaintainer: nightly update — {update.change} on {target} ({span}), restarting.")
            # History only: the startup announcement of the new process is what the chats get.
            system_note("server", f"Nightly update: {update.change} on {target}, the server is restarting.", push=False)
            self.restart(f"nightly update {span}")
            return True
        finally:
            _nightly_lock.release()

    @staticmethod
    def _updater() -> Updater:
        """The update machinery of `common/Updater`, running its git commands through
        `_run_subprocess` like every other git and gh call of the maintainer."""
        return Updater(run=CodeQualityMaintainer._run_subprocess)

    @staticmethod
    def _head_commit() -> str:
        """The commit the checkout is on, "" when git cannot say."""
        return CodeQualityMaintainer._updater().head_commit()

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
    def _pr_files(number: int) -> tuple[str, list[dict]]:
        """A PR's title and every file it touches as the REST file list gives them: `filename`,
        `previous_filename` for a renamed one (`gh pr view` only knows the new path, hence the
        REST list), `status` (added, modified, removed, renamed) and `deletions`. Raises when gh
        fails or hands back fewer files than the PR changes, since an unseen path cannot be
        vouched for."""
        def gh(*args: str) -> str:
            result = subprocess.run(["gh", *args], cwd=WORKING_DIR, capture_output=True, text=True,
                                    timeout=GIT_TIMEOUT)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or f"gh {args[0]} exited {result.returncode}")
            return result.stdout

        pr = json.loads(gh("pr", "view", str(number), "--json", "title,changedFiles"))
        files = [json.loads(line) for line in
                 gh("api", f"repos/{{owner}}/{{repo}}/pulls/{number}/files", "--paginate",
                    "--jq", ".[] | {filename, previous_filename, status, deletions}").splitlines() if line.strip()]
        if len(files) != pr.get("changedFiles"):
            raise RuntimeError(f"gh listed {len(files)} of {pr.get('changedFiles')} changed files")
        for f in files:
            if f.get("previous_filename"):
                trace(f"CodeQualityMaintainer: PR #{number} moves {f['previous_filename']} to {f['filename']}")
        return pr.get("title", ""), files

    @staticmethod
    def _pr_title_and_files(number: int) -> tuple[str, list[str]]:
        """A PR's title and every path it touches — for a renamed file both the new and the
        previous path, so moving src/x.py under tests/ still shows as a change outside tests/."""
        title, files = CodeQualityMaintainer._pr_files(number)
        paths = []
        for f in files:
            paths.append(f["filename"])
            if f.get("previous_filename"):
                paths.append(f["previous_filename"])
        return title, paths

    def _step_auto_merge(self) -> str:
        """Last step of the night, so the test-only PRs the earlier steps opened are merged with it.
        Plain code, no Claude Code run: a PR qualifies when every path it touches is under tests/.
        The report lists every PR the night merged, the test generation's own included."""
        info("CodeQualityMaintainer: auto-merge of test-only PRs...")
        merged, skipped = self._merge_test_prs()
        if merged is None and not self._merged_tonight:
            return ""
        return "\n\n".join(f"## {heading}\n\n" + ("\n".join(lines) or "None.")
                           for heading, lines in (("Merged PRs", self._merged_tonight), ("Skipped PRs", skipped)))

    def _merge_test_prs(self) -> tuple[list[str] | None, list[str]]:
        """Merges every open PR of a trusted author that touches nothing outside tests/ and returns
        the report lines of the merged and the skipped ones — (None, []) when no PR was there to
        look at. The test generation step calls it for the PR it just opened, the auto-merge step
        for whatever is left at the end of the night."""
        candidates = self._auto_merge_candidates()
        if not candidates:
            info("CodeQualityMaintainer: no open PRs from trusted authors, skipping the auto-merge.")
            return None, []
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
                    self._merged_tonight.append(merged[-1])
        finally:
            self._run_subprocess(["git", "checkout", branch], "git checkout", GIT_TIMEOUT)
        info(f"CodeQualityMaintainer: auto-merge complete, {len(merged)} merged, {len(skipped)} skipped.")
        return merged, skipped

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
                     f"{obj['duration_ms'] / 1000:.0f} s" if obj.get("duration_ms") else "",
                     "flagged as failed" if obj.get("is_error") else ""]
            facts_str = ", ".join(f for f in facts if f)
            return f"RESULT ({facts_str}): {preview}" if facts_str else f"RESULT: {preview}"
        return None

    @staticmethod
    def _run_claude(prompt, label, add_dirs: tuple = ()):
        """One `claude -p` run in the checkout. *add_dirs* are directories outside it the run
        may read and write too (`--add-dir`): the local integration folders of a job on
        them — the checkout stays the working directory, so CLAUDE.md is in force."""
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
        if add_dirs:
            cmd += ["--add-dir", *map(str, add_dirs)]
            trace(f"CodeQualityMaintainer [{label}] Claude Code may also write in {', '.join(map(str, add_dirs))}")
        process = subprocess.Popen(
            cmd, cwd=WORKING_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL, text=True, env=status.env()
        )
        timed_out = threading.Event()

        def kill():
            timed_out.set()
            process.kill()
        timer = threading.Timer(CLAUDE_TIMEOUT, kill)
        timer.start()
        # Read after stdout, a stderr past the pipe buffer would block the CLI forever
        stderr_chunks = []
        stderr_reader = threading.Thread(target=lambda: stderr_chunks.append(process.stderr.read()),
                                         name=f"claude-stderr {label}", daemon=True)
        stderr_reader.start()
        result_text = ""
        result_is_error = False
        last_assistant_text = ""
        stderr = ""
        cost_usd = None
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
                        cost_usd = obj.get("total_cost_usd") or obj.get("cost_usd")
                except json.JSONDecodeError:
                    trace(f"CodeQualityMaintainer [{label}] {line}")
            process.wait()
            stderr_reader.join()
            stderr = "".join(chunk or "" for chunk in stderr_chunks)
            if stderr and stderr.strip():
                trace(f"CodeQualityMaintainer [{label}] stderr: {stderr.strip()}")
        finally:
            timer.cancel()
        output = result_text or last_assistant_text
        if timed_out.is_set():
            # What it wrote before the kill is a report cut short, not the run's report
            reason = f"timed out after {CLAUDE_TIMEOUT // 60} minutes and was killed"
            error(f"CodeQualityMaintainer [{label}] Claude Code {reason}; last text: {output.strip()[:200]}")
            raise ClaudeRunFailedError(label, reason)
        # A successful report may legitimately discuss login errors in the code under review, so
        # the result text only counts when the CLI itself flagged the run as failed.
        if (result_is_error and CodeQualityMaintainer._is_not_logged_in(output)) \
                or CodeQualityMaintainer._is_not_logged_in(stderr):
            error(f"CodeQualityMaintainer [{label}] Claude Code is not logged in on this machine: {output.strip() or stderr.strip()}")
            raise ClaudeNotLoggedInError()
        # A flagged result is the CLI's reason, not a report: taken as one, the out-of-credits line
        # was the night's bug analysis and the fix step read "nothing Critical or Major" in it.
        if result_is_error or (len(result_text) <= USAGE_LIMIT_MESSAGE_MAX_CHARS
                               and CodeQualityMaintainer._is_usage_limit(result_text)):
            reason = (output or stderr).strip() or "no reason given"
            if CodeQualityMaintainer._is_usage_limit(reason):
                error(f"CodeQualityMaintainer [{label}] Claude Code is out of usage credits — no run tonight can "
                      f"do better: {reason}")
                raise ClaudeUsageLimitError(label, reason)
            error(f"CodeQualityMaintainer [{label}] Claude Code flagged the run as failed: {reason}")
            raise ClaudeRunFailedError(label, reason)
        if output and status.api_key:
            info(f"CodeQualityMaintainer [{label}] the run was billed to {API_KEY_CONFIG_KEY}"
                 f"{f' (${cost_usd:.2f})' if cost_usd else ''}: the report says so")
            output += api_key_cost_note(cost_usd)
        return output

    @staticmethod
    def _is_not_logged_in(text) -> bool:
        lowered = (text or "").lower()
        return any(marker in lowered for marker in NOT_LOGGED_IN_MARKERS)

    @staticmethod
    def _is_usage_limit(text) -> bool:
        lowered = (text or "").lower()
        return any(marker in lowered for marker in USAGE_LIMIT_MARKERS)

    def _step_generate_tests(self) -> str:
        """First step of the night: the tests the last days' commits lack, as a tests-only PR
        that is merged on the spot, so the fix step after it starts from a master that holds
        them. A bug the new tests expose is pinned with a strict xfail and listed in the
        report's fix list, which the fix step reads; the step itself never touches the code."""
        branch = CodeQualityMaintainer._current_branch()
        try:
            info("CodeQualityMaintainer: generating missing tests for recent commits...")
            trace("Starting Claude Code in the background...")
            report = CodeQualityMaintainer._run_claude(GENERATE_TESTS_PROMPT, "generate-tests")
            info("CodeQualityMaintainer: test generation complete.")
        except ClaudeUnavailableError:
            raise   # the night reports it and knows whether to go on
        except Exception as e:
            error(f"CodeQualityMaintainer: test generation failed: {e}")
            return ""
        finally:
            CodeQualityMaintainer._run_subprocess(["git", "checkout", branch], "git checkout", GIT_TIMEOUT)
        info("CodeQualityMaintainer: merging the test PR before the fix step...")
        try:
            merged, _ = self._merge_test_prs()
        except Exception as e:
            # The report holds the fix list the fix step reads; a merge that raises must not lose it
            error(f"CodeQualityMaintainer: merging the test PR failed, the report is kept: {e}")
            merged = None
        if merged:
            report += "\n\n## Merged into master\n\n" + "\n".join(merged)
        else:
            info("CodeQualityMaintainer: no test-only PR to merge after the test generation.")
        return report

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
        except ClaudeUnavailableError:
            raise   # the night reports it and knows whether to go on
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
        except ClaudeUnavailableError:
            raise   # the night reports it and knows whether to go on
        except Exception as e:
            error(f"CodeQualityMaintainer: bug analysis failed: {e}")
            return ""

    @staticmethod
    def _step_log_triage() -> str:
        """The problems the server actually had in the last LOG_TRIAGE_HOURS, from its logs, each
        judged serious or noise and fixed since or still open — its fix list feeds the fix step
        beside the bug analysis'. Runs every night: the logs are about the running system, not
        the day's commits."""
        try:
            info(f"CodeQualityMaintainer: log triage of the last {LOG_TRIAGE_HOURS} hours...")
            report = CodeQualityMaintainer._run_claude(LOG_TRIAGE_PROMPT.format(hours=LOG_TRIAGE_HOURS), "log-triage")
            info("CodeQualityMaintainer: log triage complete.")
            return report
        except ClaudeUnavailableError:
            raise   # the night reports it and knows whether to go on
        except Exception as e:
            error(f"CodeQualityMaintainer: log triage failed: {e}")
            return ""

    @staticmethod
    def pinned_tests() -> int:
        """How many strict xfails the unit tests hold — a cheap screen so a suite without a pin
        costs no Claude Code run. Counts markers, not tests; a `wontfix` is not one."""
        pattern = re.compile(STRICT_XFAIL_PATTERN)
        count = 0
        for path in (_AISERVER_PATH / "tests" / "unit").rglob("test_*.py"):
            try:
                count += len(pattern.findall(path.read_text()))
            except OSError as e:
                trace(f"CodeQualityMaintainer: could not read {path} for pins: {e}")
        return count

    @staticmethod
    def _step_open_pins() -> str:
        """Last step of the night: every bug a strict xfail pins, numbered, with the fix complexity
        Claude Code rates — the whole list every night, however old a pin is, so a list that grows
        says the maintainer has verdicts to give (fix, or mark `wontfix`). Nothing is changed."""
        pins = CodeQualityMaintainer.pinned_tests()
        if not pins:
            info("CodeQualityMaintainer: the unit suite holds no pinned bug, nothing to list.")
            return "No bug is pinned in the unit suite."
        try:
            info(f"CodeQualityMaintainer: open pins — {pins} strict xfail(s) in the unit suite, rating the fixes...")
            report = CodeQualityMaintainer._run_claude(OPEN_PINS_PROMPT, "open-pins")
            info("CodeQualityMaintainer: open pins complete.")
            return report
        except ClaudeUnavailableError:
            raise   # the night reports it and knows whether to go on
        except Exception as e:
            error(f"CodeQualityMaintainer: open pins failed: {e}")
            return ""

    @staticmethod
    def fixable_severities(analysis: str) -> list[str]:
        """The severities of FIX_SEVERITIES the analysis mentions at all — a cheap screen so a
        night without a Critical or Major finding costs no Claude Code run. A report that only
        says "no major issues" passes the screen; the run then fixes nothing and says so."""
        words = set(re.findall(r"[a-z]+", analysis.lower()))
        return [severity for severity in FIX_SEVERITIES if severity in words]

    @staticmethod
    def _step_fix_bugs(analysis: str) -> str:
        """Fixes the Critical and Major items of the fix lists of *analysis* — the bug analysis, the
        log triage and the test generation as `fix_input` joins them — as one PR, through
        `_pr_job`; the auto-merge leaves that PR alone, since it touches more than tests/."""
        if not analysis:
            info("CodeQualityMaintainer: no report of the night to fix from, skipping the fix step.")
            return ""
        severities = CodeQualityMaintainer.fixable_severities(analysis)
        if not severities:
            info("CodeQualityMaintainer: the reports name nothing Critical or Major, nothing to fix.")
            return "The night's reports name nothing Critical or Major — nothing to fix tonight."
        reason = CodeQualityMaintainer._github_unavailable()
        if reason:
            return reason
        info(f"CodeQualityMaintainer: the reports mention {', '.join(severities)} findings, fixing...")
        prompt = FIX_BUGS_PROMPT.format(analysis=analysis, label=NIGHTLY_FIX_LABEL,
                                        version=CodeQualityMaintainer._version())
        report, _ = CodeQualityMaintainer._pr_job(prompt, "nightly-fix")
        return report

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
    def improve_local_integration(request: str, roots: tuple, name: Optional[str] = None,
                                  folder: Optional[str] = None) -> str:
        """Claude Code works on this install's own integration folders outside the checkout
        (*roots*, the resolved `extra_integration_dirs`): on the local integration *name*
        in *folder* (its path), on a new one called *name* (*folder* None — it goes into the
        first directory), or, *name* None, on the directories themselves (a git repository
        for them, say). No PR: the folders are the user's, so the run writes there — the
        directories are passed as `--add-dir`, the checkout stays the working directory —
        and the report is the result. Returns the report."""
        if name is None:
            target = "The work is on the directories themselves, no one integration in particular."
        elif folder is None:
            target = (f'No local integration is called "{name}": build it as a new folder in {roots[0]}, '
                      f"named after its service, with the pieces an integration folder has.")
        else:
            target = f'The work is on the "{name}" integration, the folder {folder}.'
        info(f"CodeQualityMaintainer: starting local integration work on "
             f"{folder or (name and 'a new ' + name + ' folder') or 'the local directories'}...")
        prompt = LOCAL_INTEGRATION_PROMPT.format(roots=", ".join(map(str, roots)), target=target, request=request)
        trace(f"prompt: {prompt}")
        try:
            result = CodeQualityMaintainer._run_claude(prompt, "local-integration", add_dirs=tuple(roots))
        except ClaudeUnavailableError as e:
            return str(e)
        if not result:
            info("CodeQualityMaintainer: the local integration job produced no output")
            return "The local integration job produced no output."
        return result

    @staticmethod
    def analyze_cost(window: str) -> str:
        """Where the LLM spend of the last window went — the keep-cost-analysis skill, run by
        Claude Code on the local log; the summary comes back as the report."""
        info(f"CodeQualityMaintainer: starting cost analysis ({window or 'last 24 hours'})...")
        try:
            result = CodeQualityMaintainer._run_claude(COST_ANALYSIS_PROMPT.format(window=window or "24h"),
                                                       "cost-analysis")
        except ClaudeUnavailableError as e:
            return str(e)
        if not result:
            info("CodeQualityMaintainer: cost analysis produced no output.")
        return result or "Cost analysis produced no output."

    @staticmethod
    def _github_unavailable() -> Optional[str]:
        """Why nothing can be filed on the repository right now, or None."""
        return CodeQualityMaintainer._github_access()[1]

    @staticmethod
    def _github_access() -> tuple[Optional[GithubAccess], Optional[str]]:
        """Who gh is logged in as and what the account may do on the repository — or, when
        nothing can be filed from this machine (gh missing, logged out, or the repository
        unreadable), why. Checked before a run that ends in a PR or an issue, so the user
        hears it at once instead of after an hour of Claude Code."""
        def gh(*args: str) -> subprocess.CompletedProcess:
            return subprocess.run(["gh", *args], cwd=WORKING_DIR, capture_output=True, text=True,
                                  timeout=GIT_TIMEOUT, stdin=subprocess.DEVNULL)
        try:
            status = gh("auth", "status")
        except FileNotFoundError:
            info("CodeQualityMaintainer: gh is not installed")
            return None, GH_NOT_INSTALLED_MESSAGE
        except Exception as e:
            error(f"CodeQualityMaintainer: gh auth status failed: {e}")
            return None, GH_NOT_LOGGED_IN_MESSAGE
        if status.returncode != 0:
            info(f"CodeQualityMaintainer: gh is not logged in: {(status.stderr or status.stdout).strip()}")
            return None, GH_NOT_LOGGED_IN_MESSAGE
        try:
            user = gh("api", "user", "--jq", ".login")
            repo = gh("repo", "view", "--json", "nameWithOwner,viewerPermission")
            if user.returncode != 0 or repo.returncode != 0:
                raise RuntimeError((user.stderr or repo.stderr or "").strip() or "gh exited with an error")
            view = json.loads(repo.stdout)
            access = GithubAccess(login=user.stdout.strip(), repo=view["nameWithOwner"],
                                  permission=str(view.get("viewerPermission") or "READ").upper())
        except Exception as e:
            error(f"CodeQualityMaintainer: gh is logged in but the account or the repository cannot be read: {e}")
            return None, f"The GitHub CLI is logged in, but the repository cannot be read through it: {e}"
        info(f"CodeQualityMaintainer: gh is logged in as {access.login}, {access.permission} on {access.repo}")
        return access, None

    @staticmethod
    def _origin_repo() -> Optional[str]:
        """owner/name of the GitHub repository the checkout's origin is, without asking gh."""
        result = CodeQualityMaintainer._run_subprocess(["git", "remote", "get-url", "origin"], "git remote", GIT_TIMEOUT)
        match = re.search(r"github\.com[:/]([\w.-]+/[\w.-]+?)(?:\.git)?/?$", (result.stdout or "").strip())
        if not match:
            error(f"CodeQualityMaintainer: origin is not a GitHub repository: {(result.stdout or result.stderr).strip()}")
            return None
        return match.group(1)

    @staticmethod
    def _ensure_fork(access: GithubAccess) -> str:
        """owner/name of the account's fork of the repository, made on its first PR — an account
        without push access opens its PRs from there. git is pointed at gh's login, so the push
        over HTTPS needs no credentials of its own. Raises when there is no fork to push to."""
        for cmd, label in ((["gh", "repo", "fork", access.repo, "--clone=false", "--default-branch-only"], "gh repo fork"),
                           (["gh", "auth", "setup-git"], "gh auth setup-git")):
            result = CodeQualityMaintainer._run_subprocess(cmd, label, GIT_TIMEOUT)
            if result.returncode != 0:
                raise RuntimeError(f"{label} failed: {(result.stderr or result.stdout).strip()}")
        result = CodeQualityMaintainer._run_subprocess(
            ["gh", "repo", "list", access.login, "--fork", "--limit", "1000", "--json", "nameWithOwner,parent"],
            "gh repo list", GIT_TIMEOUT)
        if result.returncode != 0:
            raise RuntimeError(f"gh repo list failed: {(result.stderr or '').strip()}")
        for repo in json.loads(result.stdout or "[]"):
            parent = repo.get("parent") or {}
            if f"{(parent.get('owner') or {}).get('login')}/{parent.get('name')}".lower() == access.repo.lower():
                info(f"CodeQualityMaintainer: {access.login} has no push access to {access.repo}, "
                     f"the PR comes from the fork {repo['nameWithOwner']}")
                return repo["nameWithOwner"]
        raise RuntimeError(f"no fork of {access.repo} under {access.login} after gh repo fork")

    @staticmethod
    def _fork_instructions(access: GithubAccess, fork: str) -> str:
        """What a PR job is told when the account cannot push to the repository: the branch
        goes to the account's fork and the PR from there, without a label."""
        return (f"\nIMPORTANT: the GitHub account here ({access.login}) cannot push to {access.repo}, so the pull "
                f"request comes from its fork {fork}. This overrides the steps above: never push to origin; push "
                f"the branch with `git push https://github.com/{fork}.git HEAD:refs/heads/<branch>` and open the PR "
                f"with `gh pr create --repo {access.repo} --head {access.login}:<branch>`. Give it no label — the "
                f"account cannot set labels on {access.repo}.\n")

    @staticmethod
    def _version() -> str:
        from common import Version
        return Version.version()

    @staticmethod
    def improve_code(request: str) -> tuple[str, Optional[JobPullRequest]]:
        """A PR anywhere in the code base: the report and the PR the run opened (None for none)."""
        access, reason = CodeQualityMaintainer._github_access()
        if reason:
            return reason, None
        prompt = SELF_IMPROVEMENT_PROMPT.format(request=request, label=SELF_IMPROVEMENT_LABEL,
                                                version=CodeQualityMaintainer._version())
        return CodeQualityMaintainer._pr_job(prompt, "self-improvement", access)

    @staticmethod
    def improve_integration(name: str, folder: Optional[str], request: str) -> tuple[str, Optional[JobPullRequest]]:
        """A PR confined to one integration: a change to the integration *name* in *folder*,
        or — *folder* None, no integration of that name — a new one. After the run the files
        of the PR the run opened (that one alone, `_job_pull_request`; a run that opened none
        has nothing checked) are checked against that scope (`scope_violations`, on the
        folder `_judged_folder` settles) and a PR that strays is closed, its branch kept and
        the reason in the report, so a user's install can only ever open PRs the developers
        can review as one integration's. Returns the report and the PR left open, if any."""
        access, reason = CodeQualityMaintainer._github_access()
        if reason:
            return reason, None
        version = CodeQualityMaintainer._version()
        if folder is None:
            scope = INTEGRATION_SCOPE.format(folder="<folder>",
                                             registry=", plus the registry lines a new folder needs (" +
                                                      ", ".join(sorted(REGISTRY_FILES)) + " — additions only)")
            prompt = NEW_INTEGRATION_PROMPT.format(name=name, request=request, scope=scope,
                                                   label=INTEGRATION_PR_LABEL, version=version)
        else:
            scope = INTEGRATION_SCOPE.format(folder=folder, registry="")
            prompt = IMPROVE_INTEGRATION_PROMPT.format(name=name, request=request, scope=scope,
                                                       label=INTEGRATION_PR_LABEL, version=version)
        label = f"improve-integration {name}"
        report, pr = CodeQualityMaintainer._pr_job(prompt, label, access)
        if pr is None:
            info(f"CodeQualityMaintainer: {label} opened no PR, so nothing is checked against the integration scope")
            return report, None
        if CodeQualityMaintainer._is_tried_pr(pr.number):
            # During a trial the job pushed onto the tried PR — the developers' own, open
            # before this run and holding whatever it holds. Its files are not this job's
            # scope to judge, and closing it would end the trial the server runs.
            info(f"CodeQualityMaintainer: PR #{pr.number} is the tried PR, not checked against the integration scope")
            return report, pr
        try:
            title, files = CodeQualityMaintainer._pr_files(pr.number)
            judged = CodeQualityMaintainer._judged_folder(files, folder)
            violations = CodeQualityMaintainer.scope_violations(files, judged)
        except Exception as e:
            error(f"CodeQualityMaintainer: PR #{pr.number} could not be checked against the integration scope: {e}")
            judged, violations = folder, [f"its files could not be read ({e})"]
        if not violations:
            info(f"CodeQualityMaintainer: PR #{pr.number} (from {pr.head}) stays inside the {judged or 'new'} "
                 f"integration and is left open")
            return report, pr
        reason = f"a PR from improve_integration must stay inside one integration, and this one {'; '.join(violations)}"
        info(f"CodeQualityMaintainer: closing PR #{pr.number} (from {pr.head}, opened by {label}) — {reason}")
        CodeQualityMaintainer._close_pr(pr.number, reason)
        return (f"PR #{pr.number} was closed again: {reason}.\n\nIts branch {pr.head} is kept, so the developers "
                f"can reopen it. A change that needs more than one integration can be filed for them with "
                f"report_issue.\n\nClaude Code's report:\n{report}"), None

    @staticmethod
    def _is_tried_pr(number: int) -> bool:
        from assistant.agents.PullRequestTrial import PullRequestTrial
        trial = PullRequestTrial.current()
        return trial is not None and trial.number == number

    @staticmethod
    def _trial_instructions() -> str:
        """During a PR trial a PR job is feedback on the tried PR, and pushes to it (the
        paragraph PullRequestTrial adds to the prompt); "" on a normal branch."""
        from assistant.agents.PullRequestTrial import PullRequestTrial
        return PullRequestTrial.pr_job_instructions()

    @staticmethod
    def _pr_job(prompt: str, label: str, access: Optional[GithubAccess] = None) -> tuple[str, Optional[JobPullRequest]]:
        """One Claude Code run that ends in a pull request: on the pulled current branch,
        back on it afterwards whatever happened. During a trial the prompt gains the trial's
        paragraph (the change goes onto the tried PR); otherwise an *access* that cannot push
        gets its fork made first and the fork's paragraph. Returns the report and the PR the
        run opened (`_job_pull_request`), None when it opened none or that cannot be told."""
        delivery = CodeQualityMaintainer._trial_instructions()
        if not delivery and access is not None and not access.can_push:
            try:
                delivery = CodeQualityMaintainer._fork_instructions(access, CodeQualityMaintainer._ensure_fork(access))
            except Exception as e:
                error(f"CodeQualityMaintainer: {label} not started, no fork to open the PR from: {e}")
                return (f"{access.login} cannot push to {access.repo}, and the fork the pull request would come "
                        f"from could not be made: {e}"), None
        prompt += delivery
        branch = CodeQualityMaintainer._current_branch()
        open_before = CodeQualityMaintainer._open_pull_requests()
        try:
            info(f"CodeQualityMaintainer: starting {label}...")
            CodeQualityMaintainer._pull_current()
            trace(f"prompt: {prompt}")
            result = CodeQualityMaintainer._run_claude(prompt, label)
            left_on = CodeQualityMaintainer._current_branch()
            pr = CodeQualityMaintainer._job_pull_request(label, open_before, left_on if left_on != branch else None,
                                                         access.login if access else None)
            return (result if result else f"{label.capitalize()} produced no output."), pr
        except ClaudeUnavailableError as e:
            return str(e), None
        except Exception as e:
            error(f"CodeQualityMaintainer: {label} failed: {e}")
            return f"{label.capitalize()} failed: {e}", None
        finally:
            CodeQualityMaintainer._run_subprocess(["git", "checkout", branch], "git checkout", GIT_TIMEOUT)

    @staticmethod
    def _open_pull_requests() -> Optional[dict[int, JobPullRequest]]:
        """The repository's open PRs by number; None when gh cannot list them — then no PR
        can be told to be a job's, and none is judged."""
        try:
            result = CodeQualityMaintainer._run_subprocess(
                ["gh", "pr", "list", "--state", "open", "--limit", "200", "--json", "number,url,headRefName,author"],
                "gh pr list", GIT_TIMEOUT)
            if result.returncode != 0:
                raise RuntimeError((result.stderr or "").strip() or f"gh pr list exited {result.returncode}")
            prs = json.loads(result.stdout or "[]")
            return {pr["number"]: JobPullRequest(pr["number"], pr.get("url", ""), pr.get("headRefName", ""),
                                                 ((pr.get("author") or {}).get("login") or ""))
                    for pr in prs}
        except Exception as e:
            error(f"CodeQualityMaintainer: could not list the open PRs: {e}")
            return None

    @staticmethod
    def _job_pull_request(label: str, open_before: Optional[dict[int, JobPullRequest]],
                          work_branch: Optional[str], login: Optional[str] = None) -> Optional[JobPullRequest]:
        """The PR a job opened: during a trial the tried PR, which the job pushed onto; else
        an open PR that was not open before the run and, with *login*, opened by that account
        — on a public repository anyone's PR can appear during the hour a job runs — the one
        from *work_branch*, the branch the run left the tree on, or the only new one when the
        tree is back on its own branch. None when the job opened none, and when it cannot be
        told: a job's PR is never guessed, since the scope check closes the PR it is handed."""
        open_after = CodeQualityMaintainer._open_pull_requests()
        if open_after is None:
            info(f"CodeQualityMaintainer: {label}: the open PRs cannot be listed, so no PR is taken as the job's")
            return None
        from assistant.agents.PullRequestTrial import PullRequestTrial
        trial = PullRequestTrial.current()
        if trial is not None:
            pr = open_after.get(trial.number)
            verdict = "the PR it pushed onto" if pr else "which is no longer open, so no PR is the job's"
            info(f"CodeQualityMaintainer: {label} ran during the trial of PR #{trial.number}, {verdict}")
            return pr
        if open_before is None:
            info(f"CodeQualityMaintainer: {label}: the PRs open before the run are unknown, so no PR is taken as the job's")
            return None
        new = [pr for number, pr in open_after.items() if number not in open_before]
        if login:
            others = [pr for pr in new if pr.author.lower() != login.lower()]
            if others:
                info(f"CodeQualityMaintainer: {label}: new PR(s) by other accounts are not the job's: "
                     + ", ".join(f"#{pr.number} by {pr.author or 'unknown'}" for pr in others))
            new = [pr for pr in new if pr not in others]
        if not new:
            info(f"CodeQualityMaintainer: {label} opened no pull request")
            return None
        listed = ", ".join(f"#{pr.number} from {pr.head}" for pr in new)
        if work_branch:
            for pr in new:
                if pr.head == work_branch:
                    info(f"CodeQualityMaintainer: {label} opened PR #{pr.number} from its branch {work_branch}")
                    return pr
            info(f"CodeQualityMaintainer: {label} left the tree on {work_branch} but no PR new since the run started "
                 f"comes from it ({listed}); no PR is taken as the job's")
            return None
        if len(new) == 1:
            info(f"CodeQualityMaintainer: {label} opened PR #{new[0].number} from {new[0].head}, "
                 f"the one PR new since the run started")
            return new[0]
        info(f"CodeQualityMaintainer: {len(new)} PRs are new since {label} started ({listed}) and the tree is back "
             f"on its own branch; none is taken as the job's")
        return None

    @staticmethod
    def _folder_of_path(path: str) -> Optional[str]:
        """The integration folder a path belongs to (its code or its tests), None otherwise."""
        for root in (INTEGRATION_FOLDERS, INTEGRATION_TESTS):
            if path.startswith(root):
                rest = path[len(root):]
                return rest.split("/", 1)[0] if "/" in rest else None
        return None

    @staticmethod
    def _existing_integration_folders() -> set[str]:
        """The integration folders of this checkout — on its own branch again after a run, so
        a folder a PR adds is not among them."""
        root = REPO_DIR / INTEGRATION_FOLDERS
        if not root.is_dir():
            return set()
        return {p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")}

    @staticmethod
    def _judged_folder(files: list[dict], folder: Optional[str]) -> Optional[str]:
        """The folder the scope check judges a PR against: *folder* when the job had one; for
        a job without one (no integration of the name) the one existing folder every file sits
        in, when there is such a one — the name resolved to nothing, but Claude Code found the
        folder (2026-10-26: "music", named by its folder, got a fix confined to the music folder
        closed as "an existing integration instead of a new one"); else None, a new
        integration's."""
        if folder is not None:
            return folder
        owners = {CodeQualityMaintainer._folder_of_path(path) for f in files
                  for path in [f["filename"]] + ([f["previous_filename"]] if f.get("previous_filename") else [])}
        if len(owners) == 1 and None not in owners:
            (owner,) = owners
            if owner in CodeQualityMaintainer._existing_integration_folders():
                info(f"CodeQualityMaintainer: every file of the PR is in the existing {owner} folder, "
                     f"so it is judged as a change to that integration, not as a new one")
                return owner
        trace(f"CodeQualityMaintainer: the PR is judged as a new integration's "
              f"(folders touched: {sorted(o for o in owners if o) or 'none'})")
        return None

    @staticmethod
    def scope_violations(files: list[dict], folder: Optional[str]) -> list[str]:
        """Why a PR from improve_integration is not one integration's — empty when it is.
        *files* as `_pr_files` lists them. The rule: every path is in one integration folder
        or its tests; for a change to *folder* it is that folder and nothing else; for a new
        integration (*folder* None) that folder is new (every file in it added) and the
        REGISTRY_FILES may be touched with additions only. A renamed file counts under its
        previous path too, so nothing can be moved in from outside."""
        violations, folders, outside, not_new, removals = [], set(), [], [], []
        for f in files:
            paths = [f["filename"]] + ([f["previous_filename"]] if f.get("previous_filename") else [])
            for path in paths:
                owner = CodeQualityMaintainer._folder_of_path(path)
                if owner is not None:
                    folders.add(owner)
                    if folder is None and f.get("status") != "added":
                        not_new.append(path)
                elif folder is None and path in REGISTRY_FILES:
                    if f.get("deletions", 0):
                        removals.append(path)
                else:
                    outside.append(path)
        if not files:
            violations.append("changes no files")
        if outside:
            violations.append(f"changes files outside an integration folder: {', '.join(sorted(set(outside))[:5])}")
        if len(folders) > 1:
            violations.append(f"touches several integrations: {', '.join(sorted(folders))}")
        if folder is not None and folders - {folder}:
            violations.append(f"touches the {', '.join(sorted(folders - {folder}))} folder instead of {folder}")
        if not_new:
            violations.append(f"changes an existing integration instead of adding one: {', '.join(sorted(set(not_new))[:5])}")
        if removals:
            violations.append(f"removes lines from a registry file: {', '.join(sorted(set(removals)))}")
        trace(f"CodeQualityMaintainer: scope check of {len(files)} file(s) for {folder or 'a new integration'}: "
              f"{violations or 'inside the scope'}")
        return violations

    @staticmethod
    def _close_pr(number: int, reason: str):
        """Closes the PR with the reason as a comment. The branch stays: a closed PR is reopened
        in one step, a deleted branch is restored by hand (2026-10-26, twice)."""
        comment = (f"Closed by Keep: {reason}. The branch is kept, so the PR can be reopened; a change beyond "
                   f"one integration goes to the developers as an issue (report_issue).")
        result = CodeQualityMaintainer._run_subprocess(
            ["gh", "pr", "close", str(number), "--comment", comment],
            f"gh pr close {number}", GIT_TIMEOUT)
        if result.returncode != 0:
            error(f"CodeQualityMaintainer: could not close PR #{number}: {result.stderr.strip()}")

    @staticmethod
    def report_issue(request: str) -> tuple[str, Optional[IssueDraft]]:
        """One Claude Code run that investigates and files a GitHub issue; the report names
        the issue's URL (`issue_in`), or says why none was filed. Where gh cannot file it (not
        installed or not logged in — a normal install), the run drafts it instead and the
        draft comes back for the user to file by hand."""
        access, reason = CodeQualityMaintainer._github_access()
        version = CodeQualityMaintainer._version()
        draft_path = None
        if access is not None:
            label = (f"Give it the label `{ISSUE_LABEL}`." if access.can_label
                     else f"Give it no label — the account cannot set labels on {access.repo}.")
            parts = ISSUE_FILED
        else:
            repo = CodeQualityMaintainer._origin_repo()
            if repo is None:
                return reason, None
            ISSUE_DRAFT_DIR.mkdir(parents=True, exist_ok=True)
            draft_path = ISSUE_DRAFT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.md"
            why = CodeQualityMaintainer._not_filed_because(reason)
            info(f"CodeQualityMaintainer: gh cannot file the issue ({why}), drafting it to {draft_path}")
            label = ""
            parts = {k: v.format(repo=repo, path=draft_path) for k, v in ISSUE_DRAFTED.items()}
        info("CodeQualityMaintainer: starting the issue report...")
        prompt = REPORT_ISSUE_PROMPT.format(request=request, label=label, version=version, **parts)
        trace(f"prompt: {prompt}")
        try:
            result = CodeQualityMaintainer._run_claude(prompt, "report-issue")
        except ClaudeUnavailableError as e:
            return str(e), None
        except Exception as e:
            error(f"CodeQualityMaintainer: the issue report failed: {e}")
            return f"The issue report failed: {e}", None
        report = result if result else "The issue report produced no output."
        if draft_path is None:
            return report, None
        return report, CodeQualityMaintainer.read_issue_draft(draft_path, repo, why)

    @staticmethod
    def _not_filed_because(reason: str) -> str:
        """The clause the user hears for a `_github_access` refusal: the three ways gh cannot
        file from this machine, in the words of the chat message the draft comes with."""
        if reason == GH_NOT_INSTALLED_MESSAGE:
            return "the GitHub CLI is not installed on this machine"
        if reason == GH_NOT_LOGGED_IN_MESSAGE:
            return "there is no GitHub login on this machine"
        return "the GitHub CLI on this machine cannot read the repository"

    @staticmethod
    def read_issue_draft(path, repo: str, why: str = IssueDraft.why) -> Optional[IssueDraft]:
        """The draft a report_issue run wrote to *path* (`Title: …` or `Comment on: #N`, an
        empty line, the body), *why* being the reason it was not filed; None when it wrote
        none or one that does not read."""
        try:
            text = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            info(f"CodeQualityMaintainer: the issue run wrote no draft to {path}")
            return None
        except OSError as e:
            error(f"CodeQualityMaintainer: the issue draft {path} cannot be read: {e}")
            return None
        head, _, body = text.partition("\n")
        match = _DRAFT_HEAD.match(head)
        if not match or not body.strip():
            error(f"CodeQualityMaintainer: the issue draft {path} does not start with a title or an issue number "
                  f"and a body: {head[:120]!r}")
            return None
        number = match.group("number")
        draft = IssueDraft(repo=repo, title=(match.group("title") or "").strip(), body=body.strip(),
                           comment_on=int(number) if number else None, why=why)
        info(f"CodeQualityMaintainer: issue drafted, "
             f"{f'a comment on #{draft.comment_on}' if number else repr(draft.title)} ({len(draft.body)} characters)")
        return draft

    _ISSUE_URL = re.compile(r"https?://github\.com/[\w.-]+/[\w.-]+/issues/(\d+)(?:#issuecomment-\d+)?")

    @classmethod
    def issue_in(cls, report: str) -> tuple[int | None, str | None]:
        """(number, URL) of the issue a report_issue report names — a new issue's, or the
        one it commented on; (None, None) when it filed nothing."""
        match = cls._ISSUE_URL.search(report or "")
        if match:
            return int(match.group(1)), match.group(0)
        trace("CodeQualityMaintainer: the report names no issue")
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
        return CodeQualityMaintainer._updater().current_branch()

    @staticmethod
    def target_branch() -> str:
        """The branch restart_server pulls: master with `developer_release: true`, else release."""
        developer = bool(ConfigManager().get(DEVELOPER_RELEASE_KEY, False))
        return DEVELOPER_BRANCH if developer else RELEASE_BRANCH

    @staticmethod
    def _pull_current() -> None:
        """Bring the current branch up to date before a Claude Code run (`Updater.pull_current`).
        A failure is logged and the run goes on with the code it has."""
        CodeQualityMaintainer._updater().pull_current()

    @staticmethod
    def _sync_requirements() -> tuple[str, str]:
        """Between a checkout and the restart: the venv is brought to the pins of the code now in
        the tree (`Updater.sync`). Returns (failure, installed); a failure means do not restart."""
        return Updater.sync()

    @staticmethod
    def pull_and_restart(resumes_conversation: bool = False) -> str:
        """Pull the target branch, sync the venv and restart. *resumes_conversation* is the
        `restart_server` command's: the restart was asked for in a chat, and the new process
        picks that conversation up (`restart`)."""
        update = CodeQualityMaintainer._updater().refresh(CodeQualityMaintainer.target_branch())
        if update.pull_failure:
            return update.pull_failure
        if update.sync_failure:
            return f"The server was not restarted: {update.sync_failure}"

        changes_msg = "no new changes" if update.up_to_date else f"pulled updates: {update.pulled}"
        if update.installed:
            changes_msg += f"; {update.installed}"
        CodeQualityMaintainer.restart(f"update requested, {changes_msg}", resumes_conversation=resumes_conversation)
        return f"Update initiated ({changes_msg}). Server is restarting..."

    @staticmethod
    def restart(reason: str = "", fallback_branch: str = "", resumes_conversation: bool = False) -> str:
        """Relaunch the server in place via start.sh (graceful kill + fresh start) WITHOUT
        pulling new code. Used by pull_and_restart() after the pull, and by automatic
        recovery paths (e.g. AudioRecovery) that just need a clean process to clear an
        in-process wedge while running the same code. A PR trial names the branch to fall
        back to: start.sh then checks it out and relaunches if the new server never comes up.
        A restart the assistant started from a chat (*resumes_conversation*) leaves the marker
        the next boot resumes that conversation from; the nightly update and the recovery
        paths leave none — nobody is waiting on those."""
        if resumes_conversation:
            from common.StartupTracker import StartupTracker
            StartupTracker.write_resume_marker(reason)
        Updater.relaunch(START_SCRIPT, reason, *([fallback_branch] if fallback_branch else []))
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
    def _send_email(subject: str, html_body: str, attachments: list[tuple[str, str]] = ()) -> str | None:
        """The report goes to the household's own address: the account `send_email` sends from
        (`mail.send_from`, else the first configured one) mails it to itself. `attachments` are
        (filename, text) pairs attached as plain-text files. Returns the address it went to,
        None when no mail account is configured."""
        from assistant.integration.mail.MailAccess import MailAccess
        account = MailAccess.current().sender()
        if account is None or not account.address:
            error("CodeQualityMaintainer: no mail account configured, skipping send.")
            return None
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

        msg = MIMEMultipart("mixed" if attachments else "alternative")
        msg["Subject"] = subject
        msg["To"] = address
        msg.attach(MIMEText(full_html, "html"))
        for filename, text in attachments:
            part = MIMEText(text, "plain", "utf-8")
            part.add_header("Content-Disposition", "attachment", filename=filename)
            msg.attach(part)

        err = account.send_message(msg, [address])
        if err:
            raise RuntimeError(f"report email not sent: {err}")
        info(f"CodeQualityMaintainer: email sent to {address} from {account.name}"
             + (f" with {len(attachments)} attachment(s)" if attachments else ""))
        return address

