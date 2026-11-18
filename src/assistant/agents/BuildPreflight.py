import json
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Optional

from common.logging import info, error, trace
from common.Updater import GIT_TIMEOUT
from . import CodeQualityMaintainer as cqm
from . import CommitAuthor

# The test request `claude_credits` sends: one word asked for, no tools and no system prompt of the
# CLI's own, from a directory without a CLAUDE.md — about 600 tokens, a cent on a Fable model
# (measured 2026-11-18; from the checkout the same request carried CLAUDE.md, 17 700 tokens).
PROBE_PROMPT = "Reply with the single word OK."
PROBE_TIMEOUT = 60
# How long a passed test request is believed. A build asks again after a quarter of an hour, since
# it costs dollars where the request costs a cent; the health check, which runs every 15 minutes
# around the clock, after a day — and at once when the integration comes up (`forget_credits`),
# which is when it matters most: the setup. A refused request is never kept: the next ask sends
# another, so the state mends within one check of the account being topped up.
CREDITS_AGE_FOR_A_BUILD = 15 * 60
CREDITS_AGE_FOR_HEALTH = 24 * 60 * 60
REASON_MAX_CHARS = 200
CLAUDE_KEYS = frozenset({"claude", "credits"})   # the items without which no job runs at all, an analysis included

_credits_lock = threading.Lock()
_credits_passed: dict[tuple, float] = {}   # (CLI, credentials, model) -> when its test request last passed


def forget_credits() -> None:
    """No earlier pass stands: the next ask sends a test request. Called when the Self Improvement
    integration comes up — the server's start, `enable_integration`, a restart of it."""
    with _credits_lock:
        _credits_passed.clear()


@dataclass(frozen=True)
class Unmet:
    """One thing a job needs that the server machine lacks: what it is and how it is mended."""
    key: str
    text: str             # with the remedy: what a refused build answers with
    brief: str = ""       # the few words of it the health state carries; the text itself when it is short
    blocks: bool = True   # False: said with the rest, and the job starts all the same (the memory)


def claude_credits(status: "cqm.ClaudeStatus", max_age: float) -> Optional[str]:
    """`credits_refusal` in one sentence pair: what was refused, then the way out."""
    refusal = credits_refusal(status, max_age)
    return " ".join(refusal) if refusal else None


def credits_refusal(status: "cqm.ClaudeStatus", max_age: float) -> Optional[tuple[str, str]]:
    """Why Claude Code cannot run the jobs' model right now — what the CLI answered, and the
    way out, apart: the health state carries the first alone — None when it can: a login or a key
    says nothing about that — an account without usage credits, a plan that lacks the model
    and a spent limit all pass `claude auth status` and fail the first request, which used to
    be the build's. So one test request is sent as the jobs send theirs (the same CLI,
    credentials and model), and a pass is believed for *max_age* seconds. Only a refusal the
    CLI itself states counts: a request that timed out or gave no result (an old CLI that
    lacks a flag, no network) is logged and taken as no finding."""
    model = cqm.build_model()
    key = (status.path, status.api_key, model)
    with _credits_lock:
        passed = _credits_passed.get(key)
        if passed is not None and time.monotonic() - passed < max_age:
            trace(f"Claude Code credits: a test request for {model} passed {time.monotonic() - passed:.0f} s ago, "
                  f"not asked again")
            return None
        cmd = [status.path, "-p", PROBE_PROMPT, "--model", model, "--output-format", "json",
               "--system-prompt", PROBE_PROMPT, "--tools", "", "--no-session-persistence",
               "--strict-mcp-config", "--disable-slash-commands"]
        try:
            result = subprocess.run(cmd, cwd=tempfile.gettempdir(), capture_output=True, text=True,
                                    stdin=subprocess.DEVNULL, timeout=PROBE_TIMEOUT, env=status.env())
            answer = json.loads(result.stdout or "")
            if not isinstance(answer, dict) or answer.get("type") != "result":
                raise ValueError("no result object")
        except subprocess.TimeoutExpired:
            error(f"Claude Code credits: the test request for {model} timed out after {PROBE_TIMEOUT}s, no finding")
            return None
        except OSError as e:
            error(f"Claude Code credits: the test request could not be run ({e}), no finding")
            return None
        except ValueError:
            output = (result.stderr or result.stdout or "").strip().replace("\n", " ")[:200]
            error(f"Claude Code credits: the test request for {model} gave no result (exit {result.returncode}): "
                  f"{output} — no finding")
            return None
        if not answer.get("is_error"):
            cost = answer.get("total_cost_usd")
            info(f"Claude Code credits: a test request for {model} was answered"
                 f"{f' (${cost:.4f})' if cost else ''}, {status.detail}")
            _credits_passed[key] = time.monotonic()
            return None
        _credits_passed.pop(key, None)
        said = " ".join(str(answer.get("result") or "no reason given").split())[:REASON_MAX_CHARS]
        info(f"Claude Code credits: a test request for {model} was refused: {said}")
        paid_by = cqm.API_KEY_CONFIG_KEY if status.api_key else "its claude.ai login"
        # Never "a model it has": told that, the assistant put the builds on Sonnet (2026-11-18).
        way_out = ("The account needs usage for that model." if model == cqm.BUILD_FALLBACK_MODEL else
                   f"The account needs usage for that model, or {cqm.BUILD_MODEL_KEY} set to "
                   f"{cqm.BUILD_FALLBACK_MODEL}: {cqm.BUILD_MODEL_RULE}.")
        return f"Claude Code cannot run {model} on {paid_by}: \"{said}\"", way_out


def forks_refused(access: "cqm.GithubAccess") -> bool:
    """Whether the repository is certain to refuse a fork to an account that cannot push to it."""
    try:
        result = cqm.CodeQualityMaintainer._run_subprocess(
            ["gh", "api", f"repos/{access.repo}", "--jq", ".allow_forking"], "gh api repo", GIT_TIMEOUT)
    except (OSError, subprocess.SubprocessError) as e:
        error(f"Build preflight: could not ask whether {access.repo} allows forks ({e}), taken as allowed")
        return False
    refused = result.returncode == 0 and (result.stdout or "").strip() == "false"
    trace(f"Build preflight: {access.login} cannot push to {access.repo}, a fork is "
          f"{'refused by the repository' if refused else 'the way'}")
    return refused


@dataclass(frozen=True)
class BuildPreflight:
    """Everything a Claude Code job needs on the server machine, checked in one pass before the job starts.

    The health check of the Self Improvement integration and the commands that start a build
    ask it, so what is missing is said whole and at once: on 2026-11-17 a tester's install
    reported "ok" and then found its gaps one build at a time — no gh, no gh login, no usage
    credits, too little memory, no git author — eleven builds in all. `check()` goes through
    all of it whatever it finds on the way: the Claude Code CLI and its credentials
    (`claude_status`), that it can pay for the jobs' model (`claude_credits`), for a job that
    ends in a pull request also gh, its login and push or fork access to the checkout's
    repository (`_github_access`, `forks_refused`) and an author for the commits — git's own,
    else the gh account's, so with gh logged in there always is one (`CommitAuthor`; a job
    never picks one: `COMMIT_AUTHOR_RULE`) — and the machine's memory, which is said and
    refuses nothing (`build_memory_warning`)."""
    unmet: tuple[Unmet, ...] = ()

    @staticmethod
    def check(pull_request: bool = True, credits_max_age: float = CREDITS_AGE_FOR_A_BUILD) -> "BuildPreflight":
        """*pull_request* False is a job that pushes nothing to the repository (the local
        integration folders): Claude Code and the memory alone."""
        unmet = []
        status = cqm.claude_status()
        if not status.ok:
            unmet.append(Unmet("claude", status.detail))
        else:
            refusal = credits_refusal(status, credits_max_age)
            if refusal:
                unmet.append(Unmet("credits", " ".join(refusal), brief=refusal[0]))
        if pull_request:
            access, reason = cqm.CodeQualityMaintainer._github_access()
            if reason:
                unmet.append(Unmet("github", reason, brief=cqm.CodeQualityMaintainer._not_filed_because(reason)))
            elif not access.can_push and forks_refused(access):
                stuck = f"{access.login} can neither push to {access.repo} nor fork it"
                unmet.append(Unmet("github", f"The GitHub account {stuck} (the repository allows no forks): it "
                                             f"needs write access.", brief=stuck))
            keys = [] if reason else CommitAuthor.missing_keys()
            if keys and CommitAuthor.account() is None:
                unmet.append(Unmet("git_author", (
                    f"The commits have no author: git has no {' and '.join(keys)} on the server machine and gh "
                    f"did not name its account. The user sets them in a terminal there (`git config --global "
                    f"user.name`, `user.email`); both are public in the pull request's commits."),
                    brief="no author for the commits (git has none, gh names no account)"))
        memory = cqm.build_memory_warning()
        if memory:
            unmet.append(Unmet("memory", memory, blocks=False))
        found = BuildPreflight(tuple(unmet))
        if unmet:
            info(f"Build preflight ({'a pull request' if pull_request else 'no pull request'}): "
                 f"{len(found.blocking)} item(s) in the way [{', '.join(u.key for u in found.blocking) or 'none'}], "
                 f"{len(found.advisory)} said only [{', '.join(u.key for u in found.advisory) or 'none'}]")
        else:
            trace(f"Build preflight ({'a pull request' if pull_request else 'no pull request'}): nothing is missing")
        return found

    @property
    def blocking(self) -> tuple[Unmet, ...]:
        return tuple(u for u in self.unmet if u.blocks)

    @property
    def advisory(self) -> tuple[Unmet, ...]:
        return tuple(u for u in self.unmet if not u.blocks)

    @property
    def claude_down(self) -> bool:
        """Whether Claude Code itself cannot run, so not even an analysis would."""
        return any(u.key in CLAUDE_KEYS for u in self.blocking)

    def health_text(self) -> str:
        """What blocks, in a few words each on one line — the integration's health state,
        which `list_integrations` and `enable_integration` show and the prompt may carry, so
        it names what is missing and leaves the remedy to the refused build's answer and to
        `config_help`."""
        return "; ".join(u.brief or u.text for u in self.blocking)

    def refusal(self, label: str) -> str:
        """What a build command answers with when something blocks: every item with its
        remedy, and the memory warning beside them, so the user hears all of it in one go."""
        lines = [f"{label} not started. Missing on the server machine, tell the user all of it at once:"]
        lines += [f"{n}. {u.text}" for n, u in enumerate(self.blocking, 1)]
        lines += [f"Also: {u.text}" for u in self.advisory]
        return "\n".join(lines)
