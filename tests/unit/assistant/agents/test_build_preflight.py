"""What a build needs on the server machine, found missing in one pass (assistant/agents/BuildPreflight)."""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from assistant.agents import BuildPreflight as bp
from assistant.agents import CodeQualityMaintainer as cqm
from assistant.agents import CommitAuthor
from assistant.agents.BuildPreflight import BuildPreflight
from assistant.agents.CodeQualityMaintainer import (ClaudeStatus, CodeQualityMaintainer, GithubAccess,
                                                    COMMIT_AUTHOR_RULE, NOT_INSTALLED_SHORT)
from common.Paths import AISERVER_DIR

# The real check: the conftest answers "nothing missing" in its place for every other test, in a
# session fixture that runs after this module was imported.
check = BuildPreflight.check

MODEL = "claude-fable-5-2"
LOGGED_IN = ClaudeStatus("/opt/bin/claude", True, "logged in as m@mlab.test (max)")
ON_THE_KEY = ClaudeStatus("/opt/bin/claude", True, "no login, using the key", api_key=True)
NO_CLI = ClaudeStatus(None, False, NOT_INSTALLED_SHORT)
PUSHER = GithubAccess(login="marcus", repo="marcus/keep", permission="WRITE")
READER = GithubAccess(login="alice", repo="marcus/keep", permission="READ")
GH_MISSING = "The GitHub CLI (gh) is not installed on the server machine."
SMALL = "This machine has 4 GB of memory and no swap."


@pytest.fixture
def machine():
    """A machine that has everything a build needs; a test takes away what it means."""
    with patch.object(cqm, "claude_status", return_value=LOGGED_IN) as status, \
            patch.object(bp, "credits_refusal", return_value=None) as credits, \
            patch.object(CodeQualityMaintainer, "_github_access", return_value=(PUSHER, None)) as github, \
            patch.object(bp, "forks_refused", return_value=False) as forks, \
            patch.object(CommitAuthor, "missing_keys", return_value=[]) as author, \
            patch.object(CommitAuthor, "account", return_value={"user.name": "Marcus", "user.email": "m@mlab.test"}) as account, \
            patch.object(cqm, "build_memory_warning", return_value=None) as memory:
        yield SimpleNamespace(status=status, credits=credits, github=github, forks=forks, author=author,
                              account=account, memory=memory)


@pytest.mark.unit
class TestCheck:
    def test_the_key_standing_in_for_a_login_is_found_and_blocks_nothing(self, machine):
        """The health check warns on it (#211); a login, or a machine without the CLI, is not on the key."""
        assert not check().on_api_key
        machine.status.return_value = ON_THE_KEY
        found = check()
        assert found.on_api_key and not found.unmet

    def test_everything_missing_is_found_in_one_pass(self, machine):
        """An install found its gaps one build at a time: no gh, no login, no usage, too little
        memory, no git author — eleven builds (2026-11-17)."""
        machine.status.return_value = NO_CLI
        machine.github.return_value = (None, GH_MISSING)
        machine.author.return_value = ["user.name", "user.email"]
        machine.memory.return_value = SMALL
        found = check()
        # no author item beside the gh one: once gh is logged in, its account is the author
        assert [u.key for u in found.unmet] == ["claude", "github", "memory"]
        assert [u.key for u in found.blocking] == ["claude", "github"]
        assert [u.text for u in found.advisory] == [SMALL]      # said, and in nobody's way
        machine.credits.assert_not_called()                     # no CLI to send a test request through

    def test_a_machine_with_everything_lacks_nothing(self, machine):
        found = check()
        assert found.unmet == () and not found.claude_down

    def test_an_account_without_usage_stops_every_job(self, machine):
        machine.credits.return_value = ("Claude Code cannot run the model: out of usage credits.", "Switch to Opus.")
        found = check()
        assert [u.key for u in found.blocking] == ["credits"] and found.claude_down
        # the health state names what was refused; the way out is the refused build's to give
        assert "out of usage credits" in found.health_text() and "Switch to Opus." not in found.health_text()
        assert "Switch to Opus." in found.refusal("Self-improvement")

    def test_what_only_a_pull_request_needs_leaves_claude_code_up(self, machine):
        machine.github.return_value = (None, GH_MISSING)
        found = check()
        assert [u.key for u in found.blocking] == ["github"] and not found.claude_down

    def test_a_git_without_an_author_lacks_nothing_while_gh_names_its_account(self, machine):
        """A fresh install has no author in git's config; the run commits as the gh account."""
        machine.author.return_value = ["user.name", "user.email"]
        assert check().unmet == ()

    @pytest.mark.parametrize("keys, named, unnamed", [(["user.email"], "user.email", "user.name and"),
                                                      (["user.name", "user.email"], "user.name and user.email", None)])
    def test_no_author_from_git_or_gh_is_in_the_way_and_names_what_git_lacks(self, machine, keys, named, unnamed):
        machine.author.return_value = keys
        machine.account.return_value = None
        found = check()
        assert [u.key for u in found.blocking] == ["git_author"]
        text = found.blocking[0].text
        assert named in text and (unnamed is None or unnamed not in text)

    def test_an_account_that_can_neither_push_nor_fork_has_nowhere_to_open_a_pr_from(self, machine):
        machine.github.return_value = (READER, None)
        machine.forks.return_value = True
        found = check()
        assert [u.key for u in found.blocking] == ["github"]
        assert "alice" in found.blocking[0].text and "marcus/keep" in found.blocking[0].text

    def test_an_account_that_cannot_push_but_may_fork_lacks_nothing(self, machine):
        machine.github.return_value = (READER, None)
        assert check().unmet == ()
        machine.forks.assert_called_once_with(READER)

    def test_an_account_that_can_push_is_not_asked_about_forks(self, machine):
        check()
        machine.forks.assert_not_called()

    def test_a_job_without_a_pull_request_is_not_held_to_gh_or_an_author(self, machine):
        """The local integration folders are the user's own: nothing is pushed to the repository."""
        machine.github.return_value = (None, GH_MISSING)
        machine.author.return_value = ["user.name"]
        machine.memory.return_value = SMALL
        found = check(pull_request=False)
        assert found.blocking == () and [u.key for u in found.advisory] == ["memory"]
        machine.github.assert_not_called()
        machine.author.assert_not_called()

    def test_the_refusal_carries_every_item_and_the_memory_beside_them(self, machine):
        machine.credits.return_value = ("Claude Code cannot run the model: out of usage credits.", "Switch to Opus.")
        machine.github.return_value = (None, GH_MISSING)
        machine.memory.return_value = SMALL
        found = check()
        refusal = found.refusal("Self-improvement")
        assert len(found.unmet) == 3 and all(u.text in refusal for u in found.unmet)


def _answered(result="OK", is_error=False):
    return subprocess.CompletedProcess(args=[], returncode=1 if is_error else 0, stderr="",
                                       stdout=json.dumps({"type": "result", "is_error": is_error, "result": result}))


@pytest.mark.unit
class TestClaudeCredits:
    """A login says nothing about whether the account can pay for the jobs' model: one test request does."""

    OUT = "You're out of usage credits. Add more at claude.ai/settings/usage."

    @pytest.fixture(autouse=True)
    def nothing_passed_yet(self):
        bp._credits_passed.clear()
        with patch.object(cqm, "build_model", return_value=MODEL):
            yield
        bp._credits_passed.clear()

    @patch.object(bp.subprocess, "run", return_value=_answered())
    def test_the_request_goes_out_as_a_job_would_and_carries_no_checkout(self, run):
        with patch.object(cqm, "claude_api_key", return_value="sk-ant-test-key"):
            assert bp.claude_credits(ON_THE_KEY, 900) is None
        cmd, kwargs = run.call_args[0][0], run.call_args[1]
        assert cmd[0] == "/opt/bin/claude" and cmd[cmd.index("--model") + 1] == MODEL
        assert kwargs["env"][cqm.API_KEY_ENV] == "sk-ant-test-key"
        # from the checkout the CLI reads CLAUDE.md into the request: 17 700 tokens where 600 do
        assert not Path(kwargs["cwd"]).resolve().is_relative_to(AISERVER_DIR)
        assert cmd[cmd.index("--tools") + 1] == ""

    @patch.object(bp.subprocess, "run", return_value=_answered())
    def test_a_pass_is_believed_for_its_age_and_no_longer(self, run):
        with patch.object(bp.time, "monotonic", side_effect=[1000.0]):
            bp.claude_credits(LOGGED_IN, 900)
        with patch.object(bp.time, "monotonic", return_value=1000.0 + 899):
            assert bp.claude_credits(LOGGED_IN, 900) is None
        assert run.call_count == 1
        with patch.object(bp.time, "monotonic", return_value=1000.0 + 901):
            bp.claude_credits(LOGGED_IN, 900)
        assert run.call_count == 2

    @patch.object(bp.subprocess, "run", return_value=_answered())
    def test_a_pass_for_one_model_says_nothing_about_another(self, run):
        bp.claude_credits(LOGGED_IN, 900)
        with patch.object(cqm, "build_model", return_value="claude-opus-5-5"):
            bp.claude_credits(LOGGED_IN, 900)
        assert [c[0][0][c[0][0].index("--model") + 1] for c in run.call_args_list] == [MODEL, "claude-opus-5-5"]

    def test_a_refusal_comes_back_in_the_clis_words_and_is_asked_again(self):
        with patch.object(bp.subprocess, "run", return_value=_answered(self.OUT, is_error=True)) as run:
            first = bp.claude_credits(LOGGED_IN, 900)
            second = bp.claude_credits(LOGGED_IN, 900)
        assert self.OUT in first and MODEL in first and second == first
        assert run.call_count == 2      # mended within one check of the account being topped up

    def test_a_refusal_ends_an_earlier_pass(self):
        with patch.object(bp.time, "monotonic", return_value=1000.0), \
                patch.object(bp.subprocess, "run", return_value=_answered()):
            bp.claude_credits(LOGGED_IN, 900)
        with patch.object(bp.time, "monotonic", return_value=2000.0), \
                patch.object(bp.subprocess, "run", return_value=_answered(self.OUT, is_error=True)):
            assert bp.claude_credits(LOGGED_IN, 900)          # asked: the pass was too old for a build
            assert bp.claude_credits(LOGGED_IN, 6 * 3600)     # and no longer stands for the health check

    def test_the_refusal_says_whose_usage_it_is(self):
        with patch.object(bp.subprocess, "run", return_value=_answered(self.OUT, is_error=True)), \
                patch.object(cqm, "claude_api_key", return_value="sk-ant-test-key"):
            assert cqm.API_KEY_CONFIG_KEY in bp.claude_credits(ON_THE_KEY, 900)
            assert cqm.API_KEY_CONFIG_KEY not in bp.claude_credits(LOGGED_IN, 900)

    @pytest.mark.parametrize("outcome", [
        subprocess.TimeoutExpired(cmd="claude", timeout=60),
        OSError("exec format error"),
        subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="error: unknown option '--tools'"),
        subprocess.CompletedProcess(args=[], returncode=0, stdout='["not", "a", "result"]', stderr=""),
    ], ids=["timeout", "cannot run", "an old CLI", "no result"])
    def test_only_a_refusal_the_cli_states_counts(self, outcome):
        """A request that got no verdict blocks nothing, and is no pass to believe either."""
        raised = isinstance(outcome, Exception)
        with patch.object(bp.subprocess, "run", side_effect=outcome if raised else None,
                          return_value=None if raised else outcome) as run:
            assert bp.claude_credits(LOGGED_IN, 900) is None
            assert bp.claude_credits(LOGGED_IN, 900) is None
        assert run.call_count == 2


@pytest.mark.unit
class TestForks:
    @pytest.mark.parametrize("gh, refused", [
        (Mock(returncode=0, stdout="false\n"), True),
        (Mock(returncode=0, stdout="true\n"), False),
        (Mock(returncode=1, stdout=""), False),                  # gh could not say: a fork is tried
        (OSError("no gh"), False),
    ])
    def test_only_a_certain_no_refuses(self, gh, refused):
        raised = isinstance(gh, Exception)
        with patch.object(CodeQualityMaintainer, "_run_subprocess", side_effect=gh if raised else None,
                          return_value=None if raised else gh) as run:
            assert bp.forks_refused(READER) is refused
        assert "repos/marcus/keep" in run.call_args[0][0]


@pytest.mark.unit
class TestNoRunPicksAnAuthor:
    """Of four builds on a machine without a git author two committed under a name they made up
    and two were stopped trying (2026-11-17): every run that commits is told to do neither."""

    @staticmethod
    def _prompt_of(run) -> str:
        prompts = []
        with patch.object(CodeQualityMaintainer, "_run_claude",
                          side_effect=lambda prompt, label, **_: prompts.append(prompt) or "done"), \
                patch.object(CodeQualityMaintainer, "_run_subprocess", return_value=Mock(returncode=0, stdout="release\n", stderr="")), \
                patch.object(CodeQualityMaintainer, "_job_home", return_value="release"), \
                patch.object(CodeQualityMaintainer, "_back_home"), \
                patch.object(CodeQualityMaintainer, "_pull_current"), \
                patch.object(CodeQualityMaintainer, "_open_pull_requests", return_value={}), \
                patch.object(CodeQualityMaintainer, "_trial_instructions", return_value=""), \
                patch.object(CodeQualityMaintainer, "_merge_test_prs", return_value=(None, [])), \
                patch("assistant.agents.PullRequestTrial.PullRequestTrial.current", return_value=None):
            run()
        return prompts[0]

    @pytest.mark.parametrize("run", [
        lambda: CodeQualityMaintainer._pr_job("Fix the trunk.", "self-improvement", PUSHER),
        lambda: CodeQualityMaintainer()._step_generate_tests(),
        lambda: CodeQualityMaintainer._step_test_health(),
        lambda: CodeQualityMaintainer.improve_local_integration("Add a feed.", (Path("/srv/keep-local"),)),
    ], ids=["a PR job", "test generation", "test health", "a local integration"])
    def test_every_run_that_commits_ends_with_the_rule(self, run):
        assert self._prompt_of(run).endswith(COMMIT_AUTHOR_RULE)

    def test_an_analysis_commits_nothing_and_is_not_told(self):
        assert COMMIT_AUTHOR_RULE not in self._prompt_of(lambda: CodeQualityMaintainer.analyze_code("why?"))
