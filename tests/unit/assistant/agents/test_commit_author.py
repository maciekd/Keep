"""Who a Claude Code job's commits are by: git's own author, else the gh account (assistant/agents/CommitAuthor)."""
import os
import subprocess
from unittest.mock import Mock, patch

import pytest

from assistant.agents import CommitAuthor
from assistant.agents import CodeQualityMaintainer as cqm
from assistant.agents.CodeQualityMaintainer import ClaudeStatus, CodeQualityMaintainer

# The real one: the conftest answers {} in its place for every other test, in a session fixture
# that runs after this module was imported.
author_env = CommitAuthor.env

# The account's own private address must never be the one committed under: the noreply address is
# built from its id and login, here those of the cast's Marcus.
GH_MARCUS = "4711\tmarcus-k\tMarcus Keeper\n"
NOREPLY = "4711+marcus-k@users.noreply.github.com"


def _machine(name="Marcus", email="m@mlab.test", gh=GH_MARCUS):
    """A machine whose git config holds *name* and *email* (None: unset, an int: git's exit code,
    an exception: git cannot run) and whose gh answers *gh* (None: logged out, an exception: no gh)."""
    config = {"user.name": name, "user.email": email}

    def run(cmd, **_kwargs):
        if cmd[0] == "gh":
            if isinstance(gh, Exception):
                raise gh
            return subprocess.CompletedProcess(cmd, 1 if gh is None else 0, gh or "", "")
        value = config[cmd[-1]]
        if isinstance(value, Exception):
            raise value
        if isinstance(value, int):
            return subprocess.CompletedProcess(cmd, value, "", "fatal")
        return subprocess.CompletedProcess(cmd, 1 if value is None else 0, "" if value is None else value + "\n", "")
    return patch.object(CommitAuthor.subprocess, "run", side_effect=run)


@pytest.mark.unit
class TestMissingKeys:
    @pytest.mark.parametrize("name, email, missing", [
        ("Marcus", "m@mlab.test", []),
        ("Marcus", None, ["user.email"]),
        (None, None, ["user.name", "user.email"]),
        ("", "m@mlab.test", ["user.name"]),                      # set to nothing is not set
        (OSError("no git"), 128, []),                            # git could not say: not counted as missing
    ])
    def test_the_keys_without_a_value_are_the_missing_ones(self, name, email, missing):
        with _machine(name, email):
            assert CommitAuthor.missing_keys() == missing

    def test_the_config_is_asked_and_not_what_git_would_make_up(self):
        """`git var GIT_AUTHOR_IDENT` answers with login@host where nothing is set: nobody's choice."""
        with _machine() as run:
            CommitAuthor.missing_keys()
        assert all(c[0][0][:2] == ["git", "config"] for c in run.call_args_list)


@pytest.mark.unit
class TestTheRunsAuthor:
    def test_an_author_git_has_stands_and_gh_is_not_asked(self):
        with _machine() as run:
            assert author_env() == {}
        assert all(c[0][0][0] == "git" for c in run.call_args_list)

    def test_a_machine_without_one_commits_as_the_gh_account_under_its_noreply_address(self):
        """A build on such a machine put a tester's own mail address into a public commit (2026-11-17)."""
        with _machine(None, None):
            env = author_env()
        assert env["GIT_AUTHOR_NAME"] == env["GIT_COMMITTER_NAME"] == "Marcus Keeper"
        assert env["GIT_AUTHOR_EMAIL"] == env["GIT_COMMITTER_EMAIL"] == NOREPLY

    def test_only_what_git_lacks_is_filled_in(self):
        """A name the owner did set is theirs: the environment would override it."""
        with _machine("Marcus", None):
            env = author_env()
        assert set(env) == {"GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"}

    def test_an_account_without_a_profile_name_commits_under_its_login(self):
        with _machine(None, None, gh="4711\tmarcus-k\t\n"):
            assert author_env()["GIT_AUTHOR_NAME"] == "marcus-k"

    @pytest.mark.parametrize("gh", [None, FileNotFoundError("gh"), "Bad credentials\n", "\tmarcus-k\tMarcus\n"],
                             ids=["logged out", "no gh", "not an account", "no id"])
    def test_without_an_account_nothing_is_made_up(self, gh):
        with _machine(None, None, gh=gh):
            assert author_env() == {}
            assert CommitAuthor.account() is None


@pytest.mark.unit
class TestARunCarriesIt:
    """`_run_claude` is where the author reaches a job: its commits are git's children of that process."""

    @staticmethod
    def _popen_env(status, author):
        process = Mock(stdout=iter([]))
        process.stderr.read.return_value = ""
        with patch.object(cqm, "claude_status", return_value=status), \
                patch.object(CommitAuthor, "env", return_value=author), \
                patch.object(cqm.subprocess, "Popen", return_value=process) as popen:
            CodeQualityMaintainer._run_claude("prompt", "label")
        return popen.call_args[1]["env"]

    AUTHOR = {"GIT_AUTHOR_NAME": "Marcus Keeper", "GIT_AUTHOR_EMAIL": NOREPLY}

    def test_the_author_joins_the_servers_own_environment(self):
        env = self._popen_env(ClaudeStatus("/opt/bin/claude", True, "logged in"), self.AUTHOR)
        assert env["GIT_AUTHOR_EMAIL"] == NOREPLY
        assert env["PATH"] == os.environ["PATH"]      # the rest is inherited, not replaced by two variables

    def test_the_author_joins_the_api_key_and_neither_pushes_the_other_out(self):
        with patch.object(cqm, "claude_api_key", return_value="sk-ant-test-key"):
            env = self._popen_env(ClaudeStatus("/opt/bin/claude", True, "on the key", api_key=True), self.AUTHOR)
        assert env[cqm.API_KEY_ENV] == "sk-ant-test-key" and env["GIT_AUTHOR_NAME"] == "Marcus Keeper"
