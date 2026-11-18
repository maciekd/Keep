"""Who the commits of a Claude Code job are by: git's own author, or the GitHub account gh is logged in as.

A fresh install has no `user.name` and `user.email` in git's config, and a job that found none used
to make an author up — on 2026-11-17 one put a tester's mail address into a public commit unasked,
and of the three builds after it one more did the same and two were stopped trying. The author is
never the job's to pick (`COMMIT_AUTHOR_RULE`): where git has one it stands, and where it has none
the run's environment carries the account that opens the pull request, under the noreply address
GitHub gives it — no private address is published, and GitHub still shows the commit as the
account's. git's config is never written. The module imports nothing of the assistant.
"""
import subprocess
from typing import Optional

from common.logging import info, error, trace
from common.Paths import REPO_DIR
from common.Updater import GIT_TIMEOUT

KEYS = ("user.name", "user.email")
# The variables git reads an author and a committer from, by the config key each stands in for.
ENVIRONMENT = {"user.name": ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"),
               "user.email": ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL")}
NOREPLY = "{id}+{login}@users.noreply.github.com"


def missing_keys() -> list[str]:
    """Which of git's author keys (`user.name`, `user.email`) have no value for the checkout.
    Asked of the config alone: where they are unset git may still make an author up from the
    login and the host name, which is nobody's choice either. A key git could not be asked
    about is not counted. The values themselves are read by nothing here and reach no log."""
    missing = []
    for key in KEYS:
        try:
            result = subprocess.run(["git", "config", "--get", key], cwd=str(REPO_DIR), capture_output=True,
                                    text=True, stdin=subprocess.DEVNULL, timeout=GIT_TIMEOUT)
        except (OSError, subprocess.SubprocessError) as e:
            error(f"Commit author: git could not be asked for {key} ({e}), not counted as missing")
            continue
        if result.returncode == 1 or (result.returncode == 0 and not (result.stdout or "").strip()):
            missing.append(key)
        elif result.returncode != 0:
            error(f"Commit author: `git config --get {key}` exited {result.returncode}, not counted as missing")
    return missing


def account() -> Optional[dict]:
    """The author the GitHub account gh is logged in as makes, by git's keys: its profile name
    (the login where it has none), as GitHub signs a commit made on the web, and its noreply
    address. None when gh is missing, logged out or does not answer."""
    try:
        result = subprocess.run(["gh", "api", "user", "--jq", '[(.id|tostring), .login, (.name // "")] | @tsv'],
                                cwd=str(REPO_DIR), capture_output=True, text=True, stdin=subprocess.DEVNULL,
                                timeout=GIT_TIMEOUT)
    except (OSError, subprocess.SubprocessError) as e:
        info(f"Commit author: gh could not be asked for its account ({e})")
        return None
    fields = (result.stdout or "").rstrip("\n").split("\t")
    if result.returncode != 0 or len(fields) != 3 or not fields[0].isdigit() or not fields[1]:
        info(f"Commit author: gh named no account (exit {result.returncode})")
        return None
    user_id, login, name = (field.strip() for field in fields)
    return {"user.name": name or login, "user.email": NOREPLY.format(id=user_id, login=login)}


def env() -> dict:
    """What a run's environment gains so that its commits have an author: nothing where git
    has one, else the gh account's for the keys git lacks — and nothing again when gh names no
    account, which `BuildPreflight` reports before a build starts."""
    missing = missing_keys()
    if not missing:
        trace("Commit author: git has one, the runs commit under it")
        return {}
    who = account()
    if who is None:
        info(f"Commit author: git has no {' and '.join(missing)} and gh names no account — a run has no author")
        return {}
    info(f"Commit author: git has no {' and '.join(missing)}, the run commits as the gh account "
         f"(its noreply address); git's config is left alone")
    return {variable: who[key] for key in missing for variable in ENVIRONMENT[key]}
