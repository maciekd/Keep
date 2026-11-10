"""The running system's version, read from git: `vYYYY.MM.N` — the month of the last commit
and the number of commits behind HEAD, which only grows along master so it names one commit
(`git rev-list --reverse master | sed -n <N>p`). A release is a master commit that release.sh
moved the `release` branch to and tagged with this very string, so an install on `release`
reports the same version as master did at that commit. `detail()` adds what the short form
cannot tell: the hash (two side branches can share a count), the branch the checkout follows
(release on an install, master on the development machine) and whether the tree has local
edits. `release_notes()` is the what's new release.sh wrote into that tag's body."""
import subprocess
from functools import lru_cache
from typing import Optional

from common.Paths import REPO_DIR
from common.logging import info

UNKNOWN = "unknown"


def _git(*args: str) -> Optional[str]:
    try:
        result = subprocess.run(["git", *args], cwd=REPO_DIR, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as e:
        info(f"Version: git {args[0]} failed: {e}")
        return None
    if result.returncode != 0:
        info(f"Version: git {args[0]} exited {result.returncode}: {result.stderr.strip()}")
        return None
    return result.stdout.strip()


@lru_cache(maxsize=None)
def version() -> str:
    month = _git("log", "-1", "--format=%cd", "--date=format:%Y.%m")
    count = _git("rev-list", "--count", "HEAD")
    if not month or not count:
        info("Version: not a git checkout, the version is unknown")
        return UNKNOWN
    return f"v{month}.{count}"


@lru_cache(maxsize=None)
def release_notes() -> str:
    """The what's new release.sh wrote into the annotated tag of the running commit; "" when
    HEAD is not a tagged release (master between releases, an install on another branch), when
    the tag of this version names another commit (a side branch can share the count) or when the
    tag carries no note (a release made while Claude was unavailable)."""
    tag = version()
    if tag == UNKNOWN:
        return ""
    tagged = _git("tag", "-l", "--format=%(*objectname)", tag)   # the commit of an annotated tag, "" otherwise
    head = _git("rev-parse", "HEAD")
    if not tagged or not head or tagged != head:
        info(f"Version: HEAD is not the tagged release {tag}, no release notes")
        return ""
    body = (_git("tag", "-l", "--format=%(contents:body)", tag) or "").strip()
    info(f"Version: release {tag} carries {'a' if body else 'no'} what's new ({len(body)} chars)")
    return body


@lru_cache(maxsize=None)
def branch() -> Optional[str]:
    """The branch the checkout follows, None when detached or not a checkout."""
    name = _git("rev-parse", "--abbrev-ref", "HEAD")
    return name if name and name != "HEAD" else None


@lru_cache(maxsize=None)
def commit() -> str:
    """The full hash of the commit the process runs, "" when git cannot say. Read once, at the
    first call — a pull the process has not restarted on does not move it — and that call is
    the version line every process logs when it starts (`detail`), not whoever asks first later:
    an integration enabled after the checkout moved would name a commit the process does not
    run. It is what a server tells its iMessage satellite to follow and what the satellite
    compares that with."""
    return _git("rev-parse", "HEAD") or ""


@lru_cache(maxsize=None)
def label() -> str:
    """What the TUI shows: `Keep v2026.09.64` on a release install, `Keep v2026.09.64 dev` on
    master, the branch's own name on any other — a release needs no badge, running something
    else should be visible."""
    name = branch()
    if name is None or name == "release":
        return f"Keep {version()}"
    return f"Keep {version()} {'dev' if name == 'master' else name}"


@lru_cache(maxsize=None)
def detail() -> str:
    """`v2026.09.3868 (1b6c9ce7, release, dirty)` — for the log and /health. The branch is
    left out of a detached checkout."""
    short = commit()[:8]
    if version() == UNKNOWN or not short:
        return version()
    parts = [short]
    if branch():
        parts.append(branch())
    if _git("status", "--porcelain"):
        parts.append("dirty")
    return f"{version()} ({', '.join(parts)})"
