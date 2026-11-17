"""common/Platform.py is the one place that asks the OS what it is."""
import re
import sys

import pytest

from common import Platform
from common.Paths import AISERVER_DIR, SRC_DIR

# Modules that read the OS to report it, never to behave differently. Shrink, never grow: a
# behaviour switch goes through Platform, where a test can patch it.
REPORTS_THE_OS = {"src/assistant/Telemetry.py"}

_ASKS_THE_OS = re.compile(r"\bsys\.platform\b|\bos\.(?:name|uname)\b|^\s*(?:import|from)\s+platform\b"
                          r"|^\s*from\s+sys\s+import\b.*\bplatform\b", re.MULTILINE)


# Every alternative of the pattern holds one of these words: the pattern, which tries each line
# start of five megabytes of source, runs only over a file that has one.
_WORDS = ("platform", "os.name", "os.uname")


def _asks_the_os(text: str) -> bool:
    return any(word in text for word in _WORDS) and bool(_ASKS_THE_OS.search(text))


def _modules_asking_the_os() -> set[str]:
    files = [*SRC_DIR.rglob("*.py"), *AISERVER_DIR.glob("*.py")]
    assert len(files) > 100, "the glob is looking in the wrong place"
    return {str(path.relative_to(AISERVER_DIR)) for path in files
            if path != SRC_DIR / "common" / "Platform.py" and _asks_the_os(path.read_text())}


@pytest.mark.parametrize("line", ["if sys.platform == 'darwin':", "import platform", "    from platform import machine",
                                  "from sys import argv, platform", "if os.name == 'posix':", "os.uname().sysname"])
def test_each_way_of_asking_the_os_is_seen(line):
    assert _asks_the_os(f"import os\n\n{line}\n")


def test_no_module_but_platform_asks_the_os_what_it_is():
    asking = _modules_asking_the_os()
    assert asking - REPORTS_THE_OS == set(), "branch on common.Platform instead of sys.platform / the platform module"
    assert REPORTS_THE_OS - asking == set(), "no longer asks the OS — take it off the allowlist"


@pytest.mark.parametrize("raw, name", [("darwin", Platform.MAC), ("linux", Platform.LINUX), ("win32", "win32")])
def test_current_names_the_supported_systems_and_passes_any_other_through(monkeypatch, raw, name):
    monkeypatch.setattr(sys, "platform", raw)
    assert Platform.current() == name
    assert Platform.is_mac() == (raw == "darwin")
    assert Platform.sys_platform() == raw


def test_the_helpers_follow_a_patched_current(monkeypatch):
    monkeypatch.setattr(Platform, "current", lambda: Platform.MAC)
    on_the_mac = Platform.machine()
    monkeypatch.setattr(Platform, "current", lambda: Platform.LINUX)
    assert not Platform.is_mac()
    assert Platform.machine() != on_the_mac and "Mac" not in Platform.machine()
    refusal = Platform.mac_only("The Photos library")
    assert "The Photos library" in refusal and "Linux" in refusal
