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


# The machine's facts the installer's local AI gate reads (common/install/AiChoice.py).
@pytest.mark.parametrize("system, chip, silicon", [(Platform.MAC, "arm64", True), (Platform.MAC, "x86_64", False),
                                                   (Platform.LINUX, "arm64", False), (Platform.LINUX, "aarch64", False)])
def test_apple_silicon_is_an_arm_mac_and_no_other_arm_machine(monkeypatch, system, chip, silicon):
    monkeypatch.setattr(Platform, "current", lambda: system)
    monkeypatch.setattr(Platform.platform, "machine", lambda: chip)
    assert Platform.is_apple_silicon() is silicon


@pytest.mark.parametrize("system, reported, version", [(Platform.MAC, "15.6.1", 15), (Platform.MAC, "26.0", 26), (Platform.MAC, "14", 14),
                                                       (Platform.MAC, "", 0), (Platform.LINUX, "15.6.1", 0)],
                         ids=["Sequoia", "Tahoe", "no minor", "unreadable", "Linux"])
def test_the_macos_version_is_its_major_number_and_none_off_a_mac(monkeypatch, system, reported, version):
    monkeypatch.setattr(Platform, "current", lambda: system)
    monkeypatch.setattr(Platform.platform, "mac_ver", lambda: (reported, ("", "", ""), "arm64"))
    assert Platform.macos_version() == version


@pytest.mark.parametrize("system", [Platform.MAC, Platform.LINUX])
@pytest.mark.parametrize("pages, gb", [(32 * 2 ** 30 // 4096, 32), (int(31.3 * 2 ** 30) // 4096, 32), (int(3.8 * 2 ** 30) // 4096, 4),
                                       (36 * 2 ** 30 // 4096, 36)], ids=["a Mac's 32", "a kernel's 31.3", "a machine sold as 4", "36"])
def test_memory_is_what_the_machine_was_sold_with_on_either_system(monkeypatch, system, pages, gb):
    """A Linux kernel reports what is left after its own reservations: rounded up, as install.sh's ram_gb does."""
    monkeypatch.setattr(Platform, "current", lambda: system)
    monkeypatch.setattr(Platform.os, "sysconf", lambda name: {"SC_PHYS_PAGES": pages, "SC_PAGE_SIZE": 4096}[name])
    assert Platform.memory_gb() == gb


def test_memory_that_cannot_be_read_is_none_not_a_crash(monkeypatch):
    def unreadable(name):
        raise ValueError("unrecognized configuration name")
    monkeypatch.setattr(Platform.os, "sysconf", unreadable)
    assert Platform.memory_gb() == 0
