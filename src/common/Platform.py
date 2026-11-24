"""The operating system the process runs on, asked in this one place.

Every behaviour that differs between a Mac and a Linux server branches on `current()` /
`is_mac()` and nothing else in the tree tests `sys.platform` (tests/unit/common/test_platform.py
fails otherwise), so a test patches `Platform.current` and runs the other system's branch on
whichever machine it is on. The module imports nothing of the tree: `RequirementsChecker` and the
iMessage satellite use it before anything else is loadable.
"""
import math
import os
import platform
import sys

MAC, LINUX = "mac", "linux"
_NAMES = {"darwin": MAC, "linux": LINUX}
_LABELS = {MAC: "macOS", LINUX: "Linux"}


def current() -> str:
    """MAC, LINUX, or the raw `sys.platform` of anything else."""
    return _NAMES.get(sys.platform, sys.platform)


def is_mac() -> bool:
    return current() == MAC


def label() -> str:
    """The system's name as a user reads it: "macOS", "Linux"."""
    system = current()
    return _LABELS.get(system, system)


def sys_platform() -> str:
    """`current()` in the spelling of `sys.platform` — what a requirement's environment marker
    (`; sys_platform == "darwin"`) is written against."""
    system = current()
    return next((raw for raw, name in _NAMES.items() if name == system), system)


def is_apple_silicon() -> bool:
    return is_mac() and platform.machine() == "arm64"


def macos_version() -> int:
    """The major version of macOS (15, 26); 0 off a Mac or when it cannot be read."""
    if not is_mac():
        return 0
    try:
        return int(platform.mac_ver()[0].split(".")[0])
    except ValueError:
        return 0


def memory_gb() -> int:
    """The machine's memory in whole GB, rounded up (a kernel reports a little under what is
    installed); 0 when it cannot be read."""
    try:
        return math.ceil(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2 ** 30)
    except (ValueError, OSError):
        return 0


def machine() -> str:
    """The noun user-facing texts use for the computer the server runs on."""
    return "the Mac" if is_mac() else "the server"


def mac_only(what: str) -> str:
    """The one wording of every "this needs macOS" answer; `what` names the feature."""
    return f"{what}: only on a Mac — this server runs on {label()}."
