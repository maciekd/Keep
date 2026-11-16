"""The operating system the process runs on, asked in this one place.

Every behaviour that differs between a Mac and a Linux server branches on `current()` /
`is_mac()` and nothing else in the tree tests `sys.platform` (tests/unit/common/test_platform.py
fails otherwise), so a test patches `Platform.current` and runs the other system's branch on
whichever machine it is on. The module imports nothing of the tree: `RequirementsChecker` and the
iMessage satellite use it before anything else is loadable.
"""
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


def machine() -> str:
    """The noun user-facing texts use for the computer the server runs on."""
    return "the Mac" if is_mac() else "the server"


def mac_only(what: str) -> str:
    """The one wording of every "this needs macOS" answer; `what` names the feature."""
    return f"{what}: only on a Mac — this server runs on {label()}."
