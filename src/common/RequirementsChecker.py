import importlib
import importlib.metadata
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from common import IntegrationDirs, Platform
from common.logging import error, info, trace
from common.Paths import AISERVER_DIR, LOG_DIR

REQUIREMENTS_FILE = AISERVER_DIR / "requirements.txt"
VOICE_REQUIREMENTS_FILE = AISERVER_DIR / "requirements-voice.txt"
PIP_LOG = LOG_DIR / "pip-update.log"

REQUIRED_PYTHON = (3, 12)
# torch alone is hundreds of MB; a pin bump of the voice stack can take a while on a slow line.
PIP_TIMEOUT = 15 * 60
# On Linux x86_64 PyPI's torch is the multi-GB CUDA build; speaker id runs on the CPU one, which this index
# serves as `<version>+cpu` — the higher version, so pip prefers it for the same pin.
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

_REQ_LINE_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)(?:\[[^\]]+\])?\s*==\s*(\S+)\s*$")
_DIRECT_REF_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*@\s*(\S+)\s*$")
_INCLUDE_RE = re.compile(r"^(?:-r|--requirement)[ =]\s*(\S+)")
# pip's rule: a `#` opens a comment at the start of a line or after whitespace, so a URL's #fragment stays
_COMMENT_RE = re.compile(r"(^|\s+)#.*$")
_ARCHIVE_EXTENSIONS = (".whl", ".tar.gz", ".tar.bz2", ".zip")


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _split_marker(line: str) -> tuple[str, str]:
    """(requirement, environment marker) of a line, the marker "" when it has none. A URL may hold
    a `;` of its own, so after a direct reference a marker needs whitespace in front (PEP 508)."""
    separator = r"\s+;\s*" if "@" in line.split(";")[0] else r"\s*;\s*"
    parts = re.split(separator, line, maxsplit=1)
    return parts[0], parts[1].strip() if len(parts) > 1 else ""


def _marker_applies(marker: str, requirement: str) -> bool:
    """Whether a pin with this environment marker is one this interpreter installs — on the
    system `Platform` names, so a test of the other system's pins patches that. Anything
    short of a clear "no" — a marker that does not parse, `packaging` not importable yet in a
    venv pip has not filled — counts as yes: the pin is then checked, and pip has the last word."""
    try:
        from packaging.markers import Marker
        return Marker(marker).evaluate({"sys_platform": Platform.sys_platform()})
    except Exception as e:
        info(f"Requirements check: marker {marker!r} of {requirement} not evaluated ({e!r}); the pin is checked")
        return True


def _same_version(installed: str, pinned: str) -> bool:
    """A pin without a local version accepts any local build of it, as pip does (PEP 440):
    torch from the CPU index is installed as `2.4.1+cpu` under the pin `2.4.1`."""
    return installed == pinned or ("+" not in pinned and installed.partition("+")[0] == pinned)


def _version_from_direct_ref(name: str, url: str) -> str | None:
    """Extract the pinned version from a direct-reference URL's archive name,
    e.g. .../en_core_web_sm-3.8.0-py3-none-any.whl -> '3.8.0'."""
    basename = url.rsplit("/", 1)[-1]
    for ext in _ARCHIVE_EXTENSIONS:
        if basename.endswith(ext):
            basename = basename[: -len(ext)]
            break
    else:
        return None
    m = re.match(r"([A-Za-z0-9_.\-]+?)-([0-9][A-Za-z0-9.!+]*)(?:-|$)", basename)
    if not m or _normalize(m.group(1)) != _normalize(name):
        return None
    return m.group(2)


class RequirementsChecker:
    """Verifies at startup that the Python version and pinned requirements match what is installed.

    Both groups are checked whatever the config enables: the assistant can switch the voice
    interface on from the chat (config_set "interfaces.voice.enabled"), and the restart that
    follows must not be the first time the install learns it lacks requirements-voice.txt.
    Any missing or mismatched package aborts the server with install instructions.
    A pin whose environment marker (`; sys_platform == "darwin"`) is false for this interpreter
    is not checked — pip does not install it here either.

    A `-r other.txt` line is followed the way pip follows it — relative to the file it stands
    in — which is how the pins an integration keeps in its own folder
    (src/integrations/<name>/requirements.txt) are checked. Each file is read once per start:
    the voice file's `-r requirements.txt` finds the base group already checked. The local
    integration folders (`extra_integration_dirs`, common/IntegrationDirs.py) are outside the
    repository, so no file of it includes theirs: each one's `requirements.txt` is a group of its own.

    The same walk serves the updater: `sync` is what restart_server and the nightly update run
    between the pull and the restart, so a release that bumps a pin installs it while the old
    server still serves and the new one never dies on this check. pip runs over the voice file,
    which includes the rest — exactly what install.sh runs.
    """

    @staticmethod
    def _groups() -> list[tuple[str, Path]]:
        return [("base", REQUIREMENTS_FILE), ("voice", VOICE_REQUIREMENTS_FILE)] + RequirementsChecker._local_groups()

    @staticmethod
    def _local_groups() -> list[tuple[str, Path]]:
        return [(f"local {path.parent.name}", path) for path in IntegrationDirs.files("requirements.txt", local_only=True)]

    @staticmethod
    def verify_or_exit():
        """Stops the start when the venv is behind the pins — after one try with pip, so a start
        that would otherwise fail (a hand pull that bumped a pin) is slow instead of dead."""
        RequirementsChecker._check_python_version()
        missing, mismatched, checked = RequirementsChecker.drift()
        if missing or mismatched:
            info(f"Requirements check: {RequirementsChecker._describe(missing, mismatched)} — "
                 f"installing before the start")
            if not RequirementsChecker.install():
                missing, mismatched, checked = RequirementsChecker.drift()
        if missing or mismatched:
            RequirementsChecker._fail(missing, mismatched, [p.name for _, p in RequirementsChecker._groups()])
            return
        trace(f"All {checked} pinned packages match installed versions "
              f"({', '.join(g for g, _ in RequirementsChecker._groups())}).")

    @staticmethod
    def drift() -> tuple[list[str], list[tuple[str, str, str]], int]:
        """(missing, mismatched, checked) across both groups, read fresh from the venv."""
        missing: list[str] = []
        mismatched: list[tuple[str, str, str]] = []
        checked = 0
        seen: set[Path] = set()
        for group, path in RequirementsChecker._groups():
            if not path.exists():
                error(f"{path} not found; skipping {group} requirements check.")
                continue
            g_missing, g_mismatched, g_checked = RequirementsChecker._check_file(path, seen)
            missing += g_missing
            mismatched += g_mismatched
            checked += g_checked
        return missing, mismatched, checked

    @staticmethod
    def sync() -> tuple[str, str]:
        """Brings the venv to the pins when they drifted. Returns (failure, done): *failure* says
        why the venv is still behind and is "" when it matches; *done* names what was installed,
        "" when there was nothing to do. A failure means the caller must not restart — the new
        process would die on verify_or_exit."""
        missing, mismatched, checked = RequirementsChecker.drift()
        if not (missing or mismatched):
            info(f"Requirements sync: all {checked} pins match the venv, nothing to install.")
            return "", ""
        change = RequirementsChecker._describe(missing, mismatched)
        info(f"Requirements sync: {change} — running pip")
        failure = RequirementsChecker.install()
        if failure:
            return failure, ""
        missing, mismatched, _ = RequirementsChecker.drift()
        if missing or mismatched:
            left = RequirementsChecker._describe(missing, mismatched)
            error(f"Requirements sync: pip finished but the venv is still behind — {left}")
            return f"pip ran but the venv is still behind the pins ({left}); see {PIP_LOG}.", ""
        info(f"Requirements sync: done — {change}")
        return "", f"installed {change}"

    @staticmethod
    def install() -> str:
        """Runs pip in this venv over the voice file (it includes everything in the repository) and
        the local integration folders' files, appending pip's output to PIP_LOG. On Linux pip is
        given PyTorch's CPU index as well, so a pin bump never swaps the CPU build of torch for the
        CUDA one. Returns "" when pip exited 0, else what went wrong."""
        cmd = [sys.executable, "-m", "pip", "install", "--no-input", "--disable-pip-version-check"]
        if Platform.current() == Platform.LINUX:
            cmd += ["--extra-index-url", TORCH_CPU_INDEX]
        cmd += ["-r", str(VOICE_REQUIREMENTS_FILE)]
        for _, path in RequirementsChecker._local_groups():
            cmd += ["-r", str(path)]
        PIP_LOG.parent.mkdir(parents=True, exist_ok=True)
        info(f"pip install: {' '.join(cmd)} (output in {PIP_LOG})")
        started = datetime.now()
        try:
            with open(PIP_LOG, "a") as log:
                log.write(f"\n===== {started:%Y-%m-%d %H:%M:%S} {' '.join(cmd)}\n")
                log.flush()
                result = subprocess.run(cmd, cwd=AISERVER_DIR, stdout=log, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, timeout=PIP_TIMEOUT)
        except subprocess.TimeoutExpired:
            error(f"pip install: no result within {PIP_TIMEOUT}s")
            return f"pip did not finish within {PIP_TIMEOUT // 60} minutes; see {PIP_LOG}."
        except Exception as e:
            error(f"pip install could not run: {e}")
            return f"pip could not run: {e}"
        # The new dist-info directories must be visible to the drift check that follows.
        importlib.invalidate_caches()
        took = (datetime.now() - started).total_seconds()
        if result.returncode != 0:
            tail = RequirementsChecker._log_tail()
            error(f"pip install exited {result.returncode} after {took:.0f}s: {tail}")
            return f"pip exited {result.returncode} ({tail}); the full output is in {PIP_LOG}."
        info(f"pip install finished in {took:.0f}s")
        return ""

    @staticmethod
    def _log_tail(lines: int = 3) -> str:
        try:
            tail = [l.strip() for l in PIP_LOG.read_text().splitlines() if l.strip()][-lines:]
        except OSError:
            return "no output"
        return " | ".join(tail) or "no output"

    @staticmethod
    def _describe(missing: list[str], mismatched: list[tuple[str, str, str]]) -> str:
        parts = [f"{name} (missing)" for name in missing]
        parts += [f"{name} {actual} → {expected}" for name, expected, actual in mismatched]
        n = len(parts)
        return f"{n} package{'s' if n != 1 else ''}: {', '.join(parts)}"

    @staticmethod
    def _check_file(path: Path, seen: set[Path] | None = None) -> tuple[list[str], list[tuple[str, str, str]], int]:
        """(missing, mismatched, checked) for *path* and every file it includes. *seen* holds
        the files already read — passed along so an include cycle, or a file two others
        include, is read once."""
        seen = set() if seen is None else seen
        seen.add(path.resolve())
        missing: list[str] = []
        mismatched: list[tuple[str, str, str]] = []
        skipped: list[str] = []
        checked = 0

        with open(path) as f:
            for raw in f:
                line = _COMMENT_RE.sub("", raw).strip()
                if not line:
                    continue
                include = _INCLUDE_RE.match(line)
                if include:
                    target = path.parent / include.group(1)
                    if target.resolve() in seen:
                        trace(f"{path.name}: {include.group(1)} is already checked")
                    elif not target.exists():
                        error(f"{target} (included from {path.name}) not found; skipping its requirements check.")
                    else:
                        i_missing, i_mismatched, i_checked = RequirementsChecker._check_file(target, seen)
                        missing += i_missing
                        mismatched += i_mismatched
                        checked += i_checked
                    continue
                line, marker = _split_marker(line)
                if marker and not _marker_applies(marker, line):
                    trace(f"{path.name}: {line} is not for this platform ({marker}), not checked")
                    continue
                m = _REQ_LINE_RE.match(line)
                if m:
                    name, expected = m.group(1), m.group(2)
                else:
                    d = _DIRECT_REF_RE.match(line)
                    if not d:
                        skipped.append(line)
                        continue
                    name, url = d.group(1), d.group(2)
                    expected = _version_from_direct_ref(name, url)
                    if expected is None:
                        trace(f"Direct reference for {name}: no version in URL, checking presence only")
                try:
                    actual = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    missing.append(name)
                    continue
                if expected is not None and not _same_version(actual, expected):
                    mismatched.append((name, expected, actual))
                checked += 1

        for s in skipped:
            trace(f"Skipped unpinned/unparseable line: {s!r}")
        return missing, mismatched, checked

    @staticmethod
    def _check_python_version():
        actual = sys.version_info[:2]
        ver = sys.version.replace('\n', ' ')
        if actual == REQUIRED_PYTHON:
            trace(f"Python version: {ver}")
            return
        required_str = ".".join(str(x) for x in REQUIRED_PYTHON)
        bar = "=" * 70
        error("")
        error(bar)
        error("  PYTHON VERSION CHECK FAILED — server will not start")
        error(bar)
        error(f"  Required: Python {required_str}.x")
        error(f"  Found:    {ver}")
        error(bar)
        error(f"  Activate the {required_str} venv or rebuild it with python{required_str}.")
        error(bar)
        error("")
        sys.exit(1)

    @staticmethod
    def _fail(missing: list[str], mismatched: list[tuple[str, str, str]], files: list[str]):
        bar = "=" * 70
        error("")
        error(bar)
        error("  DEPENDENCY CHECK FAILED — server will not start")
        error(bar)
        if missing:
            error(f"  Missing packages ({len(missing)}):")
            for name in missing:
                error(f"    - {name}")
        if mismatched:
            error(f"  Version mismatches ({len(mismatched)}):")
            for name, expected, actual in mismatched:
                error(f"    - {name}: expected {expected}, installed {actual}")
        error(bar)
        error(f"  Fix with: pip install -r {files[-1]}")
        error("  (the server tried that itself just now — see log/pip-update.log)")
        error(bar)
        error("")
        sys.exit(1)
