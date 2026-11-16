import importlib.metadata

import pytest

from common import RequirementsChecker as rc_module
from common.RequirementsChecker import (
    RequirementsChecker,
    _REQ_LINE_RE,
    _version_from_direct_ref,
)


# ---- Requirements line regex ----

def test_regex_matches_simple_pinned():
    m = _REQ_LINE_RE.match("anthropic==0.39.0")
    assert m is not None
    assert m.group(1) == "anthropic"
    assert m.group(2) == "0.39.0"


def test_regex_matches_extras_spec():
    m = _REQ_LINE_RE.match("Hypercorn[trio,uvloop]==0.17.3")
    assert m is not None
    assert m.group(1) == "Hypercorn"
    assert m.group(2) == "0.17.3"


def test_regex_tolerates_whitespace():
    m = _REQ_LINE_RE.match("  google-cloud-speech == 2.27.0  ")
    assert m is not None
    assert m.group(1) == "google-cloud-speech"
    assert m.group(2) == "2.27.0"


@pytest.mark.parametrize("line", [
    "anthropic>=0.39.0",        # unpinned lower-bound
    "anthropic~=0.39.0",        # compatible release
    "anthropic",                # bare name
    "anthropic==",              # missing version
    "==0.39.0",                 # missing name
    "git+https://example.com/repo.git",
])
def test_regex_rejects_non_pinned_lines(line):
    assert _REQ_LINE_RE.match(line) is None


_REAL_INSTALL = RequirementsChecker.install


@pytest.fixture(autouse=True)
def no_pip(monkeypatch):
    """verify_or_exit tries pip before giving up: in a test pip is a stub that fixes nothing
    unless the test installs its own (see _Venv.pip) or asks for the real one (real_pip)."""
    monkeypatch.setattr(RequirementsChecker, "install", staticmethod(lambda: "no pip in tests"))


@pytest.fixture
def real_pip(monkeypatch):
    """The real install(), with subprocess.run patched by the test."""
    monkeypatch.setattr(RequirementsChecker, "install", staticmethod(_REAL_INSTALL))


# ---- Direct-reference version extraction ----

SPACY_URL = (
    "https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
)


def test_direct_ref_version_from_wheel_url():
    assert _version_from_direct_ref("en_core_web_sm", SPACY_URL) == "3.8.0"


def test_direct_ref_version_from_sdist_url():
    url = "https://example.com/dist/pkg-a-1.2.3.tar.gz"
    assert _version_from_direct_ref("pkg-a", url) == "1.2.3"


def test_direct_ref_version_normalizes_name_separators():
    url = "https://example.com/dist/pkg_a-1.2.3-py3-none-any.whl"
    assert _version_from_direct_ref("pkg-a", url) == "1.2.3"


def test_direct_ref_version_none_when_name_mismatch():
    assert _version_from_direct_ref("other-pkg", SPACY_URL) is None


def test_direct_ref_version_none_for_non_archive_url():
    url = "https://example.com/repo.git"
    assert _version_from_direct_ref("pkg-a", url) is None


# ---- Python version check ----

def test_check_python_version_exits_on_mismatch(monkeypatch):
    monkeypatch.setattr(rc_module, "REQUIRED_PYTHON", (9, 99))
    with pytest.raises(SystemExit) as exc_info:
        RequirementsChecker._check_python_version()
    assert exc_info.value.code == 1


# ---- verify_or_exit() ----

@pytest.fixture(autouse=True)
def _no_voice_group(tmp_path, monkeypatch):
    """The real requirements-voice.txt is not under test here: point the group at a missing file."""
    monkeypatch.setattr(rc_module, "VOICE_REQUIREMENTS_FILE", tmp_path / "no-voice.txt")


def _write_reqs(tmp_path, body: str):
    f = tmp_path / "requirements.txt"
    f.write_text(body)
    return f


def test_verify_or_exit_missing_file_returns_without_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(rc_module, "REQUIREMENTS_FILE", tmp_path / "nope.txt")
    # Should NOT raise SystemExit
    RequirementsChecker.verify_or_exit()


def test_verify_or_exit_all_match_does_not_exit(tmp_path, monkeypatch):
    reqs = _write_reqs(tmp_path, "pkg-a==1.0.0\npkg-b==2.0.0\n")
    monkeypatch.setattr(rc_module, "REQUIREMENTS_FILE", reqs)

    versions = {"pkg-a": "1.0.0", "pkg-b": "2.0.0"}
    monkeypatch.setattr(
        importlib.metadata, "version", lambda name: versions[name]
    )

    RequirementsChecker.verify_or_exit()


def test_verify_or_exit_exits_on_missing_package(tmp_path, monkeypatch):
    reqs = _write_reqs(tmp_path, "pkg-a==1.0.0\n")
    monkeypatch.setattr(rc_module, "REQUIREMENTS_FILE", reqs)

    def _raise(_name):
        raise importlib.metadata.PackageNotFoundError("pkg-a")

    monkeypatch.setattr(importlib.metadata, "version", _raise)

    with pytest.raises(SystemExit) as exc_info:
        RequirementsChecker.verify_or_exit()
    assert exc_info.value.code == 1


def test_verify_or_exit_exits_on_version_mismatch(tmp_path, monkeypatch):
    reqs = _write_reqs(tmp_path, "pkg-a==1.0.0\n")
    monkeypatch.setattr(rc_module, "REQUIREMENTS_FILE", reqs)
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "9.9.9")

    with pytest.raises(SystemExit) as exc_info:
        RequirementsChecker.verify_or_exit()
    assert exc_info.value.code == 1


def test_verify_or_exit_ignores_blanks_and_comments(tmp_path, monkeypatch):
    body = (
        "\n"
        "# top-level comment\n"
        "pkg-a==1.0.0\n"
        "   \n"
        "# inline-style comment\n"
        "pkg-b==2.0.0\n"
    )
    reqs = _write_reqs(tmp_path, body)
    monkeypatch.setattr(rc_module, "REQUIREMENTS_FILE", reqs)

    seen = []

    def _version(name):
        seen.append(name)
        return {"pkg-a": "1.0.0", "pkg-b": "2.0.0"}[name]

    monkeypatch.setattr(importlib.metadata, "version", _version)

    RequirementsChecker.verify_or_exit()
    assert seen == ["pkg-a", "pkg-b"]


def test_verify_or_exit_checks_a_pin_with_a_trailing_comment(tmp_path, monkeypatch):
    reqs = _write_reqs(tmp_path, "pkg-a==1.0.0   # what needs it\n")
    monkeypatch.setattr(rc_module, "REQUIREMENTS_FILE", reqs)
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "9.9.9")

    with pytest.raises(SystemExit) as exc_info:
        RequirementsChecker.verify_or_exit()
    assert exc_info.value.code == 1


def test_verify_or_exit_keeps_the_fragment_of_a_direct_reference_url(tmp_path, monkeypatch):
    reqs = _write_reqs(tmp_path, f"en_core_web_sm @ {SPACY_URL}#sha256=abc123  # the NER model\n")
    monkeypatch.setattr(rc_module, "REQUIREMENTS_FILE", reqs)
    captured = []

    def _version(name):
        captured.append(name)
        return "3.8.0"

    monkeypatch.setattr(importlib.metadata, "version", _version)

    RequirementsChecker.verify_or_exit()
    assert captured == ["en_core_web_sm"]


def test_verify_or_exit_extras_spec_validates_against_base_name(tmp_path, monkeypatch):
    reqs = _write_reqs(tmp_path, "Hypercorn[trio,uvloop]==0.17.3\n")
    monkeypatch.setattr(rc_module, "REQUIREMENTS_FILE", reqs)

    captured = []

    def _version(name):
        captured.append(name)
        return "0.17.3"

    monkeypatch.setattr(importlib.metadata, "version", _version)

    RequirementsChecker.verify_or_exit()
    assert captured == ["Hypercorn"]  # extras spec must be stripped


def test_verify_or_exit_skips_unpinned_lines_without_exit(tmp_path, monkeypatch):
    # Unpinned lines should be skipped (logged at trace level) — they must
    # NOT trigger a failure, otherwise legitimate non-pinned lines (e.g. a
    # future git+ URL) would crash startup.
    reqs = _write_reqs(tmp_path, "pkg-a>=1.0.0\npkg-b==2.0.0\n")
    monkeypatch.setattr(rc_module, "REQUIREMENTS_FILE", reqs)

    seen = []

    def _version(name):
        seen.append(name)
        return "2.0.0"

    monkeypatch.setattr(importlib.metadata, "version", _version)

    RequirementsChecker.verify_or_exit()
    assert seen == ["pkg-b"]  # unpinned line never checked


def test_verify_or_exit_checks_direct_reference_version(tmp_path, monkeypatch):
    reqs = _write_reqs(tmp_path, f"en_core_web_sm @ {SPACY_URL}\n")
    monkeypatch.setattr(rc_module, "REQUIREMENTS_FILE", reqs)

    seen = []

    def _version(name):
        seen.append(name)
        return "3.8.0"

    monkeypatch.setattr(importlib.metadata, "version", _version)

    RequirementsChecker.verify_or_exit()
    assert seen == ["en_core_web_sm"]


def test_verify_or_exit_exits_on_direct_reference_mismatch(tmp_path, monkeypatch):
    reqs = _write_reqs(tmp_path, f"en_core_web_sm @ {SPACY_URL}\n")
    monkeypatch.setattr(rc_module, "REQUIREMENTS_FILE", reqs)
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "3.7.0")

    with pytest.raises(SystemExit) as exc_info:
        RequirementsChecker.verify_or_exit()
    assert exc_info.value.code == 1


def test_verify_or_exit_direct_reference_without_version_checks_presence(tmp_path, monkeypatch):
    reqs = _write_reqs(tmp_path, "pkg-a @ https://example.com/repo.git\n")
    monkeypatch.setattr(rc_module, "REQUIREMENTS_FILE", reqs)

    seen = []

    def _version(name):
        seen.append(name)
        return "0.0.1"  # any version accepted when URL has no version

    monkeypatch.setattr(importlib.metadata, "version", _version)

    RequirementsChecker.verify_or_exit()
    assert seen == ["pkg-a"]


# ---- Grouped (base + voice) checks ----

def _write_voice_reqs(tmp_path, body: str):
    f = tmp_path / "requirements-voice.txt"
    f.write_text("-r requirements.txt\n" + body)
    return f


def _point_at(monkeypatch, base, voice):
    monkeypatch.setattr(rc_module, "REQUIREMENTS_FILE", base)
    monkeypatch.setattr(rc_module, "VOICE_REQUIREMENTS_FILE", voice)


def test_both_groups_are_checked(tmp_path, monkeypatch):
    base = _write_reqs(tmp_path, "pkg-a==1.0.0\n")
    voice = _write_voice_reqs(tmp_path, "torch-ish==2.0.0\n")
    _point_at(monkeypatch, base, voice)

    seen = []

    def _version(name):
        seen.append(name)
        return {"pkg-a": "1.0.0", "torch-ish": "2.0.0"}[name]

    monkeypatch.setattr(importlib.metadata, "version", _version)

    RequirementsChecker.verify_or_exit()
    assert seen == ["pkg-a", "torch-ish"]  # the -r include line is not re-resolved


def test_missing_voice_package_exits_even_with_voice_off(tmp_path, monkeypatch):
    base = _write_reqs(tmp_path, "pkg-a==1.0.0\n")
    voice = _write_voice_reqs(tmp_path, "torch-ish==2.0.0\n")
    _point_at(monkeypatch, base, voice)

    def _version(name):
        if name == "torch-ish":
            raise importlib.metadata.PackageNotFoundError(name)
        return "1.0.0"

    monkeypatch.setattr(importlib.metadata, "version", _version)

    with pytest.raises(SystemExit) as exc_info:
        RequirementsChecker.verify_or_exit()
    assert exc_info.value.code == 1


def test_real_requirements_files_parse_and_voice_includes_base():
    # the autouse fixture points the voice group at a missing file; this test wants the real ones
    voice_lines = (rc_module.AISERVER_DIR / "requirements-voice.txt").read_text().splitlines()
    assert "-r requirements.txt" in voice_lines
    base_names = {l.split("==")[0] for l in (rc_module.AISERVER_DIR / "requirements.txt").read_text().splitlines()
                  if "==" in l}
    voice_names = {l.split("==")[0].split(" @ ")[0] for l in voice_lines
                   if "==" in l or " @ " in l}
    assert not (base_names & voice_names), "a package must live in exactly one group"
    for name in ("torch", "mlx-whisper", "sounddevice", "spacy", "en_core_web_sm"):
        assert name in voice_names


def test_every_line_of_the_real_requirements_files_is_one_the_checker_reads(monkeypatch):
    # a line the checker cannot parse is skipped with a trace, so a pin could go unchecked unnoticed
    skipped = []
    monkeypatch.setattr(rc_module, "trace", lambda message: skipped.append(message) if "Skipped" in message else None)
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0")
    seen = set()
    for name in ("requirements.txt", "requirements-voice.txt"):
        RequirementsChecker._check_file(rc_module.AISERVER_DIR / name, seen)
    assert not skipped


# ---- -r includes (an integration's own pins, src/integrations/<name>/requirements.txt) ----

def _installed(monkeypatch, versions: dict) -> list:
    """Pretend exactly *versions* are installed; returns the names asked for, in order."""
    asked = []

    def _version(name):
        asked.append(name)
        if name not in versions:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(importlib.metadata, "version", _version)
    return asked


def _tree(tmp_path, files: dict):
    for name, body in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    return tmp_path / "requirements.txt"


def test_an_included_file_is_checked_where_its_line_stands(tmp_path, monkeypatch):
    reqs = _tree(tmp_path, {"requirements.txt": "lamp-driver==1.0.0\n-r folders/kettle/requirements.txt\nradio-tuner==3.0.0\n",
                            "folders/kettle/requirements.txt": "kettle-sdk==2.0.0\n"})
    asked = _installed(monkeypatch, {"lamp-driver": "1.0.0", "kettle-sdk": "2.0.0", "radio-tuner": "3.0.0"})

    assert RequirementsChecker._check_file(reqs) == ([], [], 3)
    assert asked == ["lamp-driver", "kettle-sdk", "radio-tuner"]


def test_a_nested_include_is_relative_to_the_file_it_stands_in(tmp_path, monkeypatch):
    reqs = _tree(tmp_path, {"requirements.txt": "-r folders/kettle/requirements.txt\n",
                            "folders/kettle/requirements.txt": "-r parts/heater.txt\n",
                            "folders/kettle/parts/heater.txt": "kettle-heater==4.1.0\n",
                            "parts/heater.txt": "wrong-file==9.9.9\n"})
    asked = _installed(monkeypatch, {"kettle-heater": "4.1.0"})

    assert RequirementsChecker._check_file(reqs) == ([], [], 1)
    assert asked == ["kettle-heater"]


def test_a_wrong_version_in_an_included_file_stops_the_server(tmp_path, monkeypatch):
    reqs = _tree(tmp_path, {"requirements.txt": "lamp-driver==1.0.0\n-r folders/kettle/requirements.txt\n",
                            "folders/kettle/requirements.txt": "kettle-sdk==2.0.0\nradio-tuner==3.0.0\n"})
    monkeypatch.setattr(rc_module, "REQUIREMENTS_FILE", reqs)
    _installed(monkeypatch, {"lamp-driver": "1.0.0", "kettle-sdk": "2.5.0"})

    assert RequirementsChecker._check_file(reqs) == (["radio-tuner"], [("kettle-sdk", "2.0.0", "2.5.0")], 2)
    with pytest.raises(SystemExit) as exc_info:
        RequirementsChecker.verify_or_exit()
    assert exc_info.value.code == 1


def test_a_missing_include_is_skipped_and_the_rest_is_still_checked(tmp_path, monkeypatch):
    reqs = _tree(tmp_path, {"requirements.txt": "-r folders/kettle/requirements.txt\nlamp-driver==1.0.0\n"})
    monkeypatch.setattr(rc_module, "REQUIREMENTS_FILE", reqs)
    asked = _installed(monkeypatch, {"lamp-driver": "1.0.0"})

    RequirementsChecker.verify_or_exit()   # no SystemExit, no exception
    assert asked == ["lamp-driver"]


def test_files_that_include_each_other_are_each_read_once(tmp_path, monkeypatch):
    reqs = _tree(tmp_path, {"requirements.txt": "lamp-driver==1.0.0\n-r other.txt\n",
                            "other.txt": "-r requirements.txt\nkettle-sdk==2.0.0\n-r other.txt\n"})
    asked = _installed(monkeypatch, {"lamp-driver": "1.0.0", "kettle-sdk": "2.0.0"})

    assert RequirementsChecker._check_file(reqs) == ([], [], 2)
    assert asked == ["lamp-driver", "kettle-sdk"]


def _included_files(path, seen=None) -> set:
    seen = set() if seen is None else seen
    seen.add(path.resolve())
    for line in path.read_text().splitlines():
        if line.startswith("-r "):
            target = (path.parent / line[3:].strip())
            if target.resolve() not in seen:
                _included_files(target, seen)
    return seen


def test_every_integration_folders_requirements_are_installed_with_the_base_set():
    """`pip install -r requirements.txt` is the documented install: a folder's pins the root
    file does not include would never be installed nor checked."""
    from common.Paths import INTEGRATIONS_DIR
    included = _included_files(rc_module.AISERVER_DIR / "requirements.txt")
    folder_files = {p.resolve() for p in INTEGRATIONS_DIR.glob("*/requirements.txt")}
    assert folder_files, "no integration folder pins anything — the glob is looking in the wrong place"
    assert folder_files <= included, f"not included from requirements.txt: {sorted(map(str, folder_files - included))}"

    pins = [line.split("==")[0].strip().lower()
            for path in _included_files(rc_module.AISERVER_DIR / "requirements-voice.txt")
            for line in path.read_text().splitlines() if "==" in line and not line.lstrip().startswith("#")]
    assert sorted(set(pins)) == sorted(pins), "a package is pinned in two files"


# ---- sync: what restart_server runs between the pull and the restart ----

class _Venv:
    """A fake venv: *installed* maps package → version and a pip run can rewrite it."""

    def __init__(self, monkeypatch, tmp_path, pins: str, installed: dict):
        self.installed = installed
        self.pip_calls = []
        monkeypatch.setattr(rc_module, "REQUIREMENTS_FILE", _write_reqs(tmp_path, pins))
        monkeypatch.setattr(rc_module, "VOICE_REQUIREMENTS_FILE", tmp_path / "no-voice.txt")
        monkeypatch.setattr(rc_module, "PIP_LOG", tmp_path / "pip-update.log")

        def version(name):
            if name not in self.installed:
                raise importlib.metadata.PackageNotFoundError(name)
            return self.installed[name]
        monkeypatch.setattr(importlib.metadata, "version", version)

    def pip(self, monkeypatch, after: dict | None = None, failure: str = ""):
        """pip that leaves the venv as *after* (unchanged when None) and reports *failure*."""
        def install():
            self.pip_calls.append(1)
            if after is not None:
                self.installed.clear()
                self.installed.update(after)
            return failure
        monkeypatch.setattr(RequirementsChecker, "install", staticmethod(install))


def test_sync_runs_no_pip_when_the_venv_matches_the_pins(tmp_path, monkeypatch):
    venv = _Venv(monkeypatch, tmp_path, "pkg-a==1.0.0\n", {"pkg-a": "1.0.0"})
    venv.pip(monkeypatch)
    assert RequirementsChecker.sync() == ("", "")
    assert venv.pip_calls == []


def test_sync_installs_a_bumped_pin_and_reports_it(tmp_path, monkeypatch):
    venv = _Venv(monkeypatch, tmp_path, "pkg-a==2.0.0\npkg-b==1.0.0\n", {"pkg-a": "1.0.0"})
    venv.pip(monkeypatch, after={"pkg-a": "2.0.0", "pkg-b": "1.0.0"})
    failure, installed = RequirementsChecker.sync()
    assert failure == ""
    assert len(venv.pip_calls) == 1
    assert "pkg-a" in installed and "2.0.0" in installed and "pkg-b" in installed


def test_sync_reports_a_failed_pip_and_does_not_recheck(tmp_path, monkeypatch):
    venv = _Venv(monkeypatch, tmp_path, "pkg-a==2.0.0\n", {"pkg-a": "1.0.0"})
    venv.pip(monkeypatch, failure="pip exited 1 (no matching distribution)")
    failure, installed = RequirementsChecker.sync()
    assert "no matching distribution" in failure and installed == ""


def test_sync_fails_when_pip_succeeds_but_the_venv_is_still_behind(tmp_path, monkeypatch):
    venv = _Venv(monkeypatch, tmp_path, "pkg-a==2.0.0\n", {"pkg-a": "1.0.0"})
    venv.pip(monkeypatch)   # exits 0 without changing anything
    failure, installed = RequirementsChecker.sync()
    assert "still behind" in failure and "pkg-a" in failure and installed == ""


def test_verify_or_exit_runs_pip_and_starts_once_the_venv_matches(tmp_path, monkeypatch):
    venv = _Venv(monkeypatch, tmp_path, "pkg-a==2.0.0\n", {"pkg-a": "1.0.0"})
    venv.pip(monkeypatch, after={"pkg-a": "2.0.0"})
    RequirementsChecker.verify_or_exit()
    assert len(venv.pip_calls) == 1


def test_verify_or_exit_still_exits_when_pip_does_not_fix_it(tmp_path, monkeypatch):
    venv = _Venv(monkeypatch, tmp_path, "pkg-a==2.0.0\n", {"pkg-a": "1.0.0"})
    venv.pip(monkeypatch, failure="pip exited 1")
    with pytest.raises(SystemExit):
        RequirementsChecker.verify_or_exit()


def test_verify_or_exit_runs_no_pip_when_the_venv_matches(tmp_path, monkeypatch):
    venv = _Venv(monkeypatch, tmp_path, "pkg-a==1.0.0\n", {"pkg-a": "1.0.0"})
    venv.pip(monkeypatch)
    RequirementsChecker.verify_or_exit()
    assert venv.pip_calls == []


def test_install_runs_pip_of_this_venv_over_the_voice_file_and_logs_its_output(tmp_path, monkeypatch, real_pip):
    """The voice file includes the base file and every integration folder's pins — the one
    command install.sh runs, so an update installs exactly what a fresh install does."""
    import subprocess
    import sys
    voice = tmp_path / "requirements-voice.txt"
    voice.write_text("-r requirements.txt\n")
    log = tmp_path / "pip-update.log"
    monkeypatch.setattr(rc_module, "VOICE_REQUIREMENTS_FILE", voice)
    monkeypatch.setattr(rc_module, "PIP_LOG", log)
    runs = []

    def run(cmd, **kwargs):
        runs.append((cmd, kwargs))
        kwargs["stdout"].write("Successfully installed pkg-a-2.0.0\n")
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(rc_module.subprocess, "run", run)

    assert RequirementsChecker.install() == ""
    (cmd, kwargs), = runs
    assert cmd[:3] == [sys.executable, "-m", "pip"] and cmd[-2:] == ["-r", str(voice)]
    assert kwargs["stdin"] is subprocess.DEVNULL and kwargs["timeout"] >= 60
    assert "Successfully installed pkg-a-2.0.0" in log.read_text()


def test_install_reports_the_tail_of_a_failed_pip(tmp_path, monkeypatch, real_pip):
    import subprocess
    monkeypatch.setattr(rc_module, "PIP_LOG", tmp_path / "pip-update.log")

    def run(cmd, **kwargs):
        kwargs["stdout"].write("Collecting pkg-a\nERROR: No matching distribution found for pkg-a==9.9.9\n")
        return subprocess.CompletedProcess(cmd, 1)
    monkeypatch.setattr(rc_module.subprocess, "run", run)

    failure = RequirementsChecker.install()
    assert "No matching distribution" in failure and "pip-update.log" in failure


def test_install_reports_a_timeout(tmp_path, monkeypatch, real_pip):
    import subprocess
    monkeypatch.setattr(rc_module, "PIP_LOG", tmp_path / "pip-update.log")

    def run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
    monkeypatch.setattr(rc_module.subprocess, "run", run)

    assert "did not finish" in RequirementsChecker.install()


# ---- environment markers and Linux (docs/plans/LINUX_PLAN.md §1) ----

NOWHERE = 'sys_platform == "plan9"'
EVERYWHERE = 'sys_platform != "plan9"'


def test_a_pin_whose_marker_is_false_here_is_neither_missing_nor_looked_up(tmp_path, monkeypatch):
    reqs = _write_reqs(tmp_path, f"pkg-a==1.0.0\npkg-elsewhere==2.0.0 ; {NOWHERE}   # another system's\n")
    asked = _installed(monkeypatch, {"pkg-a": "1.0.0"})
    assert RequirementsChecker._check_file(reqs) == ([], [], 1)
    assert asked == ["pkg-a"]


def test_a_pin_whose_marker_is_true_here_is_checked_like_any_other(tmp_path, monkeypatch):
    reqs = _write_reqs(tmp_path, f"pkg-a==1.0.0;{EVERYWHERE}\npkg-b[extra]==2.0.0 ; {EVERYWHERE}\npkg-c==3.0.0 ; {EVERYWHERE}\n")
    _installed(monkeypatch, {"pkg-a": "1.0.0", "pkg-b": "2.5.0"})
    assert RequirementsChecker._check_file(reqs) == (["pkg-c"], [("pkg-b", "2.0.0", "2.5.0")], 2)


def test_a_marker_nobody_can_evaluate_leaves_the_pin_checked(tmp_path, monkeypatch):
    reqs = _write_reqs(tmp_path, "pkg-a==1.0.0 ; phase_of_moon == 'full'\n")
    _installed(monkeypatch, {})
    assert RequirementsChecker._check_file(reqs) == (["pkg-a"], [], 0)


def test_a_direct_reference_keeps_the_semicolon_of_its_url_and_obeys_its_marker(tmp_path, monkeypatch):
    url = "https://files.example/get;v=3/pkg_a-1.2.0-py3-none-any.whl"
    reqs = _write_reqs(tmp_path, f"pkg-a @ {url}\npkg-a @ {url} ; {EVERYWHERE}\npkg-b @ {url} ; {NOWHERE}\n")
    asked = _installed(monkeypatch, {"pkg-a": "1.2.0"})
    assert RequirementsChecker._check_file(reqs) == ([], [], 2)
    assert asked == ["pkg-a", "pkg-a"]


def test_a_local_build_of_the_pinned_version_matches_the_pin(tmp_path, monkeypatch):
    """torch from PyTorch's CPU index reports itself as 2.4.1+cpu; pip takes that for ==2.4.1."""
    reqs = _write_reqs(tmp_path, "pkg-a==2.4.1\npkg-b==2.4.1\npkg-c==2.4.1+cpu\n")
    _installed(monkeypatch, {"pkg-a": "2.4.1+cpu", "pkg-b": "2.4.10+cpu", "pkg-c": "2.4.1"})
    assert RequirementsChecker._check_file(reqs) == (
        [], [("pkg-b", "2.4.1", "2.4.10+cpu"), ("pkg-c", "2.4.1+cpu", "2.4.1")], 3)


def _pip_command(tmp_path, monkeypatch, system) -> list:
    import subprocess
    from common import Platform
    monkeypatch.setattr(Platform, "current", lambda: system)
    monkeypatch.setattr(rc_module, "PIP_LOG", tmp_path / "pip-update.log")
    commands = []
    monkeypatch.setattr(rc_module.subprocess, "run",
                        lambda cmd, **kwargs: commands.append(cmd) or subprocess.CompletedProcess(cmd, 0))
    assert RequirementsChecker.install() == ""
    return commands[0]


def test_on_linux_pip_is_given_the_cpu_torch_index_and_on_the_mac_its_command_is_what_it_was(
        tmp_path, monkeypatch, real_pip):
    from common import Platform
    on_the_mac = _pip_command(tmp_path, monkeypatch, Platform.MAC)
    on_linux = _pip_command(tmp_path, monkeypatch, Platform.LINUX)
    assert not any("index" in argument for argument in on_the_mac)
    index = on_linux.index("--extra-index-url")
    assert on_linux[index + 1].startswith("https://download.pytorch.org/") and on_linux[index + 1].endswith("/cpu")
    assert on_linux[:index] + on_linux[index + 2:] == on_the_mac


def test_a_linux_server_is_not_asked_for_what_only_the_macs_own_audio_and_apis_need(monkeypatch):
    """The real files, read as Linux reads them: no pyobjc, no MLX Whisper, no audio device —
    and still everything the phones' side of voice runs on."""
    from common import Platform
    monkeypatch.setattr(Platform, "current", lambda: Platform.LINUX)
    asked = []
    monkeypatch.setattr(importlib.metadata, "version", lambda name: asked.append(name.lower()) or "0")
    seen = set()
    for name in ("requirements.txt", "requirements-voice.txt"):
        RequirementsChecker._check_file(rc_module.AISERVER_DIR / name, seen)
    assert not {"pyobjc", "mlx-whisper", "sounddevice", "silero-vad", "lingua-language-detector"} & set(asked)
    assert {"torch", "torchaudio", "speechbrain", "soundfile", "spacy", "bleak"} <= set(asked)
    monkeypatch.setattr(Platform, "current", lambda: Platform.MAC)
    on_the_mac = []
    monkeypatch.setattr(importlib.metadata, "version", lambda name: on_the_mac.append(name.lower()) or "0")
    seen = set()
    for name in ("requirements.txt", "requirements-voice.txt"):
        RequirementsChecker._check_file(rc_module.AISERVER_DIR / name, seen)
    assert {"pyobjc", "mlx-whisper", "sounddevice", "silero-vad", "lingua-language-detector"} <= set(on_the_mac)
