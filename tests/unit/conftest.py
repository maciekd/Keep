import gc
import pytest
from unittest.mock import patch
import sys
import os
from pathlib import Path

# Collection allocates the whole tree (every module, 7000 items) and none of it is garbage, yet
# the collector keeps walking it: off until pytest_collection_finish freezes that heap.
gc.disable()

# Add the parent directory to the path so we can import AI modules
# Dynamically determine the path based on where this conftest.py is located
test_dir = Path(__file__).parent.resolve()  # .../AIServer/tests/unit/
aiserver_dir = test_dir.parent.parent  # .../AIServer/

# Change to AIServer directory
os.chdir(aiserver_dir)

# Add AIServer/src to Python path (mirrors pytest.ini pythonpath for non-pytest imports)
sys.path.insert(0, str(aiserver_dir / "src"))

# No unit test plays or records: importing the real sounddevice initialises PortAudio, and eight
# workers doing that at once queue up behind coreaudiod (~0.3 s of the wall time). Tests that
# care about a call patch it where it is used, exactly as they would the real module.
import types
from unittest.mock import MagicMock


class _SoundDeviceStub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = MagicMock(name=f"sounddevice.{name}")
        setattr(self, name, value)
        return value


_sounddevice = _SoundDeviceStub("sounddevice")
_sounddevice.PortAudioError = type("PortAudioError", (Exception,), {})
sys.modules["sounddevice"] = _sounddevice

# No unit test relaunches the server: `bash start.sh <pid>` is how CodeQualityMaintainer.restart
# relaunches the process it runs in, and a test that reaches it unstubbed (a playback test did,
# through the stall forensics and AudioRecovery) kills the developer's running server and the
# pytest worker, then opens a Terminal window. A subclass, so a test patching Popen or checking
# isinstance sees nothing unusual; the maintainer's own restart tests patch Popen themselves.
# common/Updater relaunches through whatever script its caller names and is what pulls, switches
# branches and runs pip, so the same guard refuses any script handed this process's pid, a git
# command that moves the checkout the suite runs in, and a pip install into its venv.
import subprocess

_GIT_MOVES = {"pull", "fetch", "checkout", "switch", "merge", "rebase", "reset", "stash", "push"}


class _NoRelaunchPopen(subprocess.Popen):
    def __init__(self, args, *rest, **kwargs):
        words = list(map(str, args)) if isinstance(args, (list, tuple)) else []
        if words[:1] == ["bash"] and (words[1:2] == ["start.sh"] or words[2:3] == [str(os.getpid())]):
            raise RuntimeError(f"a unit test tried to relaunch the server: {words}")
        if words[:1] == ["git"] and _GIT_MOVES.intersection(words[1:2]) \
                and Path(kwargs.get("cwd") or os.getcwd()).resolve() == aiserver_dir:
            raise RuntimeError(f"a unit test tried to move the checkout: {words}")
        if words[1:4] == ["-m", "pip", "install"]:
            raise RuntimeError(f"a unit test tried to install into the venv: {words}")
        if words[:1] == ["osascript"] and any('application "Messages"' in w for w in words):
            raise RuntimeError("a unit test tried to script the real Messages.app")
        # The satellite brings the Mac to its `watchdog` key and registers a login item by itself
        # at each start (common/imessage/Satellite.stay_up): unstubbed, that would load or unload
        # the developer's own launchd agent and add a login item to their account. Every launchctl
        # verb is refused, the read-only ones too, and so is the applet's build: `Watchdog.sync`
        # asks launchd what is loaded before it builds, writes or removes anything, so it gets no
        # further — and the plist it would write is not the one under ~/Library/LaunchAgents
        # (`isolated_launch_agent` below).
        if words[:1] == ["launchctl"]:
            raise RuntimeError(f"a unit test tried to reach this Mac's launchd agents: {words}")
        if words[:1] == ["osacompile"]:
            raise RuntimeError(f"a unit test tried to build an applet in the checkout: {words}")
        if words[:1] == ["osascript"] and any("login item" in w for w in words):
            raise RuntimeError("a unit test tried to read or change this Mac's login items")
        super().__init__(args, *rest, **kwargs)


subprocess.Popen = _NoRelaunchPopen

# openai is imported here, at collection, rather than by the first ChatGPTLLM test on each
# worker: the import is ~80 ms, and the test that paid it was recorded as slow, moved to the
# front of the next run's collection and paid it again. anthropic is already imported at module
# level by the ClaudeLLM tests; google.genai is left to the few workers that need it.
import openai  # noqa: F401

# Disable all file logging during tests
import common.logging
common.logging.disable_file_logging = True

# The suite runs against whatever config.yaml the machine has, so the keys the tests' literals
# depend on are pinned to the cast's values here: the owner is Marcus ("Marcus to Alice: …"),
# the wake word is Michelle with the example config's aliases and near-misses (the transcript
# gate's constants read these once, at import, so the pin must be in place before collection
# imports them — not in a fixture), and the AI footer is on. Pinned under `get` of the shared
# instance only: a test patching `config.get` still wins, and the ConfigManager tests' own
# instances still read their files.
import common.ConfigManager as _config_module

CAST_CONFIG = {
    "owner_name": "Marcus",
    "wake_word": "Michelle",
    "wake_word_aliases": ["michel"],
    "wake_word_near_misses": ["mission", "missions", "we shall"],
    "ai_message_footer": True,
    # The install's local integration folders are not the repository's: no test sees them.
    "extra_integration_dirs": [],
    # An install whose iMessage account is on a satellite names its address here: no test may build
    # a host that would read that Mac's real messages. The tests of the remote host patch the key.
    "imessage.host": "local",
    # The same for the iMessage chat interface, which is also off for the suite whatever the
    # install says: its tests build the transport over a fake host and patch the keys they need.
    "imessage_chat.host": "local",
    "interfaces.imessage.enabled": False,
}
_real_config_get = _config_module.config.get


def _cast_config_get(key, *args, **kwargs):
    return CAST_CONFIG[key] if key in CAST_CONFIG else _real_config_get(key, *args, **kwargs)


_config_module.config.get = _cast_config_get

# Cheap vault keys: every test vault derived a real scrypt key (~90 ms), about a third of
# the suite's time. Sixteen rounds derive in microseconds and no test cares about strength.
import common.vault.Vault
common.vault.Vault.SCRYPT_N = 2 ** 4

# No fsync per commit: hundreds of tests write a throwaway SQLite file under tmp_path, and
# waiting for the disk on each commit was ~0.4 s of the wall time. Nothing here survives a crash.
import sqlite3
_sqlite_connect = sqlite3.connect


# The Mac's own Messages database and AddressBook hold real people. A test reads a fixture
# database under tmp_path; one that reaches the real folders (a default path left unpinned did,
# 2026-11-09) is refused here whatever path it built, and the defaults of common/imessage point
# at a folder that does not exist.
_REAL_PEOPLE = (str(Path.home() / "Library" / "Messages"),
                str(Path.home() / "Library" / "Application Support" / "AddressBook"))


def _connect_without_fsync(*args, **kwargs):
    database = str(args[0] if args else kwargs.get("database", ""))
    if any(folder in database for folder in _REAL_PEOPLE):
        raise RuntimeError(f"a unit test tried to open the Mac's own Messages or AddressBook database: {database}")
    conn = _sqlite_connect(*args, **kwargs)
    try:
        conn.execute("PRAGMA synchronous=OFF")
    except sqlite3.Error:
        pass
    return conn


sqlite3.connect = _connect_without_fsync

import common.imessage.Contacts as _contacts_module
import common.imessage.MessagesDatabase as _messages_database_module
_NO_SUCH_MAC = aiserver_dir / "cache" / "no-such-mac"
_messages_database_module.CHAT_DB_PATH = str(_NO_SUCH_MAC / "Messages" / "chat.db")
_contacts_module.ADDRESS_BOOK_DIR = _NO_SUCH_MAC / "AddressBook"

# The typing dots are made through the Mac's Accessibility: a write into the compose field of the
# real Messages shows dots on a real person's phone, and the windows it opens are the developer's.
# Every pyobjc call of common/imessage — Accessibility and the login session's state — goes
# through one loader, refused here; the tests put an in-memory tree in its place (ax_fakes).
import common.imessage.Accessibility as _accessibility_module


def _no_real_accessibility():
    raise RuntimeError("a unit test tried to reach this Mac's Accessibility API or its login session")


_accessibility_module._frameworks = _no_real_accessibility


@pytest.fixture(autouse=True)
def integrations_enabled_unless_said_otherwise():
    """The developer's config.yaml decides `enabled_integrations`; the suite's fake integrations
    must not depend on it. Tests of the switch itself patch `Integration.enabled` closer in."""
    from assistant.integration.Integration import Integration
    stub = classmethod(lambda cls, config: True)
    stub.original = Integration.__dict__["enabled"]   # for the tests of the switch itself
    with patch.object(Integration, "enabled", stub):
        yield


@pytest.fixture(autouse=True)
def first_run_files_stay_out_of_the_tree(tmp_path):
    """The repo root's first_run.json and cache/setup_activity.json are the developer's install;
    a test that does not name its own paths reads and writes a scratch pair instead (a setup
    command notes its activity on every run)."""
    from assistant import FirstRun
    with patch.object(FirstRun, "STATE_FILE", tmp_path / "first_run.json"), \
            patch.object(FirstRun, "SETUP_ACTIVITY_FILE", tmp_path / "setup_activity.json"):
        yield


@pytest.fixture(autouse=True, scope="session")
def isolated_vault_file(tmp_path_factory):
    """No unit test opens the developer's vault.bin (which the cheap key could not read anyway):
    the Vault singleton gets its own, empty file for the session."""
    common.vault.Vault.VAULT_PATH = tmp_path_factory.mktemp("vault") / "vault.bin"


@pytest.fixture(autouse=True, scope="session")
def isolated_job_home_record(tmp_path_factory):
    """No unit test writes or removes the install's cache/job_home_branch — the record a running
    server keeps while one of its Claude Code jobs has the checkout on a work branch."""
    from assistant.agents import CodeQualityMaintainer
    CodeQualityMaintainer.JOB_HOME_FILE = tmp_path_factory.mktemp("job_home") / "job_home_branch"


@pytest.fixture(autouse=True, scope="session")
def isolated_imessage_chat_state(tmp_path_factory):
    """No unit test reads or writes the install's cache/imessage_chat.json — the iMessage chat's
    cursor into the assistant's own Messages account and its send counts: a transport built
    without a path of its own keeps its state in a scratch file."""
    from interfaces.imessage import ChatState
    ChatState.STATE_PATH = tmp_path_factory.mktemp("imessage_chat") / "imessage_chat.json"


@pytest.fixture(autouse=True, scope="session")
def isolated_imessage_typing_record(tmp_path_factory):
    """No unit test reads or writes the install's cache/imessage_typing_windows.json — the names
    of the dots windows a real host left open: a host built without an indicator of its own
    keeps that record in a scratch file."""
    from common.imessage import LocalMessagesHost
    LocalMessagesHost.TYPING_RECORD = tmp_path_factory.mktemp("imessage_typing") / "imessage_typing_windows.json"


@pytest.fixture(autouse=True, scope="session")
def isolated_launch_agent(tmp_path_factory):
    """No unit test writes or removes the watchdog's plist in ~/Library/LaunchAgents — launchd
    loads whatever lies there at the next login, whoever wrote it: a `Watchdog.sync` that names
    no plist of its own gets a scratch one, and the log the plist names is beside it."""
    from common import Watchdog
    root = tmp_path_factory.mktemp("launch_agent")
    Watchdog.PLIST = root / "LaunchAgents" / Watchdog.PLIST.name
    Watchdog.OUT_LOG = root / "Logs" / Watchdog.OUT_LOG.name


@pytest.fixture(autouse=True, scope="session")
def isolated_memory_stores(tmp_path_factory):
    """No unit test reads or writes the developer's memory/watchlist and memory/procedures (or
    their log/memory trail, or the curator's dream journals in log/memory — a journal written
    there after 04:00 would make the live server skip the night's curation): the stores'
    default directories are empty ones for the session. A test of the stores passes or
    patches its own tmp_path."""
    from assistant import Procedures, VoiceCandidates, WatchList
    from assistant.agents import LongTermMemoryCurator
    root = tmp_path_factory.mktemp("memory")
    WatchList.WATCHLIST_DIR, WatchList.LOG_DIR = root / "watchlist", root / "log" / "watchlist"
    Procedures.PROCEDURES_DIR, Procedures.LOG_DIR = root / "procedures", root / "log" / "procedures"
    VoiceCandidates.CANDIDATES_DIR, VoiceCandidates.LOG_DIR = root / "voice_candidates", root / "log" / "voice_id"
    LongTermMemoryCurator.MEMORY_LOG_DIR = root / "log"


@pytest.fixture(autouse=True, scope="session")
def isolated_database(tmp_path_factory):
    """No unit test opens the developer's live floxy.db: the conftest chdirs to AIServer/, so the
    cwd-relative default would be the real one, and anything reaching the singleton — a launcher
    test's `build()`, StartupTracker, a MessageHistory — would write into it. A test that wants
    the singleton gets a fresh, empty database per worker instead."""
    from common.FloxyDatabase import FloxyDatabase
    FloxyDatabase.DB_PATH = str(tmp_path_factory.mktemp("db") / "floxy.db")
    FloxyDatabase.reset_instance()
    yield
    FloxyDatabase.reset_instance()


@pytest.fixture(autouse=True)
def config_singleton_is_the_module_level_config():
    """`ConfigManager.reset_instance()` (the config tests' fixtures) leaves `get_instance()`
    building a second ConfigManager, while everything that did `from common.ConfigManager import
    config` keeps the first. A later test on that worker then patches `config.get` and code
    reading through `get_instance()` never sees it — it reads the developer's config.yaml
    instead (test_launcher's voice-less boot imported the voice stack that way, depending on
    which tests worksteal had put before it)."""
    import common.ConfigManager as config_module
    yield
    config_module.ConfigManager._instance = config_module.config


@pytest.fixture(autouse=True)
def reset_system_note():
    """Every test starts without a SystemNote history. `floxy_ai_server.build()` installs the
    prompt manager's MessageHistory process-wide and nothing detaches it, so without this a
    later test's `system_note()` (an iCloud 2FA test, say) lands in whatever history the last
    launcher test left behind."""
    from assistant import SystemNote
    SystemNote.set_history(None)
    yield
    SystemNote.set_history(None)


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file for testing"""
    config_content = """
api_keys:
  openai: test_openai_key
  anthropic: test_anthropic_key
  elevenlabs: test_elevenlabs_key
llm:
  models:
    chatgpt:
      model: gpt-4
voice_cache_size_in_gb: 2
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def file_logging_enabled(monkeypatch):
    """For a test of something the file-logging gate guards (prompt dumps, command replies,
    images, debug WAVs): opens the gate without starting the LogWriter thread, which would
    otherwise appear on the first log line and stay for the worker's lifetime."""
    monkeypatch.setattr(common.logging, "disable_file_logging", False)
    monkeypatch.setattr(common.logging, "_ensure_writer_started", lambda: None)


@pytest.fixture(autouse=True)
def stop_command_workers(monkeypatch):
    """Every CommandProcessor a test builds (directly or through a DialogManager) gets its worker
    thread stopped afterwards, so no test leaves a thread blocked on command_queue.get()."""
    from assistant.commands.CommandProcessor import CommandProcessor
    built = []
    real_init = CommandProcessor.__init__

    def tracking_init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        built.append(self)

    monkeypatch.setattr(CommandProcessor, "__init__", tracking_init)
    yield
    for processor in built:
        processor._stop_worker()
        processor.worker_thread.join(timeout=2)


@pytest.fixture(autouse=True)
def reset_assistant_observers():
    """Every test starts with an empty observer hub; subscribe a fake to assert on events."""
    from assistant.AssistantEvents import observers
    observers().clear()
    yield
    observers().clear()


@pytest.fixture(autouse=True)
def reset_notifier():
    """Every test starts with the NullNotifier; a test that wants to see notifications
    installs its own fake via set_notifier() or patch('assistant.Notifier._notifier')."""
    from assistant.Notifier import set_notifier
    set_notifier(None)
    yield
    set_notifier(None)


# The unit suite's wall-clock time is the last worker's finish plus the shutdown, so the two
# hooks below spread the slow tests evenly and make the workers exit fast.
SLOW_TESTS_KEY = "floxy/slow-tests"
SLOW_TEST_SECONDS = 0.05
_durations = {}


WONTFIX_MARKER = "wontfix"


def pytest_itemcollected(item):
    """A `wontfix` test is a non-strict xfail: the test documents a bug the maintainer has decided
    not to fix, so neither the bug staying nor its going away may fail the suite. The agents never
    add the marker (the nightly prompts say so); the test health audit removes it from a test that
    passes, so the record does not outlive the bug."""
    marker = item.get_closest_marker(WONTFIX_MARKER)
    if marker is not None:
        reason = marker.kwargs.get("reason") or (marker.args[0] if marker.args else WONTFIX_MARKER)
        item.add_marker(pytest.mark.xfail(strict=False, reason=f"{WONTFIX_MARKER}: {reason}"))


def pytest_collection_modifyitems(session, config, items):
    """Spread the previous run's slow tests through the collection, longest first, each placed
    where its share of the slow time falls. xdist hands every worker a contiguous slice of the
    collection (worksteal then steals contiguous tails), so in file order the 3 s crystal test
    sits at item ~5400 of 6400 and starts when the other workers are nearly done, holding the
    run open, and the slow tests clumped in one file (photos, webaccess) all land on one worker.
    The durations are what the controller cached in pytest_sessionfinish; a fresh checkout runs
    in file order once."""
    cache = getattr(config, "cache", None)
    slow = cache.get(SLOW_TESTS_KEY, {}) if cache else {}
    heavy = sorted((item for item in items if item.nodeid in slow), key=lambda item: -slow[item.nodeid])
    if not heavy:
        return
    rest = [item for item in items if item.nodeid not in slow]
    total = sum(slow[item.nodeid] for item in heavy)
    targets, placed = [], 0.0
    for item in heavy:
        targets.append((int(placed / total * len(rest)), item))
        placed += slow[item.nodeid]
    ordered, next_target = [], 0
    for index, item in enumerate(rest):
        while next_target < len(targets) and targets[next_target][0] <= index:
            ordered.append(targets[next_target][1])
            next_target += 1
        ordered.append(item)
    ordered.extend(item for _, item in targets[next_target:])
    items[:] = ordered


def pytest_collection_finish(session):
    """The collected heap goes into the permanent generation, so the collections the tests
    trigger walk only what the tests allocate; the higher threshold makes those rarer too.
    Together ~1 s of the wall time."""
    gc.freeze()
    gc.set_threshold(50000, 20, 100)
    gc.enable()


def pytest_runtest_logreport(report):
    _durations[report.nodeid] = _durations.get(report.nodeid, 0.0) + report.duration


def pytest_sessionfinish(session):
    """Nothing in the heap (every module, 6400 items, their fixtures) is used after the last
    test, yet pytest's unraisable-exception hook collects it five times at unconfigure and the
    interpreter once more at exit: ~0.5 s per worker, which the controller waits for. Freezing
    it into the permanent generation the collector skips makes both passes free. The price is
    that an unawaited coroutine left in the last test's garbage is no longer reported at exit.
    The controller (or the single process of a `-n 0` run) also records the slow tests for the
    next run's ordering."""
    gc.freeze()
    cache = getattr(session.config, "cache", None)
    if hasattr(session.config, "workerinput") or cache is None:
        return
    slow = cache.get(SLOW_TESTS_KEY, {})
    for nodeid, seconds in _durations.items():
        if seconds >= SLOW_TEST_SECONDS:
            slow[nodeid] = round(seconds, 3)
        else:
            slow.pop(nodeid, None)
    cache.set(SLOW_TESTS_KEY, slow)
