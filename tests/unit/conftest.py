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

# openai is imported here, at collection, rather than by the first ChatGPTLLM test on each
# worker: the import is ~80 ms, and the test that paid it was recorded as slow, moved to the
# front of the next run's collection and paid it again. anthropic is already imported at module
# level by the ClaudeLLM tests; google.genai is left to the few workers that need it.
import openai  # noqa: F401

# Disable all file logging during tests
import common.logging
common.logging.disable_file_logging = True

# Cheap vault keys: every test vault derived a real scrypt key (~90 ms), about a third of
# the suite's time. Sixteen rounds derive in microseconds and no test cares about strength.
import common.vault.Vault
common.vault.Vault.SCRYPT_N = 2 ** 4

# No fsync per commit: hundreds of tests write a throwaway SQLite file under tmp_path, and
# waiting for the disk on each commit was ~0.4 s of the wall time. Nothing here survives a crash.
import sqlite3
_sqlite_connect = sqlite3.connect


def _connect_without_fsync(*args, **kwargs):
    conn = _sqlite_connect(*args, **kwargs)
    try:
        conn.execute("PRAGMA synchronous=OFF")
    except sqlite3.Error:
        pass
    return conn


sqlite3.connect = _connect_without_fsync


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
def owner_is_the_cast_owner():
    """The developer's config.yaml decides `owner_name`; tests asserting on rendered messages
    ("Marcus to Alice: …") must not depend on it. Pinned under `get` on the shared instance only:
    a test patching `config.get` still wins, and the ConfigManager tests' own instances still
    read their files."""
    import common.ConfigManager as config_module
    real_get = config_module.config.get

    def get(key, *args, **kwargs):
        return "Marcus" if key == "owner_name" else real_get(key, *args, **kwargs)

    with patch.object(config_module.config, "get", get):
        yield


@pytest.fixture(autouse=True, scope="session")
def isolated_vault_file(tmp_path_factory):
    """No unit test opens the developer's vault.bin (which the cheap key could not read anyway):
    the Vault singleton gets its own, empty file for the session."""
    common.vault.Vault.VAULT_PATH = tmp_path_factory.mktemp("vault") / "vault.bin"


@pytest.fixture(autouse=True, scope="session")
def isolated_memory_stores(tmp_path_factory):
    """No unit test reads or writes the developer's memory/watchlist and memory/procedures (or
    their log/memory trail): the stores' default directories are empty ones for the session.
    A test of the stores passes or patches its own tmp_path."""
    from assistant import Procedures, WatchList
    root = tmp_path_factory.mktemp("memory")
    WatchList.WATCHLIST_DIR, WatchList.LOG_DIR = root / "watchlist", root / "log" / "watchlist"
    Procedures.PROCEDURES_DIR, Procedures.LOG_DIR = root / "procedures", root / "log" / "procedures"


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
