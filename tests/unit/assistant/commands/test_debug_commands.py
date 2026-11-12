import os
import sqlite3
import threading
import time
import zipfile
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

import assistant.commands.DebugCommands as dc_mod
from assistant.commands.DebugCommands import DebugCommands


def _write_wav(path, mtime=None):
    with open(path, "wb") as f:
        f.write(b"RIFF....")
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def test_is_command():
    assert DebugCommands.is_command("//FP")
    assert DebugCommands.is_command("  //TP  ")
    assert not DebugCommands.is_command("/FP")  # single slash is a Telegram bot command
    assert not DebugCommands.is_command("hello")
    assert not DebugCommands.is_command("")


def test_recent_audio_returns_newest_first(tmp_path, monkeypatch):
    audio = tmp_path / "audio"
    audio.mkdir()
    _write_wav(audio / "old.wav", mtime=time.time() - 100)
    _write_wav(audio / "new.wav", mtime=time.time())
    # subdir + non-wav must be ignored
    (audio / "identification").mkdir()
    _write_wav(audio / "identification" / "ignored.wav", mtime=time.time() + 100)
    (audio / "notes.txt").write_text("x")

    monkeypatch.setattr(dc_mod, "AUDIO_DIR", str(audio))
    names = [os.path.basename(p) for p in DebugCommands()._recent_audio()]
    assert names == ["new.wav", "old.wav"]


def test_recent_audio_caps_at_count(tmp_path, monkeypatch):
    audio = tmp_path / "audio"
    audio.mkdir()
    for i in range(15):
        _write_wav(audio / f"clip{i:02d}.wav", mtime=time.time() + i)

    monkeypatch.setattr(dc_mod, "AUDIO_DIR", str(audio))
    monkeypatch.setattr(dc_mod, "RECENT_AUDIO_COUNT", 10)
    recent = DebugCommands()._recent_audio()
    assert len(recent) == 10
    # newest first -> clip14 down to clip05
    assert os.path.basename(recent[0]) == "clip14.wav"
    assert os.path.basename(recent[-1]) == "clip05.wav"


def test_recent_audio_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(dc_mod, "AUDIO_DIR", str(tmp_path / "missing"))
    assert DebugCommands()._recent_audio() == []


def test_cmd_fp_and_tp_copy_to_dest(tmp_path, monkeypatch):
    audio = tmp_path / "audio"
    audio.mkdir()
    for i in range(3):
        _write_wav(audio / f"phrase{i}.wav", mtime=time.time() + i)
    fp = tmp_path / "false-positive"
    tp = tmp_path / "en"

    monkeypatch.setattr(dc_mod, "AUDIO_DIR", str(audio))
    monkeypatch.setattr(dc_mod, "FALSE_POSITIVE_DIR", str(fp))
    monkeypatch.setattr(dc_mod, "TRUE_POSITIVE_DIR", str(tp))

    assert "Copied" in DebugCommands().handle("//FP")
    assert sorted(p.name for p in fp.glob("*.wav")) == ["phrase0.wav", "phrase1.wav", "phrase2.wav"]

    assert "Copied" in DebugCommands().handle("//tp")  # case-insensitive
    assert sorted(p.name for p in tp.glob("*.wav")) == ["phrase0.wav", "phrase1.wav", "phrase2.wav"]


def test_cmd_fp_copies_to_the_configured_dir(tmp_path, monkeypatch):
    audio = tmp_path / "audio"
    audio.mkdir()
    _write_wav(audio / "phrase.wav", mtime=time.time())
    configured = tmp_path / "corpus" / "false-positive"
    monkeypatch.setattr(dc_mod, "AUDIO_DIR", str(audio))
    monkeypatch.setattr(dc_mod, "FALSE_POSITIVE_DIR", str(tmp_path / "default"))

    from common.ConfigManager import config
    with patch.object(config, "get", lambda key, *a, **k: str(configured) if key == "debug.false_positive_dir" else ""):
        assert "Copied" in DebugCommands().handle("//FP")

    assert [p.name for p in configured.glob("*.wav")] == ["phrase.wav"]
    assert not (tmp_path / "default").exists()


def test_cmd_fp_no_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(dc_mod, "AUDIO_DIR", str(tmp_path / "missing"))
    assert "No audio" in DebugCommands().handle("//FP")


# --- //help and dispatch ------------------------------------------------------------------

def _cmd_zztest(self, args: str) -> str:
    """Echo the arguments (test only)."""
    return f"args=<{args}>"


def test_help_lists_every_command_with_its_docstring(monkeypatch):
    monkeypatch.setattr(DebugCommands, "cmd_ZZTEST", _cmd_zztest, raising=False)
    text = DebugCommands().handle("//help")
    assert text.startswith("Debug commands:")
    for name in ("HELP", "FP", "TP", "WIPE", "ZZTEST"):
        assert f"//{name}" in text
    assert "Echo the arguments (test only)." in text
    assert "//wipe confirm" in text


def test_arguments_reach_the_command(monkeypatch):
    monkeypatch.setattr(DebugCommands, "cmd_ZZTEST", _cmd_zztest, raising=False)
    assert DebugCommands().handle("//zztest one  two ") == "args=<one  two>"
    assert DebugCommands().handle("//ZZTEST") == "args=<>"


def test_unknown_command_shows_help():
    text = DebugCommands().handle("//NOPE")
    assert "Unknown command: //NOPE" in text and "//HELP" in text


# --- //wipe -------------------------------------------------------------------------------

@pytest.fixture
def wipe_env(tmp_path, monkeypatch):
    """A fake AIServer tree with every kind of user state, and the wipe wired to it."""
    root = tmp_path / "AIServer"
    cache = root / "cache"
    cache.mkdir(parents=True)
    db = root / "floxy.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE t (v TEXT)")
        conn.execute("INSERT INTO t VALUES ('kept')")
    (root / "floxy.db-shm").write_bytes(b"stale sidecar")
    (root / "floxy.db.old").write_bytes(b"old backup, not a target")
    (root / "config.yaml").write_text("api_keys:\n  openai: sk-real\n")
    (root / "vault.bin").write_bytes(b"FLOXYVAULT1\n")
    (root / "memory.md").write_text("# The Family\n")
    (cache / "icloud_session").mkdir()
    (cache / "icloud_session" / "cookies").write_text("c")
    (cache / "chrome_profile" / "Default").mkdir(parents=True)
    (cache / "chrome_profile" / "Default" / "Cookies").write_bytes(b"x" * 100)
    external = tmp_path / "tmp" / "ring_floxy"
    external.mkdir(parents=True)
    (external / "token.cache").write_text("ring")

    stops = []
    monkeypatch.setattr(dc_mod, "DATABASE_PATH", db)
    monkeypatch.setattr(dc_mod, "DATABASE_SIDECARS", [root / "floxy.db-wal", root / "floxy.db-shm"])
    monkeypatch.setattr(dc_mod, "CONFIG_PATH", root / "config.yaml")
    monkeypatch.setattr(dc_mod, "DUMMY_CONFIG_PATH", root / "config.dummy.yaml")
    monkeypatch.setattr(dc_mod, "BACKUP_DIR", cache / "wipe-backups")
    monkeypatch.setattr(dc_mod, "WIPE_TARGETS", [
        (db, "database"),
        (root / "config.yaml", "config"),
        (root / "vault.bin", "vault"),
        (root / "memory.md", "memory"),
        (root / "missing.txt", "never existed"),
        (cache / "icloud_session", "iCloud session"),
    ])
    monkeypatch.setattr(dc_mod, "WIPE_ONLY_TARGETS", [(cache / "chrome_profile", "browser profile")])
    monkeypatch.setattr(dc_mod, "external_targets", lambda: [(external / "token.cache", "Ring token")])
    monkeypatch.setattr(dc_mod, "integration_targets", lambda: [])
    monkeypatch.setattr(dc_mod, "_stop_server", lambda: stops.append(True))
    monkeypatch.setattr(dc_mod, "STOP_DELAY_SECONDS", 0.0)
    return root, stops


def _wait_for_stop(stops):
    """The wipe's tail (browser close, profile delete, stop) runs on a timer thread."""
    deadline = time.time() + 3
    while not stops and time.time() < deadline:
        time.sleep(0.01)
    assert stops == [True]


def test_restart_replies_first_then_pulls_and_restarts(monkeypatch):
    """The reply must be out before start.sh kills the process, so the pull and restart
    run on a timer; the outcome goes to the dialog as a system note, like restart_server."""
    from assistant.agents.CodeQualityMaintainer import CodeQualityMaintainer
    import assistant.SystemNote as sn_mod
    calls, notes = [], []
    monkeypatch.setattr(CodeQualityMaintainer, "target_branch", staticmethod(lambda: "release"))
    monkeypatch.setattr(CodeQualityMaintainer, "own_branch", staticmethod(lambda: None))   # whatever branch the suite runs on
    monkeypatch.setattr(CodeQualityMaintainer, "pull_and_restart",
                        staticmethod(lambda: calls.append(True) or "Update initiated (no new changes). Server is restarting..."))
    monkeypatch.setattr(sn_mod, "system_note", lambda source, text: notes.append((source, text)))
    monkeypatch.setattr(dc_mod, "STOP_DELAY_SECONDS", 0.0)

    text = DebugCommands().handle("//restart")

    assert "Pulling release and restarting the server" in text
    deadline = time.time() + 3
    while not notes and time.time() < deadline:
        time.sleep(0.01)
    assert calls == [True]
    assert notes == [("server", "Server restart (//restart): Update initiated (no new changes). Server is restarting...")]


def test_restart_on_a_branch_of_the_installs_own_does_not_claim_a_pull(monkeypatch):
    """The answer is all the user sees before the process dies: on an own branch nothing is pulled,
    and saying "pulling release" would have them expect the fix they are waiting for."""
    from assistant.agents.CodeQualityMaintainer import CodeQualityMaintainer
    monkeypatch.setattr(CodeQualityMaintainer, "target_branch", staticmethod(lambda: "release"))
    monkeypatch.setattr(CodeQualityMaintainer, "own_branch", staticmethod(lambda: "feat/garden-lights"))
    monkeypatch.setattr(dc_mod, "_pull_and_restart", lambda: None)
    monkeypatch.setattr(dc_mod, "STOP_DELAY_SECONDS", 0.0)

    text = DebugCommands().handle("//restart")

    assert "feat/garden-lights" in text and "without a pull" in text
    assert "release" not in text


def test_wipe_without_confirm_only_shows_the_plan(wipe_env):
    root, stops = wipe_env
    text = DebugCommands().handle("//wipe")
    assert "would back up 6 item(s)" in text
    assert "floxy.db: database" in text and "Ring token" in text and "chrome_profile" in text
    assert "missing.txt" not in text
    assert "//wipe confirm" in text
    assert (root / "floxy.db").exists() and (root / "vault.bin").exists()
    assert not (root / "cache" / "wipe-backups").exists()
    assert stops == []


def test_wipe_confirm_backs_up_deletes_resets_config_and_stops(wipe_env):
    root, stops = wipe_env
    text = DebugCommands().handle("//wipe confirm")

    zips = list((root / "cache" / "wipe-backups").glob("floxy-wipe-*.zip"))
    assert len(zips) == 1
    with zipfile.ZipFile(zips[0]) as zf:
        names = set(zf.namelist())
        assert {"external/" + str(root / "config.yaml").lstrip("/"),
                "external/" + str(root / "vault.bin").lstrip("/"),
                "external/" + str(root / "memory.md").lstrip("/"),
                "external/" + str(root / "cache" / "icloud_session" / "cookies").lstrip("/")} <= names
        assert not any("chrome_profile" in n for n in names)
        assert any(n.endswith("ring_floxy/token.cache") for n in names)
        db_copy = root / "restored.db"
        db_copy.write_bytes(zf.read("external/" + str(root / "floxy.db").lstrip("/")))
    with sqlite3.connect(db_copy) as conn:
        assert conn.execute("SELECT v FROM t").fetchall() == [("kept",)]

    assert "Wiped 7 item(s)" in text and "writes a skeleton" in text and "Stopping the server" in text
    assert "Deleted once the browser is closed:" in text and "chrome_profile: browser profile" in text
    _wait_for_stop(stops)
    for gone in ("floxy.db", "floxy.db-shm", "vault.bin", "memory.md", "cache/icloud_session", "cache/chrome_profile"):
        assert not (root / gone).exists(), gone
    assert not (root.parent / "tmp" / "ring_floxy" / "token.cache").exists()
    assert (root / "floxy.db.old").exists()
    assert not (root / "config.yaml").exists()
    assert not list((root / "cache" / "wipe-backups").glob("*.snapshot"))


def test_second_wipe_confirm_while_one_runs_is_refused(wipe_env, monkeypatch):
    """Two concurrent wipes once wrote into one shared database snapshot, zipping a torn
    copy; the second must wait for the first's reply instead."""
    root, stops = wipe_env
    backing_up, release = threading.Event(), threading.Event()
    original = DebugCommands._backup

    def slow_backup(self, targets):
        backing_up.set()
        assert release.wait(3)
        return original(self, targets)

    monkeypatch.setattr(DebugCommands, "_backup", slow_backup)
    first = []
    t = threading.Thread(target=lambda: first.append(DebugCommands().handle("//wipe confirm")))
    t.start()
    assert backing_up.wait(3)
    second = DebugCommands().handle("//wipe confirm")
    assert "already running" in second
    assert (root / "floxy.db").exists()
    release.set()
    t.join(5)
    assert "Wiped 7 item(s)" in first[0]
    _wait_for_stop(stops)
    zips = list((root / "cache" / "wipe-backups").glob("floxy-wipe-*.zip"))
    assert len(zips) == 1
    assert not list((root / "cache" / "wipe-backups").glob("*.snapshot"))


def test_wipe_installs_the_dummy_config_when_there_is_one(wipe_env):
    root, stops = wipe_env
    (root / "config.dummy.yaml").write_text("api_keys:\n  anthropic: sk-dev\n")
    plan = DebugCommands().handle("//wipe")
    assert "config.dummy.yaml is copied to config.yaml" in plan

    text = DebugCommands().handle("//wipe confirm")
    assert "config.dummy.yaml is copied to config.yaml" in text and "writes a skeleton" not in text
    _wait_for_stop(stops)
    assert (root / "config.yaml").read_text() == "api_keys:\n  anthropic: sk-dev\n"
    assert (root / "config.dummy.yaml").exists()
    with zipfile.ZipFile(next((root / "cache" / "wipe-backups").glob("floxy-wipe-*.zip"))) as zf:
        assert zf.read("external/" + str(root / "config.yaml").lstrip("/")) == b"api_keys:\n  openai: sk-real\n"


def test_wipe_carries_on_past_an_item_it_cannot_delete(wipe_env, monkeypatch):
    root, stops = wipe_env
    real_rmtree = dc_mod.shutil.rmtree

    def _stubborn(path, *args, **kwargs):
        if Path(path).name == "icloud_session":
            raise OSError(66, "Directory not empty")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(dc_mod.shutil, "rmtree", _stubborn)
    text = DebugCommands().handle("//wipe confirm")
    assert "Wiped 6 item(s)" in text
    assert "Could not delete (see the log):" in text and "icloud_session" in text and "Directory not empty" in text
    assert (root / "cache" / "icloud_session").exists()
    assert not (root / "floxy.db").exists() and not (root / "config.yaml").exists()
    _wait_for_stop(stops)


def test_wipe_backup_failure_deletes_nothing(wipe_env, monkeypatch):
    root, stops = wipe_env

    def _boom(self, targets):
        raise OSError("disk full")

    monkeypatch.setattr(DebugCommands, "_backup", _boom)
    text = DebugCommands().handle("//wipe confirm")
    assert "Backup failed, nothing was deleted: disk full" in text
    assert (root / "floxy.db").exists() and (root / "vault.bin").exists() and (root / "cache" / "chrome_profile").exists()
    assert (root / "config.yaml").read_text() == "api_keys:\n  openai: sk-real\n"
    assert stops == []


def test_wipe_on_a_fresh_install(wipe_env):
    root, stops = wipe_env
    DebugCommands().handle("//wipe confirm")
    _wait_for_stop(stops)
    stops.clear()
    text = DebugCommands().handle("//wipe confirm")
    assert text == "Nothing to wipe: the install is already fresh."
    assert DebugCommands().handle("//wipe") == "Nothing to wipe: the install is already fresh."


# --- //feed -------------------------------------------------------------------------------

class _FakeRunner:
    def __init__(self, name, state="catching_up"):
        self.name = name
        self.state = state
        self.paused = False

    def pause(self):
        if self.paused:
            return False
        self.paused = True
        return True

    def resume(self):
        if not self.paused:
            return False
        self.paused = False
        return True

    def status(self):
        return {"state": self.state, "paused": self.paused, "progress": 0.5 if self.state == "catching_up" else None,
                "remaining": 10 if self.state == "catching_up" else 0, "last_error": None}


class _FakeFeeds:
    def __init__(self, runners):
        self._runners = {r.name: r for r in runners}

    def names(self):
        return list(self._runners)

    def runner(self, name):
        return self._runners.get(name)

    def status(self, name):
        return self._runners[name].status()

    def find(self, name):
        return next((r for k, r in self._runners.items() if k.lower() == name.strip().lower()), None)


@pytest.fixture
def feeds(monkeypatch):
    from assistant.integration.FeedRunner import Feeds
    fake = _FakeFeeds([_FakeRunner("iMessage"), _FakeRunner("Telegram", state="live"), _FakeRunner("iCloud Photos")])
    monkeypatch.setattr(Feeds, "get_instance", classmethod(lambda cls: fake))
    return fake


def test_feed_lists_the_feeds(feeds):
    assert DebugCommands().handle("//feed") == \
        "Feeds:\n  iMessage: catching up 50%, 10 left\n  Telegram: live\n  iCloud Photos: catching up 50%, 10 left"


def test_feed_pause_a_feed_whose_name_has_a_space(feeds):
    assert DebugCommands().handle("//feed pause iCloud Photos") == "Paused iCloud Photos (paused, catching up 50%, 10 left)"
    assert feeds.runner("iCloud Photos").paused and not feeds.runner("iMessage").paused


def test_feed_pause_and_resume_one_feed_case_insensitively(feeds):
    assert DebugCommands().handle("//feed pause imessage") == "Paused iMessage (paused, catching up 50%, 10 left)"
    assert feeds.runner("iMessage").paused and not feeds.runner("Telegram").paused
    assert DebugCommands().handle("//feed pause iMessage") == "Already paused iMessage (paused, catching up 50%, 10 left)"
    assert DebugCommands().handle("//feed resume IMESSAGE") == "Resumed iMessage (catching up 50%, 10 left)"
    assert DebugCommands().handle("//feed resume imessage") == "Not paused iMessage (catching up 50%, 10 left)"


def test_feed_pause_all(feeds):
    text = DebugCommands().handle("//feed pause all")
    assert text == "Paused iMessage (paused, catching up 50%, 10 left)\nPaused Telegram (paused, live)\n" \
                   "Paused iCloud Photos (paused, catching up 50%, 10 left)"
    assert all(r.paused for r in feeds._runners.values())
    assert DebugCommands().handle("//feed resume ALL").startswith("Resumed iMessage")
    assert not any(r.paused for r in feeds._runners.values())


def test_feed_bad_input(feeds):
    assert DebugCommands().handle("//feed pause nope") == "No feed named 'nope'. Feeds: iMessage, Telegram, iCloud Photos"
    assert DebugCommands().handle("//feed stop all") == "Usage: //feed <pause|resume> <all|iMessage|Telegram|iCloud Photos>"
    assert DebugCommands().handle("//feed pause") == "Usage: //feed <pause|resume> <all|iMessage|Telegram|iCloud Photos>"


# --- //memory_curate --------------------------------------------------------------------

@pytest.fixture
def curation(monkeypatch):
    """Capture the span handed to the curator instead of running an LLM pass; the thread is
    joined so the assertion sees the call."""
    calls = []
    threads = []

    histories = []

    class FakeCurator:
        @staticmethod
        def run_span(start, end, history=None):
            calls.append((start, end))
            histories.append(history)
            return {}

    import assistant.agents.LongTermMemoryCurator as curator_mod
    monkeypatch.setattr(curator_mod, "LongTermMemoryCurator", FakeCurator)
    real_thread = dc_mod.threading.Thread

    class JoinedThread(real_thread):
        def start(self):
            super().start()
            threads.append(self)

    monkeypatch.setattr(dc_mod.threading, "Thread", JoinedThread)

    def run(query, history=None):
        reply = DebugCommands(history=history).handle(query)
        for t in threads:
            t.join(timeout=5)
        return reply
    run.calls = calls
    run.histories = histories
    return run


def test_memory_curate_span_end_is_inclusive(curation):
    # the span reaches the curator as inclusive dates; it does the chunking (8 of 4 days here)
    from datetime import date
    reply = curation("//memory_curate 20260801 20260830")
    assert curation.calls == [(date(2026, 8, 1), date(2026, 8, 30))]
    assert "2026-08-01 .. 2026-08-30 (30 days, 8 chunks of up to 4 days)" in reply


def test_memory_curate_single_day(curation):
    from datetime import date
    reply = curation("//memory_curate 20260901")
    assert curation.calls == [(date(2026, 9, 1), date(2026, 9, 1))]
    assert "2026-09-01" in reply and "chunks" not in reply


def test_memory_curate_clamps_the_end_to_today(curation):
    from datetime import date
    curation("//memory_curate 20260901 29991231")
    assert curation.calls == [(date(2026, 9, 1), date.today())]


def test_memory_curate_hands_the_history_to_the_curator(curation):
    # The completion row lands in the assistant's dialog history; without one, the curator
    # is told so (None) rather than handed something else.
    history = object()
    curation("//memory_curate 20260901", history=history)
    curation("//memory_curate 20260901")
    assert curation.histories == [history, None]


# --- //memory_condense ------------------------------------------------------------------

@pytest.fixture
def condense(monkeypatch):
    """A bank of a chosen size behind //memory_condense, the cut handed to the curator captured
    instead of an LLM pass; the thread is joined so the assertion sees the call."""
    import assistant.agents.LongTermMemoryCurator as curator_mod
    from assistant import MemoryFile
    calls, threads = [], []

    def run_condense(cut, history=None):
        calls.append((cut, history))
        return {}

    monkeypatch.setattr(curator_mod.LongTermMemoryCurator, "run_condense", staticmethod(run_condense))
    real_thread = dc_mod.threading.Thread

    class JoinedThread(real_thread):
        def start(self):
            super().start()
            threads.append(self)

    monkeypatch.setattr(dc_mod.threading, "Thread", JoinedThread)

    def run(query, words: int, history=None):
        monkeypatch.setattr(MemoryFile, "read", lambda path=None: "# Memory\n" + " ".join(["w"] * (words - 2)))
        reply = DebugCommands(history=history).handle(query)
        for t in threads:
            t.join(timeout=5)
        return reply
    run.calls = calls
    return run


def test_memory_condense_over_the_limit_asks_the_nightly_cut(condense):
    from assistant.agents.LongTermMemoryCurator import LongTermMemoryCurator, MEMORY_WORD_LIMIT
    history = object()
    reply = condense("//memory_condense", MEMORY_WORD_LIMIT + 300, history=history)
    assert condense.calls == [(LongTermMemoryCurator.condense_cut(MEMORY_WORD_LIMIT + 300), history)]
    assert str(MEMORY_WORD_LIMIT + 300) in reply


def test_memory_condense_within_the_limit_does_nothing_without_a_number(condense):
    from assistant.agents.LongTermMemoryCurator import MEMORY_WORD_LIMIT
    reply = condense("//memory_condense", MEMORY_WORD_LIMIT - 10)
    assert condense.calls == [] and "nothing to condense" in reply


def test_memory_condense_takes_the_words_to_cut(condense):
    from assistant.agents.LongTermMemoryCurator import MEMORY_WORD_LIMIT
    condense("//memory_condense 250", MEMORY_WORD_LIMIT - 10)
    assert condense.calls == [(250, None)]


def test_memory_condense_rejects_a_bad_count(condense):
    from assistant.agents.LongTermMemoryCurator import MEMORY_WORD_LIMIT
    for args in ("abc", "0", "1 2"):
        assert "Usage" in condense(f"//memory_condense {args}", MEMORY_WORD_LIMIT + 300)
    assert condense.calls == []


def test_for_assistant_takes_the_history_from_the_prompt_manager():
    from unittest.mock import patch, Mock
    assistant = Mock()
    assert DebugCommands.for_assistant(assistant).history is assistant.prompt_manager.message_history
    # a stand-in assistant (a hub with nothing attached yet, tests) has no prompt manager
    assert DebugCommands.for_assistant(object()).history is None
    assistant.prompt_manager = None
    assert DebugCommands.for_assistant(assistant).history is None


def test_memory_curate_rejects_bad_input(curation):
    assert curation("//memory_curate").startswith("Usage")
    assert curation("//memory_curate 2026-10-01").startswith("Usage")
    assert curation("//memory_curate 20261030 20261001").startswith("The end")
    assert "future" in curation("//memory_curate 29991231")
    assert curation.calls == []


# --- //first_run -------------------------------------------------------------------------

@pytest.fixture
def first_run_state(tmp_path, monkeypatch):
    """The pipeline state at a tmp path. The command reaches the file two ways: `FirstRun.STATE_FILE`
    (exists/unlink) and `State.load()` / `save()` whose default path was bound when FirstRun was
    imported — so both are pointed at the tmp file here."""
    from assistant import FirstRun
    path = tmp_path / "first_run.json"
    monkeypatch.setattr(FirstRun, "STATE_FILE", path)
    load, save = FirstRun.State.load.__func__, FirstRun.State.save
    monkeypatch.setattr(FirstRun.State, "load", classmethod(lambda cls, p=path: load(cls, p)))
    monkeypatch.setattr(FirstRun.State, "save", lambda self, p=path: save(self, p))
    return path


def test_first_run_without_state(first_run_state):
    assert DebugCommands().handle("//first_run") == "No first-run pipeline on this install (established: no first_run.json)."
    assert DebugCommands().handle("//first_run reset") == "No first-run state to reset."


def test_first_run_status(first_run_state, feeds):
    from assistant import FirstRun
    state = FirstRun.State(stage=FirstRun.Stage.CURATING, started_at=time.mktime((2026, 9, 14, 9, 0, 0, 0, 0, -1)),
                           entered_at=time.mktime((2026, 9, 14, 11, 30, 0, 0, 0, -1)),
                           curation_start="2026-08-13", curation_end="2026-09-14", chunks_done=3, chunks_total=9)
    state.save()
    assert DebugCommands().handle("//first_run") == (
        "First-run pipeline: curating since 2026-09-14 11:30 (started 2026-09-14 09:00)\n"
        "First curation: 2026-08-13 .. 2026-09-14, chunk 3/9\n"
        "iMessage: catching up 50%, 10 left\n"
        "iCloud Photos: catching up 50%, 10 left")


def test_first_run_status_without_span_or_feeds(first_run_state, monkeypatch):
    from assistant import FirstRun
    from assistant.integration.FeedRunner import Feeds
    monkeypatch.setattr(Feeds, "get_instance", classmethod(lambda cls: _FakeFeeds([])))
    FirstRun.State.fresh().save()
    lines = DebugCommands().handle("//first_run").split("\n")
    assert lines[0].startswith("First-run pipeline: setup since ")
    assert lines[1:] == ["iMessage: not started", "iCloud Photos: not started"]


def test_first_run_stage_is_forced(first_run_state):
    from assistant import FirstRun
    state = FirstRun.State(stage=FirstRun.Stage.CURATING, started_at=1.0, entered_at=2.0,
                           curation_start="2026-08-13", curation_end="2026-09-14", chunks_done=3, chunks_total=9)
    state.save()
    assert DebugCommands().handle("//first_run stage expanding") == \
        "First-run stage set to expanding (the pipeline picks it up at its next poll)."
    saved = FirstRun.State.load()
    assert saved.stage == FirstRun.Stage.EXPANDING and saved.chunks_done == 0 and saved.entered_at > 2.0
    assert saved.started_at == 1.0 and saved.span == (date(2026, 8, 13), date(2026, 9, 14))


def test_first_run_stage_creates_the_state_when_there_is_none(first_run_state):
    from assistant import FirstRun
    DebugCommands().handle("//first_run stage indexing")
    assert FirstRun.State.load().stage == FirstRun.Stage.INDEXING


def test_first_run_bad_input(first_run_state):
    usage = "Usage: //first_run stage <setup, indexing, curating, expanding, done>"
    assert DebugCommands().handle("//first_run stage bogus") == usage
    assert DebugCommands().handle("//first_run stage") == usage
    assert DebugCommands().handle("//first_run stage curating done") == usage
    assert DebugCommands().handle("//first_run nope") == "Usage: //first_run [stage <name> | reset]"
    assert not first_run_state.exists()


def test_first_run_reset_deletes_the_state(first_run_state):
    from assistant import FirstRun
    FirstRun.State.fresh().save()
    assert DebugCommands().handle("//first_run reset") == \
        "First-run state deleted: the pipeline stops and the install counts as established."
    assert not first_run_state.exists()


def test_wipe_logs_the_telegram_session_out_before_deleting_it(wipe_env, monkeypatch):
    """Telethon's log-out deletes the file itself; the listing still names it. A failed
    log-out is noted and the file deleted anyway. The session is the Telegram folder's own
    wipe target, gathered by discovery."""
    root, stops = wipe_env
    session = root / "telegram_feed.session"
    session.write_bytes(b"sqlite")
    from integrations.telegram_feed.TelegramFeedIntegration import TelegramFeedIntegration
    monkeypatch.setattr(TelegramFeedIntegration, "wipe_targets", classmethod(lambda cls: [(session, "Telegram feed login")]))
    monkeypatch.setattr("assistant.integration.bootstrap.integration_classes", lambda: [TelegramFeedIntegration])
    monkeypatch.setattr(dc_mod, "integration_targets", lambda: [(session, "Telegram feed login")])

    def log_out():
        session.unlink()
        return "logged out on Telegram"

    with patch.object(TelegramFeedIntegration, "log_out", side_effect=log_out) as logout:
        text = DebugCommands().handle("//wipe confirm")
    logout.assert_called_once_with()
    assert "telegram_feed.session: Telegram feed login — logged out on Telegram" in text
    assert "Wiped 8 item(s)" in text
    assert not session.exists()
    _wait_for_stop(stops)

    session.write_bytes(b"sqlite")
    stops.clear()
    with patch.object(TelegramFeedIntegration, "log_out", return_value="not logged out on Telegram (OSError: no network)"):
        text = DebugCommands().handle("//wipe confirm")
    assert "telegram_feed.session: Telegram feed login — not logged out on Telegram (OSError: no network)" in text
    assert not session.exists()
    _wait_for_stop(stops)


def test_wipe_without_a_telegram_session_never_touches_the_integration(wipe_env):
    from integrations.telegram_feed.TelegramFeedIntegration import TelegramFeedIntegration
    with patch.object(TelegramFeedIntegration, "log_out") as logout:
        DebugCommands().handle("//wipe")
    logout.assert_not_called()


# --- //toolkits and //toolkit --------------------------------------------------------------

def test_toolkits_lists_every_toolkit_with_its_window_state():
    from unittest.mock import Mock
    from assistant.commands.Toolkits import TOOLKITS
    history = Mock()
    history.loaded_toolkits.return_value = {"email": time.mktime((2026, 9, 14, 9, 5, 0, 0, 0, -1))}
    text = DebugCommands(history).handle("//toolkits")
    lines = text.split("\n")
    assert lines[0] == f"Toolkits ({len(TOOLKITS)}):"
    assert len(lines) == len(TOOLKITS) + 1
    email = next(line for line in lines if line.split()[0] == "email")
    assert "loaded 09:05" in email and "latest_email" in email and "MISSING FILE" not in text
    web = next(line for line in lines if line.split()[0] == "web")
    assert "not loaded" in web and "open_url, google" in web


def test_toolkits_without_a_history_shows_nothing_loaded():
    text = DebugCommands().handle("//toolkits")
    assert "loaded " not in text.replace("not loaded", "")


def test_toolkit_renders_one_reference_through_the_prompt_manager(monkeypatch):
    from unittest.mock import Mock
    from assistant.agents.AutonomousActionManager import AutonomousActionManager
    monkeypatch.setattr(AutonomousActionManager, "_instance", None)
    raw = DebugCommands().handle("//toolkit photos")
    assert raw.startswith("# Toolkit: photos\n") and "{PHOTO_COUNT}" in raw, "no manager: the placeholders stay"

    aam = Mock()
    aam.prompt_manager.render_template = lambda text: text.replace("{PHOTO_COUNT}", "~1234")
    monkeypatch.setattr(AutonomousActionManager, "_instance", aam)
    assert "~1234 indexed photos" in DebugCommands().handle("//toolkit photos")
    assert DebugCommands().handle('//toolkit "Email"').startswith("# Toolkit: email\n")


def test_toolkit_unknown_or_missing_name_shows_usage():
    for query in ("//toolkit", "//toolkit nope"):
        text = DebugCommands().handle(query)
        assert text.startswith("Usage: //toolkit <name>") and "email" in text


# --- //memory_split ---------------------------------------------------------------------

def test_memory_split_rejects_an_unknown_argument(monkeypatch, tmp_path):
    monkeypatch.setattr(dc_mod, "MEMORY_LOG_DIR", tmp_path)
    started = []
    monkeypatch.setattr(dc_mod.threading, "Thread", lambda **kw: started.append(kw))
    reply = DebugCommands().handle("//memory_split now")
    assert "//memory_split apply" in reply
    assert started == [] and list(tmp_path.iterdir()) == []


def test_memory_split_apply_without_a_proposal_says_so(monkeypatch, tmp_path):
    import assistant.agents.LongTermMemoryCurator as curator_mod
    monkeypatch.setattr(dc_mod, "MEMORY_LOG_DIR", tmp_path)
    (tmp_path / "split_20260301_120000").mkdir()   # an aborted proposal: no bank.md in it
    created = []
    monkeypatch.setattr(curator_mod, "LongTermMemoryCurator", lambda *a, **kw: created.append(1))
    reply = DebugCommands().handle("//memory_split apply")
    assert "no split proposal" in reply.lower()
    assert created == []


def test_memory_split_apply_takes_the_latest_proposal(monkeypatch, tmp_path):
    import assistant.agents.LongTermMemoryCurator as curator_mod
    monkeypatch.setattr(dc_mod, "MEMORY_LOG_DIR", tmp_path)
    for name in ("split_20260301_120000", "split_20260302_090000"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "bank.md").write_text("# Bank\n")
    applied = []

    class FakeCurator:
        def apply_split(self, source):
            applied.append(source)
            return {'watch_items': ["parcel", "dentist", "plumber"], 'procedures': ["shop-checkout"],
                    'kept': ["school-portal-login"], 'backup': "somewhere"}

    monkeypatch.setattr(curator_mod, "LongTermMemoryCurator", FakeCurator)
    reply = DebugCommands().handle("//memory_split apply")
    assert applied == [tmp_path / "split_20260302_090000"]
    assert "split_20260302_090000" in reply and "3" in reply and "school-portal-login" in reply


def test_memory_split_writes_its_proposal_in_the_background(monkeypatch, tmp_path):
    import assistant.agents.LongTermMemoryCurator as curator_mod
    monkeypatch.setattr(dc_mod, "MEMORY_LOG_DIR", tmp_path)
    targets, threads = [], []

    class FakeCurator:
        def propose_split(self, target):
            targets.append(target)

    monkeypatch.setattr(curator_mod, "LongTermMemoryCurator", FakeCurator)
    real_thread = dc_mod.threading.Thread

    class JoinedThread(real_thread):
        def start(self):
            super().start()
            threads.append(self)

    monkeypatch.setattr(dc_mod.threading, "Thread", JoinedThread)
    reply = DebugCommands().handle("//memory_split")
    for t in threads:
        t.join(timeout=5)
    assert len(targets) == 1 and targets[0].parent == tmp_path and targets[0].name.startswith("split_")
    assert str(targets[0]) in reply


# --- //nightly --------------------------------------------------------------------------

def _nightly(monkeypatch, query, unavailable=None):
    """Runs the query with the run itself recorded instead of executed; returns (reply, runs)."""
    import assistant.agents.CodeQualityMaintainer as cqm
    runs, threads = [], []
    monkeypatch.setattr(cqm.CodeQualityMaintainer, "unavailable_reason", staticmethod(lambda: unavailable))
    monkeypatch.setattr(cqm.CodeQualityMaintainer, "_execute_nightly_run", lambda self, **kw: runs.append(kw))
    real_thread = dc_mod.threading.Thread

    class JoinedThread(real_thread):
        def start(self):
            super().start()
            threads.append(self)

    monkeypatch.setattr(dc_mod.threading, "Thread", JoinedThread)
    reply = DebugCommands().handle(query)
    for t in threads:
        t.join(timeout=5)
    return reply, runs


def test_nightly_runs_the_whole_night_even_with_the_schedule_off(monkeypatch):
    _, runs = _nightly(monkeypatch, "//nightly")
    assert runs == [{'force': True, 'only': None}]


def test_nightly_runs_one_named_step(monkeypatch):
    _, runs = _nightly(monkeypatch, "//nightly Health")
    assert runs == [{'force': True, 'only': "health"}]


def test_nightly_rejects_an_unknown_step(monkeypatch):
    reply, runs = _nightly(monkeypatch, "//nightly everything")
    assert runs == [] and "health" in reply


def test_nightly_does_not_start_while_the_integration_is_off(monkeypatch):
    reply, runs = _nightly(monkeypatch, "//nightly", unavailable="Self Improvement is disabled")
    assert runs == [] and reply == "Self Improvement is disabled"


def test_nightly_does_not_start_a_second_run(monkeypatch):
    import assistant.agents.CodeQualityMaintainer as cqm
    with cqm._nightly_lock:
        reply, runs = _nightly(monkeypatch, "//nightly")
    assert runs == [] and "already" in reply


# --- //voice_discover -------------------------------------------------------------------

def test_voice_discover_without_a_discovery_says_so(monkeypatch):
    monkeypatch.setattr(DebugCommands, "voice_discovery", None)
    reply = DebugCommands().handle("//voice_discover")
    assert "not running" in reply


def test_voice_discover_runs_the_handed_over_run_and_leaves_the_summary_in_the_history(monkeypatch):
    """The voice interface hands its run over; the command runs it on a thread and the summary
    lands in the dialog as a system row. A second call after the first finished runs again."""
    runs, rows, threads = [], [], []

    def run():
        runs.append(1)
        return {"filed": 4}

    class History:
        def add_context_message(self, role, text, when, source=None):
            rows.append((role, text, source))

    real_thread = dc_mod.threading.Thread

    class JoinedThread(real_thread):
        def start(self):
            super().start()
            threads.append(self)

    monkeypatch.setattr(dc_mod.threading, "Thread", JoinedThread)
    monkeypatch.setattr(DebugCommands, "voice_discovery", staticmethod(run))
    commands = DebugCommands(history=History())
    reply = commands.handle("//voice_discover")
    for t in threads:
        t.join(timeout=5)
    assert "background" in reply and runs == [1]
    assert rows and rows[0][0] == "system" and "'filed': 4" in rows[0][1]
    commands.handle("//voice_discover")
    for t in threads:
        t.join(timeout=5)
    assert runs == [1, 1]


def test_voice_discover_runs_one_at_a_time_and_a_failed_run_frees_the_slot(monkeypatch):
    """A second //voice_discover while the first is still embedding is refused, not queued —
    two runs would cluster the same clips twice. A run that raises leaves the failure in the
    dialog and the slot free, so the next call runs again instead of being refused for good."""
    started, release = threading.Event(), threading.Event()
    runs, rows, threads = [], [], []

    def run():
        runs.append(1)
        started.set()
        release.wait(timeout=5)
        raise RuntimeError("no clips this week")

    class History:
        def add_context_message(self, role, text, when, source=None):
            rows.append(text)

    real_thread = dc_mod.threading.Thread

    class JoinedThread(real_thread):
        def start(self):
            super().start()
            threads.append(self)

    monkeypatch.setattr(dc_mod.threading, "Thread", JoinedThread)
    monkeypatch.setattr(DebugCommands, "voice_discovery", staticmethod(run))
    commands = DebugCommands(history=History())
    try:
        commands.handle("//voice_discover")
        assert started.wait(timeout=5)
        second = commands.handle("//voice_discover")
        assert runs == [1], "the run in progress was not started again"
        assert "progress" in second
    finally:
        release.set()
    for t in threads:
        t.join(timeout=5)
    assert len(rows) == 1 and "failed" in rows[0]
    third = commands.handle("//voice_discover")
    for t in threads:
        t.join(timeout=5)
    assert runs == [1, 1] and "progress" not in third


# --- //testcase -------------------------------------------------------------------------

class _ReplyHistory:
    def __init__(self, entries):
        self._entries = entries

    def get_merged_entries(self):
        return self._entries


@pytest.fixture
def testcase_prompt(tmp_path, monkeypatch):
    prompt = tmp_path / "current_prompt.md"
    prompt.write_text("# [SYSTEM]\n\nYou are Michelle.\n\n# [USER]\n\nWhere is my car?\n")
    monkeypatch.setattr(dc_mod, "CURRENT_PROMPT_FILE", str(prompt))
    return prompt


def _mailer(monkeypatch, result="m@mlab.test", side_effect=None):
    from assistant.agents.CodeQualityMaintainer import CodeQualityMaintainer
    sent = {}

    def send(subject, html_body, attachments=()):
        if side_effect:
            raise side_effect
        sent.update(subject=subject, body=html_body, attachments=list(attachments))
        return result

    monkeypatch.setattr(CodeQualityMaintainer, "_send_email", staticmethod(send))
    return sent


def test_testcase_mails_the_prompt_with_the_comment_and_the_last_reply(testcase_prompt, monkeypatch):
    sent = _mailer(monkeypatch)
    history = _ReplyHistory([
        {"role": "Marcus", "content": "Where is my car?"},
        {"role": "assistant", "content": "Your car is at home."},
        {"role": "command", "content": "result: ..."},
    ])
    reply = DebugCommands(history=history).handle("//testcase the ai should not have said my car is at home")

    assert "the ai should not have said my car is at home" in sent["subject"]
    assert "the ai should not have said my car is at home" in sent["body"]
    assert "Your car is at home." in sent["body"]
    [(name, text)] = sent["attachments"]
    sent_at = datetime.fromtimestamp(testcase_prompt.stat().st_mtime)
    assert name == f"the-ai-should-not-have-said-my-car-is-at-home-{sent_at:%Y%m%d-%H%M%S}.md"
    assert text == testcase_prompt.read_text()
    assert "m@mlab.test" in reply


def test_testcase_attachment_name_keeps_only_letters_and_digits_of_the_comment(testcase_prompt, monkeypatch):
    sent = _mailer(monkeypatch)
    DebugCommands().handle("//testcase  Bilen: står hemma?! (Nr 2) ")

    [(name, _)] = sent["attachments"]
    stem = name.rsplit("-", 2)[0]
    assert stem == "bilen-st-r-hemma-nr-2"


def test_testcase_attachment_name_is_cut_for_a_long_comment(testcase_prompt, monkeypatch):
    sent = _mailer(monkeypatch)
    DebugCommands().handle("//testcase " + "word " * 40)

    [(name, _)] = sent["attachments"]
    stem = name.rsplit("-", 2)[0]
    assert len(stem) <= 60 and not stem.endswith("-") and stem.startswith("word-word")


def test_testcase_without_a_comment_or_history_still_mails_the_prompt(testcase_prompt, monkeypatch):
    sent = _mailer(monkeypatch)
    reply = DebugCommands().handle("//TESTCASE")

    [(name, text)] = sent["attachments"]
    assert name.startswith("prompt-") and name.endswith(".md")
    assert text == testcase_prompt.read_text()
    assert "no comment" in reply
    assert "No assistant reply" in sent["body"]


def test_testcase_without_a_prompt_dump_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(dc_mod, "CURRENT_PROMPT_FILE", str(tmp_path / "missing.md"))
    sent = _mailer(monkeypatch)
    reply = DebugCommands().handle("//testcase a comment")
    assert not sent
    assert "No prompt has been sent" in reply


def test_testcase_without_a_mail_account_says_so(testcase_prompt, monkeypatch):
    _mailer(monkeypatch, result=None)
    reply = DebugCommands().handle("//testcase")
    assert "No mail account" in reply


def test_testcase_reports_a_failed_send(testcase_prompt, monkeypatch):
    _mailer(monkeypatch, side_effect=RuntimeError("report email not sent: SMTP boom"))
    reply = DebugCommands().handle("//testcase")
    assert "not mailed" in reply and "SMTP boom" in reply
