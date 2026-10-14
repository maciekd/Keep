"""Debug commands invoked by sending a query that starts with '//'.

The '//' prefix (not a single '/', which Telegram intercepts as a bot command)
marks a debug command. These are intercepted at the interfaces (the HTTP /stream
route and every chat transport) and never reach the LLM. Each command is a
`cmd_<NAME>(self, args: str) -> str` method, dispatched dynamically by name, so
adding a new command is just adding a method; `args` is whatever followed the
name. The first line of the method's docstring is its description in //help.
A command returns a human-readable string that is relayed back as the reply.

    //help          - list the commands
    //FP            - last conversations were False Positives -> copy their audio to false-positive/
    //TP            - last conversations were True Positives  -> copy their audio to en/
    //wipe confirm  - back up and wipe the user state, then stop the server (FTUE testing)
    //restart       - pull the target branch and restart the server, as the restart_server command does (fallback when the LLM is unusable)
    //feed <pause|resume> <all|name> - hold / carry on filling the history from a feed; bare //feed lists them
    //memory_curate <start> [end]    - curate the long-term memory over a span of days (YYYYMMDD, end inclusive), in 4-day chunks
    //memory_split [apply]           - propose the split of an old one-file bank into bank / watch list / procedures; apply the latest proposal
    //nightly [bugs|fix|tests|health|merge] - run the nightly code-quality run now, or one step of it (fix runs the bug analysis first)
    //first_run [stage <name>|reset] - the first-run pipeline's state; force a stage or delete the state (FTUE testing)
    //toolkits      - the toolkits with their size and whether each is in the LLM's history window
    //toolkit <name> - the reference of one toolkit as the LLM receives it
"""

import os
import shutil
import signal
import sqlite3
import threading
import zipfile
from contextlib import closing
from datetime import date, datetime, timedelta
from pathlib import Path

from assistant import FirstRun
from common.logging import info, error
from common.Paths import (AISERVER_DIR, CACHE_DIR, CONFIG_PATH, DUMMY_CONFIG_PATH, ICLOUD_SESSION_DIR, LOG_DIR,
                          MEMORY_DIR, MEMORY_LOG_DIR, VAULT_PATH)

# Anchored on common.Paths so cwd doesn't matter.
AUDIO_DIR = str(LOG_DIR / "audio")
# Where //FP and //TP collect labelled audio; debug.false_positive_dir / debug.true_positive_dir
# point them at an STT test corpus kept elsewhere.
FALSE_POSITIVE_DIR = str(CACHE_DIR / "stt_samples" / "false-positive")
TRUE_POSITIVE_DIR = str(CACHE_DIR / "stt_samples" / "en")

# How many of the latest samples //FP and //TP grab on each invocation.
RECENT_AUDIO_COUNT = 10

# --- //wipe -----------------------------------------------------------------------------
# Everything a returning user has that a fresh install hasn't. Files and directories
# alike; a missing one is skipped. Each is zipped into BACKUP_DIR before it is deleted.
DATABASE_PATH = AISERVER_DIR / "floxy.db"
BACKUP_DIR = CACHE_DIR / "wipe-backups"
# The sessions an integration keeps in the tree (the Telegram feed's, WhatsApp's) are the
# integration's own `wipe_targets`, gathered by integration_targets().
WIPE_TARGETS = [
    (DATABASE_PATH, "database: dialog history, usage stats, photo index, feeds"),
    (CONFIG_PATH, "config.yaml (the next start writes a skeleton and asks for the required keys)"),
    (VAULT_PATH, "vault"),
    (MEMORY_DIR, "long-term memory: bank, procedures, watch list"),
    (AISERVER_DIR / "memory.md", "long-term memory bank of an install from before memory/"),
    (FirstRun.STATE_FILE, "first-run pipeline state"),
    (ICLOUD_SESSION_DIR, "iCloud session, trust token and cookies"),
    (CACHE_DIR / "config-backups", "config backups (hold the old credentials)"),
    (CACHE_DIR / "last_startup.json", "startup tracker"),
    (CACHE_DIR / "tui_state.json", "TUI state"),
    (CACHE_DIR / "audio_restarts.json", "audio recovery state"),
]
# Deleted without a backup: too big to zip, and rebuilt by logging in again. Deleted
# only once the browser is closed: Chrome writes into its profile as long as it runs,
# so an rmtree of a live profile ends in "Directory not empty".
WIPE_ONLY_TARGETS = [
    (CACHE_DIR / "chrome_profile", "browser profile (web logins)"),
]
# SQLite sidecars: their content is in the consistent copy the backup API makes.
DATABASE_SIDECARS = [DATABASE_PATH.with_name(DATABASE_PATH.name + s) for s in ("-wal", "-shm", "-journal")]
# The reply must reach the user before the process goes down.
STOP_DELAY_SECONDS = 2.0
# Held for the whole of one `//wipe confirm`. A second one sent while the first is still
# zipping (a 270 MB database takes ~30 s with no visible progress) used to run alongside
# it and write into the same database snapshot the first was zipping: a torn backup.
_wipe_lock = threading.Lock()


def integration_targets() -> list[tuple[Path, str]]:
    """The signed-in state the integration folders keep in the tree, each folder's own
    `Integration.wipe_targets`, in discovery order."""
    from assistant.integration.bootstrap import integration_classes
    targets = []
    for cls in integration_classes():
        targets += cls.wipe_targets()
    return targets


def external_targets() -> list[tuple[Path, str]]:
    """Login tokens the camera libraries and pyicloud keep outside the tree. pyicloud's
    temp-dir session is one to get: ICloudIntegration adopts it whenever
    cache/icloud_session is empty, which would quietly undo the wipe."""
    from common.ConfigManager import config
    targets = [
        (Path(config.get("ring.token_path", "/tmp/ring_floxy/token.cache")), "Ring login token"),
        (Path(config.get("arlo.storage_dir", "/tmp/pyaarlo_floxy")), "Arlo session"),
    ]
    try:
        from pyicloud.base import resolve_cookie_directory
        targets.append((Path(resolve_cookie_directory()), "pyicloud temp-dir session (legacy)"))
    except Exception as e:
        info(f"//wipe: pyicloud session dir not resolvable ({e}), skipping it")
    return targets


def _stop_server():
    """The same graceful shutdown as Ctrl-C: HttpInterface's SIGTERM handler ends serve(),
    main() stops every interface. Nothing starts it again — that is the user's move."""
    info("//wipe: stopping the server")
    os.kill(os.getpid(), signal.SIGTERM)


def _pull_and_restart():
    """//restart's tail, run once the reply is out: exactly what the restart_server command
    does — pull the target branch, relaunch through start.sh, note the outcome in the dialog."""
    from assistant.agents.CodeQualityMaintainer import CodeQualityMaintainer
    from assistant.SystemNote import system_note
    try:
        result = CodeQualityMaintainer.pull_and_restart()
        info(f"//restart: {result}")
        system_note("server", f"Server restart (//restart): {result}")
    except Exception as e:
        error(f"//restart failed: {e}")


def _close_browser():
    """Close Web Access's Chrome (and kill whatever of it lingers) so its profile can go.
    Only an already constructed WebAccess is touched: a wipe never starts a browser."""
    from assistant.tools.WebAccess import WebAccess
    if WebAccess._instance is None:
        info("//wipe: no browser running, nothing to close")
        return
    info("//wipe: closing the browser before deleting its profile")
    WebAccess.reset_instance()


def _install_dummy_config():
    """Development stand-in: config.dummy.yaml becomes the new config.yaml, so the next start
    has keys and boots straight into the first run instead of asking for them."""
    if not DUMMY_CONFIG_PATH.exists():
        info(f"//wipe: no {DUMMY_CONFIG_PATH.name}, the next start writes the skeleton config")
        return
    try:
        shutil.copyfile(DUMMY_CONFIG_PATH, CONFIG_PATH)
    except OSError as e:
        error(f"//wipe: could not install {DUMMY_CONFIG_PATH.name} as {CONFIG_PATH.name}: {e}")
        return
    info(f"//wipe: installed {DUMMY_CONFIG_PATH.name} as {CONFIG_PATH.name}")


def _finish_wipe(after_browser: list[tuple[Path, str]]):
    """The wipe's tail, run once the reply is out: close the browser, delete what needs
    it closed, put the dummy config in place, stop the server."""
    if after_browser:
        _close_browser()
        for path, _ in after_browser:
            DebugCommands._delete(path)
    _install_dummy_config()
    _stop_server()


class DebugCommands:
    """Handles the `//NAME` debug queries that are intercepted before they reach the LLM.
    `history` is the assistant's MessageHistory, for the commands that leave a system row
    in the dialog (//memory_curate's completion); None when there is no assistant."""
    def __init__(self, history=None):
        self.history = history

    @classmethod
    def for_assistant(cls, assistant) -> 'DebugCommands':
        """The commands with the assistant's history; a stand-in assistant without a prompt
        manager (tests, a hub with nothing attached yet) gets the commands without one."""
        prompt_manager = getattr(assistant, "prompt_manager", None)
        return cls(history=prompt_manager.message_history if prompt_manager is not None else None)

    @staticmethod
    def is_command(query: str) -> bool:
        return query.strip().startswith('//')

    def handle(self, query: str) -> str:
        parts = query.strip()[2:].split(None, 1)
        name = parts[0] if parts else ""
        args = parts[1].strip() if len(parts) > 1 else ""
        method = getattr(self, f"cmd_{name.upper()}", None)
        if not method:
            info(f"Unknown debug command: //{name}")
            return f"Unknown command: //{name}\n{self.cmd_HELP('')}"
        info(f"Executing debug command: //{name} {args}".rstrip())
        return method(args)

    def commands(self) -> list[tuple[str, str]]:
        """(name, description) of every cmd_* method, read off the class on each call so a
        new method shows up in //help at once."""
        result = []
        for attr in sorted(dir(self)):
            if attr.startswith("cmd_"):
                doc = (getattr(self, attr).__doc__ or "").strip()
                result.append((attr[4:], doc.splitlines()[0] if doc else ""))
        return result

    def cmd_HELP(self, args: str) -> str:
        """List the debug commands."""
        commands = self.commands()
        width = max(len(name) for name, _ in commands)
        lines = ["Debug commands:"]
        lines += [f"  //{name.ljust(width)}  {desc}" for name, desc in commands]
        return "\n".join(lines)

    def cmd_TOOLKITS(self, args: str) -> str:
        """The toolkits: size, commands, and when each was loaded into today's history window."""
        from . import Toolkits
        from assistant.integration.IntegrationMonitor import IntegrationMonitor
        loaded = self.history.loaded_toolkits(window=None) if self.history is not None else {}
        monitor = IntegrationMonitor.get_instance()
        lines = [f"Toolkits ({len(Toolkits.TOOLKITS)}):"]
        width = max(len(t.name) for t in Toolkits.TOOLKITS)
        for toolkit in Toolkits.TOOLKITS:
            size = f"{len(toolkit.text()):5d} chars" if toolkit.path.exists() else "  MISSING FILE"
            if not toolkit.offered_by(monitor):
                state = "integration off"
            elif toolkit.name in loaded:
                state = "loaded " + datetime.fromtimestamp(loaded[toolkit.name]).strftime("%H:%M")
            else:
                state = "not loaded"
            lines.append(f"  {toolkit.name.ljust(width)}  {size}  {state:15s} {', '.join(toolkit.commands)}")
        return "\n".join(lines)

    def cmd_TOOLKIT(self, args: str) -> str:
        """The reference of one toolkit as the LLM receives it: //toolkit <name>."""
        from . import Toolkits
        toolkit = Toolkits.by_name(args) if args.strip() else None
        if toolkit is None:
            return f"Usage: //toolkit <name>. Toolkits: {', '.join(Toolkits.names())}"
        from assistant.agents.AutonomousActionManager import AutonomousActionManager
        aam = AutonomousActionManager.current()
        substitute = aam.prompt_manager.render_template if aam and aam.prompt_manager else (lambda text: text)
        return Toolkits.render(toolkit, substitute)

    def cmd_FP(self, args: str) -> str:
        """Last conversations were false positives: copy their audio to the false-positive sample folder."""
        dest = self._sample_dir("debug.false_positive_dir", FALSE_POSITIVE_DIR)
        return self._copy_recent_audio(dest, "false-positive")

    def cmd_TP(self, args: str) -> str:
        """Last conversations were true positives: copy their audio to the true-positive sample folder."""
        dest = self._sample_dir("debug.true_positive_dir", TRUE_POSITIVE_DIR)
        return self._copy_recent_audio(dest, "en")

    @staticmethod
    def _sample_dir(config_key: str, default: str) -> str:
        from common.ConfigManager import config
        dest = os.path.expanduser(config.get(config_key, "") or default)
        info(f"Labelled audio goes to {dest} ({config_key if dest != default else 'default'}).")
        return dest

    def _recent_audio(self, count: int = RECENT_AUDIO_COUNT) -> list[str]:
        """The newest-mtime .wav files directly in log/audio/ (subdirs are
        skipped), newest first, capped at `count`."""
        if not os.path.isdir(AUDIO_DIR):
            return []
        wavs = [
            os.path.join(AUDIO_DIR, f)
            for f in os.listdir(AUDIO_DIR)
            if f.lower().endswith(".wav") and os.path.isfile(os.path.join(AUDIO_DIR, f))
        ]
        wavs.sort(key=os.path.getmtime, reverse=True)
        return wavs[:count]

    def _copy_recent_audio(self, dest_dir: str, label: str) -> str:
        srcs = self._recent_audio()
        if not srcs:
            info(f"No audio files found in {AUDIO_DIR}, nothing to copy.")
            return "No audio files found to label."
        copied = []
        try:
            os.makedirs(dest_dir, exist_ok=True)
            for src in srcs:
                dest = os.path.join(dest_dir, os.path.basename(src))
                shutil.copy2(src, dest)
                copied.append(os.path.basename(src))
        except OSError as e:
            error(f"Failed to copy {src} to {dest_dir}: {e}")
            return f"Failed to copy audio: {e}"
        info(f"Copied {len(copied)} file(s) to {label}/: {', '.join(copied)}")
        return f"Copied {len(copied)} file(s) to {label}: {', '.join(copied)}"

    # --- //feed ---------------------------------------------------------------------------

    def cmd_FEED(self, args: str) -> str:
        """//feed <pause|resume> <all|name>: hold or carry on filling the history from a feed (iMessage, Telegram…); bare //feed lists them."""
        from assistant.integration.FeedRunner import Feeds, describe_feed_status
        feeds = Feeds.get_instance()
        names = feeds.names()
        if not names:
            info("//feed: no feeds are running")
            return "No feeds are running."
        parts = args.split(None, 1)
        if not parts:
            return "Feeds:\n" + "\n".join(f"  {n}: {describe_feed_status(feeds.status(n))}" for n in names)
        action = parts[0].lower()
        if action not in ("pause", "resume") or len(parts) != 2:
            return f"Usage: //feed <pause|resume> <all|{'|'.join(names)}>"
        if parts[1].lower() == "all":
            targets = [feeds.runner(n) for n in names]
        else:
            runner = feeds.find(parts[1])
            if runner is None:
                info(f"//feed: no feed named '{parts[1]}'")
                return f"No feed named '{parts[1]}'. Feeds: {', '.join(names)}"
            targets = [runner]
        lines = []
        for runner in targets:
            changed = runner.pause() if action == "pause" else runner.resume()
            verb = ("Paused" if action == "pause" else "Resumed") if changed \
                else ("Already paused" if action == "pause" else "Not paused")
            lines.append(f"{verb} {runner.name} ({describe_feed_status(runner.status())})")
        return "\n".join(lines)

    # --- //memory_curate ------------------------------------------------------------------

    def cmd_MEMORY_CURATE(self, args: str) -> str:
        """//memory_curate <start> [end]: curate the long-term memory over those days (YYYYMMDD, end inclusive, default = start); runs in the background, result in log/memory/."""
        usage = "Usage: //memory_curate <start> [end], dates as YYYYMMDD (end inclusive), e.g. //memory_curate 20261001 20261030"
        parts = args.split()
        if not 1 <= len(parts) <= 2:
            return usage
        try:
            start = datetime.strptime(parts[0], "%Y%m%d").date()
            end = datetime.strptime(parts[-1], "%Y%m%d").date()
        except ValueError:
            info(f"//memory_curate: bad date(s) in '{args}'")
            return usage
        if end < start:
            return f"The end ({end:%Y-%m-%d}) is before the start ({start:%Y-%m-%d}). {usage}"
        if start > date.today():
            info(f"//memory_curate: {start:%Y-%m-%d} is in the future, nothing to curate")
            return f"{start:%Y-%m-%d} is in the future: no history to curate yet."
        from assistant.agents.LongTermMemoryCurator import CURATION_CHUNK_DAYS
        end = min(end, date.today())
        days = (end - start).days + 1
        chunks = -(-days // CURATION_CHUNK_DAYS)
        info(f"//memory_curate: starting curation for {start:%Y-%m-%d} .. {end:%Y-%m-%d} "
             f"({chunks} chunk(s)) in the background")
        threading.Thread(target=self._curate_memory, args=(start, end, self.history), daemon=True).start()
        span = f"{start:%Y-%m-%d}" if days == 1 else \
            f"{start:%Y-%m-%d} .. {end:%Y-%m-%d} ({days} days, {chunks} chunks of up to {CURATION_CHUNK_DAYS} days)"
        return (f"Curating the long-term memory over {span} in the background. "
                f"It takes a few minutes per chunk; the bank and its diff land in log/memory/, progress is in the log.")

    @staticmethod
    def _curate_memory(start: date, end: date, history):
        """The curation run itself, on its own thread: the span in chunks, one full LLM pass
        each; every completion lands in `history` as a system row."""
        from assistant.agents.LongTermMemoryCurator import LongTermMemoryCurator
        try:
            summary = LongTermMemoryCurator.run_span(start, end, history=history)
        except Exception as e:
            error(f"//memory_curate: curation failed: {e}")
            return
        if summary is None:
            info("//memory_curate: skipped, another curation was already running")

    # --- //memory_split -------------------------------------------------------------------

    def cmd_MEMORY_SPLIT(self, args: str) -> str:
        """//memory_split [apply]: split a one-file memory bank into bank / watch list / procedures — a proposal in log/memory/split_<time>/ to read and edit first; `apply` puts the latest proposal in place."""
        action = args.strip().lower()
        if action not in ("", "apply"):
            return "Usage: //memory_split (writes a proposal) or //memory_split apply (puts the latest proposal in place)"
        from assistant.agents.LongTermMemoryCurator import LongTermMemoryCurator
        if action == "apply":
            proposals = sorted(p for p in MEMORY_LOG_DIR.glob("split_*") if (p / "bank.md").is_file())
            if not proposals:
                info("//memory_split apply: no proposal in log/memory/")
                return "There is no split proposal yet. Run //memory_split first."
            try:
                done = LongTermMemoryCurator().apply_split(proposals[-1])
            except Exception as e:
                error(f"//memory_split apply: failed: {e}")
                return f"Applying {proposals[-1].name} failed: {e}"
            kept = f" {len(done['kept'])} already existed and were kept: {', '.join(done['kept'])}." \
                if done['kept'] else ""
            return (f"Applied {proposals[-1].name}: the bank is replaced, {len(done['watch_items'])} watch item(s) and "
                    f"{len(done['procedures'])} procedure(s) are in place.{kept} "
                    f"The memory as it was is in {done['backup']}.")
        target = MEMORY_LOG_DIR / f"split_{datetime.now():%Y%m%d_%H%M%S}"
        info(f"//memory_split: writing a proposal to {target} in the background")
        threading.Thread(target=self._propose_split, args=(target,), daemon=True).start()
        return (f"Splitting the memory bank in the background (two AI passes, a few minutes). The proposal lands in "
                f"{target} — bank.md, watchlist/, procedures/. Nothing live changes until //memory_split apply.")

    @staticmethod
    def _propose_split(target: Path):
        from assistant.agents.LongTermMemoryCurator import LongTermMemoryCurator
        try:
            LongTermMemoryCurator().propose_split(target)
        except Exception as e:
            error(f"//memory_split: the proposal failed: {e}")

    # --- //nightly ------------------------------------------------------------------------

    def cmd_NIGHTLY(self, args: str) -> str:
        """//nightly [bugs|fix|tests|health|merge]: run the nightly code-quality run now (bug analysis, a PR fixing its Critical and Major findings, test generation, test health, auto-merge), or one step alone (fix runs the bug analysis first, it reads from it) — in the background, also with self_improvement.nightly_runs off; the report is emailed."""
        from assistant.agents import CodeQualityMaintainer as cqm
        only = args.strip().lower() or None
        if only and only not in cqm.NIGHTLY_STEPS:
            return f"Usage: //nightly (the whole run) or //nightly <{'|'.join(cqm.NIGHTLY_STEPS)}> (one step)"
        reason = cqm.CodeQualityMaintainer.unavailable_reason()
        if reason:
            info(f"//nightly: not started — {reason}")
            return reason
        if cqm._nightly_lock.locked():
            info("//nightly: not started, a nightly run is already going")
            return "A nightly run is already going. Follow it in log/floxy.out."
        info(f"//nightly: starting {only or 'the whole run'} in the background")
        threading.Thread(target=cqm.CodeQualityMaintainer()._execute_nightly_run, name="NightlyDebugRun",
                         kwargs={'force': True, 'only': only}, daemon=True).start()
        return (f"Started {f'the {only} step' if only else 'the nightly run'} in the background. "
                f"Each Claude Code step can take up to an hour; follow it in log/floxy.out "
                f"(CodeQualityMaintainer). The report is emailed when it is done.")

    # --- //first_run ----------------------------------------------------------------------

    def cmd_FIRST_RUN(self, args: str) -> str:
        """//first_run [stage <name> | reset]: the first-run pipeline's stage and progress; force a stage, or delete the state so the install counts as established."""
        parts = args.split()
        if parts and parts[0].lower() == "reset":
            if not FirstRun.STATE_FILE.exists():
                info("//first_run reset: no state file")
                return "No first-run state to reset."
            FirstRun.STATE_FILE.unlink()
            info("//first_run reset: state file deleted")
            return "First-run state deleted: the pipeline stops and the install counts as established."
        if parts and parts[0].lower() == "stage":
            names = ", ".join(s.value for s in FirstRun.Stage)
            if len(parts) != 2 or parts[1].lower() not in [s.value for s in FirstRun.Stage]:
                return f"Usage: //first_run stage <{names}>"
            state = FirstRun.State.load() or FirstRun.State.fresh()
            state.enter(FirstRun.Stage(parts[1].lower()))
            state.chunks_done = 0
            state.save()
            info(f"//first_run: stage forced to {state.stage.value}")
            return f"First-run stage set to {state.stage.value} (the pipeline picks it up at its next poll)."
        if parts:
            return "Usage: //first_run [stage <name> | reset]"
        state = FirstRun.State.load()
        if state is None:
            return "No first-run pipeline on this install (established: no first_run.json)."
        lines = [f"First-run pipeline: {state.stage.value} since {datetime.fromtimestamp(state.entered_at):%Y-%m-%d %H:%M}"
                 f" (started {datetime.fromtimestamp(state.started_at):%Y-%m-%d %H:%M})"]
        if state.span:
            lines.append(f"First curation: {state.curation_start} .. {state.curation_end}, "
                         f"chunk {state.chunks_done}/{state.chunks_total}")
        from assistant.integration.FeedRunner import Feeds, describe_feed_status
        for name in (FirstRun.IMESSAGE_FEED, FirstRun.PHOTOS_FEED):
            runner = Feeds.get_instance().runner(name)
            lines.append(f"{name}: {describe_feed_status(runner.status() if runner else None)}")
        return "\n".join(lines)

    # --- //restart ------------------------------------------------------------------------

    def cmd_RESTART(self, args: str) -> str:
        """Pull the target branch and restart the server, exactly as the restart_server command does; the fallback when the LLM is unusable."""
        from assistant.agents.CodeQualityMaintainer import CodeQualityMaintainer
        target = CodeQualityMaintainer.target_branch()
        threading.Timer(STOP_DELAY_SECONDS, _pull_and_restart).start()
        info(f"//restart: pulling {target} and restarting in {STOP_DELAY_SECONDS:g}s")
        return (f"Pulling {target} and restarting the server in {STOP_DELAY_SECONDS:g}s. "
                f"The outcome lands in the log and as a system note.")

    # --- //wipe ---------------------------------------------------------------------------

    def cmd_WIPE(self, args: str) -> str:
        """Simulate a new user: zip the user state into cache/wipe-backups, delete it (config.yaml included) and stop the server. Needs `//wipe confirm`."""
        present = [(p, d) for p, d in WIPE_TARGETS + integration_targets() + external_targets() if p.exists()]
        wipe_only = [(p, d) for p, d in WIPE_ONLY_TARGETS if p.exists()]
        if args.lower() != "confirm":
            info(f"//wipe without confirm: {len(present)} item(s) would be backed up and wiped")
            return self._wipe_plan(present, wipe_only)
        if not present and not wipe_only:
            info("//wipe: nothing to wipe, the install is already fresh")
            return "Nothing to wipe: the install is already fresh."
        if not _wipe_lock.acquire(blocking=False):
            info("//wipe confirm while a wipe is already running: ignored")
            return "A wipe is already running (the backup takes a while); wait for its reply."
        try:
            return self._wipe(present, wipe_only)
        finally:
            _wipe_lock.release()

    def _wipe(self, present, wipe_only) -> str:
        try:
            zip_path = self._backup(present) if present else None
        except Exception as e:
            error(f"//wipe: backup failed, nothing deleted: {e}")
            return f"Backup failed, nothing was deleted: {e}"

        notes = self._log_out(present)
        deleted, failed = [], []
        for path, desc in present + [(p, "database sidecar") for p in DATABASE_SIDECARS]:
            note = f" — {notes[path]}" if path in notes else ""
            if not path.exists():
                if note:    # Telethon's log-out deletes the file itself
                    deleted.append(f"  - {self._display(path)}: {desc}{note}")
                continue
            problem = self._delete(path)
            if problem:
                failed.append(f"  - {self._display(path)}: {desc}{note} — {problem}")
            else:
                deleted.append(f"  - {self._display(path)}: {desc}{note}")

        threading.Timer(STOP_DELAY_SECONDS, _finish_wipe, args=(wipe_only,)).start()
        info(f"//wipe: done, {len(deleted)} item(s) wiped, {len(failed)} failed, "
             f"{len(wipe_only)} deleted after the browser closes, server stops in {STOP_DELAY_SECONDS:g}s")
        size = f" ({zip_path.stat().st_size / 1e6:.0f} MB)" if zip_path else ""
        lines = [f"Wiped {len(deleted)} item(s)"
                 + (f", backup: {self._display(zip_path)}{size}" if zip_path else ", nothing to back up")]
        lines += deleted
        if failed:
            lines.append("Could not delete (see the log):")
            lines += failed
        if wipe_only:
            lines.append("Deleted once the browser is closed:")
            lines += [f"  - {self._display(p)}: {d}" for p, d in wipe_only]
        lines.append(f"Stopping the server in {STOP_DELAY_SECONDS:g}s. Start it again yourself: {self._next_start()}")
        return "\n".join(lines)

    @staticmethod
    def _log_out(present) -> dict[Path, str]:
        """The server-side log-outs before the files go: a Telethon session stays authorized
        on the account after its file is deleted, a WhatsApp link stays a linked device. Every
        integration with a wipe target present gets its `log_out`; what happened goes on its
        first target, for the listing."""
        from assistant.integration.bootstrap import integration_classes
        present_paths = {path for path, _ in present}
        notes: dict[Path, str] = {}
        for cls in integration_classes():
            targets = [path for path, _ in cls.wipe_targets() if path in present_paths]
            if not targets:
                continue
            try:
                outcome = cls.log_out()
            except Exception as e:
                error(f"//wipe: {cls.__name__}.log_out raised {type(e).__name__}: {e}")
                outcome = f"log-out failed ({type(e).__name__}: {e})"
            if outcome:
                notes[targets[0]] = outcome
        return notes

    def _wipe_plan(self, present, wipe_only) -> str:
        if not present and not wipe_only:
            return "Nothing to wipe: the install is already fresh."
        lines = [f"//wipe would back up {len(present)} item(s) to {self._display(BACKUP_DIR)}/ and delete them:"]
        lines += [f"  - {self._display(p)}: {d}" for p, d in present]
        if wipe_only:
            lines.append("Deleted without a backup, once the browser is closed:")
            lines += [f"  - {self._display(p)}: {d}" for p, d in wipe_only]
        lines.append(f"The server then stops (no restart). {self._next_start()}")
        lines.append("Run `//wipe confirm` to do it. The backup takes a while (about 30 s for a large database); "
                     "the reply comes when it is done.")
        return "\n".join(lines)

    @staticmethod
    def _next_start() -> str:
        """What the next start finds: the development stand-in config, or no config at all."""
        if DUMMY_CONFIG_PATH.exists():
            return f"{DUMMY_CONFIG_PATH.name} is copied to config.yaml (development stand-in), so it starts with keys."
        return "Without config.yaml it writes a skeleton and asks for the anthropic and gemini keys."

    def _backup(self, targets) -> Path:
        """One zip of every target, the database as a consistent copy made by SQLite's
        backup API (other threads keep writing while we zip)."""
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = BACKUP_DIR / f"floxy-wipe-{datetime.now().strftime('%Y%m%d-%H%M%S')}.zip"
        info(f"//wipe: backing up {len(targets)} item(s) to {zip_path}")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for path, _ in targets:
                if path == DATABASE_PATH:
                    self._backup_database(zf, path, zip_path.with_suffix(".snapshot"))
                elif path.is_dir():
                    for f in sorted(p for p in path.rglob("*") if p.is_file()):
                        zf.write(f, self._arcname(f))
                else:
                    zf.write(path, self._arcname(path))
        info(f"//wipe: backup written, {zip_path.stat().st_size / 1e6:.1f} MB")
        return zip_path

    def _backup_database(self, zf: zipfile.ZipFile, path: Path, snapshot: Path):
        """`snapshot` is this backup's own scratch file, named after its zip: two backups
        must never share one."""
        snapshot.unlink(missing_ok=True)
        try:
            with closing(sqlite3.connect(path)) as src, closing(sqlite3.connect(snapshot)) as dst:
                src.backup(dst)
            zf.write(snapshot, self._arcname(path))
        finally:
            snapshot.unlink(missing_ok=True)

    @staticmethod
    def _arcname(path: Path) -> str:
        try:
            return str(path.relative_to(AISERVER_DIR))
        except ValueError:
            return "external/" + str(path).lstrip("/")

    @staticmethod
    def _display(path: Path) -> str:
        try:
            return str(path.relative_to(AISERVER_DIR))
        except ValueError:
            return str(path)

    @staticmethod
    def _delete(path: Path) -> str | None:
        """Delete a file or tree; the problem as text when it fails. One stubborn item must
        not abort the wipe and leave the server running on a wiped config."""
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except OSError as e:
            error(f"//wipe: could not delete {path}: {e}")
            return str(e)
        info(f"//wipe: deleted {path}")
        return None
