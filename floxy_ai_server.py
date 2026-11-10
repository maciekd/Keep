"""Floxy AI Server launcher: config → common singletons → Assistant → the enabled
interfaces (http, console, voice, telegram, imessage) → serve.

`build()` is importable without serving (tests boot a Telegram-only process with it);
`main()` runs only under __main__. A disabled interface's module is never imported.

With the root config key `role: imessage_satellite` (common/Role.py) the same entry point runs
no assistant at all: `_launch` hands over to `_serve_satellite`, which imports nothing of the
assistant, integrations or interfaces trees."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # packages live under AIServer/src

from common import grpc_env  # noqa: F401  — sets gRPC env vars; ahead of anything that could pull google.genai in

# Ahead of every other import: a library that builds its SSL context at import time (aiohttp)
# would otherwise keep the bundled CA list for good.
from common import SystemTrust
_TLS_TRUST = SystemTrust.install()

import asyncio
import atexit
import faulthandler
import logging
import os
import time

from colorama import init

import common.DevelopmentConfig as DevelopmentConfig
from common.DevelopmentConfig import PREPEND_USER_QUERY_OVERRIDE
from common.logging import error, info, trace, note
from common import ConfigMigrations, Migrations, Role, Version
from common.ConfigManager import config, REQUIRED_KEYS
from common.RequirementsChecker import RequirementsChecker
from common.StderrLogger import StderrLogger

init(autoreset=True, strip=False)

# Start order; stop() runs in reverse. http first (owns the Quart app), console early so
# its log tab sees the rest of the boot, voice before telegram (the Notifier), imessage last
# (its approvals are asked in the chats started before it).
INTERFACE_ORDER = ("http", "console", "voice", "telegram", "imessage")


def _interface_class(name: str):
    """Import an interface module only when it is enabled (voice pulls in the audio stack)."""
    if name == "http":
        from interfaces.http.HttpInterface import HttpInterface
        return HttpInterface
    if name == "console":
        from interfaces.console.ConsoleInterface import ConsoleInterface
        return ConsoleInterface
    if name == "voice":
        from interfaces.voice.VoiceInterface import VoiceInterface
        return VoiceInterface
    if name == "telegram":
        from interfaces.telegram.TelegramInterface import TelegramInterface
        return TelegramInterface
    if name == "imessage":
        from interfaces.imessage.IMessageInterface import IMessageInterface
        return IMessageInterface
    raise ValueError(name)


def _enabled(name: str, running_as_testsuite: bool) -> bool:
    """`interfaces.<name>.enabled`, with ConfigStore.INTERFACE_ENABLED_BY_DEFAULT (voice off) when
    unset — decided before the interface's module is imported."""
    from common import ConfigStore
    if name == "http":
        return True  # every other interface registers routes on its app
    if name in ("telegram", "console", "imessage") and running_as_testsuite:
        # The prompt suite runs one server per worker; a console per server would mean each
        # racing for the one TUI port and a tui.py client landing in a test server.
        info(f"{name.capitalize()} interface skipped in testsuite mode.")
        return False
    return ConfigStore.interface_enabled(name)


def _pool_size(running_as_testsuite: bool) -> int:
    if running_as_testsuite:
        return 15
    if 'PYCHARM_HOSTED' in os.environ:
        return 10
    return 25


def _register_llm_pre_heat(prompt_manager):
    from assistant.llm import create_llm, parse_llm_config
    pre_heat_llms = {}
    for tier in ('fast', 'smart'):
        llm_name, model = parse_llm_config(tier)
        key = f"{llm_name}:{model}" if model else llm_name
        if key not in pre_heat_llms:
            pre_heat_llms[key] = create_llm(llm_name, model)
    if not pre_heat_llms:
        trace("No LLMs to pre-heat.")
        return

    def _pre_heat_callback(messages):
        for llm in pre_heat_llms.values():
            llm.pre_heat_cache(messages)

    prompt_manager.on_prompt_assembled(_pre_heat_callback)
    info(f"KV cache pre-heating registered for: {', '.join(pre_heat_llms.keys())}")


def _post_startup_message(prompt_manager) -> str:
    """Leave the startup row (and the change log since the last start) in the history and
    return the announcement `_announce_startup` sends once the interfaces are up."""
    from common.StartupTracker import StartupTracker
    previous = StartupTracker.read_previous_startup()
    commits = StartupTracker.get_commits_since(previous) if previous else []
    whats_new = StartupTracker.unannounced_release_notes()
    message = StartupTracker.build_history_message(commits, previous, whats_new)
    trace(f"Posting startup message to history: {message!r}")
    prompt_manager.message_history.add_context_message("system", message, time.time(), "startup")
    announced = Version.version() if whats_new or previous is None else None   # a first boot skips the note for good
    StartupTracker.write_current_startup(time.time(), announced)
    from assistant.agents.PullRequestTrial import PullRequestTrial
    return StartupTracker.build_announcement(commits, Version.label(), whats_new, PullRequestTrial.announcement_note())


def _announce_startup(server: "Server"):
    """Tell the owner's chats the server is up again and what changed since the last start.
    The startup row is already in the history, so a chat that draws the history (the TUI's
    Home tab) is skipped; a chat still connecting (Telegram) delivers it once it is up.
    Sent quietly: a restart is routine and may happen at night."""
    from assistant import SystemNote
    from assistant.Notifier import notifier
    if not server.startup_announcement:
        trace("No startup announcement to send.")
    else:
        sent = notifier().send_message(server.startup_announcement, disable_notification=True, in_history=True)
        info(f"Startup announcement {'sent to the chats' if sent else 'not sent, no chat took it'}: "
             f"{server.startup_announcement!r}")
    migration_note = ConfigMigrations.take_note()
    if migration_note:
        # What the config load moved, told once — here and not at build time, when no chat is up
        # yet to take it (2026-10-07: the note reached the history but the Telegram push was dropped).
        SystemNote.system_note("Config", migration_note)
    from assistant.agents.PullRequestTrial import PullRequestTrial
    rollback_note = PullRequestTrial.take_rollback_note()
    if rollback_note:
        # start.sh gave up on a PR trial's server and relaunched the previous branch: the user
        # asked for that trial and must hear it is over, and why.
        SystemNote.system_note("server", rollback_note)


def _open_first_conversation(server: "Server | None" = None):
    """In the FTUE the assistant speaks first: the first-run brief goes to it as a system
    turn in the owner's chat and its introduction is on screen before the user types. Runs
    once the assistant's pool is up and the chat interfaces are started. When the console
    runs, the brief waits until its startup intro has played for the first TUI client, so
    the assistant starts thinking as the chat appears, not while the intro is on."""
    from assistant import FirstRun
    if not FirstRun.brief_due():
        info("The first-run brief is retired: the assistant waits to be spoken to.")
        return
    _open_conversation(server, FirstRun.brief, "the first-run brief")


def _resume_conversation(server: "Server | None" = None):
    """A restart the assistant started from a chat (`restart_server`, `try_pull_request`) left
    a marker; the conversation it interrupted is resumed with a system turn, so the assistant
    reports the outcome and keeps what it promised instead of waiting for the next periodic
    check (issue #113). The announcement went out first, so the turn need not repeat it. A
    nightly update, a recovery or a manual start leaves no marker and resumes nothing."""
    from assistant import FirstRun
    from common.StartupTracker import StartupTracker
    marker = StartupTracker.take_resume_marker()
    if marker is None:
        return
    if FirstRun.brief_due():
        info("The restart's conversation is not resumed: the first-run brief opens the conversation instead.")
        return
    _open_conversation(server, lambda: StartupTracker.build_resume_turn(marker, Version.label()),
                       "the conversation the restart interrupted")


def _open_conversation(server: "Server | None", query, what: str):
    """Open the owner's conversation with the system turn *query()* builds. When the console
    runs, the turn waits until its startup intro has played for the first TUI client, so the
    assistant starts thinking as the chat appears, not while the intro is on."""
    from interfaces.chat.ChatHub import ChatHub

    def send():
        info(f"The assistant opens the conversation with {what}.")
        ChatHub.get_instance().open_conversation(query())

    console = next((i for i in server.interfaces if i.name == "console"), None) if server else None
    if console is None:
        send()
        return
    info(f"{what[0].upper()}{what[1:]} waits for the console's startup intro.")
    console.after_intro(send)


class Server:
    """Everything build() created, in start order, so main() can serve and stop it."""

    def __init__(self, assistant, interfaces: list, agents: list, integration_monitor, running_as_testsuite: bool):
        self.assistant = assistant
        self.interfaces = interfaces
        self.agents = agents
        self.integration_monitor = integration_monitor
        self.running_as_testsuite = running_as_testsuite
        self.startup_announcement: str | None = None
        self._stopped = False

    @property
    def http(self):
        return self.interfaces[0]

    def interface(self, name: str):
        return next((i for i in self.interfaces if i.name == name), None)

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        if not self.running_as_testsuite:
            try:
                from common.StartupTracker import StartupTracker
                StartupTracker.record_shutdown()
            except Exception:
                pass
            try:
                from assistant.Telemetry import Telemetry
                Telemetry.get_instance().stop()
            except Exception as e:
                trace(f"Telemetry stop failed: {e}")
        for iface in reversed(self.interfaces):
            try:
                iface.stop()
            except Exception as e:
                trace(f"{iface.name} interface stop failed: {e}")
        if self.integration_monitor is not None:
            try:
                from assistant.integration.FeedRunner import Feeds
                Feeds.get_instance().stop_all()
            except Exception as e:
                trace(f"Stopping feeds failed: {e}")
        for obj in [*self.agents, self.integration_monitor]:
            try:
                if obj is not None:
                    obj.stop()
            except Exception:
                pass
        try:
            self.assistant.stop()
        except Exception as e:
            trace(f"Assistant stop failed: {e}")
        try:
            from common.logging import shutdown_logging
            shutdown_logging()
        except Exception:
            pass


def _isolate_database(port: int):
    """A test server runs from the same root as the developer's live server, so its dialog
    would land in the live floxy.db and reach that server's window at its next reload. It
    gets a fresh file of its own instead, one per port."""
    from common.FloxyDatabase import FloxyDatabase
    from common.Paths import CACHE_DIR
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"testsuite-{port}.db"
    path.unlink(missing_ok=True)
    FloxyDatabase.DB_PATH = str(path)
    info(f"Testsuite server: database isolated at {path}")


def build(port: int, running_as_testsuite: bool = False) -> Server:
    """Construct the assistant and start every enabled interface; nothing serves yet."""
    from assistant.Assistant import Assistant
    from assistant.prompt.PromptManager import PromptManager
    from assistant.RequestProcessor import RequestProcessor
    from assistant import SystemNote
    from common.FloxyDatabase import FloxyDatabase
    from common.TempMonitor import TempMonitor
    from common.logging import set_log_masker
    from common.vault.Redactor import Redactor
    from common.vault.Vault import Vault

    info(f"Version {Version.detail()}")
    _trace_prepend_setting()
    TempMonitor.check_installed()
    if running_as_testsuite:
        _isolate_database(port)
    FloxyDatabase.get_instance()
    # A vault value the LLM writes back through a placeholder (a config_set of an api_hash)
    # is restored before the command runs and would otherwise land in floxy.out in clear.
    set_log_masker(Redactor(Vault.get_instance()).mask)
    if not running_as_testsuite:
        from assistant import FirstRun, MemoryFile
        seeded = MemoryFile.seed()  # before the prompt manager's memory section first reads the file
        FirstRun.begin_if_fresh(seeded)

    prompt_manager = PromptManager(running_as_testsuite, static_prompt=running_as_testsuite)
    SystemNote.set_history(prompt_manager.message_history)   # integrations' notes land in the dialog
    if running_as_testsuite:
        prompt_manager.load_test_suite_history("res/testsuite_history.json")

    agents = []
    integration_monitor = None
    startup_announcement = None
    if not running_as_testsuite:
        from assistant.agents.AutonomousActionManager import AutonomousActionManager
        from assistant.agents.CacheAuditor import CacheAuditor
        from assistant.agents.CodeQualityMaintainer import CodeQualityMaintainer
        from assistant.integration.bootstrap import discover_integrations
        _register_llm_pre_heat(prompt_manager)
        autonomous = AutonomousActionManager.get_instance(prompt_manager)
        startup_announcement = _post_startup_message(prompt_manager)
        agents = [CacheAuditor(), CodeQualityMaintainer()]
        for agent in agents:
            agent.start()
        agents.append(autonomous)  # its constructor already started it; stopped with the rest
        integration_monitor = discover_integrations()
        from assistant.Telemetry import Telemetry
        Telemetry.get_instance().start(prompt_stats=prompt_manager.get_section_stats)

    # The voice interface (if enabled) supplies the playback/TTS/turn-gate factories; it is
    # started below, before Assistant.start() fills the handler pool.
    interfaces = []

    def _new_request_handler() -> RequestProcessor:
        """The only place a RequestProcessor is built: fully wired for this process's interfaces."""
        voice = next((i for i in interfaces if i.name == "voice"), None)
        return RequestProcessor(**(voice.handler_kwargs() if voice else {}))

    assistant = Assistant(prompt_manager, _new_request_handler, _pool_size(running_as_testsuite))

    for name in INTERFACE_ORDER:
        if not _enabled(name, running_as_testsuite):
            info(f"{name} interface disabled (interfaces.{name}.enabled) — not imported or started.")
            continue
        note(f"Starting interface {name}...")
        cls = _interface_class(name)
        if not cls.enabled(config):
            continue
        if name == "http":
            iface = cls(port)
        elif name == "voice":
            iface = cls(port, running_as_testsuite)
        else:
            iface = cls()
        http = interfaces[0] if interfaces else None
        iface.start(assistant, http.app if http else None)
        interfaces.append(iface)
        if name == "voice":
            http.attach_voice(iface.text_stream, iface.microphone_routes)
            console = next((i for i in interfaces if i.name == "console"), None)
            if console is not None:
                console.attach_voice(iface)
    note(f"Interfaces started: {', '.join(i.name for i in interfaces)}")

    if not running_as_testsuite:
        from assistant.integration.bootstrap import start_feeds
        start_feeds(prompt_manager.message_history)
        integration_monitor.start()
        pipeline = FirstRun.Pipeline(prompt_manager.message_history)   # a no-op on an established install
        pipeline.start()
        agents.append(pipeline)
        from common import Watchdog
        Watchdog.sync(bool(config.get(Watchdog.KEY, False)))   # the launchd agent follows the key

    server = Server(assistant, interfaces, agents, integration_monitor, running_as_testsuite)
    server.startup_announcement = startup_announcement
    return server


async def main(server: Server):
    server.assistant.start()
    for iface in server.interfaces:
        await iface.start_async()
    if not server.running_as_testsuite:
        _announce_startup(server)
        _open_first_conversation(server)
        _resume_conversation(server)
    try:
        await server.http.serve()
    finally:
        note("Shutting down server!")
        for iface in reversed(server.interfaces):
            await iface.stop_async()
        # Stop everything here, not from atexit: the interpreter joins every non-daemon thread
        # BEFORE the atexit handlers run, so a thread that only ends once stop() tells it to
        # kept the process alive after "Clean shutdown complete" until start.sh's SIGKILL.
        _arm_exit_deadline()
        info("Interfaces stopped — stopping the assistant, agents, integrations and logging")
        server.stop()
        info("Clean shutdown complete")   # stdout only by now: the log writer stopped with the server


EXIT_DEADLINE_SECONDS = 5


def _arm_exit_deadline():
    """A stop step that hangs or a thread that outlives the shutdown is named, not waited
    for: after the deadline faulthandler writes every thread's stack to the real stderr
    (log/raw-floxy.log under start.sh) and ends the process itself, before start.sh would
    force-kill it with nothing to show for it."""
    if sys.__stderr__ is None:
        info("Exit deadline off: this process has no stderr to name a lingering thread on")
        return
    faulthandler.dump_traceback_later(EXIT_DEADLINE_SECONDS, exit=True, file=sys.__stderr__)
    trace(f"Exit deadline armed: a thread still alive in {EXIT_DEADLINE_SECONDS}s is dumped to stderr and the process ends")


def _enable_crash_traces():
    """A crash in native code (a C library aborting or segfaulting) raises no Python exception
    and so reaches no log: the process is just gone. faulthandler writes every thread's Python
    stack on the way down — to the real stderr (log/raw-floxy.log under start.sh), since it
    needs a file descriptor that outlives the interpreter's own objects."""
    if sys.__stderr__ is None:
        info("Crash traces off: this process has no stderr to write them to")
        return
    faulthandler.enable(file=sys.__stderr__, all_threads=True)
    trace("Crash traces on: a native crash writes every thread's Python stack to stderr")


def _parse_args():
    sys.stderr = StderrLogger(sys.stderr)
    # httpx logs every request URL at INFO, and the Telegram Bot API's URL carries the bot token (issue #97)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if "--developer-mode" in sys.argv:
        sys.argv.remove("--developer-mode")
        DevelopmentConfig.LOCAL_DEVELOPMENT_MODE = True
    port = 8888 if len(sys.argv) < 2 else int(sys.argv[1])
    running_as_testsuite = len(sys.argv) >= 3 and sys.argv[2] == "testsuite"
    if running_as_testsuite:
        import common.logging
        common.logging.disable_file_logging = True
        # A real server process (not pytest itself), so RUNNING_UNDER_TEST wasn't
        # auto-detected — set it so integrations don't start real background work.
        DevelopmentConfig.RUNNING_UNDER_TEST = True
        note("Running as TestSuite")

    note("Keep AI Server starting...")
    if DevelopmentConfig.LOCAL_DEVELOPMENT_MODE:
        trace("LOCAL_DEVELOPMENT_MODE enabled")
    return port, running_as_testsuite


def _trace_prepend_setting():
    """Which prepend setting the dialog requests start under; the config key is re-read at each
    request, so this is only how the server boots."""
    source = ("FLOXY_PREPEND_USER_QUERY" if PREPEND_USER_QUERY_OVERRIDE is not None
              else f"config key {DevelopmentConfig.PREPEND_USER_QUERY_KEY}")
    override = DevelopmentConfig.prepend_user_query_override()
    if override is None:
        trace("Prepend user query: per-LLM default — set prepend_user_query: true/false in config.yaml "
              "or FLOXY_PREPEND_USER_QUERY=0/1 to override")
    elif override:
        trace(f"Prepend user query: forced ON for every LLM ({source}) — no prompt/KV cache reads across requests")
    else:
        trace(f"Prepend user query: forced OFF for every LLM ({source}) — caching preserved, accuracy may drop")


def _required_keys() -> tuple[str, ...]:
    """REQUIRED_KEYS plus the api_keys entry the `smart` tier's backend reads — nothing extra
    when that is a keyless local model, so an Ollama setup boots without any cloud key."""
    from assistant.llm import parse_llm_config, tier_api_key_name
    llm_class, _ = parse_llm_config('smart')
    key = tier_api_key_name('smart')
    if key is None:
        trace(f"Config check: smart tier runs on {llm_class}, no API key required")
        return REQUIRED_KEYS
    trace(f"Config check: smart tier runs on {llm_class}, requires {key}")
    return (key,) + REQUIRED_KEYS


def _verify_config_or_exit():
    """A config without the API key of the LLM the smart tier runs on, or without the vault
    passphrase, stops the start here, with the fix spelled out, instead of a traceback from
    the first LLM constructor."""
    missing = config.missing_required(_required_keys())
    if not missing:
        trace("Config check: required keys present")
        return
    names = " and ".join(filter(None, [", ".join(missing[:-1]), missing[-1]]))
    lines = [f"Please set {names} in {os.path.basename(config.config_path())} and restart."]
    if config.created_skeleton:
        lines.insert(0, f"A new {os.path.basename(config.config_path())} was written at {config.config_path()}")
    _refuse_to_start("server", lines)


def _refuse_to_start(what: str, lines: list[str]):
    bar = "=" * 70
    error("")
    error(bar)
    error(f"  CONFIG CHECK FAILED — {what} will not start")
    error(bar)
    for line in lines:
        error(f"  {line}")
    error(bar)
    error("")
    sys.exit(1)


def _serve_satellite():
    """`role: imessage_satellite`: no Assistant, no interface, no integration — the process
    serves this Mac user's Messages account to a Keep server and follows that server's version
    (common/imessage/Satellite.py). The port is `imessage_satellite.port`, not argv's: a second
    macOS user shares the port space of the server's own."""
    from common.imessage import Satellite
    problem = Satellite.config_problem()
    if problem:
        _refuse_to_start("satellite", [f"{problem}.", f"Set it in {config.config_path()} and start again."])
    asyncio.run(Satellite.build().serve())
    info("Clean shutdown complete")


def _launch():
    _enable_crash_traces()
    info(_TLS_TRUST)
    port, testsuite = _parse_args()
    RequirementsChecker.verify_or_exit()
    Migrations.run()
    role = Role.current()
    if role == Role.IMESSAGE_SATELLITE:
        _serve_satellite()
        return
    if role != Role.SERVER:
        _refuse_to_start("Keep", [f"{Role.KEY}: {role!r} in {config.config_path()} is not a role.",
                                  f"Remove the key for a Keep server, or set it to {Role.IMESSAGE_SATELLITE}."])
    _verify_config_or_exit()
    info("Loading libraries...")
    server = build(port, testsuite)
    atexit.register(server.stop)   # the fallback for a start that never reached main()'s finally
    asyncio.run(main(server))


if __name__ == "__main__":
    _launch()
