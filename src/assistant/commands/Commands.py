import inspect
import queue
import re
import threading
import time
import urllib.parse
from collections import deque
from typing import Callable, Optional

from assistant.integration.Accounts import render_accounts_status
from assistant.integration.mail.MailAccess import MailAccess
from assistant.integration.calendar.CalendarAccess import CalendarAccess
from assistant.integration import HomeAutomation as home
from assistant.tools.WebAccess import WebAccess, NETWORK_LOG_HEADER
from assistant.agents.AutonomousActionManager import AutonomousActionManager
from . import CommandSet, Toolkits, ToolkitPreloader
from assistant import Destinations, FirstRun, MemoryFile, Procedures, WatchList
from common.logging import trace, info, note, error, fine
from assistant.agents.CodeQualityMaintainer import CodeQualityMaintainer
from common.ConfigManager import ConfigManager
from common import ConfigSchema, ConfigStore
from common.ConfigStore import ConfigStoreError
from assistant.ChannelProfile import AUTONOMOUS
from assistant.Notifier import SecretRejected, notifier
from assistant.SystemNote import system_note
from assistant.Telemetry import Telemetry
from .CommandEcho import describe_completion
from assistant.integration import CameraProvider as cameras
from assistant.integration import VehicleProvider as vehicles
from common.FloxyDatabase import FloxyDatabase
from common.vault.Vault import Vault, VaultError, DuplicateSecret, KINDS
from common.vault.Redactor import redact_for_storage


def normalize_llm_escapes(text: str) -> str:
    """Turn the literal escape sequences the LLM emits (backslash + n/r/t) into
    real whitespace. Single source of truth so the approval preview and the
    delivered message agree — see _gate_sensitive_command."""
    return text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")


# noinspection PyMethodMayBeStatic
# Returned to the LLM when a photo/report had nowhere to go (NullNotifier installed).
NO_NOTIFICATION_CHANNEL = "failed: no notification channel configured (no chat interface running), nothing was delivered to the user."

# What server_logs() hands the LLM: a window of the main log (cwd-relative, the server runs from the root) —
# the last SERVER_LOG_TAIL_LINES lines, or the first that many from a given time.
SERVER_LOG_PATH = "log/floxy.out"
# Opens every server_logs result; is_reference_result knows the dumps by it.
SERVER_LOG_HEADER = "# Keep AI Server Logs"
SERVER_LOG_TAIL_LINES = 400


def is_reference_result(body: str) -> bool:
    """Whether a command result is reference text nobody learns anything from later: a toolkit
    or procedure reference, the note a preload puts in front of one, a server log dump or a
    browse network listing. The memory curator leaves them out of its transcript (over
    2026-10-07..09 they were 433KB of the 683KB of command results), and search_history /
    show_history leave them out of their hits: a toolkit loaded a dozen times a day matched
    every broad query and filled the 30 hits with its own text. What stays is what was read
    for the user — pages, mails, device status. A round is judged by its opening: a batch
    (`_publish_bg_results`) puts the reference results after the others, so a round that
    loaded a toolkit beside a real read opens with the read and is kept; the toolkit text
    itself has blank lines in it, so cutting the body into parts would not tell them apart."""
    return (body.startswith((Toolkits.HEADER_PREFIX, Procedures.HEADER_PREFIX, SERVER_LOG_HEADER, NETWORK_LOG_HEADER))
            or ToolkitPreloader.is_preface(body))


# A log line's timestamp; the ANSI colour codes around it never match this shape, so the raw line is searched.
_LOG_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
# What server_logs(from) accepts: a full timestamp (space or T), a date (midnight) or a clock time (today).
_LOG_TIME_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M",
                     "%Y-%m-%d", "%H:%M:%S", "%H:%M")


class Commands:
    """Runs every command the LLM can invoke, from lights to email to photos.

    The core commands are the `cmd_*` methods here; an integration's own commands are a
    CommandSet in its folder (`src/integrations/<folder>/<Name>Commands.py`), one instance of
    which is built per Commands. `_owners` maps a set's command names to it, and `run`,
    `describe` and the flags (sensitive, pure, multi-param, approval note) ask the command's
    owner — this object for a core command. The command name list handed to the parser is
    derived from the `cmd_` method names of all of them; SENSITIVE_COMMANDS go through the
    notifier's approval gate before running.
    """
    SENSITIVE_COMMANDS = frozenset({"send_email", "restart_server", "try_pull_request"})
    # What the approval prompt shows under the title when the command has no parameters
    # to preview — the user should know what saying yes does before the screen goes blank.
    APPROVAL_NOTES = {
        "restart_server": "Pulls the latest code from git (the release branch, or master when "
                          "developer_release is true; a PR trial ends here) and relaunches the server. "
                          "The console goes blank for a moment and reconnects by itself.",
        "try_pull_request": "Fetches the pull request's code, checks it out and relaunches the server on it "
                            "as a trial. restart_server (or the nightly update at 03:00) goes back to the "
                            "normal code. The console goes blank for a moment and reconnects by itself.",
    }
    # Commands whose handler accepts multiple positional arguments. The dispatcher
    # splats the parsed parameter list (`f(*params)`); the handler decides how to
    # interpret arity (e.g. the iMessage set's send_imessage merges trailing tokens into the body
    # so under-quoted LLM output like `send_imessage(John, Hi, friend)` still works).
    # Anything not in this set gets `f(params[0] if params else "")` for compat
    # with the historical single-string signature.
    MULTI_PARAM_COMMANDS = frozenset({
        "send_email", "dim", "search_history", "show_history",
        "browse_inspect", "browse_click_at", "browse_wait", "move_email", "latest_email", "vault_add",
        "vault_request", "config_set", "load_toolkit", "pr_feedback",
        "load_watch", "save_watch", "delete_watch",
        "load_procedure", "save_procedure", "delete_procedure", "post_message",
        "create_event", "delete_event", "move_event", "load_calendar", "fetch_camera_images", "car_navigate",
    })
    # Side-effect-free commands that the prompt test suite still runs: a toolkit the LLM
    # never receives would fail every expectation that depends on its reference.
    PURE_COMMANDS = frozenset({"load_toolkit", "load_procedure", "load_watch", "load_calendar"})
    AI_MESSAGE_SUFFIX = "[Sent with the help of AI 🤖]"
    # The footer is on unless config.yaml says ai_message_footer: false — read at each send.
    AI_FOOTER_KEY = "ai_message_footer"

    # Integrations are looked up per call: get_instance() is the cache, so a restarted
    # integration (restart_integration) is what the next command uses.
    @staticmethod
    def _get_web_access() -> WebAccess:
        return WebAccess.get_instance()

    @property
    def web_access(self) -> WebAccess:
        return self._get_web_access()

    @property
    def _email_provider(self) -> MailAccess:
        return self._mail_access()

    false_positive_timestamps = deque(maxlen=10)
    # Installed by the launcher when the voice interface is running
    # (interfaces.voice.microphones.FalsePositiveFlood.report_false_positive_flood);
    # None in a text-only process.
    false_positive_reporter: Callable[[], None] | None = None

    def with_ai_footer(self, body: str) -> str:
        """*body* with the AI disclosure footer, unless the user turned it off."""
        if ConfigManager.get_instance().get(self.AI_FOOTER_KEY, True) is False:
            trace(f"{self.AI_FOOTER_KEY} is off — the message goes out without the AI footer")
            return body
        return f"{body}\n\n{self.AI_MESSAGE_SUFFIX}"

    @staticmethod
    def _home_name(name: str) -> str:
        # The prompt shows every "(Room)" in a device or action name as "[Room]" and the LLM
        # copies that form back; the backend gets the raw name (HomeAutomation.from_display).
        return home.HomeAutomation.from_display(name.strip())

    def __init__(self, cp, command_sets=None):
        """`command_sets` are the CommandSet classes to host; the discovered ones when None."""
        self.command_processor = cp
        self._owners: dict[str, object] = self._host(CommandSet.discover() if command_sets is None else command_sets)
        # Set by CommandProcessor before each run: the policy of the channel that asked
        # and the RequestContext of the request (None for autonomous runs).
        self._current_profile = AUTONOMOUS
        self._current_context = None
        # The notifier's handle for the running command's chat echo; background_command
        # takes it along so the background worker can complete the echo when the work ends.
        self._current_echo = None
        # Background batch state — see the "background commands" section below.
        # bg_lock serializes submit / finish / publish so commands running on several
        # lane threads can't race the batch bookkeeping.
        self.bg_lock = threading.Lock()
        self._bg_outstanding = 0
        self._bg_seq = 0
        self._bg_results: dict[int, object] = {}
        self.bg_results_contain_error = False
        self._bg_batch_started = 0.0
        self._bg_batch_count = 0
        self._bg_busy_seconds = 0.0
        self._bg_idle = threading.Event()
        self._bg_idle.set()
        # One queue + at most one drain thread per serial lane (browser, config).
        self._serial_queues: dict[str, queue.Queue] = {}
        self._serial_workers: dict[str, threading.Thread] = {}

    def _host(self, command_sets) -> dict[str, object]:
        """One instance of each set, by the command names it owns. A name with two owners
        would make the dispatch depend on the walk order, so it is refused here."""
        core = set(CommandSet.command_names(type(self)))
        owners: dict[str, object] = {}
        for cls in command_sets:
            command_set = cls(self)
            for name in CommandSet.command_names(cls):
                if name in core or name in owners:
                    other = type(self) if name in core else type(owners[name])
                    raise ValueError(f"Command '{name}' has two owners: {other.__module__}.{other.__name__} "
                                     f"and {cls.__module__}.{cls.__name__}")
                owners[name] = command_set
        trace(f"Commands: hosting {len(owners)} command(s) from {len(command_sets)} integration command set(s)")
        return owners

    def _owner(self, command: str):
        """Who implements `command`: its CommandSet, or this object for a core (or unknown) one."""
        return self._owners.get(command, self)

    @classmethod
    def command_names(cls) -> list[str]:
        """Every command name, core and discovered sets, for a caller without an instance."""
        names = set(CommandSet.command_names(cls))
        for command_set in CommandSet.discover():
            names.update(CommandSet.command_names(command_set))
        return sorted(names)

    @staticmethod
    def _mail_access() -> MailAccess:
        """Every mail account of the enabled and configured integrations (their short-id
        caches live on the monitored singletons, so they survive between a listing and a
        read), gathered per call so a restarted or newly configured account is used."""
        return MailAccess.current()

    def cmd_home_automation_logs(self, parameter_string: str):
        self.background_command(self._fetch_home_automation_logs, None)

    @staticmethod
    def _fetch_home_automation_logs(_params) -> str:
        logs = home.active().error_logs()
        info(f"Home automation logs fetched: {len(logs)} chars")
        return f"# Home Automation Error Logs\n{logs}"

    def cmd_server_logs(self, parameter_string: str):
        self.background_command(self._fetch_server_logs, self._first_argument(parameter_string) or None)

    @staticmethod
    def _fetch_server_logs(from_time: Optional[str]) -> str:
        start = Commands._parse_log_time(from_time) if from_time else None
        if from_time and start is None:
            info(f"server_logs: cannot read the time '{from_time}'")
            return (f"server_logs failed: cannot read the time '{from_time}'; "
                    "use YYYY-MM-DD HH:MM[:SS], YYYY-MM-DD or HH:MM[:SS] (today).")
        try:
            # The log is redirected process output; a stray non-UTF-8 byte must not take the
            # assistant's own diagnostics down with it.
            with open(SERVER_LOG_PATH, "r", errors="replace") as f:
                lines = f.readlines()
        except FileNotFoundError:
            error(f"Server log {SERVER_LOG_PATH} not found")
            return f"{SERVER_LOG_HEADER}\nLog file not found."
        if start is None:
            chunk = lines[-SERVER_LOG_TAIL_LINES:]
            window = f"last {len(chunk)} lines"
        else:
            first = Commands._first_log_line_at(lines, start)
            if first is None:
                info(f"Server logs: nothing logged at or after {start} ({len(lines)} lines)")
                return f"{SERVER_LOG_HEADER}\nNothing logged at or after {start}; the log has {len(lines)} lines."
            chunk = lines[first:first + SERVER_LOG_TAIL_LINES]
            window = f"{len(chunk)} lines from {start} (line {first + 1})"
        text = re.sub(r'\x1b\[[0-9;]*m', '', "".join(chunk))
        info(f"Server logs fetched: {window} of {len(lines)}, {len(text)} chars")
        return f"{SERVER_LOG_HEADER} ({window})\n{text}"

    @staticmethod
    def _parse_log_time(text: str):
        from datetime import datetime
        for fmt in _LOG_TIME_FORMATS:
            try:
                parsed = datetime.strptime(text, fmt)
            except ValueError:
                continue
            if "%Y" not in fmt:
                parsed = datetime.combine(datetime.now().date(), parsed.time())
            return parsed
        return None

    @staticmethod
    def _first_log_line_at(lines: list, start) -> Optional[int]:
        """Index of the first line stamped at or after `start`; continuation lines (tracebacks) carry no stamp and are skipped."""
        key = start.strftime("%Y-%m-%d %H:%M:%S")
        for index, line in enumerate(lines):
            match = _LOG_TIMESTAMP.search(line)
            if match and match.group() >= key:
                return index
        return None

    def cmd_open_url(self, parameter_string: str):
        self.background_command(self.web_access.fetch_webpage, parameter_string, lane=self.BROWSER_LANE)

    def cmd_google(self, parameter_string: str):
        self.background_command(self.web_access.search, parameter_string)

    def cmd_browse_open(self, parameter_string: str):
        self.background_command(self.web_access.browse_open, parameter_string, lane=self.BROWSER_LANE)

    def cmd_browse_act(self, parameter_string: str):
        self.background_command(self.web_access.browse_act, parameter_string, lane=self.BROWSER_LANE)

    def cmd_browse_key(self, parameter_string: str):
        self.background_command(self.web_access.browse_key, parameter_string, lane=self.BROWSER_LANE)

    def cmd_browse_eval(self, parameter_string: str):
        self.background_command(self.web_access.browse_eval, parameter_string, lane=self.BROWSER_LANE)

    def cmd_browse_inspect(self, *args: str):
        # Accepts either a single quoted spec ("3,href") or two positional args
        # (3, href). Re-join with comma so the underlying parser sees a single
        # spec string in the format it expects.
        spec = ",".join(a.strip() for a in args)
        self.background_command(self.web_access.browse_inspect, spec, lane=self.BROWSER_LANE)

    def cmd_browse_snapshot(self, parameter_string: str):
        self.background_command(self.web_access.browse_snapshot, parameter_string, lane=self.BROWSER_LANE)

    def cmd_browse_click_at(self, *args: str):
        # "412, 583" arrives as two positional args — re-join so WebAccess parses one spec.
        spec = ",".join(a.strip() for a in args)
        self.background_command(self.web_access.browse_click_at, spec, lane=self.BROWSER_LANE)

    def cmd_browse_wait(self, *args: str):
        # `browse_wait("Continue|callback", 45)` arrives as two positional args — re-join so
        # WebAccess parses one "<pattern>, <seconds>" spec (a comma inside the pattern survives).
        spec = ", ".join(a.strip() for a in args)
        self.background_command(self.web_access.browse_wait, spec, lane=self.BROWSER_LANE)

    def cmd_browse_network(self, parameter_string: str):
        self.background_command(self.web_access.browse_network, parameter_string, lane=self.BROWSER_LANE)

    def cmd_browse_extract(self, parameter_string: str):
        self.background_command(self.web_access.browse_extract, parameter_string, lane=self.BROWSER_LANE)

    def cmd_browse_screenshot(self, parameter_string: str):
        self.background_command(self.web_access.browse_screenshot, parameter_string, lane=self.BROWSER_LANE)

    def cmd_send_screenshot(self, parameter_string: str):
        route = self._request_route()
        self.background_command(lambda caption: self._run_send_screenshot(caption, route=route),
                                parameter_string, lane=self.BROWSER_LANE)

    def _request_route(self) -> Optional[dict]:
        """The chat a request-driven send answers to, as the notifier's source/conversation
        kwargs; None when no request is being processed (the autonomous check), which the
        notifier takes as a proactive push. Captured on the command worker: by the time a
        background command runs, another request may have replaced _current_context."""
        ctx = self._current_context
        source = getattr(ctx, "source", None)
        if source is None:
            return None
        return {"source": source, "conversation": getattr(ctx, "conversation", None)}

    def _run_send_screenshot(self, caption: str, route: Optional[dict] = None) -> str:
        if not self._current_profile.screenshot_allowed:
            return "send_screenshot failed: not available in voice mode (user has no screen to view it on)."

        png_bytes, path = self.web_access.get_last_screenshot()
        if not png_bytes:
            return "send_screenshot failed: no screenshot available. Call browse_screenshot() first."

        from assistant.tools.ImageAnalyzer import resize_to_jpeg
        jpeg_bytes = resize_to_jpeg(png_bytes, max_dim=2048)
        if jpeg_bytes is None:
            note("send_screenshot: resize failed, sending original PNG")
            jpeg_bytes = png_bytes

        caption = caption.strip()
        if not notifier().send_photo(jpeg_bytes, caption=caption, **(route or {})):
            return NO_NOTIFICATION_CHANNEL
        info(f"Sent screenshot ({path}, {len(jpeg_bytes)} bytes) to notifier with caption: {caption!r}")
        return ""

    def cmd_browse_close(self, parameter_string: str):
        self.fire_and_forget(self.web_access.browse_close, parameter_string)

    _UNREAD_TOKENS = frozenset({"unread", "true", "1", "yes", "unread_only"})

    def cmd_latest_email(self, folder: str = "", unread_only: str = ""):
        unread = str(unread_only).strip().strip('"').strip("'").lower() in self._UNREAD_TOKENS
        self.background_command(self._email_provider.fetch_latest, [folder, unread])

    def cmd_search_email(self, parameter_string: str):
        self.background_command(self._email_provider.search, parameter_string)

    def cmd_read_email(self, parameter_string: str):
        self.background_command(self._email_provider.read, parameter_string)

    def cmd_move_email(self, email_id: str, dest_folder: str):
        self.background_command(self._email_provider.move, [email_id, dest_folder])

    def cmd_list_folders(self, parameter_string: str):
        self.background_command(self._email_provider.list_folders, parameter_string)

    def cmd_create_event(self, *args: str):
        self.background_command(self._run_create_event, list(args))

    _TIME_RE = re.compile(r"^\d{1,2}:\d{2}$")

    @classmethod
    def _run_create_event(cls, *args: str) -> str:
        try:
            if len(args) < 4:
                return "create_event failed: expected 4 parameters (title, date, start_time, end_time)"
            args = [a.strip() for a in args]
            # A fifth argument names the calendar; the end time is the last argument otherwise.
            calendar_name = ""
            if len(args) > 4 and not cls._TIME_RE.match(args[-1]) and cls._TIME_RE.match(args[-2]):
                calendar_name = args.pop().strip("'\"")
            # If the LLM under-quoted and produced extra tokens, the title may have
            # been split at internal commas — re-join leading tokens into the title
            # and pull the last 3 tokens as date/start/end.
            title = ", ".join(args[:-3])
            date_str, start_time, end_time = args[-3:]
            CalendarAccess.current().create(title, date_str, start_time, end_time, calendar_name)
            return ""
        except Exception as e:
            return f"create_event failed: {e}"

    _DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    @classmethod
    def _split_title(cls, args: list[str], command: str) -> tuple[str, list[str]]:
        """The title and what follows it: the title is everything before the first date
        token, rejoined when the LLM under-quoted a title with a comma in it."""
        first_date = next((i for i, a in enumerate(args) if cls._DATE_RE.match(a)), None)
        if not first_date:
            raise ValueError(f"{command} failed: expected a title followed by its date (YYYY-MM-DD)")
        return ", ".join(args[:first_date]), args[first_date:]

    def cmd_delete_event(self, *args: str):
        self.background_command(self._run_delete_event, list(args))

    @classmethod
    def _run_delete_event(cls, *args: str) -> str:
        """`delete_event(title, date, calendar = "")`."""
        try:
            args = [a.strip() for a in args if a.strip()]
            title, rest = cls._split_title(args, "delete_event")
            if len(rest) > 2:
                return "delete_event failed: expected title, date and at most a calendar name"
            calendar_name = rest[1].strip("'\"") if len(rest) > 1 else ""
            return CalendarAccess.current().delete(title, rest[0], calendar_name)
        except Exception as e:
            return f"delete_event failed: {e}"

    def cmd_move_event(self, *args: str):
        self.background_command(self._run_move_event, list(args))

    @classmethod
    def _run_move_event(cls, *args: str) -> str:
        """`move_event(title, date, new_date, start_time = "", end_time = "", calendar = "")`:
        the two dates follow the title, then both times or neither, then the calendar."""
        try:
            args = [a.strip() for a in args if a.strip()]
            title, rest = cls._split_title(args, "move_event")
            if len(rest) < 2 or not cls._DATE_RE.match(rest[1]):
                return "move_event failed: expected title, date and new date (YYYY-MM-DD), then optionally start and end time"
            date_str, new_date, rest = rest[0], rest[1], rest[2:]
            times = [t for t in rest[:2] if cls._TIME_RE.match(t)]
            if len(times) == 1:
                return "move_event failed: give both the start and the end time, or neither to keep the event's own"
            rest = rest[len(times):]
            if len(rest) > 1:
                return "move_event failed: expected at most a calendar name after the times"
            calendar_name = rest[0].strip("'\"") if rest else ""
            start_time, end_time = times if times else ("", "")
            return CalendarAccess.current().move(title, date_str, new_date, start_time, end_time, calendar_name)
        except Exception as e:
            return f"move_event failed: {e}"

    def cmd_load_calendar(self, *names: str):
        self.background_command(self._run_load_calendar, list(names))

    @staticmethod
    def _run_load_calendar(*names: str) -> str:
        """The named calendars over the window the prompt's calendar section covers."""
        names = [n for n in names if n and n.strip()]
        if not names:
            return "load_calendar failed: expected a calendar's name"
        try:
            return CalendarAccess.current().load(names)
        except Exception as e:
            return f"load_calendar failed: {e}"

    def cmd_schedule_wakeup(self, parameter_string: str):
        # Just AFTER the wanted time, otherwise the AI may see it as not yet due.
        AutonomousActionManager.get_instance().schedule(float(parameter_string) + 0.1)

    def cmd_caffeinate(self, parameter_string: str):
        AutonomousActionManager.get_instance().caffeinate()

    @staticmethod
    def _media_tag(photo: dict) -> str:
        """The kind tag shown next to a search hit: screenshots and videos (with their length)."""
        photo_type = photo.get("photo_type") or ""
        if "screenshot" in photo_type:
            return " (screenshot)"
        if photo_type.startswith("video"):
            duration = photo.get("duration_seconds")
            return f" ({photo_type}, {duration:.0f}s)" if duration else f" ({photo_type})"
        return ""

    def cmd_search_history(self, *args: str):
        self.background_command(self._run_search_history, list(args))

    @staticmethod
    def _run_search_history(*args: str) -> str:
        from datetime import datetime
        query = args[0].replace("str = ", "").replace("str=", "").strip() if args else ""
        sort = "relevance"
        if len(args) > 1:
            sort = args[1].replace("str = ", "").replace("str=", "").strip().lower()
            if sort not in FloxyDatabase.HISTORY_SORT_OPTIONS:
                info(f"Invalid sort '{sort}', falling back to relevance")
                sort = "relevance"
        if not query:
            return "No query provided for history search."

        db = FloxyDatabase.get_instance()

        entries = []
        seen_content = set()

        for r in db.search_history(query, sort=sort, limit=30, skip_command_result=is_reference_result):
            content = r["content"]
            if content in seen_content:
                continue
            seen_content.add(content)
            if len(content) > 600:
                content = content[:600] + "..."
            source_tag = f" ({r['source']})" if r.get("source") else ""
            location_tag = f" [{r['location']}]" if r.get("location") else ""
            date_str = datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d %H:%M")
            entries.append((
                r["timestamp"],
                r.get("fts_relevance", 0),
                f"[{date_str}]{location_tag} {r['role']}{source_tag}: {content}"
            ))

        photo_count = 0
        for r in db.search_photos(query, limit=5, sort_by="relevance"):
            photo_count += 1
            ts = r.get("creation_date", 0)
            date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else ""
            location = r.get("location_text") or "Unknown location"
            pretty = r.get("pretty_score", 0)
            family = r.get("family_importance_score", 0)
            persons_tag = f' [{r["persons"]}]' if r.get("persons") else ""
            entries.append((
                ts,
                r.get("fts_relevance", 0),
                f"[{date_str}] photo#{r['id']}: {r['short_description']}{Commands._media_tag(r)} ({location}) [pretty: {pretty}, family: {family}]{persons_tag}"
            ))

        if not entries:
            return f'No history entries or photos found matching "{query}".'

        if sort == "oldest":
            entries.sort(key=lambda e: e[0])
        elif sort == "newest":
            entries.sort(key=lambda e: e[0], reverse=True)
        else:
            entries.sort(key=lambda e: e[1], reverse=True)

        info(f"History search for '{query}' returned {len(entries) - photo_count} history + {photo_count} photo results")
        return "\n".join(e[2] for e in entries)

    def cmd_show_history(self, *args: str):
        self.background_command(self._run_show_history, list(args))

    @staticmethod
    def _run_show_history(*args: str) -> str:
        from datetime import datetime
        if len(args) < 2:
            return "show_history failed: expected two parameters (from, to) in YYYY-MM-DD HH:MM format."
        from_str = args[0].strip()
        to_str = args[1].strip()

        try:
            from_dt = datetime.strptime(from_str, "%Y-%m-%d %H:%M")
            to_dt = datetime.strptime(to_str, "%Y-%m-%d %H:%M")
        except ValueError:
            return "show_history failed: timestamps must be in YYYY-MM-DD HH:MM format."

        from_ts = from_dt.timestamp()
        to_ts = to_dt.timestamp()

        db = FloxyDatabase.get_instance()
        results = db.get_history_range(from_ts, to_ts, skip_command_result=is_reference_result)

        if not results:
            return f'No history entries found between {from_str} and {to_str}.'

        lines = []
        seen_content = set()
        for r in results:
            content = r["content"]
            if content in seen_content:
                continue
            seen_content.add(content)
            if len(content) > 600:
                content = content[:600] + "..."
            source_tag = f" ({r['source']})" if r.get("source") else ""
            location_tag = f" [{r['location']}]" if r.get("location") else ""
            date_str = datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d %H:%M")
            lines.append(f"[{date_str}]{location_tag} {r['role']}{source_tag}: {content}")

        info(f"show_history({from_str} -> {to_str}) returned {len(lines)} entries")
        return "\n".join(lines)

    def cmd_send_email(self, *args: str) -> str:
        if len(args) < 3:
            return "send_email failed: expected 'to, subject, body' parameters"
        access = self._email_provider
        args = list(args)
        account_name = ""
        # A fourth argument is the sending account — but only when it names one: anything else
        # is the tail of an under-quoted body, which the re-join below puts back.
        if len(args) > 3 and access.named(args[-1]) is not None:
            account_name = args.pop().strip().strip("'\"")
            trace(f"send_email: sending from the named account {account_name!r}")
        to = args[0].strip()
        subject = args[1].strip()
        # Re-join trailing tokens so an under-quoted body with commas survives.
        body = ", ".join(a for a in args[2:]).strip()
        if not to or not subject or not body:
            return "send_email failed: expected 'to, subject, body' parameters"

        # LLM emits literal "\n" escape sequences — turn them into real newlines.
        body = normalize_llm_escapes(body)
        return access.send(to, subject, body, account_name)

    def cmd_fetch_camera_images(self, *names: str):
        """Stills from every camera of every enabled camera integration, or from the cameras
        the arguments name (substring match)."""
        only = [n.strip() for n in names if n and n.strip()] or None
        # background_command splats a list into positional arguments: the one list goes in a tuple
        self.background_command(self._fetch_all_camera_images, (only,))

    @staticmethod
    def _provider_snapshots(provider, only) -> list:
        try:
            blocks = provider.fetch_snapshots(only)
            info(f"Cameras: {provider.name} returned {sum(1 for b in blocks if b.get('type') == 'image')} image(s)")
            return blocks
        except Exception as e:
            error(f"Cameras: {provider.name} snapshot fetch failed: {e}")
            return []

    @staticmethod
    def _fetch_all_camera_images(only) -> object:
        from concurrent.futures import ThreadPoolExecutor

        providers = cameras.active()
        if not providers:
            return cameras.no_provider_hint()
        with ThreadPoolExecutor(max_workers=len(providers)) as pool:
            futures = [pool.submit(Commands._provider_snapshots, p, only) for p in providers]
            results = [block for f in futures for block in f.result()]

        if not results:
            if only:
                known = [name for p in providers for name in p.camera_names()]
                info(f"Cameras: nothing matched {only}; the cameras are {known}")
                return (f"No camera matches {', '.join(only)}. The cameras: {', '.join(known)}." if known
                        else f"No camera matches {', '.join(only)}, and no camera is reachable right now.")
            return "No camera images available."

        results.insert(0, {"type": "text", "text": "Here are the current camera images from around the house:"})
        return results

    # --- The cars: every VehicleProvider's, resolved per call through the fleet ---

    @staticmethod
    def _car_label(vehicle_id: str) -> str:
        return vehicle_id.strip() or "the car"

    @staticmethod
    def _run_car_action(vehicle_id: str, verb: str, *args, silent: bool = False) -> str:
        """Resolve the car and run one provider action on it. The text for the LLM: the
        provider's line (nothing when `silent` — the LLM's own reply stands), or why it
        failed: no provider ready, an id that names no car, a brand that refuses."""
        fleet = vehicles.fleet()
        if not fleet.providers:
            return vehicles.no_provider_hint()
        try:
            provider, spec = fleet.resolve(vehicle_id)
            info(f"Cars: {verb} on {spec.label} through {provider.name}")
            result = getattr(provider, verb)(spec, *args)
            info(f"Cars: {provider.name} — {result}")
            return "" if silent else result
        except vehicles.VehicleError as e:
            info(f"Cars: {verb} failed — {e}")
            return f"Car command failed: {e}"

    def _car_command(self, vehicle_id: str, verb: str):
        self.background_command(lambda p: self._run_car_action(p, verb, silent=True), vehicle_id)

    def cmd_car_max_defrost(self, parameter_string: str):
        self._car_command(parameter_string, "max_defrost")

    def cmd_car_climate_on(self, parameter_string: str):
        self._car_command(parameter_string, "climate_on")

    def cmd_car_climate_off(self, parameter_string: str):
        self._car_command(parameter_string, "climate_off")

    def cmd_car_open_trunk(self, parameter_string: str):
        self._car_command(parameter_string, "open_trunk")

    @staticmethod
    def _navigation_args(args: tuple) -> tuple[str, str]:
        """(vehicle_id, destination) from the LLM's arguments: the vehicle leads and the rest
        is the destination, re-joined so an unquoted address keeps its commas; a lone argument
        is a destination for the default car."""
        parts = [a.strip() for a in args if a and a.strip()]
        if len(parts) == 1:
            return "", parts[0]
        return (parts[0] if parts else ""), ", ".join(parts[1:])

    def cmd_car_navigate(self, *args: str):
        vehicle_id, destination = self._navigation_args(args)
        if not destination:
            return "car_navigate failed: expected 'vehicle_id, destination' parameters"
        self.background_command(lambda v, d: self._run_car_action(v, "navigate_to", d), (vehicle_id, destination))

    def cmd_self_analysis(self, parameter_string: str):
        reason = CodeQualityMaintainer.unavailable_reason()
        if reason:
            info(f"self_analysis not started: {reason}")
            return reason

        def _run_and_deliver(params):
            try:
                report = CodeQualityMaintainer.analyze_code(params)
                trace("Received self-analysis report from Code Quality Maintainer")
                fine(f"Report: {report}")
                # The chats get the report in their dialect, the console included; the LLM
                # gets the markdown whole as a `command` row — never cut at render time, unlike
                # an assistant row, which keeps 2000 characters of a report the toolkit tells
                # the LLM to feed into a fix or an issue — and the console's Home tab, which
                # hides command rows, draws the pushed copy instead.
                sent = notifier().send_message(f"Self-analysis report:\n\n{notifier().markdown_to_html(report)}")
                aam = AutonomousActionManager.get_instance()
                prompt_manager = aam.prompt_manager if aam else None
                # "[Self-analysis complete]" is the tag the LLM keys off; keep it first.
                delivery = "" if sent else f" ({NO_NOTIFICATION_CHANNEL})"
                if prompt_manager is not None:
                    prompt_manager.add_command_result(f"[Self-analysis complete]{delivery}\n{report}")
                else:
                    info("Self-analysis: no AutonomousActionManager yet, the report is not in the history")
            except Exception as e:
                error(f"Self-analysis failed: {e}")

        self.fire_and_forget(_run_and_deliver, parameter_string)

    def cmd_self_improvement(self, parameter_string: str):
        reason = CodeQualityMaintainer.unavailable_reason()
        if reason:
            info(f"self_improvement not started: {reason}")
            return reason
        self._run_pr_job("Self-improvement", lambda: CodeQualityMaintainer.improve_code(parameter_string))

    def cmd_improve_integration(self, parameter_string: str):
        """`improve_integration(name, request)`: a PR confined to one integration — a fix to
        the named one, or a new one when no integration has that name. The parser hands the
        payload over verbatim (a free-text command), so the name is split off here."""
        reason = CodeQualityMaintainer.unavailable_reason()
        if reason:
            info(f"improve_integration not started: {reason}")
            return reason
        name, _, request = normalize_llm_escapes(parameter_string or "").partition(",")
        name, request = name.strip().strip("\"'`"), request.strip().strip("\"'`")
        if not name or not request:
            return "improve_integration failed: expected 'integration name, request' parameters"
        folder = self._integration_folder(name)
        if folder is None:
            info(f"improve_integration: no integration named '{name}' — a new one")
        else:
            info(f"improve_integration: '{name}' is the {folder} folder")
        self._run_pr_job("Integration improvement",
                         lambda: CodeQualityMaintainer.improve_integration(name, folder, request))

    def _integration_folder(self, name: str) -> Optional[str]:
        """The integration folder behind *name* (`src/integrations/<folder>/`), enabled or
        not; None when no registered integration has that name."""
        monitor = self._integration_monitor()
        return monitor.folder_named(name) if monitor else None

    def _run_pr_job(self, label: str, job: Callable[[], str]):
        """Run a Claude Code job that ends in a pull request in the background and deliver its
        report: pushed to the chats, and summed up in the history (`_pr_outcome`)."""
        def _run_and_deliver(_params):
            try:
                report = job()
                trace(f"Received {label.lower()} report from Code Quality Maintainer")
                fine(f"Report: {report}")
                html_report = notifier().markdown_to_html(report + self._trial_invitation(report))
                sent = notifier().send_message(f"{label} report:\n\n{html_report}")
                aam = AutonomousActionManager.get_instance()
                if aam and aam.prompt_manager:
                    aam.prompt_manager.update_history(None, None, self._pr_outcome(label, report, sent))
                else:
                    info(f"{label}: no AutonomousActionManager yet, the outcome is not in the history")
            except Exception as e:
                error(f"{label} failed: {e}")

        self.fire_and_forget(_run_and_deliver, None)

    @staticmethod
    def _trial_invitation(report: str) -> str:
        """The paragraph under a report that opened a PR: the message is pushed to the chat as
        is, no LLM speaks after a job, so the user is invited here to run the PR as a trial and
        send their verdict — a tried PR with the tester's feedback is what the developers merge
        first. "" for a report without a PR."""
        number, _ = CodeQualityMaintainer.pull_request_in(report)
        if number is None:
            return ""
        return (f"\n\n**Try it:** say \"try PR #{number}\" and the server runs it as a trial. Then tell me whether it "
                f"works — I post your verdict on the PR. A PR that was tried and has your feedback is much more "
                f"likely to be merged, and fast.")

    @staticmethod
    def _pr_outcome(label: str, report: str, sent: bool) -> str:
        """The history row a finished PR job leaves — the PR it opened (number, title line,
        URL) or that none was, so the LLM knows the outcome without the report: the "report
        sent" marker it left before told it nothing, and on 2026-10-11 it took a job for
        still running two hours after the PR was open. Without a chat to deliver to, the
        row carries the whole report. "[<label> complete]" is the tag the LLM keys off;
        keep it first."""
        number, url = CodeQualityMaintainer.pull_request_in(report)
        title = CodeQualityMaintainer.first_line(report)
        delivery = "" if sent else f" ({NO_NOTIFICATION_CHANNEL})"
        if number is not None:
            text = f"[{label} complete]{delivery} PR #{number} open: {title}"
            if url:
                text += f"\n{url}"
            text += (f"\nThe user was invited to run it with try_pull_request({number}) and to post their verdict "
                     f"with pr_feedback({number}, text).")
        else:
            text = f"[{label} complete]{delivery} no PR was created: {title}"
        if not sent:
            text += f"\n\nReport:\n{report}"
        return text

    def cmd_report_issue(self, parameter_string: str):
        """`report_issue(report)`: Claude Code investigates and files a GitHub issue on the
        repository (or comments on an open one about the same thing); the user gets the link."""
        reason = CodeQualityMaintainer.unavailable_reason()
        if reason:
            info(f"report_issue not started: {reason}")
            return reason

        def _run_and_deliver(params):
            try:
                report = CodeQualityMaintainer.report_issue(params)
                trace("Received the issue report from Code Quality Maintainer")
                fine(f"Report: {report}")
                number, url = CodeQualityMaintainer.issue_in(report)
                if url:
                    message = f"Your issue is now live here: {url}"
                else:
                    message = f"Issue report:\n\n{notifier().markdown_to_html(report)}"
                sent = notifier().send_message(message)
                aam = AutonomousActionManager.get_instance()
                if aam and aam.prompt_manager:
                    aam.prompt_manager.update_history(None, None, self._issue_outcome(report, number, url, sent))
                else:
                    info("report_issue: no AutonomousActionManager yet, the outcome is not in the history")
            except Exception as e:
                error(f"report_issue failed: {e}")

        self.fire_and_forget(_run_and_deliver, parameter_string)

    @staticmethod
    def _issue_outcome(report: str, number, url, sent: bool) -> str:
        """The history row a finished report_issue job leaves: the issue's number and link
        (the user was sent the link), or that none was filed and why — the report's title
        line, and the whole report when no chat could show it. "[Issue reported]" /
        "[Issue report complete]" is the tag the LLM keys off; keep it first."""
        delivery = "" if sent else f" ({NO_NOTIFICATION_CHANNEL})"
        if number is not None:
            text = f"[Issue reported]{delivery} issue #{number} is live: {CodeQualityMaintainer.first_line(report)}"
            if url:
                text += f"\n{url}"
        else:
            text = f"[Issue report complete]{delivery} no issue was filed: {CodeQualityMaintainer.first_line(report)}"
        if not sent:
            text += f"\n\nReport:\n{report}"
        return text

    def cmd_restart_server(self, parameter_string: str):
        def _run(_params):
            try:
                result = CodeQualityMaintainer.pull_and_restart()
                info(f"Restart server result: {result}")
                system_note("server", f"Server restart: {result}")
            except Exception as e:
                error(f"Restart server failed: {e}")

        self.fire_and_forget(_run, parameter_string)

    def cmd_try_pull_request(self, parameter_string: str):
        """`try_pull_request(number)`: run one of the install's own open PRs as a trial. The
        vetting answers at once (a refusal is the command result); the fetch, the pip sync and
        the restart follow in the background and report through a system note, as a restart does."""
        from assistant.agents.PullRequestTrial import PullRequestTrial, TrialRefused, parse_number
        number = parse_number(parameter_string)
        if number is None:
            return "try_pull_request failed: expected the pull request's number"
        reason = CodeQualityMaintainer.unavailable_reason()
        if reason:
            info(f"try_pull_request not started: {reason}")
            return reason
        try:
            pr = PullRequestTrial.vet(number)
        except TrialRefused as e:
            info(f"try_pull_request: PR #{number} refused — {e}")
            return f"Cannot try PR #{number}: {e}."

        def _run(_params):
            try:
                result = PullRequestTrial.start(number)
                info(f"try_pull_request result: {result}")
                system_note("server", f"PR trial: {result}")
            except Exception as e:
                error(f"try_pull_request failed: {e}")
                system_note("server", f"PR trial of #{number} failed: {e}")

        self.fire_and_forget(_run, parameter_string)
        return f"PR #{number} ({pr.get('title', '')}) checked out for a trial, the server is switching to it and restarting."

    def cmd_pr_feedback(self, *args: str) -> str:
        """`pr_feedback(number, text)`: post the user's verdict on a PR as a comment. Without a
        number the tried PR is meant; the text may carry commas, so the rest of the arguments
        are the text."""
        from assistant.agents.PullRequestTrial import PullRequestTrial, parse_number
        args = [a for a in args if a is not None]
        number = parse_number(args[0]) if args else None
        text = ", ".join(a.strip() for a in (args[1:] if number is not None else args))
        if number is None:
            trial = PullRequestTrial.current()
            if trial is None:
                return "pr_feedback failed: expected 'number, text' — no PR is being tried, so the number is needed"
            number = trial.number
        if not text.strip():
            return "pr_feedback failed: nothing to post"
        self.background_command(PullRequestTrial.feedback, (number, text))

    def cmd_execute(self, parameter_string: str) -> str:
        return home.active().execute(self._home_name(parameter_string))

    def cmd_vault_add(self, *args: str) -> str:
        """Files a value the user has already said out loud into the vault, so every later
        request (and the history that already holds it) is redacted from here on."""
        if len(args) < 3:
            return "vault_add failed: expected 'kind, description, value' parameters"
        kind, description = args[0].strip().lower(), args[1].strip()
        # An unquoted value with commas in it arrives split — the rest of the args are the value.
        value = ", ".join(a.strip() for a in args[2:])
        if kind not in KINDS:
            return f"vault_add failed: unknown kind '{kind}', expected one of {', '.join(KINDS)}"
        if not description or not value:
            return "vault_add failed: description and value must not be empty"
        vault = Vault.get_instance()
        if not vault.enabled:
            error(f"vault_add: vault disabled — {vault.error}")
            return f"vault_add failed: the vault is disabled ({vault.error})"
        try:
            secret = vault.add(kind, description, value)
        except DuplicateSecret as e:
            # The value is in the vault, which is what was asked — not a failure. The id is
            # in the prompt's vault section, so the LLM needs nothing back here.
            info(f"vault_add: value already vaulted as '{e.existing_id}' — nothing new stored")
            return ""
        except VaultError as e:
            return f"vault_add failed: {e}"
        info(f"vault_add: stored '{secret.id}' ({kind}) — redacted from the next request on")
        return ""

    def cmd_vault_request(self, *args: str) -> str:
        """Asks the user for a value through a masked prompt in the chat UIs and files it in
        the vault; the LLM only ever gets the placeholder back. The wait runs as a background
        command so its outcome reaches the LLM as a plain reply, not an error report."""
        if len(args) < 2:
            return "vault_request failed: expected 'kind, description' parameters"
        kind = args[0].strip().lower()
        # An unquoted description with commas in it arrives split — the rest of the args are the description.
        description = ", ".join(a.strip() for a in args[1:])
        if kind not in KINDS:
            return f"vault_request failed: unknown kind '{kind}', expected one of {', '.join(KINDS)}"
        if not description:
            return "vault_request failed: description must not be empty"
        vault = Vault.get_instance()
        if not vault.enabled:
            error(f"vault_request: vault disabled — {vault.error}")
            return f"vault_request failed: the vault is disabled ({vault.error})"
        # Captured now, on the command worker: by the time the bg worker runs, another
        # request may have replaced _current_context. The prompt opens only in this chat.
        ctx = self._current_context
        source = getattr(ctx, "source", None)
        conversation = getattr(ctx, "conversation", None)
        self.background_command(self._run_vault_request, (kind, description, source, conversation))

    def _run_vault_request(self, kind: str, description: str, source=None, conversation=None) -> str:
        """Runs on the bg command worker: blocks in the notifier until the user answered the
        prompt (or cancelled / timed out). The value only ever exists inside `accept`.
        *source*/*conversation* are the asking request's, so the prompt opens in that chat."""
        vault = Vault.get_instance()
        outcome: dict = {}

        def accept(value: str) -> str:
            try:
                secret = vault.add(kind, description, value)
            except DuplicateSecret as e:
                outcome["duplicate"] = e.existing_id
                info(f"vault_request: value already vaulted as '{e.existing_id}'")
                return f"Already in the vault as {e.existing_id}"
            except VaultError as e:
                outcome["reason"] = str(e)
                raise SecretRejected(str(e))
            outcome["secret"] = secret
            info(f"vault_request: stored '{secret.id}' ({kind}) — redacted from the next request on")
            return f"Stored as {secret.id} ({secret.masked})"

        decision = notifier().request_secret(title=f"{description} ({kind})", details="", accept=accept,
                                             source=source, conversation=conversation)
        if decision == "accepted":
            if "duplicate" in outcome:
                return (f"That value is already in the vault as {{{{secret:{outcome['duplicate']}}}}} — "
                        f"nothing new was stored.")
            secret = outcome["secret"]
            return (f"Stored as {{{{secret:{secret.id}}}}} ({kind}: {description}). Use that placeholder from "
                    f"now on; the value itself was never shown to you.")
        info(f"vault_request: {decision} for '{description}' ({kind}) — nothing stored")
        messages = {
            "cancel": "The user cancelled the prompt — nothing was stored.",
            "timeout": "The user did not answer the prompt in time — nothing was stored.",
            "rejected": f"The value was refused ({outcome.get('reason', 'it could not be stored')}) — "
                        "nothing was stored.",
            "unavailable": "No chat interface can show the prompt right now — ask the user to add the value "
                           "in the Vault tab of the console instead.",
        }
        return messages.get(decision, f"vault_request ended with '{decision}' — nothing was stored.")

    # --- self-setup: settings and integrations (plans/SELF_SETUP_PLAN.md) --------------------
    # The LLM never sees config.yaml: it gets key names, set / not set, and the example's text.
    # All four run in the background so their text reaches the LLM as a plain reply.

    _STATE_LABELS = {"ok": "ok", "warning": "warning", "error": "error", "not_configured": "not configured",
                     "disabled": "disabled", "needs_2fa": "needs 2FA", "rate_limited": "rate limited",
                     "unknown": "not checked yet"}
    @staticmethod
    def _setup_activity():
        """A command that changes the setup ran — every command on the CONFIG_LANE (config_set,
        config_request, enable / disable / restart_integration, the sign-ins and pairings of the
        integration folders), see background_command: on a young install setup is in progress
        for FTUE_SETUP_GRACE_HOURS more (FirstRun.in_ftue) — the setup toolkit stays preloaded
        and the INTEGRATIONS section stays in the prompt. A read-only command (config_help,
        list_integrations) is not setup activity: the assistant reading the schema a week in
        kept the ~10 KB setup reference in every request for two more days."""
        FirstRun.note_setup_activity()

    # Integrations that are not Integration classes but follow the same switch, with their keys.
    @staticmethod
    def _integration_monitor():
        from assistant.integration.IntegrationMonitor import IntegrationMonitor
        return IntegrationMonitor.get_instance()

    # What a command set in an integration folder (telegram_login, show_photo, search_photos)
    # uses of the core's helpers; it reaches them through its host.
    def integration_monitor(self):
        return self._integration_monitor()

    def state_label(self, state, default: str) -> str:
        return self._STATE_LABELS.get(state, default)

    def describe_restart(self, status: dict, restartable: bool) -> str:
        return self._describe_restart(status, restartable)

    def first_argument(self, p: str) -> str:
        return self._first_argument(p)

    def request_route(self) -> Optional[dict]:
        return self._request_route()

    def asked_from_chat(self) -> bool:
        """Whether a chat asked for the running command — its reply shows there, so a text
        meant for the user's eyes (a sign-in link) needs no note pushed to the chats. False
        for voice, the HTTP clients and an autonomous run. Captured on the command worker,
        like `request_route`."""
        return (self._request_route() or {}).get("conversation") is not None

    def media_tag(self, photo: dict) -> str:
        return self._media_tag(photo)

    @staticmethod
    def _keys_summary(keys: list[dict]) -> str:
        return ", ".join(f"{k['key']}: {k.get('summary') or ('set' if k['set'] else 'not set')}" for k in keys)

    def cmd_list_integrations(self, parameter_string: str = ""):
        self.background_command(self._run_list_integrations, "")

    def _run_list_integrations(self, _="") -> str:
        statuses = self._integration_monitor().describe_all()
        if not statuses:
            info("list_integrations: nothing registered")
            return "No integrations are registered."
        lines = []
        for status in sorted(statuses, key=lambda s: s["name"].lower()):
            line = f"{status['name']} — {self._STATE_LABELS.get(status['state'], status['state'])}"
            if status["state"] == "disabled":
                line += f'; enable: enable_integration("{status["name"]}")'
            if status["last_error"] and status["state"] in ("warning", "error", "needs_2fa", "rate_limited"):
                line += f": {status['last_error'][:160]}"
            if status["last_check_time"]:
                line += f" (checked {self._age(status['last_check_time'])})"
            if status["keys"]:
                line += f" [{self._keys_summary(status['keys'])}]"
            if status.get("accounts"):
                line += f"; accounts: {render_accounts_status(status['accounts'])}"
            if "feed" in status:
                from assistant.integration.FeedRunner import describe_feed_status
                # A disabled feed has no runner to describe, but it is still a feed.
                line += "; feed" if status["state"] == "disabled" else f"; feed: {describe_feed_status(status['feed'])}"
            line += "; restartable" if status["restartable"] else "; needs a server restart to change"
            if status.get("sensitive"):
                line += "; private data: enabling it ends the beta's transcript telemetry"
            lines.append(line)
        info(f"list_integrations: {len(lines)} integrations")
        return "\n".join(lines)

    @staticmethod
    def _age(epoch: float) -> str:
        seconds = max(0, int(time.time() - epoch))
        if seconds < 60:
            return f"{seconds}s ago"
        if seconds < 3600:
            return f"{seconds // 60} min ago"
        return f"{seconds // 3600} h ago"

    def cmd_config_help(self, parameter_string: str = ""):
        self.background_command(self._run_config_help, parameter_string.strip().strip('"\''))

    def _run_config_help(self, section: str = "") -> str:
        if not section:
            lines = []
            for name in ConfigSchema.sections():
                counts = ConfigStore.section_summary(name)
                lines.append(f"{name} ({counts})" if counts else name)
            info(f"config_help: listed {len(lines)} sections")
            return ("Sections of config.example.yaml (set = config.yaml holds a real value):\n" + "\n".join(lines)
                    + "\n\nconfig_help(section) shows one section with its comments.")
        text = ConfigSchema.section_text(section)
        if text is None:
            info(f"config_help: no section '{section}'")
            return f"No section '{section}' in config.example.yaml. Sections: {', '.join(ConfigSchema.sections())}"
        name = section.strip().lower()
        if not ConfigSchema.leaves(name):
            info(f"config_help: section '{section}' (documentation only, no keys)")
            return text
        info(f"config_help: section '{section}'")
        return f"{text}\n\nCurrently: {ConfigStore.summary([name])}"

    def cmd_config_request(self, parameter_string: str) -> str:
        """Asks the user for one setting through the masked prompt (the vault_request
        machinery with a config-writing accept) — the LLM gets 'saved', never the value."""
        key = ConfigSchema.normalize_key(parameter_string.strip().strip('"\''))
        if not key:
            return "config_request failed: expected a setting key such as imap.accounts[0].password"
        try:
            ConfigStore.check_key(key)
        except ConfigStoreError as e:
            info(f"config_request: refused — {e}")
            return f"config_request failed: {e}"
        ctx = self._current_context
        source = getattr(ctx, "source", None)
        conversation = getattr(ctx, "conversation", None)
        self.background_command(self._run_config_request, (key, source, conversation), lane=self.CONFIG_LANE)

    def _run_config_request(self, key: str, source=None, conversation=None) -> str:
        outcome: dict = {}
        replaces = ConfigStore.is_set(key)

        def accept(value: str) -> str:
            try:
                ConfigStore.set_value(key, value)
            except ConfigStoreError as e:
                outcome["reason"] = str(e)
                raise SecretRejected(str(e))
            outcome["saved"] = True
            return f"Saved {key}"

        details = "This replaces the current value." if replaces else ""
        decision = notifier().request_secret(title=f"Setting {key}", details=details, accept=accept,
                                             source=source, conversation=conversation)
        if decision == "accepted":
            return f"{key} saved. {self._restart_hint(key)}"
        info(f"config_request: {decision} for '{key}' — nothing stored")
        messages = {
            "cancel": "The user cancelled the prompt — nothing was saved.",
            "timeout": "The user did not answer the prompt in time — nothing was saved.",
            "rejected": f"The value was refused ({outcome.get('reason', 'it could not be stored')}) — "
                        "nothing was saved.",
            "unavailable": "No chat interface can show the prompt right now — ask the user to open the console "
                           "or Telegram and try again.",
        }
        return messages.get(decision, f"config_request ended with '{decision}' — nothing was saved.")

    RESTART_HINT = "Restart the integration that uses it (restart_integration) for the change to take effect."
    TELEGRAM_RESTART_HINT = 'Run restart_integration("Telegram") for the bot to pick it up — no server restart.'
    # The chat-id lists are read live by the bot (TelegramTransport._live_chat_ids). The FTUE
    # assistant once restarted the bot from the Telegram chat that had just sent its chat id,
    # which killed that very conversation and left its spinner stuck.
    TELEGRAM_LIVE_HINT = "In force from the next message on — the bot reads it live, no restart needed."
    TELEGRAM_RESTART_KEYS = frozenset({"telegram_bot.bot_token", "telegram_bot.link_previews"})
    SERVER_RESTART_HINT = ("The server reads interfaces.* only at startup — offer restart_server "
                           "(developer toolkit) for the change to take effect.")
    # The FTUE assistant once saved api_keys.gemini, found no integration owning it in the
    # listing and concluded the key is read at startup — then offered a server restart mid-setup.
    # The consumers that matter for setup build their LLM on first use, so say so.
    MODEL_KEY_HINT = ("Model-provider key. Photo indexing, video descriptions and any integration enabled "
                      "from now on use it straight away — setup needs no restart.")
    MODEL_PROVIDER_KEYS = frozenset({"openai", "anthropic", "gemini", "grok", "openrouter"})
    DEVELOPER_RELEASE_HINT = ("Takes effect at the next restart_server or nightly update, which checks out master "
                              "(true) or release (false) before pulling — offer the restart if the user wants to "
                              "switch now.")
    TELEMETRY_HINT = ("Telemetry reads it for every event: false is in force from the next event on, "
                      "no restart needed. Switching telemetry.enabled back to true takes a restart_server.")
    AI_FOOTER_HINT = "Read at each send: in force from the next message on, no restart needed."

    IS_DEVELOPER_HINT = ("Read at each prompt build: the self_improvement toolkit is offered (true) or "
                         "withdrawn (false) from the next turn on, no restart needed.")

    @classmethod
    def _restart_hint(cls, key: str) -> str:
        if key == "developer_release":
            return cls.DEVELOPER_RELEASE_HINT
        if key == "is_developer":
            return cls.IS_DEVELOPER_HINT
        if key == cls.AI_FOOTER_KEY:
            return cls.AI_FOOTER_HINT
        if key.startswith("telemetry."):
            return cls.TELEMETRY_HINT
        if key.startswith("interfaces."):
            return cls.SERVER_RESTART_HINT
        if key.startswith("telegram_bot."):
            return cls.TELEGRAM_RESTART_HINT if key.split("[", 1)[0] in cls.TELEGRAM_RESTART_KEYS else cls.TELEGRAM_LIVE_HINT
        if key.startswith("api_keys.") and key.split(".", 1)[1] in cls.MODEL_PROVIDER_KEYS:
            return cls.MODEL_KEY_HINT
        return cls.RESTART_HINT

    def cmd_config_set(self, *args: str) -> str:
        """Writes a value the assistant states itself — a plain setting the user said in the chat,
        or a credential it already holds (an API key it just created in the browser). A credential
        the user has to supply preferably goes through config_request's masked prompt."""
        if len(args) < 2:
            return "config_set failed: expected (key, value), e.g. config_set(\"owner_name\", \"Anna\")"
        key = ConfigSchema.normalize_key(args[0].strip().strip('"\''))
        # An unquoted value with commas arrives split; re-join it like send_imessage does.
        value = ", ".join(args[1:]).strip().strip('"\'')
        try:
            ConfigStore.check_key(key)
        except ConfigStoreError as e:
            info(f"config_set: refused — {e}")
            return f"config_set failed: {e}"
        self.background_command(self._run_config_set, [key, value], lane=self.CONFIG_LANE)

    def _run_config_set(self, key: str, value: str) -> str:
        interface = ConfigStore.interface_flag(key)
        before = ConfigStore.interface_enabled(interface) if interface else None
        try:
            ConfigStore.set_value(key, value)
        except ConfigStoreError as e:
            info(f"config_set: refused — {e}")
            return f"config_set failed: {e}"
        if interface and ConfigStore.interface_enabled(interface) == before:
            # Telegram and console are on by default: writing "true" for them changes nothing,
            # and the restart hint would send the LLM off to restart_server for no reason.
            info(f"config_set: {key} was already {'on' if before else 'off'} in effect — no restart needed")
            return (f"{key} saved, but the {interface} interface was already {'on' if before else 'off'} "
                    f"(its default) — nothing changes and no restart is needed.")
        return f"{key} saved. {self._restart_hint(key)}"

    TELEGRAM_FEED = "Telegram Feed"

    def cmd_restart_integration(self, parameter_string: str):
        name = parameter_string.strip().strip('"\'')
        if not name:
            return "restart_integration failed: expected an integration name from list_integrations"
        self.background_command(self._run_restart_integration, name, lane=self.CONFIG_LANE)

    def _run_restart_integration(self, name: str) -> str:
        monitor = self._integration_monitor()
        result = monitor.restart(name)
        if result is None:
            known = ", ".join(sorted(s["name"] for s in monitor.get_all_statuses()))
            return f"No integration named '{name}'. Known: {known}"
        return self._describe_restart(result["status"], result["restartable"])

    def cmd_enable_integration(self, parameter_string: str):
        """Adds the integration to enabled_integrations by name and rebuilds it — one call,
        so the LLM never juggles list indices (a second index write used to push the first
        name out)."""
        name = parameter_string.strip().strip('"\'')
        if not name:
            return "enable_integration failed: expected an integration name from list_integrations"
        self.background_command(self._run_enable_integration, name, lane=self.CONFIG_LANE)

    def cmd_disable_integration(self, parameter_string: str):
        name = parameter_string.strip().strip('"\'')
        if not name:
            return "disable_integration failed: expected an integration name from list_integrations"
        self.background_command(self._run_disable_integration, name, lane=self.CONFIG_LANE)

    def _run_enable_integration(self, name: str) -> str:
        return self._switch_integration(name, enable=True)

    def _run_disable_integration(self, name: str) -> str:
        return self._switch_integration(name, enable=False)

    def _switch_integration(self, name: str, enable: bool) -> str:
        """Edit enabled_integrations, then restart the integration so the list takes effect now."""
        verb = "enable" if enable else "disable"
        canonical = self._canonical_integration_name(name)
        if canonical is None:
            known = ", ".join(sorted(self._known_integration_names()))
            info(f"{verb}_integration: no integration named '{name}'")
            return f"No integration named '{name}'. Known: {known}"
        was_listed = ConfigManager.get_instance().integration_enabled(canonical)
        try:
            changed = (ConfigStore.enable_integration if enable else ConfigStore.disable_integration)(canonical)
        except ConfigStoreError as e:
            info(f"{verb}_integration: refused — {e}")
            return f"{verb}_integration failed: {e}"
        result = self._integration_monitor().restart(canonical)
        if was_listed != enable:
            Telemetry.get_instance().integration_switch(canonical, enable, result["status"]["state"] if result else None)
        else:
            trace(f"{verb}_integration: {canonical} was already {verb}d, no telemetry event")
        if result is None:
            return changed
        return f"{changed}; {self._describe_restart(result['status'], result['restartable'])}"

    def _known_integration_names(self) -> list[str]:
        return [s["name"] for s in self._integration_monitor().get_all_statuses()]

    def _canonical_integration_name(self, name: str) -> Optional[str]:
        """The name as list_integrations shows it for any spelling the config accepts."""
        from common.ConfigManager import integration_key
        wanted = integration_key(name)
        for known in self._known_integration_names():
            if integration_key(known) == wanted:
                return known
        return None

    def _describe_restart(self, status: dict, restartable: bool) -> str:
        if not restartable:
            return f"{status['name']} cannot be restarted from here — restart the server for a config change to apply."
        state = status["state"]
        line = f"{status['name']} restarted: {self._STATE_LABELS.get(state, state)}"
        if state == "disabled":
            line += f' — not in enabled_integrations; enable_integration("{status["name"]}") adds it'
        elif state == "not_configured":
            missing = [k["key"] for k in ConfigStore.describe_keys(status["config_keys"]) if not k["set"]]
            if missing:
                line += f" — not set: {', '.join(missing)}"
            elif status["last_error"]:
                line += f" — {status['last_error']}"
        elif state == "needs_2fa":
            line += " — " + (status.get("login_hint")
                             or "the login must be completed in the console's tab for it (the code arrives by SMS or email)")
        elif status["last_error"]:
            line += f" — {status['last_error'][:300]}"
        return line

    def cmd_remember(self, parameter_string: str):
        MemoryFile.append(parameter_string)

    def cmd_post_message(self, destination: str = "", *text_parts: str):
        """Say something somewhere other than the asking chat: the owner's chats, a group the
        bot is in, a room. Synchronous and silent on success — the command echo in the owner's
        chat is the audit trail, and a result would cost the model another round; a failure
        comes back as the error so the model can correct the destination in the same turn.
        The text is vault-redacted first: a group or a speaker is not the owner's eyes, and
        VaultLLM may have restored a real value in the reply."""
        destination = destination.strip().strip("\"'`")
        # An under-quoted text with commas arrives in pieces: put it back together.
        text = ", ".join(part.strip() for part in text_parts if part.strip())
        if not destination or not text:
            return "post_message failed: expected 'destination, text' — available destinations: " + Destinations.describe()
        redacted = redact_for_storage(text)
        if redacted != text:
            trace(f"post_message: {len(text) - len(redacted)} chars of vault values redacted before sending")
        return Destinations.send(destination, redacted) or None

    # The watch list and the procedures: the argument handling lives with the stores
    # (the memory curator's command block runs the same commands through them).
    def cmd_load_watch(self, *names: str):
        self.background_command(self._run_load_watch, list(names))

    @staticmethod
    def _run_load_watch(*names: str) -> str:
        """The named watch items in full, splatted like `_run_load_procedure`."""
        names = [n for n in names if n and n.strip()]
        if not names:
            return "load_watch failed: expected an item's name. Open: " + (", ".join(i.name for i in WatchList.items()) or "none")
        return "\n".join(WatchList.load(name) for name in names)

    def cmd_save_watch(self, *args: str) -> str:
        return WatchList.run("save_watch", list(args))

    def cmd_checked_watch(self, *args: str) -> str:
        return WatchList.run("checked_watch", list(args))

    def cmd_delete_watch(self, *args: str) -> str:
        return WatchList.run("delete_watch", list(args))

    def cmd_load_procedure(self, *names: str):
        self.background_command(self._run_load_procedure, list(names))

    @staticmethod
    def _run_load_procedure(*names: str) -> str:
        """The named procedures, each under its header. Splatted like every list-parameter
        background command (`function(*params)`)."""
        names = [n for n in names if n and n.strip()]
        if not names:
            return "load_procedure failed: expected a procedure name. Available: " + (", ".join(Procedures.names()) or "none")
        return "\n\n".join(Procedures.load(name) for name in names)

    def cmd_save_procedure(self, *args: str) -> str:
        return Procedures.run("save_procedure", list(args))

    def cmd_delete_procedure(self, *args: str) -> str:
        return Procedures.run("delete_procedure", list(args))

    def cmd_false_positive(self, parameter_string: str):
        info("We detected a false positive!")

        current_time = time.time()
        Commands.false_positive_timestamps.append(current_time)

        time_window_in_seconds = 10
        five_seconds_ago = current_time - time_window_in_seconds
        recent_count = sum(1 for ts in Commands.false_positive_timestamps if ts > five_seconds_ago)

        trace(f"False positives in last {time_window_in_seconds} seconds: {recent_count}")

        if recent_count >= 2:
            self.report_fp_flood()

    @staticmethod
    def report_fp_flood():
        if Commands.false_positive_reporter is None:
            info("No voice interface — false-positive flood report ignored.")
            return
        note("False positive flood! We're listening to gibberish - reporting to the voice interface.")
        Commands.false_positive_reporter()

    def cmd_turn_on(self, parameter_string: str) -> str:
        return home.active().turn_on(self._home_name(parameter_string))

    def cmd_turn_off(self, parameter_string: str) -> str:
        return home.active().turn_off(self._home_name(parameter_string))

    def cmd_stop(self, parameter_string: str) -> str:
        return home.active().stop(self._home_name(parameter_string))

    def cmd_dim(self, *args: str) -> str:
        if len(args) < 2:
            return "dim failed: expected 'name, level' parameters"
        try:
            level = int(float(args[1].strip().rstrip("%")))
        except ValueError:
            return f"dim failed: '{args[1].strip()}' is not a level"
        return home.active().dim(self._home_name(args[0]), level)

    def fire_and_forget(self, function, params):
        def wrapper():
            try:
                function(params)
            except Exception as e:
                error(f"Error in fire_and_forget thread: {e}")

        threading.Thread(target=wrapper).start()

    # --- background commands -------------------------------------------------------------
    # A command that has to wait on the outside world (a search, a page fetch, IMAP) is
    # handed to a lane instead of running on the command worker. Lanes:
    #   IO_LANE      — runs concurrently, capped by _IO_SLOTS across all Commands instances.
    #                  For independent HTTP / file work: an LLM reply that emits google(),
    #                  search_x() and search_email() together waits for the slowest one,
    #                  not the sum.
    #   BROWSER_LANE — strictly one at a time, in submission order. Every browser command
    #                  drives the one shared Chrome tab (open_url's tier-1 fetch included),
    #                  and interleaving two of them wedges both.
    #   CONFIG_LANE  — strictly one at a time, in submission order. A reply that emits
    #                  config_set("icloud.devices[0].id"), …[1].id and restart_integration
    #                  together only works in that order: each write reads config.yaml back
    #                  before the next, and the restart must see them all (2026-09-08 log:
    #                  run in parallel, seven of eight writes were refused or lost).
    # All lanes feed one batch: results are published together, in submission order, when
    # the last outstanding command of the batch finishes.
    IO_LANE = "io"
    BROWSER_LANE = "browser"
    CONFIG_LANE = "config"
    SERIAL_LANES = frozenset({BROWSER_LANE, CONFIG_LANE})
    MAX_PARALLEL_IO = 5
    _IO_SLOTS = threading.BoundedSemaphore(MAX_PARALLEL_IO)

    # Runs on: command worker thread. Hands the work to a lane thread.
    def background_command(self, function, params, lane: str = IO_LANE):
        echo, self._current_echo = self._current_echo, None
        if lane == self.CONFIG_LANE:
            self._setup_activity()
        with self.bg_lock:
            seq = self._bg_seq
            self._bg_seq += 1
            # _bg_outstanding is the source of truth for "a batch is in flight": the last
            # command out decrements it to zero under this lock and publishes there, so a
            # submit either joins the running batch or starts a fresh one — never lands
            # between the two.
            if self._bg_outstanding == 0:
                self.command_processor.background_command_start()
                # _bg_results is not reset here: an earlier batch of the same reply may have
                # published already, and the next publish must carry its results too. The
                # processor forgets them when the reply is consumed (forget_batch_results).
                self._bg_batch_started = time.time()
                self._bg_batch_count = 0
                self._bg_busy_seconds = 0.0
                self._bg_idle.clear()
            self._bg_outstanding += 1
            task = (seq, function, params, echo)
            if lane in self.SERIAL_LANES:
                self._serial_queues.setdefault(lane, queue.Queue()).put(task)
                if lane not in self._serial_workers:
                    worker = threading.Thread(target=self._serial_lane_worker, args=(lane,),
                                              name=f"Bg{lane.capitalize()}Worker", daemon=True)
                    self._serial_workers[lane] = worker
                    worker.start()
            else:
                threading.Thread(target=self._io_lane_run, args=(task,),
                                 name=f"BgCommand-{seq}", daemon=True).start()

    def wait_for_background(self, timeout: Optional[float] = None) -> bool:
        """Blocks until every background command of this instance has finished and the
        batch is published. True when idle, False on timeout."""
        return self._bg_idle.wait(timeout)

    def background_outstanding(self) -> int:
        with self.bg_lock:
            return self._bg_outstanding

    # Runs on: its own daemon thread, one per IO_LANE command.
    def _io_lane_run(self, task):
        with self._IO_SLOTS:
            self._run_one(*task)

    # Runs on: a serial lane's daemon thread. Drains that lane's queue in order and
    # retires when it is empty.
    def _serial_lane_worker(self, lane: str):
        tasks = self._serial_queues[lane]
        while True:
            try:
                task = tasks.get_nowait()
            except queue.Empty:
                # Retire atomically: re-check under the lock so a concurrent
                # background_command() either enqueues before we retire (we keep going)
                # or spawns a fresh worker after we've dropped this one.
                with self.bg_lock:
                    if tasks.empty():
                        del self._serial_workers[lane]
                        return
                continue
            self._run_one(*task)
            tasks.task_done()

    # Runs on: a lane thread. Executes one command, completes its chat echo and files
    # its result; the last result of the batch triggers the publish.
    def _run_one(self, seq: int, function, params, echo):
        started = time.time()
        result = None
        failed = True
        try:
            # When the caller passes a list/tuple, splat as positional args so
            # multi-arg helpers (e.g. _run_send_email(to, subject, body)) work.
            if isinstance(params, (list, tuple)):
                result = function(*params)
            else:
                result = function(params)
            failed = False
        except Exception as e:
            # Never let a failing command take the batch down with it — that would leave
            # command_running_in_background stuck True and hang the request. Surface
            # the error to the LLM instead.
            error(f"Background command {getattr(function, '__name__', function)} failed: {e}")
            result = f"Command failed: {e}"
        finally:
            if echo is not None:
                self._complete_echo(echo, describe_completion(started, None if failed else result, failed=failed))
            self._finish_one(seq, result, failed, time.time() - started)

    def _finish_one(self, seq: int, result, failed: bool, elapsed: float):
        with self.bg_lock:
            self._bg_results[seq] = result
            if failed:
                self.bg_results_contain_error = True
            self._bg_batch_count += 1
            self._bg_busy_seconds += elapsed
            self._bg_outstanding -= 1
            if self._bg_outstanding > 0:
                return
            wall = time.time() - self._bg_batch_started
            if self._bg_batch_count > 1:
                info(f"{self._bg_batch_count} background commands done in {wall:.1f}s "
                     f"(would have taken {self._bg_busy_seconds:.1f}s one after another).")
            self._publish_bg_results()
            self.command_processor.background_command_done()
            self._bg_idle.set()

    @staticmethod
    def _complete_echo(echo, detail: str):
        try:
            notifier().complete_command_notification(echo, detail)
        except Exception as e:
            error(f"Failed to complete command notification: {e}")

    @property
    def accumulated_bg_results(self) -> list:
        """The batch's results so far, in submission order — the order the LLM emitted
        the commands, whatever order they finished in."""
        return [self._bg_results[k] for k in sorted(self._bg_results)]

    # Runs on: a lane thread, under bg_lock. Publishes everything accumulated since the
    # reply was last consumed as one batch reply — every result, not only this batch's, so
    # the processor can replace its batch part whole. What the synchronous path appended
    # (errors, denials) is the processor's own and goes in front when it composes the reply.
    def _publish_bg_results(self):
        cp = self.command_processor
        contains_error = self.bg_results_contain_error
        results = self.accumulated_bg_results
        list_results = [r for r in results if isinstance(r, list)]
        # The reference text (a toolkit loaded beside a read) goes last, so the round's opening
        # says whether anything in it was read for the user — see is_reference_result.
        str_results = sorted((r for r in results if isinstance(r, str) and r.strip()), key=is_reference_result)
        if str_results and is_reference_result(str_results[-1]) and not is_reference_result(str_results[0]):
            trace("Background results: the reference text is placed after the other results")
        if list_results:
            combined = []
            for lr in list_results:
                combined.extend(lr)
            if str_results:
                combined.append({"type": "text", "text": "\n\n".join(str_results)})
            cp.set_command_reply(combined, contains_error=contains_error)
        elif str_results:
            cp.set_command_reply("\n\n".join(str_results), contains_error=contains_error)
        else:
            trace(f"{len(results)} background result(s), all empty — nothing to publish.")

    # Runs on: the thread that consumed the reply (see CommandProcessor.clear_command_reply).
    def forget_batch_results(self):
        with self.bg_lock:
            if self._bg_outstanding:
                trace(f"Reply consumed with {self._bg_outstanding} background command(s) still "
                      "running — their batch keeps the results published so far.")
                return
            self._bg_results = {}
            self.bg_results_contain_error = False

    def list_commands(self) -> list[str]:
        # Name check first: getattr on every attribute would evaluate the properties too, and
        # `web_access` builds the WebAccess singleton (and its Chrome loop thread) on first use.
        core = [func.replace("cmd_", "", 1) for func in dir(self) if
                func.startswith("cmd_") and callable(getattr(self, func))]
        return sorted(core + list(self._owners))

    def is_sensitive(self, command_name: str) -> bool:
        return command_name in self._owner(command_name).SENSITIVE_COMMANDS

    def approval_note(self, command_name: str) -> str:
        return self._owner(command_name).APPROVAL_NOTES.get(command_name, "")

    def is_pure(self, command_name: str) -> bool:
        return command_name in self._owner(command_name).PURE_COMMANDS

    def is_multi_param(self, command_name: str) -> bool:
        return command_name in self._owner(command_name).MULTI_PARAM_COMMANDS

    # --- toolkits: the command references the LLM loads on demand (plans/TOOLKITS_PLAN.md) ---

    @staticmethod
    def _prompt_manager():
        """The live PromptManager, reached the way the bug-fix commands reach the history;
        None in a process without one (tests)."""
        aam = AutonomousActionManager.current()
        return aam.prompt_manager if aam is not None else None

    def cmd_load_toolkit(self, *names: str):
        self.background_command(self._run_load_toolkit, list(names))

    def _run_load_toolkit(self, *names: str) -> str:
        """The references of the named toolkits, each already in the LLM's history window
        answered with a pointer to it instead of the text again. Splatted like every
        list-parameter background command (`function(*params)`)."""
        names = [n for n in names if n and n.strip()]
        monitor = self._integration_monitor()
        if not names:
            return "load_toolkit failed: expected a toolkit name. Available: " + ", ".join(Toolkits.names(monitor))
        prompt_manager = self._prompt_manager()
        in_window = prompt_manager.message_history.loaded_toolkits() if prompt_manager else {}
        # The autonomous check's rounds live in its own messages, never in the history: what
        # it is sent must not become the baseline the history's readers are diffed against.
        record = self._current_profile is not AUTONOMOUS
        parts, loaded, pointed, unknown, off = [], [], [], [], []
        for name in names:
            toolkit = Toolkits.by_name(name)
            if toolkit is None:
                unknown.append(name.strip())
                parts.append(f"No toolkit named '{name.strip()}'. Available: {', '.join(Toolkits.names(monitor))}")
            elif not toolkit.offered_by(monitor):
                off.append(toolkit.name)
                parts.append(self._toolkit_off(toolkit, monitor))
            elif toolkit.name in in_window:
                # The reference is static, the carried data is not: a pointer back to the
                # reference, the data again as it is now.
                pointed.append(toolkit.name)
                at = time.strftime("%H:%M", time.localtime(in_window[toolkit.name]))
                parts.append(f"The {toolkit.name} toolkit was loaded at {at}: its reference is above in this "
                             f"conversation, use it from there." + self._carried_data(toolkit, prompt_manager, record))
            else:
                loaded.append(toolkit.name)
                parts.append(ToolkitPreloader.render_load(toolkit, prompt_manager, record=record))
        result = "\n\n".join(parts)
        info(f"load_toolkit: loaded {loaded or 'nothing'}, pointed back to {pointed or 'nothing'}, "
             f"unknown {unknown or 'none'}, integration off {off or 'none'} ({len(result)} chars, channel "
             f"{self._current_profile.name}{'' if record else ', baseline untouched'})")
        return result

    @staticmethod
    def _toolkit_off(toolkit, monitor) -> str:
        """The answer to a toolkit, or a command of one, whose integration is off: which
        integration it is and how the setup toolkit turns it on — or, for a toolkit a config
        key gates, which key."""
        if toolkit.requires is not None and not Toolkits.config.get(toolkit.requires, False):
            return (f"The {toolkit.name} toolkit is for a developer's install only: it needs "
                    f"`{toolkit.requires}: true` in config.yaml. Users work through the developer toolkit.")
        behind = toolkit.behind(monitor)
        if len(behind) == 1:
            return (f"The {toolkit.name} toolkit belongs to the {behind[0].name} integration, which is off — "
                    f"enable_integration(\"{behind[0].name}\") in the setup toolkit turns it on.")
        names = ", ".join(i.name for i in behind)
        return (f"The {toolkit.name} toolkit needs one of these integrations, and every one is off: {names}. "
                f"enable_integration(name) in the setup toolkit turns one on.")

    def refused(self, command: str) -> str | None:
        """Why *command* cannot run now: it belongs to a toolkit whose integration is off
        (Toolkit.offered_by). None when it may run."""
        toolkit = Toolkits.owner_of(command)
        if toolkit is None:
            return None
        monitor = self._integration_monitor()
        if toolkit.offered_by(monitor):
            return None
        info(f"Command {command} refused: the {toolkit.name} toolkit's integration or config key is off")
        return f"{command} failed: " + self._toolkit_off(toolkit, monitor)

    @staticmethod
    def _carried_data(toolkit, prompt_manager, record: bool = True) -> str:
        """The toolkit's carried sections (the home data) to send after its reference, current
        as of now (PromptManager.carried_data); empty for a toolkit that carries none, or
        without a PromptManager to take them from."""
        if not toolkit.sections or prompt_manager is None:
            return ""
        data = prompt_manager.carried_data(toolkit, record=record)
        return "\n\n" + data if data else ""

    @staticmethod
    def _short_url(u: str, max_len: int = 50) -> str:
        u = (u or "").strip()
        if len(u) <= max_len:
            return u
        try:
            parsed = urllib.parse.urlparse(u)
            host = parsed.netloc or u
            path = parsed.path or ""
            head = host + path
            if len(head) <= max_len:
                return head
            return head[:max_len]
        except Exception:
            return u[:max_len]

    # Returns a human-readable progress string for the given command, or None
    # if no describe_X method exists or the description is intentionally empty.
    # Accepts either a list[str] (canonical) or a str (legacy callers); the
    # describe_* methods themselves all take a single joined string for simplicity.
    def describe(self, command: str, params) -> Optional[str]:
        command = command.strip()
        f = getattr(self._owner(command), "describe_" + command, None)
        if f is None:
            return None
        joined = ", ".join(params) if isinstance(params, list) else (params or "")
        try:
            # A hook taking a second parameter gets the parsed arguments too — the joined text
            # cannot tell a comma inside one argument ("Alice, Bob", a group) from the separator.
            if len(inspect.signature(f).parameters) >= 2:
                text = f(joined, params if isinstance(params, list) else None)
            else:
                text = f(joined)
            return text if text else None
        except Exception as e:
            error(f"describe_{command} raised: {e}")
            return None

    def echo_text(self, command: str, params) -> Optional[str]:
        """What a chat shows for a command call: its description, the call itself for a
        command without a describe_ method, None for one whose description is empty."""
        text = self.describe(command, params)
        if text is None and not hasattr(self._owner(command.strip()), "describe_" + command.strip()):
            params_display = ", ".join(params) if isinstance(params, list) else str(params)
            return f"{command}({params_display})"
        return text

    @staticmethod
    def _first_argument(p: str) -> str:
        return p.split(",", 1)[0].strip().strip("\"'`")

    def describe_home_automation_logs(self, p):   return "Reading home automation logs"
    def describe_server_logs(self, p):            return "Reading server logs"
    def describe_open_url(self, p):               return f"Reading {self._short_url(p)}"
    def describe_google(self, p):                 return f"Searching for {p.strip()}"
    def describe_browse_open(self, p):            return f"Browsing {self._short_url(p)}"
    def describe_browse_eval(self, p):            return "Inspecting the page"
    def describe_browse_snapshot(self, p):        return "Re-reading the page"
    def describe_browse_click_at(self, p):        return f"Clicking at {p.strip()}" if p.strip() else "Clicking"
    def describe_browse_extract(self, p):         return "Extracting page text"
    def describe_browse_network(self, p):         return "Reading the page's network traffic"
    def describe_browse_wait(self, p):
        pattern, timeout = WebAccess._parse_wait_spec(p)
        timeout = max(1.0, min(timeout, WebAccess._WAIT_MAX_S))
        return f"Waiting up to {timeout:.0f}s for '{pattern}'" if pattern else "Waiting for the page"
    def describe_browse_screenshot(self, p):      return "Taking a screenshot"
    def describe_send_screenshot(self, p):        return "Sending screenshot"
    def describe_browse_close(self, p):           return "Closing browser"
    def describe_latest_email(self, p):
        parts = [x.strip().strip('"').strip("'") for x in p.split(",")] if p else []
        folder = parts[0] if parts and parts[0] else ""
        unread = len(parts) > 1 and parts[1].lower() in self._UNREAD_TOKENS
        scope = "unread email" if unread else "email"
        return f"Checking {scope}" + (f" in {folder}" if folder else "")
    def describe_search_email(self, p):           return f"Searching email for {p.strip()}"
    def _friendly_email_subject(self, email_id):
        """Cached subject for an email id, trimmed and truncated for a progress
        notification, or None when no usable subject is known (cache miss,
        "(no subject)" placeholder, or lookup error). Never raises and never
        leaks the raw internal id, so callers can safely fall back to a generic
        message."""
        if not email_id:
            return None
        try:
            subject = self._email_provider.subject_for(email_id)
        except Exception as e:
            trace(f"_friendly_email_subject: subject lookup failed for '{email_id}': {e}")
            return None
        if subject and subject.strip() and subject.strip() != "(no subject)":
            s = subject.strip()
            if len(s) > 50:
                s = s[:50].rstrip() + "…"
            return s
        return None
    def describe_read_email(self, p):
        s = self._friendly_email_subject(p)
        return f"Reading email '{s}'" if s else "Reading email…"
    def describe_move_email(self, p):
        if not p:
            return "Moving email to folder"
        head, sep, tail = p.rpartition(",")
        folder = (tail if sep else p).strip()
        email_id = head.strip() if sep else ""
        s = self._friendly_email_subject(email_id)
        if s:
            return f"Moving email '{s}' to {folder}"
        return f"Moving email to {folder}"
    def describe_list_folders(self, p):           return "Listing email folders"
    def describe_load_calendar(self, p):          return f"Looking at the calendar: {self._first_argument(p)}" if p.strip() else "Looking at a calendar"
    def describe_create_event(self, p):
        title = p.split(",", 1)[0].strip()
        return f"Creating calendar event '{title}'" if title else "Creating calendar event"
    def describe_delete_event(self, p):
        title = self._first_argument(p)
        return f"Deleting calendar event '{title}'" if title else "Deleting a calendar event"
    def describe_move_event(self, p):
        title = self._first_argument(p)
        return f"Moving calendar event '{title}'" if title else "Moving a calendar event"
    def describe_schedule_wakeup(self, p):     return f"Scheduling wakeup in {p.strip()} min"
    def describe_caffeinate(self, p):             return "Staying awake"
    def describe_send_email(self, p):
        recipient = p.split(",", 1)[0].strip() if p else ""
        return f"Sending email to {recipient}" if recipient else "Sending email"
    def describe_fetch_camera_images(self, p):    return "Fetching camera images"
    def describe_car_max_defrost(self, p):        return f"Turning on max defrost in {self._car_label(p)}"
    def describe_car_climate_on(self, p):         return f"Turning on the climate in {self._car_label(p)}"
    def describe_car_climate_off(self, p):        return f"Turning off the climate in {self._car_label(p)}"
    def describe_car_open_trunk(self, p):         return f"Opening the trunk of {self._car_label(p)}"

    def describe_car_navigate(self, p):
        _vehicle, destination = self._navigation_args(tuple(p.split(",")) if p else ())
        return f"Sending the car's navigation to {destination}" if destination else "Sending the car a navigation destination"
    def describe_self_analysis(self, p):          return "Running self-analysis"
    def describe_self_improvement(self, p):       return "Running self-improvement"
    def describe_improve_integration(self, p):    return f"Working on the {self._first_argument(p)} integration"
    def describe_report_issue(self, p):           return "Reporting an issue to the developers"
    def describe_restart_server(self, p):         return "Restarting server"
    def describe_try_pull_request(self, p):       return f"Trying PR #{p.strip().lstrip('#').strip() or '?'}"
    def describe_pr_feedback(self, p):            return f"Posting feedback on PR #{self._first_argument(p).lstrip('#') or '?'}"
    def describe_execute(self, p):                return f"Running action {p.strip()}"
    def describe_vault_add(self, p):
        # Never the value: the notification is exactly where a secret must not be echoed.
        parts = [x.strip() for x in p.split(",")]
        if len(parts) >= 2 and parts[1]:
            return f"Adding {parts[1]} to the vault ({parts[0]})"
        return "Adding a secret to the vault"
    def describe_vault_request(self, p):      return None  # the masked prompt itself says what is asked for
    def describe_list_integrations(self, p):  return "Listing integrations"
    def describe_config_help(self, p):        return f"Reading the settings for {p.strip()}" if p.strip() else "Listing the settings sections"
    def describe_config_request(self, p):     return f"Asking for setting {p.strip()}" if p.strip() else "Asking for a setting"
    def describe_config_set(self, p):
        key = p.split(",", 1)[0].strip().strip('"\'')
        return f"Setting {key}" if key else "Setting a value"
    def describe_restart_integration(self, p): return f"Restarting {p.strip()}" if p.strip() else "Restarting an integration"
    def describe_enable_integration(self, p):  return f"Enabling {p.strip()}" if p.strip() else "Enabling an integration"
    def describe_disable_integration(self, p): return f"Disabling {p.strip()}" if p.strip() else "Disabling an integration"
    def describe_remember(self, p):               return f"Saving to memory: {p.strip()}" if p.strip() else "Saving to memory"
    def describe_post_message(self, p):           return f"Posting to {self._first_argument(p)}" if p.strip() else "Posting a message"
    def describe_load_watch(self, p):             return f"Looking at: {self._first_argument(p)}" if p.strip() else "Looking at a watch item"
    def describe_save_watch(self, p):             return f"Watching: {self._first_argument(p)}" if p.strip() else "Saving a watch item"
    def describe_checked_watch(self, p):          return f"Checked: {self._first_argument(p)}" if p.strip() else "Marking a watch item checked"
    def describe_delete_watch(self, p):           return f"No longer watching: {self._first_argument(p)}" if p.strip() else "Deleting a watch item"
    def describe_load_procedure(self, p):         return f"Loading the {self._first_argument(p)} procedure" if p.strip() else "Loading a procedure"
    def describe_save_procedure(self, p):         return f"Saving the {self._first_argument(p)} procedure" if p.strip() else "Saving a procedure"
    def describe_delete_procedure(self, p):       return f"Deleting the {self._first_argument(p)} procedure" if p.strip() else "Deleting a procedure"
    def describe_false_positive(self, p):         return None  # noisy / dev-only — suppress
    def describe_load_toolkit(self, p):
        names = [n.strip().strip("\"'`") for n in p.split(",") if n.strip()]
        if not names:
            return "Loading a toolkit"
        if len(names) == 1:
            return f"Loading the {names[0]} toolkit"
        return f"Loading the {', '.join(names[:-1])} and {names[-1]} toolkits"
    def describe_turn_on(self, p):                return f"Turning on {p.strip()}"
    def describe_turn_off(self, p):               return f"Turning off {p.strip()}"
    def describe_stop(self, p):                   return f"Stopping {p.strip()}"

    def describe_dim(self, p):
        parts = [x.strip() for x in p.split(",")]
        if len(parts) >= 2:
            return f"Dimming {parts[0]} to {parts[1]}%"
        return f"Dimming {p.strip()}"

    def describe_browse_act(self, p):
        spec = (p or '').strip()
        ops = WebAccess._split_act_batch(spec)
        if len(ops) > 1:
            return ' · '.join(self._describe_one_act(op) for op in ops)
        return self._describe_one_act(spec)

    def _describe_one_act(self, spec):
        idx_str, _, value = spec.partition('=') if '=' in spec else (spec, '', None)
        idx_str = idx_str.strip()
        try:
            idx = int(idx_str)
        except ValueError:
            return f"Acting on {spec}"
        label = self.web_access.get_snapshot_label(idx)
        target = label if label else f"element {idx}"
        if value is not None:
            value = value.strip()
            if value.endswith('\\n'):
                return f"Filling {target} with '{value[:-2].rstrip()}' and pressing Enter"
            return f"Filling {target} with '{value}'"
        return f"Clicking on {target}"

    def describe_browse_key(self, p):
        spec = (p or '').strip()
        if ',' in spec:
            idx_str, _, key = spec.partition(',')
            key = key.strip()
            try:
                idx = int(idx_str.strip())
            except ValueError:
                return f"Pressing {spec}"
            label = self.web_access.get_snapshot_label(idx)
            target = label if label else f"element {idx}"
            return f"Pressing {key} on {target}"
        return f"Pressing {spec}"

    def describe_browse_inspect(self, p):
        spec = (p or '').strip()
        idx_str, _, attr = spec.partition(',') if ',' in spec else (spec, '', '')
        idx_str = idx_str.strip()
        attr = attr.strip()
        try:
            idx = int(idx_str)
        except ValueError:
            return f"Inspecting {spec}"
        label = self.web_access.get_snapshot_label(idx)
        target = label if label else f"element {idx}"
        if attr:
            return f"Reading '{attr}' of {target}"
        return f"Inspecting {target}"

    def describe_search_history(self, p):
        q = p.split(",", 1)[0].strip()
        return f"Searching history for {q}" if q else "Searching history"

    def describe_show_history(self, p):
        parts = [x.strip() for x in p.split(",", 1)]
        if len(parts) == 2:
            return f"Reading history from {parts[0]} to {parts[1]}"
        return "Reading history"

    # Runs on: command worker thread (called from CommandProcessor._command_worker).
    # Returns "" when the command ran and the reason when it did not — the contract of
    # every cmd_ method (see CommandSet): a synchronous command has no result to hand back,
    # one with data to return runs it through background_command.
    def run(self, command: str, params: list[str]) -> str:
        command = command.strip()

        if command == "dim" and len(params) >= 2 and params[1].strip() == "0":
            trace("Changing dim command to turn_off since dim value is zero and we want clean data in stats.")
            command = "turn_off"
            params = [params[0]]

        params_display = ", ".join(params)
        note("Running command: " + command + "(" + params_display + ")")
        f = getattr(self._owner(command), "cmd_" + command, None)
        if f is None:
            error(f"Unknown command: {command}")
            return f"Unknown command: {command}"
        if refusal := self.refused(command):
            return refusal

        try:
            if self.is_multi_param(command):
                result = f(*params)
            else:
                result = f(params[0] if params else "")
            return result if isinstance(result, str) else ""
        except Exception as e:
            error(f"Exception running command {command}({params_display}): {e}")
            # An exception without a message (a bare TimeoutError()) would read as success.
            return str(e) or f"{command} failed: {type(e).__name__}"
