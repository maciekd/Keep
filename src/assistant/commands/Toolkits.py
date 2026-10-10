"""The toolkits: the command references the LLM loads on demand (plans/TOOLKITS_PLAN.md).

The system prompt carries `load_toolkit(name)` and a one-line index of these; each toolkit's
reference is a Markdown file that comes back as the command's result. Nothing is inlined into
the prompt: every toolkit, `home` included, is loaded on demand.

A toolkit more than one integration is behind, or none (`home`, `email`, `web`, `vault` …), is
declared in `CORE_TOOLKITS` with its reference in `res/toolkits/<name>.md`. A toolkit one
integration owns lives in that integration's folder, `src/integrations/<folder>/toolkit.md`:
a front-matter block (`name`, `summary`, `commands`, and `sections` / `hidden` when set) over
the reference, found by globbing the folders — no Python of the folder is imported to read
it. `TOOLKITS` is the two together, in `INDEX_ORDER`.

A toolkit can carry prompt sections (`Toolkit.sections`; home carries the house's state): their
placeholder is not in the prompt template, so the prompt never holds the data — `load_toolkit`
sends them, current, after the reference, and again on a pointer-back, since the reference is
static but the data is not. While the toolkit is in the LLM's window, later changes follow as
delta notes against what was sent (PromptManager._on_carried_section_updated_locked).

A hidden toolkit (`voice` — how to talk on the voice channel, and `false_positive`) is not in
the index: the channel it belongs to always has it preloaded (`ChannelProfile.fixed_toolkits`),
so a text chat's prompt never carries a word of it.

A toolkit is offered only while the integration behind it is on (`Toolkit.offered_by`, the
IntegrationMonitor's verdict): a folder toolkit while its folder's integration is enabled, a
core toolkit that fronts a contract (`needs` — home automation, the mail and calendar
sources) while an enabled integration implements it. One that is not offered is out of the
index, `load_toolkit` and the preloads answer that its integration is off, and its commands
are refused — the LLM never hears of a car the house does not have.
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import yaml

from assistant.integration.CameraProvider import CameraProvider
from assistant.integration.HomeAutomation import HomeAutomation
from assistant.integration.VehicleProvider import VehicleProvider
from assistant.integration.calendar.CalendarSource import CalendarSource
from assistant.integration.mail.MailSource import MailSource
from common.Paths import INTEGRATIONS_DIR, RES_DIR
from common.logging import error, trace

TOOLKITS_DIR = RES_DIR / "toolkits"
# An integration folder's own toolkit: front matter, then the reference.
FOLDER_TOOLKIT_FILE = "toolkit.md"
_FRONT_MATTER_RE = re.compile(r"\A---\n(.*?\n)---\n", re.S)
_FRONT_MATTER_KEYS = {"name", "summary", "commands", "sections", "hidden"}
# The header of a rendered toolkit; MessageHistory recognises loaded toolkits by it.
HEADER_PREFIX = "# Toolkit: "
# A reference is not a web page: the history stores it whole, so the files stay bounded.
MAX_FILE_CHARS = 10_000


@dataclass(frozen=True)
class Toolkit:
    name: str
    # One line in the index: what the toolkit is for and when to load it.
    summary: str
    commands: tuple[str, ...]
    # The prompt sections (PromptManager.sections keys) the toolkit carries, see the module doc.
    sections: tuple[str, ...] = ()
    # Left out of the index and the "Available:" lists; `load_toolkit` still resolves it.
    hidden: bool = False
    # An integration folder's toolkit.md; None for a core toolkit, whose file is in res/toolkits.
    file: Optional[Path] = None
    # A core toolkit that fronts a contract: offered while an enabled integration implements it.
    needs: Optional[type] = None

    @property
    def path(self) -> Path:
        return self.file or TOOLKITS_DIR / f"{self.name}.md"

    @property
    def folder(self) -> Optional[str]:
        """The integration folder a folder toolkit belongs to; None for a core toolkit."""
        return self.file.parent.name if self.file else None

    def offered_by(self, monitor) -> bool:
        """Whether the LLM is offered the toolkit: not while the IntegrationMonitor knows the
        integration behind it to be off (`folder_enabled` / `contract_enabled` False). None
        from the monitor — nothing registered for it yet — leaves it offered; without a
        monitor everything is."""
        if monitor is None:
            return True
        if self.folder is not None:
            return monitor.folder_enabled(self.folder) is not False
        if self.needs is not None:
            return monitor.contract_enabled(self.needs) is not False
        return True

    def behind(self, monitor) -> list:
        """The registered integrations the toolkit depends on (for the "it is off" answers)."""
        if monitor is None:
            return []
        if self.folder is not None:
            return monitor.folder_integrations(self.folder)
        if self.needs is not None:
            return monitor.contract_integrations(self.needs)
        return []

    def text(self) -> str:
        """The reference as the LLM receives it: a folder toolkit's front matter is not part of it."""
        text = self.path.read_text(encoding="utf-8")
        return _FRONT_MATTER_RE.sub("", text, count=1) if self.file else text


class ToolkitFileError(Exception):
    """A folder's toolkit.md that cannot be a toolkit; the message names the file."""


def read_folder_toolkit(path: Path) -> Toolkit:
    """The Toolkit a folder's toolkit.md declares in its front matter."""
    match = _FRONT_MATTER_RE.match(path.read_text(encoding="utf-8"))
    if match is None:
        raise ToolkitFileError(f"{path}: no front matter (a `---` block with name, summary, commands)")
    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError as e:
        raise ToolkitFileError(f"{path}: front matter is not YAML: {e}") from e
    if not isinstance(meta, dict):
        raise ToolkitFileError(f"{path}: front matter is not a mapping")
    unknown = set(meta) - _FRONT_MATTER_KEYS
    if unknown:
        raise ToolkitFileError(f"{path}: unknown front matter key(s) {sorted(unknown)}")
    name, summary = meta.get("name"), meta.get("summary")
    commands, sections, hidden = meta.get("commands"), meta.get("sections", []), meta.get("hidden", False)
    if not (isinstance(name, str) and name.strip() and isinstance(summary, str) and summary.strip()):
        raise ToolkitFileError(f"{path}: front matter needs a name and a summary")
    for key, value, may_be_empty in (("commands", commands, False), ("sections", sections, True)):
        if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value) or not (value or may_be_empty):
            raise ToolkitFileError(f"{path}: front matter `{key}` must be a list of names")
    if not isinstance(hidden, bool):
        raise ToolkitFileError(f"{path}: front matter `hidden` must be true or false")
    return Toolkit(name.strip(), summary.strip(), tuple(commands), tuple(sections), hidden, file=path)


def discover(root: Path = INTEGRATIONS_DIR) -> list[Toolkit]:
    """The toolkits the integration folders under *root* own, in folder order. A file that
    cannot be read as one is logged and left out: one broken folder must not take the
    server down (for the folders in this repository the unit suite fails instead)."""
    found = []
    for path in sorted(root.glob(f"*/{FOLDER_TOOLKIT_FILE}")):
        try:
            found.append(read_folder_toolkit(path))
        except (ToolkitFileError, OSError) as e:
            error(f"Toolkit left out — {e}")
    return found


def assemble(core: Iterable[Toolkit], discovered: Iterable[Toolkit], order: tuple[str, ...]) -> tuple[Toolkit, ...]:
    """Core and folder toolkits as one registry. A folder toolkit whose name, or one of whose
    commands, is already taken is logged and left out — the core and the folder read before
    it keep theirs. The names in *order* come first, in that order; any other toolkit after
    them, by name."""
    accepted = list(core)
    for toolkit in discovered:
        taken = {t.name: t for t in accepted}
        owners = {command: t for t in accepted for command in t.commands}
        clashes = sorted(c for c in toolkit.commands if c in owners)
        if toolkit.name in taken:
            error(f"Toolkit left out — {toolkit.path}: the name '{toolkit.name}' is already "
                  f"{taken[toolkit.name].path}'s")
        elif clashes:
            error(f"Toolkit left out — {toolkit.path}: {', '.join(clashes)} already belong(s) to "
                  f"the {owners[clashes[0]].name} toolkit")
        else:
            accepted.append(toolkit)
    rank = {name: i for i, name in enumerate(order)}
    return tuple(sorted(accepted, key=lambda t: (rank.get(t.name, len(rank)), t.name)))


CORE_TOOLKITS: tuple[Toolkit, ...] = (
    Toolkit("home", "lights, covers, actions and the home automation log, with the current state of every "
                    "light, cover, action and variable of the house",
            ("turn_on", "turn_off", "dim", "stop", "execute", "home_automation_logs"),
            sections=("home",), needs=HomeAutomation),
    Toolkit("cameras", "a still image from every camera of the house, or from the ones you name",
            ("fetch_camera_images",), needs=CameraProvider),
    Toolkit("cars", "the cars: defrost, climate, trunk, navigation",
            ("car_max_defrost", "car_climate_on", "car_climate_off", "car_open_trunk", "car_navigate"),
            needs=VehicleProvider),
    Toolkit("web", "read a web page or google something; anything that needs current information",
            ("open_url", "google"), sections=("web",)),
    Toolkit("browser", "drive a real browser when a page needs clicking, typing or a date picker before it "
                       "shows the answer; send a screenshot of it",
            ("browse_open", "browse_act", "browse_key", "browse_eval", "browse_inspect", "browse_snapshot",
             "browse_wait", "browse_network", "browse_click_at", "browse_extract", "browse_screenshot",
             "browse_close", "send_screenshot")),
    Toolkit("email", "list, search, read and move the user's mail; send an email",
            ("latest_email", "search_email", "read_email", "move_email", "list_folders", "send_email"),
            needs=MailSource),
    Toolkit("calendar", "create a calendar event; the coming week of a calendar the prompt does not show",
            ("create_event", "load_calendar"), needs=CalendarSource),
    Toolkit("timers", "anything the user wants done LATER, or watched for until it happens",
            ("schedule_wakeup", "caffeinate")),
    Toolkit("history", "search everything that ever happened, when long-term memory does not cover it",
            ("search_history", "show_history")),
    Toolkit("vault", "store a secret the user has said, or ask them for one you must never see",
            ("vault_add", "vault_request")),
    Toolkit("setup", "see which integrations are on, off, broken or unconfigured, and configure one",
            ("list_integrations", "config_help", "config_request", "config_set", "restart_integration",
             "enable_integration", "disable_integration", "telegram_login", "microsoft_login", "google_login",
             "google_login_code", "volvo_login", "volvo_login_code", "spotify_login", "spotify_login_code", "hue_pair")),
    Toolkit("developer", "study or change your own code, or check the machine you run on (network, disk, updates): analyse a bug or a question, build a fix or a small feature as a PR; read your own log, restart the server",
            ("self_analysis", "self_improvement", "server_logs", "restart_server")),
    Toolkit("voice", "how to reply when the user is speaking: spoken style, emotion tags, false positives",
            ("false_positive",), hidden=True),
)

# The order of the index, which is part of the prompt: a toolkit keeps its place wherever its
# file lives. A toolkit not named here (a new folder's) comes after these, by name.
INDEX_ORDER = ("home", "cameras", "web", "browser", "x", "email", "imessage", "whatsapp", "telegram", "music", "photos", "location", "calendar",
               "shopping_list", "weather", "timers", "cars", "husdata", "history", "vault", "setup", "developer", "voice")

TOOLKITS: tuple[Toolkit, ...] = assemble(CORE_TOOLKITS, discover(), INDEX_ORDER)
trace(f"Toolkits: {len(CORE_TOOLKITS)} core, {len(TOOLKITS) - len(CORE_TOOLKITS)} from integration folders")

# Documented in the prompt itself, outside any toolkit.
CORE_COMMANDS = frozenset({"load_toolkit", "remember",
                           "load_watch", "save_watch", "checked_watch", "delete_watch",
                           "load_procedure", "save_procedure", "delete_procedure"})
# Runnable but deliberately offered nowhere (Commands.py carries the todo asking whether to keep it).
UNDOCUMENTED = frozenset({"sonos_enqueue_and_play"})

_BY_NAME = {t.name: t for t in TOOLKITS}
# Section key → the toolkit that carries it.
CARRIED_SECTIONS: dict[str, str] = {key: t.name for t in TOOLKITS for key in t.sections}
# The heading a carried section is sent under; the prompt's own heading for it, once.
SECTION_HEADINGS = {"home": "HOME AUTOMATION DATA", "web": "WEB BROWSER"}


def offered(monitor=None) -> list[Toolkit]:
    """The toolkits the LLM may load: every one whose integration *monitor* does not know to
    be off (Toolkit.offered_by); all of them without a monitor."""
    return [t for t in TOOLKITS if t.offered_by(monitor)]


def names(monitor=None) -> list[str]:
    """The toolkits the LLM is offered by name (the hidden ones and the ones whose integration
    is off are not)."""
    return [t.name for t in offered(monitor) if not t.hidden]


def by_name(name: str) -> Toolkit | None:
    """Case-insensitive; tolerates quotes and an "email toolkit" / "the email" phrasing."""
    cleaned = name.strip().strip("\"'`").lower()
    for word in ("the ", "toolkit "):
        cleaned = cleaned.removeprefix(word)
    cleaned = cleaned.removesuffix(" toolkit").strip()
    return _BY_NAME.get(cleaned)


def owner_of(command: str) -> Toolkit | None:
    for toolkit in TOOLKITS:
        if command in toolkit.commands:
            return toolkit
    return None


def render_index(monitor=None) -> str:
    """The `{TOOLKITS}` text: one line per toolkit the LLM may load itself — the hidden ones
    and those of an integration *monitor* knows to be off are left out."""
    lines = [f"- `{t.name}` — {t.summary}" for t in offered(monitor) if not t.hidden]
    left_out = [t.name for t in TOOLKITS if not t.hidden and not t.offered_by(monitor)]
    trace(f"Toolkit index: {len(lines)} toolkit(s) offered"
          + (f", {', '.join(left_out)} left out (integration off)" if left_out else ""))
    return "\n".join(lines)


def render(toolkit: Toolkit, substitute: Callable[[str], str] = lambda text: text) -> str:
    """A toolkit as the LLM receives it: the header MessageHistory looks for, then the file
    with the prompt placeholders ({WAKE_INTERVAL_MINUTES}, {PHOTO_COUNT}) filled in."""
    return f"{HEADER_PREFIX}{toolkit.name}\n{substitute(toolkit.text()).rstrip()}\n"


def render_sections(toolkit: Toolkit, contents: dict[str, str]) -> str:
    """The toolkit's carried sections as `load_toolkit` sends them after the reference: each
    under its heading, from `contents` (section key → current text); empty for a toolkit
    that carries none. A section with nothing to say (the web one while Chrome is installed)
    is left out, heading and all."""
    return "\n\n".join(f"## {SECTION_HEADINGS.get(key, key.upper())}\n{contents[key].rstrip()}\n"
                       for key in toolkit.sections if contents.get(key, "").strip())


def loaded_name(text: str) -> str | None:
    """The toolkit name a rendered reference (a `command` history row) starts with, else None."""
    if not text.startswith(HEADER_PREFIX):
        return None
    return text[len(HEADER_PREFIX):].split("\n", 1)[0].strip() or None
