"""Reads a Claude Code session file (`~/.claude/projects/<project>/<session>.jsonl`) into the
facts the journal keeps: who started it, the prompts typed, the files edited, the git work.

One JSON record per line: `user` (a typed prompt, or a tool result), `assistant` (text and
`tool_use` blocks — Edit, Write, Bash…), `pr-link`, `custom-title` / `ai-title`, `cost-state`
and a few bookkeeping kinds. A session the owner typed into has `entrypoint` "cli" (the
terminal) or "claude-desktop"; one a script ran with `claude -p` — Keep's own self-improvement
runs — has "sdk-cli", which is how they are told apart (`is_scripted`). The file is appended
to while the session runs, so `read_records` reads from an offset and stops at the last
complete line; the offset it returns is the next read's start. Pure functions, no state."""
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from common.logging import trace

SCRIPTED_ENTRYPOINTS = ("sdk-cli",)      # `claude -p`: Keep's self-improvement runs, any script
EDIT_TOOLS = {"Edit": "file_path", "Write": "file_path", "MultiEdit": "file_path", "NotebookEdit": "notebook_path"}
PROMPT_MAX_CHARS = 600                   # what a prompt row keeps of a long prompt
REPLY_MAX_CHARS = 1500                   # what the digest keeps of the last assistant text

# A typed prompt that is really the CLI's own: slash commands, the output of local commands,
# the IDE's open file, a hook's note.
_CLI_PSEUDO_PROMPT = re.compile(r"^\s*<(?:command-name|command-message|local-command-stdout|local-command-stderr|"
                                r"system-reminder|ide_opened_file|ide_selection|user-prompt-submit-hook|task-notification)")
_COMMIT_SUBJECT = re.compile(r"-m\s+(?:\"\$\(cat\s+<<['\"]?EOF['\"]?\n([^\n]*)|\"((?:[^\"\\]|\\.)*)\"|'([^']*)'|(\S+))")
_NEW_BRANCH = re.compile(r"\bgit\s+(?:checkout\s+-b|switch\s+-c|branch)\s+([^\s;&|]+)")
_PUSH = re.compile(r"\bgit\s+push\b(?:\s+-u)?(?:\s+[^\s;&|-][^\s;&|]*)?(?:\s+([^\s;&|-][^\s;&|]*))?")
_PR_CREATE = re.compile(r"\bgh\s+pr\s+create\b")


@dataclass
class Digest:
    """What a whole session (or the part after its last summary) amounts to."""
    session_id: str
    project: str = ""
    cwd: str = ""
    branch: str = ""
    title: str = ""
    started: Optional[float] = None
    ended: Optional[float] = None
    prompts: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    git: list[str] = field(default_factory=list)
    pr_links: list[str] = field(default_factory=list)
    last_reply: str = ""
    lines_added: Optional[int] = None
    lines_removed: Optional[int] = None

    @property
    def duration_text(self) -> str:
        if self.started is None or self.ended is None:
            return ""
        minutes = max(0, int((self.ended - self.started) // 60))
        return f"{minutes // 60} h {minutes % 60} min" if minutes >= 60 else f"{minutes} min"

    @property
    def span_text(self) -> str:
        if self.started is None or self.ended is None:
            return ""
        start, end = datetime.fromtimestamp(self.started), datetime.fromtimestamp(self.ended)
        day = start.strftime("%Y-%m-%d ") if start.date() != end.date() else ""
        return f"{start.strftime('%H:%M')}–{day}{end.strftime('%H:%M')}"


def read_records(path: str, offset: int) -> tuple[list[dict], int]:
    """The complete JSON lines of `path` from byte `offset` on, and the offset just past the
    last of them. A line still being written (no newline yet) waits for the next read; a line
    that is not JSON is traced and skipped."""
    records = []
    with open(path, "rb") as handle:
        handle.seek(offset)
        data = handle.read()
    end = data.rfind(b"\n")
    if end < 0:
        return records, offset
    for line in data[:end].split(b"\n"):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            trace(f"Claude Code: skipped a line of {os.path.basename(path)} that is not JSON: {type(e).__name__}")
            continue
        if isinstance(record, dict):
            records.append(record)
    return records, offset + end + 1


def record_time(record: dict) -> Optional[float]:
    """The record's `timestamp` (ISO 8601, UTC) as an epoch, None without one."""
    value = record.get("timestamp")
    if isinstance(value, (int, float)):
        return value / 1000 if value > 1e12 else float(value)
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def entrypoint(records: list[dict]) -> Optional[str]:
    """The `entrypoint` of the first record that carries one; None when none does yet."""
    for record in records:
        if record.get("entrypoint"):
            return record["entrypoint"]
    return None


def is_scripted(entry: Optional[str]) -> bool:
    """A session run by a script (`claude -p`) rather than typed into."""
    return entry in SCRIPTED_ENTRYPOINTS


def project_label(cwd: str) -> str:
    """`/Users/marcus/Keep` as `~/Keep`; the Claude desktop app's scratch folder by that name,
    not its 150-character path; an empty cwd as "?"."""
    if not cwd:
        return "?"
    if "/scratch-workspaces/" in cwd:
        return "Claude app scratch workspace"
    home = os.path.expanduser("~")
    if cwd == home:
        return "~"
    if home and cwd.startswith(home + os.sep):
        return "~" + cwd[len(home):]
    return cwd


def prompt_text(record: dict) -> Optional[str]:
    """The text the owner typed, for a `user` record that is a prompt; None for a tool result,
    a sidechain (a subagent's) record or one of the CLI's own pseudo-prompts."""
    if record.get("type") != "user" or record.get("isSidechain"):
        return None
    content = (record.get("message") or {}).get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        if any(isinstance(block, dict) and block.get("type") == "tool_result" for block in content):
            return None
        text = "\n".join(block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text")
    else:
        return None
    text = text.strip()
    if not text or _CLI_PSEUDO_PROMPT.match(text):
        return None
    return text


def tool_uses(record: dict) -> list[tuple[str, dict]]:
    """(tool name, input) of every `tool_use` block of an `assistant` record."""
    if record.get("type") != "assistant":
        return []
    content = (record.get("message") or {}).get("content")
    if not isinstance(content, list):
        return []
    return [(block.get("name") or "", block.get("input") or {}) for block in content
            if isinstance(block, dict) and block.get("type") == "tool_use"]


def assistant_text(record: dict) -> str:
    content = (record.get("message") or {}).get("content")
    if not isinstance(content, list):
        return ""
    return "\n".join(block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text")


def edited_file(name: str, tool_input: dict, cwd: str) -> Optional[str]:
    """The path an edit tool touched, relative to the session's cwd when under it."""
    key = EDIT_TOOLS.get(name)
    if not key:
        return None
    path = tool_input.get(key)
    if not isinstance(path, str) or not path:
        return None
    if cwd and path.startswith(cwd.rstrip(os.sep) + os.sep):
        return path[len(cwd.rstrip(os.sep)) + 1:]
    return path


def git_events(command: str) -> list[str]:
    """What a shell command does to the repository, in words: a commit (with its subject),
    a new branch, a push, a pull request."""
    events = []
    if not command:
        return events
    if re.search(r"\bgit\s+commit\b", command):
        match = _COMMIT_SUBJECT.search(command)
        subject = next((group for group in match.groups() if group), "") if match else ""
        subject = subject.strip().strip("\"'")
        events.append(f'committed "{subject}"' if subject else "committed")
    for match in _NEW_BRANCH.finditer(command):
        events.append(f"created branch {match.group(1)}")
    push = _PUSH.search(command)
    if push:
        branch = push.group(1)
        events.append(f"pushed {branch}" if branch else "pushed")
    if _PR_CREATE.search(command):
        events.append("opened a pull request")
    return events


def truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def digest(session_id: str, records: list[dict]) -> Digest:
    """The session's facts from its records, in order: project and branch from the first
    record that names them, the typed prompts, the files edited (each once), the git work,
    the title Claude Code gave it and the last thing the assistant said."""
    result = Digest(session_id)
    seen_files: set[str] = set()
    for record in records:
        kind = record.get("type")
        if not result.cwd and record.get("cwd"):
            result.cwd = record["cwd"]
            result.project = project_label(result.cwd)
        if not result.branch and record.get("gitBranch") and record["gitBranch"] != "HEAD":
            result.branch = record["gitBranch"]
        when = record_time(record)
        if when is not None and kind in ("user", "assistant", "system"):
            result.started = when if result.started is None else result.started
            result.ended = when if result.ended is None else max(result.ended, when)
        if kind == "user":
            text = prompt_text(record)
            if text:
                result.prompts.append(truncate(text, PROMPT_MAX_CHARS))
        elif kind == "assistant":
            text = assistant_text(record)
            if text.strip():
                result.last_reply = truncate(text, REPLY_MAX_CHARS)
            for name, tool_input in tool_uses(record):
                path = edited_file(name, tool_input, result.cwd)
                if path and path not in seen_files:
                    seen_files.add(path)
                    result.files.append(path)
                if name == "Bash":
                    result.git.extend(git_events(tool_input.get("command") or ""))
        elif kind == "pr-link" and record.get("prUrl"):
            result.pr_links.append(record["prUrl"])
        elif kind in ("custom-title", "ai-title"):
            result.title = record.get("customTitle") or record.get("aiTitle") or result.title
        elif kind == "cost-state":
            result.lines_added = record.get("totalLinesAdded", result.lines_added)
            result.lines_removed = record.get("totalLinesRemoved", result.lines_removed)
    return result

