"""memory/procedures/ — the how-tos the assistant has learnt, one markdown file per procedure.

A procedure is a sequence that worked: how to log in somewhere, how to pay, how to read the
pool display. Every step and literal matters, so a procedure is never condensed: the prompt
carries only an index line per file (built here from the file, so nobody can erode it) and the
`load_procedure` command loads the body when it is needed. The index line is the name and the
start of the `When:` line, as short as a watch item's (`WHEN_CHARS`): a save with a longer
`when` is refused, and a line an older file holds is cut in the index alone. `save_procedure` replaces a whole
procedure and `delete_procedure` removes one; a replaced or deleted version is copied to
`log/memory/procedures/` first, which is a debugging trail the LLM never reads. The memory
curator alone appends dated notes (`procedure_note`): it may add to a procedure, never rewrite it.

The commands and the memory curator's command block both go through `run`; the curator is not
allowed to replace or delete (`allow_replace=False`).
"""
import re
import threading
from datetime import date, datetime
from pathlib import Path

from assistant import WatchList
from common.logging import info, trace
from common.Paths import MEMORY_DIR, MEMORY_LOG_DIR
from common.vault.Redactor import redact_for_storage

PROCEDURES_DIR = MEMORY_DIR / "procedures"
LOG_DIR = MEMORY_LOG_DIR / "procedures"
COMMANDS = ("save_procedure", "procedure_note", "delete_procedure")
# A procedure comes back whole as a command result and stays in the history: keep it bounded.
MAX_FILE_CHARS = 10_000
# The header of a loaded procedure, as `load_procedure(name)` returns it.
HEADER_PREFIX = "# Procedure: "
NOTES_HEADING = "## Notes"
WHEN_PREFIX = "When:"
# The longest `when` a save may write, and how much of one the index carries: every procedure costs
# its index line in every prompt, so the line is as short as a watch item's.
WHEN_CHARS = WatchList.INDEX_NOTE_CHARS
EMPTY = "No procedures saved yet."

lock = threading.RLock()


def slug(name: str) -> str:
    """The file name a procedure answers to: 'VKlass login' and "vklass-login" are the same one."""
    return re.sub(r"[^a-z0-9]+", "-", name.strip().strip("\"'`").lower()).strip("-")


def _path(name: str, directory: Path) -> Path:
    return directory / f"{slug(name)}.md"


def names(directory: Path = None) -> list[str]:
    directory = directory or PROCEDURES_DIR
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob("*.md"))


def read(name: str, directory: Path = None) -> str | None:
    """The procedure's file, or None when there is none by that name."""
    directory = directory or PROCEDURES_DIR
    if not slug(name):
        return None
    path = _path(name, directory)
    with lock:
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return None


def when(text: str) -> str:
    """The procedure's `When:` line — what the index says about it."""
    for line in text.splitlines()[:5]:
        if line.startswith(WHEN_PREFIX):
            return line[len(WHEN_PREFIX):].strip()
    return ""


def _log_copy(name: str, text: str, log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    (log_dir / f"{slug(name)}.{stamp}.md").write_text(text, encoding="utf-8")


def save(name: str, when_text: str, text: str, directory: Path = None, log_dir: Path = None,
         allow_replace: bool = True) -> str:
    """Create the procedure, or replace it whole. Returns '' or the reason it was refused."""
    directory = directory or PROCEDURES_DIR
    if not slug(name):
        return "save_procedure failed: the name must contain letters or digits"
    if not text.strip():
        return "save_procedure failed: the procedure text is empty"
    when_text = " ".join(when_text.split())
    if len(when_text) > WHEN_CHARS:
        info(f"Procedures: save of '{slug(name)}' refused — `when` is {len(when_text)} chars, limit {WHEN_CHARS}")
        return (f"save_procedure failed: `when` is {len(when_text)} chars, the limit is {WHEN_CHARS} — "
                "shorten it and save again")
    content = redact_for_storage(f"# {name.strip()}\n{WHEN_PREFIX} {when_text}\n\n{text.strip()}\n")
    if len(content) > MAX_FILE_CHARS:
        return (f"save_procedure failed: {len(content)} chars, the limit is {MAX_FILE_CHARS} — "
                "split it into two procedures")
    with lock:
        previous = read(name, directory)
        if previous is not None and not allow_replace:
            info(f"Procedures: refused to replace '{slug(name)}' (replacing is not allowed here)")
            return (f"save_procedure failed: '{slug(name)}' exists and may not be replaced here — "
                    "add to it with procedure_note")
        if previous is not None:
            _log_copy(name, previous, log_dir or LOG_DIR)
        directory.mkdir(parents=True, exist_ok=True)
        _path(name, directory).write_text(content, encoding="utf-8")
    if previous is None:
        info(f"Procedures: '{slug(name)}' created ({len(content.split())} words)")
    else:
        info(f"Procedures: '{slug(name)}' replaced ({len(previous.split())} → {len(content.split())} words, "
             "previous version in the log)")
    return ""


def note(name: str, text: str, directory: Path = None) -> str:
    """Append a dated lesson under the procedure's notes. Returns '' or the reason it was refused."""
    directory = directory or PROCEDURES_DIR
    if not text.strip():
        return "procedure_note failed: the note is empty"
    text = redact_for_storage(text)
    with lock:
        current = read(name, directory)
        if current is None:
            return _unknown("procedure_note", name, directory)
        addition = f"- {date.today().isoformat()}: {text.strip()}\n"
        if NOTES_HEADING not in current:
            addition = f"\n{NOTES_HEADING}\n{addition}"
        if len(current) + len(addition) > MAX_FILE_CHARS:
            return (f"procedure_note failed: '{slug(name)}' would exceed {MAX_FILE_CHARS} chars — "
                    "rewrite it with save_procedure, folding the notes into the steps")
        _path(name, directory).write_text(current.rstrip("\n") + "\n" + addition, encoding="utf-8")
    info(f"Procedures: note added to '{slug(name)}'")
    return ""


def delete(name: str, directory: Path = None, log_dir: Path = None) -> str:
    directory = directory or PROCEDURES_DIR
    with lock:
        current = read(name, directory)
        if current is None:
            return _unknown("delete_procedure", name, directory)
        _log_copy(name, current, log_dir or LOG_DIR)
        _path(name, directory).unlink()
    info(f"Procedures: '{slug(name)}' deleted (last version in the log)")
    return ""


def _unknown(command: str, name: str, directory: Path) -> str:
    available = ", ".join(names(directory)) or "none"
    return f"{command} failed: no procedure named '{slug(name) or name.strip()}'. Available: {available}"


def load(name: str, directory: Path = None) -> str:
    """What `load_procedure(name)` answers: the procedure under its header, or what exists instead."""
    text = read(name, directory)
    if text is None:
        info(f"Procedures: '{name.strip()}' asked for, no such procedure")
        return _unknown("load_procedure", name, directory or PROCEDURES_DIR)
    trace(f"Procedures: '{slug(name)}' loaded ({len(text)} chars)")
    return f"{HEADER_PREFIX}{slug(name)}\n{text.rstrip()}\n"


def render_index(directory: Path = None) -> str:
    """The `{PROCEDURES}` text: one line per procedure, from the files themselves — the name and
    the start of its `When:` line."""
    directory = directory or PROCEDURES_DIR
    lines = []
    for name in names(directory):
        summary = WatchList.index_start(when(read(name, directory) or ""))
        lines.append(f"- `{name}`" + (f" — {summary}" if summary else ""))
    return "\n".join(lines) or EMPTY


def version(directory: Path = None) -> tuple:
    """Changes whenever the index would."""
    directory = directory or PROCEDURES_DIR
    try:
        return tuple(sorted((p.name, p.stat().st_mtime_ns) for p in directory.glob("*.md")))
    except OSError:
        return ()


def run(command: str, args: list[str], directory: Path = None, log_dir: Path = None,
        allow_replace: bool = True) -> str:
    """One procedure-writing command with the arguments as the command parser split them.
    Returns '' when it ran and the reason when it could not — the error the LLM is told.
    Without `allow_replace` an existing procedure can be added to but not replaced or deleted."""
    if command == "save_procedure":
        if len(args) < 3:
            return "save_procedure failed: expected 'name, when, text' parameters"
        # An unquoted text with commas in it arrives split — the rest of the args are the text.
        return save(args[0], args[1], ", ".join(args[2:]), directory, log_dir, allow_replace)
    if command == "procedure_note":
        if len(args) < 2:
            return "procedure_note failed: expected 'name, text' parameters"
        return note(args[0], ", ".join(args[1:]), directory)
    if command == "delete_procedure":
        if not args or not args[0].strip():
            return "delete_procedure failed: expected the procedure's name"
        if not allow_replace:
            info(f"Procedures: refused to delete '{slug(args[0])}' (deleting is not allowed here)")
            return "delete_procedure failed: procedures may not be deleted here"
        return delete(args[0], directory, log_dir)
    return f"{command} is not a procedure command"
