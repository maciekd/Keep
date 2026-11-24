"""A folder of named markdown files the assistant loads on demand — the base of `Procedures` and `Memos`.

Each file is a name, a `When:` line and a text. The prompt carries only an index line per file
(built here from the file, so nobody can erode it) and the load command brings the body when it
is needed. The index line is the name and the start of the `When:` line, as short as a watch
item's (`WHEN_CHARS`): a save with a longer `when` is refused, and a line an older file holds is
cut in the index alone. A save replaces a whole file and a delete removes one; a replaced or
deleted version is copied to the log folder first, which is a debugging trail the LLM never reads.

A subclass states its folder, its log folder, its words for itself and its command names. The
commands and the memory curator's command block both go through `run`.
"""
import re
import threading
from datetime import datetime
from pathlib import Path

from assistant import WatchList
from common.logging import info, trace
from common.vault.Redactor import redact_for_storage


class MemoryDocs:
    DIR: Path = None
    LOG_DIR: Path = None
    NOUN = ""
    PLURAL = ""
    LOAD = ""
    SAVE = ""
    DELETE = ""
    # The header of a loaded file, as the load command returns it.
    HEADER_PREFIX = ""
    EMPTY = ""
    # What a refused replace says to do instead.
    REPLACE_HINT = ""
    # A file comes back whole as a command result and stays in the history: keep it bounded.
    MAX_FILE_CHARS = 10_000
    WHEN_PREFIX = "When:"
    # The longest `when` a save may write, and how much of one the index carries: every file costs
    # its index line in every prompt, so the line is as short as a watch item's.
    WHEN_CHARS = WatchList.INDEX_NOTE_CHARS

    lock = threading.RLock()

    @staticmethod
    def slug(name: str) -> str:
        """The file name a document answers to: 'VKlass login' and "vklass-login" are the same one."""
        return re.sub(r"[^a-z0-9]+", "-", name.strip().strip("\"'`").lower()).strip("-")

    @classmethod
    def _label(cls) -> str:
        return cls.PLURAL.capitalize()

    @classmethod
    def _path(cls, name: str, directory: Path) -> Path:
        return directory / f"{cls.slug(name)}.md"

    @classmethod
    def names(cls, directory: Path = None) -> list[str]:
        directory = directory or cls.DIR
        if not directory.is_dir():
            return []
        return sorted(path.stem for path in directory.glob("*.md"))

    @classmethod
    def read(cls, name: str, directory: Path = None) -> str | None:
        """The file, or None when there is none by that name."""
        directory = directory or cls.DIR
        if not cls.slug(name):
            return None
        with cls.lock:
            try:
                return cls._path(name, directory).read_text(encoding="utf-8")
            except OSError:
                return None

    @classmethod
    def when(cls, text: str) -> str:
        """The file's `When:` line — what the index says about it."""
        for line in text.splitlines()[:5]:
            if line.startswith(cls.WHEN_PREFIX):
                return cls._unlabelled(line)
        return ""

    @classmethod
    def _unlabelled(cls, when_text: str) -> str:
        """`when` without the `When:` label: a model that has read a file writes the label into the value."""
        when_text = " ".join(when_text.split())
        while when_text.lower().startswith(cls.WHEN_PREFIX.lower()):
            when_text = when_text[len(cls.WHEN_PREFIX):].strip()
        return when_text

    @classmethod
    def _log_copy(cls, name: str, text: str, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        (log_dir / f"{cls.slug(name)}.{stamp}.md").write_text(text, encoding="utf-8")

    @classmethod
    def save(cls, name: str, when_text: str, text: str, directory: Path = None, log_dir: Path = None,
             allow_replace: bool = True) -> str:
        """Create the file, or replace it whole. Returns '' or the reason it was refused."""
        directory = directory or cls.DIR
        slug = cls.slug(name)
        if not slug:
            return f"{cls.SAVE} failed: the name must contain letters or digits"
        if not text.strip():
            return f"{cls.SAVE} failed: the {cls.NOUN} text is empty"
        labelled, when_text = when_text, cls._unlabelled(when_text)
        if labelled.split() != when_text.split():
            trace(f"{cls._label()}: '{slug}' — the `{cls.WHEN_PREFIX}` label was written into `when`, dropped")
        if len(when_text) > cls.WHEN_CHARS:
            info(f"{cls._label()}: save of '{slug}' refused — `when` is {len(when_text)} chars, limit {cls.WHEN_CHARS}")
            return (f"{cls.SAVE} failed: `when` is {len(when_text)} chars, the limit is {cls.WHEN_CHARS} — "
                    "shorten it and save again")
        content = redact_for_storage(f"# {name.strip()}\n{cls.WHEN_PREFIX} {when_text}\n\n{text.strip()}\n")
        if len(content) > cls.MAX_FILE_CHARS:
            info(f"{cls._label()}: save of '{slug}' refused — {len(content)} chars, limit {cls.MAX_FILE_CHARS}")
            return (f"{cls.SAVE} failed: {len(content)} chars, the limit is {cls.MAX_FILE_CHARS} — "
                    f"split it into two {cls.PLURAL}")
        with cls.lock:
            previous = cls.read(name, directory)
            if previous is not None and not allow_replace:
                info(f"{cls._label()}: refused to replace '{slug}' (replacing is not allowed here)")
                return f"{cls.SAVE} failed: '{slug}' exists and may not be replaced here{cls.REPLACE_HINT}"
            if previous is not None:
                cls._log_copy(name, previous, log_dir or cls.LOG_DIR)
            directory.mkdir(parents=True, exist_ok=True)
            cls._path(name, directory).write_text(content, encoding="utf-8")
        if previous is None:
            info(f"{cls._label()}: '{slug}' created ({len(content.split())} words)")
        else:
            info(f"{cls._label()}: '{slug}' replaced ({len(previous.split())} → {len(content.split())} words, "
                 "previous version in the log)")
        return ""

    @classmethod
    def delete(cls, name: str, directory: Path = None, log_dir: Path = None) -> str:
        directory = directory or cls.DIR
        with cls.lock:
            current = cls.read(name, directory)
            if current is None:
                return cls._unknown(cls.DELETE, name, directory)
            cls._log_copy(name, current, log_dir or cls.LOG_DIR)
            cls._path(name, directory).unlink()
        info(f"{cls._label()}: '{cls.slug(name)}' deleted (last version in the log)")
        return ""

    @classmethod
    def _unknown(cls, command: str, name: str, directory: Path) -> str:
        available = ", ".join(cls.names(directory)) or "none"
        return f"{command} failed: no {cls.NOUN} named '{cls.slug(name) or name.strip()}'. Available: {available}"

    @classmethod
    def load(cls, name: str, directory: Path = None) -> str:
        """What the load command answers: the file under its header, or what exists instead."""
        text = cls.read(name, directory)
        if text is None:
            info(f"{cls._label()}: '{name.strip()}' asked for, no such {cls.NOUN}")
            return cls._unknown(cls.LOAD, name, directory or cls.DIR)
        trace(f"{cls._label()}: '{cls.slug(name)}' loaded ({len(text)} chars)")
        return f"{cls.HEADER_PREFIX}{cls.slug(name)}\n{text.rstrip()}\n"

    @classmethod
    def load_all(cls, names: list[str]) -> str:
        """What the load command answers for the names it was given, each file under its header."""
        names = [n for n in names if n and n.strip()]
        if not names:
            return f"{cls.LOAD} failed: expected a {cls.NOUN} name. Available: " + (", ".join(cls.names()) or "none")
        return "\n\n".join(cls.load(name) for name in names)

    @classmethod
    def render_index(cls, directory: Path = None) -> str:
        """The index the prompt carries: one line per file, from the files themselves — the name
        and the start of its `When:` line."""
        directory = directory or cls.DIR
        lines = []
        for name in cls.names(directory):
            summary = WatchList.index_start(cls.when(cls.read(name, directory) or ""))
            lines.append(f"- `{name}`" + (f" — {summary}" if summary else ""))
        return "\n".join(lines) or cls.EMPTY

    @classmethod
    def version(cls, directory: Path = None) -> tuple:
        """Changes whenever the index would."""
        directory = directory or cls.DIR
        try:
            return tuple(sorted((p.name, p.stat().st_mtime_ns) for p in directory.glob("*.md")))
        except OSError:
            return ()

    @classmethod
    def run(cls, command: str, args: list[str], directory: Path = None, log_dir: Path = None,
            allow_replace: bool = True) -> str:
        """One writing command with the arguments as the command parser split them. Returns ''
        when it ran and the reason when it could not — the error the LLM is told. Without
        `allow_replace` an existing file can be neither replaced nor deleted."""
        if command == cls.SAVE:
            if len(args) < 3:
                return f"{cls.SAVE} failed: expected 'name, when, text' parameters"
            # An unquoted text with commas in it arrives split — the rest of the args are the text.
            return cls.save(args[0], args[1], ", ".join(args[2:]), directory, log_dir, allow_replace)
        if command == cls.DELETE:
            if not args or not args[0].strip():
                return f"{cls.DELETE} failed: expected the {cls.NOUN}'s name"
            if not allow_replace:
                info(f"{cls._label()}: refused to delete '{cls.slug(args[0])}' (deleting is not allowed here)")
                return f"{cls.DELETE} failed: {cls.PLURAL} may not be deleted here"
            return cls.delete(args[0], directory, log_dir)
        return f"{command} is not a {cls.NOUN} command"
