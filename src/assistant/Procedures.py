"""memory/procedures/ — the how-tos the assistant has learnt, one markdown file per procedure.

A procedure is a sequence that worked: how to log in somewhere, how to pay, how to read the
pool display. Every step and literal matters, so a procedure is never condensed: the prompt
carries only its index line and `load_procedure` loads the body when it is needed (the
mechanics are `MemoryDocs`). `save_procedure` replaces a whole procedure and `delete_procedure`
removes one. The memory curator alone appends dated notes (`procedure_note`): it may add to a
procedure, never rewrite it — it runs the commands with `allow_replace=False`.
"""
from datetime import date
from pathlib import Path

from assistant.MemoryDocs import MemoryDocs
from common.logging import info
from common.Paths import MEMORY_DIR, MEMORY_LOG_DIR
from common.vault.Redactor import redact_for_storage


class Procedures(MemoryDocs):
    DIR = MEMORY_DIR / "procedures"
    LOG_DIR = MEMORY_LOG_DIR / "procedures"
    NOUN = "procedure"
    PLURAL = "procedures"
    LOAD = "load_procedure"
    SAVE = "save_procedure"
    DELETE = "delete_procedure"
    NOTE = "procedure_note"
    COMMANDS = (SAVE, NOTE, DELETE)
    HEADER_PREFIX = "# Procedure: "
    EMPTY = "No procedures saved yet."
    REPLACE_HINT = f" — add to it with {NOTE}"
    NOTES_HEADING = "## Notes"

    @classmethod
    def note(cls, name: str, text: str, directory: Path = None) -> str:
        """Append a dated lesson under the procedure's notes. Returns '' or the reason it was refused."""
        directory = directory or cls.DIR
        if not text.strip():
            return f"{cls.NOTE} failed: the note is empty"
        text = redact_for_storage(text)
        with cls.lock:
            current = cls.read(name, directory)
            if current is None:
                return cls._unknown(cls.NOTE, name, directory)
            addition = f"- {date.today().isoformat()}: {text.strip()}\n"
            if cls.NOTES_HEADING not in current:
                addition = f"\n{cls.NOTES_HEADING}\n{addition}"
            if len(current) + len(addition) > cls.MAX_FILE_CHARS:
                info(f"Procedures: note to '{cls.slug(name)}' refused — it would exceed {cls.MAX_FILE_CHARS} chars")
                return (f"{cls.NOTE} failed: '{cls.slug(name)}' would exceed {cls.MAX_FILE_CHARS} chars — "
                        f"rewrite it with {cls.SAVE}, folding the notes into the steps")
            cls._path(name, directory).write_text(current.rstrip("\n") + "\n" + addition, encoding="utf-8")
        info(f"Procedures: note added to '{cls.slug(name)}'")
        return ""

    @classmethod
    def run(cls, command: str, args: list[str], directory: Path = None, log_dir: Path = None,
            allow_replace: bool = True) -> str:
        if command == cls.NOTE:
            if len(args) < 2:
                return f"{cls.NOTE} failed: expected 'name, text' parameters"
            return cls.note(args[0], ", ".join(args[1:]), directory)
        return super().run(command, args, directory, log_dir, allow_replace)
