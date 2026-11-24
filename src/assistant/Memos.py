"""memory/memos/ — what the assistant knows but need not have in front of it, one markdown file per memo.

A memo is lookup material: a trip's bookings, a list of wines, a round of test results. It is
needed when its subject comes up and is dead weight in every other turn, so it lives outside the
memory bank: the prompt carries only its index line and `load_memo` loads the text (the
mechanics are `MemoryDocs`). Unlike a procedure a memo is kept current — `save_memo` replaces
it, `delete_memo` removes it, and the memory curator may do both.
"""
from assistant.MemoryDocs import MemoryDocs
from common.Paths import CACHE_DIR, MEMORY_DIR, MEMORY_LOG_DIR


class Memos(MemoryDocs):
    DIR = MEMORY_DIR / "memos"
    LOG_DIR = MEMORY_LOG_DIR / "memos"
    USED_FILE = CACHE_DIR / "memos_used.json"
    NOUN = "memo"
    PLURAL = "memos"
    LOAD = "load_memo"
    SAVE = "save_memo"
    DELETE = "delete_memo"
    COMMANDS = (SAVE, DELETE)
    HEADER_PREFIX = "# Memo: "
    EMPTY = "No memos saved yet."
