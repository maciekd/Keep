"""memory/bank.md — the long-term memory bank — and the one lock every writer of it shares.

The bank holds what is durably true of the household: people, preferences, relationships,
contacts, standing rules. How-tos live in `Procedures`, things being tracked in `WatchList`;
the three together are the `memory/` directory.

Two things write the file: the curator's full rewrite (a read, minutes of LLM time, then a
write) and the `remember` command's append. The lock serialises the append with the curator's
final read-merge-write, so a note taken while the curator was thinking is carried over instead
of overwritten; `write` replaces the file atomically so a reader (the prompt's memory section,
the console's Memory tab) never sees a half-written bank.

The prompt shows the bank in two places (`split`): the part the curator last wrote, near the
top, and what `remember` appended since, lower down. A `remember` then leaves the top of the
prompt byte-identical, which keeps the LLM's prompt cache alive between curations. `write`
records where the curated part ends in a sidecar, `<bank>.curated`.

The bank never holds a literal think tag (`without_think_tags`, applied by both writers).
"""
import hashlib
import json
import os
import re
import shutil
import threading

from common.logging import info
from common.Paths import RES_DIR
from common.vault.Redactor import redact_for_storage

MEMORY_FILE = os.path.join("memory", "bank.md")
# Where the bank lived before the memory got a directory of its own; `seed` moves it over.
LEGACY_MEMORY_FILE = "memory.md"
# What a fresh install starts from: a placeholder heading plus the concepts the assistant
# may teach the user over time. The first curation rewrites it into a real bank.
FIRST_RUN_STUB = RES_DIR / "first_run_memory.md"
# What the stub opens with; a bank that still starts so has never been curated.
PLACEHOLDER_HEADING = "# Long-term memory bank (placeholder)"

# The tags a local model's server marks the thinking with, in any spacing and case.
THINK_TAG = re.compile(r'<\s*(/?)\s*think(?:ing)?\s*>', re.IGNORECASE)

lock = threading.RLock()


def without_think_tags(text: str) -> str:
    """`text` with every literal think tag put in words. The bank is part of every prompt, and
    a model that quotes one from it while thinking has closed its thinking as far as its server
    is concerned: the rest of the thinking then arrives as the reply (ds4, 2026-11-17 — a note
    about a reasoning bug cost the nightly curation four runs out of eight). The same for an
    opening tag quoted in an answer, which swallows what follows."""
    cleaned, count = THINK_TAG.subn(lambda tag: "think-close tag" if tag.group(1) else "think-open tag", text)
    if count:
        info(f"Memory bank: {count} literal think tag(s) put in words before storing")
    return cleaned


def _ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def seed(path: str = MEMORY_FILE, stub=FIRST_RUN_STUB, legacy: str = None) -> bool:
    """Copy the first-run stub into place when there is no memory bank yet. Runs at every
    start, so a wiped install (`//wipe`) comes back with the stub too. An install from before
    the `memory/` directory has its bank moved there first: `legacy` is where it lived, and
    only the real bank path looks for the real legacy file — seeding any other path never
    moves it. Returns whether it copied the stub."""
    if legacy is None and path == MEMORY_FILE:
        legacy = LEGACY_MEMORY_FILE
    with lock:
        if not os.path.exists(path) and legacy and os.path.exists(legacy):
            _ensure_parent(path)
            shutil.move(legacy, path)
            info(f"Memory bank moved from {legacy} to {path}")
        if os.path.exists(path):
            info(f"Memory bank {path} exists, nothing to seed.")
            return False
        if not os.path.exists(stub):
            info(f"No memory bank at {path} and no first-run stub at {stub}, starting with none.")
            return False
        _ensure_parent(path)
        shutil.copyfile(stub, path)
    info(f"No memory bank at {path}: seeded it from {stub}")
    return True


def read(path: str = MEMORY_FILE) -> str:
    with lock:
        if not os.path.exists(path):
            return ""
        with open(path, "r") as f:
            return f.read()


def is_placeholder(text: str) -> bool:
    """Whether `text` is still the first-run stub, i.e. no curation has rewritten it yet."""
    return text.startswith(PLACEHOLDER_HEADING)


def append(text: str, path: str = MEMORY_FILE):
    text = without_think_tags(redact_for_storage(text))
    with lock:
        _ensure_parent(path)
        with open(path, "a") as f:
            f.write("\n" + text.strip() + "\n")
    info(f"Appended {len(text)} chars to {path}")


def write(text: str, path: str = MEMORY_FILE):
    """Replace the whole file in one step (write a sibling temp file, then rename over)."""
    text = without_think_tags(redact_for_storage(text))
    with lock:
        _ensure_parent(path)
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, path)
        _mark_curated(path, text)


def _mark_path(path: str) -> str:
    return f"{path}.curated"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _mark_curated(path: str, text: str):
    with open(_mark_path(path), "w") as f:
        json.dump({"length": len(text), "sha256": _digest(text)}, f)


def _read_mark(path: str) -> dict | None:
    try:
        with open(_mark_path(path), "r") as f:
            mark = json.load(f)
        return mark if isinstance(mark.get("length"), int) and isinstance(mark.get("sha256"), str) else None
    except (FileNotFoundError, json.JSONDecodeError, AttributeError):
        return None


def curated_length(path: str = MEMORY_FILE) -> int | None:
    """Where the curated part of the bank ends, for a reader that only looks (the console's
    Bank tab): None for a bank without a valid mark, which `split` would count as curated
    whole. Unlike `split` it marks nothing."""
    with lock:
        text = read(path)
        mark = _read_mark(path)
    if mark is None or _digest(text[:mark["length"]]) != mark["sha256"]:
        return None
    return mark["length"]


def split(path: str = MEMORY_FILE) -> tuple[str, str]:
    """The bank as (curated, appended since): the text the last `write` left, and what
    `append` added after it. A bank without a valid mark — never curated here, or edited by
    hand above the mark — counts as curated whole and is marked as such, so the next append
    starts a fresh "since"."""
    with lock:
        if not os.path.exists(path):
            return "", ""
        text = read(path)
        mark = _read_mark(path)
        if mark is not None:
            curated = text[:mark["length"]]
            if len(curated) == mark["length"] and _digest(curated) == mark["sha256"]:
                return curated, text[mark["length"]:]
            info(f"Memory bank {path} no longer matches its curated mark (edited by hand?) — "
                 f"treating all {len(text)} chars as curated")
        else:
            info(f"Memory bank {path} has no curated mark yet — treating all {len(text)} chars as curated")
        _mark_curated(path, text)
        return text, ""
