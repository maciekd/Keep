"""The frames a streaming chat draft shows before it has any text, and the test that tells such a
placeholder from a message.

Keep's Telegram transport posts a reply bubble as a spinner frame the moment a turn starts and
edits the text into it as it streams. A feed reading a group through the owner's account gets
that bubble as a new message — the first frame, never the edit — so another Keep's draft in a
shared group would read as a message of one `█`. Both sides use this module: the transport
draws its frames and phase labels from it, the feed recognises them (the labels in Telegram's
HTML or in the Markdown Telethon renders a message's text as).
"""
import re

SPINNER_FRAMES_C64 = ("█", "░")
SPINNER_FRAMES_BRAILLE = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
SPINNER_FRAME_CHARS = frozenset(SPINNER_FRAMES_C64 + SPINNER_FRAMES_BRAILLE)

# The line a draft puts over its frame while it waits for something in particular; a detail
# in parentheses may follow ("Thinking (↓ 1.2k tokens · 45 t/s)…", "Reading (▓▓▓▓░░░░░░)…").
THINKING_LABEL = "Thinking"
PREFILL_LABEL = "Reading"
COMMAND_LABEL = "Running command"
COMMANDS_LABEL = "Running commands"
QUEUED_LABEL = "Waiting in queue"
PHASE_LABELS = (THINKING_LABEL, PREFILL_LABEL, QUEUED_LABEL, COMMAND_LABEL, COMMANDS_LABEL)

_MARKUP = re.compile(r"</?code>|`")


def progress_bar(fraction: float, width: int, *, fine: bool = True, full: str = "█", empty: str = "░") -> tuple[str, str]:
    """A `width`-cell bar as (filled, empty) text. `fine` adds an eighth-block for the
    remainder (terminal cells abut, so the console and the Dashboard use it); without it
    the bar moves in whole cells. Telegram draws the filled part in `▓`: its client places
    glyphs at fractional pixels, which leaves hairline seams between `█` blocks, while the
    shade glyphs cover their full advance and join seamlessly."""
    eighths = round(max(0.0, min(1.0, fraction)) * width * 8)
    full_cells, rem = divmod(eighths, 8)
    filled = full * full_cells
    if fine and rem and full_cells < width:
        filled += " ▏▎▍▌▋▊▉"[rem]
    return filled, empty * (width - len(filled))
_PHASE_LINE = re.compile(r"^(?:" + "|".join(re.escape(label) for label in PHASE_LABELS) + r")(?: \([^)]*\))?…$")


def is_draft_placeholder(text: str) -> bool:
    """True when *text* is nothing but a draft's placeholder: spinner frames, with or without a
    phase line over them, or a phase line alone (a label with a live count has no frame under
    it), in the HTML the transport sends or the Markdown Telethon reads back."""
    if not text:
        return False
    lines = [line.strip() for line in _MARKUP.sub("", text).splitlines() if line.strip()]
    if not lines:
        return False
    *labels, last = lines
    if not all(ch in SPINNER_FRAME_CHARS or ch.isspace() for ch in last):
        labels = lines
    return all(_PHASE_LINE.match(label) for label in labels)
