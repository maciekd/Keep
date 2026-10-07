"""
SystemNote — an event the household should see and the assistant should know about.

Integrations push things at the user outside a dialog turn (a 2FA challenge, a code
fished out of iMessage and submitted). A plain `notifier().send_message` reaches the chat
UIs but never the LLM, which then talks about the challenge as if nothing had happened.
`system_note` leaves the event as a `system` row in the dialog history — the Home tab
draws it as "System (<source>)" through its HistoryFeed and the LLM reads it on its next
turn — and pushes the same text to the chats that do not render the history (Telegram).
A routine note (voice came up as usual) passes `push=False` and stays a history row only.

The LLM sees a system row as "System (HH:MM): text" without the source, so the text has
to name the integration itself. The history is the process-wide one the launcher installs
with `set_history`; without one (a test, an external script) the note is only pushed.
"""
import time
from typing import Optional

from assistant.Notifier import notifier
from common.logging import info, trace

_history = None


def set_history(history) -> None:
    """Install the process-wide MessageHistory the notes are written to; None detaches."""
    global _history
    _history = history
    info(f"SystemNote history {'set' if history is not None else 'cleared'}")


def history():
    return _history


def system_note(source: str, text: str, timestamp: Optional[float] = None, push: bool = True,
                quiet: bool = False) -> bool:
    """Record *text* as a system row from *source* and push it to the chats that do not
    show the history — without a sound when *quiet*; with *push* off it is a history row
    only. Returns True when it reached the history or a chat."""
    text = text.strip()
    if not text:
        trace(f"SystemNote: empty note from {source}, skipped")
        return False
    stored = False
    if _history is None:
        info(f"SystemNote: no history installed, {source} note only pushed: {text[:120]!r}")
    else:
        _history.add_context_message("system", text, timestamp or time.time(), source=source)
        stored = True
        info(f"SystemNote: {source} → history: {text[:120]!r}")
    if not push:
        trace(f"SystemNote: {source} note is history-only, not pushed to the chats")
        return stored
    pushed = notifier().send_message(text, disable_notification=quiet, in_history=stored)
    return stored or pushed
