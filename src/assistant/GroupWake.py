"""The group wake rule: which message of a group chat is said to the assistant, for every chat platform.

In a group the assistant answers a message that replies to one of its own bubbles, one that
names it (the wake word or an alias, as a whole word), and the one that comes right after a
message of its own — people forget the name and the reply gesture when they answer. The rest
is read as context. Two consumers decide by it: the Telegram feed (the owner's account sees
the groups the bot is in; the bot itself is handed only the replies to its bubbles) and the
iMessage chat transport (the assistant's own account sees every message, so it decides alone).

What is shared is the decision, not the platform: `reason` says why a message wakes, `brake`
is the hourly cap that stops two assistants answering each other for ever, and `settle`
keeps who wrote last in each group. The "right after" rule has no time limit and reads that
last sender; every live message moves it but one the chat layer refused to take as a turn (a
guest's while a reply was in flight): the group never saw that one answered, so the next
message still follows the assistant. A consumer keeps one `GroupWake` and names its groups
and senders by whatever ids its platform has.
"""
import re
import time
from collections import deque
from typing import Callable, Hashable, Iterable, Optional

from common.ConfigManager import config
from common.logging import trace

TURN, ROW, REFUSED = "turn", "row", "refused"       # what became of a live group message
REPLY, NAMED, FOLLOWS = "reply", "named", "follows"     # why a message wakes the assistant
BOT_WAKE_CAP = 6            # wakes by another assistant per group per hour: the brake if two of them loop
BOT_WAKE_WINDOW = 3600.0

HOW = {REPLY: "replied to one of the assistant's own messages", NAMED: "named the assistant",
       FOLLOWS: "wrote right after the assistant's own message"}


def assistant_names(source=None) -> list[str]:
    """The names the assistant answers to: `wake_word` and `wake_word_aliases`. *source* is
    the config to read them from, the process's own when left out."""
    source = source or config
    return [n for n in (source.get('wake_word', 'Michelle'), *(source.get('wake_word_aliases', []) or [])) if n]


def names_assistant(text: str, names: Optional[Iterable[str]] = None) -> bool:
    """Whether *text* says one of the assistant's names as a whole word, in any case."""
    names = assistant_names() if names is None else [n for n in names if n]
    if not names:
        trace("GroupWake: no wake word configured, no name can wake the assistant in a group")
        return False
    pattern = r"\b(" + "|".join(re.escape(n) for n in names) + r")\b"
    return re.search(pattern, text or "", re.IGNORECASE) is not None


class GroupWake:
    """The bookkeeping of one consumer: who wrote last in each group, and how often the capped
    senders woke the assistant there."""

    def __init__(self, clock: Callable[[], float] = time.time):
        self._clock = clock
        self.last_sender: dict = {}     # group → the sender of its latest live message
        self._wakes: dict = {}          # brake key → the times it let a wake through

    def reason(self, group: Hashable, text: str, own: Hashable, names: Optional[Iterable[str]] = None,
               replies_to_own: bool = False) -> str:
        """Why a message with this text, just written in *group* by someone else, wakes the
        assistant — REPLY, NAMED or FOLLOWS — or "" when it does not. *own* is the id the
        assistant's own messages are settled under (None: it has none here, nothing follows
        it); a message without text wakes nobody."""
        if not (text or "").strip():
            return ""
        if replies_to_own:
            return REPLY
        if names_assistant(text, names):
            return NAMED
        return FOLLOWS if self.follows(group, own) else ""

    def brake(self, key: Hashable, cap: int = BOT_WAKE_CAP, window: float = BOT_WAKE_WINDOW) -> bool:
        """Whether one more wake under *key* (a group, for the assistants in it) is within
        *cap* per *window*; a yes is counted."""
        recent = self._wakes.setdefault(key, deque())
        now = self._clock()
        while recent and now - recent[0] > window:
            recent.popleft()
        if len(recent) >= cap:
            return False
        recent.append(now)
        return True

    def settle(self, group: Hashable, sender: Hashable, outcome: str = ROW) -> None:
        """A live message from *sender* in *group* came to *outcome*: it is the group's latest
        now, unless the chat layer refused it."""
        if outcome == REFUSED:
            trace(f"GroupWake: [{group}] the refused turn from {sender} leaves the last sender as is")
            return
        self.last_sender[group] = sender

    def spoke(self, group: Hashable, own: Hashable) -> None:
        """The assistant's own message is the latest in *group*: the next one follows it."""
        self.last_sender[group] = own

    def follows(self, group: Hashable, own: Hashable) -> bool:
        return own is not None and self.last_sender.get(group) == own

    def following(self, own: Hashable) -> list:
        """The groups whose latest message is the assistant's own."""
        return [group for group, sender in self.last_sender.items() if sender == own]
