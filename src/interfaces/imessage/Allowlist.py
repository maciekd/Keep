"""Who the assistant's iMessage account serves: the handles on `imessage_chat.owners`, `guests` and `bots`.

An entry is a handle — a phone number in international form (`+46701740605`) or an email — with
the person's name in front of it when the assistant should know one: `Charlotte +46701740606`.
An owner's turn has every right the owner has anywhere; a guest's is the chat layer's guest
turn. `bots` marks handles that are another household's assistant: guests the chat layer
tells apart, and that wake this one only so often an hour.

A phone number is matched as a whole, never by its last digits, and one without the leading
`+` is refused: an entry decides who commands the house. The lists are read from the config
at every use, so a change is in force at the next message.
"""
import re
from dataclasses import dataclass
from typing import Optional

from common.ConfigManager import config
from common.imessage.Handles import normalize_phone, prefer_handle
from common.logging import info, note, trace

SECTION = "imessage_chat"
OWNERS_KEY = f"{SECTION}.owners"
GUESTS_KEY = f"{SECTION}.guests"
BOTS_KEY = f"{SECTION}.bots"

_PHONE_RE = re.compile(r"^(?P<name>.*?)\s*(?P<phone>\+[\d\s().-]{6,})$")
_MIN_PHONE_DIGITS = 7


def normalize(handle: Optional[str]) -> str:
    """A handle as the lists and chat.db are compared: an email in lower case, a phone as `+` and digits."""
    return prefer_handle((handle or "").strip()) or ""


@dataclass(frozen=True)
class Person:
    handle: str             # normalised
    name: str = ""          # as the entry gave it, "" when it gave none
    owner: bool = False
    bot: bool = False       # another household's assistant

    @property
    def guest(self) -> bool:
        return not self.owner


def parse_entry(entry) -> tuple[str, str]:
    """(name, handle) of one list entry; the handle is "" when the entry names none."""
    text = " ".join(str(entry or "").split())
    if "@" in text:
        *name, email = text.split(" ")
        email = email.strip("<>").lower()
        return (" ".join(name), email) if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email) else ("", "")
    match = _PHONE_RE.match(text)
    if not match:
        return "", ""
    phone = normalize_phone(match.group("phone"))
    if len(phone) <= _MIN_PHONE_DIGITS:
        return "", ""
    return match.group("name").strip(" :,<>"), phone


class Allowlist:
    """Who the assistant's iMessage account serves: the people of the three lists as config holds them now.

    Read again at every use; a change is logged once."""

    def __init__(self):
        self._seen: Optional[tuple] = None
        self._people: dict[str, Person] = {}
        self.problems: list[str] = []       # the entries that name nobody, in words for the owner

    @staticmethod
    def _raw(key: str) -> list:
        value = config.get(key, []) or []
        return list(value) if isinstance(value, (list, tuple)) else [value]

    def refresh(self) -> bool:
        """Read the lists again; True when they are not what they were at the last read."""
        raw = (self._raw(OWNERS_KEY), self._raw(GUESTS_KEY), self._raw(BOTS_KEY))
        seen = tuple(tuple(str(item) for item in part) for part in raw)
        if seen == self._seen:
            return False
        first = self._seen is None
        self._seen = seen
        people: dict[str, Person] = {}
        problems: list[str] = []
        for key, entries, owner, bot in ((OWNERS_KEY, raw[0], True, False), (GUESTS_KEY, raw[1], False, False),
                                         (BOTS_KEY, raw[2], False, True)):
            for index, entry in enumerate(entries):
                name, handle = parse_entry(entry)
                if not handle:
                    # an owner's entry is never echoed: what was typed there is the user's alone
                    problems.append(f"{key}[{index}] names no handle — an entry is a phone number starting with + "
                                    f"and the country code, or an email, with a name in front of it if wanted")
                    continue
                known = people.get(handle)
                if known is not None and known.owner:
                    info(f"iMessage chat: {key}[{index}] is an owner's handle already — it stays an owner")
                    continue
                people[handle] = Person(handle, name or (known.name if known else ""), owner, bot)
        self._people, self.problems = people, problems
        owners = sum(1 for p in people.values() if p.owner)
        bots = sum(1 for p in people.values() if p.bot)
        summary = f"{owners} owner(s), {len(people) - owners - bots} guest(s), {bots} other assistant(s)"
        if first:
            info(f"iMessage chat: serving {summary}")
        else:
            note(f"iMessage chat: the allowlist changed — now {summary}, in force from the next message")
        for problem in problems:
            info(f"iMessage chat: {problem}")
        return True

    def snapshot(self) -> tuple:
        """The three lists as config holds them now, as a value to compare: whoever acts on a
        change keeps the snapshot it acted on, since `refresh` answers True to one caller only."""
        self.refresh()
        return self._seen

    def find(self, handle: Optional[str]) -> Optional[Person]:
        self.refresh()
        person = self._people.get(normalize(handle))
        if person is None:
            trace(f"iMessage chat: {handle} is on no list")
        return person

    def people(self) -> list[Person]:
        self.refresh()
        return list(self._people.values())

    def __len__(self) -> int:
        self.refresh()
        return len(self._people)
