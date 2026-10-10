"""Plejd's site and mesh states as the HomeAutomation data model, pure functions. The MAC of
an advertising unit comes out of Plejd's manufacturer data (CoreBluetooth never shows an
address, and pyplejd keys its hardware by the MAC the cloud lists); a light's state reads as
on / on 80% / off on Plejd's 0–255 dim scale and a cover's (a blind, a zipscreen, an awning)
as open / 40% open / closed / opening / closing on pyplejd's 0–100 position scale; the visible
lights, relays and covers fill the LIGHTS list and the visible scenes the ACTIONS list; and
the name index turns the names the LLM hands back — `name (room)` as `from_display` gives it,
or the bare name — into devices."""
import difflib
import re
from typing import Optional

from assistant.integration.HomeAutomation import Entity, HomeAutomation, HomeSnapshot

MANUFACTURER_ID = 887           # Plejd's Bluetooth SIG company id; the unit's MAC sits in that advertisement data
LIGHT_TYPES = ("LIGHT", "RELAY")
COVER_TYPE = "COVERABLE"        # pyplejd's outputType for a PlejdCover: a blind, a zipscreen, an awning
SCENE_TYPE = "SCENE"
_SPACES = re.compile(r"\s+")
_PAREN_SUFFIX = re.compile(r"^(.*?)\s*\(([^()]*)\)\s*$")


class Unresolved(Exception):
    """A name the index cannot turn into one device; the message is for the LLM."""


def mac_from_advertisement(manufacturer_data: dict) -> Optional[str]:
    """The MAC of the advertising unit the way pyplejd keys its hardware ("AABBCCDDEEFF"):
    bytes 4–10 of Plejd's manufacturer data, reversed. None for anything else on the air."""
    data = manufacturer_data.get(MANUFACTURER_ID)
    if not data or len(data) < 10:
        return None
    return bytes(data[4:10])[::-1].hex().upper()


def to_plejd_level(percent: int) -> int:
    """A dim percentage on Plejd's 0–255 scale, never 0 for a level above zero (the command
    turns 0 into off)."""
    return max(1, round(min(max(int(percent), 0), 100) * 255 / 100))


def value_of(state: dict) -> str:
    """How a light reads in the prompt: "on 80%", "on", "off", or "unknown" before the mesh
    reported it. The dim is on the 0–255 scale whichever way it arrived (a light level poll
    hands it over as a float, an event as an int)."""
    if "state" not in state:
        return "unknown"
    if not state["state"]:
        return "off"
    dim = state.get("dim")
    if dim is None:
        return "on"
    return f"on {min(100, round(float(dim) / 255 * 100))}%"


def cover_value_of(state: dict) -> str:
    """How a cover reads in the prompt: "opening" / "closing" while the motor runs, else
    "open", "closed" or "40% open" (100 is fully open, as pyplejd and Home Assistant count),
    and "unknown" before the mesh reported it or when pyplejd lost track of the position."""
    if "moving" not in state and "position" not in state:
        return "unknown"
    if state.get("moving"):
        return "opening" if state.get("opening") else "closing"
    position = state.get("position")
    if position is None:
        return "unknown"
    position = min(100, max(0, round(float(position))))
    if position >= 100:
        return "open"
    if position <= 0:
        return "closed"
    return f"{position}% open"


def kind_of(device) -> str:
    """LIGHT, RELAY, COVERABLE, SCENE, ... — pyplejd's outputType as text."""
    return str(getattr(device, "outputType", "") or "")


def is_cover(device) -> bool:
    return kind_of(device) == COVER_TYPE


def state_text(device, state: dict) -> str:
    """The device's state as the prompt shows it, by what the device is."""
    return cover_value_of(state) if is_cover(device) else value_of(state)


def room_of(device) -> Optional[str]:
    room = getattr(device, "roomData", None)
    title = getattr(room, "title", None) if room is not None else None
    return title.strip() if title and title.strip() else None


class NameIndex:
    """Name → device for the commands. Every device is reachable as `name (room)` (what
    `from_display` hands the backend for the `"name [Room]"` the prompt showed) and as its
    bare name when that is unique."""

    def __init__(self):
        self._holders: dict[str, list] = {}     # normalized key → every (device, display) behind it

    def add(self, device, name: str, room: Optional[str]):
        display = HomeAutomation.to_display(f"{name} ({room})" if room else name)
        self._put(name, device, display)
        if room:
            self._put(f"{name} ({room})", device, display)

    def _put(self, key: str, device, display: str):
        holders = self._holders.setdefault(normalize(key), [])
        if all(held is not device for held, _ in holders):
            holders.append((device, display))

    def resolve(self, name: str):
        """(device, its display name) behind *name*, else Unresolved with the text for the LLM."""
        key = normalize(name)
        if not key:
            raise Unresolved("No name given.")
        for candidate in self._candidates(key):
            holders = self._holders.get(candidate)
            if holders:
                if len(holders) == 1:
                    return holders[0]
                raise Unresolved(f'"{name}" could be {", ".join(d for _, d in holders)} — say which.')
        unique = [k for k, h in self._holders.items() if len(h) == 1]
        close = difflib.get_close_matches(key, unique, n=3, cutoff=0.6)
        suggestions = sorted({self._holders[k][0][1] for k in close})
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise Unresolved(f'No Plejd light, relay, cover or scene named "{name}".{hint}')

    @staticmethod
    def _candidates(key: str) -> list[str]:
        """The key itself, then — for `name (place)` — the bare name: the LLM may have
        invented the place or translated it."""
        match = _PAREN_SUFFIX.match(key)
        if match and match.group(1):
            return [key, match.group(1).strip()]
        return [key]

    def __len__(self):
        return len({id(d) for holders in self._holders.values() for d, _ in holders})


def normalize(text: str) -> str:
    return _SPACES.sub(" ", HomeAutomation.from_display(text)).strip().lower()


def build(devices, states: dict) -> tuple[HomeSnapshot, NameIndex]:
    """The snapshot — the visible lights, relays and covers with their room as LIGHTS, the
    visible scenes as ACTIONS, each list sorted by name — and the index of what it lists.
    *states* is keyed by the device's `identifier`; a device absent from it reads as unknown."""
    snapshot, index = HomeSnapshot(), NameIndex()
    for device in devices:
        kind = kind_of(device)
        name = (getattr(device, "name", "") or "").strip()
        if not name or getattr(device, "hidden", False):
            continue
        if kind == SCENE_TYPE:
            snapshot.actions.append(Entity(name, "scene"))
            index.add(device, name, None)
        elif kind in LIGHT_TYPES or kind == COVER_TYPE:
            room = room_of(device)
            snapshot.lights.append(Entity(name, state_text(device, states.get(device.identifier, {})), room))
            index.add(device, name, room)
    snapshot.lights.sort(key=lambda e: (e.name.lower(), e.area or ""))
    snapshot.actions.sort(key=lambda e: e.name.lower())
    return snapshot, index
