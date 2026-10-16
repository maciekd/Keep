"""HomeClaw's accessories, scenes and events as the HomeAutomation data model, pure functions.
An accessory is one entity, named as the app names it with its room as the area — twins (three
"Aqara TempHumidity Sensor T1" in one room) are numbered #1, #2, #3 in the order of their UUIDs,
so a name stays the same accessory from one snapshot to the next. Lights, covers, the garage
door and the irrigation zones fill the LIGHTS list with a state the house commands act on;
sensors, buttons and the thermostat fill SENSORS with one line of readings; the named scenes
fill ACTIONS (the app also lists HomeKit's own action sets under a bare UUID — those are
hidden). The accessories of a bridge another integration owns (`homeclaw.exclude`) are left
out, and the name index turns the names the LLM hands back into targets."""
import difflib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from assistant.integration.Feed import FeedEntry
from assistant.integration.HomeAutomation import Entity, HomeAutomation, HomeSnapshot

VALVE_SERVICE = "000000D0-0000-1000-8000-0026BB765291"    # HomeKit's Valve service: one per zone of an irrigation control
LIGHT, COVER, GARAGE, IRRIGATION, VALVE = "lightbulb", "window_covering", "garage_door", "irrigation", "valve"
THERMOSTAT, SENSOR, BUTTON, SCENE, BRIDGE = "thermostat", "sensor", "programmable_switch", "scene", "bridge"
SWITCHED = (LIGHT, "switch", "outlet", "fan")           # categories whose `power` the house commands switch
DEFAULT_EXCLUDE = ("Plejd", "Signify", "Philips Hue")   # the makers Keep's own integrations already list
SOIL_MAKERS = ("gardena",)                              # makers whose humidity reading is the soil's moisture
UNREAD = "--"                                           # the app's value for a characteristic it has not read
_TRUE = {"true", "1", "on", "yes"}
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_SPACES = re.compile(r"\s+")
_PAREN_SUFFIX = re.compile(r"^(.*?)\s*\(([^()]*)\)\s*$")


class Unresolved(Exception):
    """A name the index cannot turn into one target; the message is for the LLM."""


@dataclass(frozen=True)
class Target:
    """What a name resolves to: an accessory (with the service a control must pick, for an
    irrigation zone) or a scene, with the state the last snapshot showed."""
    id: str
    kind: str                           # the category, VALVE for a zone, SCENE for a scene
    display: str                        # as the prompt shows it: "Name [Room]"
    service: Optional[str] = None
    state: dict = field(default_factory=dict)


def clean(text) -> str:
    """One space between words, no non-breaking spaces (the app passes them on from HomeKit)."""
    return _SPACES.sub(" ", str(text or "")).strip()


def is_read(value) -> bool:
    return value is not None and str(value).strip() not in ("", UNREAD)


def excluded(accessory: dict, filters) -> bool:
    """Whether the accessory, or the bridge it hangs on, matches one of the exclusion strings
    (case-insensitive substrings of the maker, the bridge's maker, model or name)."""
    bridge = accessory.get("bridge") or {}
    haystack = " | ".join(str(accessory.get(key) or "") for key in ("manufacturer",)) + " | " + \
        " | ".join(str(bridge.get(key) or "") for key in ("manufacturer", "model", "name"))
    haystack = haystack.lower()
    return any(str(f).strip() and str(f).strip().lower() in haystack for f in filters or ())


def is_hidden_scene(scene: dict) -> bool:
    """HomeKit's own action sets come through named by their UUID; nobody asks for those."""
    return bool(_UUID.match(clean(scene.get("name"))))


def percent(value) -> Optional[int]:
    try:
        return min(100, max(0, round(float(str(value).rstrip("%")))))
    except (TypeError, ValueError):
        return None


def position_text(value) -> str:
    """A cover position as the prompt shows it: open, closed or "40% open" (100 is fully open)."""
    level = percent(value)
    if level is None:
        return "unknown"
    return "open" if level >= 100 else "closed" if level <= 0 else f"{level}% open"


def light_value(state: dict) -> str:
    """on, on 70%, off — or unknown before the app read the accessory."""
    power = state.get("power")
    if not is_read(power):
        return "unknown"
    if str(power).strip().lower() not in _TRUE:
        return "off"
    level = percent(state.get("brightness")) if is_read(state.get("brightness")) else None
    return "on" if level is None or level >= 100 else f"on {level}%"


def cover_value(state: dict) -> str:
    """opening / closing while the target is ahead of the position, else the position."""
    current, target = percent(state.get("current_position")), percent(state.get("target_position"))
    if current is None:
        return "unknown"
    if target is not None and target != current:
        return "opening" if target > current else "closing"
    return position_text(current)


def garage_value(state: dict) -> str:
    current = clean(state.get("current_door_state")).lower()
    return current if is_read(current) else "unknown"


def valve_value(active) -> str:
    if not is_read(active):
        return "unknown"
    return "watering" if str(active).strip().lower() in _TRUE else "off"


def thermostat_value(state: dict) -> str:
    """"19.5°C, target 21.0°C, heating, battery 55%"."""
    parts = []
    if is_read(state.get("current_temperature")):
        parts.append(clean(state["current_temperature"]))
    if is_read(state.get("target_temperature")):
        parts.append(f"target {clean(state['target_temperature'])}")
    mode = clean(state.get("current_heating_cooling")).lower()
    if is_read(mode):
        parts.append({"heat": "heating", "cool": "cooling", "off": "idle"}.get(mode, mode))
    if is_read(state.get("battery_level")):
        parts.append(f"battery {percent(state['battery_level'])}%")
    return ", ".join(parts) or "no reading"


def sensor_value(state: dict, soil: bool = False) -> str:
    """"16.1°C, 82% humidity, battery 42%" — the readings the accessory has, in that order; a
    soil sensor's humidity is the soil moisture."""
    parts = []
    if is_read(state.get("current_temperature")):
        parts.append(clean(state["current_temperature"]))
    if is_read(state.get("current_humidity")):
        level = percent(state["current_humidity"])
        parts.append(f"soil moisture {level}%" if soil else f"{level}% humidity")
    if is_read(state.get("occupancy_detected")):
        parts.append("motion" if str(state["occupancy_detected"]).strip().lower() in _TRUE else "no motion")
    if is_read(state.get("motion_detected")):
        parts.append("motion" if str(state["motion_detected"]).strip().lower() in _TRUE else "no motion")
    if is_read(state.get("contact_state")):
        parts.append(f"contact {clean(state['contact_state']).lower()}")
    if is_read(state.get("battery_level")):
        parts.append(f"battery {percent(state['battery_level'])}%")
    return ", ".join(parts) or "no reading"


class NameIndex:
    """Name → target for the commands. Every target is reachable as `name (room)` (what
    `from_display` hands the backend for the `"name [Room]"` the prompt showed), as its bare
    name when that is unique, and by its UUID."""

    def __init__(self):
        self._holders: dict[str, list[Target]] = {}

    def add(self, target: Target, name: str, room: Optional[str]):
        self._put(name, target)
        self._put(target.id, target)
        if room:
            self._put(f"{name} ({room})", target)

    def _put(self, key: str, target: Target):
        holders = self._holders.setdefault(normalize(key), [])
        if all(held is not target for held in holders):
            holders.append(target)

    def resolve(self, name: str) -> Target:
        key = normalize(name)
        if not key:
            raise Unresolved("No name given.")
        for candidate in self._candidates(key):
            holders = self._holders.get(candidate)
            if holders:
                if len(holders) == 1:
                    return holders[0]
                raise Unresolved(f'"{name}" could be {", ".join(t.display for t in holders)} — say which.')
        unique = [k for k, h in self._holders.items() if len(h) == 1 and not _UUID.match(k)]
        close = difflib.get_close_matches(key, unique, n=3, cutoff=0.6)
        suggestions = sorted({self._holders[k][0].display for k in close})
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise Unresolved(f'No HomeClaw accessory or scene named "{name}".{hint}')

    @staticmethod
    def _candidates(key: str) -> list[str]:
        match = _PAREN_SUFFIX.match(key)
        if match and match.group(1):
            return [key, match.group(1).strip()]
        return [key]

    def targets(self) -> list[Target]:
        seen, out = set(), []
        for holders in self._holders.values():
            for target in holders:
                if id(target) not in seen:
                    seen.add(id(target))
                    out.append(target)
        return out

    def __len__(self):
        return len(self.targets())


def normalize(text: str) -> str:
    return clean(HomeAutomation.from_display(text)).lower()


def _display(name: str, room: Optional[str]) -> str:
    return HomeAutomation.to_display(f"{name} ({room})" if room else name)


def _numbered(accessories: list[dict]) -> dict[str, str]:
    """Accessory id → the name to show: the app's name, numbered #1, #2, … (by UUID) where
    several accessories of one room carry the same name."""
    groups: dict[tuple, list[dict]] = {}
    for accessory in accessories:
        groups.setdefault((_name_of(accessory).lower(), clean(accessory.get("room")).lower()), []).append(accessory)
    names = {}
    for twins in groups.values():
        twins.sort(key=lambda a: str(a.get("id")))
        for number, accessory in enumerate(twins, 1):
            name = _name_of(accessory)
            names[str(accessory.get("id"))] = f"{name} #{number}" if len(twins) > 1 else name
    return names


def _name_of(accessory: dict) -> str:
    return clean(accessory.get("name")) or clean(accessory.get("display_name")) or str(accessory.get("id"))


def build(accessories: list[dict], scenes: list[dict], zones: dict[str, list[dict]],
          filters=DEFAULT_EXCLUDE) -> tuple[HomeSnapshot, NameIndex]:
    """The snapshot and the index of what it lists. `zones` maps an irrigation accessory's id
    to its Valve services (name, active) when the backend read them; an irrigation accessory
    without an entry is one entity with the app's collapsed state."""
    snapshot, index = HomeSnapshot(), NameIndex()
    kept = [a for a in accessories
            if not a.get("is_bridge") and a.get("category") != BRIDGE and not excluded(a, filters)]
    names = _numbered(kept)
    for accessory in kept:
        uid, name = str(accessory.get("id")), names[str(accessory.get("id"))]
        room = clean(accessory.get("room")) or None
        state = dict(accessory.get("state") or {})
        category = str(accessory.get("category") or "")
        reachable = accessory.get("reachable", True) is not False
        if category == IRRIGATION and zones.get(uid):
            for number, service in enumerate(zones[uid], 1):
                zone = clean(service.get("name")) if is_read(service.get("name")) else ""
                zone_name = f"{name} {zone or f'Zon {number}'}"
                target = Target(uid, VALVE, _display(zone_name, room), service=zone or None, state=dict(service))
                snapshot.lights.append(Entity(zone_name, valve_value(service.get("active")), room))
                index.add(target, zone_name, room)
            continue
        target = Target(uid, category, _display(name, room), state=state)
        if category in SWITCHED or (category not in (COVER, GARAGE, IRRIGATION, THERMOSTAT, SENSOR, BUTTON) and "power" in state):
            value = light_value(state)
        elif category == COVER:
            value = cover_value(state)
        elif category == GARAGE:
            value = garage_value(state)
        elif category == IRRIGATION:
            value = valve_value(state.get("active"))
        elif category == THERMOSTAT:
            value = thermostat_value(state)
        else:
            value = sensor_value(state, soil=clean(accessory.get("manufacturer")).lower() in SOIL_MAKERS)
        if not reachable:
            value = "unreachable" if value in ("unknown", "no reading") else f"{value}, unreachable"
        section = snapshot.sensors if category in (THERMOSTAT, SENSOR, BUTTON) or value == "no reading" else snapshot.lights
        section.append(Entity(name, value, room))
        index.add(target, name, room)
    for scene in scenes:
        name = clean(scene.get("name"))
        if not name or is_hidden_scene(scene):
            continue
        kind = clean(scene.get("type")).replace("_", " ")
        value = "scene" if kind in ("", "user defined") else f"scene, {kind}"
        snapshot.actions.append(Entity(name, value))
        index.add(Target(str(scene.get("id")), SCENE, _display(name, None)), name, None)
    for section in (snapshot.lights, snapshot.sensors):
        section.sort(key=lambda e: ((e.area or "").lower(), e.name.lower()))
    snapshot.actions.sort(key=lambda e: e.name.lower())
    return snapshot, index


# --- events -----------------------------------------------------------------------------------

def event_time(stamp) -> Optional[float]:
    """The epoch second of an ISO timestamp the app writes ("2026-10-16T17:09:58Z"); None for anything else."""
    try:
        parsed = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def iso_time(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _verb(kind: str, characteristic: str, value) -> Optional[str]:
    """What the row says happened, or None for a change the dialog does not want: a sensor
    ticking, a brightness step, a cover's position on its way."""
    text = clean(value).lower()
    if characteristic == "power":
        return "turned on" if text in _TRUE else "turned off"
    if characteristic == "current_door_state":
        return f"is {text}" if is_read(text) else None
    if characteristic == "target_position":
        return f"set to {position_text(text)}"
    if characteristic == "active":
        if kind in (VALVE, IRRIGATION):
            return "started watering" if text in _TRUE else "stopped watering"
        return "is active" if text in _TRUE else "is inactive"
    if characteristic in ("occupancy_detected", "motion_detected"):
        return "detected motion" if text in _TRUE else "sees no motion"
    if characteristic == "input_event":
        return f"was pressed ({text})" if is_read(text) else None
    if characteristic == "target_temperature":
        return f"target set to {clean(value)}"
    if characteristic == "target_heating_cooling":
        return f"mode set to {text}"
    if characteristic == "contact_state":
        return f"contact {text}"
    return None


def feed_entry(event: dict, known: dict[str, list[Target]]) -> Optional[FeedEntry]:
    """An app event as a history row, or None: not a characteristic change, an accessory the
    snapshot does not list (another integration's, a bridge), or a change `_verb` drops."""
    if event.get("type") != "characteristic_change":
        return None
    accessory = event.get("accessory") or {}
    targets = known.get(str(accessory.get("id")), [])
    if not targets:
        return None
    stamp = event_time(event.get("timestamp"))
    if stamp is None:
        return None
    service = clean(event.get("service"))
    target = next((t for t in targets if t.service and t.service == service), targets[0])
    verb = _verb(target.kind, str(event.get("characteristic") or ""), event.get("value"))
    if verb is None:
        return None
    return FeedEntry(stamp, f"{target.display} {verb}")
