"""HomeClaw as a HomeAutomation backend over the app's local socket (HomeClawClient). A
snapshot is `list_accessories` and `list_scenes` — a cache read in the app, a few
milliseconds — plus one `get_accessory` per irrigation control for its zones, mapped by
HomeClawModel; a command resolves its name through the index of the last read and re-reads
once for a name the index does not know (a rename in the Home app). Every write is `control`
with verify, a scene `trigger_scene`, and `recent_events` reads the app's `events` log for
the homeclaw_events command. The health check is `status`; the app not running is the one
failure the user can fix, and every answer says so. Not a Feed yet: the event rows are
already shaped as FeedEntry, so making it one is adding the base class, a cursor and the
registration."""
import threading
import time
from typing import List, Optional

from assistant.integration.Feed import FeedEntry
from assistant.integration.HomeAutomation import HomeAutomation, HomeSnapshot
from assistant.integration.Integration import Integration
from common.ConfigManager import config
from common.logging import info, trace
from .HomeClawClient import HomeClawClient, HomeClawError
from .HomeClawModel import (
    COVER, DEFAULT_EXCLUDE, GARAGE, IRRIGATION, SCENE, SWITCHED, THERMOSTAT, VALVE, VALVE_SERVICE, NameIndex, Target,
    Unresolved, build, feed_entry, is_read, iso_time, percent)

EVENT_LIMIT = 500           # events per read; the app answers newest first, so a full page means older ones are cut
SET_HINT = "use homeclaw_set(name, characteristic, value) from the homeclaw toolkit"


class HomeClawIntegration(Integration, HomeAutomation):
    """HomeKit through the HomeClaw app: sensors, covers, garage door, scenes.

    Makes the HomeKit home a home automation backend: Keep reads the sensors room by room, the
    thermostat, covers, the garage door, the irrigation valves and the lights no other
    integration owns, drives them and the scenes, and tells what changed in the house.

    The app must be running on the server's Mac; nothing to sign in, the app holds the HomeKit
    access. One JSON request per socket connection; the exclusion list is `homeclaw.exclude`,
    matched against the maker and the bridge. Webhooks (the app's `set_webhook`) are not used
    yet, but nothing here stands in their way."""

    def __init__(self):
        super().__init__()
        self._client: Optional[HomeClawClient] = None
        self._lock = threading.Lock()
        self._index: Optional[NameIndex] = None
        self._known: dict[str, list[Target]] = {}       # accessory id → its targets, for the events

    def _on_reset(self):
        self._client = None

    # --- Integration ------------------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "HomeClaw"

    @property
    def config_keys(self) -> List[str]:
        return []

    @property
    def config_section(self) -> str:
        return "homeclaw"

    def client(self) -> HomeClawClient:
        if self._client is None:
            self._client = HomeClawClient(str(config.get("homeclaw.socket", "") or ""))
        return self._client

    def check_health(self) -> bool:
        status = self.client().call("status") or {}
        if not status.get("ready"):
            raise RuntimeError("the HomeClaw app is running but not ready — it is still loading the home")
        if not status.get("homes"):
            raise RuntimeError("the HomeClaw app sees no HomeKit home: open it and check its HomeKit access")
        trace(f"HomeClaw: {status.get('accessories', '?')} accessories in {status.get('homes')} home(s)")
        return True

    # --- HomeAutomation: state ------------------------------------------------------------------

    def snapshot(self) -> HomeSnapshot:
        snapshot, _ = self._read()
        return snapshot

    def _exclusions(self) -> list:
        return list(config.get("homeclaw.exclude", list(DEFAULT_EXCLUDE)) or [])

    def _read(self) -> tuple[HomeSnapshot, NameIndex]:
        client = self.client()
        accessories = client.call("list_accessories") or []
        if isinstance(accessories, dict):
            accessories = accessories.get("accessories") or []
        scenes = client.call("list_scenes") or []
        if isinstance(scenes, dict):
            scenes = scenes.get("scenes") or []
        snapshot, index = build(accessories, scenes, self._zones(accessories), self._exclusions())
        snapshot.taken_at = time.time()
        known: dict[str, list[Target]] = {}
        for target in index.targets():
            known.setdefault(target.id, []).append(target)
        with self._lock:
            self._index, self._known = index, known
        trace(f"HomeClaw: {len(snapshot.lights)} controllable, {len(snapshot.sensors)} sensor readings, "
              f"{len(snapshot.actions)} scenes from {len(accessories)} accessories")
        return snapshot, index

    def _zones(self, accessories: list) -> dict[str, list[dict]]:
        """The Valve services of every irrigation control: `get_accessory` without a refresh
        is a cache read. A control the app cannot describe stays one entity."""
        zones = {}
        for accessory in accessories:
            if accessory.get("category") != IRRIGATION:
                continue
            uid = str(accessory.get("id"))
            try:
                detail = self.client().call("get_accessory", id=uid, refresh=False) or {}
            except (HomeClawError, TimeoutError) as e:
                info(f"HomeClaw: cannot read the zones of {accessory.get('name')}: {e}")
                continue
            valves = []
            for service in detail.get("services") or []:
                if service.get("type") != VALVE_SERVICE:
                    continue
                chars = {c.get("name"): c.get("value") for c in service.get("characteristics") or []}
                valves.append({"name": service.get("name"), "active": chars.get("active")})
            if valves:
                zones[uid] = valves
        return zones

    # --- HomeAutomation: commands ---------------------------------------------------------------

    def turn_on(self, name: str) -> str:
        return self._switch(name, True)

    def turn_off(self, name: str) -> str:
        return self._switch(name, False)

    def _switch(self, name: str, on: bool) -> str:
        target, text = self._resolve(name)
        if target is None:
            return text
        if target.kind in SWITCHED or (target.kind not in (COVER, GARAGE, IRRIGATION, VALVE, THERMOSTAT, SCENE)
                                       and "power" in target.state):
            return self._control(target, "power", "true" if on else "false")
        if target.kind == COVER:
            return self._control(target, "target_position", "100" if on else "0")
        if target.kind == GARAGE:
            return self._control(target, "target_door_state", "open" if on else "closed")
        if target.kind in (VALVE, IRRIGATION):
            return self._control(target, "active", "1" if on else "0")
        return self._wrong_command(target, "turn_on" if on else "turn_off")

    def dim(self, name: str, level: int) -> str:
        level = int(level)
        if level <= 0:
            return self.turn_off(name)
        level = min(level, 100)
        target, text = self._resolve(name)
        if target is None:
            return text
        if target.kind == COVER:
            return self._control(target, "target_position", str(level))
        if target.kind in SWITCHED:
            if not is_read(target.state.get("brightness")) and "brightness" not in target.state:
                return f"{target.display} is not dimmable."
            failed = self._control(target, "power", "true")
            return failed or self._control(target, "brightness", str(level))
        return self._wrong_command(target, "dim")

    def stop(self, name: str) -> str:
        """A cover halts where it is: HomeKit has no stop, so the target becomes the position."""
        target, text = self._resolve(name)
        if target is None:
            return text
        if target.kind != COVER:
            return self._wrong_command(target, "stop")
        position = percent(target.state.get("current_position"))
        if position is None:
            return f"{target.display}: the position is not known, so it cannot be stopped where it is."
        return self._control(target, "target_position", str(position))

    def execute(self, name: str) -> str:
        target, text = self._resolve(name)
        if target is None:
            return text
        if target.kind != SCENE:
            return self._wrong_command(target, "execute")
        try:
            self.client().call("trigger_scene", id=target.id)
        except (HomeClawError, TimeoutError) as e:
            info(f"HomeClaw: trigger_scene {target.display} failed: {e}")
            return self._failure(e)
        info(f"HomeClaw: scene {target.display} triggered")
        return ""

    def set_characteristic(self, name: str, characteristic: str, value: str) -> str:
        """`homeclaw_set`: any writable characteristic of an accessory, verified. Returns the
        outcome for the LLM — the new value, or why it failed."""
        target, text = self._resolve(name)
        if target is None:
            return text
        if target.kind == SCENE:
            return f"{target.display} is a scene; use execute."
        failed, data = self._write(target, characteristic, value)
        if failed:
            return failed
        shown = data.get("new_value", value) if isinstance(data, dict) else value
        if isinstance(data, dict) and data.get("verified") is False:
            return (f"{target.display}: {characteristic} was set to {value}, but the accessory reports "
                    f"{data.get('current_value', 'something else')}.")
        return f"{target.display}: {characteristic} is now {shown}."

    @staticmethod
    def _wrong_command(target: Target, verb: str) -> str:
        if target.kind == SCENE:
            return f"{target.display} is a scene; use execute."
        if target.kind == THERMOSTAT:
            return f"{target.display} is a thermostat; {SET_HINT} (target_temperature, target_heating_cooling)."
        if target.kind == COVER:
            return f"{target.display} is a cover; use turn_on (open) / turn_off (close) / dim (position) / stop."
        if target.kind == GARAGE:
            return f"{target.display} is a garage door; use turn_on (open) / turn_off (close)."
        if target.kind in (VALVE, IRRIGATION):
            return f"{target.display} is an irrigation valve; use turn_on (water) / turn_off."
        if target.kind in SWITCHED:
            return f"{target.display} is a light; use turn_on / turn_off / dim."
        return f"{target.display} is a sensor reading and cannot be controlled with {verb}."

    def _resolve(self, name: str) -> tuple[Optional[Target], str]:
        """(the target, "") or (None, the text for the LLM); a name the index does not hold
        reads the app again first."""
        try:
            with self._lock:
                index = self._index
            if index is None:
                _, index = self._read()
            try:
                return index.resolve(name), ""
            except Unresolved:
                info(f"HomeClaw: '{name}' is not in the index — reading the app again")
                _, index = self._read()
                return index.resolve(name), ""
        except Unresolved as e:
            info(f"HomeClaw: {e}")
            return None, str(e)
        except (HomeClawError, TimeoutError) as e:
            info(f"HomeClaw: cannot resolve '{name}': {e}")
            return None, self._failure(e)

    def _control(self, target: Target, characteristic: str, value: str) -> str:
        failed, _ = self._write(target, characteristic, value)
        return failed

    def _write(self, target: Target, characteristic: str, value: str) -> tuple[str, Optional[dict]]:
        """One verified `control`; ("", the app's data) or (the text for the LLM, None)."""
        try:
            data = self.client().call("control", id=target.id, characteristic=characteristic, value=str(value),
                                      verify=True, service_name=target.service)
        except (HomeClawError, TimeoutError) as e:
            info(f"HomeClaw: control {target.display} {characteristic}={value} failed: {e}")
            return self._failure(e), None
        info(f"HomeClaw: {target.display} ← {characteristic}={value}"
             + (f" (verified {data.get('verified')})" if isinstance(data, dict) and "verified" in data else ""))
        return "", data if isinstance(data, dict) else {}

    @staticmethod
    def _failure(e: Exception) -> str:
        return "Request timed out" if isinstance(e, TimeoutError) else str(e)

    # --- events -------------------------------------------------------------------------------

    def recent_events(self, hours: float) -> list[FeedEntry]:
        """What changed in the house in the last `hours`, oldest first: the app's `events` log
        minus what the dialog does not want (sensor ticks, brightness steps, the accessories
        of another integration). Raises when the app does not answer."""
        since = iso_time(time.time() - hours * 3600)
        data = self.client().call("events", limit=EVENT_LIMIT, since=since) or {}
        events = list((data.get("events") if isinstance(data, dict) else data) or [])
        with self._lock:
            known = self._known
        if not known:
            self._read()
            with self._lock:
                known = self._known
        if len(events) >= EVENT_LIMIT:
            info(f"HomeClaw: the event page is full ({EVENT_LIMIT}) — the oldest changes since {since} are cut")
        entries = [entry for entry in (feed_entry(event, known) for event in reversed(events)) if entry is not None]
        entries.sort(key=lambda e: e.timestamp)
        trace(f"HomeClaw: {len(events)} events since {since}, {len(entries)} kept")
        return entries
