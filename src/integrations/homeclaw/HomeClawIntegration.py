"""HomeKit through the HomeClaw app's local socket (HomeClawClient): the accessories Keep's Hue
and Plejd integrations do not already bring, for the HOMEKIT prompt section and the homeclaw
toolkit, and the app's log of what changed in the house.

A refresh is `list_accessories` and `list_scenes` — cache reads in the app — plus one
`get_accessory` per irrigation control for its zones; HomeClawModel turns them into the
targets and the Index the commands resolve against. Every write is one `control` with
verify, a zone's addressed by its Valve service's UUID; a dry run asks the app to validate
the same request without acting. A thermostat target outside `homeclaw.thermostat_range` is
refused here, since the app validates the format, not the range.

Not a Feed yet: `events` already turns the app's log into FeedEntry rows, so making it one —
the changes poured into the history as they happen — is adding the base class, a cursor over
the event time and the class to the feed set tests/unit/assistant/integration/test_bootstrap.py
pins."""
import threading
from typing import List, Optional

from assistant.integration.Feed import FeedEntry
from assistant.integration.Integration import Integration
from common.ConfigManager import config
from common.logging import info, trace
from .HomeClawAutomations import is_hidden, render_detail, render_summary
from .HomeClawClient import HomeClawClient, HomeClawError, HomeClawUnavailable
from .HomeClawModel import (
    COVER, DEFAULT_BRIDGE_FILTERS, GARAGE, IRRIGATION, SWITCHED, THERMOSTAT, Index, Target, Unresolved,
    build_targets, event_entry, excluded, real_scenes, render, zones_of)

EVENT_LIMIT = 500           # events per read; the app answers newest first, so a full page means older ones are cut
DEFAULT_EVENT_SKIP = ("current_temperature", "current_humidity", "battery_level", "color_temperature", "hue",
                      "saturation", "brightness")


class HomeClawIntegration(Integration):
    """HomeKit through the HomeClaw app: sensors, covers, garage, irrigation.

    Shows the HomeKit sensors, thermostat, covers, garage door, irrigation valves and the lights
    Hue and Plejd do not already cover, room by room, and opens, closes, waters, heats, switches
    and runs scenes when asked. Tells what changed in the house from the app's event log.

    The HomeClaw app must be running on the server's Mac as the same user; there is nothing to
    sign in, the app holds the HomeKit access. One JSON request per socket connection. Everything
    is keyed on the HomeKit UUID; the exclusion of the Hue and Plejd accessories is
    `homeclaw.exclude_bridges`, overridden per UUID by `homeclaw.include` / `homeclaw.exclude`."""

    def __init__(self):
        super().__init__()
        self._client: Optional[HomeClawClient] = None
        self._lock = threading.Lock()
        self._index: Optional[Index] = None
        self._scenes: list[dict] = []
        self._by_accessory: dict[str, list[Target]] = {}
        self._left_out: set[str] = set()        # accessory UUIDs excluded at the last refresh

    def _on_reset(self):
        self._client = None

    @property
    def name(self) -> str:
        return "HomeClaw"

    @property
    def config_keys(self) -> List[str]:
        return []

    def client(self) -> HomeClawClient:
        if self._client is None:
            self._client = HomeClawClient(str(config.get("homeclaw.socket", "") or ""))
        return self._client

    def check_health(self) -> bool:
        status = self.client().call("status") or {}
        if not status.get("ready"):
            raise HomeClawError("HomeClaw is running but not ready — it is still loading the home")
        if not status.get("homes"):
            raise HomeClawError("HomeClaw sees no HomeKit home: open the app and check its HomeKit access")
        trace(f"HomeClaw: {status.get('accessories', '?')} accessories in {status.get('homes')} home(s)")
        return True

    # --- reading ----------------------------------------------------------------------------------

    @staticmethod
    def _list(data, key: str) -> list:
        if isinstance(data, dict):
            data = data.get(key)
        return list(data or [])

    def refresh(self) -> Index:
        """Reads the app and rebuilds the index the commands resolve against."""
        client = self.client()
        accessories = self._list(client.call("list_accessories"), "accessories")
        scenes = self._list(client.call("list_scenes"), "scenes")
        bridge_filters = config.get("homeclaw.exclude_bridges", list(DEFAULT_BRIDGE_FILTERS)) or []
        include = config.get("homeclaw.include", []) or []
        exclude = config.get("homeclaw.exclude", []) or []
        kept, left_out = [], set()
        for accessory in accessories:
            (left_out.add(str(accessory.get("id") or "").upper()) if excluded(accessory, bridge_filters, include, exclude)
             else kept.append(accessory))
        index = Index(build_targets(kept, self._zones(kept)))
        shown = real_scenes(scenes)
        by_accessory: dict[str, list[Target]] = {}
        for target in index.targets:
            by_accessory.setdefault(target.id, []).append(target)
        with self._lock:
            self._index, self._scenes, self._by_accessory, self._left_out = index, shown, by_accessory, left_out
        trace(f"HomeClaw: {len(index.targets)} targets from {len(kept)} accessories ({len(left_out)} left out), "
              f"{len(shown)} scenes ({len(scenes) - len(shown)} hidden ones dropped)")
        return index

    def _zones(self, accessories: list[dict]) -> dict[str, list[dict]]:
        zones = {}
        for accessory in accessories:
            if accessory.get("category") != IRRIGATION:
                continue
            uid = str(accessory.get("id") or "").upper()
            try:
                found = zones_of(self.client().call("get_accessory", id=uid, refresh=False) or {})
            except HomeClawError as e:
                info(f"HomeClaw: cannot read the zones of {accessory.get('name')}: {e} — kept as one valve")
                continue
            if found:
                zones[uid] = found
        return zones

    def snapshot(self) -> str:
        """The prompt section's body, from a fresh read."""
        index = self.refresh()
        with self._lock:
            scenes = list(self._scenes)
        body = render(index, scenes)
        try:
            summary = render_summary(self.automations())
        except HomeClawUnavailable:
            raise
        except HomeClawError as e:
            info(f"HomeClaw: automations left out of the section: {e}")
            summary = ""
        return f"{body}\n\n{summary}" if summary else body

    def _resolve(self, ref: str, kinds, what: str) -> Target:
        with self._lock:
            index = self._index
        if index is None:
            index = self.refresh()
        try:
            return index.resolve(ref, kinds, what)
        except Unresolved:
            trace(f"HomeClaw: '{ref}' does not resolve to one {what} — reading the app again")
            return self.refresh().resolve(ref, kinds, what)

    # --- automations ------------------------------------------------------------------------------

    def automations(self) -> list[dict]:
        """Every automation of the home with its conditions: `list_automations` plus one
        `get_automation` each (the list leaves the conditions out); a row whose detail cannot be
        read is kept as listed."""
        rows = self._list(self.client().call("list_automations"), "automations")
        detailed = []
        for row in rows:
            try:
                detail = self.client().call("get_automation", id=row.get("id"))
            except HomeClawUnavailable:
                raise
            except HomeClawError as e:
                info(f"HomeClaw: cannot read automation {row.get('name')}: {e} — shown without its conditions")
                detail = None
            detailed.append({**row, **detail} if isinstance(detail, dict) else row)
        trace(f"HomeClaw: {len(detailed)} automations")
        return detailed

    def describe_automations(self, wanted: str = "") -> str:
        """The `homekit_automations` reply: every automation, or those whose name holds `wanted`
        (or whose id it is), each with what it runs — the actions of a hidden action set always,
        those of a named scene only when automations were asked for by name."""
        automations = self.automations()
        wanted = str(wanted or "").strip().strip('"').casefold()
        if wanted:
            automations = [a for a in automations if wanted in str(a.get("name") or "").casefold()
                           or wanted == str(a.get("id") or "").casefold()]
            if not automations:
                trace(f"HomeClaw: no automation matches '{wanted}'")
                return f"No HomeKit automation called '{wanted}'; call homekit_automations() for the list."
        if not automations:
            return "The home has no HomeKit automations."
        scenes = {}
        for automation in automations:
            for action_set in automation.get("action_sets") or []:
                if (wanted or is_hidden(action_set)) and action_set.get("id") not in scenes:
                    try:
                        scenes[action_set["id"]] = self.client().call("get_scene", id=action_set["id"]) or {}
                    except HomeClawUnavailable:
                        raise
                    except HomeClawError as e:
                        info(f"HomeClaw: cannot read the actions of {automation.get('name')}: {e}")
        return render_detail(automations, scenes, expand_named=bool(wanted))

    # --- writing ----------------------------------------------------------------------------------

    def _control(self, target: Target, characteristic: str, value: str, dry_run: bool, shown: str) -> str:
        """One `control` with verify (or a dry run of it); the outcome for the LLM."""
        data = self.client().call("control", id=target.id, characteristic=characteristic, value=value,
                                  verify="true", dry_run="true" if dry_run else None, service_id=target.service_id)
        data = data if isinstance(data, dict) else {}
        if dry_run:
            if data.get("valid") is False:
                info(f"HomeClaw: dry run {target.label} {characteristic}={value} refused: {data}")
                return f"Dry run: HomeClaw would refuse to set {target.label} to {shown}: {data.get('error') or data}"
            trace(f"HomeClaw: dry run {target.label} {characteristic}={value} valid")
            return (f"Dry run: {target.label} would go to {shown} ({characteristic} "
                    f"{data.get('current_value', '?')} → {data.get('new_value', value)}); nothing was changed.")
        info(f"HomeClaw: {target.label} ({target.key}) ← {characteristic}={value}, verified {data.get('verified')}")
        if data.get("verified") is False:
            return (f"{target.label}: {characteristic} was sent {value}, but the accessory reports "
                    f"{data.get('current_value', data.get('actual_value', 'something else'))}.")
        return f"{target.label} is set to {shown}."

    def set_cover(self, ref: str, position: int, dry_run: bool = False) -> str:
        target = self._resolve(ref, {COVER}, "cover")
        shown = "open" if position >= 100 else "closed" if position <= 0 else f"{position} % open"
        return self._control(target, "target_position", str(position), dry_run, shown)

    def set_garage(self, ref: str, open_it: bool, dry_run: bool = False) -> str:
        target = self._resolve(ref, {GARAGE}, "garage door")
        state = "open" if open_it else "closed"
        return self._control(target, "target_door_state", state, dry_run, state)

    def set_valve(self, ref: str, on: bool, dry_run: bool = False) -> str:
        target = self._resolve(ref, {IRRIGATION}, "irrigation valve")
        return self._control(target, "active", "1" if on else "0", dry_run, "watering" if on else "off")

    def set_light(self, ref: str, on: bool, dry_run: bool = False) -> str:
        target = self._resolve(ref, SWITCHED, "light")
        return self._control(target, "power", "true" if on else "false", dry_run, "on" if on else "off")

    def set_thermostat(self, ref: str, celsius: float, dry_run: bool = False) -> str:
        low, high = (float(v) for v in (config.get("homeclaw.thermostat_range", [10, 30]) or [10, 30])[:2])
        if not low <= celsius <= high:
            raise HomeClawError(f"{celsius:g} °C is outside the allowed thermostat range {low:g}–{high:g} °C "
                                f"(homeclaw.thermostat_range in config.yaml)")
        target = self._resolve(ref, {THERMOSTAT}, "thermostat")
        return self._control(target, "target_temperature", f"{celsius:g}", dry_run, f"{celsius:g} °C")

    def trigger_scene(self, name: str, dry_run: bool = False) -> str:
        scene = self._scene(name)
        if dry_run:
            return f"Dry run: the scene {scene['name']} would run; nothing was changed."
        self.client().call("trigger_scene", id=scene["id"])
        info(f"HomeClaw: scene {scene['name']} ({scene['id']}) triggered")
        return f"The scene {scene['name']} ran."

    def _scene(self, name: str) -> dict:
        wanted = str(name or "").strip().strip('"').lower()
        if not wanted:
            raise Unresolved("Name the scene.")
        for attempt in (0, 1):
            with self._lock:
                scenes = list(self._scenes) if self._index is not None else None
            if scenes is None or attempt:
                self.refresh()
                with self._lock:
                    scenes = list(self._scenes)
            exact = [s for s in scenes if str(s.get("name")).lower() == wanted or str(s.get("id")).lower() == wanted]
            partial = [s for s in scenes if wanted in str(s.get("name")).lower()]
            hits = exact or partial
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                raise Unresolved(f"Several scenes fit '{name}': " + ", ".join(s["name"] for s in hits))
            trace(f"HomeClaw: no scene '{name}'{' after a re-read' if attempt else ''}")
        raise Unresolved(f"No scene called '{name}' in HomeClaw; the scenes are: "
                         + ", ".join(sorted(str(s.get('name')) for s in scenes)))

    # --- events -----------------------------------------------------------------------------------

    def events(self, since: str = "1h") -> list[FeedEntry]:
        """The changes since `since` (the app's form: "1h", "30m", an ISO time), oldest first,
        minus `homeclaw.event_skip` and the accessories left out."""
        data = self.client().call("events", limit=EVENT_LIMIT, since=since or "1h", type="characteristic_change")
        events = self._list(data, "events")
        if any(str((e.get("accessory") or {}).get("id") or "").upper() not in self._known_ids() for e in events):
            self.refresh()
        if len(events) >= EVENT_LIMIT:
            info(f"HomeClaw: the event page is full ({EVENT_LIMIT}) — changes before the oldest one are cut")
        skip = {str(s) for s in (config.get("homeclaw.event_skip", list(DEFAULT_EVENT_SKIP)) or [])}
        with self._lock:
            targets = dict(self._by_accessory)
        entries = [entry for event in events if event.get("characteristic") not in skip
                   and (entry := event_entry(event, targets)) is not None]
        entries.sort(key=lambda e: e.timestamp)
        trace(f"HomeClaw: {len(events)} events since {since}, {len(entries)} kept")
        return entries

    def _known_ids(self) -> set[str]:
        with self._lock:
            return set(self._by_accessory) | self._left_out
