"""The HomeKit automations as HomeClaw lists them, made into a line each for the prompt and a
block each for `homekit_automations`.

`list_automations` gives a row per automation: `trigger_type` (calendar, significant_time,
button, characteristic, presence, timer, orphaned), `event_summary` ("at 08:00 (+1 condition)",
"at sunset-45m", "Knapp single_press", "when the last person leaves"), `enabled` and the
`action_sets` it runs. `get_automation` adds `conditions` (weekdays, characteristic predicates,
each with a `summary`) and `events`. An automation made in the Home app runs a hidden action set
named by a bare UUID; `get_scene` lists its `actions` ({accessory, characteristic, value}), which
are shown per accessory and grouped by what they do. A named scene is shown by its name."""
import re
from typing import Optional

from .HomeClawModel import is_uuid

WEEKDAYS = {(1, 2, 3, 4, 5, 6, 7): "daily", (2, 3, 4, 5, 6): "Mon–Fri", (1, 7): "Sat–Sun"}
HEATING_MODES = {"0": "mode off", "1": "heat", "2": "cool", "3": "auto"}
COLOUR = {"hue", "saturation", "color_temperature"}
_CONDITION_COUNT = re.compile(r"\s*\(\+\d+ conditions?\)\s*$")


def is_hidden(action_set: dict) -> bool:
    """An action set HomeKit made for one automation, not a scene a person named."""
    return bool(action_set.get("hidden")) or is_uuid(action_set.get("name"))


def trigger_text(automation: dict) -> str:
    """When the automation fires, in the app's words without its condition count."""
    summary = _CONDITION_COUNT.sub("", str(automation.get("event_summary") or "")).replace("_press", " press").strip()
    if not summary:
        fire_time = automation.get("fire_time")
        summary = f"at {fire_time}" if fire_time else str(automation.get("trigger_type") or "unknown trigger")
    if automation.get("trigger_type") == "orphaned":
        summary += " — orphaned: what it watched is gone, it never fires"
    return summary


def condition_text(condition: dict) -> str:
    if condition.get("type") == "weekdays":
        days = tuple(sorted(int(d) for d in condition.get("weekdays") or []))
        if days in WEEKDAYS:
            return WEEKDAYS[days]
    if condition.get("summary"):
        return str(condition["summary"])
    parts = [str(condition[k]) for k in ("accessory", "characteristic", "value") if condition.get(k) not in (None, "")]
    return " ".join(parts) or str(condition.get("type") or "condition")


def _value_text(characteristic: str, value) -> Optional[str]:
    value = str(value).strip()
    if characteristic == "power":
        return "on" if value.lower() in ("1", "true", "on") else "off"
    if characteristic == "brightness":
        return f"{value} %"
    if characteristic in ("target_position", "target_tilt_angle"):
        return "open" if value == "100" else "closed" if value == "0" else f"{value} % open"
    if characteristic == "target_temperature":
        return f"{value} °C"
    if characteristic in ("target_heating_cooling", "target_heating_cooling_state"):
        return HEATING_MODES.get(value, f"mode {value}")
    if characteristic == "active":
        return "on" if value == "1" else "off"
    if characteristic in COLOUR:
        return "colour"
    return f"{characteristic.replace('_', ' ')} {value}"


def _rank(part: str) -> int:
    """The order of an accessory's settings: switched or mode first, the colour last."""
    return 0 if part in ("on", "off", *HEATING_MODES.values()) else 2 if part == "colour" else 1


def actions_text(actions: list[dict]) -> str:
    """The actions of an action set, what each accessory is set to, the accessories with the
    same setting grouped: "off: A, B; on 30 %: C"."""
    per_accessory: dict[str, tuple[str, list[str]]] = {}
    for action in actions:
        key = str(action.get("accessory_id") or action.get("accessory") or "?")
        name = str(action.get("accessory") or key)
        text = _value_text(str(action.get("characteristic") or ""), action.get("value"))
        _, parts = per_accessory.setdefault(key, (name, []))
        if text not in parts:
            parts.append(text)
    groups: dict[str, list[str]] = {}
    for name, parts in per_accessory.values():
        setting = "off" if "off" in parts and "on" not in parts else " ".join(sorted(parts, key=_rank))
        groups.setdefault(setting, [])
        if name not in groups[setting]:
            groups[setting].append(name)
    return "; ".join(f"{setting}: {', '.join(names)}" for setting, names in groups.items()) or "no actions"


def head(automation: dict) -> str:
    """Name, trigger and conditions: the part of an automation every listing shows."""
    conditions = [condition_text(c) for c in automation.get("conditions") or []]
    text = f"{automation.get('name') or automation.get('id')}: {trigger_text(automation)}"
    if conditions:
        text += ", " + ", ".join(conditions)
    return text + ("" if automation.get("enabled", True) else " (disabled)")


def _runs(automation: dict) -> str:
    """What the automation runs, briefly: the named scenes and the count of the other actions."""
    named = [str(s.get("name")) for s in automation.get("action_sets") or [] if not is_hidden(s)]
    count = sum(int(s.get("action_count") or 0) for s in automation.get("action_sets") or [] if is_hidden(s))
    parts = [f"scene {n}" for n in named] + ([f"{count} action{'s' if count != 1 else ''}"] if count else [])
    return ", ".join(parts) or "nothing"


def sort_key(automation: dict):
    return str(automation.get("name") or "").casefold()


def render_summary(automations: list[dict]) -> str:
    """The prompt's `## Automations` block: a line per automation."""
    if not automations:
        return ""
    lines = [f"- {head(a)} → {_runs(a)}" for a in sorted(automations, key=sort_key)]
    return "## Automations\n" + "\n".join(lines)


def render_detail(automations: list[dict], scenes: dict[str, dict], expand_named: bool) -> str:
    """The reply of `homekit_automations`: per automation its head and, indented, what it runs;
    `scenes` maps an action set id to its `get_scene` answer. A named scene is only named unless
    `expand_named`."""
    blocks = []
    for a in sorted(automations, key=sort_key):
        lines = [head(a)]
        for action_set in a.get("action_sets") or []:
            detail = scenes.get(str(action_set.get("id")))
            actions = actions_text(detail.get("actions") or []) if detail else f"{action_set.get('action_count', '?')} actions"
            if is_hidden(action_set):
                lines.append(f"  → {actions}")
            else:
                lines.append(f"  → scene {action_set.get('name')}" + (f": {actions}" if expand_named and detail else ""))
        if len(lines) == 1:
            lines.append("  → nothing")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)
