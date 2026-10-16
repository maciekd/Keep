"""A fake HomeClaw app: a Unix socket server answering one JSON line per connection, as the
real app does, from a dict of command → reply (a value, or a callable of the args), and a
fictional house to serve."""
import json
import os
import shutil
import socket
import tempfile
import threading
from unittest.mock import patch

import pytest

from integrations.homeclaw.HomeClawIntegration import HomeClawIntegration
from integrations.homeclaw.HomeClawModel import VALVE_SERVICE

PLEJD_BRIDGE = {"category": "bridge", "id": "C0000000-0000-4000-8000-000000000001", "manufacturer": "Plejd",
                "name": "Plejd Gateway", "model": "GWY-01"}
HUE_BRIDGE = {"category": "bridge", "id": "C0000000-0000-4000-8000-000000000002", "manufacturer": "Signify Netherlands B.V.",
              "name": "Philips hue - 0A0A0A", "model": "BSB002"}
SENSOR_HUB = {"category": "bridge", "id": "C0000000-0000-4000-8000-000000000003", "manufacturer": "Aqara", "name": "Aqara Hub"}

SENSOR_A = "A1000001-0000-4000-8000-000000000001"
SENSOR_B = "A1000002-0000-4000-8000-000000000002"
SENSOR_C = "A1000003-0000-4000-8000-000000000003"
AWNING = "B2000001-0000-4000-8000-000000000001"
GARAGE_DOOR = "B3000001-0000-4000-8000-000000000001"
WATER_CONTROL = "B4000001-0000-4000-8000-000000000001"
ZONE_PATIO = "D4000001-0000-4000-8000-000000000001"
ZONE_GREENHOUSE = "D4000002-0000-4000-8000-000000000002"
THERMOSTAT = "B5000001-0000-4000-8000-000000000001"
FLOOR_LAMP = "B6000001-0000-4000-8000-000000000001"
PLEJD_LAMP = "E1000001-0000-4000-8000-000000000001"
HUE_LAMP = "E2000001-0000-4000-8000-000000000001"


def _sensor(uid, temperature, humidity, battery):
    return {"id": uid, "name": "Climate Sensor T1", "room": "Crawl space", "category": "sensor", "manufacturer": "Aqara",
            "reachable": True, "bridge": SENSOR_HUB,
            "state": {"current_temperature": f"{temperature}°C", "current_humidity": humidity, "battery_level": battery}}


ACCESSORIES = [
    {**PLEJD_BRIDGE, "is_bridge": True, "room": "Utility", "reachable": True},
    {**HUE_BRIDGE, "is_bridge": True, "room": "Utility", "reachable": True},
    _sensor(SENSOR_A, "16.1", "82.42", "42"),
    _sensor(SENSOR_B, "15.7", "79.2", "45"),
    _sensor(SENSOR_C, "18.4", "--", "57"),
    {"id": AWNING, "name": "Awning Roof", "room": "Patio", "category": "window_covering", "manufacturer": "Somfy",
     "reachable": True, "bridge": {"manufacturer": "Somfy", "name": "TaHoma switch"},
     "state": {"current_position": "100", "target_position": "40"}},
    {"id": GARAGE_DOOR, "name": "Garage Door", "room": "Garage", "category": "garage_door", "manufacturer": "Meross",
     "reachable": True, "bridge": None, "state": {"current_door_state": "closed", "target_door_state": "closed"}},
    {"id": WATER_CONTROL, "name": "Dual Water Control", "room": "Greenhouse", "category": "irrigation",
     "manufacturer": "GARDENA", "reachable": True, "bridge": {"manufacturer": "GARDENA", "name": "smart Gateway"},
     "state": {"active": "0", "battery_level": "100"}},
    {"id": THERMOSTAT, "name": "Radiator", "room": "Game room", "category": "thermostat", "manufacturer": "Eve Systems",
     "reachable": True, "bridge": None,
     "state": {"current_temperature": "20.0°C", "target_temperature": "21.0°C", "target_heating_cooling": "heat"}},
    {"id": FLOOR_LAMP, "name": "Floor Lamp", "room": "Game room", "category": "lightbulb", "manufacturer": "Nanoleaf",
     "reachable": True, "bridge": None, "state": {"power": "true", "brightness": "70"}},
    {"id": PLEJD_LAMP, "name": "Ceiling", "room": "Kitchen", "category": "lightbulb", "manufacturer": "Plejd",
     "reachable": True, "bridge": PLEJD_BRIDGE, "state": {"power": "false", "brightness": "50"}},
    {"id": HUE_LAMP, "name": "Behind Sofa", "room": "Kitchen", "category": "lightbulb",
     "manufacturer": "Signify Netherlands B.V.", "reachable": True, "bridge": HUE_BRIDGE, "state": {"power": "true"}},
]

WATER_CONTROL_DETAIL = {"id": WATER_CONTROL, "name": "Dual Water Control", "services": [
    {"id": "D4000000-0000-4000-8000-000000000000", "name": "Dual Water Control",
     "type": "000000CF-0000-1000-8000-0026BB765291", "characteristics": [{"name": "active", "value": "0"}]},
    {"id": ZONE_PATIO, "name": "Patio", "type": VALVE_SERVICE, "characteristics": [{"name": "active", "value": "1"}]},
    {"id": ZONE_GREENHOUSE, "name": "Greenhouse", "type": VALVE_SERVICE, "characteristics": [{"name": "active", "value": "0"}]},
]}

SCENES = [
    {"id": "F1000001-0000-4000-8000-000000000001", "name": "Good Night", "type": "sleep"},
    {"id": "F1000002-0000-4000-8000-000000000002", "name": "Movie Evening", "type": "user_defined"},
    {"id": "F1000003-0000-4000-8000-000000000003", "name": "F1000003-0000-4000-8000-0000000000AA", "type": "user_defined"},
]

MORNING = "A7000001-0000-4000-8000-000000000001"
BUTTON = "A7000002-0000-4000-8000-000000000002"
HIDDEN_SET = "F2000001-0000-4000-8000-000000000001"
AUTOMATIONS = [
    {"id": MORNING, "name": "Weekday Morning", "enabled": True, "trigger_type": "calendar", "fire_time": "06:30",
     "event_summary": "at 06:30 (+1 condition)", "scene_count": 1, "scenes": ["F2000001-0000-4000-8000-0000000000AA"],
     "action_sets": [{"id": HIDDEN_SET, "name": "F2000001-0000-4000-8000-0000000000AA", "hidden": True, "action_count": 4}]},
    {"id": BUTTON, "name": "Bedside Button", "enabled": False, "trigger_type": "button",
     "event_summary": "Bedside Switch single_press", "scene_count": 1, "scenes": ["Good Night"],
     "action_sets": [{"id": SCENES[0]["id"], "name": "Good Night", "action_count": 2}]},
]
AUTOMATION_DETAIL = {MORNING: {"conditions": [{"type": "weekdays", "weekdays": [2, 3, 4, 5, 6], "summary": "Mon, Tue, Wed, Thu, Fri"}],
                               "events": []},
                     BUTTON: {"conditions": None, "events": [{"type": "characteristic", "trigger_type": "button"}]}}
SCENE_ACTIONS = {
    HIDDEN_SET: [{"accessory": "Floor Lamp", "accessory_id": FLOOR_LAMP, "characteristic": "power", "value": "1"},
                 {"accessory": "Floor Lamp", "accessory_id": FLOOR_LAMP, "characteristic": "brightness", "value": "30"},
                 {"accessory": "Awning Roof", "accessory_id": AWNING, "characteristic": "target_position", "value": "0"},
                 {"accessory": "Radiator", "accessory_id": THERMOSTAT, "characteristic": "target_temperature", "value": "21"}],
    SCENES[0]["id"]: [{"accessory": "Floor Lamp", "accessory_id": FLOOR_LAMP, "characteristic": "power", "value": "0"}],
}


class FakeHomeClaw:
    """Serves `replies` on a Unix socket under a short temp dir (macOS caps the path at 104
    bytes, which pytest's tmp_path can pass); `requests` records every request decoded."""

    def __init__(self, replies: dict):
        self.dir = tempfile.mkdtemp(prefix="hc", dir="/tmp")
        self.path = os.path.join(self.dir, "homeclaw.sock")
        self.replies = replies
        self.requests: list[dict] = []
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(self.path)
        self._server.listen(8)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while True:
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            with conn:
                raw = b""
                while not raw.endswith(b"\n"):
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    raw += chunk
                request = json.loads(raw)
                self.requests.append(request)
                reply = self.replies.get(request["command"], {"success": False, "error": f"Unknown command: {request['command']}"})
                if callable(reply):
                    reply = reply(request.get("args") or {})
                conn.sendall(reply if isinstance(reply, bytes) else (json.dumps(reply) + "\n").encode())

    def of(self, command: str) -> list[dict]:
        return [r.get("args") or {} for r in self.requests if r["command"] == command]

    def close(self):
        self._server.close()
        shutil.rmtree(self.dir, ignore_errors=True)


def ok(data):
    return {"success": True, "data": data}


def house_replies() -> dict:
    return {
        "status": ok({"ready": True, "homes": 1, "accessories": len(ACCESSORIES)}),
        "list_accessories": ok(ACCESSORIES),
        "list_scenes": ok(SCENES),
        "get_accessory": lambda args: ok(WATER_CONTROL_DETAIL) if args.get("id") == WATER_CONTROL
        else {"success": False, "error": "Accessory not found"},
        "control": lambda args: ok({"valid": True, "dry_run": True, "current_value": "0", "new_value": args["value"]}
                                   if args.get("dry_run") else {"verified": True, "new_value": args["value"]}),
        "trigger_scene": ok({"triggered": True}),
        "events": ok({"events": []}),
        "list_automations": ok(AUTOMATIONS),
        "get_automation": lambda args: ok({**next(a for a in AUTOMATIONS if a["id"] == args["id"]),
                                           **AUTOMATION_DETAIL[args["id"]]}),
        "get_scene": lambda args: ok({"id": args["id"], "actions": SCENE_ACTIONS[args["id"]]}),
    }


@pytest.fixture
def app():
    fake = FakeHomeClaw(house_replies())
    yield fake
    fake.close()


def settings_patch(values: dict):
    """Patches the integration's config with `values`; any other key gets the caller's default."""
    def get(key, default=None):
        if key in values:
            return values[key]
        if default is None:
            raise KeyError(key)
        return default
    return patch("integrations.homeclaw.HomeClawIntegration.config", **{"get.side_effect": get})


@pytest.fixture
def homeclaw(app):
    with settings_patch({"homeclaw.socket": app.path}):
        integration = HomeClawIntegration()
        yield integration
