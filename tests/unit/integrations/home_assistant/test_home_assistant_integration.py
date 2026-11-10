"""HomeAssistantIntegration with the websocket client faked: the contract, the mirror and the
snapshot, the commands' service calls and answers, the error log, the health check and the
feed (plans/HOME_ASSISTANT_PLAN.md)."""
import json
import queue
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from integrations.home_assistant.EntityRules import EntityRules
from integrations.home_assistant.HomeAssistantClient import HAError, NotConnected
from integrations.home_assistant.HomeAssistantIntegration import HomeAssistantIntegration, ERROR_LOG_LINES, REGISTRY_EVENTS, REGISTRY_TTL
from integrations.home_assistant.HomeAssistantModel import DEFAULT_FEED_DOMAINS, SECTION_OF, TICKING_DOMAINS
from common import ConfigSchema, ConfigStore
from common.ConfigManager import ConfigManager
from assistant.integration.HomeAutomation import render

MODULE = "integrations.home_assistant.HomeAssistantIntegration"
DATA = Path(__file__).parents[2] / "data" / "home" / "ha"
CONFIG = {"home_assistant.url": "http://ha.local:8123/", "home_assistant.token": "tok", "home_assistant.exposed_only": True}
# Every domain the feed can follow — the scenarios below are about lamps, covers and doors, which the default
# `feed_domains` (the house's doings and its people) leaves out.
FED = [d for d in SECTION_OF if d not in TICKING_DOMAINS]


class FakeClient:
    """Answers `call` from a table, remembers every call, and lets a test flip its state."""

    def __init__(self):
        raw = json.loads((DATA / "registries.json").read_text())
        self.results = {
            "get_states": json.loads((DATA / "states.json").read_text()),
            "config/area_registry/list": raw["areas"],
            "config/entity_registry/list": raw["entities"],
            "config/device_registry/list": raw["devices"],
            "homeassistant/expose_entity/list": {"exposed_entities": raw["exposed"]},
            "homeassistant/expose_new_entities/get": {"expose_new": raw["expose_new"]},
            "config/auth/list": [{"id": "u1", "name": "Marcus"}],
            "call_service": {"context": {"id": "ctx"}},
            "ping": None,
        }
        self.calls: list[tuple[str, dict]] = []
        self.handlers: dict[str, list] = {}
        self.hooks = []
        self.url = "ws://ha.local:8123/api/websocket"
        self.connected = True
        self.ready = True
        self.auth_failed = False
        self.last_error = None
        self.disconnected_at = None
        self.started = 0
        self.stopped = 0

    def subscribe(self, event_type, handler):
        self.handlers.setdefault(event_type, []).append(handler)

    def on_connect(self, hook):
        self.hooks.append(hook)

    def start(self):
        self.started += 1
        if self.connected:      # the real client runs the hooks only after auth_ok
            for hook in self.hooks:
                try:
                    hook()
                except Exception:   # logged and swallowed by the real client, which counts as ready anyway
                    pass

    def stop(self):
        self.stopped += 1

    def wait_connected(self, timeout):
        return self.connected

    def wait_ready(self, timeout):
        return self.ready

    def call(self, type, timeout=10, **payload):
        self.calls.append((type, payload))
        if not self.connected:
            raise NotConnected("not connected")
        result = self.results.get(type, KeyError)
        if result is KeyError:
            raise HAError("unknown_command", f"Unknown command {type}")
        if isinstance(result, Exception):
            raise result
        return result

    def push(self, event_type, data):
        for handler in self.handlers.get(event_type, ()):
            handler({"event_type": event_type, "data": data})


def state(entity_id, value, user_id=None, **attributes):
    return {"entity_id": entity_id, "state": value, "attributes": attributes,
            "last_changed": "2026-09-15T10:00:00+00:00", "last_updated": "2026-09-15T10:00:00+00:00",
            "context": {"id": "c", "parent_id": None, "user_id": user_id}}


@pytest.fixture
def cfg():
    return dict(CONFIG)


@pytest.fixture
def rules_path(tmp_path):
    with patch.object(EntityRules, "PATH", tmp_path / "rules.txt"):
        yield tmp_path / "rules.txt"


@pytest.fixture
def integration(cfg, rules_path):
    HomeAssistantIntegration.reset_instance()
    with patch(f"{MODULE}.config") as config:
        config.get.side_effect = lambda key, default=None: cfg.get(key, default)
        yield HomeAssistantIntegration()
    HomeAssistantIntegration.reset_instance()


@pytest.fixture
def fake(integration):
    client = FakeClient()
    with patch(f"{MODULE}.HomeAssistantClient", return_value=client):
        yield client


@pytest.mark.unit
class TestContract:
    def test_base_url_is_normalised(self, integration, cfg):
        assert integration.base_url == "http://ha.local:8123"
        cfg["home_assistant.url"] = "ha.local:8123"
        assert integration.base_url == "http://ha.local:8123"

@pytest.mark.unit
class TestClientWiring:
    def test_client_is_built_from_config_started_once_and_wired(self, integration, fake):
        with patch(f"{MODULE}.HomeAssistantClient", return_value=fake) as cls:
            assert integration.client() is fake
            assert integration.client() is fake
        cls.assert_called_once_with("http://ha.local:8123", "tok")
        assert fake.started == 1
        assert set(fake.handlers) == {"state_changed", *REGISTRY_EVENTS}
        assert fake.hooks == [integration._refill]

    def test_refill_loads_states_registries_exposure_and_users(self, integration, fake):
        integration.client()
        kinds = [c[0] for c in fake.calls]
        assert kinds[:1] == ["get_states"]
        assert {"config/area_registry/list", "config/entity_registry/list", "config/device_registry/list",
                "homeassistant/expose_entity/list", "homeassistant/expose_new_entities/get", "config/auth/list"} <= set(kinds)
        assert ("homeassistant/expose_new_entities/get", {"assistant": "conversation"}) in fake.calls
        assert integration._users == {"u1": "Marcus"}
        assert len(integration._states) == 25
        assert integration._registries.areas["kok"]["name"] == "Kök"

    def test_refused_registries_keep_the_last_ones(self, integration, fake):
        integration.client()
        fake.results["config/area_registry/list"] = HAError("unauthorized", "Unauthorized")
        fake.results["homeassistant/expose_entity/list"] = HAError("unauthorized", "Unauthorized")
        fake.results["config/auth/list"] = HAError("unauthorized", "Unauthorized")
        integration._load_registries()
        assert integration._registries.areas["kok"]["name"] == "Kök"
        assert integration._registries.exposed["sensor.power"] == {"conversation": True}

    def test_a_failed_fill_at_connect_is_retried_by_the_next_snapshot(self, integration, fake):
        """The client swallows a failing on_connect hook and counts as ready: without a retry
        the mirror would stay empty (an empty house) until the next socket drop."""
        fake.results["get_states"] = TimeoutError("get_states timed out after 10s")
        integration.client()
        assert integration._filled is False
        with pytest.raises(TimeoutError):
            integration.snapshot()
        fake.results["get_states"] = json.loads((DATA / "states.json").read_text())
        assert integration.snapshot().is_empty() is False
        assert integration._filled is True
        assert [c for c, _ in fake.calls].count("get_states") == 3

    def test_a_refill_that_fails_marks_the_mirror_unfilled(self, integration, fake):
        integration.client()
        assert integration._filled is True
        fake.results["get_states"] = NotConnected("connection dropped")
        with pytest.raises(NotConnected):
            integration._refill()
        assert integration._filled is False

    def test_reset_stops_the_feed_and_the_client(self, integration, fake):
        integration.client()
        with patch("assistant.integration.FeedRunner.Feeds.get_instance") as feeds:
            integration._on_reset()
        feeds.return_value.stop.assert_called_once_with("Home Assistant")
        assert fake.stopped == 1 and integration._client is None


@pytest.mark.unit
class TestSnapshot:
    def test_snapshot_renders_the_golden_text(self, integration, fake):
        snapshot = integration.snapshot()
        assert render(snapshot) == (DATA / "rendered.md").read_text()
        assert snapshot.taken_at > 0
        assert snapshot.flag("Night") is True

    def test_snapshot_is_a_memory_read_after_the_first(self, integration, fake):
        integration.snapshot()
        calls = len(fake.calls)
        integration.snapshot()
        assert len(fake.calls) == calls

    def test_events_update_the_mirror_and_rebuild_lazily(self, integration, fake):
        integration.snapshot()
        fake.push("state_changed", {"entity_id": "light.hall", "old_state": state("light.hall", "on"),
                                    "new_state": state("light.hall", "off", friendly_name="Hall light")})
        fake.push("state_changed", {"entity_id": "light.new", "old_state": None,
                                    "new_state": state("light.new", "on", friendly_name="New lamp")})
        fake.push("state_changed", {"entity_id": "light.garage", "old_state": state("light.garage", "unavailable"),
                                    "new_state": None})
        lights = {e.name: e.value for e in integration.snapshot().lights}
        assert lights["Hall light"] == "off" and lights["New lamp"] == "on" and "Garage light" not in lights

    def test_registry_event_reloads_the_registries_on_the_next_snapshot(self, integration, fake):
        integration.snapshot()
        fake.results["config/area_registry/list"] = [{"area_id": "hall", "name": "Hallway", "aliases": []}]
        fake.push("area_registry_updated", {"action": "update", "area_id": "hall"})
        calls = len(fake.calls)
        lights = {e.name: e.area for e in integration.snapshot().lights}
        assert lights["Hall light"] == "Hallway"
        assert len(fake.calls) > calls

    def test_registries_reload_after_the_ttl(self, integration, fake):
        integration.snapshot()
        with patch(f"{MODULE}.time.time", return_value=integration._registries_at + REGISTRY_TTL + 1):
            integration.snapshot()
        assert [c[0] for c in fake.calls].count("config/area_registry/list") == 2

    def test_snapshot_raises_while_the_socket_is_down(self, integration, fake):
        fake.ready = False
        fake.disconnected_at = 1_789_466_400.0
        fake.last_error = "ConnectionRefusedError"
        with pytest.raises(RuntimeError, match="websocket down since .*ConnectionRefusedError"):
            integration.snapshot()

    def test_snapshot_raises_with_the_auth_error(self, integration, fake):
        fake.auth_failed = True
        fake.last_error = "token rejected"
        with pytest.raises(RuntimeError, match="token rejected"):
            integration.snapshot()

    def test_exposed_only_false_shows_everything_but_the_hidden(self, integration, fake, cfg):
        cfg["home_assistant.exposed_only"] = False
        names = {e.name for e in integration.snapshot().lights}
        assert "Bedroom lamp" in names
        assert "binary_sensor.hidden_motion" not in integration._index

    def test_a_tick_of_an_entity_the_data_never_shows_costs_no_rebuild(self, integration, fake):
        integration.snapshot()
        fake.push("state_changed", {"entity_id": "sensor.pressure", "old_state": state("sensor.pressure", "1001"),
                                    "new_state": state("sensor.pressure", "1002")})       # not exposed
        fake.push("state_changed", {"entity_id": "binary_sensor.hidden_motion",
                                    "old_state": state("binary_sensor.hidden_motion", "off"),
                                    "new_state": state("binary_sensor.hidden_motion", "on")})
        assert integration._dirty is False
        assert integration._states["sensor.pressure"]["state"] == "1002"      # the mirror still follows
        fake.push("state_changed", {"entity_id": "sensor.power", "old_state": state("sensor.power", "100"),
                                    "new_state": state("sensor.power", "120")})           # exposed, listed
        assert integration._dirty is True

    def test_a_listed_entity_vanishing_rebuilds(self, integration, fake):
        integration.snapshot()
        fake.push("state_changed", {"entity_id": "light.hall", "old_state": state("light.hall", "on"), "new_state": None})
        assert integration._dirty is True


@pytest.mark.unit
class TestRules:
    def test_an_exclude_rule_takes_its_entities_out_of_the_data_and_persists(self, integration, fake, rules_path):
        integration.snapshot()
        reply = integration.exclude("sensor")
        assert "sensor.*" in reply and "sensor.power" in reply
        assert integration.snapshot().sensors and not [e for e in integration.snapshot().sensors if e.name == "Power"]
        assert "sensor.power" not in integration._index
        assert EntityRules(rules_path).load().verdict("sensor.power") == "exclude"

    def test_an_excluded_entity_no_longer_dirties_nor_feeds(self, integration, fake):
        integration.snapshot()
        integration.exclude("binary_sensor.front_*")
        integration.snapshot()
        fake.push("state_changed", {"entity_id": "binary_sensor.front_door",
                                    "old_state": state("binary_sensor.front_door", "off"),
                                    "new_state": state("binary_sensor.front_door", "on", device_class="door")})
        assert integration._dirty is False
        assert integration.fetch(None, backfill=False).entries == []

    def test_an_include_rule_shows_a_hidden_entity_and_its_changes(self, integration, fake, cfg):
        cfg["home_assistant.feed_domains"] = FED
        integration.snapshot()
        reply = integration.include("binary_sensor.hidden_motion")
        assert "1 entity" in reply
        integration.snapshot()
        assert "binary_sensor.hidden_motion" in integration._index
        fake.push("state_changed", {"entity_id": "binary_sensor.hidden_motion",
                                    "old_state": state("binary_sensor.hidden_motion", "off"),
                                    "new_state": state("binary_sensor.hidden_motion", "on", friendly_name="Hidden motion")})
        assert [e.content for e in integration.fetch(None, backfill=False).entries] == ["Hidden motion [Hall] turned on"]

    def test_clearing_a_rule_restores_home_assistants_settings(self, integration, fake):
        integration.snapshot()
        integration.include("binary_sensor.hidden_motion")
        integration.snapshot()
        assert "include" in integration.clear_rule("binary_sensor.hidden_motion")
        integration.snapshot()
        assert "binary_sensor.hidden_motion" not in integration._index
        assert integration.clear_rule("binary_sensor.hidden_motion").startswith("No rule")

    def test_a_pattern_matching_nothing_says_so_and_a_bad_one_is_refused(self, integration, fake):
        integration.snapshot()
        assert "no entity right now" in integration.exclude("sensor.unifi_*")
        assert "sensor.unifi_*" in integration.rules_text()
        assert "not an entity id pattern" in integration.exclude("the kitchen lamp")
        assert "No pattern" in integration.include("")

    def test_the_rules_survive_a_restart_of_the_integration(self, integration, fake):
        integration.exclude("sensor.pressure")
        assert HomeAssistantIntegration().rules_text().startswith("exclude sensor.pressure")


@pytest.mark.unit
class TestCommands:
    def last_service(self, fake):
        return [c for c in fake.calls if c[0] == "call_service"][-1][1]

    def test_turn_on_and_off_by_domain(self, integration, fake):
        assert integration.turn_on("Hall light (Hall)") == ""
        assert self.last_service(fake) == {"domain": "light", "service": "turn_on", "target": {"entity_id": "light.hall"}}
        assert integration.turn_off("Blinds") == ""
        assert self.last_service(fake) == {"domain": "cover", "service": "close_cover", "target": {"entity_id": "cover.blinds"}}
        assert integration.turn_on("Front door lock") == ""
        assert self.last_service(fake)["service"] == "lock"
        assert integration.turn_off("Front door lock") == ""
        assert self.last_service(fake)["service"] == "unlock"
        assert integration.turn_on("Night") == ""
        assert self.last_service(fake) == {"domain": "input_boolean", "service": "turn_on", "target": {"entity_id": "input_boolean.night"}}
        assert integration.turn_off("Sunset lights") == ""
        assert self.last_service(fake)["domain"] == "automation" and self.last_service(fake)["service"] == "turn_off"

    def test_dim_by_domain_and_clamping(self, integration, fake):
        assert integration.dim("Hall light [Hall]", 40) == ""
        assert self.last_service(fake) == {"domain": "light", "service": "turn_on", "target": {"entity_id": "light.hall"},
                                           "service_data": {"brightness_pct": 40}}
        assert integration.dim("Bedroom fan", 150) == ""
        assert self.last_service(fake)["service_data"] == {"percentage": 100}
        assert integration.dim("Blinds", "25") == ""
        assert self.last_service(fake) == {"domain": "cover", "service": "set_cover_position",
                                           "target": {"entity_id": "cover.blinds"}, "service_data": {"position": 25}}

    def test_stop_halts_a_cover_and_nothing_else(self, integration, fake):
        """The home toolkit promises stop(name) for every cover in the house list, and Home
        Assistant lists its covers there."""
        assert integration.stop("Blinds") == ""
        assert self.last_service(fake) == {"domain": "cover", "service": "stop_cover", "target": {"entity_id": "cover.blinds"}}
        calls = len(fake.calls)
        answer = integration.stop("Hall light")
        assert "cover" in answer and "Hall light" in answer
        assert len(fake.calls) == calls, "a light is not sent a stop"

    def test_dim_to_zero_turns_off(self, integration, fake):
        assert integration.dim("Hall light", 0) == ""
        assert self.last_service(fake)["service"] == "turn_off"

    def test_dim_on_something_not_dimmable(self, integration, fake):
        assert integration.dim("Front door lock", 50) == "Front door lock [Hall] is not dimmable."
        assert not [c for c in fake.calls if c[0] == "call_service"]

    def test_execute_by_domain(self, integration, fake):
        assert integration.execute("Movie night") == ""
        assert self.last_service(fake) == {"domain": "scene", "service": "turn_on", "target": {"entity_id": "scene.movie_night"}}
        assert integration.execute("Goodnight") == ""
        assert self.last_service(fake)["domain"] == "script"
        assert integration.execute("Sunset lights") == ""
        assert self.last_service(fake)["service"] == "trigger"


    def test_the_domains_rendered_as_sensors_still_switch(self, integration, fake):
        """A thermostat or a speaker is listed under SENSORS, yet turn_on / turn_off act on
        it — the pairing a refactor of the tables would break first."""
        for name, domain in (("Kitchen spots switch", "switch"), ("Bedroom fan", "fan"),
                             ("Thermostat", "climate"), ("Living room speaker", "media_player")):
            assert integration.turn_on(name) == ""
            assert (self.last_service(fake)["domain"], self.last_service(fake)["service"]) == (domain, "turn_on")
            assert integration.turn_off(name) == ""
            assert (self.last_service(fake)["domain"], self.last_service(fake)["service"]) == (domain, "turn_off")

    def test_a_button_is_pressed(self, integration, fake, cfg):
        cfg["home_assistant.exposed_only"] = False    # a button is not exposed to Assist by default
        integration.snapshot()
        with integration._lock:
            integration._states["button.doorbell"] = state("button.doorbell", "unknown", friendly_name="Doorbell")
            integration._dirty = True
        assert integration.execute("Doorbell") == ""
        assert self.last_service(fake) == {"domain": "button", "service": "press", "target": {"entity_id": "button.doorbell"}}
        assert integration.turn_on("Doorbell") == "Doorbell is a button; use execute."

    def test_the_wrong_command_points_at_the_right_one(self, integration, fake):
        assert integration.execute("Hall light") == "Hall light [Hall] is a light; use turn_on / turn_off."
        assert integration.turn_on("Movie night") == "Movie night is a scene; use execute."
        assert integration.turn_on("Temperature (Kök)") == "Temperature [Kök] is a sensor and cannot be controlled with turn_on."
        assert not [c for c in fake.calls if c[0] == "call_service"]

    def test_unresolved_names_answer_with_the_index_text(self, integration, fake):
        assert integration.turn_on("Temperature") == '"Temperature" could be Temperature [Sovrum], Temperature [Kök] — say which.'
        assert integration.turn_on("Nigth") == 'No entity named "Nigth". Did you mean: Night?'

    def test_server_errors_come_back_as_text(self, integration, fake):
        integration.snapshot()
        fake.results["call_service"] = HAError("service_validation_error", "Validation error: brightness out of range")
        assert integration.dim("Hall light", 40) == "Validation error: brightness out of range"
        fake.results["call_service"] = TimeoutError("call_service timed out after 10s")
        assert integration.turn_on("Hall light") == "Request timed out"
        fake.results["call_service"] = NotConnected("connection dropped")
        assert integration.turn_on("Hall light") == "Could not connect to Home Assistant"

    @pytest.mark.parametrize("failure, expected", [
        (TimeoutError("config/entity_registry/list timed out after 10s"), "Request timed out"),
        (NotConnected("connection dropped"), "Could not connect to Home Assistant"),
    ])
    def test_a_registry_reload_failing_under_a_command_answers_with_text(self, integration, fake, failure, expected):
        """A stale registry is reloaded on the way to resolving a name; the reload's
        errors must become the LLM's answer, not escape the command."""
        integration.snapshot()
        fake.push("entity_registry_updated", {"action": "update"})
        fake.results["config/entity_registry/list"] = failure
        assert integration.turn_on("Hall light") == expected
        assert not any(c == "call_service" for c, _ in fake.calls)

    def test_commands_while_down_cannot_connect(self, integration, fake):
        fake.ready = False
        assert integration.turn_on("Hall light") == "Could not connect to Home Assistant"
        assert integration.execute("Goodnight") == "Could not connect to Home Assistant"


@pytest.mark.unit
class TestHealthAndErrorLog:
    def test_health_is_a_ping(self, integration, fake):
        assert integration.check_health() is True
        assert ("ping", {}) in fake.calls

    def test_health_raises_the_auth_error(self, integration, fake):
        fake.auth_failed = True
        fake.last_error = "token rejected (Invalid access token)"
        with pytest.raises(RuntimeError, match="token rejected"):
            integration.check_health()

    def test_health_raises_when_not_connected(self, integration, fake):
        fake.connected = False
        fake.last_error = "ConnectionRefusedError"
        with pytest.raises(RuntimeError, match="ConnectionRefusedError"):
            integration.check_health()

    def test_error_log_is_the_last_lines_over_rest(self, integration):
        lines = [f"2026-09-15 10:00:{i:02d} ERROR (MainThread) [homeassistant.components.hue] failure (attempt {i})" for i in range(100)]
        response = Mock(text="\n".join(lines) + "\n\n")
        with patch(f"{MODULE}.requests.get", return_value=response) as get:
            text = integration.error_logs()
        get.assert_called_once_with("http://ha.local:8123/api/error_log", timeout=10,
                                    headers={"Authorization": "Bearer tok"})
        body = text.split("\n")[1:]
        assert text.startswith("## HOME ASSISTANT ERROR LOG\n")
        assert len(body) == ERROR_LOG_LINES and body[-1].endswith("[attempt 99]")

    def test_empty_error_log(self, integration):
        with patch(f"{MODULE}.requests.get", return_value=Mock(text="\n")):
            assert integration.error_logs() == "Home Assistant reported no errors."

    def test_error_log_http_failure_raises(self, integration):
        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError("401")
        with patch(f"{MODULE}.requests.get", return_value=response), pytest.raises(requests.HTTPError):
            integration.error_logs()


@pytest.mark.unit
class TestFeed:
    def test_open_starts_the_client_and_fetch_drains_the_queue(self, integration, fake, cfg):
        cfg["home_assistant.feed_domains"] = FED
        assert integration.open_feed() is True
        assert fake.started == 1
        assert integration.fetch(None, backfill=False).entries == []
        integration.snapshot()
        fake.push("state_changed", {"entity_id": "light.hall", "old_state": state("light.hall", "off"),
                                    "new_state": state("light.hall", "on", user_id="u1", brightness=51)})
        fake.push("state_changed", {"entity_id": "sensor.kitchen_temperature",
                                    "old_state": state("sensor.kitchen_temperature", "21.4"),
                                    "new_state": state("sensor.kitchen_temperature", "21.5")})
        batch = integration.fetch(None, backfill=False)
        assert [e.content for e in batch.entries] == ["Hall light [Hall] turned on 20% by Marcus"]
        assert batch.cursor is None and batch.exhausted
        assert integration.fetch(None, backfill=False).entries == []

    def test_an_unknown_user_id_names_nobody(self, integration, fake, cfg):
        cfg["home_assistant.feed_domains"] = FED
        integration.snapshot()
        fake.push("state_changed", {"entity_id": "light.hall", "old_state": state("light.hall", "off"),
                                    "new_state": state("light.hall", "on", user_id="ghost")})
        assert integration.fetch(None, backfill=False).entries[0].content == "Hall light [Hall] turned on"

    def test_events_before_the_first_snapshot_use_the_friendly_name(self, integration, fake, cfg):
        cfg["home_assistant.feed_domains"] = FED
        integration.client()
        fake.push("state_changed", {"entity_id": "light.hall", "old_state": state("light.hall", "off"),
                                    "new_state": state("light.hall", "on", friendly_name="Hall light")})
        assert integration.fetch(None, backfill=False).entries[0].content == "Hall light turned on"

    def test_a_malformed_event_is_ignored(self, integration, fake):
        integration.client()
        fake.push("state_changed", {})
        fake.push("state_changed", {"entity_id": "light.hall"})
        assert integration.fetch(None, backfill=False).entries == []
        assert isinstance(integration._live, queue.Queue)

    def test_without_the_key_the_feed_tells_what_the_house_did_not_which_lamp_went_on(self, integration, fake, cfg):
        """`feed_domains` unset reads like Indigo's narrated log: an automation triggered is a
        row, a lamp switched by hand is in the house data but no row."""
        assert "home_assistant.feed_domains" not in cfg
        integration.snapshot()
        fake.push("state_changed", {"entity_id": "light.hall", "old_state": state("light.hall", "off"),
                                    "new_state": state("light.hall", "on", user_id="u1")})
        fake.push("state_changed", {"entity_id": "automation.sunset_lights",
                                    "old_state": state("automation.sunset_lights", "on", last_triggered="2026-09-14T18:45:00+00:00"),
                                    "new_state": state("automation.sunset_lights", "on", last_triggered="2026-09-15T18:45:00+00:00")})
        assert [e.content for e in integration.fetch(None, backfill=False).entries] == ["Sunset lights [automation] triggered"]
        assert integration.snapshot().lights, "the lamp is still house data"

    def test_a_config_file_without_the_key_is_the_default_list_not_an_error(self, integration, tmp_path, monkeypatch):
        """Issue #174: the dict standing in for the config answers a missing key with the default;
        the real ConfigManager raised, on every event of every install without the key."""
        (tmp_path / "config.yaml").write_text("home_assistant:\n  url: http://ha.local:8123\n")
        monkeypatch.setenv("FLOXY_CONFIG", str(tmp_path / "config.yaml"))
        with patch(f"{MODULE}.config", ConfigManager()):
            assert integration.feed_domains == DEFAULT_FEED_DOMAINS

    def test_the_example_lists_the_default_so_one_domain_can_be_set_by_index(self):
        """Issue #174: with the key commented out the schema knew no list there, and
        config_set refused `feed_domains[0]`."""
        assert tuple(ConfigSchema.example_value("home_assistant.feed_domains")) == DEFAULT_FEED_DOMAINS
        ConfigStore.check_key("home_assistant.feed_domains[0]")

    def test_a_changed_key_is_read_again_without_a_restart(self, integration, fake, cfg):
        integration.snapshot()
        lamp = {"entity_id": "light.hall", "old_state": state("light.hall", "off"), "new_state": state("light.hall", "on")}
        fake.push("state_changed", lamp)
        assert integration.fetch(None, backfill=False).entries == []
        cfg["home_assistant.feed_domains"] = ["light"]
        fake.push("state_changed", lamp)
        assert [e.content for e in integration.fetch(None, backfill=False).entries] == ["Hall light [Hall] turned on"]
        cfg["home_assistant.feed_domains"] = []
        fake.push("state_changed", lamp)
        assert integration.fetch(None, backfill=False).entries == [], "an empty list is a silent feed"

    def test_the_key_is_parsed_once_per_value_not_once_per_event(self, integration, fake, cfg):
        """The parse logs what it leaves out; asked per event, it would say so per event."""
        from integrations.home_assistant import HomeAssistantModel
        integration.snapshot()
        lamp = {"entity_id": "light.hall", "old_state": state("light.hall", "off"), "new_state": state("light.hall", "on")}
        with patch(f"{MODULE}.feed_domains", wraps=HomeAssistantModel.feed_domains) as parse:
            for _ in range(3):
                fake.push("state_changed", lamp)
            assert parse.call_count == 1
            cfg["home_assistant.feed_domains"] = ["light", "sensor"]
            fake.push("state_changed", lamp)
            cfg["home_assistant.feed_domains"] = ["light", "sensor"]
            fake.push("state_changed", lamp)
            assert parse.call_count == 2, "a new value is parsed once; the same value again is not"
        assert len(integration.fetch(None, backfill=False).entries) == 2


T0 = 1789466400.0       # 2026-09-15T10:00:00+00:00, the fixture's clock


def at(seconds: float) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(T0 + seconds, tz=timezone.utc).isoformat()


FIXTURE_ATTRIBUTES = {s["entity_id"]: s.get("attributes") or {} for s in json.loads((DATA / "states.json").read_text())}


def timed(entity_id, value, seconds, **attributes):
    """The fixture's entity (its friendly name and attributes kept, so the mirror still names it
    as the snapshot does) in state `value` at T0 + `seconds`."""
    s = state(entity_id, value, **{**FIXTURE_ATTRIBUTES.get(entity_id, {}), **attributes})
    s["last_changed"] = s["last_updated"] = at(seconds)
    return s


@pytest.mark.unit
class TestAvailabilityHold:
    """An entity going unavailable is held UNAVAILABLE_HOLD_S before it is a row; a hub hiccup
    is nothing at all (plans/PROMPT_CHURN_PLAN.md, phase 3)."""

    HUB = ("light.hall", "light.kitchen_spots", "cover.blinds")

    @pytest.fixture(autouse=True)
    def lamps_fed(self, cfg):
        cfg["home_assistant.feed_domains"] = FED

    @pytest.fixture
    def hub(self, integration, fake):
        """The three entities hang off the Hue spots device, which hangs off the Hue bridge."""
        for entry in fake.results["config/entity_registry/list"]:
            if entry["entity_id"] in self.HUB:
                entry["device_id"] = "dev_hue"
        integration.snapshot()
        return {"light.hall": "on", "light.kitchen_spots": "on", "cover.blinds": "open"}

    def _outage(self, fake, hub, seconds):
        for entity_id, value in hub.items():
            fake.push("state_changed", {"entity_id": entity_id, "old_state": timed(entity_id, value, 0),
                                        "new_state": timed(entity_id, "unavailable", seconds)})

    def _recovery(self, fake, hub, seconds, values=None):
        for entity_id, value in (values or hub).items():
            fake.push("state_changed", {"entity_id": entity_id, "old_state": timed(entity_id, "unavailable", 0),
                                        "new_state": timed(entity_id, value, seconds)})

    def test_a_hub_away_for_forty_seconds_is_nothing(self, integration, fake, hub):
        self._outage(fake, hub, 0)
        assert integration.fetch(None, backfill=False, now=T0 + 20).entries == []
        self._recovery(fake, hub, 37)
        assert integration.fetch(None, backfill=False, now=T0 + 40).entries == []
        assert integration.fetch(None, backfill=False, now=T0 + 900).entries == [], "nothing is released later either"

    def test_a_hub_away_past_the_hold_is_one_row_and_one_row_back(self, integration, fake, hub):
        self._outage(fake, hub, 0)
        assert integration.fetch(None, backfill=False, now=T0 + 299).entries == []
        rows = integration.fetch(None, backfill=False, now=T0 + 360).entries
        assert len(rows) == 1
        assert rows[0].content.startswith("Hue bridge")
        for name in ("Hall light [Hall]", "Kitchen spots light [Kök]", "Blinds [Sovrum]"):
            assert name in rows[0].content
        assert rows[0].timestamp == T0 + 360
        assert integration.fetch(None, backfill=False, now=T0 + 600).entries == [], "an outage is reported once"
        self._recovery(fake, hub, 700)
        back = integration.fetch(None, backfill=False, now=T0 + 702).entries
        assert len(back) == 1 and back[0].content.startswith("Hue bridge")
        assert "Blinds [Sovrum]" in back[0].content and back[0].timestamp == T0 + 700
        assert integration.fetch(None, backfill=False, now=T0 + 900).entries == []

    def _reconnect(self, fake, states):
        """The socket came back: the handshake's refill loads a fresh table where the hub's
        entities stand in `states` (an event for their recovery never came)."""
        table = {s["entity_id"]: s for s in fake.results["get_states"]}
        for entity_id, (value, seconds) in states.items():
            table[entity_id] = timed(entity_id, value, seconds)
        fake.results["get_states"] = list(table.values())
        fake.start()

    def test_an_outage_still_held_at_a_reconnect_ends_silently_when_the_fresh_table_shows_the_entity_back(self, integration, fake, hub):
        self._outage(fake, hub, 0)
        self._reconnect(fake, {e: ("on" if e.startswith("light") else "open", 100) for e in hub})
        assert integration.fetch(None, backfill=False, now=T0 + 360).entries == [], "no row for an outage long over"
        assert integration._outages == {}

    def test_a_reported_outage_over_at_a_reconnect_is_one_row_back(self, integration, fake, hub):
        self._outage(fake, hub, 0)
        assert len(integration.fetch(None, backfill=False, now=T0 + 360).entries) == 1
        self._reconnect(fake, {e: ("on" if e.startswith("light") else "open", 700) for e in hub})
        back = integration.fetch(None, backfill=False, now=T0 + 702).entries
        assert len(back) == 1 and back[0].content.startswith("Hue bridge") and back[0].timestamp == T0 + 700
        assert integration._outages == {}

    def test_a_reported_outage_of_an_entity_gone_from_the_fresh_table_ends_without_a_row(self, integration, fake, hub):
        # The entity was removed from HA while the socket was down: nothing is "available again",
        # so the reported outage closes without the recovery row, and nothing waits for one.
        self._outage(fake, hub, 0)
        assert len(integration.fetch(None, backfill=False, now=T0 + 360).entries) == 1
        table = {s["entity_id"]: s for s in fake.results["get_states"]}
        table.pop("light.hall")
        for entity_id in ("light.kitchen_spots", "cover.blinds"):
            table[entity_id] = timed(entity_id, "on" if entity_id.startswith("light") else "open", 700)
        fake.results["get_states"] = list(table.values())
        fake.start()
        assert integration._outages == {}
        back = integration.fetch(None, backfill=False, now=T0 + 702).entries
        assert len(back) == 1 and "Hall light" not in back[0].content
        assert integration.fetch(None, backfill=False, now=T0 + 800).entries == []

    def test_an_entity_still_unavailable_at_a_reconnect_keeps_its_outage(self, integration, fake, hub):
        self._outage(fake, hub, 0)
        self._reconnect(fake, {"light.hall": ("on", 100), "light.kitchen_spots": ("unavailable", 0), "cover.blinds": ("unavailable", 0)})
        assert set(integration._outages) == {"light.kitchen_spots", "cover.blinds"}
        rows = integration.fetch(None, backfill=False, now=T0 + 360).entries
        assert len(rows) == 1 and "Hall light" not in rows[0].content and "Blinds [Sovrum]" in rows[0].content

    def test_a_cover_that_moved_during_the_hiccup_is_the_one_real_change(self, integration, fake, hub):
        self._outage(fake, hub, 0)
        self._recovery(fake, hub, 30, {**hub, "cover.blinds": "closed"})
        rows = integration.fetch(None, backfill=False, now=T0 + 31).entries
        assert [r.content for r in rows] == ["Blinds [Sovrum] closed"]

    def test_the_snapshot_holds_the_last_available_state_through_the_hold(self, integration, fake, hub):
        self._outage(fake, hub, 0)
        with patch(f"{MODULE}.time") as clock:
            clock.time.return_value = T0 + 120
            lights = {e.name: e.value for e in integration.snapshot().lights}
            assert lights["Hall light"] == "on" and lights["Blinds"].startswith("open")
            clock.time.return_value = T0 + 400
            lights = {e.name: e.value for e in integration.snapshot().lights}
            assert lights["Hall light"] == "unavailable" and lights["Blinds"] == "unavailable"

    def test_an_outage_that_began_before_the_process_watched_ends_silently(self, integration, fake, hub):
        self._recovery(fake, {"light.hall": "on"}, 10)
        assert integration.fetch(None, backfill=False, now=T0 + 11).entries == []

    def test_a_diagnostic_entity_and_a_vanished_one_never_become_rows(self, integration, fake):
        for entry in fake.results["config/entity_registry/list"]:
            if entry["entity_id"] == "light.hall":
                entry["entity_category"] = "diagnostic"
        integration.snapshot()
        fake.push("state_changed", {"entity_id": "light.hall", "old_state": timed("light.hall", "on", 0),
                                    "new_state": timed("light.hall", "unavailable", 0)})
        fake.push("state_changed", {"entity_id": "cover.blinds", "old_state": timed("cover.blinds", "open", 0),
                                    "new_state": timed("cover.blinds", "unavailable", 0)})
        fake.push("state_changed", {"entity_id": "cover.blinds", "old_state": timed("cover.blinds", "unavailable", 0),
                                    "new_state": None})
        assert integration.fetch(None, backfill=False, now=T0 + 900).entries == []

    def test_loose_entities_are_grouped_without_a_device_name(self, integration, fake):
        integration.snapshot()
        for entity_id, value in (("light.hall", "on"), ("cover.blinds", "open")):
            fake.push("state_changed", {"entity_id": entity_id, "old_state": timed(entity_id, value, 0),
                                        "new_state": timed(entity_id, "unavailable", 5)})
        rows = integration.fetch(None, backfill=False, now=T0 + 400).entries
        assert len(rows) == 1 and "None" not in rows[0].content
        assert "Hall light [Hall]" in rows[0].content and "Blinds [Sovrum]" in rows[0].content


@pytest.mark.unit
class TestAvailabilityHoldOutsideTheFeed:
    """The hold is the house data's, not the feed's: under the default `feed_domains` a lamp's
    hub dropping out still keeps the snapshot at the last available state through the hold, and
    neither the released outage nor the recovery is a row."""

    HUB = ("light.hall", "light.kitchen_spots", "cover.blinds")

    @pytest.fixture
    def hub(self, integration, fake, cfg):
        assert "home_assistant.feed_domains" not in cfg, "the default list, which leaves lamps and covers out"
        for entry in fake.results["config/entity_registry/list"]:
            if entry["entity_id"] in self.HUB:
                entry["device_id"] = "dev_hue"
        integration.snapshot()
        return {"light.hall": "on", "light.kitchen_spots": "on", "cover.blinds": "open"}

    def _outage(self, fake, hub, seconds):
        for entity_id, value in hub.items():
            fake.push("state_changed", {"entity_id": entity_id, "old_state": timed(entity_id, value, 0),
                                        "new_state": timed(entity_id, "unavailable", seconds)})

    def _recovery(self, fake, hub, seconds):
        for entity_id, value in hub.items():
            fake.push("state_changed", {"entity_id": entity_id, "old_state": timed(entity_id, "unavailable", 0),
                                        "new_state": timed(entity_id, value, seconds)})

    def test_a_hub_blip_leaves_the_house_data_as_it_was(self, integration, fake, hub):
        self._outage(fake, hub, 0)
        assert set(integration._outages) == set(self.HUB)
        with patch(f"{MODULE}.time") as clock:
            clock.time.return_value = T0 + 30
            lights = {e.name: e.value for e in integration.snapshot().lights}
        assert lights["Hall light"] == "on" and lights["Blinds"].startswith("open")
        self._recovery(fake, hub, 40)
        assert integration._outages == {}
        assert integration.fetch(None, backfill=False, now=T0 + 41).entries == []

    def test_an_outage_past_the_hold_shows_in_the_house_data_but_is_no_row_coming_or_going(self, integration, fake, hub):
        self._outage(fake, hub, 0)
        assert integration.fetch(None, backfill=False, now=T0 + 360).entries == []
        with patch(f"{MODULE}.time") as clock:
            clock.time.return_value = T0 + 400
            lights = {e.name: e.value for e in integration.snapshot().lights}
        assert lights["Hall light"] == "unavailable" and lights["Blinds"] == "unavailable"
        self._recovery(fake, hub, 700)
        assert integration._outages == {}
        assert integration.fetch(None, backfill=False, now=T0 + 702).entries == []

    def test_a_domain_listed_after_the_outage_began_gets_its_row_when_the_hold_runs_out(self, integration, fake, hub, cfg):
        self._outage(fake, hub, 0)
        cfg["home_assistant.feed_domains"] = ["light", "cover"]
        rows = integration.fetch(None, backfill=False, now=T0 + 360).entries
        assert len(rows) == 1 and rows[0].content.startswith("Hue bridge")
        self._recovery(fake, hub, 700)
        back = integration.fetch(None, backfill=False, now=T0 + 702).entries
        assert len(back) == 1 and back[0].content.startswith("Hue bridge")


@pytest.mark.unit
class TestUnknownGap:
    """A person entity goes `unknown` while its trackers blip and comes back to the state it
    had; nobody moved, so there is no row (issue #140: four "arrived home" and "left home"
    rows per 15 s Wi-Fi outage, 2026-10-30). A real move during the gap is the one row."""

    PEOPLE = {"person.marcus": "home", "person.charlotte": "home", "person.frances": "not_home",
              "person.ingeborg": "not_home"}

    @pytest.fixture(autouse=True)
    def doors_fed(self, cfg):
        cfg["home_assistant.feed_domains"] = FED

    @pytest.fixture
    def household(self, integration, fake, cfg):
        """Before the first snapshot every entity is fed and named by its friendly name, so the
        whole cast is on the feed without touching the fixture's house; `exposed_only` is off
        since only Marcus is exposed in the fixture and a not-exposed entity's event is dropped
        before the feed sees it (issue #141)."""
        cfg["home_assistant.exposed_only"] = False
        integration.client()
        return {entity_id: (value, entity_id.split(".")[1].capitalize()) for entity_id, value in self.PEOPLE.items()}

    def _move(self, fake, entity_id, before, after, seconds, name):
        fake.push("state_changed", {"entity_id": entity_id, "old_state": timed(entity_id, before, 0, friendly_name=name),
                                    "new_state": timed(entity_id, after, seconds, friendly_name=name)})

    def _blip(self, fake, household, seconds, back=None):
        for entity_id, (value, name) in household.items():
            self._move(fake, entity_id, value, "unknown", seconds, name)
        for entity_id, (value, name) in household.items():
            self._move(fake, entity_id, "unknown", (back or {}).get(entity_id, value), seconds + 15, name)

    def test_the_whole_household_back_from_unknown_to_where_it_was_is_nothing(self, integration, fake, household):
        self._blip(fake, household, 0)
        assert integration.fetch(None, backfill=False, now=T0 + 20).entries == []
        assert integration._known == {}
        self._blip(fake, household, 600)
        assert integration.fetch(None, backfill=False, now=T0 + 700).entries == [], "and the next blip is nothing too"

    def test_a_move_during_the_gap_is_the_one_real_change(self, integration, fake, household):
        self._blip(fake, household, 0, back={"person.marcus": "not_home", "person.frances": "home"})
        rows = integration.fetch(None, backfill=False, now=T0 + 20).entries
        assert sorted(r.content for r in rows) == ["Frances arrived home", "Marcus left home"]
        assert all(r.timestamp == T0 + 15 for r in rows), "stamped when the state came back, not when it went"

    def test_a_person_known_after_a_gap_that_began_before_the_process_watched_is_no_row(self, integration, fake, household):
        self._move(fake, "person.ingeborg", "unknown", "not_home", 5, "Ingeborg")
        self._move(fake, "person.charlotte", "unknown", "home", 5, "Charlotte")
        assert integration.fetch(None, backfill=False, now=T0 + 10).entries == []

    def test_a_person_in_the_snapshot_is_compared_the_same_way(self, integration, fake):
        integration.snapshot()
        self._move(fake, "person.marcus", "home", "unknown", 0, "Marcus")
        self._move(fake, "person.marcus", "unknown", "home", 15, "Marcus")
        assert integration.fetch(None, backfill=False, now=T0 + 20).entries == []
        self._move(fake, "person.marcus", "home", "unknown", 100, "Marcus")
        self._move(fake, "person.marcus", "unknown", "Work", 115, "Marcus")
        assert [r.content for r in integration.fetch(None, backfill=False, now=T0 + 120).entries] == ["Marcus is at Work"]

    def test_a_door_sensor_back_from_unknown_is_no_row_either(self, integration, fake):
        integration.snapshot()
        self._move(fake, "binary_sensor.front_door", "off", "unknown", 0, "Front door")
        self._move(fake, "binary_sensor.front_door", "unknown", "off", 15, "Front door")
        assert integration.fetch(None, backfill=False, now=T0 + 20).entries == []
        self._move(fake, "binary_sensor.front_door", "off", "unknown", 100, "Front door")
        self._move(fake, "binary_sensor.front_door", "unknown", "on", 115, "Front door")
        assert [r.content for r in integration.fetch(None, backfill=False, now=T0 + 120).entries] == ["Front door [Hall] opened"]

    def test_a_scene_activated_out_of_unknown_is_still_a_row(self, integration, fake):
        integration.snapshot()
        self._move(fake, "scene.movie_night", "unknown", "2026-09-15T20:00:00+00:00", 10, "Movie night")
        assert [r.content for r in integration.fetch(None, backfill=False, now=T0 + 20).entries] == ["Movie night [scene] activated"]
        assert integration._known == {}

    def test_unknown_then_unavailable_is_held_as_the_state_before_both(self, integration, fake):
        integration.snapshot()
        self._move(fake, "person.marcus", "home", "unknown", 0, "Marcus")
        self._move(fake, "person.marcus", "unknown", "unavailable", 5, "Marcus")
        assert integration._known == {} and integration._outages["person.marcus"].before["state"] == "home"
        with patch(f"{MODULE}.time") as clock:
            clock.time.return_value = T0 + 60
            assert {e.name: e.value for e in integration.snapshot().variables}["Marcus"] == "home"
        self._move(fake, "person.marcus", "unavailable", "home", 30, "Marcus")
        assert integration.fetch(None, backfill=False, now=T0 + 40).entries == []
        self._move(fake, "person.marcus", "home", "unknown", 100, "Marcus")
        self._move(fake, "person.marcus", "unknown", "unavailable", 105, "Marcus")
        self._move(fake, "person.marcus", "unavailable", "not_home", 130, "Marcus")
        assert [r.content for r in integration.fetch(None, backfill=False, now=T0 + 140).entries] == ["Marcus left home"]

    def test_a_person_gone_from_home_assistant_while_unknown_is_forgotten(self, integration, fake):
        integration.snapshot()
        self._move(fake, "person.marcus", "home", "unknown", 0, "Marcus")
        fake.push("state_changed", {"entity_id": "person.marcus", "old_state": timed("person.marcus", "unknown", 0),
                                    "new_state": None})
        assert integration._known == {}
        assert integration.fetch(None, backfill=False, now=T0 + 20).entries == []

    def test_a_reconnect_drops_the_kept_state_the_fresh_table_outdates(self, integration, fake):
        """The socket dropped while Marcus was unknown and the fresh table shows him away: the
        kept "home" would turn his next blip into a departure that happened who knows when."""
        integration.snapshot()
        self._move(fake, "person.marcus", "home", "unknown", 0, "Marcus")
        table = {s["entity_id"]: s for s in fake.results["get_states"]}
        table["person.marcus"] = timed("person.marcus", "not_home", 300)
        fake.results["get_states"] = list(table.values())
        fake.start()
        assert integration._known == {}
        self._move(fake, "person.marcus", "not_home", "unknown", 400, "Marcus")
        self._move(fake, "person.marcus", "unknown", "not_home", 415, "Marcus")
        assert integration.fetch(None, backfill=False, now=T0 + 420).entries == []

    def test_a_reconnect_keeps_the_state_of_an_entity_still_unknown(self, integration, fake):
        integration.snapshot()
        self._move(fake, "person.marcus", "home", "unknown", 0, "Marcus")
        table = {s["entity_id"]: s for s in fake.results["get_states"]}
        table["person.marcus"] = timed("person.marcus", "unknown", 0)
        fake.results["get_states"] = list(table.values())
        fake.start()
        self._move(fake, "person.marcus", "unknown", "not_home", 415, "Marcus")
        assert [r.content for r in integration.fetch(None, backfill=False, now=T0 + 420).entries] == ["Marcus left home"]
