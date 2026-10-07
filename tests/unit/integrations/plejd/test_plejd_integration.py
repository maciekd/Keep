"""PlejdIntegration with the link faked: the contract, the snapshot, the commands' routing and
answers, and the health check."""
import asyncio
from unittest.mock import patch

import pytest

from assistant.integration.HomeAutomation import render
from integrations.plejd.PlejdIntegration import NO_MESH, PlejdIntegration
from integrations.plejd.PlejdLink import NotConnected

MODULE = "integrations.plejd.PlejdIntegration"
CONFIG = {"plejd.username": "m@mlab.test", "plejd.password": "secret", "plejd.site": "Villa Nyström"}


class Room:
    def __init__(self, title):
        self.title = title


class Light:
    outputType = "LIGHT"
    hidden = False

    def __init__(self, name, room, dimmable=True):
        self.name, self.roomData, self.dimmable = name, Room(room), dimmable
        self.identifier = ("dev", name)
        self.calls = []

    async def turn_on(self, dim=None):
        self.calls.append(("turn_on", dim))

    async def turn_off(self):
        self.calls.append(("turn_off",))


class Scene:
    outputType = "SCENE"
    hidden = False

    def __init__(self, name):
        self.name = name
        self.identifier = (name,)
        self.calls = []

    async def activate(self):
        self.calls.append(("activate",))


class FakeLink:
    def __init__(self, *args):
        self.args = args
        self.error = None
        self.ready = True
        self.connected = True
        self.site_title = "Villa Nyström"
        self.waiting = "the Plejd site is still loading"
        self.units_seen = 2
        self.devices = [Light("Taklampa", "Kök"), Light("Fläkt", "Badrum", dimmable=False), Scene("Kväll")]
        self._states = {("dev", "Taklampa"): {"state": True, "dim": 255}}
        self.started = self.stopped = 0
        self.failure = None

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    def wait_ready(self, timeout):
        return self.ready

    def states(self):
        return dict(self._states)

    def run(self, coro, timeout):
        if self.failure is not None:
            coro.close()
            raise self.failure
        return asyncio.run(coro)


@pytest.fixture
def integration():
    PlejdIntegration.reset_instance()
    with patch(f"{MODULE}.config") as config:
        config.get.side_effect = lambda key, default=None: CONFIG.get(key, default)
        yield PlejdIntegration()
    PlejdIntegration.reset_instance()


@pytest.fixture
def link(integration):
    with patch(f"{MODULE}.PlejdLink", FakeLink):
        yield integration.link()


def device(link, name):
    return next(d for d in link.devices if d.name == name)


@pytest.mark.unit
class TestLink:
    def test_the_link_is_built_from_config_and_started_once(self, integration, link):
        assert link.args == ("m@mlab.test", "secret", "Villa Nyström") and link.started == 1
        assert integration.link() is link and link.started == 1

    def test_reset_stops_the_link(self, integration, link):
        integration._on_reset()
        assert link.stopped == 1


@pytest.mark.unit
class TestSnapshot:
    def test_the_snapshot_is_the_devices_with_their_last_states(self, integration, link):
        snapshot = integration.snapshot()
        assert render(snapshot) == ('## LIGHTS\n"Fläkt [Badrum]": unknown\n"Taklampa [Kök]": on 100%\n\n'
                                    '## ACTIONS\n"Kväll": scene\n\n')
        assert snapshot.taken_at > 0

    def test_a_link_without_a_site_raises_its_reason(self, integration, link):
        link.error = "Plejd cloud login refused"
        with pytest.raises(RuntimeError, match="login refused"):
            integration.snapshot()
        link.error, link.ready = None, False
        with pytest.raises(RuntimeError, match="still loading"):
            integration.snapshot()
        link.waiting = "the Plejd cloud is unreachable"
        with pytest.raises(RuntimeError, match="cloud is unreachable"):
            integration.snapshot()


@pytest.mark.unit
class TestCommands:
    def test_turn_on_and_off_reach_the_light_by_any_of_its_names(self, integration, link):
        assert integration.turn_on("Taklampa [Kök]") == ""
        assert integration.turn_off("taklampa") == ""
        assert device(link, "Taklampa").calls == [("turn_on", None), ("turn_off",)]

    def test_dim_is_turn_on_at_plejds_scale_and_zero_is_off(self, integration, link):
        assert integration.dim("Taklampa", 50) == ""
        assert integration.dim("Taklampa", 0) == ""
        assert integration.dim("Taklampa", 140) == ""
        assert device(link, "Taklampa").calls == [("turn_on", 128), ("turn_off",), ("turn_on", 255)]

    def test_a_relay_is_not_dimmable(self, integration, link):
        assert "not dimmable" in integration.dim("Fläkt", 40)
        assert device(link, "Fläkt").calls == []

    def test_scenes_run_with_execute_only_and_lights_never_do(self, integration, link):
        assert integration.execute("Kväll") == ""
        assert device(link, "Kväll").calls == [("activate",)]
        assert "use execute" in integration.turn_on("Kväll")
        assert "use turn_on" in integration.execute("Taklampa")

    def test_an_unknown_name_is_words_not_a_call(self, integration, link):
        answer = integration.turn_on("Poolbelysning")
        assert "Poolbelysning" in answer and all(d.calls == [] for d in link.devices)

    def test_without_a_mesh_link_nothing_is_sent(self, integration, link):
        link.connected = False
        assert integration.turn_on("Taklampa") == NO_MESH
        assert device(link, "Taklampa").calls == []

    def test_a_site_that_is_not_there_answers_with_the_reason(self, integration, link):
        link.error = "Plejd cloud login refused"
        assert "login refused" in integration.turn_on("Taklampa")

    def test_timeouts_and_errors_are_words_not_exceptions(self, integration, link):
        link.failure = TimeoutError()
        assert integration.turn_on("Taklampa") == "Request timed out"
        link.failure = RuntimeError("Bluetooth is off")
        assert integration.turn_on("Taklampa") == "Bluetooth is off"

    def test_a_loop_that_stopped_before_the_write_is_no_mesh(self, integration, link):
        link.failure = NotConnected("the Plejd link is not running")
        assert integration.turn_on("Taklampa") == NO_MESH
        assert device(link, "Taklampa").calls == []


@pytest.mark.unit
class TestHealth:
    def test_connected_is_healthy(self, integration, link):
        assert integration.check_health() is True

    def test_no_mesh_link_or_no_site_is_a_failure_that_says_why(self, integration, link):
        link.connected = False
        with pytest.raises(RuntimeError, match="2 of its units heard"):
            integration.check_health()
        link.units_seen = 0
        with pytest.raises(RuntimeError, match="none of its units"):
            integration.check_health()
        link.error = "The Plejd account has 2 sites"
        with pytest.raises(RuntimeError, match="2 sites"):
            integration.check_health()
