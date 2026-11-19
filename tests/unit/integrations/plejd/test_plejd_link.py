"""PlejdLink without Bluetooth: the site choice, the site load against a faked cloud and
manager, what an advertisement does to the manager, the ping, the state store, `run` without
a loop, and the thread giving up on a refused login or an unexpected failure."""
import asyncio
from unittest.mock import patch

import pytest

from integrations.plejd.PlejdModel import MANUFACTURER_ID
from integrations.plejd.PlejdLink import NotConnected, PlejdLink, SiteChoice, WriteFailed, choose_site
from pyplejd import AuthenticationError
from pyplejd import ConnectionError as PlejdCloudUnreachable
from tests.unit.waiting import FAILURE_BOUND

MODULE = "integrations.plejd.PlejdLink"
SITES = [{"siteId": "abc123", "title": "Villa Nyström", "deviceCount": 12},
         {"siteId": "def456", "title": "Sommarstugan", "deviceCount": 3}]


@pytest.mark.unit
class TestChooseSite:
    def test_the_only_site_needs_no_choice(self):
        assert choose_site(SITES[:1], "")["siteId"] == "abc123"

    def test_a_wanted_site_is_found_by_title_or_id_regardless_of_case(self):
        assert choose_site(SITES, "sommarstugan")["siteId"] == "def456"
        assert choose_site(SITES, "ABC123")["title"] == "Villa Nyström"

    def test_several_sites_without_a_choice_name_them(self):
        with pytest.raises(SiteChoice) as e:
            choose_site(SITES, "")
        assert "Villa Nyström" in str(e.value) and "Sommarstugan" in str(e.value) and "plejd.site" in str(e.value)

    def test_a_wanted_site_the_account_lacks_names_the_ones_it_has(self):
        with pytest.raises(SiteChoice) as e:
            choose_site(SITES, "Kontoret")
        assert "Kontoret" in str(e.value) and "Sommarstugan" in str(e.value)
        with pytest.raises(SiteChoice):
            choose_site([], "")


class Hardware:
    def __init__(self):
        self.seen = []

    def see(self, rssi, device):
        self.seen.append((rssi, device))
        return len(self.seen) == 1


class Manager:
    def __init__(self, hardware):
        self.hardware = hardware
        self.devices = []
        self.connected = False


class Advertisement:
    def __init__(self, mac, rssi=-60, company=MANUFACTURER_ID):
        self.manufacturer_data = {company: b"\x01\x02\x03\x04" + bytes.fromhex(mac)[::-1] + b"\x00\x00"}
        self.rssi = rssi


class Device:
    def __init__(self, name):
        self.name = name
        self.identifier = ("dev", name)


@pytest.mark.unit
class TestAdvertisements:
    @pytest.fixture
    def link(self):
        link = PlejdLink("m@mlab.test", "secret")
        link.manager = Manager({"AABBCCDDEEFF": Hardware()})
        return link

    def test_a_unit_of_the_site_is_handed_to_pyplejd_and_the_first_sighting_asks_for_a_connect(self, link):
        link._on_advertisement("ble-device", Advertisement("aabbccddeeff", rssi=-55))
        assert link.manager.hardware["AABBCCDDEEFF"].seen == [(-55, "ble-device")]
        assert link.units_seen == 1 and link._ping_wanted
        link._ping_wanted = False
        link._on_advertisement("ble-device", Advertisement("aabbccddeeff", rssi=-70))
        assert link.units_seen == 1 and not link._ping_wanted

    def test_a_stranger_and_another_manufacturer_are_ignored(self, link):
        link._on_advertisement("x", Advertisement("001122334455"))
        link._on_advertisement("x", Advertisement("aabbccddeeff", company=76))
        assert link.units_seen == 0 and not link._ping_wanted
        assert link.manager.hardware["AABBCCDDEEFF"].seen == []

    def test_before_the_site_is_loaded_nothing_happens(self):
        link = PlejdLink("m@mlab.test", "secret")
        link._on_advertisement("x", Advertisement("aabbccddeeff"))
        assert link.units_seen == 0


class SiteDevice:
    """A device as pyplejd hands it out: it keeps the one callback the link subscribes."""

    def __init__(self, name, output_type="LIGHT"):
        self.name, self.identifier, self.outputType = name, ("dev", name), output_type
        self.callback = None

    def subscribe(self, callback):
        self.callback = callback


class SiteManager:
    """What PlejdManager looks like to _load_site: built from the credentials, init'd, then read."""

    def __init__(self, username, password, site_id):
        self.site_id = site_id
        self.devices = [SiteDevice("Taklampa"), SiteDevice("Fläkt"), SiteDevice("Kväll", "SCENE")]
        self.hardware = {"AABBCCDDEEFF": Hardware()}
        self.connected = False
        self.connection_monitor = None
        self.mesh = Mesh()

    async def init(self):
        pass


@pytest.mark.unit
class TestLoadSite:
    @pytest.fixture
    def link(self):
        link = PlejdLink("m@mlab.test", "secret")
        link.slept = []

        async def no_sleep(seconds):
            link.slept.append(seconds)

        link._sleep = no_sleep
        return link

    def test_an_unreachable_cloud_is_waited_out_and_said_so(self, link):
        attempts = []

        async def get_sites(username, password):
            attempts.append(1)
            if len(attempts) == 1:
                raise PlejdCloudUnreachable("cloud down")
            return SITES[:1]

        with patch(f"{MODULE}.get_sites", get_sites), patch(f"{MODULE}.PlejdManager", SiteManager):
            manager = asyncio.run(link._load_site())
        assert len(attempts) == 2 and len(link.slept) == 1
        assert "unreachable" in link.waiting
        assert isinstance(manager, SiteManager) and link.manager is manager
        assert link.site_title == "Villa Nyström" and link.error is None

    def test_each_device_reports_into_its_own_state(self, link):
        async def get_sites(username, password):
            return SITES[:1]

        with patch(f"{MODULE}.get_sites", get_sites), patch(f"{MODULE}.PlejdManager", SiteManager):
            manager = asyncio.run(link._load_site())
        taklampa, flakt = manager.devices[0], manager.devices[1]
        taklampa.callback({"state": True, "dim": 255})
        flakt.callback({"state": False})
        states = link.states()
        assert states[("dev", "Taklampa")] == {"state": True, "dim": 255}
        assert states[("dev", "Fläkt")] == {"state": False}
        assert manager.connection_monitor == link._on_connection

    def test_two_sites_without_a_choice_give_up_with_the_key_to_set(self, link):
        async def get_sites(username, password):
            return SITES

        with patch(f"{MODULE}.get_sites", get_sites), patch(f"{MODULE}.PlejdManager", SiteManager):
            assert asyncio.run(link._load_site()) is None
        assert "plejd.site" in link.error and link.manager is None and link.slept == []


class Backend:
    def __init__(self):
        self.disconnected_callback = None

    def set_disconnected_callback(self, callback):
        self.disconnected_callback = callback


class BleakClient:
    """The bleak client pyplejd keeps: it can lose its connection while pyplejd still holds it."""

    def __init__(self):
        self.is_connected = True
        self._backend = Backend()

    def lose(self):
        self.is_connected = False
        if self._backend.disconnected_callback:
            self._backend.disconnected_callback()


class Mesh:
    def __init__(self):
        self._client = None
        self._gateway_node = None
        self.writes = []
        self.write_ok = True

    async def _write(self, payloads):
        self.writes.append(payloads)
        return self.write_ok


class MeshManager:
    """pyplejd's manager as the link sees it: its connected flag is only whether it holds a
    client — the 0.22.0 behaviour that left a lost client in place —, ping connects when it
    holds none and answers only over a live one, disconnect drops the client and calls back."""

    def __init__(self, failure=None, answers=True):
        self.mesh = Mesh()
        self.hardware = {}
        self.pings = self.connects = self.disconnects = 0
        self.failure = failure
        self.answers = answers
        self.connection_monitor = None

    @property
    def connected(self):
        return self.mesh._client is not None

    def connect_callback(self, connected):
        if self.connection_monitor:
            self.connection_monitor(connected)

    async def ping(self, retry=True):
        self.pings += 1
        if self.failure is not None:
            raise self.failure
        if self.mesh._client is None:
            self.connects += 1
            self.mesh._client = BleakClient()
            self.connect_callback(True)
        return self.answers and self.mesh._client.is_connected

    async def disconnect(self):
        self.disconnects += 1
        self.mesh._client = None
        self.connect_callback(False)


@pytest.mark.unit
class TestPing:
    @pytest.fixture
    def link(self):
        link = PlejdLink("m@mlab.test", "secret")
        link.manager = MeshManager()
        link.manager.connection_monitor = link._on_connection
        link.units_seen = 1
        return link

    def test_nothing_heard_on_the_air_means_no_ping(self, link):
        link.units_seen = 0
        asyncio.run(link._ping())
        assert link.manager.pings == 0 and link.connected_at is None

    def test_a_failing_ping_is_swallowed_and_leaves_the_link_down(self, link):
        link.manager = MeshManager(failure=OSError("Bluetooth is off"))
        asyncio.run(link._ping())
        assert link.manager.pings == 1 and not link.connected and link.connected_at is None

    def test_connected_at_marks_the_ping_that_connected_and_no_later_one(self, link):
        asyncio.run(link._ping())
        assert link.connected and link.connected_at is not None
        link.connected_at = 1234.5
        asyncio.run(link._ping())
        assert link.manager.pings == 2 and link.connected_at == 1234.5

    def test_a_waiter_is_released_by_the_ping_that_connects_and_not_by_a_drop(self, link):
        assert not link.wait_connected(0)
        asyncio.run(link._ping())
        assert link.wait_connected(0)
        link.manager.mesh._client = None
        link._on_connection(False)
        assert not link.wait_connected(0) and link._ping_wanted

    def test_a_failing_ping_releases_no_waiter(self, link):
        link.manager = MeshManager(failure=OSError("Bluetooth is off"))
        asyncio.run(link._ping())
        assert not link.wait_connected(0)


@pytest.mark.unit
class TestDeadLink:
    """pyplejd 0.22.0 keeps a bleak client that lost its connection; the link must notice and heal."""

    @pytest.fixture
    def link(self):
        link = PlejdLink("m@mlab.test", "secret")
        link.manager = MeshManager()
        link.manager.connection_monitor = link._on_connection
        link.units_seen = 1
        asyncio.run(link._ping())
        link._ping_wanted = False
        return link

    def lose_silently(self, link):
        """A drop pyplejd's own callback would miss: the client is down, the manager still holds it."""
        link.manager.mesh._client.is_connected = False

    def test_a_held_client_that_is_down_is_not_connected(self, link):
        assert link.connected
        self.lose_silently(link)
        assert link.manager.connected and not link.connected and not link.wait_connected(0)

    def test_the_next_ping_drops_the_stale_client_and_connects_afresh(self, link):
        self.lose_silently(link)
        asyncio.run(link._ping())
        assert link.manager.disconnects == 1 and link.manager.connects == 2
        assert link.connected and link.wait_connected(0)

    def test_a_ping_that_fails_on_a_held_client_disconnects_and_asks_for_a_reconnect(self, link):
        link.manager.answers = False
        asyncio.run(link._ping())
        assert link.manager.disconnects == 1 and not link.manager.connected
        assert not link.connected and link._ping_wanted
        link.manager.answers, link._ping_wanted = True, False
        asyncio.run(link._ping())
        assert link.manager.connects == 2 and link.connected

    def test_a_unit_that_connects_but_does_not_answer_waits_for_the_regular_ping(self, link):
        link.manager.mesh._client = None
        link._mesh.clear()
        link.manager.answers = False
        asyncio.run(link._ping())
        assert link.manager.disconnects == 1 and not link._ping_wanted

    def test_the_drop_is_logged_once(self, link):
        link.manager.answers = False
        with patch(f"{MODULE}.info") as info:
            asyncio.run(link._ping())
            asyncio.run(link._ping())
        assert sum("dropped" in c.args[0] for c in info.call_args_list) == 1

    def test_bluetooth_reporting_the_loss_marks_the_link_down_and_asks_for_a_reconnect_at_once(self, link):
        link.manager.mesh._client.lose()
        assert not link.connected and not link._mesh.is_set() and link._ping_wanted

    def test_the_loss_of_a_client_already_replaced_is_ignored(self, link):
        old = link.manager.mesh._client
        asyncio.run(link.manager.disconnect())
        asyncio.run(link._ping())
        link._ping_wanted = False
        old.lose()
        assert link.connected and not link._ping_wanted

    def test_a_write_pyplejd_could_not_deliver_fails_the_send_and_asks_for_a_reconnect(self, link):
        link._watch_writes(link.manager.mesh)
        mesh = link.manager.mesh

        async def action():
            await mesh._write(["payload"])

        asyncio.run(link.send(action))
        self.lose_silently(link)
        mesh.write_ok = False
        link._ping_wanted = False
        with pytest.raises(WriteFailed):
            asyncio.run(link.send(action))
        assert link._ping_wanted and len(mesh.writes) == 2


@pytest.mark.unit
class TestScanWindow:
    def test_before_the_scanner_starts_there_is_no_window_to_wait_out(self):
        assert PlejdLink("m@mlab.test", "secret").scan_time_left(20) == 0

    def test_the_window_counts_down_from_the_scanner_start_and_ends_at_zero(self):
        link = PlejdLink("m@mlab.test", "secret")
        with patch(f"{MODULE}.time.monotonic", return_value=1000.0):
            link.scanning_since = 1000.0
            assert link.scan_time_left(20) == 20
        with patch(f"{MODULE}.time.monotonic", return_value=1012.5):
            assert link.scan_time_left(20) == 7.5
        with patch(f"{MODULE}.time.monotonic", return_value=1100.0):
            assert link.scan_time_left(20) == 0


@pytest.mark.unit
class TestConnectionMonitor:
    def test_a_drop_asks_for_a_ping_unless_the_link_is_stopping(self):
        link = PlejdLink("m@mlab.test", "secret")
        link._on_connection(True)
        assert not link._ping_wanted
        link._on_connection(False)
        assert link._ping_wanted
        link._ping_wanted, link._stopping = False, True
        link._on_connection(False)
        assert not link._ping_wanted


@pytest.mark.unit
class TestStates:
    def test_states_are_kept_per_device_as_copies(self):
        link = PlejdLink("m@mlab.test", "secret")
        state = {"state": True, "dim": 128}
        link._on_state(Device("Taklampa"), state)
        state["state"] = False
        assert link.states() == {("dev", "Taklampa"): {"state": True, "dim": 128}}
        link.states()[("dev", "Taklampa")]["dim"] = 0
        assert link.states()[("dev", "Taklampa")]["dim"] == 128

    def test_run_without_a_loop_is_not_connected_and_closes_the_coroutine(self):
        async def work():
            return 1

        coro = work()
        with pytest.raises(NotConnected):
            PlejdLink("m@mlab.test", "secret").run(coro, 1)
        with pytest.raises(RuntimeError):     # closed: cannot be awaited any more
            coro.send(None)


@pytest.mark.unit
class TestThread:
    def test_a_refused_login_ends_the_thread_with_the_reason(self):
        async def refused(username, password):
            raise AuthenticationError("Invalid username/password")

        link = PlejdLink("m@mlab.test", "wrong")
        with patch(f"{MODULE}.get_sites", refused):
            link.start()
            link._thread.join(FAILURE_BOUND)
            assert not link._thread.is_alive()
        assert not link.wait_ready(0)
        assert "login refused" in link.error and "plejd.password" in link.error
        link.stop()

    def test_an_unexpected_failure_ends_the_thread_naming_it(self):
        async def broken(username, password):
            raise ValueError("bad site data")

        link = PlejdLink("m@mlab.test", "secret")
        with patch(f"{MODULE}.get_sites", broken):
            link.start()
            link._thread.join(FAILURE_BOUND)
            assert not link._thread.is_alive()
        assert not link.wait_ready(0)
        assert "Plejd stopped" in link.error and "ValueError" in link.error and "bad site data" in link.error
        link.stop()
