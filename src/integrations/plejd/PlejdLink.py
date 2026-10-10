"""The asyncio side of the Plejd integration on a thread of its own. It loads the site from the
Plejd cloud through pyplejd (retrying while the cloud is unreachable, giving up on a refused
login or a site it cannot pick), runs the Bluetooth scanner that tells pyplejd which mesh units
are in range, keeps the link alive with pyplejd's ping — which connects to the strongest unit
seen and reconnects after a drop —, syncs the mesh clock hourly, and keeps every device's last
reported state. The sync integration hands it coroutines to run and reads the states."""
import asyncio
import threading
import time
from typing import Optional

from bleak import BleakScanner
from pyplejd import PLEJD_SERVICE, AuthenticationError, PlejdManager, get_sites
from pyplejd import ConnectionError as PlejdCloudUnreachable

from common.logging import info, trace
from .PlejdModel import COVER_TYPE, SCENE_TYPE, kind_of, mac_from_advertisement, state_text

PING_INTERVAL = 180         # pyplejd's own keep-alive period: the mesh drops an idle link
TIME_SYNC_INTERVAL = 3600   # the mesh clock drifts; Home Assistant's component syncs it hourly too
CLOUD_RETRY = 60            # seconds before the site is fetched again when the cloud was unreachable
TICK = 5                    # how long the loop sleeps between looks at its timers
STOP_TIMEOUT = 10


class SiteChoice(Exception):
    """The account's sites do not give one to use; the message says what to set."""


class NotConnected(Exception):
    """The link's loop is not running, so nothing can be sent."""


def choose_site(sites: list[dict], wanted: str) -> dict:
    """The site (`siteId`, `title`, `deviceCount`) *wanted* names by id or title, case-
    insensitively; with nothing wanted, the only site the account has."""
    wanted = (wanted or "").strip()
    names = ", ".join(f'"{s["title"]}"' for s in sites)
    if wanted:
        for site in sites:
            if wanted.lower() in (site["siteId"].lower(), site["title"].lower()):
                return site
        raise SiteChoice(f'The Plejd account has no site "{wanted}"; it has {names or "none"} — set plejd.site to one of them.')
    if len(sites) == 1:
        return sites[0]
    if not sites:
        raise SiteChoice("The Plejd account has no site.")
    raise SiteChoice(f"The Plejd account has {len(sites)} sites — set plejd.site to one of {names}.")


class PlejdLink:
    """One site's cloud data and mesh link, driven by an event loop on a daemon thread."""

    def __init__(self, username: str, password: str, site: str = ""):
        self._credentials = (username, password, site)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._wake: Optional[asyncio.Event] = None
        self._stopping = False
        self._ping_wanted = False
        self._ready = threading.Event()         # the site is loaded and the scanner runs
        self._mesh = threading.Event()          # the mesh link is up (cleared on a drop)
        self._lock = threading.Lock()
        self._states: dict[tuple, dict] = {}    # device identifier → its last reported state
        self.manager: Optional[PlejdManager] = None
        self.site_title: Optional[str] = None
        self.error: Optional[str] = None        # why there will be no site: the health check's verdict
        self.waiting = "the Plejd site is still loading"     # why there is none yet
        self.units_seen = 0                     # mesh units of the site heard on the air since start
        self.connected_at: Optional[float] = None
        self.scanning_since: Optional[float] = None     # monotonic time the scanner started

    # --- lifecycle, from any thread -----------------------------------------------------------------

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._thread_main, name="plejd", daemon=True)
        self._thread.start()

    def stop(self):
        self._stopping = True
        self._nudge()
        if self._thread is not None:
            self._thread.join(STOP_TIMEOUT)
            self._thread = None

    def wait_ready(self, timeout: float) -> bool:
        return self._ready.wait(timeout)

    def wait_connected(self, timeout: float) -> bool:
        """Blocks until the mesh link is up or *timeout* passes; whether it is up."""
        return self._mesh.wait(timeout) or self.connected

    def scan_time_left(self, window: float) -> float:
        """Seconds of the first *window* seconds of scanning still ahead — 0 once they have
        passed or while the scanner has not started. A fresh scanner hears the units of a
        site one by one over several seconds and the link follows the first one heard, so
        a verdict on the mesh inside that window is early."""
        if self.scanning_since is None:
            return 0.0
        return max(0.0, window - (time.monotonic() - self.scanning_since))

    @property
    def connected(self) -> bool:
        return self.manager is not None and bool(self.manager.connected)

    @property
    def devices(self) -> list:
        return list(self.manager.devices) if self.manager is not None else []

    def states(self) -> dict[tuple, dict]:
        with self._lock:
            return {key: dict(state) for key, state in self._states.items()}

    def run(self, coro, timeout: float):
        """Runs *coro* on the link's loop and returns its result; TimeoutError past *timeout*,
        NotConnected when the loop is not running."""
        if self._loop is None or not self._loop.is_running():
            coro.close()
            raise NotConnected("the Plejd link is not running")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout)
        except TimeoutError:
            future.cancel()
            raise

    def request_ping(self):
        """Asks the loop to ping (connect) at its next turn, from any thread."""
        self._ping_wanted = True
        self._nudge()

    def _nudge(self):
        if self._loop is not None and self._wake is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._wake.set)

    # --- the loop thread ----------------------------------------------------------------------------

    def _thread_main(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except Exception as e:
            self.error = f"Plejd stopped: {type(e).__name__}: {e}"
            info(self.error)
        finally:
            self._ready.clear()
            self._mesh.clear()
            self._loop.close()

    async def _main(self):
        self._wake = asyncio.Event()
        manager = await self._load_site()
        if manager is None:
            return
        scanner = BleakScanner(self._on_advertisement, service_uuids=[PLEJD_SERVICE])
        await scanner.start()
        info(f"Plejd: scanning for the units of {self.site_title}")
        self.scanning_since = time.monotonic()
        self._ready.set()
        last_ping = last_sync = 0.0
        try:
            while not self._stopping:
                now = time.monotonic()
                if self._ping_wanted or now - last_ping >= PING_INTERVAL:
                    self._ping_wanted, last_ping = False, now
                    await self._ping()
                if self.connected and now - last_sync >= TIME_SYNC_INTERVAL:
                    last_sync = now
                    await self._sync_time()
                await self._sleep(TICK)
        finally:
            await scanner.stop()
            if manager.connected:
                await manager.disconnect()
            info("Plejd: link stopped")

    async def _sleep(self, seconds: float):
        """Sleeps unless stop() or request_ping() wakes the loop first."""
        try:
            await asyncio.wait_for(self._wake.wait(), timeout=seconds)
        except TimeoutError:
            pass
        self._wake.clear()

    async def _load_site(self) -> Optional[PlejdManager]:
        username, password, wanted = self._credentials
        while not self._stopping:
            try:
                site = choose_site(await get_sites(username, password), wanted)
                manager = PlejdManager(username, password, site["siteId"])
                await manager.init()
            except AuthenticationError as e:
                self.error = f"Plejd cloud login refused: {e} — check plejd.username and plejd.password."
                info(self.error)
                return None
            except SiteChoice as e:
                self.error = str(e)
                info(self.error)
                return None
            except PlejdCloudUnreachable:
                self.waiting = f"the Plejd cloud is unreachable — the site is fetched again every {CLOUD_RETRY} s"
                info(f"Plejd: the cloud is unreachable — trying again in {CLOUD_RETRY} s")
                await self._sleep(CLOUD_RETRY)
                continue
            self.site_title = site["title"]
            manager.connection_monitor = self._on_connection
            for device in manager.devices:
                device.subscribe(lambda state, d=device: self._on_state(d, state))
            self.manager = manager
            kinds = [kind_of(d) for d in manager.devices]
            scenes, covers = kinds.count(SCENE_TYPE), kinds.count(COVER_TYPE)
            info(f"Plejd: site {site['title']} loaded — {len(kinds) - scenes - covers} devices, {covers} covers, "
                 f"{scenes} scenes, {len(manager.hardware)} mesh units")
            return manager
        return None

    def _on_advertisement(self, device, advertisement):
        """A unit of the site on the air: pyplejd learns it (with the BLEDevice to connect to),
        and the first one triggers the connect."""
        mac = mac_from_advertisement(advertisement.manufacturer_data)
        hardware = self.manager.hardware.get(mac) if mac and self.manager is not None else None
        if hardware is None:
            return
        if hardware.see(advertisement.rssi, device):
            self.units_seen += 1
            info(f"Plejd: unit {mac} in range (rssi {advertisement.rssi})")
            self.request_ping()

    async def _ping(self):
        if self.units_seen == 0:
            trace("Plejd: no unit of the site in Bluetooth range yet")
            return
        was_connected = self.connected
        try:
            ok = await self.manager.ping(retry=False)
        except Exception as e:
            info(f"Plejd: ping failed: {type(e).__name__}: {e}")
            ok = False
        if ok and not was_connected:
            self.connected_at = time.time()
            info("Plejd: connected to the mesh")
        elif not ok:
            info("Plejd: the mesh did not answer — reconnecting at the next ping")
        if ok:
            self._mesh.set()
        else:
            self._mesh.clear()

    async def _sync_time(self):
        try:
            await self.manager.broadcast_time()
        except Exception as e:
            info(f"Plejd: clock sync failed: {type(e).__name__}: {e}")

    def _on_connection(self, connected: bool):
        if connected:
            return
        info("Plejd: the mesh link dropped")
        self._mesh.clear()
        if not self._stopping:
            self.request_ping()

    def _on_state(self, device, state: dict):
        with self._lock:
            before = state_text(device, self._states.get(device.identifier, {}))
            self._states[device.identifier] = dict(state)
        after = state_text(device, state)
        if after != before:
            trace(f"Plejd: {device.name} is {after}")
