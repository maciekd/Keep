"""Plejd as a HomeAutomation backend, over Bluetooth from the server's own Mac. The Plejd
cloud (the app's login) gives the site — lights, rooms, scenes and the mesh key — and
PlejdLink keeps a link to the strongest powered unit in range on its own thread: the states
come from the mesh's notifications and the commands are writes to it. Nothing here blocks the
caller for long: a snapshot is a read from memory, a command waits COMMAND_TIMEOUT for the
loop, and the health check says why the site or the mesh is not there."""
import threading
import time
from typing import Callable, List, Optional

from assistant.integration.HomeAutomation import HomeAutomation, HomeSnapshot
from assistant.integration.Integration import Integration
from common.ConfigManager import config
from common.logging import info
from .PlejdLink import NotConnected, PlejdLink
from .PlejdModel import SCENE_TYPE, Unresolved, build, kind_of, to_plejd_level

CONNECT_WAIT = 10       # seconds a snapshot, a command or the health check waits for the site to load
MESH_WAIT = 20          # seconds of scanning the health check gives the mesh link before judging it
COMMAND_TIMEOUT = 10    # seconds a command waits for the mesh write
NO_MESH = "No Plejd unit in Bluetooth range — the mesh is not connected."


class PlejdIntegration(Integration, HomeAutomation):
    """Plejd lights over Bluetooth.

    Makes the Plejd mesh a home automation backend: Keep sees every light, relay and scene of
    the site and switches, dims and runs them through any powered Plejd unit in Bluetooth range
    of the server, no gateway needed. Beside Indigo or Home Assistant it is one more part of the
    same house.

    pyplejd (the library behind the Home Assistant component) does the cloud call and the mesh
    protocol; PlejdLink runs its asyncio side — scanner, link, keep-alive — on a thread, and
    PlejdModel maps the devices and states onto the snapshot and the name index. pyplejd drops
    a write it cannot deliver (no link, a Bluetooth error) without raising, so a command only
    reports what it could check: the link being up before the write."""

    def __init__(self):
        super().__init__()
        self._link: Optional[PlejdLink] = None
        self._lock = threading.Lock()

    def _on_reset(self):
        if self._link is not None:
            self._link.stop()
            self._link = None

    # --- Integration ------------------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "Plejd"

    @property
    def config_keys(self) -> List[str]:
        return ["plejd.username", "plejd.password"]

    @property
    def config_section(self) -> str:
        return "plejd"

    def check_health(self) -> bool:
        """Healthy with the mesh link up. Right after a start the scanner is still hearing
        the units and the link follows the first one, so inside the first MESH_WAIT seconds of
        scanning the check waits for the link instead of calling its absence a failure."""
        link = self._ready_link()
        if not link.connected:
            left = link.scan_time_left(MESH_WAIT)
            if left > 0:
                info(f"Plejd: health check waits up to {left:.0f} s for the first scan to bring the mesh link up")
                if link.wait_connected(left):
                    info(f"Plejd: the mesh link came up within the first scan — {link.units_seen} units heard")
        if not link.connected:
            seen = f"{link.units_seen} of its units heard since start" if link.units_seen else "none of its units heard since start"
            raise RuntimeError(f"site {link.site_title} loaded, but no mesh link: {seen}")
        return True

    # --- the link -----------------------------------------------------------------------------------

    def link(self) -> PlejdLink:
        """The link, built from config and started on first use."""
        with self._lock:
            if self._link is None:
                self._link = PlejdLink(str(config.get("plejd.username", "") or ""),
                                       str(config.get("plejd.password", "") or ""),
                                       str(config.get("plejd.site", "") or ""))
                self._link.start()
            return self._link

    def _ready_link(self) -> PlejdLink:
        """The link once its site is loaded; RuntimeError with the reason when it is not."""
        link = self.link()
        if link.error:
            raise RuntimeError(link.error)
        if not link.wait_ready(CONNECT_WAIT):
            raise RuntimeError(link.error or link.waiting)
        return link

    # --- HomeAutomation: state ------------------------------------------------------------------

    def snapshot(self) -> HomeSnapshot:
        link = self._ready_link()
        snapshot, _ = build(link.devices, link.states())
        snapshot.taken_at = time.time()
        if not link.connected:
            info("Plejd: snapshot without a mesh link — the states are the last reported ones")
        return snapshot

    # --- HomeAutomation: commands ---------------------------------------------------------------

    def turn_on(self, name: str) -> str:
        return self._command(name, "turn_on")

    def turn_off(self, name: str) -> str:
        return self._command(name, "turn_off")

    def dim(self, name: str, level: int) -> str:
        level = int(level)
        if level <= 0:
            return self.turn_off(name)
        return self._command(name, "dim", min(level, 100))

    def execute(self, name: str) -> str:
        return self._command(name, "execute")

    def _command(self, name: str, verb: str, level: Optional[int] = None) -> str:
        try:
            link = self._ready_link()
        except RuntimeError as e:
            info(f"Plejd: cannot {verb} '{name}': {e}")
            return str(e)
        try:
            _, index = build(link.devices, {})
            device, display = index.resolve(name)
        except Unresolved as e:
            info(f"Plejd: {e}")
            return str(e)
        action = self._action(device, display, verb, level)
        if isinstance(action, str):
            return action
        if not link.connected:
            info(f"Plejd: cannot {verb} {display} — no mesh link")
            return NO_MESH
        try:
            link.run(action(), COMMAND_TIMEOUT)
        except TimeoutError:
            info(f"Plejd: {verb} {display} timed out")
            return "Request timed out"
        except NotConnected:
            return NO_MESH
        except Exception as e:
            info(f"Plejd: {verb} {display} failed: {type(e).__name__}: {e}")
            return str(e)
        info(f"Plejd: {verb} → {display}{f' {level}%' if level is not None else ''}")
        return ""

    @staticmethod
    def _action(device, display: str, verb: str, level: Optional[int]) -> Callable | str:
        """What to run on the loop for *verb*, or the text for the LLM when the device cannot."""
        if kind_of(device) == SCENE_TYPE:
            return device.activate if verb == "execute" else f"{display} is a scene; use execute."
        if verb == "execute":
            return f"{display} is a light; use turn_on / turn_off."
        if verb == "dim":
            if not getattr(device, "dimmable", False):
                return f"{display} is not dimmable."
            return lambda: device.turn_on(dim=to_plejd_level(level))
        return device.turn_on if verb == "turn_on" else device.turn_off
