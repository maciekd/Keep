"""The iMessage satellite process: this Mac user's Messages account, lent to a Keep server over HTTP.

What `floxy_ai_server.py` runs under `role: imessage_satellite` instead of an assistant: a
`LocalMessagesHost` behind a `SatelliteService`, with a `ServerFollower` that keeps the install
on the server's code. It lives where the Apple ID is signed in — a second macOS user on the
server's Mac, or another Mac — and holds the grants there. Its config is the section
`imessage_satellite`: `port`, `token` (what the server must present) and `lan_access` (false:
only this Mac reaches it, the case of a second user; true: the network does). A switch that is
not a real true or false — `lan_access`, the root key `watchdog` — is a config problem the
process does not start on (`config_problem`), never a guess.

Nobody sits at a satellite and no assistant runs there, so what the server gets from its
first-time setup and from `config_set` the process does for itself at each start (`stay_up`):
the launchd agent follows the root key `watchdog`, and the login item is registered once."""
import asyncio
import signal
import threading
import time
from typing import Optional

from common import Autostart, Version, Watchdog
from common.ConfigManager import config
from common.ConfigMigrations import looks_like_placeholder
from common.Updater import Updater
from common.imessage import Hosts, Wire
from common.imessage.MessagesHost import MessagesHost, NO_ACCESS
from common.imessage.SatelliteService import SatelliteService, hypercorn_config
from common.imessage.ServerFollower import ServerFollower
from common.Paths import CACHE_DIR
from common.logging import error, info, note, set_log_masker, trace

SECTION = "imessage_satellite"
PORT_KEY = f"{SECTION}.port"
TOKEN_KEY = f"{SECTION}.token"
LAN_ACCESS_KEY = f"{SECTION}.lan_access"
DEFAULT_PORT = 8898
MIN_TOKEN_LENGTH = 16       # install.sh writes 32 random characters; a short one typed by hand can be guessed
LOOPBACK = "127.0.0.1"
ALL_INTERFACES = "0.0.0.0"
MASKED_TOKEN = "<satellite token>"
# Written when the login item was made: a satellite registers it once, so one the user removed
# in System Settings stays removed.
LOGIN_ITEM_MARKER = CACHE_DIR / "satellite_login_item"


def _token() -> str:
    return str(config.get(TOKEN_KEY, "") or "").strip()


def _port() -> Optional[int]:
    value = config.get(PORT_KEY, DEFAULT_PORT)
    return value if isinstance(value, int) and not isinstance(value, bool) and 0 < value < 65536 else None


def _switch(key: str) -> Optional[bool]:
    """A yes-or-no key as config.yaml has it: off when absent or empty, None for anything that
    is not a boolean. A quoted "false" is a text, and Python calls every text but "" true — it
    would open the satellite to the network, or load the watchdog, on a line that says no."""
    value = config.get(key, False)
    if value is None:
        return False
    return value if isinstance(value, bool) else None


def config_problem() -> str:
    """Why this install cannot run as a satellite as config.yaml stands, "" when it can —
    without ever quoting the token."""
    token = _token()
    if not token or looks_like_placeholder(token):
        return f"{TOKEN_KEY} is not set: the token a Keep server must present (at least {MIN_TOKEN_LENGTH} random characters)"
    if len(token) < MIN_TOKEN_LENGTH:
        return f"{TOKEN_KEY} is shorter than {MIN_TOKEN_LENGTH} characters: too easy to guess for what it guards"
    if not Wire.token_fits(token):
        return f"{TOKEN_KEY} holds a space or a character that is not plain ASCII: no server could present it"
    if _port() is None:
        return f"{PORT_KEY} is not a port number"
    for key in (LAN_ACCESS_KEY, Watchdog.KEY):
        if _switch(key) is None:
            return (f"{key} is {str(config.get(key))[:20]!r}, which is neither true nor false: write the word "
                    f"without quotes")
    return ""


def bind_host() -> str:
    """Every interface of the Mac when `imessage_satellite.lan_access` is on, else the loopback."""
    if _switch(LAN_ACCESS_KEY) is True:
        note(f"The satellite answers the whole network ({LAN_ACCESS_KEY} is on): the token is all that "
             f"stands between the network and this account's messages, and it travels unencrypted.")
        return ALL_INTERFACES
    info(f"The satellite answers this Mac only ({LAN_ACCESS_KEY} is off); a Keep server on another Mac "
         f"needs the key on.")
    return LOOPBACK


def stay_up(readable: bool) -> None:
    """Bring the Mac to what keeps this satellite running: the launchd agent that relaunches a
    dead process follows the `watchdog` key (as the server's start does), and `Keep Autostart.app`
    becomes a login item at the first start that can read Messages — the server's setup waits for
    Full Disk Access the same way, so macOS asks one thing at a time. Both may wait on a dialog
    at the Mac's screen, so this runs beside the serving, never before it."""
    try:
        Watchdog.sync(_switch(Watchdog.KEY) is True)
        if LOGIN_ITEM_MARKER.exists():
            trace("Satellite: the login item was registered by an earlier start — left as the user has it")
            return
        if not readable:
            info("Satellite: Messages is not readable yet — the login item is registered by the first start that reads it")
            return
        failure = Autostart.enable()
        if failure:
            info(f"Satellite: not a login item yet ({failure}) — the next start tries again")
            return
        LOGIN_ITEM_MARKER.parent.mkdir(parents=True, exist_ok=True)
        LOGIN_ITEM_MARKER.write_text(f"{time.time():.0f}\n")
        note(f"The satellite starts at login from now on ({Autostart.SETTINGS_PATH}, where it can be removed).")
    except Exception as e:
        error(f"Satellite: the watchdog and the login item could not be set up ({type(e).__name__}: {e}) — "
              f"the satellite serves, but nothing restarts it")


class Satellite:
    """The iMessage satellite process: this Mac user's Messages account served to a Keep server until a signal ends it."""

    def __init__(self, host: MessagesHost, service: SatelliteService, bind: str, port: int):
        self.host = host
        self.service = service
        self.bind = bind
        self.port = port
        self._shutdown: Optional[asyncio.Event] = None

    def check_host(self) -> bool:
        """Reads the Messages database and the contacts once before serving, so macOS asks for
        its grants while someone may be at the Mac and the log says what is missing. Returns
        whether Messages is readable."""
        health = self.host.health(refresh=True)
        if health.ok:
            info(f"Satellite: Messages is readable here ({health.contacts} contact entries)")
            return True
        if health.reason == NO_ACCESS:
            error(f"Satellite: this Mac keeps the process out of Messages ({health.detail}). Grant Full Disk Access "
                  f"to Terminal in System Settings > Privacy & Security, quit Terminal and start Keep again.")
        else:
            error(f"Satellite: Messages cannot be read here — {health.detail}")
        return False

    async def serve(self) -> None:
        """Run hypercorn until SIGINT/SIGTERM (or `request_shutdown`)."""
        from hypercorn.asyncio import serve

        readable = await asyncio.to_thread(self.check_host)
        threading.Thread(target=stay_up, args=(readable,), daemon=True, name="satellite-stay-up").start()
        self._shutdown = asyncio.Event()
        self._install_signal_handlers()
        info(f"Satellite listening on {self.bind}:{self.port}...")
        note("Ready to serve this Mac's Messages to the Keep server.")
        try:
            await serve(self.service.app, hypercorn_config(f"{self.bind}:{self.port}"),
                        shutdown_trigger=self._shutdown.wait)
        finally:
            self.host.close()       # the typing dots and their windows do not outlive the process
            note("Satellite stopped.")

    def _install_signal_handlers(self) -> None:
        """SIGTERM is how start.sh and stop.sh end the process; both signals end the serving."""
        loop = asyncio.get_running_loop()
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._on_signal, sig.name)
        except (NotImplementedError, RuntimeError, ValueError) as e:
            info(f"Satellite: no signal handlers here ({type(e).__name__}: {e}) — only request_shutdown() stops it")

    def _on_signal(self, name: str) -> None:
        note(f"Received {name} — stopping the satellite")
        self.request_shutdown()

    def request_shutdown(self) -> None:
        if self._shutdown is not None:
            self._shutdown.set()


def build() -> Satellite:
    """The satellite config.yaml describes; nothing serves yet. `config_problem()` is checked first."""
    token = _token()
    # Nothing here logs the token, and should a library ever do it the line is masked.
    set_log_masker(lambda text: text.replace(token, MASKED_TOKEN) if token else text)
    note(f"Keep iMessage satellite, version {Version.detail()}")
    host = Hosts.local()
    follower = ServerFollower(Version.branch() or "", Version.commit(), Updater())
    return Satellite(host, SatelliteService(host, token, follower), bind_host(), _port())
