"""The phones that take notifications from this server: per phone the address Apple knows it
by, the key its notifications are encrypted with, and how many it has been sent since
somebody last looked at it (`docs/plans/APP_PUSH_PLAN.md`).

A phone registers itself at every launch (`POST /app/device`), so a lost file heals by
itself. The file holds the keys and is the owner's alone to read."""
import base64
import json
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from common import PrivateFile
from common.logging import error, info, trace

FILE_NAME = "app_devices.json"
MAX_DEVICES = 8     # a household's phones and tablets; the one registered longest ago leaves first
KEY_BYTES = 32
ENVIRONMENTS = ("production", "sandbox")
_TOKEN = re.compile(r"^[0-9a-f]{64,200}$")
_ID = re.compile(r"^[A-Za-z0-9-]{8,64}$")


class DeviceRefused(ValueError):
    """What a phone sent to register cannot be taken: why, in a few words the app logs."""


@dataclass
class Device:
    id: str             # the app's own, made up at its first launch
    token: str          # Apple's address of the phone, hex
    environment: str    # which of Apple's two services knows the token
    key: str            # base64 of the 32 bytes the phone opens its notifications with
    name: str = ""
    at: float = 0.0     # when it last registered
    unseen: int = 0     # notifications since the phone was last looked at: its badge

    def key_bytes(self) -> bytes:
        return base64.b64decode(self.key)

    def __str__(self) -> str:
        return f"{self.name or 'phone'} [{self.id[:8]}]"


class Devices:
    """The registry, in `<directory>/app_devices.json`; `directory` None keeps nothing on disk."""

    def __init__(self, directory: Optional[Path] = None):
        self._path = Path(directory) / FILE_NAME if directory is not None else None
        self._lock = threading.Lock()
        self._devices: dict[str, Device] = {}
        self._load()

    def all(self) -> list[Device]:
        with self._lock:
            return list(self._devices.values())

    def get(self, device_id: str) -> Optional[Device]:
        with self._lock:
            return self._devices.get(device_id)

    def register(self, device_id: str, token: str, environment: str, key: str, name: str = "") -> bool:
        """The phone *device_id* is reached at *token* from now on, and has just been looked
        at. True when this is news (a phone not seen before, or one whose token, environment
        or key changed). `DeviceRefused` for fields that are not what the wire says."""
        device_id, token, name = str(device_id or ""), str(token or "").lower(), str(name or "")[:40]
        if not _ID.match(device_id):
            raise DeviceRefused("device must be the app's own id")
        if not _TOKEN.match(token):
            raise DeviceRefused("token must be a device token in hex")
        if environment not in ENVIRONMENTS:
            raise DeviceRefused("environment must be production or sandbox")
        try:
            usable = len(base64.b64decode(str(key or ""), validate=True)) == KEY_BYTES
        except ValueError:
            usable = False
        if not usable:
            raise DeviceRefused(f"key must be {KEY_BYTES} bytes in base64")
        with self._lock:
            known = self._devices.pop(device_id, None)
            device = Device(device_id, token, environment, key, name, at=time.time())
            self._devices[device_id] = device
            news = known is None or (known.token, known.environment, known.key) != (token, environment, key)
            # One phone, one address: a token that moved to another id (the app was reinstalled).
            moved = [d for d in self._devices.values() if d.token == token and d.id != device_id]
            for other in moved:
                del self._devices[other.id]
            dropped = []
            while len(self._devices) > MAX_DEVICES:
                dropped.append(self._devices.pop(next(iter(self._devices))))
            if news or (known is not None and known.unseen) or moved or dropped:
                self._save()
        if news:
            info(f"App push: {device} registered ({environment}, token {token[:8]}…)"
                 + (" — its token, environment or key changed" if known is not None else ""))
        else:
            trace(f"App push: {device} is here again, nothing changed")
        for other in moved:
            info(f"App push: {other} forgotten, its token is {device}'s now")
        for other in dropped:
            info(f"App push: {other} forgotten, more than {MAX_DEVICES} devices are registered")
        return news

    def forget(self, device_id: str) -> bool:
        """The phone takes no notifications any more. False when it was not registered."""
        with self._lock:
            device = self._devices.pop(str(device_id or ""), None)
            if device is not None:
                self._save()
        if device is None:
            trace(f"App push: no device [{str(device_id)[:8]}] to forget")
            return False
        info(f"App push: {device} forgotten, its notifications were turned off")
        return True

    def drop_token(self, token: str, why: str) -> None:
        """Apple no longer knows *token*: its phone registers again at its next launch."""
        with self._lock:
            dead = [d for d in self._devices.values() if d.token == token]
            for device in dead:
                del self._devices[device.id]
            if dead:
                self._save()
        for device in dead:
            info(f"App push: {device} forgotten, Apple refused its token ({why})")
        if not dead:
            trace(f"App push: the token {token[:8]}… Apple refused ({why}) is of no registered device")

    def pushed(self, device_id: str) -> int:
        """One more notification is on its way to the phone: how many it has not seen now."""
        with self._lock:
            device = self._devices.get(device_id)
            if device is None:
                trace(f"App push: device [{device_id[:8]}] is gone, nothing to count")
                return 0
            device.unseen += 1
            self._save()
            return device.unseen

    # --- the file -----------------------------------------------------------------------------

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            trace("App push: no device registry on disk yet")
            return
        try:
            saved = json.loads(self._path.read_text())
            self._devices = {d["id"]: Device(**d) for d in saved["devices"]}
        except Exception as e:
            error(f"App push: {self._path.name} cannot be read ({type(e).__name__}: {e}) — starting empty, "
                  f"each phone registers again at its next launch")
            self._devices = {}
            return
        info(f"App push: {len(self._devices)} registered device(s) loaded")

    def _save(self) -> None:
        # Caller holds _lock.
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            PrivateFile.write(self._path, json.dumps({"devices": [asdict(d) for d in self._devices.values()]}, indent=1))
        except OSError as e:
            error(f"App push: could not write {self._path.name}: {e}")
