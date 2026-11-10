"""An Interface is one way for people to reach the assistant: HTTP, VoiceToVoice, Telegram,
the TUI console, iMessage. Each is a self-contained unit the launcher starts only when its
`interfaces.<name>.enabled` flag says so — a disabled interface's module is never even
imported, which is what lets a Telegram-only install boot on the base requirements.

Start order (see floxy_ai_server.build): http first (it owns the Quart app the others
register routes on), then console, voice, telegram, imessage. `stop()` runs in reverse."""
from abc import ABC, abstractmethod

from common.logging import info, note

# Whether the sockets answer the network or only this Mac: an install starts loopback-only and
# the assistant may open it (`config_set`), read at startup like the enabled flags.
LAN_ACCESS_KEY = "interfaces.lan_access"
LOOPBACK = "127.0.0.1"
ALL_INTERFACES = "0.0.0.0"


def bind_host(config, what: str) -> str:
    """The address `what` (a socket's name, for the log) binds: every interface of the machine
    when `interfaces.lan_access` is on, else the loopback — a server nobody but this Mac reaches."""
    if bool(config.get(LAN_ACCESS_KEY, False)):
        note(f"{what} answers the whole network ({LAN_ACCESS_KEY} is on): anyone who reaches "
             f"this Mac's address can use the assistant as the owner.")
        return ALL_INTERFACES
    info(f"{what} answers this Mac only ({LAN_ACCESS_KEY} is off); phones and a TUI on another "
         f"machine need the key on.")
    return LOOPBACK


class Interface(ABC):
    """One way for people to reach the assistant, started by the launcher only when enabled."""

    name: str = ""
    # What `interfaces.<name>.enabled` means when the config does not say. Voice is off: it
    # needs requirements-voice.txt, microphones and speakers; so is iMessage, which needs an
    # Apple ID of the assistant's own.
    enabled_by_default: bool = True

    @classmethod
    def enabled(cls, config) -> bool:
        """Read `interfaces.<name>.enabled` from the ConfigManager, `enabled_by_default` when unset."""
        return bool(config.get(f'interfaces.{cls.name}.enabled', cls.enabled_by_default))

    @abstractmethod
    def start(self, assistant, app) -> None:
        """Bring the interface up against `assistant`; `app` is the Quart app (None for
        the http interface itself, which creates it)."""

    async def start_async(self) -> None:
        """Hook run on the server event loop just before serving (for interfaces that own
        asyncio tasks, e.g. the voice interface's outpost TCP server)."""

    async def stop_async(self) -> None:
        """Counterpart of start_async, awaited when serve() returns."""

    def stop(self) -> None:
        """Shut down; called from the launcher's atexit hook, in reverse start order."""
