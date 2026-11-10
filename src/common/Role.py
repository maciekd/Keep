"""Which process an install's `floxy_ai_server.py` is: the Keep server, or an iMessage satellite.

The root config key `role` decides, read once at the start. Absent or `server`, the launcher
builds the assistant as it always did. `imessage_satellite` makes the same entry point — and
so the same start.sh, login applet, requirements check and updater — run a process without an
assistant: the HTTP service that lends this Mac user's Messages account to a Keep server
elsewhere (`common/imessage/Satellite.py`)."""
from common.ConfigManager import config
from common.logging import info

KEY = "role"
SERVER = "server"
IMESSAGE_SATELLITE = "imessage_satellite"
ROLES = (SERVER, IMESSAGE_SATELLITE)


def current() -> str:
    """The role config.yaml names, lowercased; `server` when it names none. A value that is no
    role comes back as written — the launcher refuses to start on it, since guessing would
    start an assistant on a Mac meant to have none."""
    role = str(config.get(KEY, SERVER) or SERVER).strip().lower()
    info(f"Role: {role}" + ("" if role in ROLES else f" — not one of {', '.join(ROLES)}"))
    return role
