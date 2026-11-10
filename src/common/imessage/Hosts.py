"""Builds the MessagesHost a consumer's config section names: this Mac's own Messages or a satellite's.

A consumer — the iMessage integration, the iMessage chat interface — has two keys in its own
section: `host`, which is `local` (the default: Messages.app of the user the server runs as) or
the address of a satellite (`http://<host>:8898`), and `host_token`, the token that satellite's
config.yaml holds under `imessage_satellite.token`. This is the one place that reads them."""
from typing import Optional
from urllib.parse import urlsplit

from common.ConfigManager import config
from common.imessage.LocalMessagesHost import LocalMessagesHost
from common.imessage.MessagesHost import MessagesHost
from common.imessage.RemoteMessagesHost import RemoteMessagesHost
from common.logging import info

HOST_KEY = "host"
TOKEN_KEY = "host_token"
LOCAL = "local"
_LOOPBACK = ("localhost", "127.0.0.1", "::1")

_local: Optional[LocalMessagesHost] = None


def _address(section: str) -> str:
    return str(config.get(f"{section}.{HOST_KEY}", LOCAL) or LOCAL).strip()


def is_remote(section: str) -> bool:
    """Whether the section's `host` names a satellite rather than this Mac's own Messages."""
    return _address(section).lower() != LOCAL


def account(section: str) -> str:
    """Which Messages account the section's `host` names, as a value two sections can be
    compared by: `local`, or the satellite's scheme, host and port — one spelling for this
    Mac's loopback names. Two addresses of one satellite that are spelled otherwise (its name
    and its IP) are not seen to be the same."""
    address = _address(section)
    if address.lower() == LOCAL:
        return LOCAL
    try:
        parts = urlsplit(address)
        host, port = (parts.hostname or "").lower(), parts.port
    except ValueError:
        return address.lower().rstrip("/")
    return f"{parts.scheme.lower()}://{'loopback' if host in _LOOPBACK else host}:{port or ''}"


def same_account(section: str, other: str) -> bool:
    """Whether two consumers' sections name the same Messages account: every message it
    receives would then reach both."""
    return account(section) == account(other)


def local() -> LocalMessagesHost:
    """This process's own Messages account — one object however many consumers use it, since the
    send rate limit and the contact cache belong to the account."""
    global _local
    if _local is None:
        _local = LocalMessagesHost()
        info("iMessage host: this Mac's own Messages (local)")
    return _local


def build(section: str) -> MessagesHost:
    """The host `<section>.host` names. A remote one is built anew at each call, so a consumer
    that is restarted picks up a changed address or token."""
    if not is_remote(section):
        info(f"iMessage host for '{section}': local")
        return local()
    info(f"iMessage host for '{section}': the satellite at {section}.{HOST_KEY}")
    return RemoteMessagesHost(_address(section), str(config.get(f"{section}.{TOKEN_KEY}", "") or ""))
