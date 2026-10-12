"""The destinations: the places the assistant can post a message to, registered by the interfaces.

A turn answers where it was asked; `post_message(destination, text)` says something somewhere
else — the owner's chats, a Telegram group the bot is in, a room's speakers. The assistant never
imports `interfaces.*`, so each interface registers what it can deliver to when it starts (the
way `set_notifier` hands over the chat hub) and drops it when it stops: the chat hub `owner`,
the Telegram transport one destination per group on its allow list, the voice interface a room.

The kind decides the rendering: a chat or a group takes the text as the chat transport renders it
(Telegram HTML); a voice destination gets the plain words, tags stripped. Lookup is
case-insensitive over the names and the aliases (a group's chat id as text). Listeners — the
PromptManager, whose `{DESTINATIONS}` placeholder lists what can be reached — are told after every
change; they are held weakly, so a manager a test dropped is not called back.
"""
import html
import re
import threading
import weakref
from dataclasses import dataclass, field
from typing import Callable

from common.logging import error, info, trace

KIND_CHAT = "chat"      # the owner's own chats
KIND_GROUP = "group"    # a group chat others read too
KIND_VOICE = "voice"    # a room's speakers

_KIND_LABELS = {KIND_CHAT: "your chats", KIND_GROUP: "Telegram group", KIND_VOICE: "speakers"}
NONE_YET = "none yet — no interface has registered one"


@dataclass
class Destination:
    name: str                       # what the LLM writes: "owner", "Maciek & Erik - Michelle test", "kitchen"
    kind: str                       # KIND_CHAT | KIND_GROUP | KIND_VOICE — decides the rendering
    send: Callable[[str], bool]     # delivers the rendered text; True when the platform took it
    aliases: tuple[str, ...] = field(default_factory=tuple)

    def answers_to(self, name: str) -> bool:
        wanted = name.strip().lower()
        return wanted == self.name.lower() or any(wanted == alias.lower() for alias in self.aliases)


_lock = threading.RLock()
_destinations: list[Destination] = []
_listeners: list[weakref.ref] = []


def register(destination: Destination) -> None:
    """Add *destination*, replacing one of the same name (a restarted interface registers again)."""
    with _lock:
        before = len(_destinations)
        _destinations[:] = [d for d in _destinations if d.name.lower() != destination.name.lower()]
        _destinations.append(destination)
        replaced = len(_destinations) == before
    info(f"Destination {'replaced' if replaced else 'registered'}: {destination.name!r} ({destination.kind}"
         f"{', aliases ' + ', '.join(destination.aliases) if destination.aliases else ''})")
    _notify()


def unregister(name: str) -> None:
    with _lock:
        kept = [d for d in _destinations if d.name.lower() != name.lower()]
        gone = len(kept) < len(_destinations)
        _destinations[:] = kept
    if gone:
        info(f"Destination unregistered: {name!r}")
        _notify()
    else:
        trace(f"Destination {name!r} was not registered, nothing to drop")


def names() -> list[str]:
    with _lock:
        return [d.name for d in _destinations]


def find(name: str) -> Destination | None:
    with _lock:
        return next((d for d in _destinations if d.answers_to(name)), None)


def describe() -> str:
    """The line the prompts show: `owner (your chats), "Maciek & Erik - Michelle test" (Telegram group), kitchen (speakers)`."""
    with _lock:
        if not _destinations:
            return NONE_YET
        parts = []
        for d in _destinations:
            shown = d.name if re.fullmatch(r"[\w-]+", d.name) else f'"{d.name}"'
            parts.append(f"{shown} ({_KIND_LABELS.get(d.kind, d.kind)})")
    return ", ".join(parts)


def send(name: str, text: str) -> str:
    """Deliver *text* to the destination called *name*. Returns '' or the reason it failed."""
    destination = find(name)
    if destination is None:
        info(f"post_message: no destination answers to {name!r} (registered: {', '.join(names()) or 'none'})")
        return f"post_message failed: unknown destination {name!r} — available: {describe()}"
    rendered = plain_text(text) if destination.kind == KIND_VOICE else text
    if not rendered.strip():
        return "post_message failed: the text is empty"
    info(f"post_message: {len(rendered)} chars to {destination.name!r} ({destination.kind})")
    try:
        taken = destination.send(rendered)
    except Exception as e:
        error(f"post_message: {destination.name!r} raised {type(e).__name__}: {e}")
        return f"post_message failed: {destination.name!r} could not take the message ({type(e).__name__})"
    if not taken:
        return f"post_message failed: {destination.name!r} did not take the message (is it running?)"
    return ""


def plain_text(text: str) -> str:
    """The words of a chat-formatted text: tags gone, entities decoded — what a voice can say."""
    return html.unescape(re.sub(r"<[^>]+>", "", text)).strip()


def add_listener(callback: Callable[[], None]) -> None:
    """Call *callback* (a bound method, held weakly) after every change of the registry."""
    with _lock:
        _listeners.append(weakref.WeakMethod(callback) if hasattr(callback, "__self__") else weakref.ref(callback))


def _notify() -> None:
    with _lock:
        refs = list(_listeners)
    for ref in refs:
        callback = ref()
        if callback is None:
            with _lock:
                if ref in _listeners:
                    _listeners.remove(ref)
            continue
        try:
            callback()
        except Exception as e:
            error(f"Destinations: a listener raised {type(e).__name__}: {e}")


def clear() -> None:
    """Forget every destination and listener — for tests."""
    with _lock:
        _destinations.clear()
        _listeners.clear()
