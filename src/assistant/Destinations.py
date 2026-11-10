"""The destinations: the places the assistant can post a message to, registered by the interfaces.

A turn answers where it was asked; `post_message(destination, text)` says something somewhere
else — the owner's chats, a Telegram group the bot is in, a room's speakers. The assistant never
imports `interfaces.*`, so each interface registers what it can deliver to when it starts (the
way `set_notifier` hands over the chat hub) and drops it when it stops: the chat hub `owner`,
the Telegram transport one destination per group on its allow list, the iMessage transport one
per person and group chat it serves, the voice interface a room.

The kind decides the rendering: a chat or a group takes the text as the chat transport renders it
(Telegram HTML); a voice destination gets the plain words, tags stripped. Lookup is
case-insensitive over the names and the aliases (a group's chat id as text). Listeners — the
PromptManager, whose `{DESTINATIONS}` placeholder lists what can be reached — are told after every
change; they are held weakly, so a manager a test dropped is not called back.

Two ways in. `register` / `unregister` go by name: the registrant owns the name and a second
`register` replaces the first (a restarted interface). `claim` / `release` go by the object: a
registrant whose names are not its own to choose (a chat's title, which the people in it set)
gets a name only while nobody answers to it, and takes back only the very registration it made
— never whatever holds that name by then.

A name is somebody's free text and the list is part of every prompt, so `describe` shows no
name that could end its own quotes or its line, and puts every name between quotes that is a
group's — its members chose it — or more than plain ASCII letters and digits. Every way in
compares names as `find` does (`name_key`), so two spellings of one name are one name.
"""
import html
import re
import threading
import weakref
from dataclasses import dataclass, field, replace
from typing import Callable

from common.logging import error, info, trace

KIND_CHAT = "chat"      # the owner's own chats
KIND_GROUP = "group"    # a group chat others read too
KIND_VOICE = "voice"    # a room's speakers

_KIND_LABELS = {KIND_CHAT: "your chats", KIND_GROUP: "Telegram group", KIND_VOICE: "speakers"}
# A destination's `send` answers True when the platform took the text; False, or the reason as
# a text ("the daily limit of 200 messages to Charlotte is reached"), when it did not.
NONE_YET = "none yet — no interface has registered one"


@dataclass
class Destination:
    name: str                       # what the LLM writes: "owner", "Maciek & Erik - Michelle test", "kitchen"
    kind: str                       # KIND_CHAT | KIND_GROUP | KIND_VOICE — decides the rendering
    send: Callable[..., bool | str]     # delivers the rendered text; True when the platform took it, else False or why not
    aliases: tuple[str, ...] = field(default_factory=tuple)
    # True when `send` takes the sender as a second argument — the RequestContext of the turn
    # whose post_message it is, None for an autonomous run. A room's speakers wait for the
    # house to be quiet, and that turn must not count as the house being busy.
    sender_aware: bool = False
    # What the prompt's list calls it, when the kind's own label would be wrong: "iMessage" for a
    # person's chat with the assistant, "iMessage group".
    label: str = ""
    # False for a destination that is found by its name and left out of the prompt's list: a
    # person's second address, which is not a name to show.
    listed: bool = True

    def answers_to(self, name: str) -> bool:
        wanted = _key(name)
        return wanted == _key(self.name) or any(wanted == _key(alias) for alias in self.aliases)


_QUOTES = dict.fromkeys(map(ord, '"“”„‟«»＂'), "'")


def shown_name(name: str) -> str:
    """A destination's name as the prompt may show it: one line, no control character, and no
    double quote of any kind — a name is put between double quotes there, and one that could
    close them would go on as the prompt's own words."""
    one_line = " ".join("".join(c if c.isprintable() else " " for c in str(name)).split())
    return one_line.translate(_QUOTES)


def name_key(name: str) -> str:
    """What a name is looked up by: as it is shown, in lower case — so the spelling the prompt
    lists finds the destination whatever the registered name's quotes were."""
    return shown_name(name).lower()


_key = name_key


_lock = threading.RLock()
_destinations: list[Destination] = []
_listeners: list[weakref.ref] = []


def register(destination: Destination) -> None:
    """Add *destination*, replacing one of the same name (a restarted interface registers again)."""
    with _lock:
        before = len(_destinations)
        _destinations[:] = [d for d in _destinations if _key(d.name) != _key(destination.name)]
        _destinations.append(destination)
        replaced = len(_destinations) == before
    info(f"Destination {'replaced' if replaced else 'registered'}: {destination.name!r} ({destination.kind}"
         f"{', aliases ' + ', '.join(destination.aliases) if destination.aliases else ''})")
    _notify()


def unregister(name: str) -> None:
    with _lock:
        kept = [d for d in _destinations if _key(d.name) != _key(name)]
        gone = len(kept) < len(_destinations)
        _destinations[:] = kept
    if gone:
        info(f"Destination unregistered: {name!r}")
        _notify()
    else:
        trace(f"Destination {name!r} was not registered, nothing to drop")


def claim(destination: Destination) -> Destination | None:
    """Add *destination* unless something answers to its name already; None then, and nothing
    changes. An alias somebody else answers to is left out. Returns the registration as it
    stands — what `release` and `holds` take."""
    with _lock:
        if any(d.answers_to(destination.name) for d in _destinations):
            trace(f"Destination {destination.name!r} not claimed: the name is taken")
            return None
        free = tuple(alias for alias in destination.aliases
                     if _key(alias) != _key(destination.name) and not any(d.answers_to(alias) for d in _destinations))
        claimed = replace(destination, aliases=free)
        _destinations.append(claimed)
    info(f"Destination registered: {claimed.name!r} ({claimed.kind}"
         f"{', aliases ' + ', '.join(claimed.aliases) if claimed.aliases else ''})")
    _notify()
    return claimed


def release(destination: Destination) -> bool:
    """Drop this very registration (what `claim` returned). False, and nothing changes, when
    the registry no longer holds it — its name may be somebody else's by now."""
    with _lock:
        held = any(d is destination for d in _destinations)
        _destinations[:] = [d for d in _destinations if d is not destination]
    if held:
        info(f"Destination unregistered: {destination.name!r}")
        _notify()
    else:
        trace(f"Destination {destination.name!r} is no longer this registration, nothing to drop")
    return held


def holds(destination: Destination) -> bool:
    """Whether the registry still holds this very registration."""
    with _lock:
        return any(d is destination for d in _destinations)


def names() -> list[str]:
    with _lock:
        return [d.name for d in _destinations]


def find(name: str) -> Destination | None:
    with _lock:
        return next((d for d in _destinations if d.answers_to(name)), None)


def describe() -> str:
    """The line the prompts show: `owner (your chats), "Maciek & Erik - Michelle test" (Telegram group), kitchen (speakers)`."""
    with _lock:
        listed = [d for d in _destinations if d.listed]
    if not listed:
        return NONE_YET
    parts = []
    for d in listed:
        bare = d.kind != KIND_GROUP and re.fullmatch(r"[A-Za-z0-9_-]+", d.name)
        shown = d.name if bare else f'"{shown_name(d.name)}"'
        parts.append(f"{shown} ({d.label or _KIND_LABELS.get(d.kind, d.kind)})")
    return ", ".join(parts)


def send(name: str, text: str, sender=None) -> str:
    """Deliver *text* to the destination called *name*. Returns '' or the reason it failed.
    *sender* is the RequestContext of the turn posting it (None outside a turn), handed on
    to a sender-aware destination only."""
    destination = find(name)
    if destination is None:
        info(f"post_message: no destination answers to {name!r} (registered: {', '.join(names()) or 'none'})")
        return f"post_message failed: unknown destination {name!r} — available: {describe()}"
    rendered = plain_text(text) if destination.kind == KIND_VOICE else text
    if not rendered.strip():
        return "post_message failed: the text is empty"
    info(f"post_message: {len(rendered)} chars to {destination.name!r} ({destination.kind})")
    try:
        taken = destination.send(rendered, sender) if destination.sender_aware else destination.send(rendered)
    except Exception as e:
        error(f"post_message: {destination.name!r} raised {type(e).__name__}: {e}")
        return f"post_message failed: {destination.name!r} could not take the message ({type(e).__name__})"
    if isinstance(taken, str) and taken:
        info(f"post_message: {destination.name!r} did not take the message: {taken}")
        return f"post_message failed: {taken}"
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
