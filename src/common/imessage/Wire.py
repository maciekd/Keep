"""Turns the values that cross a MessagesHost into JSON-ready dicts and back, and names what the two ends of the HTTP wire agree on.

`from_wire` ignores a field it does not know and leaves a missing one at its default, so the two
ends of a wire may be one version apart. The same holds for a call's arguments (`arguments`):
one the reader does not know is dropped, one the writer did not send takes the default of the
interface.

The wire itself, between `RemoteMessagesHost` and `SatelliteService`: every abstract call of
`MessagesHost` is `POST /messages/<call>` with its arguments as one JSON object and its result
as one — derived from the interface (`calls`), so a call added there is on the wire on both
ends. The exception is `attachment`: the bytes of a file never cross as base64 in JSON, they
are the body of `GET /messages/attachment/<id>`. Every request carries the token and the
caller's wire version, branch and commit in headers; every answer carries the satellite's wire
version."""
import inspect
import re
from dataclasses import asdict, dataclass, fields, is_dataclass
from functools import lru_cache
from typing import Any, Optional, Union, get_args, get_origin, get_type_hints

from common.imessage.MessagesHost import (Count, GroupList, HostHealth, MessageBatch, MessagesHost, Recipient,
                                          SendResult, TypingResult)

# Raised when a call or a field changes meaning; a new call or a new field with a default is
# what `from_wire` and `arguments` already tolerate. Ends more than one apart refuse each other.
WIRE_VERSION = 1
WIRE_TOLERANCE = 1

TOKEN_HEADER = "Authorization"          # "Bearer <token>" — a header, never the URL: URLs end up in logs
TOKEN_SCHEME = "Bearer "
WIRE_HEADER = "X-Keep-Wire"
BRANCH_HEADER = "X-Keep-Branch"         # what the caller runs: the satellite follows it
COMMIT_HEADER = "X-Keep-Commit"

ROUTE = "/messages"
ATTACHMENT_ROUTE = f"{ROUTE}/attachment"
BINARY_CALL = "attachment"              # the one call that is not JSON on the wire

MAX_REQUEST_BYTES = 256 * 1024          # a call's arguments: the longest is the text of a send
MAX_ANSWER_BYTES = 32 * 1024 * 1024     # a JSON answer: a poll of 500 long messages is well under it
MAX_ATTACHMENT_BYTES = 100 * 1024 * 1024    # one file; iMessage itself takes no larger attachment
MAX_LIST_ITEMS = 10_000                 # a list argument (the rows a consumer tracks)
# What an answer cannot be without, by result type: the fields that say what the call came to.
# Every field of a result has a default, so that the ends may be a version apart, and a bare
# `{}` would otherwise read as a call that went well — a send that was made, a ROWID of 0. A
# satellite writes every field it knows (`to_wire`), so an answer lacking one of these is not one.
ANSWER_FIELDS = {
    HostHealth: ("ok",),
    Count: ("value",),
    MessageBatch: ("messages", "cursor"),
    Recipient: ("handle",),
    SendResult: ("service", "error"),
    GroupList: ("groups",),
    TypingResult: ("showing",),
}
_NUMBER_RE = re.compile(r"^[0-9]{1,12}$")
_TOKEN_RE = re.compile(r"^[\x21-\x7e]+$")


def number(text: str) -> Optional[int]:
    """A header's value as a number, None for anything but a few ASCII digits — `str.isdigit`
    says yes to "²", which `int` then refuses."""
    return int(text) if _NUMBER_RE.match(text or "") else None


def token_fits(token: str) -> bool:
    """Whether a token can travel in the header at all: visible ASCII, no space, no line break."""
    return bool(_TOKEN_RE.match(token or ""))


def to_wire(value) -> dict:
    return asdict(value)


def from_wire(cls, data: dict):
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    return cls(**{name: _decode(hints[name], value) for name, value in data.items() if name in known})


def _decode(hint, value) -> Any:
    if value is None:
        return None
    origin = get_origin(hint)
    if origin is Union:
        return _decode(next(arg for arg in get_args(hint) if arg is not type(None)), value)
    if origin is list:
        return [_decode(get_args(hint)[0], item) for item in value]
    if is_dataclass(hint):
        return from_wire(hint, value)
    return value


@dataclass(frozen=True)
class Call:
    """One call of the interface as the wire sees it: what it takes and what it answers with."""
    name: str
    parameters: dict        # argument name → type hint
    defaults: dict          # the arguments that may be left out, with what they then are
    result: type


@lru_cache(maxsize=1)
def calls() -> dict[str, Call]:
    """The calls that cross as JSON, by name — every abstract method of `MessagesHost` but the binary one."""
    found = {}
    for name in sorted(MessagesHost.__abstractmethods__ - {BINARY_CALL}):
        method = getattr(MessagesHost, name)
        hints = get_type_hints(method)
        result = hints.pop("return")
        defaults = {p.name: p.default for p in inspect.signature(method).parameters.values()
                    if p.default is not inspect.Parameter.empty}
        found[name] = Call(name, hints, defaults, result)
    return found


def fits(hint, value) -> bool:
    """Whether a JSON value is what an argument or a result of this type may be. A bool is not
    a number here, a list has a sane length, and a dataclass is an object whose known fields
    fit theirs — one it lacks or does not know is the other end's version, not an error."""
    if is_dataclass(hint):
        hints = get_type_hints(hint)
        return isinstance(value, dict) and all(fits(hints[name], item) for name, item in value.items() if name in hints)
    origin = get_origin(hint)
    if origin is Union:
        return value is None and type(None) in get_args(hint) or any(
            fits(arg, value) for arg in get_args(hint) if arg is not type(None))
    if origin is list:
        return isinstance(value, list) and len(value) <= MAX_LIST_ITEMS and all(
            fits(get_args(hint)[0], item) for item in value)
    if hint is bool:
        return isinstance(value, bool)
    if hint is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if hint is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return hint is str and isinstance(value, str)


def answers(result: type, data) -> bool:
    """Whether a JSON value is an answer of this result type: an object that fits it (`fits`)
    and carries what no answer of the type is without (`ANSWER_FIELDS`)."""
    return fits(result, data) and all(name in data for name in ANSWER_FIELDS[result])


def arguments(call: Call, data) -> tuple[dict, str]:
    """(the keyword arguments of the call, "") from a request's JSON, or ({}, what is wrong
    with it). An argument the interface does not have is left out — a newer caller sent it."""
    if not isinstance(data, dict):
        return {}, "the arguments are not a JSON object"
    found = {}
    for name, hint in call.parameters.items():
        if name not in data:
            if name not in call.defaults:
                return {}, f"'{name}' is missing"
            found[name] = call.defaults[name]
        elif not fits(hint, data[name]):
            return {}, f"'{name}' is not of the type the call takes"
        else:
            found[name] = data[name]
    return found, ""
