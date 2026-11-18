"""Tells a logged error that may be a defect in the code from one the code foresaw: a service
that is away, a sign-in that expired, a setting that is missing.

Both are logged with `error()`, and only the first belongs in the error tracking, where a new
issue should mean a new bug — `Telemetry.error_report` sends the other as a plain `problem`
event. The rule, first match wins:

1. nothing caught the exception → a defect
2. the error is a message with no exception behind it → foreseen: somebody wrote that
   message for that case
3. the exception is a programming error (`PROGRAMMING_ERRORS`), or a service's answer that the
   request itself was malformed (`MALFORMED_REQUEST`) → a defect, whatever the call site says
4. the call site said `error(…, outside=True)` → foreseen
5. the exception, or one it came from, is the system's or the network's (`OSError`), a client
   library's (`OUTSIDE_LIBRARIES`) or of a class this tree defines — a failure somebody gave
   a name to → foreseen
6. anything else → a defect

The doubt goes to the error tracking. An issue there that is no bug is answered by a line
here or where it is raised — the library added, a class of its own for the failure,
`outside=True` at the call site — and never by a rule that guesses from the message."""
from typing import Optional

PROGRAMMING_ERRORS = (TypeError, AttributeError, NameError, LookupError, AssertionError, ImportError,
                      SyntaxError, RecursionError, NotImplementedError, ArithmeticError)
# The statuses that blame the request's form; 401/403/404/408/409/429 are about access, what
# is there, time and rate — the state of the other side.
MALFORMED_REQUEST = frozenset({400, 405, 413, 414, 415, 422})
# Top-level modules of the libraries that talk to something outside the process: what they
# raise in a class of their own is the outside saying no.
OUTSIDE_LIBRARIES = frozenset({
    "requests", "urllib3", "http", "httpx", "httpcore", "aiohttp", "websockets", "websocket", "curl_cffi",
    "cloudscraper", "nodriver", "googlesearch",
    "anthropic", "openai", "google", "grpc", "elevenlabs",
    "telegram", "telethon", "neonize", "imaplib", "imap_tools", "caldav", "msal", "pyicloud", "spotipy",
    "pyaarlo", "ring_doorbell", "soco", "hyundai_kia_connect_api", "bleak", "bleak_retry_connector", "pyplejd",
    "sounddevice",
})
TREE_PACKAGES = frozenset({"assistant", "common", "integrations", "interfaces"})


def _status(exception: BaseException) -> Optional[int]:
    """The HTTP status *exception* carries, as the client libraries name it."""
    for holder in (exception, getattr(exception, "response", None)):
        for name in ("status_code", "status", "code"):
            value = getattr(holder, name, None)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return None


def _foreseen(exception: BaseException) -> Optional[str]:
    """Why *exception* is a failure the code foresaw — None when it is not."""
    kind = type(exception)
    package = kind.__module__.split(".", 1)[0]
    if isinstance(exception, OSError):
        return f"{kind.__name__} is the system's or the network's"
    if package in OUTSIDE_LIBRARIES:
        return f"{kind.__name__} is raised by {package}"
    if package in TREE_PACKAGES:
        return f"{kind.__name__} is a failure {kind.__module__} names"
    return None


def classify(chain: list[BaseException], handled: bool = True, outside: bool = False) -> tuple[bool, str]:
    """Whether the error is a defect, and why. *chain* is the exception it was logged for and
    the ones that came from, newest first — empty for an error that is a message alone."""
    if not handled:
        return True, "nothing caught it"
    if not chain:
        return False, "a message with no exception behind it"
    logged = chain[0]
    if isinstance(logged, PROGRAMMING_ERRORS):
        return True, f"{type(logged).__name__} is a programming error"
    if _status(logged) in MALFORMED_REQUEST:
        return True, f"status {_status(logged)} blames the request"
    if outside:
        return False, "the call site reports the outside"
    for exception in chain:
        why = _foreseen(exception)
        if why:
            return False, why
    return True, f"{type(logged).__name__} is nobody's named failure"
