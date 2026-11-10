import base64
import json
import re
import threading
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit

import requests

from common import Version
from common.imessage import Wire
from common.imessage.MessagesHost import (AttachmentData, BAD_ANSWER, Count, GroupList, HostHealth, MessageBatch,
                                          MessagesHost, NOT_CONFIGURED, REFUSED, Recipient, SatelliteStatus,
                                          SendResult, TYPING_FAILED, TypingResult, UNREACHABLE)
from common.logging import error, info, note, trace

CONNECT_TIMEOUT = 3
# Seconds without a byte from the satellite before a call is given up, by call. A send waits
# for Messages.app there (15 s of AppleScript); a refreshing health check re-reads the contacts;
# the dots take as long as a chat's window takes to open there, and a reply waits for no more.
READ_TIMEOUT = 15
READ_TIMEOUTS = {"send": 40, "send_chat": 40, "health": 30, "typing": 10}
# The read timeout is per read: a peer that sends a byte now and then never trips it. So every
# exchange also has a deadline for the whole of it — a JSON call its read timeout and this much
# for its answer to cross, a file `ATTACHMENT_TIMEOUT` in all.
TRANSFER_SECONDS = 30
ATTACHMENT_TIMEOUT = 120
# An exchange given up at its deadline still holds a thread until the peer lets go of it; with
# this many hanging nothing more is sent, so a peer that never finishes costs no thread per poll.
MAX_ABANDONED = 3
CHUNK_BYTES = 64 * 1024
UNUSABLE_ADDRESS = "an unusable address"    # stands in for one that cannot be shown: it may hold a pasted token

_ERRNO_RE = re.compile(r"\[Errno -?\d+\] ([^'\")]+)")


@dataclass
class _Answer:
    """What one exchange came to: a body to read, or the reason there is none."""
    body: bytes = b""
    reason: str = ""        # a HostHealth reason when the exchange failed
    detail: str = ""
    reached: bool = False   # a satellite answered, whatever it said
    unsure: bool = False    # the request may have arrived although no answer did
    # The wire versions are too far apart: nothing to do but wait for the end that is behind —
    # a satellite brings itself to this side's code, a server takes its own next update.
    skewed: bool = False

    @property
    def away(self) -> bool:
        """Waiting is all there is to do: nobody is there, or who is there cannot be talked to yet."""
        return self.reason == UNREACHABLE or self.skewed


class RemoteMessagesHost(MessagesHost):
    """The Messages account of another Mac, or of another user of this one, reached through its satellite over HTTP.

    Every `MessagesHost` call is one request to a `SatelliteService` (`Wire` has the routes and
    the headers). Whatever goes wrong on the way — no address, no token, no connection, no
    answer in time, a refused token, an answer that is not a satellite's — ends here as the
    error of the answer, as the contract asks; nothing is retried, the caller polls again.
    Each exchange runs on a thread of its own, so that the whole of it has a deadline whatever
    the peer does (`_send`); `away()` is the verdict of the calling thread's own last call.
    The token is sent in a header and is in no URL, log line or error text. Each request also
    says which branch and commit this process runs: the satellite follows them."""

    NO_ADDRESS = "the address is not of the form http://<host>:<port>"
    NO_TOKEN = "no token is set"
    BAD_TOKEN = "the token holds a space, a line break or a character that is not plain ASCII"

    def __init__(self, address: str, token: str, session: Optional[requests.Session] = None):
        self.address = (address or "").strip().rstrip("/")
        token = (token or "").strip()
        self._misconfigured = self._check(self.address, token)
        if self._misconfigured == self.NO_ADDRESS:
            self.address = UNUSABLE_ADDRESS
        self._session = session or requests.Session()
        self._session.trust_env = False     # no proxy and no ~/.netrc login between the two Macs
        if not self._misconfigured:
            self._session.headers.update({Wire.TOKEN_HEADER: f"{Wire.TOKEN_SCHEME}{token}",
                                          Wire.WIRE_HEADER: str(Wire.WIRE_VERSION)})
        branch, commit = Version.branch(), Version.commit()
        if branch and commit:
            self._session.headers.update({Wire.BRANCH_HEADER: branch, Wire.COMMIT_HEADER: commit})
        else:
            info(f"iMessage satellite {self.address}: this process cannot tell its branch and commit "
                 f"({branch or 'no branch'}, {commit[:8] or 'no commit'}) — the satellite will not follow it")
        self._peer_wire = 0                 # the wire version the satellite last answered with
        self._answering: Optional[bool] = None
        # What each thread's last exchange came to: a consumer's poll and a command's send share
        # the host, and one's outcome must not be read as the other's.
        self._last = threading.local()
        self._lock = threading.Lock()
        self._abandoned: list[threading.Thread] = []    # exchanges given up at their deadline and still running
        if self._misconfigured:
            info(f"iMessage satellite: not usable as configured — {self._misconfigured}")
        else:
            info(f"iMessage satellite: {self.address}, wire version {Wire.WIRE_VERSION}")

    @classmethod
    def _check(cls, address: str, token: str) -> str:
        """Why nothing can be sent with this address and token, "" when it can."""
        try:
            parts = urlsplit(address)
            usable = (parts.scheme in ("http", "https") and bool(parts.hostname) and parts.port != 0
                      and not parts.username and not parts.password
                      and parts.path in ("", "/") and not parts.query and not parts.fragment)
        except ValueError:
            usable = False
        if not usable:
            return cls.NO_ADDRESS
        if not token:
            return cls.NO_TOKEN
        if not Wire.token_fits(token):
            return cls.BAD_TOKEN
        return ""

    def _noted(self, answer: _Answer, what: str) -> _Answer:
        """Logs the satellite going quiet and coming back once each — a consumer polls every few
        seconds — and every call a satellite that does answer turned down."""
        if answer.reached != self._answering:
            self._answering = answer.reached
            if answer.reached:
                info(f"iMessage satellite {self.address}: answering (wire version {self._peer_wire})")
            else:
                info(f"iMessage satellite {self.address}: not answering — {what}: {answer.detail}")
        elif not answer.reached:
            trace(f"iMessage satellite {self.address}: still not answering — {what}: {answer.detail}")
        if answer.reached and answer.reason:
            info(f"iMessage satellite {self.address}: {what} failed — {answer.detail}")
        return answer

    def _exchange(self, method: str, path: str, what: str, arguments: Optional[dict] = None,
                  timeout: Optional[float] = None, limit: Optional[int] = None,
                  deadline: Optional[float] = None) -> _Answer:
        if self._misconfigured:
            return _Answer(reason=NOT_CONFIGURED, detail=self._misconfigured)
        timeout = timeout or READ_TIMEOUT
        answer = self._send(method, path, what, arguments, timeout, limit or Wire.MAX_ANSWER_BYTES,
                            deadline or timeout + TRANSFER_SECONDS)
        self._last.away = answer.away
        return self._noted(answer, what)

    def away(self) -> bool:
        return getattr(self._last, "away", False)

    def _send(self, method, path, what, arguments, timeout, limit, deadline) -> _Answer:
        """One exchange, given `deadline` seconds for the whole of it: it runs on a thread of its
        own and the caller waits no longer. One given up on ends at its next chunk, or when the
        peer lets go of it."""
        with self._lock:
            self._abandoned = [thread for thread in self._abandoned if thread.is_alive()]
            hanging = len(self._abandoned)
        if hanging >= MAX_ABANDONED:
            return _Answer(reason=UNREACHABLE,
                           detail=f"{hanging} earlier requests are still waiting for the end of an answer")
        outcome: list[_Answer] = []
        given_up = threading.Event()
        exchange = threading.Thread(name="imessage-satellite-call", daemon=True, target=lambda: outcome.append(
            self._request(method, path, what, arguments, timeout, limit, given_up)))
        exchange.start()
        exchange.join(deadline)
        if outcome:
            return outcome[0]
        given_up.set()
        with self._lock:
            self._abandoned.append(exchange)
        return _Answer(reason=UNREACHABLE, detail=f"{what} was not through within {deadline:.0f} s", unsure=True)

    def _request(self, method, path, what, arguments, timeout, limit, given_up: threading.Event) -> _Answer:
        try:
            with self._session.request(method, f"{self.address}{path}", json=arguments, stream=True,
                                       timeout=(CONNECT_TIMEOUT, timeout), allow_redirects=False) as response:
                problem = self._peer_problem(response)
                if problem:
                    return problem
                declared = Wire.number(response.headers.get("Content-Length", ""))
                if declared is not None and declared > limit:
                    return _Answer(reason=BAD_ANSWER, reached=True,
                                   detail=f"the answer to {what} is {declared} bytes, over the {limit} this side reads")
                body = bytearray()
                for chunk in response.iter_content(CHUNK_BYTES):
                    body.extend(chunk)
                    if len(body) > limit:
                        return _Answer(reason=BAD_ANSWER, reached=True,
                                       detail=f"the answer to {what} is over the {limit} bytes this side reads")
                    if given_up.is_set():
                        trace(f"iMessage satellite {self.address}: {what} given up at its deadline, "
                              f"{len(body)} bytes read")
                        return _Answer(reason=UNREACHABLE, reached=True)
                return self._judged(response.status_code, bytes(body), what)
        except requests.ConnectTimeout:
            return _Answer(reason=UNREACHABLE, detail=f"no connection within {CONNECT_TIMEOUT} s")
        except requests.Timeout:
            return _Answer(reason=UNREACHABLE, detail=f"no answer to {what} within {timeout:.0f} s", unsure=True)
        except requests.ConnectionError as e:
            text = str(e)
            cause = _ERRNO_RE.search(text)
            return _Answer(reason=UNREACHABLE, unsure="Failed to establish a new connection" not in text,
                           detail=f"no connection ({cause.group(1).strip() if cause else type(e).__name__})")
        except requests.RequestException as e:
            return _Answer(reason=UNREACHABLE, detail=f"the request failed ({type(e).__name__})", unsure=True)
        except Exception as e:      # the contract: whatever happens here is a value to the caller
            error(f"iMessage satellite {self.address}: {what} failed on this side — {type(e).__name__}: {e}")
            return _Answer(reason=BAD_ANSWER, detail=f"{what} failed on this side ({type(e).__name__})", unsure=True)

    def _peer_problem(self, response) -> Optional[_Answer]:
        """What the answer's status and wire version rule out before its body is read."""
        status = response.status_code
        if status == 401:
            return _Answer(reason=REFUSED, detail="the satellite refused the token")
        wire = Wire.number(response.headers.get(Wire.WIRE_HEADER, ""))
        if wire is None:
            return _Answer(reason=BAD_ANSWER,
                           detail=f"HTTP {status} without a wire version — what answers there is not a Keep iMessage satellite")
        if wire != self._peer_wire:
            self._peer_wire = wire
            if self._peer_wire != Wire.WIRE_VERSION:
                note(f"iMessage satellite {self.address}: it speaks wire version {self._peer_wire}, "
                     f"this side {Wire.WIRE_VERSION}")
        if abs(self._peer_wire - Wire.WIRE_VERSION) > Wire.WIRE_TOLERANCE:
            behind = "the satellite has updated itself" if self._peer_wire < Wire.WIRE_VERSION else "this server is updated"
            return _Answer(reason=BAD_ANSWER, reached=True, skewed=True,
                           detail=f"the satellite speaks wire version {self._peer_wire}, this side "
                                  f"{Wire.WIRE_VERSION} — too far apart to work together until {behind}")
        return None

    def _judged(self, status: int, body: bytes, what: str) -> _Answer:
        if status == 200:
            return _Answer(body, reached=True)
        said = ""
        try:
            said = str(json.loads(body).get("error", ""))[:300]
        except (ValueError, AttributeError, RecursionError):
            pass
        if status == 404 and what in Wire.calls():
            said = said or f"the satellite does not know the call '{what}'"
        skew = (f" (its wire version {self._peer_wire}, this side {Wire.WIRE_VERSION})"
                if self._peer_wire != Wire.WIRE_VERSION else "")
        return _Answer(reason=BAD_ANSWER, detail=f"{said or f'HTTP {status}'}{skew}", reached=True)

    def _ask(self, name: str, **arguments):
        """One JSON call: (the answer as the result type the interface names, how the exchange
        went) — None with the failure, which an answer of another shape is too."""
        result = Wire.calls()[name].result
        answer = self._exchange("POST", f"{Wire.ROUTE}/{name}", name, arguments,
                                timeout=READ_TIMEOUTS.get(name, READ_TIMEOUT))
        if answer.reason:
            return None, answer
        try:
            data = json.loads(answer.body)
            if Wire.answers(result, data):
                return Wire.from_wire(result, data), answer
        except Exception as e:
            # Not JSON, a field without a default missing — or JSON nested deeper than Python
            # reads, a RecursionError and no ValueError. Whatever a peer's bytes do is a value;
            # so is an object without the fields every answer of the type carries (`Wire.answers`).
            trace(f"iMessage satellite {self.address}: the answer to {name} could not be read ({type(e).__name__})")
        return None, self._noted(_Answer(reason=BAD_ANSWER, reached=True,
                                         detail=f"the answer to {name} is not a {result.__name__}"), name)

    def _call(self, name: str, **arguments):
        """One JSON call, the failure as the error of its result type."""
        decoded, answer = self._ask(name, **arguments)
        if decoded is None:
            return Wire.calls()[name].result(error=f"iMessage satellite at {self.address}: {answer.detail}")
        return decoded

    def health(self, refresh: bool = False) -> HostHealth:
        health, answer = self._ask("health", refresh=refresh)
        if health is None:
            # a satellite too far away to talk to has still said its wire version: which side is behind
            return HostHealth(False, answer.reason, answer.detail,
                              satellite=SatelliteStatus(self._peer_wire) if answer.skewed else None)
        health.satellite = health.satellite or SatelliteStatus()    # it answered, whatever it says of itself
        health.satellite.wire = self._peer_wire
        return health

    def cursor_at(self, horizon_ts: Optional[float]) -> Count:
        return self._call("cursor_at", horizon_ts=horizon_ts)

    def last_rowid(self) -> Count:
        return self._call("last_rowid")

    def remaining(self, after_rowid: int, horizon_ts: Optional[float]) -> Count:
        return self._call("remaining", after_rowid=after_rowid, horizon_ts=horizon_ts)

    def poll(self, after_rowid: int, horizon_ts: Optional[float], limit: int) -> MessageBatch:
        batch = self._call("poll", after_rowid=after_rowid, horizon_ts=horizon_ts, limit=limit)
        if batch.error:
            batch.cursor = after_rowid      # the caller's cursor stays where it was
        elif batch.cursor < after_rowid:
            # an answer without a cursor, or one that would take the caller back through its history
            info(f"iMessage satellite {self.address}: the poll after ROWID {after_rowid} was answered with the "
                 f"cursor {batch.cursor} — the caller's is kept")
            batch.cursor = after_rowid
        return batch

    def messages(self, rowids: list[int]) -> MessageBatch:
        return self._call("messages", rowids=rowids)

    def scheduled(self) -> MessageBatch:
        return self._call("scheduled")

    def resolve(self, recipient: str, owner: str = "") -> Recipient:
        return self._call("resolve", recipient=recipient, owner=owner)

    def send(self, recipient: str, body: str, owner: str = "") -> SendResult:
        return self._sent(*self._ask("send", recipient=recipient, body=body, owner=owner))

    def send_chat(self, chat: str, body: str) -> SendResult:
        return self._sent(*self._ask("send_chat", chat=chat, body=body))

    def _sent(self, result: Optional[SendResult], answer: _Answer) -> SendResult:
        if result is not None:
            return result
        error = f"iMessage satellite at {self.address}: {answer.detail}"
        if answer.unsure:
            # the request may have arrived: Messages there may have sent the text
            error += " — the message may have been sent all the same, check before sending it again"
        return SendResult(error=error)

    def groups(self) -> GroupList:
        return self._call("groups")

    def typing(self, chat: str, on: bool) -> TypingResult:
        """One request per `on` and per `off`: the satellite keeps the dots alive, and takes
        them away by itself when this side goes quiet."""
        result = self._call("typing", chat=chat, on=on)
        if result.error:
            result.reason = TYPING_FAILED
        return result

    def read_attachment(self, attachment_id: int) -> tuple[Optional[bytes], str]:
        """The file's bytes as the body of a GET, never as base64 in JSON; a file over
        `Wire.MAX_ATTACHMENT_BYTES` is refused by either end."""
        what = f"attachment {attachment_id}"
        answer = self._exchange("GET", f"{Wire.ATTACHMENT_ROUTE}/{int(attachment_id)}", what,
                                limit=Wire.MAX_ATTACHMENT_BYTES, deadline=ATTACHMENT_TIMEOUT)
        if answer.reason:
            return None, f"iMessage satellite at {self.address}: {answer.detail}"
        trace(f"iMessage satellite {self.address}: {what} read, {len(answer.body)} bytes")
        return answer.body, ""

    def attachment(self, attachment_id: int) -> AttachmentData:
        raw, reason = self.read_attachment(attachment_id)
        if raw is None:
            return AttachmentData(error=reason)
        return AttachmentData(base64.b64encode(raw).decode("ascii"))
