import asyncio
import hmac
from typing import Optional

from quart import Quart, Response, g, has_request_context, has_websocket_context, request, websocket
from werkzeug.exceptions import HTTPException

from common import Version
from common.imessage import Wire
from common.imessage.MessagesHost import HostHealth, MessagesHost, SatelliteStatus
from common.logging import error, info, note, trace

MAX_POLL_ROWS = 2000    # the most rows one poll reads, whatever its caller asks for
# How long hypercorn keeps an idle connection open. Its own 5 s is the feed's poll interval: every
# poll would then race the close of the connection it is about to reuse, and a lost race reads as
# a satellite that is away — or, for a send, as a message that "may have been sent".
KEEP_ALIVE_SECONDS = 120


def hypercorn_config(bind: str):
    """How hypercorn serves the app at `bind` (`<host>:<port>`): nothing that names the software."""
    from hypercorn.config import Config
    config = Config()
    config.bind = [bind]
    config.debug = False
    config.include_server_header = False
    config.keep_alive_timeout = KEEP_ALIVE_SECONDS
    return config


def _asked() -> str:
    """What was asked for, for a log line — an error handler also runs for what is no request."""
    if has_request_context():
        return f"{request.method} {request.path[:80]!r}"
    if has_websocket_context():
        return f"a WebSocket upgrade to {websocket.path[:80]!r}"
    return "something that is no request"


class SatelliteService:
    """Serves one MessagesHost over HTTP to the Keep server that holds its token.

    The other end of `RemoteMessagesHost`; `Wire` names the routes, the headers and the limits.
    The app answers nothing without the token — compared in constant time, before the route is
    even looked up, so a stranger on the network learns no more than that something listens —
    and nothing to a caller whose wire version is more than one from its own. A WebSocket
    upgrade is turned down the same way, before routing: no call is one. It reads no
    body over `Wire.MAX_REQUEST_BYTES` and no arguments but a JSON object of the types the call
    takes. Each authenticated request is also the news of what the server runs, handed to the
    `follower`. The host's calls block (sqlite, AppleScript), so they run off the event loop."""

    def __init__(self, host: MessagesHost, token: str, follower=None):
        self.host = host
        self.follower = follower    # a ServerFollower, or None: nobody to follow (a test, a one-off)
        self.wire = Wire.WIRE_VERSION
        self._expected = f"{Wire.TOKEN_SCHEME}{token}".encode()
        self._caller_wire: Optional[int] = None
        self.app = Quart("imessage_satellite", static_folder=None)    # no /static route: nothing but the calls
        self.app.config["MAX_CONTENT_LENGTH"] = Wire.MAX_REQUEST_BYTES
        self.app.before_request(self._gate)
        self.app.before_websocket(self._no_websocket)
        self.app.after_request(self._stamp)
        self.app.register_error_handler(HTTPException, self._refused)
        self.app.register_error_handler(Exception, self._failed)
        self.app.add_url_rule(f"{Wire.ROUTE}/<name>", view_func=self._call, methods=["POST"])
        self.app.add_url_rule(f"{Wire.ATTACHMENT_ROUTE}/<int:attachment_id>", view_func=self._attachment,
                              methods=["GET"])
        info(f"Satellite service: {len(Wire.calls())} calls under {Wire.ROUTE}, attachments under "
             f"{Wire.ATTACHMENT_ROUTE}, wire version {Wire.WIRE_VERSION}")

    def status(self) -> SatelliteStatus:
        if self.follower is not None:
            return self.follower.status()
        return SatelliteStatus(Wire.WIRE_VERSION, Version.detail(), Version.branch() or "", Version.commit())

    def _stranger(self, asked) -> str:
        """Why whoever asks is not the server — no token, a wrong one — "" when it holds the token."""
        presented = asked.headers.get(Wire.TOKEN_HEADER, "").encode("utf-8", "replace")
        if hmac.compare_digest(presented, self._expected):
            return ""
        return "a wrong token" if presented else "no token"

    async def _no_websocket(self):
        """Hypercorn hands an upgrade to the app as a WebSocket, which `before_request` never
        sees: it is answered here, before the route is looked up, like any other request."""
        stranger = self._stranger(websocket)
        info(f"Satellite service: refused {_asked()} from {websocket.remote_addr} — "
             f"{stranger or 'no call is a WebSocket'}")
        if stranger:
            return {"error": "unauthorized"}, 401
        return {"error": "the satellite's calls are plain HTTP requests, not WebSockets"}, 400

    async def _gate(self):
        """The token, then the wire version; what passes is the server, and says what it runs."""
        stranger = self._stranger(request)
        if stranger:
            info(f"Satellite service: refused {_asked()} from {request.remote_addr} — {stranger}")
            return {"error": "unauthorized"}, 401
        g.authenticated = True
        wire = Wire.number(request.headers.get(Wire.WIRE_HEADER, ""))
        if wire is None:
            info(f"Satellite service: refused {request.path[:80]!r} — the caller names no wire version")
            return {"error": f"no {Wire.WIRE_HEADER} header: this is a Keep iMessage satellite, wire version "
                             f"{self.wire}"}, 400
        if wire != self._caller_wire:
            self._caller_wire = wire
            if self._caller_wire != self.wire:
                note(f"Satellite service: the server speaks wire version {wire}, this side {self.wire}")
        if self.follower is not None:
            # before the refusal below: a server too new to talk to is the one to follow most of all
            self.follower.notice(request.headers.get(Wire.BRANCH_HEADER, ""), request.headers.get(Wire.COMMIT_HEADER, ""))
        if abs(self._caller_wire - self.wire) > Wire.WIRE_TOLERANCE:
            return {"error": f"the caller speaks wire version {wire}, this satellite {self.wire} — too far "
                             f"apart to work together"}, 409
        return None

    async def _stamp(self, response):
        if g.get("authenticated"):      # a stranger is not told what answers here
            response.headers[Wire.WIRE_HEADER] = str(self.wire)
        return response

    @staticmethod
    async def _refused(e: HTTPException):
        trace(f"Satellite service: {_asked()} answered {e.code} {e.name}")
        return {"error": e.name}, e.code

    @staticmethod
    async def _failed(e: Exception):
        error(f"Satellite service: {_asked()} failed — {type(e).__name__}: {e}")
        return {"error": f"the satellite failed ({type(e).__name__})"}, 500

    async def _call(self, name: str):
        call = Wire.calls().get(name)
        if call is None:
            info(f"Satellite service: unknown call {name[:40]!r} — a newer server, or not a server at all")
            return {"error": f"the satellite does not know the call '{name[:40]}'"}, 404
        arguments, problem = Wire.arguments(call, await request.get_json(silent=True))
        if problem:
            info(f"Satellite service: {name} refused — {problem}")
            return {"error": f"{name}: {problem}"}, 400
        if "limit" in arguments:
            arguments["limit"] = max(1, min(arguments["limit"], MAX_POLL_ROWS))
        answer = await asyncio.to_thread(getattr(self.host, name), **arguments)
        if isinstance(answer, HostHealth):
            answer.satellite = self.status()
        return Wire.to_wire(answer)

    @staticmethod
    def _too_large(attachment_id: int, size: int):
        megabyte = 1024 * 1024
        info(f"Satellite service: attachment {attachment_id} not sent, {size // megabyte} MB")
        return {"error": f"attachment {attachment_id} is {size // megabyte} MB, more than the "
                         f"{Wire.MAX_ATTACHMENT_BYTES // megabyte} MB a satellite sends"}, 413

    async def _attachment(self, attachment_id: int):
        """The file's bytes. Its size is asked first, so a file over the cap is refused unread;
        what a host that could not tell the size hands over is measured before it is sent."""
        size = await asyncio.to_thread(self.host.attachment_size, attachment_id)
        if not size.error and size.value > Wire.MAX_ATTACHMENT_BYTES:
            return self._too_large(attachment_id, size.value)
        raw, reason = await asyncio.to_thread(self.host.read_attachment, attachment_id)
        if raw is None:
            return {"error": reason}, 404
        if len(raw) > Wire.MAX_ATTACHMENT_BYTES:
            return self._too_large(attachment_id, len(raw))
        trace(f"Satellite service: attachment {attachment_id} sent, {len(raw)} bytes")
        return Response(raw, mimetype="application/octet-stream")
