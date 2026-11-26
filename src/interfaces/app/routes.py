"""The phone apps' side of the chat, under `/app`: one long response of events, a JSON
object per line, and plain POSTs for what the app does — so each has a status the app
can show. A message with pictures is the one POST that is a form and not JSON. No call names the phone: there is one conversation, and the cursor is the app's own.

Whoever reaches the address is the owner, as for `/stream`."""
import asyncio
import json

from quart import Blueprint, Response, request
from werkzeug.exceptions import RequestEntityTooLarge

from common.logging import info, trace
from interfaces.app.AppTransport import MAX_PICTURES, AppTransport, PicturesRefused
from interfaces.chat.ChatTypes import DECISION_APPROVE, DECISION_DENY
from interfaces.Interface import on_shutdown

HEARTBEAT_SECONDS = 15.0
# The least time between two batches of lines: a streaming reply reaches a phone as its
# newest state at most this often.
BATCH_SECONDS = 0.05
# The most one POST may carry: a message with every picture it may have, each as a phone's
# camera made it. Quart's own default (16 MB) is less than two such pictures.
MAX_PICTURE_BYTES = 16 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_PICTURES * MAX_PICTURE_BYTES


def _encode(line: dict) -> bytes:
    return json.dumps(line, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"


def build_blueprint(transport: AppTransport) -> Blueprint:
    bp = Blueprint("app_chat", __name__, url_prefix="/app")
    transcript = transport.transcript

    @bp.route('/events')
    async def events():
        try:
            after = max(0, int(request.args.get('after', 0) or 0))
        except ValueError:
            return {"error": "after must be a number"}, 400
        log = request.args.get('log', '')

        async def lines():
            loop = asyncio.get_running_loop()
            wake = asyncio.Event()
            backlog, subscription = transcript.subscribe(after, log, lambda: loop.call_soon_threadsafe(wake.set))
            try:
                for line in backlog:
                    yield _encode(line)
                while not subscription.ended:
                    try:
                        await asyncio.wait_for(wake.wait(), HEARTBEAT_SECONDS)
                    except asyncio.TimeoutError:
                        yield b"{}\n"
                        continue
                    wake.clear()
                    for line in subscription.drain():
                        yield _encode(line)
                    await asyncio.sleep(BATCH_SECONDS)
                trace("App chat: the server is stopping, an event stream ends")
            finally:
                transcript.unsubscribe(subscription)

        response = Response(lines(), mimetype="application/x-ndjson", headers={"Cache-Control": "no-store"})
        response.timeout = None     # Quart's 60 s RESPONSE_TIMEOUT would end the stream
        return response

    @bp.route('/send', methods=['POST'])
    async def send():
        # With pictures the message is a form: the same fields, and an `image` part per picture.
        pictures = []
        try:
            if request.mimetype == 'multipart/form-data':
                body = await request.form
                pictures = [part.read() for part in (await request.files).getlist('image')]
            else:
                body = await request.get_json(silent=True) or {}
        except RequestEntityTooLarge:
            info(f"App chat: a message of {request.content_length} bytes is over the {MAX_REQUEST_BYTES} "
                 f"one may have — refused")
            return {"error": f"the message is larger than {MAX_REQUEST_BYTES // (1024 * 1024)} MB"}, 413
        try:
            # The pictures are re-encoded on the way in: off the event loop.
            taken = await asyncio.to_thread(transport.submit, str(body.get('id') or ''), str(body.get('text') or ''),
                                            str(body.get('user') or ''), pictures)
        except PicturesRefused as e:
            return {"error": str(e)}, 400
        if not taken:
            return {"error": "empty message"}, 400
        return {"ok": True}, 200

    @bp.route('/stop', methods=['POST'])
    async def stop():
        transport.interrupt()
        return {"ok": True}, 200

    @bp.route('/approval', methods=['POST'])
    async def approval():
        body = await request.get_json(silent=True) or {}
        decision = body.get('decision')
        if decision not in (DECISION_APPROVE, DECISION_DENY):
            return {"error": "decision must be approve or deny"}, 400
        if not transport.answer_approval(str(body.get('token') or ''), decision):
            return {"error": "no longer pending"}, 410
        return {"ok": True}, 200

    @bp.route('/secret', methods=['POST'])
    async def secret():
        # The value is handed to the hub and nowhere else: never an item, never a log line.
        body = await request.get_json(silent=True) or {}
        value = body.get('value')
        if value is not None and not isinstance(value, str):
            return {"error": "value must be text or null"}, 400
        token = str(body.get('token') or '')
        # The requester's check of the value may take a moment: off the event loop.
        if not await asyncio.to_thread(transport.answer_secret, token, value):
            return {"error": "no longer pending"}, 410
        return {"ok": True}, 200

    @bp.route('/media/<media_id>')
    async def media(media_id: str):
        found = await asyncio.to_thread(transcript.media, media_id)
        if found is None:
            return {"error": "no such media"}, 404
        data, content_type = found
        trace(f"App chat: serving {media_id} ({len(data)} bytes)")
        response = Response(data, mimetype=content_type, headers={"Cache-Control": "private, max-age=604800"})
        # A phone's video player asks for byte ranges and plays nothing from a server without them.
        await response.make_conditional(request, accept_ranges=True, complete_length=len(data))
        return response

    return bp


def register(app, transport: AppTransport) -> None:
    app.register_blueprint(build_blueprint(transport))
    # The limit is the app's, so every route's: never lowered under what another set.
    app.config["MAX_CONTENT_LENGTH"] = max(app.config.get("MAX_CONTENT_LENGTH") or 0, MAX_REQUEST_BYTES)
    on_shutdown(app, transport.transcript.end_listeners)
    info("App chat routes registered (/app/events, /app/send, /app/stop, /app/approval, /app/secret, /app/media).")
