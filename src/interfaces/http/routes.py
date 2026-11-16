"""The HTTP interface: `/stream` (GET/POST), used by the iOS app, the Android outposts
and the prompt test suite.

parse → RequestContext → pre-dispatch hooks (debug commands, the microphone vote) →
Assistant.stream() → WireFormat.render(). No assistant logic lives here."""
import asyncio
import traceback
from dataclasses import dataclass, replace
from typing import Any, Optional

from quart import Blueprint, Response, request

from assistant.RequestProcessor import RequestData
from assistant.commands.DebugCommands import DebugCommands
from assistant.Assistant import Assistant, AssistantBusy
from assistant.AssistantEvents import observers
from assistant.AssistantRequest import RequestContext
from assistant.ChannelProfile import SOURCE_TEXT_CHAT
from common.logging import error, info, trace
from interfaces.http import WireFormat

# Keys a JSON POST body may carry in place of query-string args.
BODY_ARGS = ('query', 'text', 'source', 'user', 'lang', 'ai', 'location')


@dataclass
class HttpDeps:
    assistant: Assistant
    # The iOS /stream_text overlay channel (interfaces/voice/ios); None without voice.
    text_stream: Any = None
    # Cross-mic vote / Scribe gate / speaker id (interfaces/voice/microphones); None without voice.
    microphone_routes: Any = None
    # False when no voice interface runs in this process: audio-mode clients get text instead.
    voice_enabled: bool = True
    # False on a server without speakers of its own (common/Platform): a server_sound request
    # is answered as text.
    server_sound: bool = True


_voiceless_audio_client_reported = False
_server_sound_reported = False


def _downgrade_audio_client(context: RequestContext) -> RequestContext:
    """A voice-less process cannot synthesize audio: answer an `audio/mpeg` client with the
    TextChat rendering (text/plain, streamed tokens) rather than an empty audio body."""
    global _voiceless_audio_client_reported
    message = (f"Audio-mode request from {context.source} but the voice interface is disabled — "
               f"answering as TextChat text.")
    if _voiceless_audio_client_reported:
        trace(message)
    else:
        _voiceless_audio_client_reported = True
        info(message)
    return replace(context, source=SOURCE_TEXT_CHAT, text_only=True)


def _downgrade_server_sound(context: RequestContext) -> RequestContext:
    """A server without speakers of its own cannot play the reply: a `server_sound` request
    (the app in its server-sound mode) gets the TextChat rendering instead of silence."""
    global _server_sound_reported
    message = (f"Server-sound request from {context.source}, and this server has no speakers of its own — "
               f"answering as TextChat text.")
    if _server_sound_reported:
        trace(message)
    else:
        _server_sound_reported = True
        info(message)
    return replace(context, source=SOURCE_TEXT_CHAT, text_only=True, server_sound=False)


def _confidence(args: dict) -> Optional[float]:
    """The client's STT confidence, None when it sends none or nonsense."""
    try:
        return float(args['confidence'])
    except (KeyError, ValueError, TypeError):
        return None


def build_blueprint(deps: HttpDeps) -> Blueprint:
    bp = Blueprint("http_stream", __name__)

    @bp.route('/stream', methods=['GET', 'POST'])
    async def stream():
        request_id = None
        try:
            args = dict(request.args)
            json_body = None
            if request.method == 'POST':
                json_body = await request.get_json(silent=True) or {}
                for key in BODY_ARGS:
                    if key in json_body and key not in args:
                        args[key] = json_body[key]

            req_data = RequestData(request.method, request.url, args, dict(request.headers), json_body=json_body)
            context = RequestContext.from_request_data(req_data)
            if context.server_sound and not deps.server_sound:
                context = _downgrade_server_sound(context)
            elif not deps.voice_enabled and not context.text_only and not context.server_sound:
                context = _downgrade_audio_client(context)
            query = req_data.args.get('query', '')
            request_id = req_data.args.get('request_id')

            def close_text_stream(reason: str):
                # Any /stream path that returns without handing off to a handler must close
                # the companion /stream_text channel itself — only the handler's finally
                # normally emits the end-of-stream sentinel. See TextStreamRegistry.close().
                if request_id and deps.text_stream is not None:
                    trace(f"Closing text stream {request_id}: {reason}")
                    deps.text_stream.close(request_id)

            if DebugCommands.is_command(query):
                if request.method != 'POST':
                    # A GET is what an <img> on any web page can send; no Outpost sends `//` commands.
                    info(f"Refused debug command over {request.method} from {request.remote_addr} — POST it.")
                    close_text_stream("debug command over GET refused")
                    return {"error": "debug commands must be POSTed"}, 405
                # //wipe zips the database: keep it off the event loop.
                reply = await asyncio.to_thread(DebugCommands.for_assistant(deps.assistant).handle, query)
                close_text_stream("debug command handled")
                return reply, 200

            rejection = None
            if deps.microphone_routes is not None:
                rejection = deps.microphone_routes.accept_transcription(req_data, context)
            else:
                trace("No microphone routes (voice interface disabled) — dispatching without a cross-mic vote.")
            if query:
                # Every text the server hears, before the vote's verdict drops it.
                observers().on_request_received(context, mic=args.get('mic', ''),
                                                confidence=_confidence(args), rejected=rejection)
            if rejection is not None:
                close_text_stream(rejection)
                return "", 204

            if not query:
                close_text_stream("empty query")
                return "", 204

            # Tell the client (iOS) who's speaking as soon as we know, over the /stream_text channel.
            if context.user and request_id and deps.text_stream is not None:
                info(f"Announcing identified user '{context.user}' to text stream {request_id}")
                deps.text_stream.publish(request_id, {"type": "user", "name": context.user})
            else:
                trace("No identified user to announce to the text stream.")

            try:
                # Dispatch may wait up to the lock/pool timeouts — keep it off the event loop.
                events = await asyncio.to_thread(deps.assistant.stream, context, request_data=req_data)
            except AssistantBusy as busy:
                close_text_stream(str(busy))
                return {"error": str(busy)}, 503

            response = Response(WireFormat.render(events, context),
                                headers=WireFormat.response_headers(context))
            response.timeout = None  # Disable Quart's 60s RESPONSE_TIMEOUT for streaming
            return response
        except Exception as e:
            error(f"Error: {str(e)}")
            trace(traceback.format_exc())
            if request_id and deps.text_stream is not None:
                trace(f"Closing text stream {request_id}: request errored before handoff")
                deps.text_stream.close(request_id)
            return {"error": "An internal error occurred"}, 500

    return bp


def register(app, deps: HttpDeps) -> None:
    app.register_blueprint(build_blueprint(deps))
    info("HTTP stream route registered (/stream).")
