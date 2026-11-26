"""The HTTP interface: owns the Quart app, `/stream`, `/health`, the hypercorn server,
SIGINT/SIGTERM handling and the gc-freeze timer — everything that is "the server
process" rather than the assistant. Other interfaces register their blueprints on
`self.app` after `start()`; `serve()` is the launcher's final blocking call."""
import asyncio
import gc
import signal
import threading

from assistant.commands.formats.CommandFormats import CommandFormats
from common import Platform, Version
from common.ConfigManager import config
from common.logging import error, info, note, trace
from interfaces.http import CrossSiteGuard
from interfaces.http.routes import HttpDeps, register as register_stream_route
from interfaces.Interface import Interface, bind_host, run_shutdown_hooks

# Seconds after serve() starts before freezing the startup heap: late enough that the
# boot-time model loads (Whisper warm-up, SpeechBrain, Supertonic — all resident within
# ~15s) are included, yet still inside the mics' 60s CAPTURE_DROP_STARTUP_GRACE so the
# freeze's own final collect can't log a spurious AUDIO DROP.
GC_FREEZE_DELAY = 30.0
# How long hypercorn has to return once the shutdown is asked for. Its `wait_closed` waits
# for every open connection (Python 3.12.1+), so one response that never ends would keep
# serve() — and with it every stop step of the launcher — waiting until start.sh's SIGKILL.
SHUTDOWN_GRACE_SECONDS = 3.0


def freeze_startup_heap():
    """Move the boot-time heap into GC's permanent generation (capture-drop fix).

    A gen2 collection walks every tracked object with the GIL held — ~125ms on this
    model-heavy heap (torch, MLX, spacy) — which starves the PortAudio capture callbacks
    and drops live mic audio (the localoutpost AUDIO DROP errors; confirmed by forcing
    gc.collect() with live mics: every pass produced a matching simultaneous drop).
    The startup heap is effectively immortal, so freezing it exempts those objects from
    every future gen2 scan, pulling the pause back under the ~100ms drop threshold.
    Collect first so pending cyclic garbage isn't frozen along with it (= leaked)."""
    collected = gc.collect()
    gc.freeze()
    info(f"GC freeze: collected {collected} objects, froze {gc.get_freeze_count()} startup "
         f"objects — future gen2 GC passes skip them (audio capture-drop fix).")


class HttpInterface(Interface):
    """Owns the Quart app, hypercorn server, signal handling and the /stream and /health routes."""

    name = "http"

    def __init__(self, port: int):
        self.port = port
        self.app = None
        self.deps: HttpDeps | None = None
        self._shutdown_event: asyncio.Event | None = None
        self._hooks_told = False

    def start(self, assistant, app=None) -> None:
        from quart import Quart
        self.app = Quart("floxy_ai_server")
        CrossSiteGuard.install(self.app)
        self.deps = HttpDeps(assistant=assistant, voice_enabled=False, server_sound=Platform.is_mac())
        register_stream_route(self.app, self.deps)

        @self.app.route('/health')
        async def health():
            # command_format: the syntax this server reads replies in — the prompt test suite
            # judges a reply's calls by it.
            return {"status": "ok", "version": Version.detail(), "command_format": CommandFormats.active().name}, 200

        @self.app.errorhandler(Exception)
        async def unhandled(e):
            from werkzeug.exceptions import HTTPException
            if isinstance(e, HTTPException):
                return e
            error(f"HTTP: a route failed with {type(e).__name__}: {e}", e)
            return {"error": "internal server error"}, 500

        info(f"HTTP interface created on port {self.port} (/stream, /health).")

    def attach_voice(self, text_stream, microphone_routes) -> None:
        """Wire the voice interface's /stream hooks (iOS overlay channel, cross-mic vote)."""
        self.deps.text_stream = text_stream
        self.deps.microphone_routes = microphone_routes
        self.deps.voice_enabled = True
        info("HTTP /stream now uses the voice interface's text stream and microphone gate.")

    async def serve(self) -> None:
        """Run hypercorn until SIGINT/SIGTERM (or `request_shutdown()`)."""
        from hypercorn.asyncio import serve
        from hypercorn.config import Config

        timer = threading.Timer(GC_FREEZE_DELAY, freeze_startup_heap)
        timer.name = "gcfreeze"
        timer.daemon = True
        timer.start()
        trace(f"GC freeze scheduled in {GC_FREEZE_DELAY:.0f}s (after model loads settle).")

        hypercorn_config = Config()
        host = bind_host(config, "The HTTP server")
        hypercorn_config.bind = [f"{host}:{self.port}"]
        hypercorn_config.debug = False
        self._shutdown_event = asyncio.Event()
        self._install_signal_handlers()

        info(f"Web server listening on {host}:{self.port}...")
        note("Ready to handle requests now.")
        trace("(various subsystems now starting in background)")
        await self._serve_until_shutdown(serve(self.app, hypercorn_config, shutdown_trigger=self._shutdown_event.wait))

    async def _serve_until_shutdown(self, serving) -> None:
        """Await *serving* (hypercorn's serve), and no longer than `SHUTDOWN_GRACE_SECONDS`
        past the shutdown request."""
        serving = asyncio.ensure_future(serving)
        asked = asyncio.ensure_future(self._shutdown_event.wait())
        try:
            await asyncio.wait({serving, asked}, return_when=asyncio.FIRST_COMPLETED)
            if not serving.done():
                await asyncio.wait({serving}, timeout=SHUTDOWN_GRACE_SECONDS)
            if serving.done():
                trace("HTTP server returned by itself")
                serving.result()
                return
            note(f"HTTP server still waiting for a connection {SHUTDOWN_GRACE_SECONDS:.0f}s after the shutdown "
                 f"was asked for — not waited for any longer")
        finally:
            asked.cancel()
            if not serving.done():
                serving.cancel()
                await asyncio.gather(serving, return_exceptions=True)

    def request_shutdown(self) -> None:
        if self._shutdown_event is not None:
            self._shutdown_event.set()
        if self.app is not None and not self._hooks_told:
            self._hooks_told = True
            run_shutdown_hooks(self.app)

    def _install_signal_handlers(self) -> None:
        # Own SIGTERM/SIGINT instead of hypercorn's default handlers (hypercorn only
        # installs its own when no shutdown_trigger is given). Both signals trigger the
        # same graceful path: serve() returns, the launcher unwinds, and atexit runs the
        # interface stops plus WebAccess._shutdown_driver — so `kill <server>` no longer
        # orphans Chrome. Any handler another component installed earlier is chained.
        loop = asyncio.get_running_loop()

        def _handle_signal(sig_name: str, previous_handler):
            note(f"Received {sig_name} — starting graceful server shutdown")
            self.request_shutdown()
            if callable(previous_handler):
                try:
                    previous_handler(getattr(signal, sig_name), None)
                except Exception as e:
                    trace(f"Chained previous {sig_name} handler raised: {e}")

        for sig_name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, sig_name)
            previous = signal.getsignal(sig)
            if previous in (signal.SIG_DFL, signal.SIG_IGN, None):
                previous = None
            else:
                trace(f"Existing {sig_name} handler found — chaining it after ours")
            try:
                loop.add_signal_handler(sig, _handle_signal, sig_name, previous)
            except NotImplementedError:
                signal.signal(sig, lambda s, f, n=sig_name, p=previous: _handle_signal(n, p))
        info("Installed SIGINT/SIGTERM handlers for clean shutdown (Chrome cleanup via atexit)")

    def stop(self) -> None:
        self.request_shutdown()
