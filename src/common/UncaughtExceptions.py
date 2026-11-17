"""Sends every exception nothing caught through `error()`: into the log, and from there to
whatever listens to errors (`common.logging.set_error_sink` — the server's telemetry).

Without it an exception that ends a thread or the process is a traceback on stderr alone.
`install()` covers the main thread and every other thread, `watch_loop()` an asyncio loop's
tasks whose exception nobody awaited."""
import asyncio
import sys
import threading
import traceback

from common.logging import error, trace


def _log(what: str, exception: BaseException):
    """The traceback is the log's alone: it quotes source lines and where the install lives."""
    lines = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
    error(f"Uncaught exception {what}: {type(exception).__name__}: {exception}", exception,
          detail=f"\n{lines.rstrip()}")


def install():
    def main_thread(kind, value, tb):
        if issubclass(kind, KeyboardInterrupt):
            sys.__excepthook__(kind, value, tb)
            return
        _log("on the main thread, the process ends", value)

    def other_thread(args):
        if args.exc_type is SystemExit or args.exc_value is None:
            return
        name = args.thread.name if args.thread else "unknown"
        _log(f"ended thread {name}", args.exc_value)

    sys.excepthook = main_thread
    threading.excepthook = other_thread
    trace("Uncaught exceptions on any thread are logged as errors")


def watch_loop(loop: asyncio.AbstractEventLoop):
    def handler(_loop, context):
        exception = context.get("exception")
        message = context.get("message", "no message")
        if exception is None:
            error(f"Asyncio: {message}")
        else:
            _log(f"in an asyncio task ({message})", exception)

    loop.set_exception_handler(handler)
    trace("Uncaught exceptions of the event loop's tasks are logged as errors")
