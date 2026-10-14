"""The assistant facade — the only thing an interface calls.

`stream()` hands a request to a pooled RequestProcessor and returns the typed event
stream (assistant/AssistantEvents.py) the interface renders; the handler runs on whatever
thread iterates the stream. The pool, the global dispatch mutex and the set of in-flight
requests used to be module globals in floxy_ai_server.py.
"""
import threading
import traceback
from itertools import chain
from queue import Empty, Queue
from typing import Callable, Iterator, List, Optional

from assistant.RequestProcessor import RequestData, RequestProcessor
from assistant.AssistantEvents import AssistantEvent, ObserverHub, UserIdentified, observers
from assistant.AssistantRequest import RequestContext
from assistant.prompt.TimeOfDay import TimeOfDay
from assistant.tools.MessageFormatter import MessageFormatter
from common.logging import error, fine, info, note, trace


class AssistantBusy(Exception):
    """The request could not be dispatched: the dispatch lock or the handler pool timed out."""


class AssistantStream(Iterator[AssistantEvent]):
    """The event stream of one request. Iterating it runs the request; `killed` reports
    whether a later request superseded it (the stream then ends without Done)."""

    def __init__(self, handler: RequestProcessor, events: Iterator[AssistantEvent]):
        self._handler = handler
        self._events = events

    def __iter__(self):
        return self

    def __next__(self) -> AssistantEvent:
        return next(self._events)

    @property
    def killed(self) -> bool:
        return self._handler.is_killed

    def kill(self):
        self._handler.kill()


class Assistant:
    """Dispatches requests to a pool of warm RequestProcessors and returns their event streams.

    The pool, the global dispatch mutex and the set of in-flight requests live here so a new
    request can kill the one it supersedes; the handler runs on whatever thread iterates the stream.
    """
    RUNNING_CAPACITY = 10
    DISPATCH_LOCK_TIMEOUT = 5.0
    POOL_GET_TIMEOUT = 5.0
    REPLENISH_INTERVAL = 1.0
    REPLENISH_RETRY_DELAY = 5.0

    def __init__(self, prompt_manager, handler_factory: Callable[[], RequestProcessor], pool_size: int):
        """handler_factory builds one fresh, fully wired RequestProcessor (the launcher
        closes over the playback / voice / turn-gate factories); pool_size is how many are
        kept warm — a handler pre-builds its DialogManager and LLM clients, which is why
        they are pooled at all."""
        self.prompt_manager = prompt_manager
        self._handler_factory = handler_factory
        self._pool: "Queue[RequestProcessor]" = Queue(maxsize=pool_size)
        self._running: "Queue[RequestProcessor]" = Queue(maxsize=self.RUNNING_CAPACITY)
        # Serializes dispatch: kill-the-previous + take-from-pool must not interleave.
        self._dispatch_lock = threading.Lock()
        self._replenish_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # --- observers ------------------------------------------------------------------------

    @property
    def observers(self) -> ObserverHub:
        return observers()

    def subscribe(self, observer):
        observers().subscribe(observer)

    # --- pool -----------------------------------------------------------------------------

    @property
    def pool_size(self) -> int:
        return self._pool.maxsize

    def pooled_handlers(self) -> int:
        return self._pool.qsize()

    def replenish_pool(self):
        while self._pool.qsize() < self._pool.maxsize:
            self._pool.put(self._handler_factory())

    def drain_pool(self):
        drained = 0
        while True:
            try:
                handler = self._pool.get_nowait()
            except Empty:
                break
            try:
                handler.kill()
            except Exception as exc:
                trace(f"Failed to kill pooled handler: {exc}")
            drained += 1
        if drained:
            trace(f"Drained {drained} handler(s) from pool.")
        else:
            trace("Pool already empty, nothing to drain.")

    def start(self):
        """Start keeping the pool full in the background (the Replenish thread)."""
        if self._replenish_thread is not None:
            trace("Replenishment loop already running.")
            return
        info("Starting web request handler replenishment loop in background.")
        self._replenish_thread = threading.Thread(target=self._replenishment_loop, name="Replenish", daemon=True)
        self._replenish_thread.start()

    def stop(self):
        """Stop the Replenish thread, kill the pooled handlers and stop the prompt manager's
        section threads — everything the assistant keeps running between requests."""
        self._stop_event.set()
        if self._replenish_thread is not None:
            self._replenish_thread.join(timeout=2)
            self._replenish_thread = None
        self.drain_pool()
        self.prompt_manager.stop()
        info("Assistant stopped.")

    def _replenishment_loop(self):
        last_time_of_day = TimeOfDay.get_current()
        while not self._stop_event.is_set():
            try:
                if last_time_of_day != TimeOfDay.get_current():
                    info("Clearing queue because time of day changed (needed for whispering to work).")
                    self.drain_pool()
                    last_time_of_day = TimeOfDay.get_current()
                self.replenish_pool()
                self._stop_event.wait(self.REPLENISH_INTERVAL)
            except Exception as e:
                error(f"Replenishment loop error: {str(e)}. Will retry.")
                trace(traceback.format_exc())
                self._stop_event.wait(self.REPLENISH_RETRY_DELAY)
        trace("Replenishment loop stopped.")

    # --- in-flight requests ---------------------------------------------------------------

    def running_handlers(self) -> List[RequestProcessor]:
        """Snapshot of the in-flight handlers (the microphone routes read their turn gates)."""
        return list(self._running.queue)

    def _in_flight(self) -> bool:
        """Whether a dispatched request is still being served. A handler stays in `_running`
        after its turn until the next dispatch, so the flag it keeps for the turn is what
        tells; one killed before its turn began never cleared it, hence the second test."""
        return any(h.is_performing_request and not h.is_killed for h in self.running_handlers())

    def is_busy(self, source: str | None = None, except_ctx=None) -> bool:
        """Whether a request is being served right now, or with *source* one from that channel
        (ChannelProfile.SOURCE_*). An announcement through the speakers waits for a voice turn
        (interfaces.voice.Announcer), not for the Telegram turn whose post_message it is — nor
        for the voice turn whose post_message it is: *except_ctx* is that turn's RequestContext,
        and the handler serving it does not count, else the announcement waits on itself."""
        return any(h.is_performing_request and not h.is_killed
                   and (except_ctx is None or getattr(h, "ctx", None) is not except_ctx)
                   and (source is None or getattr(getattr(h, "ctx", None), "source", None) == source)
                   for h in self.running_handlers())

    def kill_all(self, reason: str = "superseded"):
        killed = 0
        while self._running.qsize() > 0:
            self._running.get().kill(reason)
            killed += 1
        if killed:
            info(f"Killed {killed} running request(s): {reason}.")
        else:
            trace(f"No running request to kill ({reason}).")

    def kill(self, request_id: str) -> bool:
        """Kill the in-flight request with this request_id (the iOS correlation id).
        Returns False when none matched."""
        for handler in self.running_handlers():
            if getattr(handler.dialog_manager, "request_id", None) == request_id:
                handler.kill(f"killed by request_id {request_id}")
                info(f"Killed request {request_id}.")
                return True
        trace(f"No running request with request_id {request_id!r} to kill.")
        return False

    # --- dispatch -------------------------------------------------------------------------

    def stream(self, req: RequestContext, query: Optional[str] = None, images: Optional[list] = None,
               request_data: Optional[RequestData] = None) -> AssistantStream:
        """Dispatch one request and return its event stream. Kills whatever was running:
        a new request always supersedes the previous one. Raises AssistantBusy when the
        dispatch lock or the pool cannot be obtained within the timeouts.

        `request_data` carries the raw HTTP args/headers the handler still reads (`reset`,
        `request_id`, the wake-word flags for the turn gate); in-process callers omit it."""
        if query is not None:
            req.user_input = query
        if images is not None:
            req.images = images
        if request_data is None:
            request_data = RequestData(None, None, {}, {})

        header = MessageFormatter.format_header(req.user, None, req.source, req.location)
        fine("")
        info(f"{header}: ")
        note(f"{req.user_input}")
        fine("")
        fine(f"{request_data.method} {request_data.args}")

        if not self._dispatch_lock.acquire(timeout=self.DISPATCH_LOCK_TIMEOUT):
            error("Failed to acquire request processing lock within timeout")
            raise AssistantBusy("Server busy processing another request")
        try:
            fine(f"Requests currently running: {self._running.qsize()}")
            if req.guest and self._in_flight():
                # A guest (someone in a Telegram group who is not an owner) must not be able
                # to abort the owners' turns by saying her name: their turn is dropped, not
                # queued — a deferred turn would mean a thread or a timer held per guest.
                info(f"Guest request from {req.user} refused: a request is in flight, a guest turn does not supersede it")
                raise AssistantBusy("a request is in flight; a guest turn does not supersede it")
            self.kill_all(f"superseded by new request {req.user_input!r}")
            try:
                handler = self._pool.get(timeout=self.POOL_GET_TIMEOUT)
            except Empty:
                error("Request handler pool exhausted")
                raise AssistantBusy("Server busy, no available handlers")
            self._running.put(handler)
        finally:
            self._dispatch_lock.release()

        handler.request_data = request_data
        handler.request_context = req
        events = handler.process(self.prompt_manager)
        if req.user:
            events = chain((UserIdentified(req.user),), events)
        return AssistantStream(handler, events)
