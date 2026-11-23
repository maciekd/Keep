"""Tracks how far a local model has got with a prompt's prefill, for the "Reading…" bar of the chat UIs.

One `PrefillTracker` per backend class holds the state of the requests in flight; the backend
brackets each request with `begin` / `first_token` / `end` and feeds the tracker the prefill
numbers from one of two sources. A server that streams them (llama.cpp's `return_progress`
extension, which oMLX speaks too: `prompt_progress` chunks with the prompt's total, cached and
processed tokens and the elapsed time) hands each chunk to `report_progress`, and needs no log.
A server that says nothing on the wire until the first token (Ollama, ds4-server) leaves its
log file as the only view into a long prefill: one line at the start of the request says how
much of the prompt the KV cache covered and how many tokens are left to prefill, and some
engines log incremental progress after that; the backend plugs in a `LogFormat` that reads its
own log lines and points the tracker at the log (`use_log`, a local file or one read over ssh
when the server is on another machine). Whichever the source, the request's state is the same
— its workload (cached, remaining), its latest progress (processed, total), its prefill time —
and so is everything below.

The estimate: the workload is the server's own once a start line or a first chunk says it, and
until then the tracker's forecast when the backend handed `begin` the prompt's text — the longest
prefix it shares with a prompt the server prefilled lately (`learn`) is taken as cached, the rest
as to prefill, at the last prompt's tokens per character. oMLX needs it: its first chunk comes
only once a whole prefill step (2048 tokens, half a minute on a small Mac) is done, and a turn
that adds less than a step gets none before its first token. Without either the estimate is 0
(nothing to draw); it is 0 too for a prefill the rolling rate puts under `MIN_BAR_SECONDS` (a bar
there would only blink), 100 at once on a full cache hit, the engine's own progress — carried on
at the measured rate between two reports — when it gives any, and otherwise the time since the
workload line was read (since the request became the one the server is on, for a forecast)
against a rolling prefill rate (`observe_rate` refines it after each reply, from
the prefill time the log reports — `prefill_seconds`) — capped at 99 until the first token
confirms the prefill is over, or the log does (`LogFormat.done`: a non-streamed request, the
memory curation say, sends no token until its whole reply is written). The clock and the rate
both start at the workload line, never at the send: the wait behind the request ahead is not
prefill, and counted as such it ran the bar up to 99 before the engine's first progress line
pulled it back, and taught the rate single digits. The rate shown beside the bar (`rate`) is
measured from the progress lines when the engine logs them, else that rolling estimate. Each
server runs one request at a time, but ours overlap: a pre-heat fired at the end of one turn is
often still prefilling when the next user request goes out, and the first-run pipeline or the
autonomous check may hold the server while the user asks.
So the tracker keeps every request in flight, in the order they were sent — the order the server
takes them in — and the bar follows the first one that has no token yet: that is the request the
server is prefilling, or will prefill next, and whatever it is, it is what the user's reply is
waiting for (a request queued behind another's generation reads 0 until its own workload line
appears). A request's view of the log opens where the log ended when it was sent, so the lines
the request ahead logged after that land in it: each prefill that ends inside a later request's
window is counted on that request (`preceding_done`), and `LogFormat.skip` drops that many
finished sections before its lines are read — else the bar showed the earlier request's final
progress line as this one's. `PrefillTracker.current_progress()` is what the interfaces poll: the backend with a
user request in flight, else any backend with a request. `currently_queued()` is the other
thing a reply can be stuck on: the server generating the reply of the request ahead — the memory
curation's for minutes, a photo description's until it stands down — so the UIs say "queued"
instead of showing a bar at 99.
"""
import os
import re
import shlex
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from common.logging import info, trace

LOCAL_HOSTS = {None, '', 'localhost', '127.0.0.1', '::1'}


def is_local_host(hostname: str | None) -> bool:
    """Whether a base_url's hostname is this machine — the log is then a file, not an ssh read."""
    return hostname in LOCAL_HOSTS


class LogSource:
    """A server's log file: on this machine, or on the serving host read over ssh.

    The ssh connections are multiplexed over a persistent control socket, so after the first call
    each read costs tens of milliseconds. Any ssh failure disables further attempts for a cooldown
    instead of stalling the request path on repeated connect timeouts.
    """
    _SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=2",
                 "-o", "ControlMaster=auto", "-o", "ControlPath=/tmp/floxy-log-ssh-%C",
                 "-o", "ControlPersist=600"]
    _FAILURE_COOLDOWN_S = 60.0
    _MAX_READ_BYTES = 1_000_000

    def __init__(self, label: str, path: str, ssh_host: str | None = None):
        self.label = label
        self.path = path
        self.ssh_host = ssh_host
        self._disabled_until = 0.0

    def __eq__(self, other):
        return isinstance(other, LogSource) and (self.path, self.ssh_host) == (other.path, other.ssh_host)

    def describe(self) -> str:
        return f"{self.path} via ssh {self.ssh_host}" if self.ssh_host else self.path

    def size(self) -> int | None:
        """Log size in bytes, 0 if the file doesn't exist, None if it is unreachable."""
        if self.ssh_host is None:
            try:
                path = os.path.expanduser(self.path)
                return os.path.getsize(path) if os.path.exists(path) else 0
            except OSError:
                return None
        out = self._run(f"wc -c 2>/dev/null < {self._remote_path()} || echo MISSING")
        if out is None:
            return None
        out = out.strip()
        if out == "MISSING":
            trace(f"{self.label} log {self.ssh_host}:{self.path} does not exist (yet).")
            return 0
        try:
            return int(out)
        except ValueError:
            self._disable(f"unexpected wc output {out!r}")
            return None

    def read_from(self, offset: int) -> str | None:
        """Log content from byte offset to EOF (capped), None if empty or unreachable."""
        if self.ssh_host is None:
            try:
                path = os.path.expanduser(self.path)
                if not os.path.exists(path) or os.path.getsize(path) <= offset:
                    return None
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    f.seek(offset)
                    return f.read()
            except OSError:
                return None
        out = self._run(f"(tail -c +{offset + 1} {self._remote_path()} 2>/dev/null || true)"
                        f" | head -c {self._MAX_READ_BYTES}")
        return out or None

    def _remote_path(self) -> str:
        """The path quoted for the remote shell, a leading `~/` left to that shell's $HOME."""
        if self.path.startswith("~/"):
            return '"$HOME"/' + shlex.quote(self.path[2:])
        return shlex.quote(self.path)

    def _run(self, remote_cmd: str) -> str | None:
        if time.monotonic() < self._disabled_until:
            return None
        try:
            result = subprocess.run(["ssh", *self._SSH_OPTS, self.ssh_host, remote_cmd],
                                    capture_output=True, text=True, errors='replace', timeout=5)
            if result.returncode != 0:
                self._disable(f"exit {result.returncode}: {result.stderr.strip()}")
                return None
            return result.stdout
        except (subprocess.SubprocessError, OSError) as e:
            self._disable(str(e))
            return None

    def _disable(self, reason: str):
        self._disabled_until = time.monotonic() + self._FAILURE_COOLDOWN_S
        info(f"{self.label} log via ssh {self.ssh_host} failed ({reason}) — "
             f"disabling prefill diagnostics for {self._FAILURE_COOLDOWN_S:.0f}s.")


class LogFormat(ABC):
    """How one server's log tells the two things the tracker needs, from the text logged since a
    request started: the request's workload and, for engines that log it, its progress."""

    @abstractmethod
    def work(self, text: str) -> tuple[int, int] | None:
        """(cached, remaining) tokens of the request — its first workload line in `text` — or
        None while the server has not logged one."""

    @abstractmethod
    def progress(self, text: str) -> tuple[int, int] | None:
        """(processed, total) of the latest progress line in `text`, None when there is none."""

    def done(self, text: str) -> int | None:
        """The index just past the line saying the prefill of the request `text` opens with is
        over, None while there is none — the one view into a non-streamed request, which says
        nothing on the wire until its whole reply is done. None for an engine whose log has no
        such line: its bar then waits for the first token as before."""
        return None

    def duration(self, text: str) -> float | None:
        """The prefill seconds the done line reports, None when the engine's log does not say."""
        return None

    def skip(self, text: str, requests: int) -> str:
        """`text` without the sections of the `requests` earlier requests whose prefills ended
        inside it, '' when fewer have ended there yet. An engine whose log has no done line
        cannot tell the sections apart and hands the text back whole."""
        return text


@dataclass
class _Request:
    """One request in flight, with what its log lines have said so far."""
    token: int
    interactive: bool          # a user request, not a pre-heat / background call
    start_ts: float
    anchor: int | None         # log EOF when the request was sent; None = log unreachable for it
    first_token: bool = False
    first_token_ts: float = 0.0
    done: bool = False                        # the log said its prefill is over (a non-streamed request)
    duration: float | None = None             # the prefill seconds the done line, or the latest progress chunk, reported
    work: tuple[int, int] | None = None       # (cached, remaining) once the workload line is logged
    work_ts: float = 0.0                      # when that line was read: the start of the clock
    prompt: str | None = None                 # the prompt's text, when the backend gave it: what `learn` keeps
    forecast: tuple[int, int] | None = None   # (cached, remaining) the tracker expects until `work` is known
    head_ts: float = 0.0                      # when it became the request the server is on: the forecast's clock
    on_work: "Callable[[int, int], None] | None" = None   # told (cached, remaining) once, when that line is read
    progress: tuple[int, int] | None = None   # (processed, total) of the latest progress line
    progress_ts: float = 0.0                  # when that line was read
    read_ts: float = 0.0
    preceding_done: int = 0                   # earlier requests whose prefill ended inside this one's window

    @property
    def generating(self) -> bool:
        """Past its prefill: the server is writing its reply."""
        return self.first_token or self.done


class PrefillTracker:
    """The prefill state of one local backend's requests in flight (see the module docstring)."""
    POLL_INTERVAL_S = 2.0   # log reads between polls: over ssh each is a round trip, and new lines are rare
    MIN_BAR_SECONDS = 2.0   # a prefill estimated shorter is not drawn: the bar would only blink
    MIN_RATE_TOKENS = 1000  # a prefill shorter teaches the rate nothing (observe_rate)
    HELD_PROMPTS = 8        # the prompts a forecast compares against: the server's cache holds more than one
    TOKENS_PER_CHAR = 0.28  # the forecast's ratio until the server has counted a prompt
    _trackers: list["PrefillTracker"] = []

    def __init__(self, label: str, log_format: LogFormat | None, seed_tps: float):
        self.label = label
        self.format = log_format   # None for a backend whose server streams its progress (`report_progress`)
        self.tps = seed_tps  # rolling estimate of the prefill rate, tokens/s; the seed stands until a reply refines it
        self._observed = False   # the seed is a guess at a machine unknown: the first measured rate replaces it
        self._held: list[tuple[str, int]] = []   # (text, tokens) of the prompts the server prefilled lately, newest last
        self.log: LogSource | None = None
        self._lock = threading.Lock()
        self._requests: list[_Request] = []   # in the order sent — the order the server serves them
        self._last_token = 0
        PrefillTracker._trackers.append(self)

    def use_log(self, path: str, ssh_host: str | None = None):
        """Point the tracker at the server's log — a file here, or `path` on `ssh_host`."""
        source = LogSource(self.label, path, ssh_host)
        if self.log == source:
            return
        self.log = source
        info(f"{self.label} prefill diagnostics: reading {source.describe()}.")

    def begin(self, interactive: bool = True, on_work: Callable[[int, int], None] | None = None,
              prompt: str | None = None) -> int:
        """Register a request about to be sent and hand back its token. `on_work` is told the
        request's (cached, remaining) tokens once, from whichever poll first reads its workload
        line — the backend's cache-break report, made while the bar is still drawn rather than
        after the reply. It runs on the polling thread (a chat UI's tick, the dashboard's
        frame, the backend's forced read) and never on a request whose log is unreachable.
        `prompt`, the request's text, gives the bar a forecast until the server says the workload."""
        # The anchor is fetched before taking the lock: over ssh it can take a moment and the
        # progress polls must not block on it. The scan from it stays non-destructive, so every
        # poll and the done-time read see the same lines.
        anchor = self.log.size() if self.log is not None else None
        if anchor is None and self.format is not None:
            trace(f"{self.label} prefill: log unreachable — prefill diagnostics disabled for this request.")
        forecast = self._forecast(prompt) if prompt else None
        with self._lock:
            self._last_token += 1
            token = self._last_token
            self._requests.append(_Request(token, interactive, time.monotonic(), anchor, on_work=on_work,
                                           prompt=prompt, forecast=forecast))
            queued = len(self._requests) - 1
            self._stamp_head()
        trace(f"{self.label} prefill: started (#{token}, {'user request' if interactive else 'background'}"
              f"{f', {queued} ahead of it' if queued else ''}"
              f"{f'; forecast {forecast[0]} cached, {forecast[1]} to prefill' if forecast else ''}).")
        return token

    def _forecast(self, prompt: str) -> tuple[int, int]:
        """(cached, remaining) as the server will likely find them: the longest prefix the prompt
        shares with one the server prefilled lately is cached, at the newest one's tokens per char."""
        with self._lock:
            held = list(self._held)
        per_char = held[-1][1] / len(held[-1][0]) if held else self.TOKENS_PER_CHAR
        shared = max((common_prefix_length(prompt, text) for text, _ in held), default=0)
        return round(shared * per_char), round((len(prompt) - shared) * per_char)

    def learn(self, token: int | None, prompt_tokens: int):
        """The server prefilled the request's whole prompt, `prompt_tokens` long (its usage says
        so): a later prompt that starts the same is forecast to find that part cached."""
        with self._lock:
            request = self._find(token)
            if request is None or not request.prompt or prompt_tokens <= 0:
                return
            others = [held for held in self._held if held[0] != request.prompt]
            self._held = others[-(self.HELD_PROMPTS - 1):] + [(request.prompt, prompt_tokens)]
        trace(f"{self.label} prefill: #{token}'s {prompt_tokens}-token prompt is held for the next forecasts")

    def _stamp_head(self):
        """Under the lock: the request the server is now on starts its forecast's clock."""
        head = next((r for r in self._requests if not r.generating), None)
        if head is not None and not head.head_ts:
            head.head_ts = time.monotonic()

    def first_token(self, token: int | None):
        """The request's first generated token means its prefill is over."""
        with self._lock:
            request = self._find(token)
            if request is not None and not request.first_token:
                request.first_token, request.first_token_ts = True, time.monotonic()
                if not request.done:
                    self._prefill_over(request)
                self._stamp_head()
                trace(f"{self.label} prefill: first token received (prefill #{token} complete).")

    def report_progress(self, token: int | None, total: int, cache: int, processed: int,
                        time_ms: float | None = None):
        """A progress chunk the server streamed for the request — llama.cpp's `prompt_progress`
        (oMLX speaks it too): `total` prompt tokens, `cache` of them already in the KV cache,
        `processed` so far (the cached ones included), `time_ms` since the prefill began. The
        first chunk is the request's workload line — (cache, total - cache), the clock starts,
        `on_work` is told —, every chunk its latest progress, counted on the uncached part as a
        log's chunk lines are, and the chunk's elapsed time stands as the prefill's duration
        until the next one, so `prefill_seconds` reads the server's own clock at the end."""
        with self._lock:
            request = self._find(token)
            if request is None:
                trace(f"{self.label} prefill: a progress chunk for a request no longer in flight (#{token}) — ignored")
                return
            now = time.monotonic()
            on_work = None
            remaining = max(total - cache, 0)
            if request.work is None:
                request.work, request.work_ts = (cache, remaining), now
                on_work, request.on_work = request.on_work, None
            request.progress, request.progress_ts = (max(processed - cache, 0), remaining), now
            if time_ms is not None and time_ms > 0:
                request.duration = time_ms / 1000.0
            work = request.work
        trace(f"{self.label} prefill: the server says #{token} is at {processed}/{total} tokens"
              f" ({cache} cached{f', {time_ms / 1000.0:.1f} s' if time_ms else ''})")
        if on_work is not None:
            try:
                on_work(*work)
            except Exception as e:
                trace(f"{self.label} prefill: #{token}'s workload callback failed ({e})")

    def _prefill_over(self, request: _Request, done_offset: int | None = None):
        """Under the lock, once per request, when its prefill is seen to be over: the requests
        sent while it was still prefilling have its done line inside their own windows and must
        skip it to reach their lines. `done_offset`, the byte the done line ends at when the
        log showed it, tells which of them; a first token gives no offset, and the line is
        taken to precede every request in flight behind it — it was logged before the token
        was sent, so a request squeezed in between is the one in a thousand this gets wrong."""
        later = self._requests[self._requests.index(request) + 1:]
        for r in later:
            if done_offset is None or r.anchor is None or r.anchor < done_offset:
                r.preceding_done += 1

    def end(self, token: int | None):
        """The request is over; the next one in line, if any, is what the bar follows. A request
        that sent no token and whose done line no poll saw (a non-streamed one with nothing
        waiting behind it) is read once more now, so the requests behind it learn to skip it."""
        with self._lock:
            request = self._find(token)
        if request is None:
            return
        if not request.generating:
            self._poll(request, force=True)
        with self._lock:
            if request in self._requests:
                self._requests.remove(request)
            self._stamp_head()

    def prefill_seconds(self, token: int | None) -> float | None:
        """How long the request's prefill took, for `observe_rate`: what the log's done line says
        (read now), else the time from its workload line to its first token; None when neither
        is known. Never the time since the request was sent — that counts the wait behind the
        request ahead, and a rate from it dragged the estimate down to single digits."""
        with self._lock:
            request = self._find(token)
        if request is None:
            return None
        self._poll(request, force=True)
        with self._lock:
            if request.duration:
                return request.duration
            if request.first_token_ts and request.work_ts:
                return request.first_token_ts - request.work_ts
        trace(f"{self.label} prefill: #{token}'s duration unknown — no done line, no workload line; the rate estimate stands.")
        return None

    def observe_rate(self, tokens: int, seconds: float):
        """Refine the prefill-rate estimate (EMA) from a finished request. A small prefill is
        skipped: the server's fixed start of a request (~0.4 s on ds4) is most of its time, so
        135 tokens read as 225 t/s where 3.5k ran at 990 — and with the cache hitting, the
        small ones are most requests and held the bar at half when the reply began."""
        if seconds <= 0:
            return
        if tokens < self.MIN_RATE_TOKENS:
            trace(f"{self.label} prefill rate estimate kept at {self.tps:.1f} t/s: {tokens} tokens are too few to measure it")
            return
        observed = tokens / seconds
        self.tps = 0.5 * self.tps + 0.5 * observed if self._observed else observed
        self._observed = True
        trace(f"{self.label} prefill rate estimate updated: {self.tps:.1f} t/s (observed {observed:.1f} t/s)")

    def work(self, token: int | None, force: bool = False) -> tuple[int, int] | None:
        """The request's (cached, remaining) tokens once its workload line is logged, else None.
        `force` reads the log now instead of waiting for the next poll slot — the done-time read."""
        with self._lock:
            request = self._find(token)
        return self._poll(request, force)[0] if request is not None else None

    def active(self) -> bool:
        return bool(self._requests)

    def interactive(self) -> bool:
        """Whether a user request is among the requests in flight."""
        return any(r.interactive for r in self._requests)

    def _head(self) -> _Request | None:
        """The request the server is on: the first in line that is not past its prefill."""
        with self._lock:
            return next((r for r in self._requests if not r.generating), None)

    def progress(self) -> float | None:
        """Estimated prefill completion (0-100) of the request the server is on, None when no
        request is in flight. 100 when every request in flight is past its prefill."""
        if not self._requests:
            return None
        head = self._head()
        if head is None:
            return 100.0
        work, progress = self._poll(head)
        if head.done:
            return 100.0   # the log said so: the server is generating the reply
        now = time.monotonic()
        start = head.work_ts
        if work is None:
            if head.forecast is None:
                return 0.0   # the server has not said the workload yet — nothing to draw
            work, start = head.forecast, head.head_ts or now
        _cached, remaining = work
        if remaining <= 0:
            return 100.0   # a full cache hit: no prefill work
        eta = remaining / self.tps if self.tps > 0 else 0.0
        if 0 < eta < self.MIN_BAR_SECONDS:
            return 0.0   # too short to be worth a bar — plain waiting
        if progress and progress[1] > 0:
            processed, total = progress
            processed += (now - head.progress_ts) * self._rate_of(head, progress)   # since the last report
            return min(99.0, processed / total * 100.0)
        if eta <= 0:
            return 99.0
        return min(99.0, (now - start) / eta * 100.0)

    def rate(self) -> float:
        """The prefill rate in tokens/s: measured from the engine's progress when the request
        the server is on has reported any — over the server's own clock when its chunks carry
        one, else ours —, else the rolling estimate the bar runs on."""
        head = self._head()
        if head is None:
            return self.tps
        _work, progress = self._poll(head)
        return self._rate_of(head, progress)

    def _rate_of(self, request: _Request, progress: tuple[int, int] | None) -> float:
        if progress and progress[0] > 0:
            if request.duration:
                return progress[0] / request.duration
            if request.progress_ts > request.work_ts:
                return progress[0] / (request.progress_ts - request.work_ts)
        return self.tps

    def _find(self, token: int | None) -> _Request | None:
        return next((r for r in self._requests if r.token == token), None)

    def _poll(self, request: _Request, force: bool = False) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
        """One throttled log read for both answers. The read runs OUTSIDE the lock — over ssh it
        can take seconds, and `first_token` and every TUI frame would stall behind it."""
        with self._lock:
            now = time.monotonic()
            due = force or now - request.read_ts >= self.POLL_INTERVAL_S
            if due:
                request.read_ts = now
            anchor, work, progress, skip = request.anchor, request.work, request.progress, request.preceding_done
        if not due or anchor is None:
            return work, progress
        raw = self.log.read_from(anchor) if self.log is not None else None
        if not raw:
            return work, progress
        text = self.format.skip(raw, skip) if skip else raw
        try:
            new_work = work or self.format.work(text)
            new_progress = self.format.progress(text) or progress
            done_at = None if request.done else self.format.done(text)
            duration = self.format.duration(text) if done_at is not None else None
        except Exception as e:
            trace(f"{self.label} prefill: log lines not understood ({e})")
            return work, progress
        on_work = None
        with self._lock:
            if request.work is None and new_work is not None:
                request.work, request.work_ts = new_work, now
                on_work, request.on_work = request.on_work, None
            if new_progress is not None and new_progress != request.progress:
                request.progress, request.progress_ts = new_progress, now
            if done_at is not None and not request.done:
                request.done, request.duration = True, duration
                if not request.first_token:
                    done_offset = anchor + len(raw[:len(raw) - len(text) + done_at].encode())
                    self._prefill_over(request, done_offset)
                self._stamp_head()
                trace(f"{self.label} prefill: the log says #{request.token}'s prompt is done — the server is "
                      f"generating its reply{' (non-streamed: no token tells us)' if not request.first_token else ''}.")
            work, progress = request.work, request.progress
        if on_work is not None:
            try:
                on_work(*work)
            except Exception as e:
                trace(f"{self.label} prefill: #{request.token}'s workload callback failed ({e})")
        return work, progress

    def queued(self) -> bool:
        """Whether the newest user request waits for the server to finish generating another
        request's reply — a non-streamed call such as the memory curation, a background photo
        description standing down. False while the server is prefilling (the bar's business)
        or on the user request itself."""
        with self._lock:
            mine = next((r for r in reversed(self._requests) if r.interactive), None)
            ahead = self._requests[0] if mine is not None and self._requests[0] is not mine else None
        if ahead is None:
            return False
        self._poll(ahead)
        return ahead.generating

    @classmethod
    def _busy(cls) -> "PrefillTracker | None":
        """The backend a user is waiting for — one with a user request in flight, else any
        with a request — or None when the local backends are idle."""
        active = [t for t in cls._trackers if t.active()]
        if not active:
            return None
        return next((t for t in active if t.interactive()), active[0])

    @classmethod
    def current_progress(cls) -> float | None:
        """The progress of what a user is waiting for, None when the local backends are idle."""
        tracker = cls._busy()
        return tracker.progress() if tracker is not None else None

    @classmethod
    def current_rate(cls) -> float | None:
        """The prefill rate (tokens/s) of what a user is waiting for, None when idle."""
        tracker = cls._busy()
        return tracker.rate() if tracker is not None else None

    @classmethod
    def currently_queued(cls) -> bool:
        """Whether the user's reply waits for the server to finish another request's reply (see `queued`)."""
        tracker = cls._busy()
        return tracker.queued() if tracker is not None else False


def common_prefix_length(a: str, b: str) -> int:
    """How many leading characters `a` and `b` share — a binary search over slice compares, which
    run in C, since prompts are tens of thousands of characters."""
    low, high = 0, min(len(a), len(b))
    while low < high:
        mid = (low + high + 1) // 2
        if a[:mid] == b[:mid]:
            low = mid
        else:
            high = mid - 1
    return low


def parse_last(pattern: str, text: str) -> tuple[int, int] | None:
    """The last match of a two-group integer pattern in `text`, as a pair — the shape of every
    progress line — or None."""
    matches = re.findall(pattern, text)
    if not matches:
        return None
    a, b = matches[-1]
    return int(a), int(b)
