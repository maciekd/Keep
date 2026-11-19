"""The ds4 background gate: a photo-indexing image request runs only while the single-slot
server is idle, streams over a raw connection of its own, and a user request arriving
mid-image shuts that socket and takes the slot."""
import json
import threading
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from assistant.llm.AbstractLLM import LLMPreempted
from assistant.llm.Ds4LLM import Ds4LLM
from assistant.llm.Ds4Replay import ReplayStore
from assistant.llm.PrefillTracker import LogSource
from tests.unit.waiting import FAILURE_BOUND, wait_until

MESSAGES = [{"role": "system", "content": "analyst"}, {"role": "user", "content": "describe"}]
_IDLE_SECONDS = Ds4LLM.background_idle_seconds   # bound before the fixture patches it away
JPEG = b"\xff\xd8\xff\xe0 not really a jpeg"


def _sse(*contents, reasoning=None):
    """The server's stream: a prefill keepalive comment, the deltas, the usage chunk, [DONE]."""
    out = [b": prefill\n", b"\n"]
    if reasoning:
        out.append(b"data: " + json.dumps({"choices": [{"delta": {"reasoning_content": reasoning}}]}).encode() + b"\n")
    for c in contents:
        out += [b"data: " + json.dumps({"choices": [{"delta": {"content": c}}]}).encode() + b"\n", b"\n"]
    out.append(b"data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 900, "completion_tokens": 7,
                                                                 "prompt_tokens_details": {"cached_tokens": 100}}}).encode() + b"\n")
    out.append(b"data: [DONE]\n")
    return out


class FakeSock:
    def __init__(self):
        self.shut = threading.Event()

    def shutdown(self, how):
        self.shut.set()


class FakeResponse:
    """Streams `lines`; with none it blocks like a socket read until the socket is shut."""
    status = 200

    def __init__(self, sock, lines):
        self._sock, self._lines = sock, list(lines)

    def readline(self):
        if self._lines:
            return self._lines.pop(0)
        self._sock.shut.wait(FAILURE_BOUND)
        raise ConnectionResetError("socket shut")

    def read(self, n=-1):
        return b""


class FakeConn:
    instances: list["FakeConn"] = []
    lines: list[bytes] = []

    def __init__(self, host, port, timeout=None):
        self.host, self.port, self.sock, self.closed = host, port, None, False
        FakeConn.instances.append(self)

    def connect(self):
        self.sock = FakeSock()

    def request(self, method, path, body=None, headers=None):
        self.sent = (method, path, json.loads(body))

    def getresponse(self):
        # ds4-server answers `Connection: close`, and http.client then hands the socket to the
        # response and drops it from the connection — a gate holding `conn.sock` shuts nothing
        response = FakeResponse(self.sock, FakeConn.lines)
        self.sock = None
        return response

    def close(self):
        self.closed = True


def _config(values: dict):
    def get(key, default=None):
        return values.get(key, default)
    return patch('assistant.llm.Ds4LLM.config.get', side_effect=get)


@pytest.fixture(autouse=True)
def _fresh_gate():
    Ds4LLM._slot.reset()
    FakeConn.instances.clear()
    FakeConn.lines = []
    Ds4LLM._replay_store = ReplayStore(None)
    with patch('http.client.HTTPConnection', FakeConn), \
         patch.object(Ds4LLM, 'background_idle_seconds', return_value=0.0), \
         patch.object(LogSource, 'size', return_value=None):
        yield
    Ds4LLM._slot.reset()
    Ds4LLM._replay_store = None
    Ds4LLM._prefill._requests.clear()


def _llm() -> Ds4LLM:
    with _config({'ds4.base_url': 'http://192.0.2.7:8000/v1'}):
        llm = Ds4LLM()
    llm.client = MagicMock()
    return llm


def _run(target):
    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t


@pytest.mark.unit
class TestBackgroundImageRequest:
    def test_streams_over_its_own_connection_to_the_servers_chat_endpoint(self):
        FakeConn.lines = _sse("A red ", "square.", reasoning="looks square")
        with _config({}):
            reply = _llm().create_completion_with_images(MESSAGES, [JPEG], background=True)
        assert reply == "A red square."
        conn = FakeConn.instances[0]
        method, path, body = conn.sent
        assert (conn.host, conn.port, method, path) == ("192.0.2.7", 8000, "POST", "/v1/chat/completions")
        assert body["stream"] is True and "tools" not in body
        image_parts = [part for part in body["messages"][1]["content"] if part["type"] == "image_url"]
        assert len(image_parts) == 1 and image_parts[0]["image_url"]["url"].startswith("data:image/")
        assert conn.closed
        assert Ds4LLM._slot.active is False and Ds4LLM._slot.sock is None
        assert not _llm().client.chat.completions.create.called, "the SDK client is not involved"

    def test_a_chat_turn_arriving_mid_image_shuts_the_socket_and_preempts_the_caller(self):
        """Once the server answered: a photo used to run its 60 s out ahead of the turn (2026-10-22)."""
        FakeConn.lines = []          # the read blocks like a prefill in progress
        outcome = {}

        def bg():
            try:
                _llm().create_completion_with_images(MESSAGES, [JPEG], background=True)
            except LLMPreempted:
                outcome["preempted"] = True
        # One config patch around both threads: `patch` swaps the shared config's `get` and puts
        # back what it found, so two patches of it unwound in the other order (the preempted
        # thread leaves its block first) left a mock behind for the rest of the worker — the
        # owner name then read "user" in whatever chat test ran next.
        with _config({}):
            t = _run(bg)
            assert wait_until(lambda: Ds4LLM._slot.sock is not None)
            sock = Ds4LLM._slot.sock
            llm = _llm()
            llm.client.chat.completions.create.return_value = iter([
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hi", tool_calls=None), finish_reason=None)], usage=None)])
            assert list(llm.create_streaming_completion([{"role": "user", "content": "hello"}])) == ["hi"]
            assert sock.shut.is_set()
            t.join(FAILURE_BOUND)
        assert outcome == {"preempted": True}
        assert Ds4LLM._slot.interactive_active == 0, "the chat turn released the gate"

    def test_an_interactive_image_request_evicts_a_background_one_too(self):
        Ds4LLM._slot.active = True
        conn = FakeConn("h", 1)
        conn.connect()
        Ds4LLM._slot.sock = conn.sock
        llm = _llm()
        llm.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content="a cat"))])
        _run(lambda: (conn.sock.shut.wait(FAILURE_BOUND), Ds4LLM._slot.release()))
        with _config({}):
            assert llm.create_completion_with_images(MESSAGES, [JPEG]) == "a cat"
        assert conn.sock.shut.is_set()
        assert llm.client.chat.completions.create.called, "the interactive request is the plain SDK call"

    def test_waits_while_a_chat_turn_is_in_flight(self):
        Ds4LLM._slot.interactive_begin()
        acquired = threading.Event()
        _run(lambda: (Ds4LLM._slot.acquire(0.0), acquired.set()))
        assert not acquired.wait(0.03)
        Ds4LLM._slot.interactive_end()
        assert acquired.wait(FAILURE_BOUND)

    def test_a_real_failure_is_not_reported_as_preemption(self):
        FakeConn.lines = [b"data: not json\n"]
        with _config({}), pytest.raises(ValueError):
            _llm().create_completion_with_images(MESSAGES, [JPEG], background=True)
        assert Ds4LLM._slot.active is False

    def test_a_refused_request_is_an_error_not_an_empty_reply(self):
        class Refused(FakeResponse):
            status = 400
        with patch.object(FakeConn, 'getresponse', lambda self: Refused(self.sock, [])), \
             _config({}), pytest.raises(RuntimeError, match="400"):
            _llm().create_completion_with_images(MESSAGES, [JPEG], background=True)

    def test_idle_seconds_is_the_ds4_section_key(self):
        # Contract with config.yaml: `ds4.background_idle_seconds`, like Ollama's key under its section.
        with _config({'ds4.background_idle_seconds': 7}):
            assert _IDLE_SECONDS() == 7.0

    def test_the_wait_is_per_thread_and_readable_by_the_caller(self):
        Ds4LLM._slot.acquire(0.0)
        now, queued, waits = [1000.0], threading.Event(), {}

        def second():
            Ds4LLM._slot.acquire(0.0)
            waits["s"] = _llm().background_wait_seconds()
        # The slot traces once, holding its lock, just before the second request first waits: the
        # release cannot come ahead of that wait, and the wait is measured on the test's clock.
        with patch('assistant.llm.BackgroundSlot.trace', side_effect=lambda _m: queued.set()), \
             patch('assistant.llm.BackgroundSlot.time', SimpleNamespace(monotonic=lambda: now[0])):
            t = _run(second)
            assert queued.wait(FAILURE_BOUND)
            now[0] += 0.5
            Ds4LLM._slot.release()
            t.join(FAILURE_BOUND)
        assert waits["s"] == 0.5
        assert _llm().background_wait_seconds() == 0
