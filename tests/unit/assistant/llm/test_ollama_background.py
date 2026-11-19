"""The Ollama background gate (the shared BackgroundSlot): photo indexing runs only while the
single-slot runner is idle, and a user request arriving mid-image shuts the image request's
socket and takes the slot."""
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch, Mock

import pytest

from assistant.llm.AbstractLLM import AbstractLLM, LLMPreempted
from assistant.llm.OllamaLLM import OllamaLLM
from assistant.llm.BackgroundSlot import BackgroundSlot
from assistant.llm.PrefillTracker import LogSource
from tests.unit.waiting import FAILURE_BOUND, wait_until

DONE = {"done": True, "prompt_eval_count": 1, "prompt_eval_duration": 1e9,
        "eval_count": 1, "eval_duration": 1e9, "total_duration": 2e9}
MESSAGES = [{"role": "system", "content": "analyst"}, {"role": "user", "content": "describe", "images": ["QUJD"]}]


def _lines(*contents):
    out = [json.dumps({"message": {"content": c}}).encode() + b"\n" for c in contents]
    out.append(json.dumps({**DONE, "message": {"content": ""}}).encode() + b"\n")
    return out


class FakeSock:
    def __init__(self):
        self.shut = threading.Event()

    def shutdown(self, how):
        self.shut.set()


class FakeResponse:
    """Streams `lines`; with none it blocks like a socket read until the socket is shut."""
    status = 200

    def __init__(self, sock: FakeSock, lines: list[bytes]):
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
        self.sock = None
        self.closed = False
        FakeConn.instances.append(self)

    def connect(self):
        self.sock = FakeSock()

    def request(self, method, path, body=None, headers=None):
        self.sent = (method, path, json.loads(body))

    def getresponse(self):
        return FakeResponse(self.sock, FakeConn.lines)

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _fresh_gate():
    def reset():
        OllamaLLM._slot.reset()
        OllamaLLM._bg_completed = 0
        OllamaLLM._last_pre_heat_messages = None
        OllamaLLM._pre_heat_in_flight = False
        OllamaLLM._pre_heat_pending_messages = None
        FakeConn.instances.clear()
        FakeConn.lines = []
    reset()
    with patch('http.client.HTTPConnection', FakeConn), \
         patch.object(OllamaLLM, 'background_idle_seconds', return_value=0.0), \
         patch.object(LogSource, 'size', return_value=None):   # the tracker gets no anchor, reads nothing
        yield
    reset()


def _llm():
    return OllamaLLM(model='gemma4:26b', base_url='http://localhost:11434')


def _run(target):
    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t


@pytest.mark.unit
class TestBackgroundCompletion:
    def test_streams_the_reply_over_its_own_connection(self):
        FakeConn.lines = _lines("A red ", "square.")
        OllamaLLM._last_payload = "the last chat payload"
        assert _llm()._background_completion(MESSAGES) == "A red square."
        conn = FakeConn.instances[0]
        assert conn.sent[1] == "/api/chat" and conn.sent[2]["stream"] is True
        assert conn.closed
        assert OllamaLLM._slot.active is False and OllamaLLM._slot.sock is None
        # an image payload must not become what the next chat turn's prefix diagnostics compare against
        assert OllamaLLM._last_payload == "the last chat payload"

    def test_interactive_arrival_shuts_the_socket_and_preempts_the_caller(self, monkeypatch):
        monkeypatch.setattr(BackgroundSlot, "PREEMPT_UNWIND_TIMEOUT_S", FAILURE_BOUND)
        FakeConn.lines = []          # the read blocks like a prefill in progress
        outcome = {}

        def bg():
            try:
                _llm()._background_completion(MESSAGES)
            except LLMPreempted:
                outcome["preempted"] = True
        t = _run(bg)
        assert wait_until(lambda: OllamaLLM._slot.sock is not None)
        sock = OllamaLLM._slot.sock
        OllamaLLM._interactive_begin()
        assert sock.shut.is_set()
        assert OllamaLLM._slot.active is False, "the interactive side waited for the background request to unwind"
        t.join(FAILURE_BOUND)
        assert outcome == {"preempted": True}
        OllamaLLM._interactive_end()

    def test_stands_down_when_preempted_before_the_request_is_sent(self):
        OllamaLLM._slot.acquire(OllamaLLM.background_idle_seconds())
        t = _run(OllamaLLM._interactive_begin)
        assert wait_until(lambda: OllamaLLM._slot.abort)
        conn = FakeConn("h", 1)
        conn.connect()
        with pytest.raises(LLMPreempted):
            OllamaLLM._slot.register_connection(conn)
        assert conn.closed
        OllamaLLM._slot.release()
        t.join(FAILURE_BOUND)
        assert not t.is_alive()

    def test_a_real_failure_is_not_reported_as_preemption(self):
        FakeConn.lines = [b"not json\n"]
        with pytest.raises(ValueError):
            _llm()._background_completion(MESSAGES)
        assert OllamaLLM._slot.active is False


@pytest.mark.unit
class TestIdleGate:
    def test_waits_out_the_idle_clock_after_an_interactive_request(self):
        with patch.object(OllamaLLM, 'background_idle_seconds', return_value=0.04):
            OllamaLLM._interactive_begin()
            OllamaLLM._interactive_end()
            t0 = time.monotonic()
            OllamaLLM._slot.acquire(OllamaLLM.background_idle_seconds())
        assert time.monotonic() - t0 >= 0.03

    def test_waits_while_an_interactive_request_is_in_flight(self):
        OllamaLLM._interactive_begin()
        acquired = threading.Event()
        _run(lambda: (OllamaLLM._slot.acquire(OllamaLLM.background_idle_seconds()), acquired.set()))
        assert not acquired.wait(0.03)
        OllamaLLM._interactive_end()
        assert acquired.wait(FAILURE_BOUND)

    def test_one_background_request_at_a_time(self):
        OllamaLLM._slot.acquire(OllamaLLM.background_idle_seconds())
        acquired = threading.Event()
        _run(lambda: (OllamaLLM._slot.acquire(OllamaLLM.background_idle_seconds()), acquired.set()))
        assert not acquired.wait(0.03)
        OllamaLLM._slot.release()
        assert acquired.wait(FAILURE_BOUND), "the second worker goes straight in — a background request does not restart the idle clock"

    def test_the_caller_can_read_how_long_its_request_queued(self):
        OllamaLLM._slot.acquire(OllamaLLM.background_idle_seconds())
        now, queued, waits = [1000.0], threading.Event(), {}
        def second():
            OllamaLLM._slot.acquire(OllamaLLM.background_idle_seconds())
            waits["s"] = _llm().background_wait_seconds()
        # The slot traces once, holding its lock, just before the second request first waits: the
        # release cannot come ahead of that wait, and the wait is measured on the test's clock.
        with patch('assistant.llm.BackgroundSlot.trace', side_effect=lambda _m: queued.set()), \
             patch('assistant.llm.BackgroundSlot.time', SimpleNamespace(monotonic=lambda: now[0])):
            t = _run(second)
            assert queued.wait(FAILURE_BOUND)
            now[0] += 0.5
            OllamaLLM._slot.release()
            t.join(FAILURE_BOUND)
        assert waits["s"] == 0.5
        assert _llm().background_wait_seconds() == 0, "the wait is per thread — this one never queued"

    def test_the_post_turn_pre_heat_evicts_a_background_request(self):
        OllamaLLM._slot.active = True
        conn = FakeConn("h", 1)
        conn.connect()
        OllamaLLM._slot.sock = conn.sock
        llm = _llm()
        with patch.object(llm, '_rendered_prompt', return_value="sys<|im_end|>\n<|im_start|>assistant\n"), \
             patch('assistant.llm.OllamaLLM.requests.post') as post:
            post.return_value.raise_for_status = Mock()
            t = _run(lambda: llm._pre_heat_raw_truncated([{"role": "user", "content": "x"}]))
            assert conn.sock.shut.wait(FAILURE_BOUND)
            OllamaLLM._slot.release()
            t.join(FAILURE_BOUND)
        assert OllamaLLM._slot.interactive_last_end > 0, "a pre-heat restarts the idle clock"


@pytest.mark.unit
class TestSnapshotRefresh:
    def test_refreshes_the_last_pre_heated_prompt_every_n_requests(self):
        OllamaLLM._last_pre_heat_messages = [{"role": "user", "content": "chat"}]
        llm = _llm()
        with patch.object(OllamaLLM, '_SNAPSHOT_REFRESH_EVERY', 2), \
             patch.object(llm, '_pre_heat_raw_truncated', return_value=True) as warm:
            for _ in range(4):
                FakeConn.lines = _lines("x")
                llm._background_completion(MESSAGES)
        assert warm.call_count == 2
        assert warm.call_args.kwargs == {"refresh": True}
        assert OllamaLLM._pre_heat_in_flight is False

    def test_refresh_does_not_restart_the_idle_clock(self):
        llm = _llm()
        with patch.object(llm, '_rendered_prompt', return_value="sys<|im_end|>\n<|im_start|>assistant\n"), \
             patch('assistant.llm.OllamaLLM.requests.post') as post:
            post.return_value.raise_for_status = Mock()
            assert llm._pre_heat_raw_truncated([{"role": "user", "content": "x"}], refresh=True)
        assert OllamaLLM._slot.interactive_last_end == 0.0 and OllamaLLM._slot.interactive_active == 0

    def test_skipped_while_a_real_pre_heat_is_in_flight(self):
        OllamaLLM._last_pre_heat_messages = [{"role": "user", "content": "chat"}]
        OllamaLLM._pre_heat_in_flight = True
        llm = _llm()
        with patch.object(OllamaLLM, '_SNAPSHOT_REFRESH_EVERY', 1), \
             patch.object(llm, '_pre_heat_raw_truncated') as warm:
            FakeConn.lines = _lines("x")
            llm._background_completion(MESSAGES)
        warm.assert_not_called()
        assert OllamaLLM._pre_heat_in_flight is True, "the refresh must not release someone else's guard"


@pytest.mark.unit
class TestContract:
    def test_retry_does_not_retry_a_preempted_request(self):
        class Dummy(AbstractLLM):
            def __init__(self):
                pass
            create_streaming_completion = create_completion = create_completion_with_images = None
            initialize = name = provider = None
        fn = Mock(side_effect=LLMPreempted("evicted"))
        with patch('assistant.llm.AbstractLLM.time.sleep') as sleep, pytest.raises(LLMPreempted):
            Dummy()._retry(fn)
        assert fn.call_count == 1
        sleep.assert_not_called()

    def test_think_passes_background_through_to_the_image_call(self):
        llm = _llm()
        with patch.object(llm, 'create_completion_with_images', return_value="ok") as call:
            llm.think("sys", "user", images=[b"jpg"], background=True)
        assert call.call_args.kwargs["background"] is True

    def test_wrappers_forward_the_flag_and_locality(self):
        from assistant.llm.CachedLLM import CachedLLM
        from assistant.llm.VaultLLM import VaultLLM
        inner = Mock()
        inner.is_local.return_value = True
        inner.create_completion_with_images.return_value = "ok"
        cached = CachedLLM(inner)
        assert cached.is_local() is True
        cached.create_completion_with_images(MESSAGES, ["QUJD"], background=True)
        assert inner.create_completion_with_images.call_args.kwargs["background"] is True
        vault = VaultLLM.__new__(VaultLLM)
        vault.llm, vault.redactor = inner, Mock(redact_messages=lambda m: m, restore=lambda t: (t, 0))
        assert vault.is_local() is True
        vault.create_completion_with_images(MESSAGES, ["QUJD"], background=True)
        assert inner.create_completion_with_images.call_args.kwargs["background"] is True

    def test_only_ollama_is_local(self):
        from assistant.llm.ClaudeLLM import ClaudeLLM
        assert _llm().is_local() is True
        assert ClaudeLLM.__new__(ClaudeLLM).is_local() is False
