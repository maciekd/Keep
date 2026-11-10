"""The background image request of OpenAICompatibleLLM: a raw SSE stream over the slot's own
connection (the OpenAI shape — `data: {chunk}` lines, `data: [DONE]` last), whose progress
chunks feed the prefill bar and whose closing usage makes the cache report."""
import json
import threading
import time
from unittest.mock import patch

import pytest

from assistant.llm.AbstractLLM import LLMPreempted
from assistant.llm.OpenAICompatibleLLM import OpenAICompatibleLLM
from assistant.llm.PrefillTracker import LogSource

MESSAGES = [{"role": "system", "content": "analyst"},
            {"role": "user", "content": [{"type": "text", "text": "describe"},
                                         {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}}]}]


def _data(obj) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n"


def _content(text) -> bytes:
    return _data({"choices": [{"index": 0, "delta": {"content": text}}]})


def _reasoning(text) -> bytes:
    return _data({"choices": [{"index": 0, "delta": {"reasoning_content": text}}]})


def _progress(total, cache, processed) -> bytes:
    return _data({"choices": [], "prompt_progress": {"total": total, "cache": cache, "processed": processed, "time_ms": 100}})


def _usage(prompt, cached, completion=4) -> bytes:
    return _data({"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                                           "prompt_tokens_details": {"cached_tokens": cached}}})


DONE = b"data: [DONE]\n"


class FakeSock:
    def __init__(self):
        self.shut = threading.Event()

    def shutdown(self, how):
        self.shut.set()


class FakeResponse:
    """Streams `lines`; with none left it blocks like a socket read until the socket is shut."""

    def __init__(self, sock, lines, status=200):
        self._sock, self._lines, self.status = sock, list(lines), status

    def readline(self):
        if self._lines:
            return self._lines.pop(0)
        self._sock.shut.wait(timeout=2)
        raise ConnectionResetError("socket shut")

    def read(self, n=-1):
        return b"server said no"


class FakeConn:
    instances: list["FakeConn"] = []
    lines: list[bytes] = []
    status = 200

    def __init__(self, host, port, timeout=None):
        self.host, self.port = host, port
        self.sock = None
        self.closed = False
        FakeConn.instances.append(self)

    def connect(self):
        self.sock = FakeSock()

    def request(self, method, path, body=None, headers=None):
        self.sent = (method, path, json.loads(body), headers or {})

    def getresponse(self):
        return FakeResponse(self.sock, FakeConn.lines, FakeConn.status)

    def close(self):
        self.closed = True


def _config(values: dict):
    def get(key, default=None):
        return values.get(key, default)
    return patch('assistant.llm.OpenAICompatibleLLM.config.get', side_effect=get)


@pytest.fixture(autouse=True)
def _fresh():
    def reset():
        OpenAICompatibleLLM._slot.reset()
        OpenAICompatibleLLM._prefill._requests.clear()
        FakeConn.instances.clear()
        FakeConn.lines = []
        FakeConn.status = 200
    reset()
    with patch('http.client.HTTPConnection', FakeConn), \
            patch.object(OpenAICompatibleLLM, 'background_idle_seconds', return_value=0.0), \
            patch.object(LogSource, 'size', return_value=None):
        yield
    reset()


def _llm(values=None) -> OpenAICompatibleLLM:
    with _config(values or {}):
        return OpenAICompatibleLLM(model='gemma-4-26b', base_url='http://192.0.2.7:8000/v1')


def _run(llm, values=None):
    with _config(values or {}):
        return llm._background_completion(MESSAGES)


@pytest.mark.unit
class TestTheStream:
    def test_the_reply_is_the_content_deltas_joined_and_the_connection_is_closed_after_done(self):
        FakeConn.lines = [b": keepalive\n", b"\n", _reasoning("hm"), _content("A red "), _content("square."), DONE,
                          _content("never read")]
        llm = _llm()
        assert _run(llm) == "A red square."
        conn = FakeConn.instances[0]
        assert conn.closed
        assert conn.sent[0] == "POST" and conn.sent[1] == "/v1/chat/completions", "the path is the base URL's plus the endpoint"
        assert (conn.host, conn.port) == ("192.0.2.7", 8000)
        assert OpenAICompatibleLLM._slot.active is False and OpenAICompatibleLLM._slot.sock is None

    def test_the_request_streams_with_usage_and_progress_and_declares_no_tool(self):
        FakeConn.lines = [_content("ok"), DONE]
        _run(_llm())
        body = FakeConn.instances[0].sent[2]
        assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
        assert body["return_progress"] is True
        assert body["model"] == "gemma-4-26b"
        assert "tools" not in body
        assert body["reasoning_effort"] == OpenAICompatibleLLM.DEFAULT_REASONING_EFFORT

    def test_return_progress_off_leaves_the_key_out_of_the_raw_request_too(self):
        FakeConn.lines = [_content("ok"), DONE]
        _run(_llm(), values={'openai_compatible.return_progress': False})
        assert "return_progress" not in FakeConn.instances[0].sent[2]

    def test_the_bearer_header_goes_out_only_with_a_configured_key(self):
        FakeConn.lines = [_content("ok"), DONE]
        _run(_llm())
        assert "Authorization" not in FakeConn.instances[0].sent[3], "the SDK's placeholder key is not a server's"
        FakeConn.lines = [_content("ok"), DONE]
        _run(_llm(), values={'openai_compatible.api_key': 'sk-local-1'})
        assert FakeConn.instances[1].sent[3]["Authorization"] == "Bearer sk-local-1"

    def test_a_stream_closed_before_done_is_a_failure_and_frees_the_slot(self):
        FakeConn.lines = [_content("half"), b""]
        with pytest.raises(RuntimeError):
            _run(_llm())
        assert OpenAICompatibleLLM._slot.active is False

    def test_a_non_200_status_is_a_failure_naming_the_status(self):
        FakeConn.status = 503
        with pytest.raises(RuntimeError, match="503"):
            _run(_llm())
        assert OpenAICompatibleLLM._slot.active is False


@pytest.mark.unit
class TestTheBarAndTheReport:
    def test_progress_chunks_drive_the_bar_and_the_first_delta_ends_the_prefill(self):
        tracker = OpenAICompatibleLLM._prefill
        seen = []

        class WatchingResponse(FakeResponse):
            def readline(self):
                seen.append(tracker.progress())
                return super().readline()

        FakeConn.lines = [_progress(total=4000, cache=0, processed=1000), _content("A"), DONE]
        with patch.object(FakeConn, 'getresponse', lambda self: WatchingResponse(self.sock, FakeConn.lines)):
            _run(_llm())
        # seen[i] is the bar before line i is read: before the progress chunk nothing is known,
        # after it a quarter of the work is done, after the first content delta the prefill is over.
        assert seen[0] == 0.0
        assert seen[1] == pytest.approx(25.0)
        assert seen[2] == 100.0
        assert tracker.progress() is None, "the request leaves the tracker with the slot"

    def test_the_closing_usage_makes_the_cache_report_from_the_raw_dict(self):
        FakeConn.lines = [_content("A"), _usage(prompt=6000, cached=4500), DONE]
        llm = _llm()
        reports = []
        with patch.object(llm, 'report_cache_usage', side_effect=lambda m, c, t, note=None: reports.append((c, t))):
            _run(llm)
        assert reports == [(4500, 6000)]

    def test_a_progress_chunk_makes_the_report_early_and_the_usage_does_not_repeat_it(self):
        FakeConn.lines = [_progress(total=6000, cache=4500, processed=5000), _content("A"), _usage(6000, 4500), DONE]
        llm = _llm()
        reports = []
        with patch.object(llm, 'report_cache_usage', side_effect=lambda m, c, t, note=None: reports.append((c, t))):
            _run(llm)
        assert reports == [(4500, 6000)]

    def test_a_usage_without_a_cached_count_makes_no_report(self):
        FakeConn.lines = [_content("A"), _data({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 1}}), DONE]
        llm = _llm()
        with patch.object(llm, 'report_cache_usage') as report:
            _run(llm)
        report.assert_not_called()


@pytest.mark.unit
class TestPreemption:
    def test_a_user_turn_shuts_the_socket_and_the_caller_reads_it_as_preempted(self):
        FakeConn.lines = [_content("half")]    # then the read blocks like a prefill in progress
        outcome = {}
        llm = _llm()

        def bg():
            try:
                _run(llm)
            except LLMPreempted:
                outcome["preempted"] = True
            except Exception as e:
                outcome["error"] = e

        t = threading.Thread(target=bg, daemon=True)
        t.start()
        deadline = time.monotonic() + 1
        while OpenAICompatibleLLM._slot.sock is None and time.monotonic() < deadline:
            time.sleep(0.005)
        assert OpenAICompatibleLLM._slot.sock is not None, "the connection was put under the gate"
        OpenAICompatibleLLM._slot.interactive_begin()
        t.join(timeout=1)
        OpenAICompatibleLLM._slot.interactive_end()
        assert outcome == {"preempted": True}, "the socket error after an eviction is the eviction, not a failure"
