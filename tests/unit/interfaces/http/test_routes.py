"""The HTTP /stream interface: the legacy wire format every existing client depends on
(iOS, Android outposts, tests/prompt), pinned per mode,
plus the route on a real Quart app with a fake Assistant.

Replaces RequestProcessor's TestKeepaliveSentinel / TestGenerateAudio content-type tests
and the source-keyed halves of TestStreamResponse."""
import asyncio
from unittest.mock import Mock

import pytest
from quart import Quart

from assistant.Assistant import AssistantBusy
from assistant.AssistantEvents import (AudioChunk, CommandRan, Done, Notice, RoundEnd, Thinking,
                                       ThinkingDone, Token, UserIdentified)
from assistant.AssistantRequest import RequestContext
from interfaces.http import WireFormat
from interfaces.http.routes import HttpDeps, register


def ctx(source, text_only=False, server_sound=False):
    return RequestContext(source=source, user="U", user_input="q", text_only=text_only, server_sound=server_sound)


def render(events, context):
    return list(WireFormat.render(iter(events), context))


# One request: a first round, a command follow-up round, then Done.
TWO_ROUNDS = [
    UserIdentified("U"), Token("Hel"), Token("lo"), CommandRan("turn_on", "kitchen"),
    RoundEnd("<thinking>t</thinking>Hello turn_on(kitchen)"),
    Token("Done!"), RoundEnd("Done!"), Done("Done!"),
]


class TestWireFormatTable:
    """(source, text_only, server_sound) → exact chunks for TWO_ROUNDS."""

    @pytest.mark.parametrize("source,text_only,server_sound,expected", [
        # prompt suite / outposts asking for text: tokens swallowed, raw text per round + \x1e
        ("VoiceToVoice", True, False,
         ["<thinking>t</thinking>Hello turn_on(kitchen)", "\x1e", "Done!", "\x1e"]),
        # server-speaker playback: nothing per round, the final raw text after Done
        ("VoiceToVoice", False, True, ["Done!"]),
        # both flags: text-mode rounds and the final text
        ("VoiceToVoice", False | True, True,
         ["<thinking>t</thinking>Hello turn_on(kitchen)", "\x1e", "Done!", "\x1e", "Done!"]),
        # Streaming chat channels: live tokens + \x1e, never the raw text
        ("TextChat", True, False, ["Hel", "lo", "\x1e", "Done!", "\x1e"]),
        # an external client still sending source=Telegram: plain TextChat rendering
        ("Telegram", True, False, ["Hel", "lo", "\x1e", "Done!", "\x1e"]),
        # unknown source behaves like TextChat
        ("TextToText", True, False, ["Hel", "lo", "\x1e", "Done!", "\x1e"]),
    ])
    def test_text_modes(self, source, text_only, server_sound, expected):
        assert render(TWO_ROUNDS, ctx(source, text_only, server_sound)) == expected

    def test_audio_mode_streams_only_audio_bytes(self):
        events = [UserIdentified("U"), AudioChunk(b"\xff\xfb"), Thinking("x", 1), AudioChunk(b"\x00\x01"),
                  RoundEnd("raw"), Done("raw")]
        assert render(events, ctx("VoiceToVoice")) == [b"\xff\xfb", b"\x00\x01"]

    def test_notice_is_emitted_verbatim_in_every_mode(self):
        for context in (ctx("VoiceToVoice", True), ctx("TextChat", True), ctx("VoiceToVoice"), ctx("Telegram", True)):
            assert render([Notice("Request denied.")], context) == ["Request denied."]

    def test_failed_follow_up_round_is_notice_then_delimiter(self):
        events = [Token("a"), RoundEnd("a"), Notice("The AI servers are unavailable right now."), RoundEnd()]
        assert render(events, ctx("VoiceToVoice", True)) == ["a", "\x1e", "The AI servers are unavailable right now.", "\x1e"]
        assert render(events, ctx("TextChat", True)) == ["a", "\x1e", "The AI servers are unavailable right now.", "\x1e"]

    def test_killed_stream_has_no_done_and_no_final_text(self):
        events = [Token("a")]  # stream ended silently
        assert render(events, ctx("VoiceToVoice", False, True)) == []

    def test_response_headers(self):
        assert "text/plain" in WireFormat.response_headers(ctx("VoiceToVoice", True))["Content-Type"]
        assert "text/plain" in WireFormat.response_headers(ctx("VoiceToVoice", False, True))["Content-Type"]
        audio = WireFormat.response_headers(ctx("VoiceToVoice"))
        assert audio["Content-Type"] == "audio/mpeg" and audio["Accept-Ranges"] == "none"


class TestTelegramOverHttpRendersLikeTextChat:
    """The bot runs in-process; the wire format has no channel-specific framing left."""

    def test_same_chunks_as_text_chat_with_thinking_before_the_tokens(self):
        events = [Thinking("hmm", 137), ThinkingDone(0.05), Token("a"), Token("b"), RoundEnd("ab")]
        assert render(events, ctx("Telegram", True)) == render(events, ctx("TextChat", True)) == ["a", "b", "\x1e"]

    def test_producer_exception_propagates(self):
        def events():
            yield Token("a")
            raise RuntimeError("tts blew up")

        gen = WireFormat.render(events(), ctx("Telegram", True))
        assert next(gen) == "a"
        with pytest.raises(RuntimeError, match="tts blew up"):
            list(gen)


# --- the route ---------------------------------------------------------------------------

class FakeStream:
    def __init__(self, events):
        self._it = iter(events)
        self.killed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)


class FakeAssistant:
    def __init__(self, events=None, busy=None):
        self.events = events if events is not None else TWO_ROUNDS
        self.busy = busy
        self.calls = []

    def stream(self, context, query=None, images=None, request_data=None):
        self.calls.append((context, request_data))
        if self.busy:
            raise AssistantBusy(self.busy)
        return FakeStream(self.events)


@pytest.fixture
def assistant():
    return FakeAssistant()


@pytest.fixture
def text_stream():
    return Mock()


@pytest.fixture
def mic_routes():
    m = Mock()
    m.accept_transcription.return_value = None
    return m


@pytest.fixture
def app(assistant, text_stream, mic_routes):
    a = Quart(__name__)
    register(a, HttpDeps(assistant=assistant, text_stream=text_stream, microphone_routes=mic_routes))
    return a


def get(app, path, headers=None):
    async def run():
        client = app.test_client()
        r = await client.get(path, headers=headers or {})
        return r, await r.get_data()
    return asyncio.run(run())


def post(app, path, json):
    async def run():
        client = app.test_client()
        r = await client.post(path, json=json)
        return r, await r.get_data()
    return asyncio.run(run())


class TestVoicelessAudioClient:
    """A process without the voice interface answers audio-mode clients with text (Phase 6)."""

    def test_audio_client_gets_textchat_text_not_empty_audio(self, assistant):
        a = Quart(__name__)
        register(a, HttpDeps(assistant=assistant, voice_enabled=False))
        r, body = get(a, "/stream?query=hi", headers={"User-Agent": "Floxy/1 CFNetwork/1.0"})
        assert r.status_code == 200
        assert "text/plain" in r.headers["Content-Type"]
        assert body.decode() == "Hello\x1eDone!\x1e"
        context, _ = assistant.calls[0]
        assert context.source == "TextChat" and context.text_only

    def test_text_and_server_sound_clients_are_untouched(self, assistant):
        a = Quart(__name__)
        register(a, HttpDeps(assistant=assistant, voice_enabled=False))
        get(a, "/stream?text=true&source=VoiceToVoice&query=hi")
        get(a, "/stream?server_sound=true&source=VoiceToVoice&query=hi")
        assert [c.source for c, _ in assistant.calls] == ["VoiceToVoice", "VoiceToVoice"]
        assert assistant.calls[0][0].text_only and assistant.calls[1][0].server_sound

class TestStreamRoute:
    def test_text_query_renders_legacy_wire(self, app, assistant):
        r, body = get(app, "/stream?text=true&test=true&source=VoiceToVoice&query=hi")
        assert r.status_code == 200
        assert "text/plain" in r.headers["Content-Type"]
        assert body.decode() == "<thinking>t</thinking>Hello turn_on(kitchen)\x1eDone!\x1e"
        context, req_data = assistant.calls[0]
        assert context.source == "VoiceToVoice" and context.text_only and context.test
        assert req_data.args["query"] == "hi"

    def test_audio_query_from_ios_user_agent(self, app, assistant):
        assistant.events = [AudioChunk(b"\x01\x02"), RoundEnd("r"), Done("r")]
        r, body = get(app, "/stream?query=hi", headers={"User-Agent": "Floxy/1 CFNetwork/1.0"})
        assert r.headers["Content-Type"] == "audio/mpeg"
        assert body == b"\x01\x02"
        assert assistant.calls[0][0].source == "VoiceToVoice"

    def test_post_body_args_are_merged(self, app, assistant):
        r, body = post(app, "/stream", {"query": "hi", "text": "true", "source": "Telegram", "user": "M",
                                         "images": [{"data": "abc", "media_type": "image/jpeg"}]})
        assert r.status_code == 200
        context, _ = assistant.calls[0]
        assert context.source == "Telegram" and context.user == "M" and context.images == [{"data": "abc", "media_type": "image/jpeg"}]

    def test_empty_query_returns_204_and_closes_text_stream(self, app, assistant, text_stream):
        r, _ = get(app, "/stream?query=&request_id=r1")
        assert r.status_code == 204
        assert assistant.calls == []
        text_stream.close.assert_called_once_with("r1")

    def test_microphone_rejection_returns_204(self, app, assistant, mic_routes, text_stream):
        mic_routes.accept_transcription.return_value = "lost the vote"
        r, _ = get(app, "/stream?query=hi&request_id=r2")
        assert r.status_code == 204
        assert assistant.calls == []
        text_stream.close.assert_called_once_with("r2")

    def test_every_transcription_is_announced_with_the_votes_verdict(self, app, assistant, mic_routes):
        from assistant.AssistantEvents import observers
        obs = Mock()
        observers().subscribe(obs)
        try:
            get(app, "/stream?query=hi&mic=kitchen&confidence=0.9")
            mic_routes.accept_transcription.return_value = "lost the vote"
            get(app, "/stream?query=hi&mic=hall&confidence=bogus")
            get(app, "/stream?query=")
        finally:
            observers().unsubscribe(obs)
        calls = obs.on_request_received.call_args_list
        assert [(c.args[0].user_input, c.args[1:]) for c in calls] == \
            [("hi", ("kitchen", 0.9, None)), ("hi", ("hall", None, "lost the vote"))]
        assert len(assistant.calls) == 1

    def test_debug_command_is_answered_directly(self, app, assistant, monkeypatch):
        from assistant.commands.DebugCommands import DebugCommands
        monkeypatch.setattr(DebugCommands, "is_command", staticmethod(lambda q: q == "!dbg"))
        monkeypatch.setattr(DebugCommands, "handle", lambda self, q: "debug reply")
        r, body = get(app, "/stream?query=!dbg")
        assert r.status_code == 200 and body == b"debug reply"
        assert assistant.calls == []

    def test_identified_user_is_announced_to_the_text_stream(self, app, text_stream):
        get(app, "/stream?query=hi&user=Anna&request_id=r3&text=true")
        text_stream.publish.assert_called_once_with("r3", {"type": "user", "name": "Anna"})

    def test_busy_assistant_returns_503(self, app, assistant, text_stream):
        assistant.busy = "Server busy, no available handlers"
        r, body = get(app, "/stream?query=hi&request_id=r4")
        assert r.status_code == 503
        assert b"no available handlers" in body
        text_stream.close.assert_called_once_with("r4")

    def test_unexpected_error_returns_500(self, app, assistant, text_stream):
        assistant.stream = Mock(side_effect=RuntimeError("boom"))
        r, body = get(app, "/stream?query=hi&request_id=r5")
        assert r.status_code == 500
        assert b"internal error" in body
        text_stream.close.assert_called_once_with("r5")

    def test_voiceless_process_has_no_mic_gate_or_text_stream(self, assistant):
        a = Quart(__name__)
        register(a, HttpDeps(assistant=assistant))
        r, body = get(a, "/stream?query=hi&text=true&source=TextChat&request_id=r6")
        assert r.status_code == 200
        assert body.decode() == "Hello\x1eDone!\x1e"
