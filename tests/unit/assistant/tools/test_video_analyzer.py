import json
import struct
from unittest.mock import patch, Mock

import pytest

from assistant.tools.VideoAnalyzer import VideoAnalyzer, short_clip_fps, DESCRIPTION_PROMPT_USER, DEFAULT_MIME_TYPE

REPLY = '{"short_description": "A dog runs", "detailed_description": "A dog runs on a beach.", "spoken_content": ""}'


def _video_llm(*replies) -> Mock:
    llm = Mock()
    llm.supports_video.return_value = True
    llm.max_video_bytes.return_value = 2 * 1024 * 1024 * 1024
    if len(replies) == 1:
        llm.think_with_video.return_value = replies[0]
    else:
        llm.think_with_video.side_effect = list(replies)
    return llm


def _mp4(duration_seconds: float | None) -> bytes:
    """An mp4 with just a movie header; no header at all when duration is None."""
    def box(box_type: bytes, payload: bytes) -> bytes:
        return struct.pack(">I4s", 8 + len(payload), box_type) + payload
    if duration_seconds is None:
        return box(b"ftyp", b"isom") + box(b"mdat", b"\x00" * 16)
    mvhd = box(b"mvhd", bytes(12) + struct.pack(">II", 600, int(duration_seconds * 600)) + bytes(80))
    return box(b"ftyp", b"isom") + box(b"moov", mvhd)


def _analyzer_with_reply(*replies) -> tuple[VideoAnalyzer, Mock]:
    llm = _video_llm(*replies)
    return VideoAnalyzer(llm=llm), llm


@pytest.mark.unit
class TestFormatDescription:
    def test_empty_spoken_content_is_omitted(self):
        assert VideoAnalyzer.format_description({
            "short_description": "A", "detailed_description": "B", "spoken_content": "  ",
        }) == "Video description: [A] B"

    def test_spoken_content_is_appended(self):
        assert VideoAnalyzer.format_description({
            "short_description": "A", "detailed_description": "B", "spoken_content": "Hi mum!",
        }) == "Video description: [A] B Spoken: Hi mum!"


@pytest.mark.unit
class TestParseResponse:
    def test_spoken_content_is_optional(self):
        data = VideoAnalyzer._parse_response('{"short_description": "A", "detailed_description": "B"}')
        assert "spoken_content" not in data

    def test_missing_required_key_raises(self):
        with pytest.raises(KeyError):
            VideoAnalyzer._parse_response('{"short_description": "A"}')

@pytest.mark.unit
class TestAnalyze:
    def test_sends_video_and_prompt(self):
        analyzer, llm = _analyzer_with_reply(REPLY)
        result = analyzer.analyze(b"vid", "video/quicktime", context="Sent by Alice")
        assert result["short_description"] == "A dog runs"
        kwargs = llm.think_with_video.call_args[1]
        assert kwargs["video_bytes"] == b"vid"
        assert kwargs["mime_type"] == "video/quicktime"
        assert kwargs["user_message"] == DESCRIPTION_PROMPT_USER
        assert "Sent by Alice" in kwargs["system_message"]

    def test_empty_context_has_no_section(self):
        analyzer, llm = _analyzer_with_reply(REPLY)
        analyzer.analyze(b"vid", "video/mp4")
        assert "Additional context" not in llm.think_with_video.call_args[1]["system_message"]

    def test_retries_on_parse_failure(self):
        analyzer, llm = _analyzer_with_reply("garbage", REPLY)
        assert analyzer.analyze(b"vid", "video/mp4")["short_description"] == "A dog runs"
        assert llm.think_with_video.call_count == 2

    def test_gives_up_after_three_parse_failures(self):
        analyzer, llm = _analyzer_with_reply("bad", "bad", "bad")
        assert analyzer.analyze(b"vid", "video/mp4") is None
        assert llm.think_with_video.call_count == 3

    def test_llm_exception_returns_none_without_retry(self):
        analyzer, llm = _analyzer_with_reply(REPLY)
        llm.think_with_video.side_effect = RuntimeError("api down")
        assert analyzer.analyze(b"vid", "video/mp4") is None
        assert llm.think_with_video.call_count == 1


@pytest.mark.unit
class TestShortClipSampling:
    def test_short_clip_fps_table(self):
        assert short_clip_fps(12.5) is None
        assert short_clip_fps(2.0) is None
        assert short_clip_fps(1.5) == 2.0
        assert short_clip_fps(0.57) == 4.0
        assert short_clip_fps(0.3) == 7.0
        assert short_clip_fps(0.1) == 20.0
        assert short_clip_fps(0.05) == 24.0   # Gemini's cap, not 40
        assert short_clip_fps(0.0) == 24.0
        assert short_clip_fps(None) == 24.0

    def test_short_gif_gets_an_fps_override(self):
        analyzer, llm = _analyzer_with_reply(REPLY)
        analyzer.analyze(_mp4(0.5), "video/mp4")
        assert llm.think_with_video.call_args[1]["fps"] == 4.0

    def test_normal_clip_keeps_default_sampling(self):
        analyzer, llm = _analyzer_with_reply(REPLY)
        analyzer.analyze(_mp4(12.0), "video/mp4")
        assert llm.think_with_video.call_args[1]["fps"] is None

    def test_unreadable_duration_keeps_default_sampling(self):
        """Never sample a clip of unknown length at 24 fps: it could be minutes long."""
        analyzer, llm = _analyzer_with_reply(REPLY)
        analyzer.analyze(_mp4(None), "video/mp4")
        assert llm.think_with_video.call_args[1]["fps"] is None
        analyzer.analyze(b"\x1a\x45\xdf\xa3 webm bytes", "video/webm")
        assert llm.think_with_video.call_args[1]["fps"] is None


@pytest.mark.unit
class TestDescribeBytes:
    def test_full_pipeline_success(self):
        analyzer, _ = _analyzer_with_reply(REPLY)
        assert analyzer.describe_bytes(b"vid", mime_type="video/mp4", context="from Alice") == \
            "Video description: [A dog runs] A dog runs on a beach."

    def test_mime_type_defaults_when_unknown(self):
        analyzer, llm = _analyzer_with_reply(REPLY)
        analyzer.describe_bytes(b"vid")
        assert llm.think_with_video.call_args[1]["mime_type"] == DEFAULT_MIME_TYPE

    def test_llm_without_video_support_returns_fallback_without_calling(self):
        llm = Mock()
        llm.supports_video.return_value = False
        llm.name.return_value = "Claude"
        analyzer = VideoAnalyzer(llm=llm)
        assert analyzer.describe_bytes(b"vid") == "[Video]"
        llm.think_with_video.assert_not_called()

    def test_too_large_returns_fallback(self):
        analyzer, llm = _analyzer_with_reply(REPLY)
        with patch('assistant.tools.VideoAnalyzer.config') as cfg:
            cfg.get.return_value = 1
            assert analyzer.describe_bytes(b"x" * (1024 * 1024 + 1)) == "[Video]"
        cfg.get.assert_called_once_with('ai.video.max_size_mb', 200)
        llm.think_with_video.assert_not_called()

    def test_clip_under_configured_cap_is_analyzed(self):
        analyzer, llm = _analyzer_with_reply(REPLY)
        with patch('assistant.tools.VideoAnalyzer.config') as cfg:
            cfg.get.return_value = 1
            assert "A dog runs" in analyzer.describe_bytes(b"x" * (1024 * 1024))
        llm.think_with_video.assert_called_once()

    def test_max_size_is_the_cap_describe_bytes_applies(self):
        llm = _video_llm(REPLY)
        llm.max_video_bytes.return_value = 5 * 1024 * 1024
        assert VideoAnalyzer(llm=llm).max_size() == 5 * 1024 * 1024

    def test_backend_limit_caps_the_configured_size(self):
        analyzer, llm = _analyzer_with_reply(REPLY)
        llm.max_video_bytes.return_value = 10
        with patch('assistant.tools.VideoAnalyzer.config') as cfg:
            cfg.get.return_value = 200
            assert analyzer.describe_bytes(b"x" * 100) == "[Video]"
        llm.think_with_video.assert_not_called()

    def test_backend_without_a_video_limit_keeps_the_configured_cap(self):
        # Every non-Gemini backend reports 0; that must not collapse the cap to zero bytes
        llm = _video_llm(REPLY)
        llm.max_video_bytes.return_value = 0
        with patch('assistant.tools.VideoAnalyzer.config') as cfg:
            cfg.get.return_value = 200
            assert VideoAnalyzer(llm=llm).max_size() == 200 * 1024 * 1024

    def test_max_size_survives_a_video_tier_that_cannot_be_built(self):
        # Integrations probe the cap before downloading; a broken tier must not take the whole
        # message down with it (the analysis itself falls back to the placeholder later)
        with patch('assistant.llm.parse_llm_config', side_effect=RuntimeError("no video tier")), \
             patch('assistant.tools.VideoAnalyzer.config') as cfg:
            cfg.get.return_value = 50
            assert VideoAnalyzer().max_size() == 50 * 1024 * 1024

    def test_analyze_returns_none_uses_fallback(self):
        analyzer, _ = _analyzer_with_reply("never valid json")
        assert analyzer.describe_bytes(b"vid") == "[Video]"

    def test_exception_returns_fallback(self):
        analyzer = VideoAnalyzer(llm=_video_llm(REPLY))
        with patch.object(analyzer, 'analyze', side_effect=RuntimeError("boom")):
            assert analyzer.describe_bytes(b"vid", fallback="[Clip]") == "[Clip]"

@pytest.mark.unit
class TestVideoLlm:
    def test_creates_from_video_tier_on_first_use(self):
        mock_llm = Mock()
        mock_llm.missing_api_key.return_value = None
        with patch('assistant.llm.parse_llm_config', return_value=("SomeLLM", "some-model")) as mock_parse, \
             patch('assistant.llm.create_llm', return_value=mock_llm) as mock_create:
            assert VideoAnalyzer().llm is mock_llm
        mock_parse.assert_called_once_with('video')
        mock_create.assert_called_once_with("SomeLLM", "some-model")
        mock_llm.initialize.assert_called_once()

    def test_caches_llm_across_calls(self):
        with patch('assistant.llm.parse_llm_config', return_value=("SomeLLM", None)), \
             patch('assistant.llm.create_llm', return_value=Mock(**{"missing_api_key.return_value": None})) as mock_create:
            analyzer = VideoAnalyzer()
            assert analyzer.llm is analyzer.llm
        mock_create.assert_called_once()


def _keyless_llm() -> Mock:
    llm = _video_llm(REPLY)
    llm.missing_api_key.return_value = "api_keys.gemini"
    return llm


@pytest.mark.unit
class TestVideoTierWithoutAKey:
    """An install that runs on Claude alone never gave the video tier's default backend a key."""

    def test_a_clip_stays_undescribed_and_no_error_is_logged(self):
        llm = _keyless_llm()
        with patch('assistant.llm.parse_llm_config', return_value=("SomeLLM", None)), \
             patch('assistant.llm.create_llm', return_value=llm), \
             patch('assistant.tools.VideoAnalyzer.error') as logged_error:
            assert VideoAnalyzer().describe_bytes(b"vid", fallback="[Clip]") == "[Clip]"
        llm.initialize.assert_not_called()
        llm.think_with_video.assert_not_called()
        logged_error.assert_not_called()

    def test_the_size_cap_is_the_configured_one_and_no_error_is_logged(self):
        with patch('assistant.llm.parse_llm_config', return_value=("SomeLLM", None)), \
             patch('assistant.llm.create_llm', return_value=_keyless_llm()), \
             patch('assistant.tools.VideoAnalyzer.error') as logged_error, \
             patch('assistant.tools.VideoAnalyzer.config') as cfg:
            cfg.get.return_value = 50
            assert VideoAnalyzer().max_size() == 50 * 1024 * 1024
        logged_error.assert_not_called()

    def test_a_key_set_later_is_picked_up_without_a_restart(self):
        llm = _keyless_llm()
        with patch('assistant.llm.parse_llm_config', return_value=("SomeLLM", None)), \
             patch('assistant.llm.create_llm', return_value=llm):
            analyzer = VideoAnalyzer()
            assert analyzer.describe_bytes(b"vid") == "[Video]"
            llm.missing_api_key.return_value = None
            assert "A dog runs" in analyzer.describe_bytes(b"vid")
