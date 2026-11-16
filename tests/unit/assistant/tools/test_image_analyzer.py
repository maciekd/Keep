import io
import json
import os
from unittest.mock import patch, Mock

import pytest
from PIL import Image

from assistant.tools.ImageAnalyzer import ImageAnalyzer, resize_to_jpeg, convert_via_sips


def _png(size=(100, 100), mode="RGB", color="red") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, color=color).save(buf, format="PNG")
    return buf.getvalue()


def _analyzer_with_reply(*replies) -> tuple[ImageAnalyzer, Mock]:
    llm = Mock()
    if len(replies) == 1:
        llm.think.return_value = replies[0]
    else:
        llm.think.side_effect = list(replies)
    return ImageAnalyzer(llm=llm), llm


@pytest.mark.unit
class TestResizeToJpeg:
    def test_resize_small_image(self):
        result = resize_to_jpeg(_png())
        assert result is not None
        assert result[:2] == b'\xff\xd8'

    def test_resize_large_image_scales_down(self):
        result = resize_to_jpeg(_png(size=(1600, 1200), color="blue"))
        assert result is not None
        assert max(Image.open(io.BytesIO(result)).size) == 1024

    def test_resize_applies_the_exif_orientation(self):
        # A phone stores a portrait JPEG as its landscape sensor frame plus an orientation tag;
        # a chat thumbnail is shown as stored, so the turn has to be in the pixels.
        img = Image.new("RGB", (200, 100), "red")
        exif = Image.Exif()
        exif[0x0112] = 6  # rotate 90° clockwise to display
        buf = io.BytesIO()
        img.save(buf, format="JPEG", exif=exif.tobytes())
        result = resize_to_jpeg(buf.getvalue())
        assert Image.open(io.BytesIO(result)).size == (100, 200)

    def test_resize_rgba_converted_to_rgb(self):
        assert resize_to_jpeg(_png(size=(200, 200), mode="RGBA", color=(255, 0, 0, 128))) is not None

    @pytest.mark.parametrize("ink", [0, 255], ids=["black_drawing", "white_drawing"])
    @pytest.mark.parametrize("form", ["RGBA", "LA", "palette", "colour_key"])
    def test_a_drawing_on_a_transparent_background_stands_out_whether_black_or_white(self, form, ink):
        """Dropped alpha leaves the colour stored under it — the drawing's own here, as in most exports."""
        box = (20, 20, 40, 40)
        buf = io.BytesIO()
        if form == "palette":
            img = Image.new("P", (60, 60), 1)
            img.putpalette([ink] * 6)
            img.paste(0, box)
            img.save(buf, format="PNG", transparency=1)
        elif form == "colour_key":
            img = Image.new("RGB", (60, 60), (ink, ink, 1))
            img.paste((ink, ink, ink), box)
            img.save(buf, format="PNG", transparency=(ink, ink, 1))
        else:
            img = Image.new("RGBA", (60, 60), (ink, ink, ink, 0))
            img.paste((ink, ink, ink, 255), box)
            img.convert(form).save(buf, format="PNG")
        out = Image.open(io.BytesIO(resize_to_jpeg(buf.getvalue()))).convert("L")
        assert abs(out.getpixel((30, 30)) - ink) < 15
        assert abs(out.getpixel((5, 5)) - out.getpixel((30, 30))) > 80

    def test_resize_invalid_bytes_tries_sips_fallback(self):
        with patch('assistant.tools.ImageAnalyzer.convert_via_sips', return_value=None) as mock_sips:
            result = resize_to_jpeg(b"not an image")
        mock_sips.assert_called_once_with(b"not an image")
        assert result is None

    def test_resize_sips_fallback_success(self):
        buf = io.BytesIO()
        Image.new("RGB", (200, 200), color=(0, 255, 0)).save(buf, format="JPEG")
        with patch('assistant.tools.ImageAnalyzer.convert_via_sips', return_value=buf.getvalue()):
            result = resize_to_jpeg(b"heic binary data")
        assert result is not None
        assert Image.open(io.BytesIO(result)).format == "JPEG"


@pytest.mark.unit
class TestConvertViaSips:
    def test_success(self):
        jpeg_data = b"\xff\xd8\xff\xe0converted_jpeg"

        def fake_sips(cmd, **kwargs):
            with open(cmd[-1], 'wb') as f:
                f.write(jpeg_data)

        with patch('assistant.tools.ImageAnalyzer.subprocess.run', side_effect=fake_sips):
            assert convert_via_sips(b"heic raw bytes") == jpeg_data

    def test_failure_returns_none(self):
        with patch('assistant.tools.ImageAnalyzer.subprocess.run',
                   side_effect=Exception("sips not found")):
            assert convert_via_sips(b"some bytes") is None

    def test_cleans_up_temp_files(self):
        created_paths = []

        def fake_sips(cmd, **kwargs):
            created_paths.append(cmd[4])   # tmp_in
            created_paths.append(cmd[-1])  # tmp_out
            with open(cmd[-1], 'wb') as f:
                f.write(b"\xff\xd8jpeg")

        with patch('assistant.tools.ImageAnalyzer.subprocess.run', side_effect=fake_sips):
            convert_via_sips(b"test data")

        for path in created_paths:
            assert not os.path.exists(path), f"Temp file not cleaned up: {path}"

@pytest.mark.unit
class TestParseResponse:
    def test_parses_valid_json(self):
        response = '{"short_description": "A cat", "detailed_description": "A fluffy cat sitting on a sofa."}'
        result = ImageAnalyzer._parse_response(response)
        assert result["short_description"] == "A cat"
        assert result["detailed_description"] == "A fluffy cat sitting on a sofa."

    def test_parses_json_with_surrounding_text(self):
        response = 'Here is the analysis:\n{"short_description": "A dog", "detailed_description": "A dog playing."}\nDone.'
        assert ImageAnalyzer._parse_response(response)["short_description"] == "A dog"

    def test_missing_key_raises(self):
        with pytest.raises(KeyError):
            ImageAnalyzer._parse_response('{"short_description": "A cat"}')

    def test_no_json_raises(self):
        with pytest.raises(ValueError):
            ImageAnalyzer._parse_response("No JSON here at all")


@pytest.mark.unit
class TestStrayQuoteRepair:
    """A local model transcribing visible text drops the backslash on a quote inside a string
    now and then; the parser escapes it instead of failing the whole analysis."""

    def test_inch_mark_inside_a_value_is_escaped(self):
        response = ('{\n  "short_description": "Fabric label",\n'
                    '  "detailed_description": "The label reads Width: 140cm (55") on its second line.",\n'
                    '  "pretty_score": 35\n}')
        result = ImageAnalyzer._parse_response(response)
        assert result["detailed_description"] == 'The label reads Width: 140cm (55") on its second line.'
        assert result["pretty_score"] == 35

    def test_quoted_prose_before_comma_and_colon_stays_inside_the_value(self):
        response = ('{"short_description": "Sign reading "OPEN", lit", '
                    '"detailed_description": "A sign says "OPEN" and "CLOSED": both lit."}')
        result = ImageAnalyzer._parse_response(response)
        assert result["short_description"] == 'Sign reading "OPEN", lit'
        assert result["detailed_description"] == 'A sign says "OPEN" and "CLOSED": both lit.'

    def test_properly_escaped_quotes_are_left_alone(self):
        response = '{"short_description": "A \\"quoted\\" sign", "detailed_description": "Plain."}'
        assert ImageAnalyzer._parse_response(response)["short_description"] == 'A "quoted" sign'

    def test_unrepairable_object_raises_the_original_error(self):
        with pytest.raises(json.JSONDecodeError):
            ImageAnalyzer._parse_response('{"short_description": "A cat", "detailed_description": }')

@pytest.mark.unit
class TestAnalyze:
    def test_successful_analysis(self):
        analyzer, llm = _analyzer_with_reply(
            '{"short_description": "A park", "detailed_description": "People in a park."}')
        result = analyzer.analyze(b"fake_jpeg", context="test")
        assert result is not None
        assert result["short_description"] == "A park"
        llm.think.assert_called_once()

    def test_retries_on_parse_failure(self):
        analyzer, llm = _analyzer_with_reply(
            "not json",
            '{"short_description": "Retry", "detailed_description": "Worked on retry."}')
        result = analyzer.analyze(b"fake_jpeg")
        assert result is not None
        assert result["short_description"] == "Retry"
        assert llm.think.call_count == 2

    def test_returns_none_after_three_parse_failures(self):
        analyzer, llm = _analyzer_with_reply("never valid json")
        assert analyzer.analyze(b"fake_jpeg") is None
        assert llm.think.call_count == 3

    def test_returns_none_on_llm_exception(self):
        analyzer, llm = _analyzer_with_reply("unused")
        llm.think.side_effect = RuntimeError("API down")
        assert analyzer.analyze(b"fake_jpeg") is None

    def test_context_included_in_system_prompt(self):
        analyzer, llm = _analyzer_with_reply('{"short_description": "X", "detailed_description": "Y"}')
        analyzer.analyze(b"fake_jpeg", context="Sent by Alice")
        assert "Sent by Alice" in llm.think.call_args[1]["system_message"]

    def test_empty_context_no_section(self):
        analyzer, llm = _analyzer_with_reply('{"short_description": "X", "detailed_description": "Y"}')
        analyzer.analyze(b"fake_jpeg", context="")
        assert "Additional context" not in llm.think.call_args[1]["system_message"]


@pytest.mark.unit
class TestVisionLlm:
    """The analyzer builds its LLM lazily from the `vision` tier, whatever backend that names:
    the tier is pinned to a fictional one so the developer's config.yaml has no say."""

    @pytest.fixture
    def vision_tier(self):
        mock_llm = Mock()
        with patch('assistant.llm.parse_llm_config', return_value=('ClaudeLLM', 'claude-test-vision')), \
             patch('assistant.llm.ClaudeLLM.ClaudeLLM', return_value=mock_llm) as mock_cls:
            yield mock_cls, mock_llm

    def test_creates_from_vision_tier_on_first_use(self, vision_tier):
        # create_llm wraps the backend in a VaultLLM, so the concrete class sits at .llm.llm.
        from assistant.llm.VaultLLM import VaultLLM
        mock_cls, mock_llm = vision_tier
        llm = ImageAnalyzer().llm
        assert isinstance(llm, VaultLLM) and llm.llm is mock_llm
        mock_cls.assert_called_once_with(model='claude-test-vision')
        mock_llm.initialize.assert_called_once()

    def test_caches_llm_across_calls(self, vision_tier):
        mock_cls, _ = vision_tier
        analyzer = ImageAnalyzer()
        assert analyzer.llm is analyzer.llm
        mock_cls.assert_called_once()

    def test_injected_llm_is_used_as_is(self):
        injected = Mock()
        with patch('assistant.llm.ClaudeLLM.ClaudeLLM') as mock_cls:
            assert ImageAnalyzer(llm=injected).llm is injected
        mock_cls.assert_not_called()
        injected.initialize.assert_not_called()

@pytest.mark.unit
class TestDescribeBytes:
    def test_full_pipeline_success(self):
        analyzer, _ = _analyzer_with_reply('{"short_description": "A cat", "detailed_description": "A fluffy cat."}')
        with patch('assistant.tools.ImageAnalyzer.resize_to_jpeg', return_value=b'jpeg'):
            result = analyzer.describe_bytes(b"raw", context="from Alice")
        assert result == "Image description: [A cat] A fluffy cat."

    def test_context_passed_to_analyze(self):
        analyzer = ImageAnalyzer(llm=Mock())
        with patch('assistant.tools.ImageAnalyzer.resize_to_jpeg', return_value=b'jpeg'), \
             patch.object(analyzer, 'analyze', return_value={
                 "short_description": "X", "detailed_description": "Y"}) as mock_analyze:
            analyzer.describe_bytes(b"raw", context="Sent by Bob")
        assert mock_analyze.call_args[1]["context"] == "Sent by Bob"

    def test_too_large_returns_fallback(self):
        analyzer = ImageAnalyzer(llm=Mock())
        with patch('assistant.tools.ImageAnalyzer.MAX_IMAGE_SIZE', 10):
            assert analyzer.describe_bytes(b"x" * 100) == "[Photo]"
        analyzer.llm.think.assert_not_called()

    def test_resize_failure_returns_fallback(self):
        analyzer = ImageAnalyzer(llm=Mock())
        with patch('assistant.tools.ImageAnalyzer.resize_to_jpeg', return_value=None):
            assert analyzer.describe_bytes(b"raw") == "[Photo]"
        analyzer.llm.think.assert_not_called()

    def test_analyze_returns_none_uses_fallback(self):
        analyzer, _ = _analyzer_with_reply("never valid json")
        with patch('assistant.tools.ImageAnalyzer.resize_to_jpeg', return_value=b'jpeg'):
            assert analyzer.describe_bytes(b"raw") == "[Photo]"

    def test_exception_returns_fallback(self):
        analyzer = ImageAnalyzer(llm=Mock())
        with patch('assistant.tools.ImageAnalyzer.resize_to_jpeg', side_effect=RuntimeError("boom")):
            assert analyzer.describe_bytes(b"raw", fallback="[Image]") == "[Image]"


@pytest.mark.unit
class TestBackgroundAnalysis:
    def test_analysis_is_background_work_and_resends_after_preemption(self):
        from assistant.llm.AbstractLLM import LLMPreempted
        reply = json.dumps({"short_description": "A cat", "detailed_description": "A cat on a chair"})
        analyzer, llm = _analyzer_with_reply(LLMPreempted("evicted"), reply)
        result = analyzer.analyze(b"jpeg")
        assert result["short_description"] == "A cat"
        assert llm.think.call_count == 2
        assert all(call.kwargs["background"] is True for call in llm.think.call_args_list)
