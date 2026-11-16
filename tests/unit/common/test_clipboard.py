"""Clipboard: the osascript and pbpaste calls behind Ctrl+V are faked; `fit` runs on real
pictures."""
import io
import subprocess
from types import SimpleNamespace

import pytest
from PIL import Image

from common import Clipboard, Platform


def png(width: int, height: int, color=(10, 20, 30)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (width, height), color).save(out, format="PNG")
    return out.getvalue()


def noisy_jpeg_source(width: int, height: int) -> bytes:
    """A PNG no encoder compresses well: random pixels."""
    import os
    image = Image.frombytes("RGB", (width, height), os.urandom(width * height * 3))
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


class TestFit:
    def test_a_small_png_is_sent_as_it_is(self):
        raw = png(400, 300)
        assert Clipboard.fit(raw) == (raw, "image/png")

    def test_a_big_picture_is_scaled_to_the_long_edge(self):
        data, mime = Clipboard.fit(png(4000, 1000))
        assert mime == "image/png"
        assert Image.open(io.BytesIO(data)).size == (Clipboard.MAX_EDGE, Clipboard.MAX_EDGE // 4)

    def test_a_jpeg_is_re_encoded_as_png(self):
        out = io.BytesIO()
        Image.new("RGB", (100, 80), (200, 0, 0)).save(out, format="JPEG")
        data, mime = Clipboard.fit(out.getvalue())
        assert mime == "image/png" and data[:4] == b"\x89PNG"

    def test_a_png_too_big_for_the_wire_goes_out_as_jpeg(self, monkeypatch):
        monkeypatch.setattr(Clipboard, "MAX_PNG_BYTES", 2000)
        data, mime = Clipboard.fit(noisy_jpeg_source(200, 200))
        assert mime == "image/jpeg" and data[:2] == b"\xff\xd8"
        assert Image.open(io.BytesIO(data)).size == (200, 200)


def fake_run(answers: dict):
    """A subprocess.run whose stdout is `answers[script]` (b"" with exit 1 when absent)."""
    calls = []

    def run(args, capture_output=True, timeout=None):
        calls.append(args)
        key = args[-1]
        if key in answers:
            return SimpleNamespace(returncode=0, stdout=answers[key], stderr=b"")
        return SimpleNamespace(returncode=1, stdout=b"", stderr=b"Can't make some data into the expected type.")
    run.calls = calls
    return run


class TestReadImage:
    @pytest.fixture(autouse=True)
    def mac(self, monkeypatch):
        monkeypatch.setattr(Platform, "current", lambda: Platform.MAC)

    def test_the_png_class_is_decoded_from_the_hex_osascript_prints(self, monkeypatch):
        raw = png(20, 10)
        monkeypatch.setattr(subprocess, "run",
                            fake_run({"get the clipboard as «class PNGf»": b"\xc2\xabdata PNGf" + raw.hex().upper().encode() + b"\xc2\xbb\n"}))
        assert Clipboard.read_image() == (raw, "image/png")

    def test_a_file_copied_in_the_finder_is_read_from_disk(self, monkeypatch, tmp_path):
        file = tmp_path / "shot.png"
        file.write_bytes(png(20, 10))
        monkeypatch.setattr(subprocess, "run",
                            fake_run({"POSIX path of (the clipboard as «class furl»)": f"{file}\n".encode()}))
        assert Clipboard.read_image() == (file.read_bytes(), "image/png")

    def test_a_copied_file_that_is_no_picture_is_nothing(self, monkeypatch, tmp_path):
        file = tmp_path / "notes.txt"
        file.write_text("hello")
        monkeypatch.setattr(subprocess, "run",
                            fake_run({"POSIX path of (the clipboard as «class furl»)": f"{file}\n".encode()}))
        assert Clipboard.read_image() is None

    def test_text_only_clipboard_has_no_picture_but_has_text(self, monkeypatch):
        run = fake_run({"pbpaste": "hej\n".encode()})
        monkeypatch.setattr(subprocess, "run", run)
        assert Clipboard.read_image() is None
        assert Clipboard.read_text() == "hej\n"

    def test_a_failing_tool_is_nothing_not_an_error(self, monkeypatch):
        def boom(*a, **k):
            raise subprocess.TimeoutExpired("osascript", Clipboard.TIMEOUT)
        monkeypatch.setattr(subprocess, "run", boom)
        assert Clipboard.read_image() is None and Clipboard.read_text() == ""


def test_other_platforms_read_nothing(monkeypatch):
    monkeypatch.setattr(Platform, "current", lambda: Platform.LINUX)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("no tool is run off macOS"))
    assert Clipboard.read_image() is None and Clipboard.read_text() == ""
