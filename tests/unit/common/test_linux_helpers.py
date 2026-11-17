"""The Linux branches of the small helpers: a browser and a clipboard only where the server has
a desktop session, HEIC through pillow-heif, a clip's rotation through ffmpeg, and the Mac's
lifecycle pieces (login item, Desktop shortcut, launchd agent, Full Disk Access) doing nothing."""
import io
import subprocess
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from assistant.tools import ImageAnalyzer
from common import Autostart, Browser, Clipboard, DesktopShortcut, FullDiskAccess, Platform, VideoRotation, Watchdog

URL = "https://keep.example/login"
NO_TOOL = lambda *a, **k: pytest.fail("no tool may be run here")   # noqa: E731


@pytest.fixture(autouse=True)
def linux(monkeypatch):
    monkeypatch.setattr(Platform, "current", lambda: Platform.LINUX)
    for name in ("DISPLAY", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(name, raising=False)


def _runs(monkeypatch, stdout=b"", returncode=0):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=b"")
    monkeypatch.setattr(subprocess, "run", run)
    return commands


def _png() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (4, 3), "red").save(out, format="PNG")
    return out.getvalue()


@pytest.mark.unit
class TestBrowser:
    @pytest.mark.parametrize("variable", ["DISPLAY", "WAYLAND_DISPLAY"])
    def test_with_a_desktop_session_the_link_goes_to_xdg_open(self, monkeypatch, variable):
        monkeypatch.setenv(variable, ":0")
        commands = _runs(monkeypatch)
        assert Browser.open_url(URL)
        assert commands == [["xdg-open", URL]]

    def test_a_server_without_a_screen_opens_nothing_and_says_so(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", NO_TOOL)
        assert Browser.open_url(URL) is False

    def test_another_system_opens_nothing_even_with_a_display_variable(self, monkeypatch):
        monkeypatch.setattr(Platform, "current", lambda: "win32")
        monkeypatch.setenv("DISPLAY", ":0")
        monkeypatch.setattr(subprocess, "run", NO_TOOL)
        assert Browser.open_url(URL) is False

    @pytest.mark.parametrize("failure", [
        FileNotFoundError("xdg-open"),
        subprocess.CalledProcessError(3, "xdg-open"),
        subprocess.TimeoutExpired("xdg-open", 5),
    ], ids=["not installed", "no browser to hand it to", "hangs"])
    def test_a_desktop_whose_xdg_open_cannot_is_a_link_not_opened_rather_than_a_crash(self, monkeypatch, failure):
        monkeypatch.setenv("DISPLAY", ":0")

        def run(command, **kwargs):
            assert kwargs.get("check"), "a non-zero exit of xdg-open would be read as an opened link"
            raise failure
        monkeypatch.setattr(subprocess, "run", run)
        assert Browser.open_url(URL) is False


@pytest.mark.unit
class TestClipboard:
    @staticmethod
    def _installed(monkeypatch, *tools):
        monkeypatch.setattr(Clipboard.shutil, "which", lambda name: f"/usr/bin/{name}" if name in tools else None)

    def test_wayland_text_and_picture_come_from_wl_paste(self, monkeypatch):
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        self._installed(monkeypatch, "wl-paste", "xclip")
        commands = _runs(monkeypatch, stdout=_png())
        assert Clipboard.read_image() == (_png(), "image/png")
        assert commands[0][0] == "wl-paste" and "image/png" in commands[0]
        commands = _runs(monkeypatch, stdout="hej då".encode())
        assert Clipboard.read_text() == "hej då"
        assert commands[0][0] == "wl-paste"

    def test_x11_goes_through_xclip(self, monkeypatch):
        monkeypatch.setenv("DISPLAY", ":0")
        self._installed(monkeypatch, "xclip")
        commands = _runs(monkeypatch, stdout=b"plain words")
        assert Clipboard.read_text() == "plain words"
        assert commands[0][0] == "xclip"

    def test_a_display_without_its_tool_reads_nothing(self, monkeypatch):
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        self._installed(monkeypatch)
        monkeypatch.setattr(subprocess, "run", NO_TOOL)
        assert Clipboard.read_text() == "" and Clipboard.read_image() is None

    def test_a_server_without_a_screen_reads_nothing_though_the_tools_are_installed(self, monkeypatch):
        self._installed(monkeypatch, "wl-paste", "xclip")
        monkeypatch.setattr(subprocess, "run", NO_TOOL)
        assert Clipboard.read_text() == "" and Clipboard.read_image() is None

    def test_a_clipboard_without_a_picture_is_no_picture(self, monkeypatch):
        monkeypatch.setenv("DISPLAY", ":0")
        self._installed(monkeypatch, "xclip")
        _runs(monkeypatch, returncode=1)
        assert Clipboard.read_image() is None

    def test_a_wayland_session_without_wl_paste_reads_through_xwaylands_xclip(self, monkeypatch):
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        monkeypatch.setenv("DISPLAY", ":0")
        self._installed(monkeypatch, "xclip")
        commands = _runs(monkeypatch, stdout=b"plain words")
        assert Clipboard.read_text() == "plain words"
        assert commands[0][0] == "xclip"

    def test_xclip_is_asked_for_the_picture_as_png_and_for_the_text_as_unicode(self, monkeypatch):
        """X11 has no `text/plain` target every application offers: asked for it, xclip answers
        an å in Latin-1 or not at all."""
        monkeypatch.setenv("DISPLAY", ":0")
        self._installed(monkeypatch, "xclip")
        commands = _runs(monkeypatch, stdout=_png())
        Clipboard.read_image()
        Clipboard.read_text()
        targets = [command[command.index("-t") + 1] for command in commands]
        assert targets == ["image/png", "UTF8_STRING"]

    def test_what_a_declining_tool_wrote_is_not_the_clipboards_text(self, monkeypatch):
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        self._installed(monkeypatch, "wl-paste")
        _runs(monkeypatch, stdout=b"Nothing is copied", returncode=1)
        assert Clipboard.read_text() == ""

    @pytest.mark.parametrize("failure", [PermissionError("wl-paste"), subprocess.TimeoutExpired("wl-paste", 5)],
                             ids=["cannot be run", "hangs"])
    def test_a_tool_that_cannot_answer_is_an_empty_clipboard_rather_than_a_crash(self, monkeypatch, failure):
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        self._installed(monkeypatch, "wl-paste")

        def run(command, **kwargs):
            raise failure
        monkeypatch.setattr(subprocess, "run", run)
        assert Clipboard.read_text() == "" and Clipboard.read_image() is None

    def test_text_that_is_not_utf8_is_still_pasted(self, monkeypatch):
        monkeypatch.setenv("DISPLAY", ":0")
        self._installed(monkeypatch, "xclip")
        _runs(monkeypatch, stdout=b"caf\xe9 Charlotte")
        assert Clipboard.read_text().endswith(" Charlotte")


@pytest.mark.unit
class TestHeic:
    HEIC = b"\x00\x00\x00\x18ftypheic not a picture Pillow opens"

    @staticmethod
    def _pillow_heif(monkeypatch, picture=None, failure=None):
        def open_heif(stream):
            if failure:
                raise failure
            return SimpleNamespace(to_pillow=lambda: picture)
        monkeypatch.setitem(sys.modules, "pillow_heif", SimpleNamespace(open_heif=open_heif))

    def test_a_picture_pillow_cannot_open_is_converted_through_pillow_heif_not_sips(self, monkeypatch):
        self._pillow_heif(monkeypatch, Image.new("RGBA", (3000, 1500), "blue"))
        monkeypatch.setattr(ImageAnalyzer.subprocess, "run", NO_TOOL)
        jpeg = ImageAnalyzer.resize_to_jpeg(self.HEIC)
        shown = Image.open(io.BytesIO(jpeg))
        assert shown.format == "JPEG" and shown.size == (ImageAnalyzer.MAX_DIMENSION, ImageAnalyzer.MAX_DIMENSION // 2)

    def test_a_picture_it_cannot_decode_is_no_picture(self, monkeypatch):
        self._pillow_heif(monkeypatch, failure=ValueError("not a HEIF file"))
        assert ImageAnalyzer.resize_to_jpeg(self.HEIC) is None

    def test_a_missing_pillow_heif_is_no_picture_rather_than_a_crash(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pillow_heif", None)
        assert ImageAnalyzer.resize_to_jpeg(self.HEIC) is None

    def test_a_mac_still_converts_through_sips(self, monkeypatch):
        monkeypatch.setattr(Platform, "current", lambda: Platform.MAC)
        monkeypatch.setitem(sys.modules, "pillow_heif", None)
        asked = []
        monkeypatch.setattr(ImageAnalyzer, "convert_via_sips", lambda raw: asked.append(raw))
        assert ImageAnalyzer.resize_to_jpeg(self.HEIC) is None
        assert asked == [self.HEIC]


@pytest.mark.unit
class TestVideoRotation:
    CLIP = b"a portrait clip"

    def test_ffmpeg_re_encodes_the_clip_and_avfoundation_is_never_loaded(self, monkeypatch):
        monkeypatch.setattr(VideoRotation, "_load_frameworks", lambda: pytest.fail("AVFoundation is the Mac's"))
        seen = {}

        def ffmpeg(command, **kwargs):
            source, target = command[command.index("-i") + 1], command[-1]
            seen["program"], seen["source"] = command[0], open(source, "rb").read()
            open(target, "wb").write(b"upright")
            return SimpleNamespace(returncode=0, stderr=b"")
        monkeypatch.setattr(subprocess, "run", ffmpeg)
        assert VideoRotation.bake_rotation(self.CLIP) == b"upright"
        assert seen == {"program": "ffmpeg", "source": self.CLIP}

    @pytest.mark.parametrize("outcome", [
        lambda command, **k: SimpleNamespace(returncode=1, stderr=b"moov atom not found"),
        lambda command, **k: SimpleNamespace(returncode=0, stderr=b""),      # exit 0 and nothing written
        lambda command, **k: (_ for _ in ()).throw(FileNotFoundError("ffmpeg")),
        lambda command, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired("ffmpeg", 1)),
    ], ids=["fails", "writes nothing", "not installed", "too slow"])
    def test_when_ffmpeg_cannot_the_clip_goes_out_as_it_is(self, monkeypatch, outcome):
        monkeypatch.setattr(subprocess, "run", outcome)
        assert VideoRotation.bake_rotation(self.CLIP) is None


@pytest.mark.unit
class TestTheMacsLifecyclePieces:
    def test_no_login_item_is_registered_and_the_caller_hears_why(self, tmp_path):
        failure = Autostart.enable(tmp_path / "Keep Autostart.app", tmp_path / "autostart.sh", run=NO_TOOL)
        assert failure and "Mac" in failure
        assert not (tmp_path / "Keep Autostart.app").exists()

    def test_no_desktop_shortcut_is_built(self, tmp_path):
        assert DesktopShortcut.ensure(desktop=tmp_path, repo=tmp_path, run=NO_TOOL) is False
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize("enabled", [True, False])
    def test_the_watchdog_has_nothing_to_bring_in_line(self, tmp_path, enabled):
        plist = tmp_path / "se.keep.watchdog.plist"
        assert Watchdog.sync(enabled, tmp_path / "Keep Autostart.app", tmp_path / "autostart.sh", plist, run=NO_TOOL) is None
        assert not plist.exists()

    def test_full_disk_access_is_neither_granted_nor_observable_nor_opened(self, tmp_path):
        chat_db = tmp_path / "chat.db"
        chat_db.write_bytes(b"")          # even where a file of that name exists
        assert FullDiskAccess.granted(chat_db) is False and FullDiskAccess.observable(chat_db) is False
        assert "Mac" in FullDiskAccess.open_settings(run=NO_TOOL)
