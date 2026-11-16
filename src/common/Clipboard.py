"""The macOS clipboard as the console's paste source: the picture on it, fitted for an LLM,
or else its text. Read through the system's own tools — `osascript` for the picture (as the
PNG class the clipboard converts any picture to, or the file a Finder copy put there) and
`pbpaste` for the text — so no AppKit runs in the server process. A terminal only ever
delivers text on its stdin, so a picture pasted into the TUI has to be fetched like this,
on the key, from the clipboard of the machine the server runs on.

On Linux the same two reads go through `wl-paste` (Wayland) or `xclip` (X11), whichever is
installed, and only while the process has a desktop session; a server without one has no
clipboard and the paste comes back empty.
"""
import io
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from PIL import Image

from common import Platform
from common.logging import error, info, trace

MAX_EDGE = 1568          # the long edge Claude resamples a picture to anyway
MAX_PNG_BYTES = 4 << 20  # a PNG this big is a photo, not a screenshot: it goes out as JPEG
JPEG_QUALITY = 85
TIMEOUT = 5              # seconds an osascript call may take before the paste is given up
_PNG_HEX = re.compile(rb"\xc2\xabdata PNGf([0-9A-Fa-f]+)\xc2\xbb")   # «data PNGf…», UTF-8


def read_image() -> Optional[tuple[bytes, str]]:
    """The picture on the clipboard as (bytes, mime type) — PNG with its long edge at most
    MAX_EDGE, or JPEG when the PNG would be bigger than MAX_PNG_BYTES — or None when the
    clipboard holds no picture."""
    raw = _clipboard_picture()
    if raw is None:
        return None
    try:
        return fit(raw)
    except Exception as e:
        error(f"Clipboard: the picture does not decode ({e}), ignored")
        return None


def read_text() -> str:
    """The clipboard's text, "" when it has none (or the system cannot say)."""
    if not Platform.is_mac():
        return _linux_clipboard("text/plain").decode("utf-8", errors="replace")
    try:
        out = subprocess.run(["pbpaste"], capture_output=True, timeout=TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as e:
        error(f"Clipboard: pbpaste failed: {e}")
        return ""
    return out.stdout.decode("utf-8", errors="replace")


def fit(raw: bytes) -> tuple[bytes, str]:
    """`raw` as the bytes to send: unchanged when it is a small enough PNG, else re-encoded
    with its long edge at MAX_EDGE — PNG for a screenshot's crisp text, JPEG for a picture
    too big for that."""
    image = Image.open(io.BytesIO(raw))
    image.load()
    width, height = image.size
    scale = MAX_EDGE / max(width, height)
    if scale >= 1 and image.format == "PNG" and len(raw) <= MAX_PNG_BYTES:
        trace(f"Clipboard: PNG {width}×{height}, {len(raw) / 1024:.0f} kB kept as it is")
        return raw, "image/png"
    if scale < 1:
        image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.LANCZOS)
    out = io.BytesIO()
    image.save(out, format="PNG")
    png = out.getvalue()
    if len(png) <= MAX_PNG_BYTES:
        info(f"Clipboard: picture {width}×{height} sent as PNG {image.size[0]}×{image.size[1]}, {len(png) / 1024:.0f} kB")
        return png, "image/png"
    out = io.BytesIO()
    image.convert("RGB").save(out, format="JPEG", quality=JPEG_QUALITY)
    jpeg = out.getvalue()
    info(f"Clipboard: picture {width}×{height} sent as JPEG {image.size[0]}×{image.size[1]}, "
         f"{len(jpeg) / 1024:.0f} kB (PNG would be {len(png) / 1024:.0f} kB)")
    return jpeg, "image/jpeg"


def _clipboard_picture() -> Optional[bytes]:
    if not Platform.is_mac():
        data = _linux_clipboard("image/png")
        if not data:
            trace("Clipboard: no picture on the clipboard")
            return None
        info(f"Clipboard: picture read ({len(data) / 1024:.0f} kB PNG)")
        return data
    # A screenshot taken to the clipboard, a picture copied in a browser or Preview: the
    # clipboard renders any picture type as PNG on request.
    out = _osascript("get the clipboard as «class PNGf»")
    match = _PNG_HEX.search(out) if out else None
    if match:
        data = bytes.fromhex(match.group(1).decode("ascii"))
        info(f"Clipboard: picture read ({len(data) / 1024:.0f} kB PNG)")
        return data
    # A file copied in the Finder: the clipboard holds its URL.
    path = _osascript("POSIX path of (the clipboard as «class furl»)")
    if path:
        file = Path(path.decode("utf-8", errors="replace").strip())
        try:
            data = file.read_bytes()
            Image.open(io.BytesIO(data)).verify()
        except Exception as e:
            trace(f"Clipboard: the copied file {file.name} is not a picture ({e})")
            return None
        info(f"Clipboard: picture file read ({file.name}, {len(data) / 1024:.0f} kB)")
        return data
    trace("Clipboard: no picture on the clipboard")
    return None


def _linux_tool(mime: str) -> Optional[list[str]]:
    """The command that writes the clipboard's *mime* rendering to stdout, None without a
    desktop session or a tool for it."""
    if Platform.current() != Platform.LINUX:
        return None
    if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-paste"):
        return ["wl-paste", "--no-newline", "--type", mime]
    if os.environ.get("DISPLAY") and shutil.which("xclip"):
        return ["xclip", "-selection", "clipboard", "-o", "-t", "UTF8_STRING" if mime == "text/plain" else mime]
    return None


def _linux_clipboard(mime: str) -> bytes:
    """The clipboard's content as *mime*, b"" when it holds none, or the server has no desktop."""
    command = _linux_tool(mime)
    if command is None:
        trace(f"Clipboard: no desktop session or clipboard tool (wl-paste, xclip) on {Platform.label()}")
        return b""
    try:
        out = subprocess.run(command, capture_output=True, timeout=TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as e:
        error(f"Clipboard: {command[0]} failed: {e}")
        return b""
    if out.returncode != 0:
        trace(f"Clipboard: {command[0]} declined: {out.stderr.decode('utf-8', errors='replace').strip()}")
        return b""
    return out.stdout


def _osascript(script: str) -> bytes:
    """The script's stdout, b"" when it fails (the clipboard holds nothing of that class)."""
    try:
        out = subprocess.run(["osascript", "-e", script], capture_output=True, timeout=TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as e:
        error(f"Clipboard: osascript failed: {e}")
        return b""
    if out.returncode != 0:
        trace(f"Clipboard: osascript declined: {out.stderr.decode('utf-8', errors='replace').strip()}")
        return b""
    return out.stdout
