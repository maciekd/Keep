#!/usr/bin/env python3
"""Builds Keep's icons from the artwork beside it (`artwork.png`), the cube rendered on white with its shadow.

    python tools/icon/build.py [--app-icon <AppIcon.png of the Apple app>]

The cube is found in the artwork (the dark and the green pixels; the shadow is neither) and a
square is cut around it that leaves `MARGIN` of the icon's width on either side of it — the cube
is taller than wide, so above and below it is less. The shadow stays and ends where the icon does.

- `res/keep_icon.png`, `res/keep.icns` — the Mac's (`common/Applet`): the cut on the standard
  824-of-1024 rounded square with its drop shadow. macOS frames an icon that is not that shape.
- `--app-icon` — the app's: the cut as it is, a full square without alpha, which iOS masks itself.
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

ARTWORK = Path(__file__).resolve().parent / "artwork.png"
RES = Path(__file__).resolve().parents[2] / "res"
MARGIN = 0.18
CANVAS, BODY, RADIUS = 1024, 824, 185
SHADOW_OFFSET, SHADOW_BLUR, SHADOW_ALPHA = 9, 14, 0.7
ICNS_SIZES = (16, 32, 128, 256, 512)


def cube_box(artwork: Image.Image) -> tuple[int, int, int, int]:
    """The cube's bounding box: its rows and columns hold dark or green pixels, the shadow's only grey."""
    pixels = np.asarray(artwork).astype(int)
    red, green = pixels[:, :, 0], pixels[:, :, 1]
    cube = (pixels.max(axis=2) < 125) | (green > red + 30)
    columns, rows = np.where(cube.sum(axis=0) > 5)[0], np.where(cube.sum(axis=1) > 5)[0]
    return int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1


def cut(artwork: Image.Image) -> Image.Image:
    """The square around the cube, on white where it reaches past the artwork."""
    left, top, right, bottom = cube_box(artwork)
    side = round((right - left) / (1 - 2 * MARGIN))
    x, y = (left + right - side) // 2, (top + bottom - side) // 2
    print(f"cube {right - left}x{bottom - top} px, cut {side} px: "
          f"{MARGIN:.0%} beside it, {(side - (bottom - top)) / 2 / side:.1%} above and below")
    square = Image.new("RGB", (side, side), "white")
    square.paste(artwork, (-x, -y))
    return square


def mac_icon(square: Image.Image) -> Image.Image:
    edge = (CANVAS - BODY) // 2
    scale = 4
    mask = Image.new("L", (CANVAS * scale, CANVAS * scale), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [edge * scale, edge * scale, (edge + BODY) * scale - 1, (edge + BODY) * scale - 1], RADIUS * scale, fill=255)
    mask = mask.resize((CANVAS, CANVAS), Image.LANCZOS)
    shadow = ImageChops.offset(mask, 0, SHADOW_OFFSET).filter(ImageFilter.GaussianBlur(SHADOW_BLUR))
    icon = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    icon.putalpha(shadow.point(lambda value: round(value * SHADOW_ALPHA)))
    body = Image.new("RGBA", (CANVAS, CANVAS), (255, 255, 255, 0))
    body.paste(square.resize((BODY, BODY), Image.LANCZOS), (edge, edge))
    body.putalpha(mask)
    return Image.alpha_composite(icon, body)


def write_icns(icon: Image.Image, path: Path):
    with tempfile.TemporaryDirectory() as folder:
        iconset = Path(folder) / "keep.iconset"
        iconset.mkdir()
        for size in ICNS_SIZES:
            icon.resize((size, size), Image.LANCZOS).save(iconset / f"icon_{size}x{size}.png")
            icon.resize((size * 2, size * 2), Image.LANCZOS).save(iconset / f"icon_{size}x{size}@2x.png")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(path)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--app-icon", type=Path)
    args = parser.parse_args()
    square = cut(Image.open(ARTWORK).convert("RGB"))
    icon = mac_icon(square)
    icon.save(RES / "keep_icon.png", optimize=True)
    write_icns(icon, RES / "keep.icns")
    print(f"wrote {RES / 'keep_icon.png'} and {RES / 'keep.icns'}")
    if args.app_icon:
        square.resize((CANVAS, CANVAS), Image.LANCZOS).save(args.app_icon, optimize=True)
        print(f"wrote {args.app_icon}")


if __name__ == "__main__":
    main()
