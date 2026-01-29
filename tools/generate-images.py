#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/generate-images.py

Rasterizes the glyph SVGs you already generated into PNGs.

Defaults:
- input SVGs:  sketches/character-*.svg
- output PNGs: dist/images/*.png

It auto-discovers all "character-*.svg" files (so it also works for ligatures, punctuation, etc.).

Backends (first available wins):
1) cairosvg (pip install cairosvg)
2) rsvg-convert (librsvg, often available on Linux)
3) inkscape (CLI)

Usage:
  python tools/generate-images.py
  python tools/generate-images.py --size 1024 --pad 90
  python tools/generate-images.py --bg "#ffffff" --fg "#000000"
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple
from io import BytesIO

from PIL import Image


# -----------------------------
# Match your glyph metrics (baseline positioning)
# -----------------------------
H_UNITS = 1000.0
BASE_UNITS = 780.0  # baseline y in glyph coordinate system
DEFAULT_BASELINE_RATIO = BASE_UNITS / H_UNITS  # 0.78


def parse_viewbox(svg_path: Path) -> Tuple[float, float]:
    """
    Returns (viewbox_width, viewbox_height). Falls back to (width, height) attrs.
    """
    root = ET.parse(svg_path).getroot()
    vb = (root.attrib.get("viewBox") or "").strip()

    if vb:
        parts = vb.split()
        if len(parts) == 4:
            try:
                w = float(parts[2])
                h = float(parts[3])
                return (w, h)
            except Exception:
                pass

    # fallback: width/height
    try:
        w = float(root.attrib.get("width", "0"))
        h = float(root.attrib.get("height", "0"))
        if w > 0 and h > 0:
            return (w, h)
    except Exception:
        pass

    # last resort (your generator always uses H=1000, width variable)
    return (700.0, 1000.0)


def hex_to_rgba(s: Optional[str], default: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    if not s:
        return default
    t = s.strip()
    if t.startswith("#"):
        t = t[1:]
    if len(t) == 3:
        t = "".join(ch * 2 for ch in t)
    if len(t) == 6:
        r = int(t[0:2], 16)
        g = int(t[2:4], 16)
        b = int(t[4:6], 16)
        return (r, g, b, 255)
    if len(t) == 8:
        r = int(t[0:2], 16)
        g = int(t[2:4], 16)
        b = int(t[4:6], 16)
        a = int(t[6:8], 16)
        return (r, g, b, a)
    raise ValueError(f"Invalid color: {s!r} (use #rgb, #rrggbb, or #rrggbbaa)")


def _render_with_cairosvg(svg_path: Path, out_w: int, out_h: int, dpi: float) -> Optional[bytes]:
    try:
        import cairosvg  # type: ignore
    except Exception:
        return None

    try:
        return cairosvg.svg2png(
            url=str(svg_path),
            output_width=out_w,
            output_height=out_h,
            dpi=dpi,
        )
    except Exception as e:
        print(f"[warn] cairosvg failed for {svg_path.name}: {e}", file=sys.stderr)
        return None


def _render_with_rsvg(svg_path: Path, out_w: int, out_h: int) -> Optional[bytes]:
    exe = shutil.which("rsvg-convert")
    if not exe:
        return None
    try:
        proc = subprocess.run(
            [exe, "-w", str(out_w), "-h", str(out_h), "-f", "png", str(svg_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            msg = proc.stderr.decode("utf-8", errors="replace").strip()
            print(f"[warn] rsvg-convert failed for {svg_path.name}: {msg}", file=sys.stderr)
            return None
        return proc.stdout
    except Exception as e:
        print(f"[warn] rsvg-convert error for {svg_path.name}: {e}", file=sys.stderr)
        return None


def _render_with_inkscape(svg_path: Path, out_w: int, out_h: int) -> Optional[bytes]:
    exe = shutil.which("inkscape")
    if not exe:
        return None

    # Inkscape usually writes to file; do that via temp file.
    try:
        with tempfile.TemporaryDirectory() as td:
            out_png = Path(td) / "out.png"
            cmd = [
                exe,
                str(svg_path),
                "--export-type=png",
                f"--export-filename={out_png}",
                f"--export-width={out_w}",
                f"--export-height={out_h}",
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if proc.returncode != 0 or not out_png.exists():
                msg = proc.stderr.decode("utf-8", errors="replace").strip()
                print(f"[warn] inkscape failed for {svg_path.name}: {msg}", file=sys.stderr)
                return None
            return out_png.read_bytes()
    except Exception as e:
        print(f"[warn] inkscape error for {svg_path.name}: {e}", file=sys.stderr)
        return None


def render_svg_png_bytes(svg_path: Path, out_w: int, out_h: int, dpi: float) -> bytes:
    # 1) cairosvg
    b = _render_with_cairosvg(svg_path, out_w, out_h, dpi)
    if b:
        return b

    # 2) rsvg-convert
    b = _render_with_rsvg(svg_path, out_w, out_h)
    if b:
        return b

    # 3) inkscape
    b = _render_with_inkscape(svg_path, out_w, out_h)
    if b:
        return b

    raise RuntimeError(
        "No SVG rasterizer backend available.\n"
        "Install ONE of:\n"
        "  - pip install cairosvg\n"
        "  - rsvg-convert (librsvg)\n"
        "  - inkscape (CLI)\n"
    )


def tint_foreground(im: Image.Image, fg_rgba: Tuple[int, int, int, int]) -> Image.Image:
    """
    Your SVGs render black shapes on transparent background.
    This recolors non-transparent pixels to fg_rgba (keeping per-pixel alpha).
    """
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    r, g, b, a = im.split()
    # Use alpha from rendered image; replace RGB with fg
    fr, fg, fb, fa = fg_rgba
    solid = Image.new("RGBA", im.size, (fr, fg, fb, 255))
    solid.putalpha(a)
    return solid


def composite_square(
    glyph_png: Image.Image,
    *,
    square_size: int,
    pad: int,
    baseline_ratio: float,
    vb_h: float,
    rendered_h: int,
    bg_rgba: Tuple[int, int, int, int],
) -> Image.Image:
    """
    Places rendered glyph into a square canvas.
    Aligns baseline consistently using baseline_ratio (default 0.78).
    """
    out = Image.new("RGBA", (square_size, square_size), bg_rgba)

    # center horizontally
    x = (square_size - glyph_png.width) // 2

    # baseline alignment
    # baseline position inside the inner area:
    inner_h = square_size - 2 * pad
    baseline_y = pad + baseline_ratio * inner_h

    # baseline position within the rendered glyph image:
    # glyph coordinate y=BASE_UNITS maps to pixels in rendered glyph height
    baseline_in_glyph_px = (BASE_UNITS / vb_h) * rendered_h

    y = int(round(baseline_y - baseline_in_glyph_px))

    # Paste with alpha
    out.alpha_composite(glyph_png, (x, y))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", type=Path, default=Path("sketches"), help="Input SVG dir")
    ap.add_argument("--out", dest="out_dir", type=Path, default=Path("dist/images"), help="Output PNG dir")
    ap.add_argument("--glob", dest="glob_pat", type=str, default="character-*.svg", help="Which SVG files to convert")
    ap.add_argument("--size", type=int, default=1080, help="Output square size in pixels")
    ap.add_argument("--pad", type=int, default=120, help="Padding inside the square in pixels")
    ap.add_argument("--dpi", type=float, default=96.0, help="DPI hint for rasterizers (used by cairosvg)")
    ap.add_argument("--baseline", type=float, default=DEFAULT_BASELINE_RATIO, help="Baseline ratio inside content box (0..1)")
    ap.add_argument("--bg", type=str, default=None, help="Background color (hex). Default: transparent")
    ap.add_argument("--fg", type=str, default="#111111", help="Glyph color (hex). Default: #111111")
    ap.add_argument("--force", action="store_true", help="Overwrite PNGs even if up-to-date")
    args = ap.parse_args()

    in_dir: Path = args.in_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    bg_rgba = hex_to_rgba(args.bg, (0, 0, 0, 0))
    fg_rgba = hex_to_rgba(args.fg, (17, 17, 17, 255))

    svg_files = sorted(in_dir.glob(args.glob_pat))
    if not svg_files:
        print(f"[err] No SVGs found in {in_dir} matching {args.glob_pat!r}", file=sys.stderr)
        sys.exit(2)

    converted = 0
    skipped = 0

    for svg_path in svg_files:
        out_png = out_dir / (svg_path.stem + ".png")

        if not args.force and out_png.exists():
            try:
                if out_png.stat().st_mtime >= svg_path.stat().st_mtime:
                    skipped += 1
                    continue
            except Exception:
                pass

        vb_w, vb_h = parse_viewbox(svg_path)

        # Fit the glyph viewBox into the square inner area
        inner = max(1, args.size - 2 * args.pad)
        scale = min(inner / vb_w, inner / vb_h)

        render_w = max(1, int(round(vb_w * scale)))
        render_h = max(1, int(round(vb_h * scale)))

        try:
            png_bytes = render_svg_png_bytes(svg_path, render_w, render_h, args.dpi)
        except Exception as e:
            print(f"[warn] Skipping {svg_path.name}: {e}", file=sys.stderr)
            continue

        glyph = Image.open(BytesIO(png_bytes)).convert("RGBA")
        glyph = tint_foreground(glyph, fg_rgba)

        out_im = composite_square(
            glyph,
            square_size=args.size,
            pad=args.pad,
            baseline_ratio=float(args.baseline),
            vb_h=vb_h,
            rendered_h=render_h,
            bg_rgba=bg_rgba,
        )

        out_im.save(out_png)
        converted += 1

    print(f"Wrote {converted} PNGs to: {out_dir}")
    if skipped:
        print(f"Skipped {skipped} (up-to-date).")


if __name__ == "__main__":
    main()
