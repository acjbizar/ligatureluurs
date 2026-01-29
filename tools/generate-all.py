#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/generate-all.py

Runs all project generators in a sensible order:

1) tools/generate-svg.py        -> sketches/character-*.svg
2) tools/generate-sheet.py      -> sketches/sheet.svg (or whatever your sheet script outputs)
3) tools/generate-images.py     -> dist/images/*.png
4) tools/generate-fonts.py      -> dist/fonts/ligatureluurs.{ttf,woff,woff2}

This script:
- uses the current Python interpreter (sys.executable)
- forwards any extra CLI args to the underlying scripts when explicitly supported via flags below
- fails fast with clear output

Usage:
  python tools/generate-all.py
  python tools/generate-all.py --stroke 90 --resolution 64 --size 1080
  python tools/generate-all.py --skip-images
  python tools/generate-all.py --skip-fonts
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd: List[str]) -> None:
    rel = cmd[:]
    # Pretty-print with root-relative paths where possible
    pretty = " ".join(rel)
    print(f"\n==> {pretty}")
    res = subprocess.run(cmd, cwd=str(ROOT))
    if res.returncode != 0:
        raise SystemExit(res.returncode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stroke", type=float, default=None, help="Stroke width for generate-svg.py")
    ap.add_argument("--resolution", type=int, default=None, help="Resolution for generate-svg.py")
    ap.add_argument("--svg-out", type=str, default=None, help="Output dir for generate-svg.py (default: sketches)")

    ap.add_argument("--sheet-out", type=str, default=None, help="Output path/dir for sheet generator (if supported)")

    ap.add_argument("--size", type=int, default=None, help="PNG square size for generate-images.py")
    ap.add_argument("--pad", type=int, default=None, help="PNG padding for generate-images.py")
    ap.add_argument("--skip-images", action="store_true")
    ap.add_argument("--skip-fonts", action="store_true")
    ap.add_argument("--skip-sheet", action="store_true")

    ap.add_argument("--font-formats", nargs="+", default=None, choices=["ttf", "woff", "woff2"],
                    help="Formats for generate-fonts.py (default: script default)")
    ap.add_argument("--font-family", type=str, default=None, help="Font family name")
    ap.add_argument("--font-style", type=str, default=None, help="Font style name")
    ap.add_argument("--font-filename", type=str, default=None, help="Base output filename (no ext)")

    args = ap.parse_args()

    tools = ROOT / "tools"
    scripts = {
        "svg": tools / "generate-svg.py",
        "sheet": tools / "generate-sheet.py",
        "images": tools / "generate-images.py",
        "fonts": tools / "generate-fonts.py",
    }

    # --- 1) SVG glyphs ---
    svg_cmd = [PY, str(scripts["svg"])]
    if args.svg_out:
        svg_cmd += ["--out", args.svg_out]
    if args.stroke is not None:
        svg_cmd += ["--stroke", str(args.stroke)]
    if args.resolution is not None:
        svg_cmd += ["--resolution", str(args.resolution)]
    run(svg_cmd)

    # --- 2) Sheet ---
    if not args.skip_sheet and scripts["sheet"].exists():
        sheet_cmd = [PY, str(scripts["sheet"])]
        # Only pass --out if your sheet script supports it; keep it optional
        if args.sheet_out:
            sheet_cmd += ["--out", args.sheet_out]
        run(sheet_cmd)
    else:
        if args.skip_sheet:
            print("\n==> (skipping sheet)")
        else:
            print("\n==> (sheet script not found: tools/generate-sheet.py)")

    # --- 3) PNG images ---
    if not args.skip_images and scripts["images"].exists():
        img_cmd = [PY, str(scripts["images"])]
        if args.size is not None:
            img_cmd += ["--size", str(args.size)]
        if args.pad is not None:
            img_cmd += ["--pad", str(args.pad)]
        run(img_cmd)
    else:
        if args.skip_images:
            print("\n==> (skipping images)")
        else:
            print("\n==> (images script not found: tools/generate-images.py)")

    # --- 4) Fonts ---
    if not args.skip_fonts and scripts["fonts"].exists():
        font_cmd = [PY, str(scripts["fonts"])]
        if args.font_formats:
            font_cmd += ["--formats", *args.font_formats]
        if args.font_family:
            font_cmd += ["--family", args.font_family]
        if args.font_style:
            font_cmd += ["--style", args.font_style]
        if args.font_filename:
            font_cmd += ["--filename", args.font_filename]
        run(font_cmd)
    else:
        if args.skip_fonts:
            print("\n==> (skipping fonts)")
        else:
            print("\n==> (fonts script not found: tools/generate-fonts.py)")

    print("\n✅ All done.")


if __name__ == "__main__":
    main()
