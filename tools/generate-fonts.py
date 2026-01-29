#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/generate-fonts.py

Builds dist/fonts/ligatureluurs.{ext} from the generated SVG glyphs in sketches/.

- Reads:  sketches/character-*.svg   (your new filename format)
- Writes: dist/fonts/ligatureluurs.ttf  (and optionally .woff/.woff2)

Supports ligatures automatically:
If a file encodes multiple codepoints (e.g. character-u0073_u0074.svg),
it will be added as a ligature in GSUB 'liga' (sub s t by lig...).

Requirements:
  pip install fonttools

Optional:
  pip install brotli   # for .woff2 output (if not installed, woff2 will be skipped)

Usage:
  python tools/generate-fonts.py
  python tools/generate-fonts.py --formats ttf woff woff2
  python tools/generate-fonts.py --family Ligatureluurs --style Regular
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.feaLib.builder import addOpenTypeFeaturesFromString


# -----------------------------
# Must match your glyph SVG coordinate system
# (y grows down, baseline is at y=BASE_UNITS)
# -----------------------------
@dataclass(frozen=True)
class Metrics:
    UPM: int = 1000
    H: int = 1000
    BASE: float = 780.0
    CAP_TOP: float = 40.0
    XH: float = 440.0  # baseline -> xheight

    @property
    def X_TOP(self) -> float:
        return self.BASE - self.XH

    @property
    def DESC_END(self) -> float:
        return float(self.H - 30)


M = Metrics()


# -----------------------------
# Filename parsing (new format)
# character-u0061_u0062.svg  => [0x61, 0x62]
# -----------------------------
FILENAME_RE = re.compile(r"^character-(?P<codes>u[0-9a-f]+(?:_u[0-9a-f]+)*)\.svg$")


def parse_codepoints_from_filename(name: str) -> Optional[List[int]]:
    m = FILENAME_RE.match(name)
    if not m:
        return None
    codes = m.group("codes").split("_")
    cps: List[int] = []
    for c in codes:
        if not c.startswith("u"):
            return None
        try:
            cps.append(int(c[1:], 16))
        except Exception:
            return None
    return cps


def glyph_name_for_codepoint(cp: int) -> str:
    # Conventional-ish naming
    if cp <= 0xFFFF:
        return f"uni{cp:04X}"
    return f"u{cp:06X}"


def ligature_name(cps: List[int]) -> str:
    return "lig_" + "_".join(f"{cp:04X}" if cp <= 0xFFFF else f"{cp:06X}" for cp in cps)


def parse_svg_viewbox_and_path(svg_path: Path) -> Tuple[float, float, str]:
    """
    Returns (vb_w, vb_h, d). Raises on parse errors.
    """
    root = ET.parse(svg_path).getroot()
    vb = (root.attrib.get("viewBox") or "").strip()
    if not vb:
        raise ValueError(f"{svg_path.name}: missing viewBox")
    parts = vb.split()
    if len(parts) != 4:
        raise ValueError(f"{svg_path.name}: invalid viewBox {vb!r}")
    vb_w = float(parts[2])
    vb_h = float(parts[3])

    ns = {"svg": "http://www.w3.org/2000/svg"}
    p = root.find(".//svg:path", ns)
    if p is None:
        p = root.find(".//path")
    if p is None:
        return (vb_w, vb_h, "")
    d = p.attrib.get("d", "") or ""
    return (vb_w, vb_h, d)


# -----------------------------
# SVG path parsing
# Your generator emits only absolute M/L/Z with numbers separated by spaces.
# -----------------------------
TOKEN_RE = re.compile(r"[MLZmlz]|-?\d+(?:\.\d+)?")

def parse_svg_d_to_contours(d: str) -> List[List[Tuple[float, float]]]:
    """
    Returns list of contours, each a list of points (x,y) without duplicated closing point.
    Supports M/L/Z (absolute) and basic implicit lineto after moveto.
    """
    d = d.replace(",", " ")
    tokens = TOKEN_RE.findall(d)
    i = 0
    cmd: Optional[str] = None

    contours: List[List[Tuple[float, float]]] = []
    cur: List[Tuple[float, float]] = []

    def flush_close() -> None:
        nonlocal cur
        if not cur:
            return
        # remove duplicated last==first if present
        if len(cur) >= 2 and cur[0] == cur[-1]:
            cur = cur[:-1]
        if len(cur) >= 2:
            contours.append(cur)
        cur = []

    def read_pair() -> Tuple[float, float]:
        nonlocal i
        if i + 1 >= len(tokens):
            raise ValueError("Unexpected end of path data")
        x = float(tokens[i]); y = float(tokens[i + 1])
        i += 2
        return (x, y)

    while i < len(tokens):
        t = tokens[i]
        if t.isalpha():
            cmd = t
            i += 1
            if cmd in ("Z", "z"):
                flush_close()
            continue

        if cmd is None:
            raise ValueError("Path data missing command")

        if cmd in ("M", "m"):
            # absolute only expected; treat like 'M'
            x, y = read_pair()
            flush_close()
            cur = [(x, y)]
            # Any subsequent coordinate pairs without a new command are implicit 'L'
            cmd = "L"
            continue

        if cmd in ("L", "l"):
            x, y = read_pair()
            cur.append((x, y))
            continue

        if cmd in ("Z", "z"):
            flush_close()
            continue

        raise ValueError(f"Unsupported SVG path command: {cmd}")

    # if no explicit Z, still flush
    flush_close()
    return contours


def svg_contours_to_ttglyph(contours: List[List[Tuple[float, float]]], baseline: float) -> Tuple[object, Tuple[int,int,int,int]]:
    """
    Converts contours to TTGlyph (glyf) using FontTools.
    Also returns bbox (xMin,yMin,xMax,yMax) in font coords.
    """
    pen = TTGlyphPen(None)

    xmins: List[int] = []
    ymins: List[int] = []
    xmaxs: List[int] = []
    ymaxs: List[int] = []

    for pts in contours:
        if len(pts) < 2:
            continue

        # transform SVG (y down) -> font (y up), with baseline at y=0
        def tx(p: Tuple[float, float]) -> Tuple[int, int]:
            x, y = p
            xf = int(round(x))
            yf = int(round(baseline - y))
            return (xf, yf)

        p0 = tx(pts[0])
        pen.moveTo(p0)

        xmins.append(p0[0]); xmaxs.append(p0[0])
        ymins.append(p0[1]); ymaxs.append(p0[1])

        for p in pts[1:]:
            q = tx(p)
            pen.lineTo(q)
            xmins.append(q[0]); xmaxs.append(q[0])
            ymins.append(q[1]); ymaxs.append(q[1])

        pen.closePath()

    glyph = pen.glyph()

    if xmins:
        bbox = (min(xmins), min(ymins), max(xmaxs), max(ymaxs))
    else:
        bbox = (0, 0, 0, 0)

    return glyph, bbox


def mean_int(xs: Iterable[int]) -> int:
    xs = list(xs)
    return int(round(sum(xs) / max(1, len(xs))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", type=Path, default=Path("sketches"))
    ap.add_argument("--out", dest="out_dir", type=Path, default=Path("dist/fonts"))
    ap.add_argument("--family", type=str, default="Ligatureluurs")
    ap.add_argument("--style", type=str, default="Regular")
    ap.add_argument("--version", type=str, default="1.0")
    ap.add_argument("--formats", nargs="+", default=["ttf", "woff", "woff2"], choices=["ttf", "woff", "woff2"])
    ap.add_argument("--filename", type=str, default="ligatureluurs", help="Base filename without extension")
    ap.add_argument("--space-width", type=int, default=300)
    ap.add_argument("--baseline", type=float, default=M.BASE, help="SVG baseline y to map to font y=0")
    ap.add_argument("--linegap", type=int, default=200)
    args = ap.parse_args()

    in_dir: Path = args.in_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    svg_files = sorted(in_dir.glob("character-*.svg"))
    if not svg_files:
        print(f"[err] No glyph SVGs found in {in_dir}/character-*.svg", file=sys.stderr)
        sys.exit(2)

    # Gather glyphs
    glyph_order: List[str] = [".notdef", "space"]
    glyf: Dict[str, object] = {}
    hmtx: Dict[str, Tuple[int, int]] = {}
    cmap: Dict[int, str] = {}  # single codepoint glyphs
    ligatures: List[Tuple[List[str], str]] = []  # ([componentGlyphNames], ligGlyphName)

    global_xmin = 0
    global_ymin = 0
    global_xmax = 0
    global_ymax = 0
    have_bbox = False

    widths_for_avg: List[int] = []

    # .notdef
    empty_pen = TTGlyphPen(None)
    glyf[".notdef"] = empty_pen.glyph()
    hmtx[".notdef"] = (int(M.UPM * 0.6), 0)

    # space
    space_pen = TTGlyphPen(None)
    glyf["space"] = space_pen.glyph()
    hmtx["space"] = (int(args.space_width), 0)
    cmap[0x20] = "space"

    # Load all SVG glyphs
    for svg_path in svg_files:
        cps = parse_codepoints_from_filename(svg_path.name)
        if not cps:
            continue

        vb_w, vb_h, d = parse_svg_viewbox_and_path(svg_path)
        adv_w = int(round(vb_w))
        widths_for_avg.append(adv_w)

        if len(cps) == 1:
            gname = glyph_name_for_codepoint(cps[0])
        else:
            gname = ligature_name(cps)

        contours: List[List[Tuple[float, float]]] = []
        if d.strip():
            try:
                contours = parse_svg_d_to_contours(d)
            except Exception as e:
                print(f"[warn] {svg_path.name}: failed to parse path; using empty glyph ({e})", file=sys.stderr)
                contours = []

        glyph, bbox = svg_contours_to_ttglyph(contours, baseline=float(args.baseline))
        glyf[gname] = glyph
        hmtx[gname] = (adv_w, 0)

        if gname not in glyph_order:
            glyph_order.append(gname)

        if len(cps) == 1:
            cmap[cps[0]] = gname
        else:
            comps = [glyph_name_for_codepoint(cp) for cp in cps]
            ligatures.append((comps, gname))

        xMin, yMin, xMax, yMax = bbox
        if not have_bbox:
            global_xmin, global_ymin, global_xmax, global_ymax = xMin, yMin, xMax, yMax
            have_bbox = True
        else:
            global_xmin = min(global_xmin, xMin)
            global_ymin = min(global_ymin, yMin)
            global_xmax = max(global_xmax, xMax)
            global_ymax = max(global_ymax, yMax)

    # Reasonable vertical metrics (include overshoots from stroke)
    asc = max(int(M.BASE - M.CAP_TOP), global_ymax)
    desc = min(-int(M.DESC_END - M.BASE), global_ymin)  # negative
    linegap = int(args.linegap)

    # Build font
    fb = FontBuilder(M.UPM, isTTF=True)
    fb.setupGlyphOrder(glyph_order)
    fb.setupGlyf(glyf)
    fb.setupHorizontalMetrics(hmtx)
    fb.setupHorizontalHeader(ascent=asc, descent=desc, lineGap=linegap)
    fb.setupMaxp()
    fb.setupPost(formatType=3.0)

    # head needs bbox
    fb.setupHead(
        unitsPerEm=M.UPM,
        fontRevision=float(args.version),
        xMin=int(global_xmin),
        yMin=int(global_ymin),
        xMax=int(global_xmax),
        yMax=int(global_ymax),
    )

    # OS/2: keep it simple but sensible
    xheight = int(round(M.XH))                  # baseline->xheight (y up)
    capheight = int(round(M.BASE - M.CAP_TOP))  # baseline->cap
    avg_w = mean_int(widths_for_avg) if widths_for_avg else int(M.UPM * 0.5)

    # cmap
    fb.setupCharacterMap(cmap)

    fb.setupOS2(
        sTypoAscender=asc,
        sTypoDescender=desc,
        sTypoLineGap=linegap,
        usWinAscent=max(0, asc),
        usWinDescent=max(0, -desc),
        sxHeight=xheight,
        sCapHeight=capheight,
        xAvgCharWidth=avg_w,
        usWeightClass=400,
        usWidthClass=5,
    )

    # name table
    family = args.family
    style = args.style
    full_name = f"{family} {style}".strip()
    ps_name = re.sub(r"[^A-Za-z0-9-]", "", full_name.replace(" ", "-"))[:63] or "Ligatureluurs-Regular"

    fb.setupNameTable(
        {
            "familyName": family,
            "styleName": style,
            "fullName": full_name,
            "psName": ps_name,
            "version": f"Version {args.version}",
        }
    )

    font = fb.font

    # Add liga substitutions for multi-codepoint glyphs
    if ligatures:
        # Only include ligatures whose components actually exist
        fea_lines = [
            "languagesystem DFLT dflt;",
            "languagesystem latn dflt;",
            "",
            "feature liga {",
        ]
        added = 0
        for comps, lig in ligatures:
            if all(c in glyf for c in comps) and lig in glyf:
                seq = " ".join(comps)
                fea_lines.append(f"  sub {seq} by {lig};")
                added += 1
        fea_lines.append("} liga;")
        fea = "\n".join(fea_lines) + "\n"

        if added:
            addOpenTypeFeaturesFromString(font, fea)

    # Write formats
    base = out_dir / args.filename
    wrote_any = False

    def save_with_flavor(ext: str, flavor: Optional[str]) -> None:
        nonlocal wrote_any
        out_path = base.with_suffix(f".{ext}")
        try:
            font.flavor = flavor
            font.save(out_path)
            print(f"Wrote: {out_path}")
            wrote_any = True
        except Exception as e:
            print(f"[warn] Could not write {out_path.name}: {e}", file=sys.stderr)

    fmts = [f.lower() for f in args.formats]
    if "ttf" in fmts:
        save_with_flavor("ttf", None)
    if "woff" in fmts:
        save_with_flavor("woff", "woff")
    if "woff2" in fmts:
        save_with_flavor("woff2", "woff2")

    if not wrote_any:
        sys.exit(1)


if __name__ == "__main__":
    main()
