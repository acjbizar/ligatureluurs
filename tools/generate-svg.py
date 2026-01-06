#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, MultiPolygon
from shapely.ops import unary_union

Geom = Union[Polygon, MultiPolygon]


# -------------------------
# Metrics (SVG y-down)
# -------------------------

@dataclass(frozen=True)
class Metrics:
    H: int = 1000
    BASE: float = 780.0
    CAP_TOP: float = 40.0
    CAP_W: float = 700.0
    LC_W: float = 600.0
    XH: float = 520.0  # baseline -> xheight distance

    @property
    def CAP_MID(self) -> float:
        return (self.CAP_TOP + self.BASE) / 2.0

    @property
    def X_TOP(self) -> float:
        return self.BASE - self.XH

    @property
    def X_MID(self) -> float:
        return (self.X_TOP + self.BASE) / 2.0

    @property
    def DESC(self) -> float:
        return float(self.H)


# -------------------------
# Monoline builder (centerline strokes -> outlines)
# -------------------------

class Mono:
    def __init__(self, stroke: float, resolution: int = 64):
        self.stroke = float(stroke)
        self.r = self.stroke / 2.0
        self.res = int(resolution)

    def _fix(self, g: Geom) -> Geom:
        try:
            gg = g.buffer(0)
            return gg if not gg.is_empty else g
        except Exception:
            return g

    def union(self, *parts: Geom) -> Geom:
        ps = [p for p in parts if p is not None and not p.is_empty]
        if not ps:
            return Polygon()
        return self._fix(unary_union(ps))

    def line(self, pts: List[Tuple[float, float]]) -> Geom:
        return self._fix(LineString(pts).buffer(self.r, cap_style=1, join_style=1, resolution=self.res))

    def vline(self, x: float, y0: float, y1: float) -> Geom:
        return self.line([(x, y0), (x, y1)])

    def hline(self, x0: float, x1: float, y: float) -> Geom:
        return self.line([(x0, y), (x1, y)])

    def arc(self, cx: float, cy: float, r: float, deg0: float, deg1: float, steps: int = 140) -> Geom:
        # degrees: 0=right, 90=down, 180=left, 270=up (SVG y-down convention)
        d0 = deg0 % 360.0
        d1 = deg1 % 360.0
        if d1 <= d0:
            d1 += 360.0
        pts: List[Tuple[float, float]] = []
        for i in range(steps):
            a = math.radians(d0 + (d1 - d0) * (i / (steps - 1)))
            pts.append((cx + math.cos(a) * r, cy + math.sin(a) * r))
        return self.line(pts)

    def ellipse_arc(self, cx: float, cy: float, rx: float, ry: float,
                    deg0: float, deg1: float, steps: int = 240) -> Geom:
        d0 = deg0 % 360.0
        d1 = deg1 % 360.0
        if d1 <= d0:
            d1 += 360.0

        pts: List[Tuple[float, float]] = []
        for i in range(steps):
            a = math.radians(d0 + (d1 - d0) * (i / (steps - 1)))
            pts.append((cx + math.cos(a) * rx, cy + math.sin(a) * ry))
        return self.line(pts)


    def ellipse_stroke(self, cx: float, cy: float, rx: float, ry: float) -> Geom:
        base = Point(cx, cy).buffer(1.0, resolution=self.res)
        ell = affinity.scale(base, xfact=rx, yfact=ry, origin=(cx, cy))
        return self._fix(ell.boundary.buffer(self.r, cap_style=1, join_style=1, resolution=self.res))

    def dot(self, cx: float, cy: float, radius: float) -> Geom:
        return Point(cx, cy).buffer(radius, resolution=self.res)


# -------------------------
# Smooth S spine (matches source: top terminal right, bottom terminal left)
# -------------------------

def s_spine_points(cx: float, y0: float, y1: float, amp1: float, amp2: float, n: int = 220) -> List[Tuple[float, float]]:
    """
    x(t) = cx + amp1*sin(2πt + π/2) + amp2*sin(πt + π/2)
    - first term gives the classic S lobe (right->left->right)
    - second term biases the end to the left (so bottom terminal is left)
    """
    pts: List[Tuple[float, float]] = []
    for i in range(n):
        t = i / (n - 1)
        y = y0 + (y1 - y0) * t
        x = (
            cx
            + amp1 * math.sin(2.0 * math.pi * t + math.pi / 2.0)
            + amp2 * math.sin(math.pi * t + math.pi / 2.0)
        )
        pts.append((x, y))
    return pts


# -------------------------
# SVG output
# -------------------------

def fmt(x: float) -> str:
    return f"{x:.3f}"

def codepoint_filename(s: str) -> str:
    cps = [f"U{ord(ch):04X}" for ch in s]
    return "_".join(cps) + ".svg"

def geom_to_svg_path(g: Geom) -> str:
    if g.is_empty:
        return ""
    polys: List[Polygon]
    if isinstance(g, Polygon):
        polys = [g]
    else:
        polys = [p for p in g.geoms if isinstance(p, Polygon)]

    def ring_to_path(coords) -> str:
        pts = list(coords)
        if len(pts) < 2:
            return ""
        return "M " + " L ".join(f"{fmt(x)} {fmt(y)}" for x, y in pts) + " Z"

    parts: List[str] = []
    for p in polys:
        parts.append(ring_to_path(p.exterior.coords))
        for hole in p.interiors:
            parts.append(ring_to_path(hole.coords))
    return " ".join(parts)

def write_svg(out_path: Path, width: float, m: Metrics, g: Geom) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    d = geom_to_svg_path(g)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {fmt(width)} {m.H}">\n'
        f'  <path d="{d}" fill="black" fill-rule="evenodd"/>\n'
        f'</svg>\n'
    )
    out_path.write_text(svg, encoding="utf-8")

def write_preview_html(out_dir: Path, items: List[Tuple[str, str]]) -> None:
    cells = []
    for label, fname in sorted(items, key=lambda t: t[1]):
        cells.append(f"""
<div class="cell">
  <div class="label">{label}</div>
  <div class="file">{fname}</div>
  <img src="{fname}" alt="{label}">
</div>
""".strip())

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Ligatureluurs sketches</title>
<style>
body{{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:16px}}
.cell{{border:1px solid #ddd;border-radius:12px;padding:10px}}
.label{{font-size:13px;font-weight:650;margin-bottom:4px}}
.file{{font-size:11px;opacity:.7;margin-bottom:8px}}
img{{width:100%;height:auto;display:block;background:#fff;border-radius:8px}}
</style>
</head>
<body>
<h1>Ligatureluurs SVG sketches</h1>
<div class="grid">
{chr(10).join(cells)}
</div>
</body>
</html>
"""
    (out_dir / "preview.html").write_text(html, encoding="utf-8")


# -------------------------
# Uppercase
# -------------------------

def build_uppercase(m: Metrics, pen: Mono) -> Dict[str, Tuple[Geom, float]]:
    W = m.CAP_W
    xL, xR = 130.0, W - 130.0
    cx = W / 2.0
    yTop, yBase, yMid = m.CAP_TOP, m.BASE, m.CAP_MID

    # shared cap ellipse (matches B/C/D proportions)
    CAP_RX = (xR - xL) / 2.0          # 220
    CAP_RY = (yBase - yTop) / 2.0     # 370
    CAP_CX = cx
    CAP_CY = yMid

    # C-style open angles (rounded terminals, like your source)
    C_A0, C_A1 = 45.0, 315.0

    glyphs: Dict[str, Tuple[Geom, float]] = {}

    # A (keep your good arch A)
    yArch = 260.0
    rArch = (xR - xL) / 2.0
    yBar = 450.0
    A = pen.union(
        pen.vline(xL, yBase, yArch),
        pen.arc(cx, yArch, rArch, 180.0, 360.0, steps=180),
        pen.vline(xR, yArch, yBase),
        pen.hline(xL, xR, yBar),
    )
    glyphs["A"] = (A, W)

    # B (as before)
    bowl_r = (yMid - yTop) / 2.0
    xFlat = 375.0
    B = pen.union(
        pen.vline(xL, yTop, yBase),
        pen.hline(xL, xFlat, yTop),
        pen.arc(xFlat, yTop + bowl_r, bowl_r, 270.0, 90.0, steps=140),
        pen.hline(xFlat, xL, yMid),
        pen.hline(xL, xFlat, yMid),
        pen.arc(xFlat, yMid + bowl_r, bowl_r, 270.0, 90.0, steps=140),
        pen.hline(xFlat, xL, yBase),
    )
    glyphs["B"] = (B, W)

    # C
    C = pen.ellipse_arc(CAP_CX, CAP_CY, CAP_RX, CAP_RY, C_A0, C_A1, steps=260)
    glyphs["C"] = (C, W)

    # D (FIX: stem + right-half bowl; no “fat rounded rectangle”)
    xJoin = cx  # 350
    D = pen.union(
        pen.vline(xL, yTop, yBase),
        pen.hline(xL, xJoin, yTop),
        pen.ellipse_arc(xJoin, CAP_CY, xR - xJoin, CAP_RY, 270.0, 90.0, steps=220),
        pen.hline(xJoin, xL, yBase),
    )
    glyphs["D"] = (D, W)

    # E/F (unchanged)
    E = pen.union(
        pen.vline(xL, yTop, yBase),
        pen.hline(xL, xR, yTop),
        pen.hline(xL, xR - 50, yMid),
        pen.hline(xL, xR, yBase),
    )
    glyphs["E"] = (E, W)

    F = pen.union(
        pen.vline(xL, yTop, yBase),
        pen.hline(xL, xR, yTop),
        pen.hline(xL, xR - 70, yMid),
    )
    glyphs["F"] = (F, W)

    # G (FIX: “C + short mid-bar”, not a wrapped/closed ellipse)
    # bar stops well before the outer stroke, like your source
    G_bar = pen.hline(CAP_CX - 10.0, CAP_CX + CAP_RX * 0.55, CAP_CY + 10.0)
    G = pen.union(pen.ellipse_arc(CAP_CX, CAP_CY, CAP_RX, CAP_RY, C_A0, C_A1, steps=260), G_bar)
    glyphs["G"] = (G, W)

    # H/I/J etc (keep stable versions)
    H = pen.union(
        pen.vline(xL, yTop, yBase),
        pen.vline(xR, yTop, yBase),
        pen.hline(xL, xR, yMid),
    )
    glyphs["H"] = (H, W)
    glyphs["I"] = (pen.vline(cx, yTop, yBase), W)

    # J (single continuous stroke)
    jx = xR - 120.0
    hook_r = 150.0
    hook_cx = jx - hook_r
    hook_cy = yBase - hook_r
    J = pen.union(
        pen.vline(jx, yTop, hook_cy),
        pen.arc(hook_cx, hook_cy, hook_r, 0.0, 180.0, steps=140),
    )
    glyphs["J"] = (J, W)

    K = pen.union(
        pen.vline(xL, yTop, yBase),
        pen.line([(xL, yMid), (xR, yTop)]),
        pen.line([(xL, yMid), (xR, yBase)]),
    )
    glyphs["K"] = (K, W)

    glyphs["L"] = (pen.union(pen.vline(xL, yTop, yBase), pen.hline(xL, xR, yBase)), W)

    M = pen.union(
        pen.vline(xL, yTop, yBase),
        pen.vline(xR, yTop, yBase),
        pen.line([(xL, yTop), (cx, yMid), (xR, yTop)]),
    )
    glyphs["M"] = (M, W)

    N = pen.union(
        pen.vline(xL, yTop, yBase),
        pen.vline(xR, yTop, yBase),
        pen.line([(xL, yTop), (xR, yBase)]),
    )
    glyphs["N"] = (N, W)

    # O (same cap ellipse)
    O = pen.ellipse_stroke(CAP_CX, CAP_CY, CAP_RX, CAP_RY)
    glyphs["O"] = (O, W)

    # P
    P = pen.union(
        pen.vline(xL, yTop, yBase),
        pen.hline(xL, xFlat, yTop),
        pen.arc(xFlat, yTop + bowl_r, bowl_r, 270.0, 90.0, steps=140),
        pen.hline(xFlat, xL, yMid),
    )
    glyphs["P"] = (P, W)

    # Q (FIX: nicer crossing diagonal tail)
    Q_tail = pen.line([
        (CAP_CX + CAP_RX * 0.18, CAP_CY + CAP_RY * 0.20),
        (CAP_CX + CAP_RX * 0.75, CAP_CY + CAP_RY * 0.85),
    ])
    glyphs["Q"] = (pen.union(O, Q_tail), W)

    # R
    glyphs["R"] = (pen.union(P, pen.line([(xFlat, yMid), (xR, yBase)])), W)

    # S (FIX: smooth snake, not “3”)
    s_y0 = yTop + 70.0
    s_y1 = yBase - 70.0
    s_amp1 = (xR - xL) * 0.18
    s_amp2 = (xR - xL) * 0.26  # >amp1 => bottom ends left
    S = pen.line(s_spine_points(cx, s_y0, s_y1, s_amp1, s_amp2, n=240))
    glyphs["S"] = (S, W)

    # T/U/V/W/X/Y/Z (keep)
    glyphs["T"] = (pen.union(pen.hline(xL, xR, yTop), pen.vline(cx, yTop, yBase)), W)

    u_end = 560.0
    U = pen.union(
        pen.vline(xL, yTop, u_end),
        pen.arc(cx, u_end, (xR - xL) / 2.0, 0.0, 180.0, steps=170),
        pen.vline(xR, u_end, yTop),
    )
    glyphs["U"] = (U, W)

    glyphs["V"] = (pen.line([(xL, yTop), (cx, yBase), (xR, yTop)]), W)
    glyphs["W"] = (pen.line([(xL, yTop), (xL + 120, yBase), (cx, yMid), (xR - 120, yBase), (xR, yTop)]), W)
    glyphs["X"] = (pen.union(pen.line([(xL, yTop), (xR, yBase)]), pen.line([(xR, yTop), (xL, yBase)])), W)
    glyphs["Y"] = (pen.union(pen.line([(xL, yTop), (cx, yMid), (xR, yTop)]), pen.vline(cx, yMid, yBase)), W)
    glyphs["Z"] = (pen.union(pen.hline(xL, xR, yTop), pen.line([(xR, yTop), (xL, yBase)]), pen.hline(xL, xR, yBase)), W)

    # ensure all letters exist
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        glyphs.setdefault(ch, (Polygon(), W))

    return glyphs


# -------------------------
# Lowercase (source-specific logic)
# -------------------------

def build_lowercase(m: Metrics, pen: Mono) -> Dict[str, Tuple[Geom, float]]:
    W = m.LC_W
    xL, xR = 130.0, W - 130.0
    cx = W / 2.0

    yBase, yXTop, yMid = m.BASE, m.X_TOP, m.X_MID
    yAsc = m.CAP_TOP

    # lowercase bowl (tuned to look like your sample)
    RX = (xR - xL) * 0.40
    RY = (yBase - yXTop) * 0.46
    BCX = cx
    BCY = yMid + 10.0

    # open arc angles (like your 'c'/'e')
    A0, A1 = 45.0, 315.0

    # small dot (FIX: not ridiculous)
    dot_r = pen.r * 0.28

    glyphs: Dict[str, Tuple[Geom, float]] = {}

    # a = o + mid bar (matches your source a pretty closely)
    a_bar_y = BCY
    a_bar = pen.hline(BCX - RX * 0.10, BCX + RX * 0.78, a_bar_y)
    glyphs["a"] = (pen.union(pen.ellipse_stroke(BCX, BCY, RX, RY), a_bar), W)

    # b = stem + right-half bowl (NOT a full bowl)
    b_stem_x = xL + 35.0
    b_join_x = BCX + 10.0
    b = pen.union(
        pen.vline(b_stem_x, yAsc, yBase),
        pen.hline(b_stem_x, b_join_x, yXTop),
        pen.ellipse_arc(b_join_x, BCY, xR - b_join_x, RY, 270.0, 90.0, steps=200),
        pen.hline(b_join_x, b_stem_x, yBase),
    )
    glyphs["b"] = (b, W)

    # c = open bowl
    glyphs["c"] = (pen.ellipse_arc(BCX, BCY, RX, RY, A0, A1, steps=220), W)

    # d = left-half bowl + right stem
    d_stem_x = xR - 35.0
    d_join_x = BCX - 10.0
    d = pen.union(
        pen.vline(d_stem_x, yAsc, yBase),
        pen.hline(d_join_x, d_stem_x, yXTop),
        pen.ellipse_arc(d_join_x, BCY, d_join_x - xL, RY, 90.0, 270.0, steps=200),
        pen.hline(d_stem_x, d_join_x, yBase),
    )
    glyphs["d"] = (d, W)

    # e = c + mid bar (like your source e)
    e_bar = pen.hline(BCX - RX * 0.70, BCX + RX * 0.60, BCY)
    glyphs["e"] = (pen.union(glyphs["c"][0], e_bar), W)

    # f = tall stem + short crossbar + tiny hook at top (more like the sample)
    fx = cx - 40.0
    f_cross_y = yXTop + 90.0
    f = pen.union(
        pen.vline(fx, yAsc, yBase),
        pen.hline(fx - 10.0, fx + 235.0, f_cross_y),
        pen.arc(fx + 55.0, yAsc + 150.0, 95.0, 180.0, 270.0, steps=90),
    )
    glyphs["f"] = (f, W)

    # g = bowl + ear + descender that curves into a base stroke (your sample "g")
    g_bowl = pen.ellipse_stroke(BCX + 10.0, BCY - 10.0, RX * 0.92, RY * 0.92)
    g_ear_y = BCY - RY * 0.92
    g_ear = pen.hline(BCX + RX * 0.10, BCX + RX * 0.62, g_ear_y + 10.0)

    desc_y = yBase + 230.0
    # descender: down then curl left into a bottom arc
    g_down = pen.line([
        (BCX + RX * 0.05, BCY + RY * 0.45),
        (BCX + RX * 0.08, desc_y - 70.0),
    ])
    g_bottom = pen.arc(BCX - 20.0, desc_y - 70.0, 160.0, 0.0, 180.0, steps=140)

    glyphs["g"] = (pen.union(g_bowl, g_ear, g_down, g_bottom), W)

    # h = left stem + angled roof + right stem (matches your source h)
    hxL = xL + 35.0
    hxR = xR - 35.0
    h_roof_yL = yXTop
    h_roof_yR = yXTop + 45.0
    h = pen.union(
        pen.vline(hxL, yAsc, yBase),
        pen.line([(hxL, h_roof_yL), (hxR, h_roof_yR)]),
        pen.vline(hxR, h_roof_yR, yBase),
    )
    glyphs["h"] = (h, W)

    # i / j = stem + small dot
    ix = cx
    i_stem = pen.vline(ix, yXTop, yBase)
    i_dot = pen.dot(ix, yXTop - 80.0, dot_r)
    glyphs["i"] = (pen.union(i_stem, i_dot), W)

    j_stem = pen.vline(ix, yXTop, m.DESC - 30.0)
    j_hook = pen.arc(ix - 120.0, m.DESC - 80.0, 120.0, 0.0, 180.0, steps=90)
    j_dot = pen.dot(ix, yXTop - 80.0, dot_r)
    glyphs["j"] = (pen.union(j_stem, j_hook, j_dot), W)

    # keep the rest stable-ish (you can refine later)
    # k/l/m/n/o/p/q/r/s/t/u/v/w/x/y/z fallback
    # (not perfect yet, but avoids total chaos)
    def empty():
        return Polygon()

    for ch in "klmnopqrstuvwxyz":
        glyphs.setdefault(ch, (empty(), W))

    # o can at least be correct
    glyphs["o"] = (pen.ellipse_stroke(BCX, BCY, RX, RY), W)

    return glyphs


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("sketches"))
    ap.add_argument("--stroke", type=float, default=90.0)
    ap.add_argument("--resolution", type=int, default=64)
    args = ap.parse_args()

    m = Metrics()
    pen = Mono(stroke=args.stroke, resolution=args.resolution)

    upper = build_uppercase(m, pen)
    lower = build_lowercase(m, pen)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    preview: List[Tuple[str, str]] = []

    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        g, w = upper[ch]
        fname = codepoint_filename(ch)
        write_svg(out / fname, w, m, g)
        preview.append((ch, fname))

    for ch in "abcdefghijklmnopqrstuvwxyz":
        g, w = lower[ch]
        fname = codepoint_filename(ch)
        write_svg(out / fname, w, m, g)
        preview.append((ch, fname))

    write_preview_html(out, preview)
    print(f"Wrote {len(preview)} SVGs to: {out}")
    print(f"Open: {out / 'preview.html'}")


if __name__ == "__main__":
    main()
