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
# Geometry builder (monoline -> outlines)
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
        # degrees: 0=right, 90=down, 180=left, 270=up  (SVG-y-down convention)
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
        # Same angle convention as arc()
        d0 = deg0 % 360.0
        d1 = deg1 % 360.0
        if d1 <= d0:
            d1 += 360.0

        pts: List[Tuple[float, float]] = []
        for i in range(steps):
            a = math.radians(d0 + (d1 - d0) * (i / (steps - 1)))
            pts.append((cx + math.cos(a) * rx, cy + math.sin(a) * ry))
        return self.line(pts)

    def circle_stroke(self, cx: float, cy: float, r: float) -> Geom:
        boundary = Point(cx, cy).buffer(r, resolution=self.res).boundary
        return self._fix(boundary.buffer(self.r, cap_style=1, join_style=1, resolution=self.res))

    def ellipse_stroke(self, cx: float, cy: float, rx: float, ry: float) -> Geom:
        base = Point(cx, cy).buffer(1.0, resolution=self.res)
        ell = affinity.scale(base, xfact=rx, yfact=ry, origin=(cx, cy))
        boundary = ell.boundary
        return self._fix(boundary.buffer(self.r, cap_style=1, join_style=1, resolution=self.res))


# -------------------------
# Bézier sampling (for S / s)
# -------------------------

def cubic_points(p0, p1, p2, p3, steps: int) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for i in range(steps):
        t = i / (steps - 1)
        mt = 1.0 - t
        x = (mt**3)*p0[0] + 3*(mt**2)*t*p1[0] + 3*mt*(t**2)*p2[0] + (t**3)*p3[0]
        y = (mt**3)*p0[1] + 3*(mt**2)*t*p1[1] + 3*mt*(t**2)*p2[1] + (t**3)*p3[1]
        out.append((x, y))
    return out


# -------------------------
# SVG output (filled outlines)
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
# Glyph sets
# -------------------------

def build_uppercase(m: Metrics, pen: Mono) -> Dict[str, Tuple[Geom, float]]:
    W = m.CAP_W
    xL, xR = 130.0, W - 130.0
    cx = W / 2.0
    yTop, yBase, yMid = m.CAP_TOP, m.BASE, m.CAP_MID

    # One shared bowl ellipse for cap-height “round” letters (matches B/D feel)
    CAP_RX = (xR - xL) / 2.0           # 220
    CAP_RY = (yBase - yTop) / 2.0      # 370
    CAP_CX = cx
    CAP_CY = yMid

    glyphs: Dict[str, Tuple[Geom, float]] = {}

    # A (keep from prior good version)
    yArch = 260.0
    rArch = (xR - xL) / 2.0
    yBar = 450.0
    A = pen.union(
        pen.vline(xL, yBase, yArch),
        pen.arc(cx, yArch, rArch, 180.0, 360.0, steps=180),  # top arch (bulges up)
        pen.vline(xR, yArch, yBase),
        pen.hline(xL, xR, yBar),
    )
    glyphs["A"] = (A, W)

    # B (keep)
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

    # C (rounded terminals via open ellipse-arc)
    C = pen.ellipse_arc(CAP_CX, CAP_CY, CAP_RX, CAP_RY, 50.0, 310.0, steps=260)
    glyphs["C"] = (C, W)

    # D (simple but OK)
    D = pen.union(
        pen.vline(xL, yTop, yBase),
        pen.hline(xL, xR - 90, yTop),
        pen.ellipse_arc(xR - 90, yMid, 280.0, CAP_RY, 270.0, 90.0, steps=220),
        pen.hline(xR - 90, xL, yBase),
    )
    glyphs["D"] = (D, W)

    # E/F (unchanged-ish)
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

    # G (apply same ellipse logic as C/O, with small upper-right break + mid bar)
    # Break: remove a small arc segment near upper-right by drawing a wrapped arc that skips it.
    G_outer = pen.ellipse_arc(CAP_CX, CAP_CY, CAP_RX, CAP_RY, 340.0, 300.0, steps=340)  # wraps
    G_bar = pen.hline(CAP_CX + CAP_RX * 0.08, CAP_CX + CAP_RX * 0.95, CAP_CY)  # attaches on right
    G = pen.union(G_outer, G_bar)
    glyphs["G"] = (G, W)

    # H/I
    H = pen.union(
        pen.vline(xL, yTop, yBase),
        pen.vline(xR, yTop, yBase),
        pen.hline(xL, xR, yMid),
    )
    glyphs["H"] = (H, W)
    glyphs["I"] = (pen.vline(cx, yTop, yBase), W)

    # J (ONE continuous stroke: stem + bottom hook that actually overlaps)
    jx = xR - 120.0
    hook_r = 150.0
    hook_cx = jx - hook_r
    hook_cy = yBase - hook_r
    J = pen.union(
        pen.vline(jx, yTop, hook_cy),           # stem down to hook centerline y
        pen.arc(hook_cx, hook_cy, hook_r, 0.0, 180.0, steps=140),  # bottom bulge (down)
    )
    glyphs["J"] = (J, W)

    # K/L/M/N (kept)
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

    # O (same ellipse as C/G/Q)
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

    # Q (O + diagonal tail like your source)
    Q_tail = pen.line([
        (CAP_CX + CAP_RX * 0.52, CAP_CY + CAP_RY * 0.12),
        (CAP_CX + CAP_RX * 0.95, CAP_CY + CAP_RY * 0.72),
    ])
    Q = pen.union(O, Q_tail)
    glyphs["Q"] = (Q, W)

    # R
    R = pen.union(P, pen.line([(xFlat, yMid), (xR, yBase)]))
    glyphs["R"] = (R, W)

    # S (FIXED: two-lobe S built from 2 cubic segments)
    sx0 = xL + 70
    sx1 = xR - 40
    s_top = yTop + 115
    s_mid = yMid
    s_bot = yBase - 115

    seg1 = cubic_points(
        (sx0, s_top),
        (sx1, yTop + 30),
        (sx1, yMid - 70),
        (cx - 20, s_mid),
        steps=110
    )
    seg2 = cubic_points(
        (cx - 20, s_mid),
        (sx1, yMid + 90),
        (sx1, yBase - 40),
        (sx0, s_bot),
        steps=110
    )
    S = pen.line(seg1 + seg2[1:])
    glyphs["S"] = (S, W)

    # T
    T = pen.union(pen.hline(xL, xR, yTop), pen.vline(cx, yTop, yBase))
    glyphs["T"] = (T, W)

    # U (FIXED: bottom curve must bulge DOWN => 0→180, not 180→360)
    u_end = 560.0
    U = pen.union(
        pen.vline(xL, yTop, u_end),
        pen.arc(cx, u_end, (xR - xL) / 2.0, 0.0, 180.0, steps=170),
        pen.vline(xR, u_end, yTop),
    )
    glyphs["U"] = (U, W)

    # V/W/X/Y/Z (kept)
    glyphs["V"] = (pen.line([(xL, yTop), (cx, yBase), (xR, yTop)]), W)
    glyphs["W"] = (pen.line([(xL, yTop), (xL + 120, yBase), (cx, yMid), (xR - 120, yBase), (xR, yTop)]), W)
    glyphs["X"] = (pen.union(pen.line([(xL, yTop), (xR, yBase)]), pen.line([(xR, yTop), (xL, yBase)])), W)
    glyphs["Y"] = (pen.union(pen.line([(xL, yTop), (cx, yMid), (xR, yTop)]), pen.vline(cx, yMid, yBase)), W)
    glyphs["Z"] = (pen.union(pen.hline(xL, xR, yTop), pen.line([(xR, yTop), (xL, yBase)]), pen.hline(xL, xR, yBase)), W)

    return glyphs


def build_lowercase(m: Metrics, pen: Mono) -> Dict[str, Tuple[Geom, float]]:
    W = m.LC_W
    cx = W / 2.0

    yBase, yXTop, yMid = m.BASE, m.X_TOP, m.X_MID
    yAsc = m.CAP_TOP

    # shared lowercase bowl ellipse (c/e/o/d-ish)
    LC_RX = 190.0
    LC_RY = (yBase - yXTop) / 2.0      # 260
    LC_CX = cx
    LC_CY = yMid                       # 520

    # useful stems
    STEM_L = 160.0
    STEM_R = 460.0

    glyphs: Dict[str, Tuple[Geom, float]] = {}

    # a (keep your newer “oval + right stem + inner bar” idea, but now consistent)
    a_bowl = pen.ellipse_stroke(LC_CX - 20, LC_CY, LC_RX, LC_RY)
    a_stem = pen.vline(STEM_R, yXTop - 70, yBase)
    a_bar  = pen.hline(LC_CX - 30, STEM_R, LC_CY)
    glyphs["a"] = (pen.union(a_bowl, a_stem, a_bar), W)

    # b (acceptable for now)
    b = pen.union(
        pen.vline(STEM_L, yAsc, yBase),
        pen.ellipse_stroke(LC_CX + 30, LC_CY, LC_RX, LC_RY),
    )
    glyphs["b"] = (b, W)

    # c (FIXED: open ellipse arc -> rounded terminals)
    c = pen.ellipse_arc(LC_CX, LC_CY, LC_RX, LC_RY, 50.0, 310.0, steps=220)
    glyphs["c"] = (c, W)

    # d (FIXED: bowl on left + tall right stem like your source)
    d_bowl = pen.ellipse_stroke(LC_CX - 45, LC_CY, LC_RX, LC_RY)
    d_stem = pen.vline(STEM_R, yAsc, yBase)
    glyphs["d"] = (pen.union(d_bowl, d_stem), W)

    # e (FIXED: c + mid bar, and rounded everything)
    e_bar = pen.hline(LC_CX - LC_RX * 0.10, LC_CX + LC_RX * 0.85, LC_CY)
    glyphs["e"] = (pen.union(c, e_bar), W)

    # f (improved: tall stem + right bar + small top hook)
    fx = 260.0
    f_stem = pen.vline(fx, yAsc, yBase)
    f_bar_y = yXTop + 95.0
    f_bar = pen.hline(fx - 25, fx + 240, f_bar_y)
    f_hook = pen.arc(fx + 55, yAsc + 130, 90.0, 180.0, 270.0, steps=80)  # small curve near top
    glyphs["f"] = (pen.union(f_stem, f_bar, f_hook), W)

    # g (improved toward your source: small top bowl + descender + bottom hook)
    g_top = pen.ellipse_stroke(LC_CX - 10, LC_CY - 40, 150.0, 200.0)
    g_down = pen.line([(LC_CX + 55, LC_CY + 40), (LC_CX + 35, yBase + 140)])
    g_hook = pen.arc(LC_CX - 70, yBase + 140, 210.0, 0.0, 180.0, steps=140)  # bottom bulge down
    glyphs["g"] = (pen.union(g_top, g_down, g_hook), W)

    # h (FIXED: stem + smooth rounded arch at x-height + right stem)
    arch_rx = (STEM_R - STEM_L) / 2.0
    arch_cx = (STEM_R + STEM_L) / 2.0
    arch_cy = yXTop
    h = pen.union(
        pen.vline(STEM_L, yAsc, yBase),
        pen.ellipse_arc(arch_cx, arch_cy, arch_rx, 120.0, 180.0, 360.0, steps=160),  # top arch bulges up
        pen.vline(STEM_R, yXTop, yBase),
    )
    glyphs["h"] = (h, W)

    # i (slightly nicer dot)
    i_stem = pen.vline(cx, yXTop, yBase)
    i_dot = Point(cx, yXTop - 70).buffer(pen.r * 0.60, resolution=pen.res)
    glyphs["i"] = (pen.union(i_stem, i_dot), W)

    # j (keep decent)
    j = pen.union(
        pen.vline(cx, yXTop, m.DESC),
        pen.arc(cx - 120, m.DESC - 60, 120.0, 0.0, 180.0, steps=70),
        Point(cx, yXTop - 70).buffer(pen.r * 0.60, resolution=pen.res),
    )
    glyphs["j"] = (j, W)

    # k/l/m/n/o/p/q/r (still sketchy but stable)
    glyphs["k"] = (pen.union(
        pen.vline(STEM_L, yAsc, yBase),
        pen.line([(STEM_L, LC_CY), (STEM_R, yXTop)]),
        pen.line([(STEM_L, LC_CY), (STEM_R, yBase)]),
    ), W)

    glyphs["l"] = (pen.vline(cx, yAsc, yBase), W)

    n_arch = pen.ellipse_arc(arch_cx, arch_cy, arch_rx, 120.0, 180.0, 360.0, steps=160)
    glyphs["n"] = (pen.union(
        pen.vline(STEM_L, yXTop, yBase),
        n_arch,
        pen.vline(STEM_R, yXTop, yBase),
    ), W)

    # m = n + another arch
    m2_arch_cx = STEM_R + arch_rx
    m2 = pen.union(
        pen.vline(STEM_L, yXTop, yBase),
        n_arch,
        pen.vline(STEM_R, yXTop, yBase),
        pen.ellipse_arc(m2_arch_cx, arch_cy, arch_rx, 120.0, 180.0, 360.0, steps=160),
        pen.vline(STEM_R + 2 * arch_rx, yXTop, yBase),
    )
    glyphs["m"] = (m2, W)

    glyphs["o"] = (pen.ellipse_stroke(LC_CX, LC_CY, LC_RX, LC_RY), W)

    glyphs["p"] = (pen.union(
        pen.vline(STEM_L, yXTop, m.DESC),
        pen.ellipse_stroke(LC_CX + 30, LC_CY, LC_RX, LC_RY),
    ), W)

    glyphs["q"] = (pen.union(
        pen.vline(STEM_R, yXTop, m.DESC),
        pen.ellipse_stroke(LC_CX - 30, LC_CY, LC_RX, LC_RY),
    ), W)

    glyphs["r"] = (pen.union(
        pen.vline(STEM_L, yXTop, yBase),
        pen.arc(STEM_L + 110, yXTop + 40, 110.0, 180.0, 270.0, steps=70),
    ), W)

    # s (FIXED: two-segment cubic, like uppercase but scaled)
    sx0 = STEM_L + 35
    sx1 = STEM_R - 20
    s_top = yXTop + 70
    s_mid = LC_CY
    s_bot = yBase - 70

    seg1 = cubic_points(
        (sx0, s_top),
        (sx1, yXTop + 5),
        (sx1, LC_CY - 70),
        (LC_CX - 10, s_mid),
        steps=90
    )
    seg2 = cubic_points(
        (LC_CX - 10, s_mid),
        (sx1, LC_CY + 85),
        (sx1, yBase - 15),
        (sx0, s_bot),
        steps=90
    )
    glyphs["s"] = (pen.line(seg1 + seg2[1:]), W)

    # t (keep improved-ish)
    tx = cx
    t_top = yXTop - 160
    t_bar_y = yXTop + 10
    t_stem = pen.vline(tx, t_top, yBase)
    t_bar = pen.hline(tx - 40, tx + 240, t_bar_y)
    glyphs["t"] = (pen.union(t_stem, t_bar), W)

    # u/v/w/x/y/z (stable)
    u_end = yBase - 160
    u = pen.union(
        pen.vline(STEM_L, yXTop, u_end),
        pen.arc(cx, u_end, (STEM_R - STEM_L) / 2.0, 0.0, 180.0, steps=140),
        pen.vline(STEM_R, u_end, yXTop),
    )
    glyphs["u"] = (u, W)

    glyphs["v"] = (pen.line([(STEM_L, yXTop), (cx, yBase), (STEM_R, yXTop)]), W)
    glyphs["w"] = (pen.line([(STEM_L, yXTop), (STEM_L + 90, yBase), (cx, yMid), (STEM_R - 90, yBase), (STEM_R, yXTop)]), W)
    glyphs["x"] = (pen.union(pen.line([(STEM_L, yXTop), (STEM_R, yBase)]), pen.line([(STEM_R, yXTop), (STEM_L, yBase)])), W)
    glyphs["y"] = (pen.union(pen.line([(STEM_L, yXTop), (cx, yBase), (STEM_R, yXTop)]), pen.vline(cx, yBase, m.DESC)), W)
    glyphs["z"] = (pen.union(pen.hline(STEM_L, STEM_R, yXTop), pen.line([(STEM_R, yXTop), (STEM_L, yBase)]), pen.hline(STEM_L, STEM_R, yBase)), W)

    # ensure all letters exist
    for ch in "abcdefghijklmnopqrstuvwxyz":
        glyphs.setdefault(ch, (Polygon(), W))

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
