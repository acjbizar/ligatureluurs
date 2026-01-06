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

    # Lowercase x-height (tuned to match your source rows better)
    XH: float = 440.0  # baseline -> xheight distance

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
    def CAP_H(self) -> float:
        return self.BASE - self.CAP_TOP

    @property
    def DESC_END(self) -> float:
        # keep a little padding above viewbox bottom
        return float(self.H - 30)


# -------------------------
# Helpers: parametric points (for continuous stroked paths)
# -------------------------

def ellipse_point(cx: float, cy: float, rx: float, ry: float, deg: float) -> Tuple[float, float]:
    a = math.radians(deg)
    return (cx + math.cos(a) * rx, cy + math.sin(a) * ry)

def ellipse_arc_points(
    cx: float, cy: float, rx: float, ry: float,
    deg0: float, deg1: float,
    clockwise: bool,
    steps: int
) -> List[Tuple[float, float]]:
    d0 = float(deg0)
    d1 = float(deg1)

    if clockwise:
        while d1 > d0:
            d1 -= 360.0
    else:
        while d1 <= d0:
            d1 += 360.0

    pts: List[Tuple[float, float]] = []
    for i in range(steps):
        t = i / (steps - 1)
        deg = d0 + (d1 - d0) * t
        pts.append(ellipse_point(cx, cy, rx, ry, deg))
    return pts

def cubic_points(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    steps: int = 60
) -> List[Tuple[float, float]]:
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    pts: List[Tuple[float, float]] = []
    for i in range(steps):
        t = i / (steps - 1)
        mt = 1.0 - t
        x = (mt**3)*x0 + 3*(mt**2)*t*x1 + 3*mt*(t**2)*x2 + (t**3)*x3
        y = (mt**3)*y0 + 3*(mt**2)*t*y1 + 3*mt*(t**2)*y2 + (t**3)*y3
        pts.append((x, y))
    return pts


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

    def arc(self, cx: float, cy: float, r: float, deg0: float, deg1: float, steps: int = 160) -> Geom:
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

    CAP_RX = (xR - xL) / 2.0
    CAP_RY = (yBase - yTop) / 2.0
    CAP_CX = cx
    CAP_CY = yMid

    glyphs: Dict[str, Tuple[Geom, float]] = {}

    # A
    yArch = 260.0
    rArch = (xR - xL) / 2.0
    yBar = 450.0
    glyphs["A"] = (pen.union(
        pen.vline(xL, yBase, yArch),
        pen.arc(cx, yArch, rArch, 180.0, 360.0, steps=180),
        pen.vline(xR, yArch, yBase),
        pen.hline(xL, xR, yBar),
    ), W)

    # B
    bowl_r = (yMid - yTop) / 2.0
    xFlat = 375.0
    glyphs["B"] = (pen.union(
        pen.vline(xL, yTop, yBase),
        pen.hline(xL, xFlat, yTop),
        pen.arc(xFlat, yTop + bowl_r, bowl_r, 270.0, 90.0, steps=150),
        pen.hline(xFlat, xL, yMid),
        pen.hline(xL, xFlat, yMid),
        pen.arc(xFlat, yMid + bowl_r, bowl_r, 270.0, 90.0, steps=150),
        pen.hline(xFlat, xL, yBase),
    ), W)

    # C
    glyphs["C"] = (pen.ellipse_arc(CAP_CX, CAP_CY, CAP_RX, CAP_RY, 45.0, 315.0, steps=280), W)

    # D
    xJoin = cx
    glyphs["D"] = (pen.union(
        pen.vline(xL, yTop, yBase),
        pen.hline(xL, xJoin, yTop),
        pen.ellipse_arc(xJoin, CAP_CY, xR - xJoin, CAP_RY, 270.0, 90.0, steps=240),
        pen.hline(xJoin, xL, yBase),
    ), W)

    # E / F
    glyphs["E"] = (pen.union(
        pen.vline(xL, yTop, yBase),
        pen.hline(xL, xR, yTop),
        pen.hline(xL, xR - 50, yMid),
        pen.hline(xL, xR, yBase),
    ), W)

    glyphs["F"] = (pen.union(
        pen.vline(xL, yTop, yBase),
        pen.hline(xL, xR, yTop),
        pen.hline(xL, xR - 70, yMid),
    ), W)

    # G (UPDATED to your chosen smooth construction)
    # Construct as one continuous centerline: long ellipse arc (clockwise, large arc),
    # then short arc (clockwise) into the right-side hook, then cubic easing into bar, then bar.
    # This matches the SVG you approved:
    #   M (cx+rx*0.5, cy-ry*0.866)
    #   A rx ry ... to (same x, cy+ry*0.866) (LONG way)
    #   A rx ry ... to near (cx+rx*0.9545, cy+ry*0.2973)
    #   C ... to (cx+rx*0.773, cy)
    #   L ... to (cx+35, cy)
    g_a0 = 300.0
    g_a1 = 60.0
    # derive end angle from desired relative y (≈ +110 on ry=370 in your markup)
    g_end_y = CAP_CY + 110.0
    g_end_sin = (g_end_y - CAP_CY) / CAP_RY
    g_end_sin = max(-1.0, min(1.0, g_end_sin))
    g_a2 = math.degrees(math.asin(g_end_sin))  # ~17°
    # points
    arc1 = ellipse_arc_points(CAP_CX, CAP_CY, CAP_RX, CAP_RY, g_a0, g_a1, clockwise=True, steps=220)
    arc2 = ellipse_arc_points(CAP_CX, CAP_CY, CAP_RX, CAP_RY, g_a1, g_a2, clockwise=True, steps=90)
    p0 = arc2[-1]
    p3 = (CAP_CX + CAP_RX * 0.773, CAP_CY)   # ~ (520, 410)
    p4 = (CAP_CX + 35.0, CAP_CY)             # ~ (385, 410)

    p1 = (p0[0], CAP_CY + 30.0)              # ~ (560, 440)
    p2 = (CAP_CX + CAP_RX * 0.864, CAP_CY)   # ~ (540, 410)

    bez = cubic_points(p0, p1, p2, p3, steps=60)
    bar = [p3, p4]

    G_pts = arc1 + arc2[1:] + bez[1:] + bar
    glyphs["G"] = (pen.line(G_pts), W)

    # H / I / J / K / L / M / N
    glyphs["H"] = (pen.union(
        pen.vline(xL, yTop, yBase),
        pen.vline(xR, yTop, yBase),
        pen.hline(xL, xR, yMid),
    ), W)

    glyphs["I"] = (pen.vline(cx, yTop, yBase), W)

    jx = xR - 120.0
    hook_r = 150.0
    hook_cx = jx - hook_r
    hook_cy = yBase - hook_r
    glyphs["J"] = (pen.union(
        pen.vline(jx, yTop, hook_cy),
        pen.arc(hook_cx, hook_cy, hook_r, 0.0, 180.0, steps=150),
    ), W)

    glyphs["K"] = (pen.union(
        pen.vline(xL, yTop, yBase),
        pen.line([(xL, yMid), (xR, yTop)]),
        pen.line([(xL, yMid), (xR, yBase)]),
    ), W)

    glyphs["L"] = (pen.union(pen.vline(xL, yTop, yBase), pen.hline(xL, xR, yBase)), W)

    glyphs["M"] = (pen.union(
        pen.vline(xL, yTop, yBase),
        pen.vline(xR, yTop, yBase),
        pen.line([(xL, yTop), (cx, yMid), (xR, yTop)]),
    ), W)

    glyphs["N"] = (pen.union(
        pen.vline(xL, yTop, yBase),
        pen.vline(xR, yTop, yBase),
        pen.line([(xL, yTop), (xR, yBase)]),
    ), W)

    # O
    O = pen.ellipse_stroke(CAP_CX, CAP_CY, CAP_RX, CAP_RY)
    glyphs["O"] = (O, W)

    # P / Q / R
    glyphs["P"] = (pen.union(
        pen.vline(xL, yTop, yBase),
        pen.hline(xL, xFlat, yTop),
        pen.arc(xFlat, yTop + bowl_r, bowl_r, 270.0, 90.0, steps=150),
        pen.hline(xFlat, xL, yMid),
    ), W)

    q_tail = pen.line([
        (CAP_CX - CAP_RX * 0.20, CAP_CY + CAP_RY * 0.45),
        (CAP_CX + CAP_RX * 0.55, CAP_CY + CAP_RY * 0.92),
    ])
    glyphs["Q"] = (pen.union(O, q_tail), W)
    glyphs["R"] = (pen.union(glyphs["P"][0], pen.line([(xFlat, yMid), (xR, yBase)])), W)

    # S (UPDATED to the calmer/boring cubic form you approved)
    # Use 3 cubics (continuous stroke), derived from the markup proportions.
    s_rx = CAP_RX
    s0 = (CAP_CX + s_rx * 0.50, yTop + 170.0)
    s1 = (CAP_CX - s_rx * 0.409, yTop + 295.0)
    s2 = (CAP_CX + s_rx * 0.409, yTop + 480.0)
    s3 = (CAP_CX - s_rx * 0.50,  yTop + 610.0)

    c01 = (CAP_CX + s_rx * 0.182, yTop + 170.0)
    c02 = (CAP_CX - s_rx * 0.409, yTop + 215.0)

    c11 = (CAP_CX - s_rx * 0.409, yTop + 395.0)
    c12 = (CAP_CX + s_rx * 0.409, yTop + 385.0)

    c21 = (CAP_CX + s_rx * 0.409, yTop + 570.0)
    c22 = (CAP_CX - s_rx * 0.136, yTop + 610.0)

    S_pts = (
        cubic_points(s0, c01, c02, s1, steps=70) +
        cubic_points(s1, c11, c12, s2, steps=70)[1:] +
        cubic_points(s2, c21, c22, s3, steps=70)[1:]
    )
    glyphs["S"] = (pen.line(S_pts), W)

    # T/U/V/W/X/Y/Z
    glyphs["T"] = (pen.union(pen.hline(xL, xR, yTop), pen.vline(cx, yTop, yBase)), W)

    u_end = 560.0
    glyphs["U"] = (pen.union(
        pen.vline(xL, yTop, u_end),
        pen.arc(cx, u_end, (xR - xL) / 2.0, 0.0, 180.0, steps=180),
        pen.vline(xR, u_end, yTop),
    ), W)

    glyphs["V"] = (pen.line([(xL, yTop), (cx, yBase), (xR, yTop)]), W)
    glyphs["W"] = (pen.line([(xL, yTop), (xL + 120, yBase), (cx, yMid), (xR - 120, yBase), (xR, yTop)]), W)
    glyphs["X"] = (pen.union(pen.line([(xL, yTop), (xR, yBase)]), pen.line([(xR, yTop), (xL, yBase)])), W)
    glyphs["Y"] = (pen.union(pen.line([(xL, yTop), (cx, yMid), (xR, yTop)]), pen.vline(cx, yMid, yBase)), W)
    glyphs["Z"] = (pen.union(pen.hline(xL, xR, yTop), pen.line([(xR, yTop), (xL, yBase)]), pen.hline(xL, xR, yBase)), W)

    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        glyphs.setdefault(ch, (Polygon(), W))

    return glyphs


# -------------------------
# Lowercase (no uppercase-scaling fallback)
# -------------------------

def build_lowercase(m: Metrics, pen: Mono) -> Dict[str, Tuple[Geom, float]]:
    W = m.LC_W
    # a bit tighter margins than caps
    xL, xR = 110.0, W - 110.0
    cx = W / 2.0

    yBase = m.BASE
    yXTop = m.X_TOP
    yMid = m.X_MID
    yAsc = m.CAP_TOP
    yDesc = m.DESC_END

    # bowl sizes for o/c/e/etc
    rx = (xR - xL) * 0.42
    ry = (yBase - yXTop) * 0.48
    bcX = cx
    bcY = yMid + 10.0

    # dot sizing/placement closer to source
    dot_r = pen.r * 0.70
    dot_y = yXTop - 150.0

    glyphs: Dict[str, Tuple[Geom, float]] = {}

    # --- a-j (keep your “source-ish” logic; still tweakable later) ---

    # c
    glyphs["c"] = (pen.ellipse_arc(bcX, bcY, rx, ry, 45.0, 315.0, steps=240), W)

    # a: bowl + right stem + mid bar (source has that “o + right closure” feel)
    a_bowl = pen.ellipse_stroke(bcX - 10.0, bcY, rx * 0.90, ry * 0.90)
    a_stem_x = (bcX + rx * 0.62)
    a_stem = pen.vline(a_stem_x, yXTop + 25.0, yBase - 5.0)
    a_bar  = pen.hline(bcX - rx * 0.05, a_stem_x, bcY + 5.0)
    glyphs["a"] = (pen.union(a_bowl, a_stem, a_bar), W)

    # b: tall stem + right half bowl
    b_stem_x = xL + 35.0
    b_cx = b_stem_x + (xR - b_stem_x) * 0.52
    b_rx = (xR - b_stem_x) * 0.52
    b_ry = (yBase - yXTop) / 2.0
    b_cy = (yXTop + yBase) / 2.0
    glyphs["b"] = (pen.union(
        pen.vline(b_stem_x, yAsc, yBase),
        pen.ellipse_arc(b_cx, b_cy, b_rx, b_ry, 270.0, 90.0, steps=240),
    ), W)

    # d: mirror of b
    d_stem_x = xR - 35.0
    d_cx = d_stem_x - (d_stem_x - xL) * 0.52
    d_rx = (d_stem_x - xL) * 0.52
    d_ry = (yBase - yXTop) / 2.0
    d_cy = (yXTop + yBase) / 2.0
    glyphs["d"] = (pen.union(
        pen.vline(d_stem_x, yAsc, yBase),
        pen.ellipse_arc(d_cx, d_cy, d_rx, d_ry, 90.0, 270.0, steps=240),
    ), W)

    # e: c + bar
    glyphs["e"] = (pen.union(
        glyphs["c"][0],
        pen.hline(bcX - rx * 0.75, bcX + rx * 0.55, bcY),
    ), W)

    # f: tall stem + mid bar + small top curl
    fx = cx - 40.0
    f_cross_y = yXTop + 95.0
    glyphs["f"] = (pen.union(
        pen.vline(fx, yAsc + 10.0, yBase),
        pen.hline(fx - 40.0, fx + 240.0, f_cross_y),
        pen.arc(fx + 15.0, yAsc + 110.0, 70.0, 180.0, 270.0, steps=90),
    ), W)

    # g: bowl + right stem down + leftward underline (source-like)
    g_bowl = pen.ellipse_stroke(bcX - 5.0, bcY - 10.0, rx * 0.92, ry * 0.92)
    g_stem_x = bcX + rx * 0.55
    g_tail = pen.vline(g_stem_x, bcY + ry * 0.15, yDesc - 60.0)
    g_under = pen.line([
        (g_stem_x, yDesc - 60.0),
        (bcX + 10.0, yDesc - 10.0),
        (bcX - rx * 0.95, yDesc - 30.0),
    ])
    glyphs["g"] = (pen.union(g_bowl, g_tail, g_under), W)

    # h: tall left stem + flat(ish) top connector + right stem (more like source than “arch”)
    hxL = xL + 35.0
    hxR = xR - 35.0
    h_top_y = yXTop + 20.0
    glyphs["h"] = (pen.union(
        pen.vline(hxL, yAsc, yBase),
        pen.hline(hxL, hxR, h_top_y),
        pen.vline(hxR, h_top_y, yBase),
    ), W)

    # i / j: shorter stems + bigger dot + more spacing
    ix = cx
    glyphs["i"] = (pen.union(
        pen.vline(ix, yXTop + 10.0, yBase),
        pen.dot(ix, dot_y, dot_r),
    ), W)

    glyphs["j"] = (pen.union(
        pen.vline(ix, yXTop + 10.0, yDesc - 10.0),
        pen.dot(ix, dot_y, dot_r),
    ), W)

    # --- k-z (NEW: lowercase-specific, no scaled caps) ---

    # k: tall stem + two diagonals from mid (source-like)
    kx = xL + 35.0
    ky_mid = (yXTop + yBase) / 2.0
    k = pen.union(
        pen.vline(kx, yAsc, yBase),
        pen.line([(kx, ky_mid), (xR - 10.0, yXTop + 20.0)]),
        pen.line([(kx, ky_mid), (xR - 10.0, yBase - 20.0)]),
    )
    glyphs["k"] = (k, W)

    # l: tall stem
    lx = cx - 120.0
    glyphs["l"] = (pen.vline(lx, yAsc, yBase), W)

    # m: three stems + flat top connectors (source is very “boring” here)
    m_x1 = xL + 35.0
    m_x2 = cx - 20.0
    m_x3 = xR - 35.0
    m_top = yXTop + 20.0
    m = pen.union(
        pen.vline(m_x1, m_top, yBase),
        pen.vline(m_x2, m_top, yBase),
        pen.vline(m_x3, m_top, yBase),
        pen.hline(m_x1, m_x2, m_top),
        pen.hline(m_x2, m_x3, m_top),
    )
    glyphs["m"] = (m, W)

    # n: two stems + flat top connector
    n_x1 = xL + 35.0
    n_x2 = xR - 35.0
    n_top = yXTop + 20.0
    n = pen.union(
        pen.vline(n_x1, n_top, yBase),
        pen.vline(n_x2, n_top, yBase),
        pen.hline(n_x1, n_x2, n_top),
    )
    glyphs["n"] = (n, W)

    # o
    glyphs["o"] = (pen.ellipse_stroke(bcX, bcY, rx, ry), W)

    # p: left stem descender + right half bowl at x-height/baseline
    px = xL + 35.0
    p_cx = px + (xR - px) * 0.52
    p_rx = (xR - px) * 0.52
    p_ry = (yBase - yXTop) / 2.0
    p_cy = (yXTop + yBase) / 2.0
    p = pen.union(
        pen.vline(px, yXTop + 20.0, yDesc - 10.0),
        pen.ellipse_arc(p_cx, p_cy, p_rx, p_ry, 270.0, 90.0, steps=240),
    )
    glyphs["p"] = (p, W)

    # q: right stem descender + left half bowl
    qx = xR - 35.0
    q_cx = qx - (qx - xL) * 0.52
    q_rx = (qx - xL) * 0.52
    q_ry = (yBase - yXTop) / 2.0
    q_cy = (yXTop + yBase) / 2.0
    q = pen.union(
        pen.vline(qx, yXTop + 20.0, yDesc - 10.0),
        pen.ellipse_arc(q_cx, q_cy, q_rx, q_ry, 90.0, 270.0, steps=240),
    )
    glyphs["q"] = (q, W)

    # r: short stem + small shoulder
    rx_stem = xL + 35.0
    r_top = yXTop + 20.0
    r = pen.union(
        pen.vline(rx_stem, r_top, yBase),
        pen.arc(rx_stem + 95.0, r_top + 45.0, 95.0, 180.0, 270.0, steps=90),
    )
    glyphs["r"] = (r, W)

    # s: calmer “boring S” at x-height
    # (scaled version of the cap S logic)
    s_rx_l = (xR - xL) * 0.42
    s_cx_l = cx
    s0 = (s_cx_l + s_rx_l * 0.50, yXTop + 70.0)
    s1 = (s_cx_l - s_rx_l * 0.409, yXTop + 165.0)
    s2 = (s_cx_l + s_rx_l * 0.409, yXTop + 310.0)
    s3 = (s_cx_l - s_rx_l * 0.50,  yXTop + 395.0)

    c01 = (s_cx_l + s_rx_l * 0.182, yXTop + 70.0)
    c02 = (s_cx_l - s_rx_l * 0.409, yXTop + 105.0)

    c11 = (s_cx_l - s_rx_l * 0.409, yXTop + 235.0)
    c12 = (s_cx_l + s_rx_l * 0.409, yXTop + 225.0)

    c21 = (s_cx_l + s_rx_l * 0.409, yXTop + 360.0)
    c22 = (s_cx_l - s_rx_l * 0.136, yXTop + 395.0)

    s_pts = (
        cubic_points(s0, c01, c02, s1, steps=60) +
        cubic_points(s1, c11, c12, s2, steps=60)[1:] +
        cubic_points(s2, c21, c22, s3, steps=60)[1:]
    )
    glyphs["s"] = (pen.line(s_pts), W)

    # t: tall stem + crossbar near x-height
    tx = cx - 20.0
    t_cross_y = yXTop + 95.0
    t = pen.union(
        pen.vline(tx, yAsc + 10.0, yBase),
        pen.hline(tx - 130.0, tx + 170.0, t_cross_y),
    )
    glyphs["t"] = (t, W)

    # u: two stems + bottom bowl (open top)
    ux1 = xL + 35.0
    ux2 = xR - 35.0
    u_bottom = yBase - 10.0
    u_mid_x = (ux1 + ux2) / 2.0
    u_r = (ux2 - ux1) / 2.0
    u = pen.union(
        pen.vline(ux1, yXTop + 20.0, u_bottom),
        pen.arc(u_mid_x, u_bottom, u_r, 0.0, 180.0, steps=130),
        pen.vline(ux2, u_bottom, yXTop + 20.0),
    )
    glyphs["u"] = (u, W)

    # v
    vx1 = xL + 50.0
    vx2 = xR - 50.0
    v = pen.line([(vx1, yXTop + 20.0), (cx, yBase), (vx2, yXTop + 20.0)])
    glyphs["v"] = (v, W)

    # w
    wx1 = xL + 30.0
    wx2 = cx - 30.0
    wx3 = cx + 30.0
    wx4 = xR - 30.0
    w = pen.line([
        (wx1, yXTop + 20.0),
        (wx2, yBase),
        (cx,  yXTop + 120.0),
        (wx3, yBase),
        (wx4, yXTop + 20.0),
    ])
    glyphs["w"] = (w, W)

    # x
    x1 = xL + 55.0
    x2 = xR - 55.0
    x_top = yXTop + 30.0
    x_bot = yBase - 10.0
    glyphs["x"] = (pen.union(
        pen.line([(x1, x_top), (x2, x_bot)]),
        pen.line([(x2, x_top), (x1, x_bot)]),
    ), W)

    # y: like v but right leg descends
    yx1 = xL + 60.0
    yx2 = xR - 60.0
    y = pen.union(
        pen.line([(yx1, yXTop + 20.0), (cx, yBase), (yx2, yXTop + 20.0)]),
        pen.vline(yx2, yXTop + 20.0, yDesc - 10.0),
    )
    glyphs["y"] = (y, W)

    # z: top bar + diagonal + bottom bar
    z_top = yXTop + 30.0
    z_bot = yBase - 10.0
    z = pen.union(
        pen.hline(xL + 40.0, xR - 40.0, z_top),
        pen.line([(xR - 40.0, z_top), (xL + 40.0, z_bot)]),
        pen.hline(xL + 40.0, xR - 40.0, z_bot),
    )
    glyphs["z"] = (z, W)

    # Ensure all lowercase exist
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
