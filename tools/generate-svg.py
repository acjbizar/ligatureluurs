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

    # Lowercase x-height
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
        return float(self.H - 30)


# -------------------------
# Helpers: parametric points
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

def norm(vx: float, vy: float) -> Tuple[float, float]:
    n = math.hypot(vx, vy)
    if n == 0:
        return (0.0, 0.0)
    return (vx / n, vy / n)


# -------------------------
# Monoline builder
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
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{fmt(width)}" height="{m.H}" '
        f'viewBox="0 0 {fmt(width)} {m.H}">\n'
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

    # G (force connectivity by building as overlapping strokes + union)
    g_a0 = 300.0
    g_a1 = 60.0
    g_end_y = CAP_CY + 110.0
    g_end_sin = (g_end_y - CAP_CY) / CAP_RY
    g_end_sin = max(-1.0, min(1.0, g_end_sin))
    g_a2 = math.degrees(math.asin(g_end_sin))  # ~17°

    arc1 = ellipse_arc_points(CAP_CX, CAP_CY, CAP_RX, CAP_RY, g_a0, g_a1, clockwise=True, steps=220)
    arc2 = ellipse_arc_points(CAP_CX, CAP_CY, CAP_RX, CAP_RY, g_a1, g_a2, clockwise=True, steps=90)

    # Outer curve stroke as its own piece
    outer = pen.line(arc1 + arc2[1:])

    p0 = arc2[-1]
    p3 = (CAP_CX + CAP_RX * 0.773, CAP_CY)   # bar start (right-ish)
    p4 = (CAP_CX + 35.0, CAP_CY)             # bar end   (toward center)

    a = math.radians(g_a2)
    tvx, tvy = (math.sin(a) * CAP_RX, -math.cos(a) * CAP_RY)
    ux, uy = norm(tvx, tvy)

    tlen1 = CAP_RX * 0.28
    tlen2 = CAP_RX * 0.22

    p1 = (p0[0] + ux * tlen1, p0[1] + uy * tlen1)
    p2 = (p3[0] + tlen2, p3[1])  # horizontal end tangent into the bar

    bez = cubic_points(p0, p1, p2, p3, steps=70)

    # Hook stroke: start a few points back on the ellipse to guarantee overlap -> no disconnection
    hook_pts = [arc2[-8]] + bez[1:] + [p4]
    hook = pen.line(hook_pts)

    glyphs["G"] = (pen.union(outer, hook), W)

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
        (CAP_CX + CAP_RX * 0.10, CAP_CY + CAP_RY * 0.38),
        (CAP_CX + CAP_RX * 0.72, CAP_CY + CAP_RY * 0.96),
    ])
    glyphs["Q"] = (pen.union(O, q_tail), W)

    glyphs["R"] = (pen.union(glyphs["P"][0], pen.line([(xFlat, yMid), (xR, yBase)])), W)

    # S (scale to match cap width/height; less narrow + not tiny)
    pad_top = 50.0
    pad_bot = 50.0
    ref_y0, ref_y3 = 210.0, 650.0
    scale_y = (yBase - yTop - (pad_top + pad_bot)) / (ref_y3 - ref_y0)
    off_y = (yTop + pad_top) - ref_y0 * scale_y

    # widen using ref min/max in X
    pad_x = 50.0
    ref_x_min, ref_x_max = 240.0, 460.0
    target_left = xL + pad_x
    target_right = xR - pad_x
    scale_x = (target_right - target_left) / (ref_x_max - ref_x_min)
    off_x = target_left - ref_x_min * scale_x

    def Sy(y_ref: float) -> float:
        return off_y + y_ref * scale_y

    def Sx(x_ref: float) -> float:
        return off_x + x_ref * scale_x

    p0 = (Sx(460.0), Sy(210.0))
    c01 = (Sx(390.0), Sy(210.0))
    c02 = (Sx(260.0), Sy(255.0))
    p1 = (Sx(260.0), Sy(335.0))

    c11 = (Sx(260.0), Sy(435.0))
    c12 = (Sx(440.0), Sy(425.0))
    p2 = (Sx(440.0), Sy(520.0))

    c21 = (Sx(440.0), Sy(610.0))
    c22 = (Sx(320.0), Sy(650.0))
    p3 = (Sx(240.0), Sy(650.0))

    S_pts = (
        cubic_points(p0, c01, c02, p1, steps=90) +
        cubic_points(p1, c11, c12, p2, steps=90)[1:] +
        cubic_points(p2, c21, c22, p3, steps=90)[1:]
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
# Lowercase (no uppercase fallback)
# -------------------------

def build_lowercase(m: Metrics, pen: Mono) -> Dict[str, Tuple[Geom, float]]:
    W = m.LC_W
    xL, xR = 110.0, W - 110.0
    cx = W / 2.0

    yBase = m.BASE
    yXTop = m.X_TOP
    yMid = m.X_MID
    yAsc = m.CAP_TOP
    yDesc = m.DESC_END

    # bowl sizes
    rx = (xR - xL) * 0.42
    ry = (yBase - yXTop) * 0.48
    bcX = cx
    bcY = yMid + 10.0

    # dot sizing/placement
    dot_r = pen.r * 0.80
    dot_y = yXTop - 165.0

    glyphs: Dict[str, Tuple[Geom, float]] = {}

    def dshape_stem_bowl(
        stem_x: float,
        top_y: float,
        bot_y: float,
        side: str,  # "right" or "left"
        stem_top: float,
        stem_bot: float,
        bowl_rx: float,
        overlap: float = 6.0,
    ) -> Geom:
        """Guaranteed connected 'D' shape: stem + top/bot connectors + half-ellipse arc."""
        cy = (top_y + bot_y) / 2.0
        by = (bot_y - top_y) / 2.0

        if side == "right":
            cx0 = stem_x + bowl_rx - overlap
            arc = pen.ellipse_arc(cx0, cy, bowl_rx, by, 270.0, 90.0, steps=260)
            top_conn = pen.hline(stem_x, cx0, top_y)
            bot_conn = pen.hline(cx0, stem_x, bot_y)
        else:
            cx0 = stem_x - bowl_rx + overlap
            arc = pen.ellipse_arc(cx0, cy, bowl_rx, by, 90.0, 270.0, steps=260)
            top_conn = pen.hline(cx0, stem_x, top_y)
            bot_conn = pen.hline(stem_x, cx0, bot_y)

        stem = pen.vline(stem_x, stem_top, stem_bot)
        return pen.union(stem, top_conn, arc, bot_conn)

    # c
    glyphs["c"] = (pen.ellipse_arc(bcX, bcY, rx, ry, 45.0, 315.0, steps=240), W)

    # a (NO horizontal bar; bowl + right stem with overlap)
    a_rx, a_ry = rx * 0.92, ry * 0.92
    a_cx, a_cy = bcX - 12.0, bcY
    a_bowl = pen.ellipse_stroke(a_cx, a_cy, a_rx, a_ry)
    a_stem_x = a_cx + a_rx * 0.82  # inside the bowl stroke -> guaranteed overlap
    a_stem = pen.vline(a_stem_x, yXTop + 25.0, yBase - 10.0)
    glyphs["a"] = (pen.union(a_bowl, a_stem), W)

    # b (connected D-shape bowl on right)
    b_stem_x = xL + 40.0
    b_top = yXTop + 15.0
    b_bot = yBase - 10.0
    b_rx = (xR - b_stem_x) * 0.52
    glyphs["b"] = (dshape_stem_bowl(
        stem_x=b_stem_x, top_y=b_top, bot_y=b_bot,
        side="right", stem_top=yAsc, stem_bot=yBase,
        bowl_rx=b_rx, overlap=10.0,
    ), W)

    # d (connected D-shape bowl on left)
    d_stem_x = xR - 40.0
    d_top = yXTop + 15.0
    d_bot = yBase - 10.0
    d_rx = (d_stem_x - xL) * 0.52
    glyphs["d"] = (dshape_stem_bowl(
        stem_x=d_stem_x, top_y=d_top, bot_y=d_bot,
        side="left", stem_top=yAsc, stem_bot=yBase,
        bowl_rx=d_rx, overlap=10.0,
    ), W)

    # e ("3"): closed loop + right-side gap + bar into the gap
    e_loop = pen.ellipse_stroke(bcX, bcY, rx, ry)
    gap_w = pen.stroke * 0.90
    gap_h = pen.stroke * 1.25
    gap_cx = bcX + rx * 0.98  # right edge area
    gap = Polygon([
        (gap_cx - gap_w/2, bcY - gap_h/2),
        (gap_cx + gap_w/2, bcY - gap_h/2),
        (gap_cx + gap_w/2, bcY + gap_h/2),
        (gap_cx - gap_w/2, bcY + gap_h/2),
    ])
    e_outer = pen._fix(e_loop.difference(gap))
    e_bar = pen.hline(bcX - rx * 0.85, bcX + rx * 0.96, bcY)
    glyphs["e"] = (pen.union(e_outer, e_bar), W)

    # f (leave as-is here; not part of this specific fix-list)
    fx = cx - 65.0
    f_top = yAsc + 10.0
    f_bot = yBase - 10.0
    f_cross_y = yXTop + 105.0
    f = pen.union(
        pen.vline(fx, f_top, f_bot),
        pen.hline(fx, fx + 265.0, f_cross_y),
        pen.hline(fx, fx + 85.0, f_top + 55.0),
    )
    glyphs["f"] = (f, W)

    # g (NO horizontal bar; bowl + right descender + hook)
    g_bowl = pen.ellipse_stroke(bcX - 10.0, bcY - 10.0, rx * 0.88, ry * 0.88)
    g_stem_x = (bcX - 10.0) + (rx * 0.88) * 0.86  # overlap into bowl
    g_stem_y0 = bcY - 10.0 + (ry * 0.88) * 0.10
    g_stem_y1 = yDesc - 70.0
    g_stem = pen.vline(g_stem_x, g_stem_y0, g_stem_y1)
    g_hook = pen.arc(g_stem_x - 115.0, g_stem_y1, 115.0, 0.0, 180.0, steps=180)
    glyphs["g"] = (pen.union(g_bowl, g_stem, g_hook), W)

    # h (rounded top-right corner via single polyline corner)
    hxL = xL + 40.0
    hxR = xR - 40.0
    h_top_y = yXTop + 20.0
    h_corner = pen.line([(hxL, h_top_y), (hxR, h_top_y), (hxR, yBase)])
    glyphs["h"] = (pen.union(
        pen.vline(hxL, yAsc, yBase),
        h_corner,
    ), W)

    # i / j
    ix = cx
    glyphs["i"] = (pen.union(
        pen.vline(ix, yXTop + 20.0, yBase),
        pen.dot(ix, dot_y, dot_r),
    ), W)

    glyphs["j"] = (pen.union(
        pen.vline(ix, yXTop + 20.0, yDesc - 10.0),
        pen.dot(ix, dot_y, dot_r),
    ), W)

    # k
    kx = xL + 40.0
    ky_mid = (yXTop + yBase) / 2.0
    glyphs["k"] = (pen.union(
        pen.vline(kx, yAsc, yBase),
        pen.line([(kx, ky_mid), (xR - 10.0, yXTop + 20.0)]),
        pen.line([(kx, ky_mid), (xR - 10.0, yBase - 20.0)]),
    ), W)

    # l
    glyphs["l"] = (pen.vline(cx - 120.0, yAsc, yBase), W)

    # m (wider; rounded top-right corners on the arches)
    m_left = 80.0
    m_right = W - 80.0
    m_x1 = m_left
    m_x2 = cx
    m_x3 = m_right
    m_top = yXTop + 20.0
    m_arch1 = pen.line([(m_x1, m_top), (m_x2, m_top), (m_x2, yBase)])
    m_arch2 = pen.line([(m_x2, m_top), (m_x3, m_top), (m_x3, yBase)])
    glyphs["m"] = (pen.union(
        pen.vline(m_x1, m_top, yBase),
        pen.vline(m_x2, m_top, yBase),
        pen.vline(m_x3, m_top, yBase),
        m_arch1,
        m_arch2,
    ), W)

    # n (rounded top-right corner like h)
    n_x1 = xL + 40.0
    n_x2 = xR - 40.0
    n_top = yXTop + 20.0
    n_corner = pen.line([(n_x1, n_top), (n_x2, n_top), (n_x2, yBase)])
    glyphs["n"] = (pen.union(
        pen.vline(n_x1, n_top, yBase),
        n_corner,
    ), W)

    glyphs["o"] = (pen.ellipse_stroke(bcX, bcY, rx, ry), W)

    # p / q
    p_stem_x = xL + 40.0
    p_top = yXTop + 15.0
    p_bot = yBase - 10.0
    p_rx = (xR - p_stem_x) * 0.52
    glyphs["p"] = (dshape_stem_bowl(
        stem_x=p_stem_x, top_y=p_top, bot_y=p_bot,
        side="right", stem_top=p_top, stem_bot=yDesc - 10.0,
        bowl_rx=p_rx, overlap=10.0,
    ), W)

    q_stem_x = xR - 40.0
    q_top = yXTop + 15.0
    q_bot = yBase - 10.0
    q_rx = (q_stem_x - xL) * 0.52
    glyphs["q"] = (dshape_stem_bowl(
        stem_x=q_stem_x, top_y=q_top, bot_y=q_bot,
        side="left", stem_top=q_top, stem_bot=yDesc - 10.0,
        bowl_rx=q_rx, overlap=10.0,
    ), W)

    # r (extend shoulder further right)
    rx_stem = xL + 40.0
    r_top = yXTop + 20.0
    glyphs["r"] = (pen.line([
        (rx_stem, yBase),
        (rx_stem, r_top),
        (rx_stem + 260.0, r_top),
    ]), W)

    # s
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
        cubic_points(s0, c01, c02, s1, steps=70) +
        cubic_points(s1, c11, c12, s2, steps=70)[1:] +
        cubic_points(s2, c21, c22, s3, steps=70)[1:]
    )
    glyphs["s"] = (pen.line(s_pts), W)

    # t
    tx = cx - 20.0
    t_cross_y = yXTop + 105.0
    glyphs["t"] = (pen.union(
        pen.vline(tx, yAsc + 10.0, yBase),
        pen.hline(tx - 130.0, tx + 170.0, t_cross_y),
    ), W)

    # u (apply same “corner family” feel by making a single continuous U stroke)
    ux1 = xL + 50.0
    ux2 = xR - 50.0
    u_top = yXTop + 55.0
    u_bot = yBase
    u_mid_x = (ux1 + ux2) / 2.0
    u_r = (ux2 - ux1) / 2.0
    arc_pts = ellipse_arc_points(u_mid_x, u_bot, u_r, u_r, 180.0, 0.0, clockwise=True, steps=150)
    u_pts = [(ux1, u_top), (ux1, u_bot)] + arc_pts[1:] + [(ux2, u_top)]
    glyphs["u"] = (pen.line(u_pts), W)

    # v
    glyphs["v"] = (pen.line([(xL + 60.0, yXTop + 20.0), (cx, yBase), (xR - 60.0, yXTop + 20.0)]), W)

    # w (source-like “ш” w; wider)
    w_left = 80.0
    w_right = W - 80.0
    w_x1 = w_left
    w_x2 = cx
    w_x3 = w_right
    w_top = yXTop + 55.0
    w_bot = yBase
    w_geom = pen.union(
        pen.vline(w_x1, w_top, w_bot),
        pen.vline(w_x2, w_top, w_bot),
        pen.vline(w_x3, w_top, w_bot),
        pen.hline(w_x1, w_x3, w_bot),
    )
    glyphs["w"] = (w_geom, W)

    # x
    x1 = xL + 55.0
    x2 = xR - 55.0
    x_top = yXTop + 30.0
    x_bot = yBase - 10.0
    glyphs["x"] = (pen.union(
        pen.line([(x1, x_top), (x2, x_bot)]),
        pen.line([(x2, x_top), (x1, x_bot)]),
    ), W)

    # y (more like h-family: top bar into right descender)
    yx1 = xL + 60.0
    yx2 = xR - 60.0
    y_top = yXTop + 20.0
    y_corner = pen.line([(yx1, y_top), (yx2, y_top), (yx2, yDesc - 10.0)])
    glyphs["y"] = (pen.union(
        pen.vline(yx1, y_top, yBase),
        y_corner,
    ), W)

    # z
    z_top = yXTop + 30.0
    z_bot = yBase - 10.0
    glyphs["z"] = (pen.union(
        pen.hline(xL + 40.0, xR - 40.0, z_top),
        pen.line([(xR - 40.0, z_top), (xL + 40.0, z_bot)]),
        pen.hline(xL + 40.0, xR - 40.0, z_bot),
    ), W)

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
