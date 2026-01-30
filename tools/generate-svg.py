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


@dataclass(frozen=True)
class Metrics:
    H: int = 1000
    BASE: float = 780.0
    CAP_TOP: float = 40.0

    CAP_W: float = 700.0
    LC_W: float = 600.0

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


def fmt(x: float) -> str:
    return f"{x:.3f}"

def codepoint_filename(s: str) -> str:
    cps = [f"u{ord(ch):04x}" for ch in s]  # lowercase u + lowercase hex
    code = "_".join(cps)
    return f"character-{code}.svg"

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
        f'viewBox="0 0 {fmt(width)} {m.H}" '
        f'width="{fmt(width)}" height="{m.H}">\n'
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

    yArch = 260.0
    rArch = (xR - xL) / 2.0
    yBar = 450.0
    glyphs["A"] = (pen.union(
        pen.vline(xL, yBase, yArch),
        pen.arc(cx, yArch, rArch, 180.0, 360.0, steps=180),
        pen.vline(xR, yArch, yBase),
        pen.hline(xL, xR, yBar),
    ), W)

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

    glyphs["C"] = (pen.ellipse_arc(CAP_CX, CAP_CY, CAP_RX, CAP_RY, 45.0, 315.0, steps=280), W)

    xJoin = cx
    glyphs["D"] = (pen.union(
        pen.vline(xL, yTop, yBase),
        pen.hline(xL, xJoin, yTop),
        pen.ellipse_arc(xJoin, CAP_CY, xR - xJoin, CAP_RY, 270.0, 90.0, steps=240),
        pen.hline(xJoin, xL, yBase),
    ), W)

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

    # G (reverted construction; union pieces so it can't disconnect)
    g_a0 = 300.0
    g_a1 = 60.0
    g_end_y = CAP_CY + 110.0
    g_end_sin = (g_end_y - CAP_CY) / CAP_RY
    g_end_sin = max(-1.0, min(1.0, g_end_sin))
    g_a2 = math.degrees(math.asin(g_end_sin))  # ~17°

    arc1_pts = ellipse_arc_points(CAP_CX, CAP_CY, CAP_RX, CAP_RY, g_a0, g_a1, clockwise=True, steps=220)
    arc2_pts = ellipse_arc_points(CAP_CX, CAP_CY, CAP_RX, CAP_RY, g_a1, g_a2, clockwise=True, steps=90)

    p0 = arc2_pts[-1]
    p3 = (CAP_CX + CAP_RX * 0.773, CAP_CY)   # bar start
    p4 = (CAP_CX + 35.0, CAP_CY)             # bar end

    a = math.radians(g_a2)
    tvx, tvy = (math.sin(a) * CAP_RX, -math.cos(a) * CAP_RY)  # clockwise tangent
    ux, uy = norm(tvx, tvy)

    tlen1 = CAP_RX * 0.28
    tlen2 = CAP_RX * 0.22

    p1 = (p0[0] + ux * tlen1, p0[1] + uy * tlen1)
    p2 = (p3[0] + tlen2, p3[1])  # horizontal end tangent into the bar

    bez_pts = cubic_points(p0, p1, p2, p3, steps=70)

    g_outer = pen.line(arc1_pts + arc2_pts[1:])
    g_join  = pen.line(bez_pts)
    g_bar   = pen.hline(p4[0], p3[0], CAP_CY)

    glyphs["G"] = (pen.union(g_outer, g_join, g_bar), W)

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

    O = pen.ellipse_stroke(CAP_CX, CAP_CY, CAP_RX, CAP_RY)
    glyphs["O"] = (O, W)

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

    # S (less dynamic + full cap height)
    S_y0 = yTop + 25.0
    S_y3 = yBase - 25.0
    S_h = S_y3 - S_y0
    S_xL = xL + 75.0
    S_xR = xR - 75.0
    S_w = S_xR - S_xL

    p0 = (S_xR, S_y0)
    p1 = (S_xL, S_y0 + S_h * 0.28)
    p2 = (S_xR, S_y0 + S_h * 0.62)
    p3 = (S_xL, S_y3)

    c01 = (S_xR - S_w * 0.35, S_y0)
    c02 = (S_xL,              S_y0 + S_h * 0.12)

    c11 = (S_xL,              S_y0 + S_h * 0.44)
    c12 = (S_xR,              S_y0 + S_h * 0.38)

    c21 = (S_xR,              S_y0 + S_h * 0.82)
    c22 = (S_xL + S_w * 0.35, S_y3)

    S_pts = (
        cubic_points(p0, c01, c02, p1, steps=90) +
        cubic_points(p1, c11, c12, p2, steps=90)[1:] +
        cubic_points(p2, c21, c22, p3, steps=90)[1:]
    )
    glyphs["S"] = (pen.line(S_pts), W)

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


def build_digits(m: Metrics, pen: Mono) -> Dict[str, Tuple[Geom, float]]:
    W = m.CAP_W

    # --- shared digit frame (optical consistency) ---
    inset_x = 150.0
    xL, xR = inset_x, W - inset_x
    cx = (xL + xR) / 2.0

    yCapTop = m.CAP_TOP
    yCapBase = m.BASE
    yMid = (yCapTop + yCapBase) / 2.0

    rx = (xR - xL) / 2.0
    ry = (yCapBase - yCapTop) / 2.0

    # Slight optical inset so straight digits don't look shorter/longer than round ones
    orx = rx * 0.96
    ory = ry * 0.96
    yTop = yMid - ory
    yBase = yMid + ory

    glyphs: Dict[str, Tuple[Geom, float]] = {}

    # 0 (match cap O style, but optically inset)
    glyphs["0"] = (pen.ellipse_stroke(cx, yMid, orx, ory), W)

    # 1 (widened with a flag + longer base so it doesn't look skinny)
    one_x = cx + orx * 0.08
    one_stem = pen.vline(one_x, yTop, yBase)
    one_base = pen.hline(cx - orx * 0.45, cx + orx * 0.35, yBase)
    one_flag = pen.line([(cx - orx * 0.28, yTop + ory * 0.22), (one_x, yTop)])
    glyphs["1"] = (pen.union(one_stem, one_base, one_flag), W)

    # 2 (SVG-based shape, then stretch vertically so it matches digit height)
    cap_h = (yBase - yTop)

    def X(frac: float) -> float:
        return W * frac

    def Y(t: float) -> float:
        return yTop + cap_h * t

    # SVG points (700×1000 preview coordinates mapped into yTop..yBase)
    p0  = (X(190/700), Y((260-40)/740))
    c01 = (X(240/700), Y((150-40)/740))
    c02 = (X(420/700), Y((130-40)/740))
    p1  = (X(500/700), Y((220-40)/740))

    c11 = (X(560/700), Y((290-40)/740))
    c12 = (X(520/700), Y((390-40)/740))
    p2  = (X(420/700), Y((460-40)/740))

    c21 = (X(350/700), Y((510-40)/740))
    c22 = (X(300/700), Y((540-40)/740))
    p3  = (X(260/700), Y((610-40)/740))

    p4  = (X(170/700), yBase)   # bottom-left
    p5  = (X(520/700), yBase)   # baseline end

    # --- stretch vertically around baseline so the top is closer to yTop ---
    shape_pts = [p0, c01, c02, p1, c11, c12, p2, c21, c22, p3]
    y_min = min(y for _, y in shape_pts)  # highest point (smallest y)

    # target top: bring the top close to yTop (tweak factor if you want)
    target_top = yTop + pen.r * 0.35

    # scale factor about baseline (keeps yBase fixed)
    s = (yBase - target_top) / (yBase - y_min)
    s = max(1.0, min(s, 1.45))  # clamp to avoid overshoot

    def SY(pt):
        x, y = pt
        # baseline points stay baseline automatically
        return (x, yBase - (yBase - y) * s)

    p0, c01, c02, p1, c11, c12, p2, c21, c22, p3 = map(SY, [p0, c01, c02, p1, c11, c12, p2, c21, c22, p3])

    seg0 = cubic_points(p0, c01, c02, p1, steps=90)
    seg1 = cubic_points(p1, c11, c12, p2, steps=90)[1:]
    seg2 = cubic_points(p2, c21, c22, p3, steps=90)[1:]

    two_pts = seg0 + seg1 + seg2 + [p4, p5]
    glyphs["2"] = (pen.line(two_pts), W)



    # 3 (top/bottom bars curve inward on the left WITHOUT S-curving back)
    y3_top = yTop + ory * 0.10
    y3_mid = yMid
    y3_bot = yBase - ory * 0.10

    x3_left = xL + orx * 0.05
    bulge3  = orx * 0.55
    x3_join = xR - bulge3

    # Move the left endpoints toward the middle so the curve is one-directional
    top_left_y = y3_top + ory * 0.10    # down toward middle
    bot_left_y = y3_bot - ory * 0.10    # up toward middle

    def one_way_bar(yR: float, yL: float) -> Geom:
        p0 = (x3_join, yR)
        p3 = (x3_left, yL)
        c1 = (x3_join - orx * 0.55, yR)      # keep right side steady
        c2 = (x3_left + orx * 0.12, yL)      # shape happens near left
        return pen.line(cubic_points(p0, c1, c2, p3, steps=90))

    three_top = one_way_bar(y3_top, top_left_y)
    three_mid = pen.hline(x3_left + orx * 0.14, x3_join, y3_mid)
    three_bot = one_way_bar(y3_bot, bot_left_y)

    cy_u = (y3_top + y3_mid) / 2.0
    ry_u = (y3_mid - y3_top) / 2.0
    upper_loop = pen.line(ellipse_arc_points(x3_join, cy_u, bulge3, ry_u, 270.0, 90.0, clockwise=False, steps=220))

    cy_l = (y3_mid + y3_bot) / 2.0
    ry_l = (y3_bot - y3_mid) / 2.0
    lower_loop = pen.line(ellipse_arc_points(x3_join, cy_l, bulge3, ry_l, 270.0, 90.0, clockwise=False, steps=220))

    glyphs["3"] = (pen.union(three_top, three_mid, three_bot, upper_loop, lower_loop), W)


    # 4 (your approved construction: crossbar + stem + single diagonal to top of stem)
    x4_stem = xR
    y4_top = yTop
    y4_cross = yMid + ory * 0.05
    x4_left = xL

    four_stem = pen.vline(x4_stem, y4_top, yBase)
    four_bar  = pen.hline(x4_left, x4_stem, y4_cross)
    four_diag = pen.line([(x4_left, y4_cross), (x4_stem, y4_top)])
    glyphs["4"] = (pen.union(four_stem, four_bar, four_diag), W)

    # 5 (your “good” style: top + left + mid + bottom + right half-loop)
    y5_top = yTop + ory * 0.10
    y5_mid = yMid + ory * 0.05
    y5_bot = yBase - ory * 0.10

    x5_left = xL + orx * 0.02
    bulge5 = orx * 0.60
    x5_join = xR - bulge5

    five_top  = pen.hline(x5_left, xR, y5_top)
    five_left = pen.vline(x5_left, y5_top, y5_mid)
    five_mid  = pen.hline(x5_left, x5_join, y5_mid)
    five_bot  = pen.hline(x5_left, x5_join, y5_bot)

    cy5 = (y5_mid + y5_bot) / 2.0
    ry5 = (y5_bot - y5_mid) / 2.0
    five_loop = pen.line(ellipse_arc_points(x5_join, cy5, bulge5, ry5, 270.0, 90.0, clockwise=False, steps=240))
    glyphs["5"] = (pen.union(five_top, five_left, five_mid, five_bot, five_loop), W)


    # 6 (match capital O height: outer top/bottom like cap O; shorter terminal)
    yTop  = m.CAP_TOP
    yBase = m.BASE

    six_rx = orx * 0.92
    six_ry = ory * 0.60
    six_cx = cx + 15.0

    # IMPORTANT:
    # Make the OUTER bottom match cap "O" outer bottom.
    # cap O outer bottom is yBase + pen.r (because ellipse_stroke expands by pen.r).
    # So we set bowl center so: cy + ry + pen.r == yBase + pen.r  -> cy = yBase - ry
    six_cy = yBase - six_ry

    six_bowl = pen.ellipse_stroke(six_cx, six_cy, six_rx, six_ry)

    # Join at left-middle of bowl
    join_x, join_y = (six_cx - six_rx, six_cy)

    # Terminal ellipse that connects at leftmost point (angle 180)
    term_rx = six_rx * 1.16
    term_cx = join_x + term_rx
    term_cy = join_y

    # IMPORTANT:
    # Make the OUTER top match cap "O" outer top.
    # cap O outer top is yTop - pen.r, and terminal outer top is term_cy - term_ry - pen.r
    # Set equal -> term_ry = term_cy - yTop
    term_ry = max(six_ry * 1.05, term_cy - yTop)

    # Terminal continues less long
    TERMINAL_END_DEG = 302.0   # try 295..310

    # Force the arc to include 270° apex so it truly reaches the top
    pts1 = ellipse_arc_points(term_cx, term_cy, term_rx, term_ry, 180.0, 270.0, clockwise=False, steps=140)
    pts2 = ellipse_arc_points(term_cx, term_cy, term_rx, term_ry, 270.0, TERMINAL_END_DEG, clockwise=False, steps=140)[1:]
    six_term = pen.line(pts1 + pts2)

    glyphs["6"] = (pen.union(six_bowl, six_term), W)


    # 7 (top bar + diagonal, normalized)
    seven_top = pen.hline(xL, xR, yTop + ory * 0.08)
    seven_diag = pen.line([(xR, yTop + ory * 0.08), (xL + orx * 0.18, yBase)])
    glyphs["7"] = (pen.union(seven_top, seven_diag), W)

    # 8 (centerlines separate, stroke outlines overlap slightly => one solid figure-8)
    rx8_top, ry8_top = orx * 0.94, ory * 0.40
    rx8_bot, ry8_bot = orx * 1.02, ory * 0.46

    overlap_outline = pen.r * 0.80
    dy = (ry8_top + ry8_bot) + 2.0 * pen.r - overlap_outline

    cy8_top = yMid - dy / 2.0
    cy8_bot = yMid + dy / 2.0

    eight = pen.union(
        pen.ellipse_stroke(cx, cy8_top, rx8_top, ry8_top),
        pen.ellipse_stroke(cx, cy8_bot, rx8_bot, ry8_bot),
    )

    glyphs["8"] = (eight, W)


    # 9 = 6 rotated 180° (exactly), around the digit box center
    g6, _w = glyphs["6"]
    origin = (W / 2.0, (m.CAP_TOP + m.BASE) / 2.0)  # center of cap-height box
    g9 = affinity.rotate(g6, 180.0, origin=origin)
    glyphs["9"] = (g9, W)



    for ch in "0123456789":
        glyphs.setdefault(ch, (Polygon(), W))

    return glyphs



def build_lowercase(m: Metrics, pen: Mono) -> Dict[str, Tuple[Geom, float]]:
    W = m.LC_W
    xL, xR = 110.0, W - 110.0
    cx = W / 2.0

    yBase = m.BASE
    yXTop = m.X_TOP
    yMid = m.X_MID
    yAsc = m.CAP_TOP
    yDesc = m.DESC_END

    rx = (xR - xL) * 0.42
    ry = (yBase - yXTop) * 0.48
    bcX = cx
    bcY = yMid + 10.0

    dot_r = pen.r * 0.95
    dot_y = yXTop - 160.0

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

    glyphs["c"] = (pen.ellipse_arc(bcX, bcY, rx, ry, 45.0, 315.0, steps=240), W)

    # a: one continuous stroke (terminal + stem + bowl + bar)
    a_p0  = (146.0, 400.0)

    a_c01 = (205.0, 320.0)
    a_c02 = (265.0, 310.0)
    a_p1  = (312.0, 315.0)

    a_c11 = (345.0, 318.0)
    a_c12 = (360.0, 335.0)
    a_p2  = (360.0, 360.0)

    seg0 = cubic_points(a_p0, a_c01, a_c02, a_p1, steps=50)
    seg1 = cubic_points(a_p1, a_c11, a_c12, a_p2, steps=30)

    stem_down = [(360.0, 770.0)]
    bottom_in = [(256.0, 770.0)]

    a_p3  = (146.0, 665.0)
    a_c21 = (195.25, 770.0)
    a_c22 = (146.0, 723.0)

    a_p4  = (256.0, 560.0)
    a_c31 = (146.0, 607.0)
    a_c32 = (195.25, 560.0)

    seg2 = cubic_points(bottom_in[-1], a_c21, a_c22, a_p3, steps=35)
    seg3 = cubic_points(a_p3, a_c31, a_c32, a_p4, steps=35)

    bar_out = [(360.0, 560.0)]

    a_pts = (
        seg0 +
        seg1[1:] +
        stem_down +
        bottom_in +
        seg2[1:] +
        seg3[1:] +
        bar_out
    )

    glyphs["a"] = (pen.line(a_pts), W)

    b_stem_x = xL + 40.0
    b_top = yXTop + 15.0
    b_bot = yBase - 10.0
    b_rx = (xR - b_stem_x) * 0.52
    glyphs["b"] = (dshape_stem_bowl(
        stem_x=b_stem_x, top_y=b_top, bot_y=b_bot,
        side="right", stem_top=yAsc, stem_bot=yBase,
        bowl_rx=b_rx, overlap=10.0,
    ), W)

    d_stem_x = xR - 40.0
    d_top = yXTop + 15.0
    d_bot = yBase - 10.0
    d_rx = (d_stem_x - xL) * 0.52
    glyphs["d"] = (dshape_stem_bowl(
        stem_x=d_stem_x, top_y=d_top, bot_y=d_bot,
        side="left", stem_top=yAsc, stem_bot=yBase,
        bowl_rx=d_rx, overlap=10.0,
    ), W)

    # e (single continuous stroke: bar -> long ellipse arc)
    e_cx, e_cy = bcX, bcY
    e_rx, e_ry = rx, ry
    e_bar_y = yXTop + 180.0

    s = (e_bar_y - e_cy) / e_ry
    s = max(-1.0, min(1.0, s))
    a_start = math.degrees(math.asin(s)) % 360.0
    a_end = 70.0

    bar_end = ellipse_point(e_cx, e_cy, e_rx, e_ry, a_start)
    bar_start = (e_cx - e_rx * 0.56, e_bar_y)

    arc_pts = ellipse_arc_points(
        e_cx, e_cy, e_rx, e_ry,
        a_start, a_end,
        clockwise=True,
        steps=220
    )

    e_pts = [bar_start, bar_end] + arc_pts[1:]
    glyphs["e"] = (pen.line(e_pts), W)

    # f (stem + top hook as ONE smooth continuous stroke)
    fx = cx - 65.0
    f_top = yAsc + 10.0
    f_bot = yBase - 10.0

    hook_rx = 150.0
    hook_ry = 80.0
    hook_cx = fx + hook_rx
    hook_cy = f_top

    # End while still visually "up" (300–340 is a good range)
    hook_end_deg = 330.0

    hook_pts = ellipse_arc_points(
        hook_cx, hook_cy, hook_rx, hook_ry,
        180.0, hook_end_deg,
        clockwise=False,   # <-- key change: makes it rise first in y-down coords
        steps=160
    )
    hook_pts[0] = (fx, f_top)  # exact join

    f_stem_hook = pen.line([(fx, f_bot), (fx, f_top)] + hook_pts[1:])

    f_cross_y = yXTop + 110.0
    f_cross = pen.hline(fx, fx + 235.0, f_cross_y)

    glyphs["f"] = (pen.union(f_stem_hook, f_cross), W)


    # g
    g_cx, g_cy = bcX - 20.0, bcY - 10.0
    g_rx, g_ry = rx * 0.88, ry * 0.88
    g_bowl = pen.ellipse_stroke(g_cx, g_cy, g_rx, g_ry)
    g_stem_x = g_cx + g_rx * 0.98
    g_stem_y0 = g_cy + g_ry * 0.20
    g_stem_y1 = yDesc - 70.0
    g_stem = pen.vline(g_stem_x, g_stem_y0, g_stem_y1)
    g_hook = pen.arc(g_stem_x - 120.0, g_stem_y1, 120.0, 0.0, 180.0, steps=180)
    glyphs["g"] = (pen.union(g_bowl, g_stem, g_hook), W)

    # h
    # h (only the TOP-RIGHT shoulder bends)
    hxL = xL + 40.0
    hxR = xR - 40.0
    h_top_y = yXTop + 20.0

    h_left = pen.vline(hxL, yAsc, yBase)

    # Curve tuning
    curve_dx = (hxR - hxL) * 0.40         # how long the shoulder runs before turning down
    curve_dy = (yBase - h_top_y) * 0.28   # how far it drops while turning into the right stem
    k = 0.62

    # Keep bend reasonable
    curve_dx = min(curve_dx, (hxR - hxL) - pen.r * 0.25)
    curve_dy = min(curve_dy, (yBase - h_top_y) - pen.r * 0.25)

    p0 = (hxR - curve_dx, h_top_y)        # start of bend on the shoulder (horizontal)
    p3 = (hxR, h_top_y + curve_dy)        # end of bend on right stem (vertical down)

    # cubic: horizontal tangent at start, vertical tangent at end
    c1 = (p0[0] + curve_dx * k, p0[1])
    c2 = (p3[0], p3[1] - curve_dy * k)

    bend = cubic_points(p0, c1, c2, p3, steps=90)

    h_shoulder_pts: List[Tuple[float, float]] = []
    h_shoulder_pts += [(hxL, h_top_y), p0]   # straight shoulder into curve
    h_shoulder_pts += bend[1:]              # smooth bend into the right stem
    h_shoulder_pts += [(hxR, yBase)]         # continue down (tangent matches)

    h_shoulder = pen.line(h_shoulder_pts)

    glyphs["h"] = (pen.union(h_left, h_shoulder), W)


    # i
    ix = cx
    glyphs["i"] = (pen.union(pen.vline(ix, yXTop + 20.0, yBase), pen.dot(ix, dot_y, dot_r)), W)
    # j
    glyphs["j"] = (pen.union(pen.vline(ix, yXTop + 20.0, yDesc - 10.0), pen.dot(ix, dot_y, dot_r)), W)

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

    # m
    n_aperture = (xR - 40.0) - (xL + 40.0)

    Wm = W + n_aperture
    xLm, xRm = 110.0, Wm - 110.0

    m_x1 = xLm + 40.0
    m_x2 = m_x1 + n_aperture
    m_x3 = m_x2 + n_aperture
    m_top = yXTop + 20.0

    # Only curve the RIGHTMOST shoulder (into m_x3)
    curve_dx = n_aperture * 0.55
    curve_dy = (yBase - m_top) * 0.38
    k = 0.62

    # Safety: ensure the curve starts to the right of m_x2
    curve_dx = min(curve_dx, (m_x3 - m_x2) - pen.r * 0.25)

    p0 = (m_x3 - curve_dx, m_top)          # start of bend on top bar
    p3 = (m_x3, m_top + curve_dy)          # end of bend on the right stem

    # cubic: horizontal tangent at start, vertical tangent at end
    c1 = (p0[0] + curve_dx * k, p0[1])
    c2 = (p3[0], p3[1] - curve_dy * k)

    shoulder = cubic_points(p0, c1, c2, p3, steps=90)

    m_pts: List[Tuple[float, float]] = []
    m_pts += [(m_x1, yBase), (m_x1, m_top)]                 # left stem up
    m_pts += [(m_x2, m_top), (m_x2, yBase), (m_x2, m_top)]  # middle stem: top corner stays "hard"
    m_pts += [p0]                                           # straight top bar to curve start
    m_pts += shoulder[1:]                                   # curve into right stem
    m_pts += [(m_x3, yBase)]                                # right stem down

    glyphs["m"] = (pen.line(m_pts), Wm)


    # n
    n_x1 = xL + 40.0
    n_x2 = xR - 40.0
    n_top = yXTop + 20.0

    # Make the top-right a real shoulder curve (horizontal -> vertical),
    # instead of a 90° corner that only gets rounded by join_style.
    curve_dx = 170.0   # how long the shoulder runs horizontally before turning down
    curve_dy = 160.0   # how far it drops while turning into the right stem
    k = 0.62           # control strength (0.55–0.70 is a useful tuning range)

    p0 = (n_x2 - curve_dx, n_top)          # start of the bend (end of the top bar)
    p3 = (n_x2, n_top + curve_dy)          # end of the bend (start of the right stem)

    # cubic with horizontal tangent at start and vertical tangent at end
    c1 = (p0[0] + curve_dx * k, p0[1])
    c2 = (p3[0], p3[1] - curve_dy * k)

    shoulder = cubic_points(p0, c1, c2, p3, steps=90)

    n_pts = [(n_x1, yBase), (n_x1, n_top), p0] + shoulder[1:] + [(n_x2, yBase)]
    glyphs["n"] = (pen.line(n_pts), W)


    # o
    glyphs["o"] = (pen.ellipse_stroke(bcX, bcY, rx, ry), W)

    desc_len = (yXTop - yAsc)
    pq_stem_bot = yBase + desc_len

    p_stem_x = xL + 40.0
    p_top = yXTop + 15.0
    p_bot = yBase - 10.0
    p_rx = (xR - p_stem_x) * 0.52
    glyphs["p"] = (dshape_stem_bowl(
        stem_x=p_stem_x, top_y=p_top, bot_y=p_bot,
        side="right", stem_top=p_top, stem_bot=pq_stem_bot,
        bowl_rx=p_rx, overlap=10.0,
    ), W)

    q_stem_x = xR - 40.0
    q_top = yXTop + 15.0
    q_bot = yBase - 10.0
    q_rx = (q_stem_x - xL) * 0.52
    glyphs["q"] = (dshape_stem_bowl(
        stem_x=q_stem_x, top_y=q_top, bot_y=q_bot,
        side="left", stem_top=q_top, stem_bot=pq_stem_bot,
        bowl_rx=q_rx, overlap=10.0,
    ), W)


    # r (stem + single smooth shoulder exiting the side, starting a bit lower)
    rx_stem = xL + 40.0
    r_stem_top = yXTop + 20.0

    # Shoulder start (lower than the stem top)
    r_start_y = r_stem_top + 120.0

    # Shoulder shape (matches the SVG we just approved)
    run = 280.0
    p0 = (rx_stem, r_start_y)
    p3 = (rx_stem + run, r_start_y - 30.0)  # slight drop on the right

    # Controls: leave p0 upward-ish, arrive at p3 gently downward
    c1 = (rx_stem + run * 0.34, r_start_y - 120.0)
    c2 = (rx_stem + run * 0.66, r_start_y - 130.0)

    r_pts = [(rx_stem, yBase), (rx_stem, r_stem_top), p0] + cubic_points(p0, c1, c2, p3, steps=90)[1:]
    glyphs["r"] = (pen.line(r_pts), W)



    s_y0 = yXTop + 35.0
    s_y3 = yBase - 25.0
    s_h = s_y3 - s_y0
    s_xL = xL + 90.0
    s_xR = xR - 90.0
    s_w = s_xR - s_xL

    sp0 = (s_xR, s_y0)
    sp1 = (s_xL, s_y0 + s_h * 0.30)
    sp2 = (s_xR, s_y0 + s_h * 0.64)
    sp3 = (s_xL, s_y3)

    sc01 = (s_xR - s_w * 0.33, s_y0)
    sc02 = (s_xL,              s_y0 + s_h * 0.12)
    sc11 = (s_xL,              s_y0 + s_h * 0.46)
    sc12 = (s_xR,              s_y0 + s_h * 0.40)
    sc21 = (s_xR,              s_y0 + s_h * 0.84)
    sc22 = (s_xL + s_w * 0.33, s_y3)

    s_pts = (
        cubic_points(sp0, sc01, sc02, sp1, steps=80) +
        cubic_points(sp1, sc11, sc12, sp2, steps=80)[1:] +
        cubic_points(sp2, sc21, sc22, sp3, steps=80)[1:]
    )
    glyphs["s"] = (pen.line(s_pts), W)

    tx = cx + 85.0
    t_top = yAsc + 10.0
    t_bot = yBase - 10.0
    t_cross_y = yXTop + 60.0
    t_left  = tx
    t_right = tx + 220.0

    glyphs["t"] = (pen.union(
        pen.vline(tx, t_top, t_bot),
        pen.hline(t_left, t_right, t_cross_y),
    ), W)

    # u
    ux1 = xL + 50.0
    ux2 = xR - 50.0
    u_top = yXTop + 20.0
    u_bot = yBase - 10.0

    # Only the LEFT bottom "shoulder" bends (left stem -> bottom bar).
    curve_dx = (ux2 - ux1) * 0.32
    curve_dy = (u_bot - u_top) * 0.34
    k = 0.62

    # Keep the bend reasonable
    curve_dx = min(curve_dx, (ux2 - ux1) - pen.r * 0.25)
    curve_dy = min(curve_dy, (u_bot - u_top) - pen.r * 0.25)

    # Corner is at (ux1, u_bot). Round it with a cubic from vertical -> horizontal.
    p0 = (ux1, u_bot - curve_dy)          # start of bend on left stem (going DOWN)
    p3 = (ux1 + curve_dx, u_bot)          # end of bend on bottom bar (going RIGHT)

    # cubic: vertical tangent at start, horizontal tangent at end
    c1 = (p0[0], p0[1] + curve_dy * k)
    c2 = (p3[0] - curve_dx * k, p3[1])

    bend = cubic_points(p0, c1, c2, p3, steps=90)

    u_pts: List[Tuple[float, float]] = []
    u_pts += [(ux1, u_top), p0]           # left stem down to bend start
    u_pts += bend[1:]                     # smooth bend into bottom bar
    u_pts += [(ux2, u_bot), (ux2, u_top)] # bottom bar to right, then right stem up (hard corner)

    glyphs["u"] = (pen.line(u_pts), W)


    # v
    glyphs["v"] = (pen.line([(xL + 60.0, yXTop + 20.0), (cx, yBase), (xR - 60.0, yXTop + 20.0)]), W)

    # w
    Ww = W + n_aperture
    xLw, xRw = 110.0, Ww - 110.0

    wx1 = xLw + 40.0
    wx2 = wx1 + n_aperture
    wx3 = wx2 + n_aperture
    w_top = yXTop + 20.0
    w_bot = yBase - 10.0

    # Only the LEFT bottom "shoulder" bends (left stem -> bottom bar).
    curve_dx = n_aperture * 0.55
    curve_dy = (w_bot - w_top) * 0.38
    k = 0.62

    # Keep the bend confined to the left bay (to the left of the middle stem)
    curve_dx = min(curve_dx, (wx2 - wx1) - pen.r * 0.25)
    curve_dy = min(curve_dy, (w_bot - w_top) - pen.r * 0.25)

    # Corner is at (wx1, w_bot). We round it with a cubic from vertical -> horizontal.
    p0 = (wx1, w_bot - curve_dy)          # start of bend on left stem (going DOWN)
    p3 = (wx1 + curve_dx, w_bot)          # end of bend on bottom bar (going RIGHT)

    # cubic: vertical tangent at start, horizontal tangent at end
    c1 = (p0[0], p0[1] + curve_dy * k)
    c2 = (p3[0] - curve_dx * k, p3[1])

    bend = cubic_points(p0, c1, c2, p3, steps=90)

    w_outer_pts: List[Tuple[float, float]] = []
    w_outer_pts += [(wx1, w_top), p0]         # left stem down to bend start
    w_outer_pts += bend[1:]                   # smooth bend into bottom bar
    w_outer_pts += [(wx3, w_bot), (wx3, w_top)]  # bottom bar to right, then right stem up

    w_outer = pen.line(w_outer_pts)
    w_mid = pen.vline(wx2, w_top, w_bot)

    glyphs["w"] = (pen.union(w_outer, w_mid), Ww)


    # x
    x1 = xL + 55.0
    x2 = xR - 55.0
    x_top = yXTop + 30.0
    x_bot = yBase - 10.0
    glyphs["x"] = (pen.union(pen.line([(x1, x_top), (x2, x_bot)]), pen.line([(x2, x_top), (x1, x_bot)])), W)

    # y
    yx1 = xL + 50.0
    yx2 = xR - 50.0
    y_top = yXTop + 20.0
    y_bot = yBase - 10.0

    desc_len = (yXTop - yAsc)
    y_desc_bot = yBase + desc_len

    # Only the LEFT bottom "shoulder" bends (left stem -> bottom bar).
    curve_dx = (yx2 - yx1) * 0.32
    curve_dy = (y_bot - y_top) * 0.34
    k = 0.62

    curve_dx = min(curve_dx, (yx2 - yx1) - pen.r * 0.25)
    curve_dy = min(curve_dy, (y_bot - y_top) - pen.r * 0.25)

    # Round the corner at (yx1, y_bot) with a cubic from vertical -> horizontal
    p0 = (yx1, y_bot - curve_dy)          # start of bend on left stem
    p3 = (yx1 + curve_dx, y_bot)          # end of bend on bottom bar

    c1 = (p0[0], p0[1] + curve_dy * k)    # vertical tangent at start
    c2 = (p3[0] - curve_dx * k, p3[1])    # horizontal tangent at end

    bend = cubic_points(p0, c1, c2, p3, steps=90)

    left_and_bar_pts: List[Tuple[float, float]] = []
    left_and_bar_pts += [(yx1, y_top), p0]
    left_and_bar_pts += bend[1:]
    left_and_bar_pts += [(yx2, y_bot)]

    left_and_bar = pen.line(left_and_bar_pts)
    right_stem = pen.vline(yx2, y_top, y_desc_bot)   # keep this junction hard

    glyphs["y"] = (pen.union(left_and_bar, right_stem), W)


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", type=str, default="", help="Generate only these characters (e.g. --chars n or --chars aen012)")
    ap.add_argument("--out", type=Path, default=Path("sketches"))
    ap.add_argument("--stroke", type=float, default=90.0)
    ap.add_argument("--resolution", type=int, default=64)
    args = ap.parse_args()

    m = Metrics()
    pen = Mono(stroke=args.stroke, resolution=args.resolution)

    upper = build_uppercase(m, pen)
    lower = build_lowercase(m, pen)
    digits = build_digits(m, pen)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    preview: List[Tuple[str, str]] = []

    chars = args.chars or ("ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "abcdefghijklmnopqrstuvwxyz" + "0123456789")

    for ch in chars:
        if ch in upper:
            g, w = upper[ch]
        elif ch in lower:
            g, w = lower[ch]
        elif ch in digits:
            g, w = digits[ch]
        else:
            continue

        fname = codepoint_filename(ch)
        write_svg(out / fname, w, m, g)
        preview.append((ch, fname))

    write_preview_html(out, preview)
    print(f"Wrote {len(preview)} SVGs to: {out}")
    print(f"Open: {out / 'preview.html'}")


if __name__ == "__main__":
    main()
