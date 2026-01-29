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
    """
    Lining digits built on the same cap metrics as uppercase:
    - width = CAP_W
    - top = CAP_TOP, baseline = BASE
    - monoline construction via pen.line/arc/ellipse_*
    """
    W = m.CAP_W
    xL, xR = 130.0, W - 130.0
    cx = W / 2.0
    yTop, yBase, yMid = m.CAP_TOP, m.BASE, m.CAP_MID

    rx = (xR - xL) / 2.0
    ry = (yBase - yTop) / 2.0

    glyphs: Dict[str, Tuple[Geom, float]] = {}

    # 0: like O
    glyphs["0"] = (pen.ellipse_stroke(cx, yMid, rx, ry), W)

    # 1: stem + base + small top flick
    one_x = cx + 30.0
    one_stem = pen.vline(one_x, yTop + 20.0, yBase)
    one_base = pen.hline(one_x - 110.0, one_x + 90.0, yBase)
    one_flick = pen.line([(one_x - 60.0, yTop + 90.0), (one_x, yTop + 20.0)])
    glyphs["1"] = (pen.union(one_stem, one_base, one_flick), W)

    # 2: top arc + diagonal + base
    yArc = yTop + 200.0
    rArc = (xR - xL) / 2.0
    two_top = pen.arc(cx, yArc, rArc, 180.0, 360.0, steps=180)
    two_diag = pen.line([(xR, yArc), (xL, yBase)])
    two_base = pen.hline(xL, xR, yBase)
    glyphs["2"] = (pen.union(two_top, two_diag, two_base), W)

    # 3: two right arcs + mid join
    top_cy = (yTop + yMid) / 2.0
    bot_cy = (yMid + yBase) / 2.0
    ryh = (yMid - yTop) / 2.0
    rx3 = rx * 0.98

    three_top = pen.ellipse_arc(cx, top_cy, rx3, ryh, 300.0, 60.0, steps=200)
    three_bot = pen.ellipse_arc(cx, bot_cy, rx3, ryh, 300.0, 60.0, steps=200)

    # small join on the right + mid bar to keep it reading as a "3"
    p_top_dn = ellipse_point(cx, top_cy, rx3, ryh, 300.0)  # down-right on top bowl
    p_bot_up = ellipse_point(cx, bot_cy, rx3, ryh, 60.0)   # up-right on bottom bowl
    three_join = pen.line([p_top_dn, p_bot_up])
    three_mid = pen.hline(cx - 20.0, xR, yMid)

    glyphs["3"] = (pen.union(three_top, three_bot, three_join, three_mid), W)

    # 4: diagonal + right stem + crossbar
    four_x = xR - 110.0
    four_y = yMid + 20.0
    four_diag = pen.line([(xL + 40.0, yTop + 80.0), (four_x, four_y)])
    four_stem = pen.vline(four_x, yTop, yBase)
    four_bar  = pen.hline(xL + 40.0, xR, four_y)
    glyphs["4"] = (pen.union(four_diag, four_stem, four_bar), W)

    # 5: top + left down + mid + bottom bowl-ish curve
    five_top = pen.hline(xL, xR, yTop)
    five_left = pen.vline(xL, yTop, yMid)
    five_mid = pen.hline(xL, xR - 40.0, yMid)

    bot_cy5 = (yMid + yBase) / 2.0
    ry5 = (yBase - yMid) / 2.0
    five_curve = pen.ellipse_arc(cx, bot_cy5, rx * 0.98, ry5 * 0.98, 200.0, 20.0, steps=240)
    five_base = pen.hline(xL, xR, yBase)

    glyphs["5"] = (pen.union(five_top, five_left, five_mid, five_curve, five_base), W)

    # 6: lower loop + top hook
    six_loop_cy = yMid + 90.0
    six_loop = pen.ellipse_stroke(cx, six_loop_cy, rx * 0.92, ry * 0.72)
    six_hook = pen.ellipse_arc(cx, yMid - 70.0, rx * 0.78, ry * 0.60, 210.0, 20.0, steps=240)
    glyphs["6"] = (pen.union(six_loop, six_hook), W)

    # 7: top bar + diagonal
    seven_top = pen.hline(xL, xR, yTop)
    seven_diag = pen.line([(xR, yTop), (xL + 40.0, yBase)])
    glyphs["7"] = (pen.union(seven_top, seven_diag), W)

    # 8: two stacked loops
    eight_top = pen.ellipse_stroke(cx, (yTop + yMid) / 2.0, rx * 0.82, (yMid - yTop) / 2.0 * 0.82)
    eight_bot = pen.ellipse_stroke(cx, (yMid + yBase) / 2.0, rx * 0.92, (yBase - yMid) / 2.0 * 0.92)
    glyphs["8"] = (pen.union(eight_top, eight_bot), W)

    # 9: top loop + right stem
    nine_loop = pen.ellipse_stroke(cx, yMid - 70.0, rx * 0.90, ry * 0.60)
    nine_stem_x = cx + rx * 0.90
    nine_stem = pen.vline(nine_stem_x, yMid - 70.0, yBase)
    glyphs["9"] = (pen.union(nine_loop, nine_stem), W)

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

    # f
    fx = cx - 65.0
    f_top = yAsc + 10.0
    f_bot = yBase - 10.0
    f_stem = pen.vline(fx, f_top, f_bot)

    hook_cx = fx + 110.0
    hook_cy = f_top + 35.0
    hook_rx = 110.0
    hook_ry = 50.0
    hook_pts = ellipse_arc_points(
        hook_cx, hook_cy, hook_rx, hook_ry,
        180.0, 25.0,
        clockwise=False,
        steps=120
    )
    f_hook = pen.line(hook_pts)

    f_cross_y = yXTop + 110.0
    f_cross = pen.hline(fx, fx + 235.0, f_cross_y)

    glyphs["f"] = (pen.union(f_stem, f_hook, f_cross), W)

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
    hxL = xL + 40.0
    hxR = xR - 40.0
    h_top_y = yXTop + 20.0
    h_left = pen.vline(hxL, yAsc, yBase)
    h_shoulder = pen.line([(hxL, h_top_y), (hxR, h_top_y), (hxR, yBase)])
    glyphs["h"] = (pen.union(h_left, h_shoulder), W)

    ix = cx
    glyphs["i"] = (pen.union(pen.vline(ix, yXTop + 20.0, yBase), pen.dot(ix, dot_y, dot_r)), W)
    glyphs["j"] = (pen.union(pen.vline(ix, yXTop + 20.0, yDesc - 10.0), pen.dot(ix, dot_y, dot_r)), W)

    kx = xL + 40.0
    ky_mid = (yXTop + yBase) / 2.0
    glyphs["k"] = (pen.union(
        pen.vline(kx, yAsc, yBase),
        pen.line([(kx, ky_mid), (xR - 10.0, yXTop + 20.0)]),
        pen.line([(kx, ky_mid), (xR - 10.0, yBase - 20.0)]),
    ), W)

    glyphs["l"] = (pen.vline(cx - 120.0, yAsc, yBase), W)

    n_aperture = (xR - 40.0) - (xL + 40.0)

    Wm = W + n_aperture
    xLm, xRm = 110.0, Wm - 110.0

    m_x1 = xLm + 40.0
    m_x2 = m_x1 + n_aperture
    m_x3 = m_x2 + n_aperture
    m_top = yXTop + 20.0

    m_pts = [
        (m_x1, yBase),
        (m_x1, m_top),
        (m_x2, m_top),
        (m_x2, yBase),
        (m_x2, m_top),
        (m_x3, m_top),
        (m_x3, yBase),
    ]
    glyphs["m"] = (pen.line(m_pts), Wm)

    n_x1 = xL + 40.0
    n_x2 = xR - 40.0
    n_top = yXTop + 20.0
    n_shape = pen.line([(n_x1, yBase), (n_x1, n_top), (n_x2, n_top), (n_x2, yBase)])
    glyphs["n"] = (n_shape, W)

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

    rx_stem = xL + 40.0
    r_top = yXTop + 20.0
    r_stem = pen.vline(rx_stem, r_top, yBase)
    r_arm_y = r_top + 60.0
    r_arm_x2 = rx_stem + 190.0
    r_arm = pen.hline(rx_stem, r_arm_x2, r_arm_y)
    r_drop = pen.vline(r_arm_x2, r_arm_y, r_arm_y + 70.0)
    glyphs["r"] = (pen.union(r_stem, r_arm, r_drop), W)

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

    ux1 = xL + 50.0
    ux2 = xR - 50.0
    u_top = yXTop + 20.0
    u_bot = yBase - 10.0
    glyphs["u"] = (pen.line([(ux1, u_top), (ux1, u_bot), (ux2, u_bot), (ux2, u_top)]), W)

    glyphs["v"] = (pen.line([(xL + 60.0, yXTop + 20.0), (cx, yBase), (xR - 60.0, yXTop + 20.0)]), W)

    Ww = W + n_aperture
    xLw, xRw = 110.0, Ww - 110.0

    wx1 = xLw + 40.0
    wx2 = wx1 + n_aperture
    wx3 = wx2 + n_aperture
    w_top = yXTop + 20.0
    w_bot = yBase - 10.0

    w_outer = pen.line([(wx1, w_top), (wx1, w_bot), (wx3, w_bot), (wx3, w_top)])
    w_mid = pen.vline(wx2, w_top, w_bot)
    glyphs["w"] = (pen.union(w_outer, w_mid), Ww)

    x1 = xL + 55.0
    x2 = xR - 55.0
    x_top = yXTop + 30.0
    x_bot = yBase - 10.0
    glyphs["x"] = (pen.union(pen.line([(x1, x_top), (x2, x_bot)]), pen.line([(x2, x_top), (x1, x_bot)])), W)

    yx1 = xL + 50.0
    yx2 = xR - 50.0
    y_top = yXTop + 20.0
    y_bot = yBase - 10.0

    desc_len = (yXTop - yAsc)
    y_desc_bot = yBase + desc_len

    y_shape = pen.union(
        pen.vline(yx1, y_top, y_bot),
        pen.hline(yx1, yx2, y_bot),
        pen.vline(yx2, y_top, y_desc_bot),
    )
    glyphs["y"] = (y_shape, W)

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

    for ch in "0123456789":
        g, w = digits[ch]
        fname = codepoint_filename(ch)
        write_svg(out / fname, w, m, g)
        preview.append((ch, fname))

    write_preview_html(out, preview)
    print(f"Wrote {len(preview)} SVGs to: {out}")
    print(f"Open: {out / 'preview.html'}")


if __name__ == "__main__":
    main()
