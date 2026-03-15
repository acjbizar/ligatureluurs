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


# ----------------------------
# Global metrics + tuning knobs
# ----------------------------

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


@dataclass(frozen=True)
class Tune:
    CAP_INSET: float = 130.0
    DIGIT_INSET: float = 150.0
    LC_INSET: float = 110.0
    STEM_INSET: float = 40.0

    K: float = 0.62
    CURVE_STEPS: int = 90

    ARC_STEPS: int = 240
    ELLIPSE_ARC_STEPS: int = 260

    ROUND_WIDEN: float = 1.14
    DIGIT_OPTICAL: float = 0.96
    SAFE_MARGIN: float = 20.0

    DOT_GAP: float = 160.0


T = Tune()


# ----------------------------
# Geometry helpers
# ----------------------------

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
        x = (mt**3) * x0 + 3 * (mt**2) * t * x1 + 3 * mt * (t**2) * x2 + (t**3) * x3
        y = (mt**3) * y0 + 3 * (mt**2) * t * y1 + 3 * mt * (t**2) * y2 + (t**3) * y3
        pts.append((x, y))
    return pts


def s_wiggle_points(
    xL: float, xR: float,
    yTop: float, yBase: float,
    amp: float = 0.46,
    taper: float = 1.7,
    steps: int = 260
) -> List[Tuple[float, float]]:
    """
    Retained for experimentation only; not used by default.
    """
    cx = (xL + xR) / 2.0
    w = (xR - xL)
    h = (yBase - yTop)

    A = (w / 2.0) * amp
    A = min(A, (w / 2.0) - 1.0)

    pts: List[Tuple[float, float]] = []
    for i in range(steps):
        t = i / (steps - 1)
        y = yTop + h * t
        env = math.sin(math.pi * t) ** taper
        x = cx + A * math.sin(2.0 * math.pi * t) * env
        pts.append((x, y))
    return pts


def norm(vx: float, vy: float) -> Tuple[float, float]:
    n = math.hypot(vx, vy)
    if n == 0:
        return (0.0, 0.0)
    return (vx / n, vy / n)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def safe_round_rx(W: float, pen_r: float, rx: float, safe_margin: float) -> float:
    return min(rx, (W / 2.0) - pen_r - safe_margin)


def corner_h_to_v(
    p0: Tuple[float, float],
    dx: float,
    dy: float,
    k: float = T.K,
    steps: int = T.CURVE_STEPS
) -> List[Tuple[float, float]]:
    x0, y0 = p0
    p3 = (x0 + dx, y0 + dy)
    c1 = (x0 + dx * k, y0)
    c2 = (p3[0], p3[1] - dy * k)
    return cubic_points(p0, c1, c2, p3, steps=steps)


def corner_v_to_h(
    p0: Tuple[float, float],
    dx: float,
    dy: float,
    k: float = T.K,
    steps: int = T.CURVE_STEPS
) -> List[Tuple[float, float]]:
    x0, y0 = p0
    p3 = (x0 + dx, y0 + dy)
    c1 = (x0, y0 + dy * k)
    c2 = (p3[0] - dx * k, p3[1])
    return cubic_points(p0, c1, c2, p3, steps=steps)


# ----------------------------
# S construction
# ----------------------------

def cubic_chain_points(
    p0: Tuple[float, float],
    segments: List[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]],
    steps: int = 72,
) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = [p0]
    cur = p0
    for c1, c2, p3 in segments:
        seg = cubic_points(cur, c1, c2, p3, steps=steps)
        pts += seg[1:]
        cur = p3
    return pts


S_CAP_SPEC = {
    "start": (0.965, 0.086),
    "segments": [
        ((0.848, 0.004), (0.649, 0.000), (0.461, 0.044)),
        ((0.251, 0.095), (0.157, 0.260), (0.286, 0.378)),
        ((0.415, 0.494), (0.696, 0.494), (0.859, 0.567)),
        ((1.000, 0.629), (0.988, 0.781), (0.848, 0.878)),
        ((0.707, 0.973), (0.485, 1.000), (0.281, 0.962)),
        ((0.169, 0.940), (0.070, 0.896), (0.000, 0.841)),
    ],
}

S_LC_SPEC = {
    "start": (0.974, 0.121),
    "segments": [
        ((0.860, 0.022), (0.682, 0.000), (0.503, 0.045)),
        ((0.325, 0.093), (0.244, 0.247), (0.357, 0.360)),
        ((0.471, 0.472), (0.714, 0.472), (0.860, 0.551)),
        ((1.000, 0.624), (0.990, 0.778), (0.844, 0.876)),
        ((0.698, 0.972), (0.477, 1.000), (0.279, 0.958)),
        ((0.169, 0.935), (0.071, 0.888), (0.000, 0.826)),
    ],
}


def s_from_spec(
    spec: dict,
    xL: float, xR: float,
    yTop: float, yBase: float,
    steps: int = 72,
) -> List[Tuple[float, float]]:
    def P(pt: Tuple[float, float]) -> Tuple[float, float]:
        x, y = pt
        return (
            xL + (xR - xL) * x,
            yTop + (yBase - yTop) * y,
        )

    p0 = P(spec["start"])
    segments = [(P(c1), P(c2), P(p3)) for c1, c2, p3 in spec["segments"]]
    return cubic_chain_points(p0, segments, steps=steps)


# ----------------------------
# Stroke pen
# ----------------------------

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


# ----------------------------
# SVG output
# ----------------------------

def fmt(x: float) -> str:
    return f"{x:.3f}"


def codepoint_filename(s: str) -> str:
    cps = [f"u{ord(ch):04x}" for ch in s]
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
        f'viewBox="0 0 {fmt(width)} {m.H}">\n'
        f'  <path d="{d}" fill="black" fill-rule="evenodd"/>\n'
        f'</svg>\n'
    )
    out_path.write_text(svg, encoding="utf-8")


# ----------------------------
# Shared “frames”
# ----------------------------

@dataclass(frozen=True)
class CapFrame:
    W: float
    xL: float
    xR: float
    cx: float
    yTop: float
    yBase: float
    yMid: float
    rx: float
    ry: float
    round_rx: float
    round_ry: float


@dataclass(frozen=True)
class DigitFrame:
    W: float
    xL: float
    xR: float
    cx: float
    yMid: float
    rx: float
    ry: float
    orx: float
    ory: float
    yTop: float
    yBase: float


@dataclass(frozen=True)
class LCFrame:
    W: float
    xL: float
    xR: float
    cx: float
    yBase: float
    yXTop: float
    yMid: float
    yAsc: float
    yDesc: float
    bowl_rx: float
    bowl_ry: float
    bowl_cx: float
    bowl_cy: float


def make_cap_frame(m: Metrics, pen: Mono) -> CapFrame:
    W = m.CAP_W
    xL, xR = T.CAP_INSET, W - T.CAP_INSET
    cx = W / 2.0
    yTop, yBase, yMid = m.CAP_TOP, m.BASE, m.CAP_MID
    rx = (xR - xL) / 2.0
    ry = (yBase - yTop) / 2.0

    round_rx = safe_round_rx(W, pen.r, rx * T.ROUND_WIDEN, T.SAFE_MARGIN)
    round_ry = ry
    return CapFrame(W, xL, xR, cx, yTop, yBase, yMid, rx, ry, round_rx, round_ry)


def make_digit_frame(m: Metrics) -> DigitFrame:
    W = m.CAP_W
    xL, xR = T.DIGIT_INSET, W - T.DIGIT_INSET
    cx = (xL + xR) / 2.0

    yCapTop = m.CAP_TOP
    yCapBase = m.BASE
    yMid = (yCapTop + yCapBase) / 2.0

    rx = (xR - xL) / 2.0
    ry = (yCapBase - yCapTop) / 2.0

    orx = rx * T.DIGIT_OPTICAL
    ory = ry * T.DIGIT_OPTICAL
    yTop = yMid - ory
    yBase = yMid + ory

    return DigitFrame(W, xL, xR, cx, yMid, rx, ry, orx, ory, yTop, yBase)


def make_lc_frame(m: Metrics) -> LCFrame:
    W = m.LC_W
    xL, xR = T.LC_INSET, W - T.LC_INSET
    cx = W / 2.0

    yBase = m.BASE
    yXTop = m.X_TOP
    yMid = m.X_MID
    yAsc = m.CAP_TOP
    yDesc = m.DESC_END

    bowl_rx = (xR - xL) * 0.42
    bowl_ry = (yBase - yXTop) * 0.48
    bowl_cx = cx
    bowl_cy = yMid + 10.0

    return LCFrame(W, xL, xR, cx, yBase, yXTop, yMid, yAsc, yDesc, bowl_rx, bowl_ry, bowl_cx, bowl_cy)


def stem_bowl(
    pen: Mono,
    stem_x: float,
    top_y: float,
    bot_y: float,
    side: str,
    stem_top: float,
    stem_bot: float,
    bowl_rx: float,
    overlap: float = 10.0,
) -> Geom:
    cy = (top_y + bot_y) / 2.0
    by = (bot_y - top_y) / 2.0

    if side == "right":
        cx0 = stem_x + bowl_rx - overlap
        arc = pen.ellipse_arc(cx0, cy, bowl_rx, by, 270.0, 90.0, steps=T.ELLIPSE_ARC_STEPS)
        top_conn = pen.hline(stem_x, cx0, top_y)
        bot_conn = pen.hline(cx0, stem_x, bot_y)
    else:
        cx0 = stem_x - bowl_rx + overlap
        arc = pen.ellipse_arc(cx0, cy, bowl_rx, by, 90.0, 270.0, steps=T.ELLIPSE_ARC_STEPS)
        top_conn = pen.hline(cx0, stem_x, top_y)
        bot_conn = pen.hline(stem_x, cx0, bot_y)

    stem = pen.vline(stem_x, stem_top, stem_bot)
    return pen.union(stem, top_conn, arc, bot_conn)


# ----------------------------
# Uppercase
# ----------------------------

def build_uppercase(m: Metrics, pen: Mono) -> Dict[str, Tuple[Geom, float]]:
    f = make_cap_frame(m, pen)
    W = f.W
    xL, xR, cx = f.xL, f.xR, f.cx
    yTop, yBase, yMid = f.yTop, f.yBase, f.yMid

    CAP_RX = f.rx
    CAP_RY = f.ry
    ROUND_RX = f.round_rx
    ROUND_RY = f.round_ry

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

    glyphs["C"] = (pen.ellipse_arc(cx, yMid, ROUND_RX, ROUND_RY, 45.0, 315.0, steps=280), W)

    xJoin = cx
    glyphs["D"] = (pen.union(
        pen.vline(xL, yTop, yBase),
        pen.hline(xL, xJoin, yTop),
        pen.ellipse_arc(xJoin, yMid, xR - xJoin, CAP_RY, 270.0, 90.0, steps=240),
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

    g_a0 = 300.0
    g_a1 = 60.0
    G_RX = ROUND_RX
    G_RY = ROUND_RY

    g_end_y = yMid + 110.0
    g_end_sin = (g_end_y - yMid) / G_RY
    g_end_sin = clamp(g_end_sin, -1.0, 1.0)
    g_a2 = math.degrees(math.asin(g_end_sin))

    arc1_pts = ellipse_arc_points(cx, yMid, G_RX, G_RY, g_a0, g_a1, clockwise=True, steps=220)
    arc2_pts = ellipse_arc_points(cx, yMid, G_RX, G_RY, g_a1, g_a2, clockwise=True, steps=90)

    p0 = arc2_pts[-1]
    p3 = (cx + G_RX * 0.773, yMid)
    p4 = (cx + 35.0, yMid)

    a = math.radians(g_a2)
    tvx, tvy = (math.sin(a) * G_RX, -math.cos(a) * G_RY)
    ux, uy = norm(tvx, tvy)

    tlen1 = G_RX * 0.28
    tlen2 = G_RX * 0.22

    p1 = (p0[0] + ux * tlen1, p0[1] + uy * tlen1)
    p2 = (p3[0] + tlen2, p3[1])

    bez_pts = cubic_points(p0, p1, p2, p3, steps=70)

    g_outer = pen.line(arc1_pts + arc2_pts[1:])
    g_join = pen.line(bez_pts)
    g_bar = pen.hline(p4[0], p3[0], yMid)
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

    glyphs["L"] = (pen.union(
        pen.vline(xL, yTop, yBase),
        pen.hline(xL, xR, yBase)
    ), W)

    MW_W = W + 120.0
    mxL = T.CAP_INSET
    mxR = MW_W - T.CAP_INSET
    mcx = MW_W / 2.0
    glyphs["M"] = (pen.union(
        pen.vline(mxL, yTop, yBase),
        pen.vline(mxR, yTop, yBase),
        pen.line([(mxL, yTop), (mcx, yMid), (mxR, yTop)]),
    ), MW_W)

    glyphs["N"] = (pen.union(
        pen.vline(xL, yTop, yBase),
        pen.vline(xR, yTop, yBase),
        pen.line([(xL, yTop), (xR, yBase)]),
    ), W)

    O = pen.ellipse_stroke(cx, yMid, ROUND_RX, ROUND_RY)
    glyphs["O"] = (O, W)

    glyphs["P"] = (pen.union(
        pen.vline(xL, yTop, yBase),
        pen.hline(xL, xFlat, yTop),
        pen.arc(xFlat, yTop + bowl_r, bowl_r, 270.0, 90.0, steps=150),
        pen.hline(xFlat, xL, yMid),
    ), W)

    q_tail = pen.line([
        (cx + ROUND_RX * 0.10, yMid + ROUND_RY * 0.38),
        (cx + ROUND_RX * 0.72, yMid + ROUND_RY * 0.96),
    ])
    glyphs["Q"] = (pen.union(O, q_tail), W)

    glyphs["R"] = (pen.union(
        glyphs["P"][0],
        pen.line([(xFlat, yMid), (xR, yBase)])
    ), W)

    S_xL = xL + 12.0
    S_xR = xR - 12.0
    S_y0 = yTop + 8.0
    S_y3 = yBase - 8.0
    S_pts = s_from_spec(S_CAP_SPEC, S_xL, S_xR, S_y0, S_y3, steps=84)
    glyphs["S"] = (pen.line(S_pts), W)

    glyphs["T"] = (pen.union(
        pen.hline(xL, xR, yTop),
        pen.vline(cx, yTop, yBase)
    ), W)

    u_end = 560.0
    glyphs["U"] = (pen.union(
        pen.vline(xL, yTop, u_end),
        pen.arc(cx, u_end, (xR - xL) / 2.0, 0.0, 180.0, steps=180),
        pen.vline(xR, u_end, yTop),
    ), W)

    glyphs["V"] = (pen.line([(xL, yTop), (cx, yBase), (xR, yTop)]), W)

    WW_W = W + 120.0
    wxL = T.CAP_INSET
    wxR = WW_W - T.CAP_INSET
    wcx = WW_W / 2.0
    wq1 = wxL + (wxR - wxL) * 0.24
    wq3 = wxR - (wxR - wxL) * 0.24
    glyphs["W"] = (pen.line([
        (wxL, yTop),
        (wq1, yBase),
        (wcx, yMid),
        (wq3, yBase),
        (wxR, yTop),
    ]), WW_W)

    glyphs["X"] = (pen.union(
        pen.line([(xL, yTop), (xR, yBase)]),
        pen.line([(xR, yTop), (xL, yBase)])
    ), W)

    glyphs["Y"] = (pen.union(
        pen.line([(xL, yTop), (cx, yMid), (xR, yTop)]),
        pen.vline(cx, yMid, yBase)
    ), W)

    glyphs["Z"] = (pen.union(
        pen.hline(xL, xR, yTop),
        pen.line([(xR, yTop), (xL, yBase)]),
        pen.hline(xL, xR, yBase)
    ), W)

    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        glyphs.setdefault(ch, (Polygon(), W))

    return glyphs


# ----------------------------
# Digits
# ----------------------------

def build_digits(m: Metrics, pen: Mono) -> Dict[str, Tuple[Geom, float]]:
    f = make_digit_frame(m)
    W = f.W
    xL, xR, cx = f.xL, f.xR, f.cx
    yTop, yBase, yMid = f.yTop, f.yBase, f.yMid
    orx, ory = f.orx, f.ory

    glyphs: Dict[str, Tuple[Geom, float]] = {}

    # ------------------------------------------------------------------
    # Shared optical frame for all digits
    # ------------------------------------------------------------------
    yD_top = yTop + ory * 0.08
    yD_bot = yBase - ory * 0.06
    yD_mid = (yD_top + yD_bot) / 2.0
    hD = yD_bot - yD_top

    # Rounded digits need a little overshoot to look equal in height.
    round_overshoot = hD * 0.015
    yR_top = yD_top - round_overshoot
    yR_bot = yD_bot + round_overshoot
    yR_mid = (yR_top + yR_bot) / 2.0
    hR = yR_bot - yR_top

    xD_left = xL + orx * 0.04
    xD_right = xR - orx * 0.04
    wD = xD_right - xD_left

    # 0 ----------------------------------------------------------------
    rx0 = wD * 0.50
    ry0 = hR * 0.50
    glyphs["0"] = (pen.ellipse_stroke(cx, yR_mid, rx0, ry0), W)

    # 1 ----------------------------------------------------------------
    y1_top = yD_top
    y1_bot = yD_bot
    x1 = cx

    flag_len = wD * 0.24
    one_flag = pen.hline(x1 - flag_len, x1, y1_top)
    one_stem = pen.vline(x1, y1_top, y1_bot)

    base_half = max(flag_len * 1.02, wD * 0.18)
    one_base = pen.hline(x1 - base_half, x1 + base_half, y1_bot)
    glyphs["1"] = (pen.union(one_flag, one_stem, one_base), W)

    # 2 ----------------------------------------------------------------
    TWO_WIDEN = 0.14
    pad2 = wD * TWO_WIDEN

    xL2 = max(pen.r + 2.0, xD_left - pad2)
    xR2 = min(W - pen.r - 2.0, xD_right + pad2)

    def X(frac: float) -> float:
        return xL2 + (xR2 - xL2) * frac

    def Y(t: float) -> float:
        return yD_top + hD * t

    p0  = (X(190 / 700), Y((260 - 40) / 740))
    c01 = (X(240 / 700), Y((150 - 40) / 740))
    c02 = (X(420 / 700), Y((130 - 40) / 740))
    p1  = (X(500 / 700), Y((220 - 40) / 740))

    c11 = (X(560 / 700), Y((290 - 40) / 740))
    c12 = (X(520 / 700), Y((390 - 40) / 740))
    p2  = (X(420 / 700), Y((460 - 40) / 740))

    c21 = (X(350 / 700), Y((510 - 40) / 740))
    c22 = (X(300 / 700), Y((540 - 40) / 740))
    p3  = (X(250 / 700), Y((610 - 40) / 740))

    p4  = (X(150 / 700), yD_bot)
    p5  = (X(530 / 700), yD_bot)

    shape_pts = [p0, c01, c02, p1, c11, c12, p2, c21, c22, p3]
    y_min = min(y for _, y in shape_pts)
    target_top = yD_top + pen.r * 0.18

    s = (yD_bot - target_top) / (yD_bot - y_min)
    s = max(1.0, min(s, 1.42))

    def SY(pt):
        x, y = pt
        return (x, yD_bot - (yD_bot - y) * s)

    p0, c01, c02, p1, c11, c12, p2, c21, c22, p3 = map(
        SY, [p0, c01, c02, p1, c11, c12, p2, c21, c22, p3]
    )

    seg0 = cubic_points(p0, c01, c02, p1, steps=90)
    seg1 = cubic_points(p1, c11, c12, p2, steps=90)[1:]
    seg2 = cubic_points(p2, c21, c22, p3, steps=90)[1:]

    two_pts = seg0 + seg1 + seg2 + [p4, p5]
    glyphs["2"] = (pen.line(two_pts), W)

    # 3 ----------------------------------------------------------------
    pad_x3 = orx * 0.10
    pad_y3 = ory * 0.02

    x3_left = xL + pad_x3
    x3_right = xR - pad_x3 * 0.70

    y3_top = yD_top + pad_y3
    y3_bot = yD_bot
    y3_mid = (y3_top + y3_bot) / 2.0

    w3 = x3_right - x3_left
    r3 = w3 * 0.44
    cx3r = x3_right - r3

    u_cy = (y3_top + y3_mid) / 2.0
    u_ry = (y3_mid - y3_top) / 2.0
    l_cy = (y3_mid + y3_bot) / 2.0
    l_ry = (y3_bot - y3_mid) / 2.0

    three_u = pen.line(
        ellipse_arc_points(cx3r, u_cy, r3, u_ry, 270.0, 90.0, clockwise=False, steps=260)
    )
    three_l = pen.line(
        ellipse_arc_points(cx3r, l_cy, r3, l_ry, 270.0, 90.0, clockwise=False, steps=260)
    )
    three_right = pen.union(three_u, three_l)

    LEFT_SCALE = 0.72
    TOP_END_DEG = 208.0
    BOT_END_DEG = 152.0

    rxL_top = r3 * LEFT_SCALE
    ryL_top = u_ry * LEFT_SCALE
    rxL_bot = r3 * LEFT_SCALE
    ryL_bot = l_ry * LEFT_SCALE

    x_kink = x3_left + rxL_top

    top_arc = ellipse_arc_points(
        cx=x_kink,
        cy=y3_top + ryL_top,
        rx=rxL_top,
        ry=ryL_top,
        deg0=270.0,
        deg1=TOP_END_DEG,
        clockwise=True,
        steps=170,
    )
    three_top = pen.line([(cx3r, y3_top), (x_kink, y3_top)] + top_arc[1:])

    bot_arc = ellipse_arc_points(
        cx=x_kink,
        cy=y3_bot - ryL_bot,
        rx=rxL_bot,
        ry=ryL_bot,
        deg0=90.0,
        deg1=BOT_END_DEG,
        clockwise=False,
        steps=170,
    )
    three_bot = pen.line([(cx3r, y3_bot), (x_kink, y3_bot)] + bot_arc[1:])

    mid_x0 = x3_left + (cx3r - x3_left) * 0.40
    mid_x1 = cx3r + r3 * 0.16
    three_mid = pen.hline(mid_x0, mid_x1, y3_mid)

    glyphs["3"] = (pen.union(three_right, three_top, three_mid, three_bot), W)

    # 4 ----------------------------------------------------------------
    x4_stem = xR - orx * 0.02
    y4_top = yD_top
    y4_cross = yD_mid + hD * 0.03
    x4_left = xL + orx * 0.04

    four_stem = pen.vline(x4_stem, y4_top, yD_bot)
    four_bar = pen.hline(x4_left, x4_stem, y4_cross)
    four_diag = pen.line([(x4_left, y4_cross), (x4_stem, y4_top)])
    glyphs["4"] = (pen.union(four_stem, four_bar, four_diag), W)

    # 5 ----------------------------------------------------------------
    y5_top = yD_top
    y5_mid = yD_top + hD * 0.45
    y5_bot = yD_bot

    x5_left = xL + orx * 0.11
    x5_right = xR - orx * 0.05

    cy5 = (y5_mid + y5_bot) / 2.0
    ry5 = (y5_bot - y5_mid) / 2.0

    bulge5 = ry5 * 0.92
    x5_join = x5_right - bulge5 * 0.92

    five_top = pen.hline(x5_left, x5_right, y5_top)
    five_left = pen.vline(x5_left, y5_top, y5_mid)
    five_mid = pen.hline(x5_left, x5_join, y5_mid)
    five_bot = pen.hline(x5_left, x5_join, y5_bot)

    five_loop = pen.line(
        ellipse_arc_points(x5_join, cy5, bulge5, ry5, 270.0, 90.0, clockwise=False, steps=260)
    )

    glyphs["5"] = (pen.union(five_top, five_left, five_mid, five_bot, five_loop), W)

    # 6 ----------------------------------------------------------------
    six_rx = orx * 0.98
    six_ry = hR * 0.34
    six_cx = cx + orx * 0.08
    six_cy = yR_bot - six_ry

    six_bowl = pen.ellipse_stroke(six_cx, six_cy, six_rx, six_ry)

    x_attach = six_cx - six_rx
    y_attach = six_cy

    # Make the top arc align optically with the rounded overshoot top.
    y_arc = yR_top + six_ry
    six_stem = pen.vline(x_attach, y_arc, y_attach)

    ARC_END_DEG = 328.0
    six_top_arc = pen.ellipse_arc(
        six_cx, y_arc, six_rx, six_ry, 180.0, ARC_END_DEG, steps=240
    )

    glyphs["6"] = (pen.union(six_bowl, six_stem, six_top_arc), W)

    # 7 ----------------------------------------------------------------
    y7_top = yD_top
    seven_top = pen.hline(xD_left, xD_right, y7_top)
    seven_diag = pen.line([(xD_right, y7_top), (xL + orx * 0.20, yD_bot)])
    glyphs["7"] = (pen.union(seven_top, seven_diag), W)

    # 8 ----------------------------------------------------------------
    # More overlap between the outer strokes, but keep two separate counters.
    h8 = yR_bot - yR_top
    rx8_top = wD * 0.46
    ry8_top = h8 * 0.235
    rx8_bot = wD * 0.49
    ry8_bot = h8 * 0.265

    cy8_top = yR_top + ry8_top
    cy8_bot = yR_bot - ry8_bot

    eight = pen.union(
        pen.ellipse_stroke(cx, cy8_top, rx8_top, ry8_top),
        pen.ellipse_stroke(cx, cy8_bot, rx8_bot, ry8_bot),
    )
    glyphs["8"] = (eight, W)

    # 9 ----------------------------------------------------------------
    g6, _ = glyphs["6"]
    origin = (W / 2.0, yR_mid)
    g9 = affinity.rotate(g6, 180.0, origin=origin)
    glyphs["9"] = (g9, W)

    for ch in "0123456789":
        glyphs.setdefault(ch, (Polygon(), W))

    return glyphs


# ----------------------------
# Lowercase
# ----------------------------

def build_lowercase(m: Metrics, pen: Mono) -> Dict[str, Tuple[Geom, float]]:
    f = make_lc_frame(m)
    W = f.W
    xL, xR, cx = f.xL, f.xR, f.cx
    yBase, yXTop, yMid, yAsc, yDesc = f.yBase, f.yXTop, f.yMid, f.yAsc, f.yDesc
    rx, ry, bcX, bcY = f.bowl_rx, f.bowl_ry, f.bowl_cx, f.bowl_cy

    dot_r = pen.r * 0.95
    dot_y = yXTop - T.DOT_GAP

    glyphs: Dict[str, Tuple[Geom, float]] = {}

    glyphs["c"] = (pen.ellipse_arc(bcX, bcY, rx, ry, 45.0, 315.0, steps=240), W)

    a_p0 = (146.0, 400.0)

    a_c01 = (205.0, 320.0)
    a_c02 = (265.0, 310.0)
    a_p1 = (312.0, 315.0)

    a_c11 = (345.0, 318.0)
    a_c12 = (360.0, 335.0)
    a_p2 = (360.0, 360.0)

    seg0 = cubic_points(a_p0, a_c01, a_c02, a_p1, steps=50)
    seg1 = cubic_points(a_p1, a_c11, a_c12, a_p2, steps=30)

    stem_down = [(360.0, 770.0)]
    bottom_in = [(256.0, 770.0)]

    a_p3 = (146.0, 665.0)
    a_c21 = (195.25, 770.0)
    a_c22 = (146.0, 723.0)

    a_p4 = (256.0, 560.0)
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

    b_stem_x = xL + T.STEM_INSET
    b_top = yXTop + 15.0
    b_bot = yBase - 10.0
    b_rx = (xR - b_stem_x) * 0.52
    glyphs["b"] = (stem_bowl(
        pen, stem_x=b_stem_x, top_y=b_top, bot_y=b_bot,
        side="right", stem_top=yAsc, stem_bot=yBase,
        bowl_rx=b_rx, overlap=10.0,
    ), W)

    d_stem_x = xR - T.STEM_INSET
    d_top = yXTop + 15.0
    d_bot = yBase - 10.0
    d_rx = (d_stem_x - xL) * 0.52
    glyphs["d"] = (stem_bowl(
        pen, stem_x=d_stem_x, top_y=d_top, bot_y=d_bot,
        side="left", stem_top=yAsc, stem_bot=yBase,
        bowl_rx=d_rx, overlap=10.0,
    ), W)

    e_cx, e_cy = bcX, bcY
    e_rx, e_ry = rx, ry
    e_bar_y = yXTop + 180.0

    s = (e_bar_y - e_cy) / e_ry
    s = clamp(s, -1.0, 1.0)
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

    Wf = W + 50.0
    fx = 260.0

    f_bot = yBase - 10.0

    y_oval_top = 50.0
    y_join = 160.0
    x_left = fx
    x_right = 560.0
    x_mid = (x_left + x_right) * 0.5

    pts: List[Tuple[float, float]] = [(fx, f_bot), (fx, y_join)]

    p0 = (x_left, y_join)
    p3 = (x_mid, y_oval_top)
    c1 = (x_left, y_oval_top + (y_join - y_oval_top) * 0.15)
    c2 = (x_mid - (x_right - x_left) * 0.18, y_oval_top)
    seg1 = cubic_points(p0, c1, c2, p3, steps=60)

    p0 = p3
    p3 = (x_right, y_join)
    c1 = (x_mid + (x_right - x_left) * 0.18, y_oval_top)
    c2 = (x_right, y_oval_top + (y_join - y_oval_top) * 0.15)
    seg2 = cubic_points(p0, c1, c2, p3, steps=60)

    pts += seg1[1:] + seg2[1:]
    f_stem_and_hook = pen.line(pts)

    f_cross_y = 435.0
    f_cross = pen.hline(fx, 430.0, f_cross_y)

    glyphs["f"] = (pen.union(f_stem_and_hook, f_cross), Wf)

    g_cx, g_cy = bcX - 20.0, bcY - 10.0
    g_rx, g_ry = rx * 0.88, ry * 0.88
    g_bowl = pen.ellipse_stroke(g_cx, g_cy, g_rx, g_ry)
    g_stem_x = g_cx + g_rx * 0.98
    g_stem_y0 = g_cy + g_ry * 0.20
    g_stem_y1 = yDesc - 70.0
    g_stem = pen.vline(g_stem_x, g_stem_y0, g_stem_y1)
    g_hook = pen.arc(g_stem_x - 120.0, g_stem_y1, 120.0, 0.0, 180.0, steps=180)
    glyphs["g"] = (pen.union(g_bowl, g_stem, g_hook), W)

    hxL = xL + T.STEM_INSET
    hxR = xR - T.STEM_INSET
    h_top_y = yXTop + 20.0

    h_left = pen.vline(hxL, yAsc, yBase)

    curve_dx = (hxR - hxL) * 0.40
    curve_dy = (yBase - h_top_y) * 0.28

    curve_dx = min(curve_dx, (hxR - hxL) - pen.r * 0.25)
    curve_dy = min(curve_dy, (yBase - h_top_y) - pen.r * 0.25)

    p0 = (hxR - curve_dx, h_top_y)
    bend = corner_h_to_v(p0, dx=curve_dx, dy=curve_dy, k=T.K, steps=90)

    h_pts: List[Tuple[float, float]] = []
    h_pts += [(hxL, h_top_y), p0]
    h_pts += bend[1:]
    h_pts += [(hxR, yBase)]
    h_shoulder = pen.line(h_pts)

    glyphs["h"] = (pen.union(h_left, h_shoulder), W)

    ix = cx
    glyphs["i"] = (pen.union(pen.vline(ix, yXTop + 20.0, yBase), pen.dot(ix, dot_y, dot_r)), W)
    glyphs["j"] = (pen.union(pen.vline(ix, yXTop + 20.0, yDesc - 10.0), pen.dot(ix, dot_y, dot_r)), W)

    kx = xL + T.STEM_INSET
    ky_mid = (yXTop + yBase) / 2.0
    glyphs["k"] = (pen.union(
        pen.vline(kx, yAsc, yBase),
        pen.line([(kx, ky_mid), (xR - 10.0, yXTop + 20.0)]),
        pen.line([(kx, ky_mid), (xR - 10.0, yBase - 20.0)]),
    ), W)

    glyphs["l"] = (pen.vline(cx - 120.0, yAsc, yBase), W)

    n_aperture = (xR - T.STEM_INSET) - (xL + T.STEM_INSET)

    Wm = W + n_aperture
    xLm, xRm = T.LC_INSET, Wm - T.LC_INSET

    m_x1 = xLm + T.STEM_INSET
    m_x2 = m_x1 + n_aperture
    m_x3 = m_x2 + n_aperture
    m_top = yXTop + 20.0

    curve_dx = n_aperture * 0.55
    curve_dy = (yBase - m_top) * 0.38

    curve_dx = min(curve_dx, (m_x3 - m_x2) - pen.r * 0.25)

    p0 = (m_x3 - curve_dx, m_top)
    shoulder = corner_h_to_v(p0, dx=curve_dx, dy=curve_dy, k=T.K, steps=90)

    m_pts: List[Tuple[float, float]] = []
    m_pts += [(m_x1, yBase), (m_x1, m_top)]
    m_pts += [(m_x2, m_top), (m_x2, yBase), (m_x2, m_top)]
    m_pts += [p0]
    m_pts += shoulder[1:]
    m_pts += [(m_x3, yBase)]

    glyphs["m"] = (pen.line(m_pts), Wm)

    n_x1 = xL + T.STEM_INSET
    n_x2 = xR - T.STEM_INSET
    n_top = yXTop + 20.0

    curve_dx = 170.0
    curve_dy = 160.0

    p0 = (n_x2 - curve_dx, n_top)
    shoulder = corner_h_to_v(p0, dx=curve_dx, dy=curve_dy, k=T.K, steps=90)

    n_pts = [(n_x1, yBase), (n_x1, n_top), p0] + shoulder[1:] + [(n_x2, yBase)]
    glyphs["n"] = (pen.line(n_pts), W)

    glyphs["o"] = (pen.ellipse_stroke(bcX, bcY, rx, ry), W)

    desc_len = (yXTop - yAsc)
    pq_stem_bot = yBase + desc_len

    p_stem_x = xL + T.STEM_INSET
    p_top = yXTop + 15.0
    p_bot = yBase - 10.0
    p_rx = (xR - p_stem_x) * 0.52
    glyphs["p"] = (stem_bowl(
        pen, stem_x=p_stem_x, top_y=p_top, bot_y=p_bot,
        side="right", stem_top=p_top, stem_bot=pq_stem_bot,
        bowl_rx=p_rx, overlap=10.0,
    ), W)

    q_stem_x = xR - T.STEM_INSET
    q_top = yXTop + 15.0
    q_bot = yBase - 10.0
    q_rx = (q_stem_x - xL) * 0.52
    glyphs["q"] = (stem_bowl(
        pen, stem_x=q_stem_x, top_y=q_top, bot_y=q_bot,
        side="left", stem_top=q_top, stem_bot=pq_stem_bot,
        bowl_rx=q_rx, overlap=10.0,
    ), W)

    rx_stem = xL + T.STEM_INSET
    r_stem_top = yXTop + 20.0
    r_start_y = r_stem_top + 120.0

    run = 280.0
    p0 = (rx_stem, r_start_y)
    p3 = (rx_stem + run, r_start_y - 30.0)

    c1 = (rx_stem + run * 0.34, r_start_y - 120.0)
    c2 = (rx_stem + run * 0.66, r_start_y - 130.0)

    r_pts = [(rx_stem, yBase), (rx_stem, r_stem_top), p0] + cubic_points(p0, c1, c2, p3, steps=90)[1:]
    glyphs["r"] = (pen.line(r_pts), W)

    s_xL = xL + 18.0
    s_xR = xR - 18.0
    s_y0 = yXTop + 10.0
    s_y3 = yBase - 10.0
    s_pts = s_from_spec(S_LC_SPEC, s_xL, s_xR, s_y0, s_y3, steps=84)
    glyphs["s"] = (pen.line(s_pts), W)

    tx = cx + 85.0
    t_top = yAsc + 10.0
    t_bot = yBase - 10.0
    t_cross_y = yXTop + 60.0
    t_left = tx
    t_right = tx + 220.0

    glyphs["t"] = (pen.union(
        pen.vline(tx, t_top, t_bot),
        pen.hline(t_left, t_right, t_cross_y),
    ), W)

    ux1 = xL + 50.0
    ux2 = xR - 50.0
    u_top = yXTop + 20.0
    u_bot = yBase - 10.0

    curve_dx = (ux2 - ux1) * 0.32
    curve_dy = (u_bot - u_top) * 0.34

    curve_dx = min(curve_dx, (ux2 - ux1) - pen.r * 0.25)
    curve_dy = min(curve_dy, (u_bot - u_top) - pen.r * 0.25)

    p0 = (ux1, u_bot - curve_dy)
    bend = corner_v_to_h(p0, dx=curve_dx, dy=curve_dy, k=T.K, steps=90)

    u_pts: List[Tuple[float, float]] = []
    u_pts += [(ux1, u_top), p0]
    u_pts += bend[1:]
    u_pts += [(ux2, u_bot), (ux2, u_top)]
    glyphs["u"] = (pen.line(u_pts), W)

    glyphs["v"] = (pen.line([(xL + 60.0, yXTop + 20.0), (cx, yBase), (xR - 60.0, yXTop + 20.0)]), W)

    Ww = W + n_aperture
    xLw, xRw = T.LC_INSET, Ww - T.LC_INSET

    wx1 = xLw + T.STEM_INSET
    wx2 = wx1 + n_aperture
    wx3 = wx2 + n_aperture
    w_top = yXTop + 20.0
    w_bot = yBase - 10.0

    curve_dx = n_aperture * 0.55
    curve_dy = (w_bot - w_top) * 0.38

    curve_dx = min(curve_dx, (wx2 - wx1) - pen.r * 0.25)
    curve_dy = min(curve_dy, (w_bot - w_top) - pen.r * 0.25)

    p0 = (wx1, w_bot - curve_dy)
    bend = corner_v_to_h(p0, dx=curve_dx, dy=curve_dy, k=T.K, steps=90)

    w_outer_pts: List[Tuple[float, float]] = []
    w_outer_pts += [(wx1, w_top), p0]
    w_outer_pts += bend[1:]
    w_outer_pts += [(wx3, w_bot), (wx3, w_top)]

    w_outer = pen.line(w_outer_pts)
    w_mid = pen.vline(wx2, w_top, w_bot)

    glyphs["w"] = (pen.union(w_outer, w_mid), Ww)

    x1 = xL + 55.0
    x2 = xR - 55.0
    x_top = yXTop + 30.0
    x_bot = yBase - 10.0
    glyphs["x"] = (pen.union(
        pen.line([(x1, x_top), (x2, x_bot)]),
        pen.line([(x2, x_top), (x1, x_bot)])
    ), W)

    yx1 = xL + 50.0
    yx2 = xR - 50.0
    y_top = yXTop + 20.0
    y_bot = yBase - 10.0

    y_desc_bot = yBase + desc_len

    curve_dx = (yx2 - yx1) * 0.32
    curve_dy = (y_bot - y_top) * 0.34

    curve_dx = min(curve_dx, (yx2 - yx1) - pen.r * 0.25)
    curve_dy = min(curve_dy, (y_bot - y_top) - pen.r * 0.25)

    p0 = (yx1, y_bot - curve_dy)
    bend = corner_v_to_h(p0, dx=curve_dx, dy=curve_dy, k=T.K, steps=90)

    left_and_bar_pts: List[Tuple[float, float]] = []
    left_and_bar_pts += [(yx1, y_top), p0]
    left_and_bar_pts += bend[1:]
    left_and_bar_pts += [(yx2, y_bot)]

    left_and_bar = pen.line(left_and_bar_pts)
    right_stem = pen.vline(yx2, y_top, y_desc_bot)

    glyphs["y"] = (pen.union(left_and_bar, right_stem), W)

    z_top = yXTop + 30.0
    z_bot = yBase - 10.0
    glyphs["z"] = (pen.union(
        pen.hline(xL + T.STEM_INSET, xR - T.STEM_INSET, z_top),
        pen.line([(xR - T.STEM_INSET, z_top), (xL + T.STEM_INSET, z_bot)]),
        pen.hline(xL + T.STEM_INSET, xR - T.STEM_INSET, z_bot),
    ), W)

    for ch in "abcdefghijklmnopqrstuvwxyz":
        glyphs.setdefault(ch, (Polygon(), W))

    return glyphs


# ----------------------------
# Punctuation
# ----------------------------

def build_punctuation(m: Metrics, pen: Mono) -> Dict[str, Tuple[Geom, float]]:
    glyphs: Dict[str, Tuple[Geom, float]] = {}

    yTop = m.CAP_TOP
    yBase = m.BASE
    yCapMid = m.CAP_MID
    yXTop = m.X_TOP
    yXMid = m.X_MID

    dot_r = pen.r * 0.92

    period_y = yBase - dot_r
    upper_dot_y = yXTop + 135.0

    W_SPACE = 320.0
    W_DOT = 260.0
    W_EXCL = 260.0
    W_APOS = 220.0
    W_QUOTE = 340.0
    W_DASH = 360.0
    W_EN_DASH = 500.0
    W_EM_DASH = 780.0
    W_SLASH = 420.0
    W_BRACKET = 300.0
    W_PAREN = 320.0
    W_QUEST = 520.0
    W_ELLIPSIS = 520.0
    W_CURLY_SINGLE = 230.0
    W_CURLY_DOUBLE = 360.0

    def right_single_curly_quote(cx: float, y0: float) -> Geom:
        pts = cubic_chain_points(
            (cx + 12.0, y0 + 48.0),
            [
                ((cx + 26.0, y0 + 92.0), (cx + 12.0, y0 + 150.0), (cx - 8.0, y0 + 186.0)),
                ((cx - 22.0, y0 + 214.0), (cx - 10.0, y0 + 154.0), (cx + 2.0, y0 + 118.0)),
            ],
            steps=56,
        )
        return pen.line(pts)

    def left_single_curly_quote(cx: float, y0: float) -> Geom:
        return affinity.scale(
            right_single_curly_quote(cx, y0),
            xfact=-1.0,
            yfact=1.0,
            origin=(cx, 0.0),
        )

    glyphs[" "] = (Polygon(), W_SPACE)

    cx = W_DOT / 2.0
    glyphs["."] = (pen.dot(cx, period_y, dot_r), W_DOT)

    comma_dot = pen.dot(cx, period_y, dot_r)
    comma_tail = pen.line([
        (cx + dot_r * 0.10, period_y + dot_r * 0.20),
        (cx - dot_r * 0.55, period_y + dot_r * 2.05),
    ])
    glyphs[","] = (pen.union(comma_dot, comma_tail), W_DOT)

    glyphs[":"] = (pen.union(
        pen.dot(cx, upper_dot_y, dot_r),
        pen.dot(cx, period_y, dot_r),
    ), W_DOT)

    semi_dot = pen.dot(cx, period_y, dot_r)
    semi_tail = pen.line([
        (cx + dot_r * 0.10, period_y + dot_r * 0.20),
        (cx - dot_r * 0.55, period_y + dot_r * 2.05),
    ])
    glyphs[";"] = (pen.union(
        pen.dot(cx, upper_dot_y, dot_r),
        semi_dot,
        semi_tail,
    ), W_DOT)

    cx = W_EXCL / 2.0
    excl_stem = pen.vline(cx, yTop + 70.0, yBase - 180.0)
    excl_dot = pen.dot(cx, period_y, dot_r)
    glyphs["!"] = (pen.union(excl_stem, excl_dot), W_EXCL)

    cx = W_QUEST / 2.0
    q0 = (150.0, yTop + 125.0)
    q_segments = [
        (
            (210.0, yTop + 35.0),
            (330.0, yTop + 20.0),
            (365.0, yTop + 120.0),
        ),
        (
            (400.0, yTop + 210.0),
            (300.0, yCapMid - 40.0),
            (cx,    yCapMid + 35.0),
        ),
    ]
    q_head_pts = cubic_chain_points(q0, q_segments, steps=84)
    q_head = pen.line(q_head_pts)
    q_stem = pen.vline(cx, yCapMid + 70.0, yCapMid + 150.0)
    q_dot = pen.dot(cx, period_y, dot_r)
    glyphs["?"] = (pen.union(q_head, q_stem, q_dot), W_QUEST)

    cx = W_APOS / 2.0
    apos = pen.line([
        (cx + 16.0, yTop + 55.0),
        (cx + 22.0, yTop + 102.0),
        (cx - 8.0,  yTop + 170.0),
    ])
    glyphs["'"] = (apos, W_APOS)

    q1 = pen.line([
        (118.0 + 16.0, yTop + 55.0),
        (118.0 + 22.0, yTop + 102.0),
        (118.0 - 8.0,  yTop + 170.0),
    ])
    q2 = pen.line([
        (222.0 + 16.0, yTop + 55.0),
        (222.0 + 22.0, yTop + 102.0),
        (222.0 - 8.0,  yTop + 170.0),
    ])
    glyphs["\""] = (pen.union(q1, q2), W_QUOTE)

    glyphs["-"] = (pen.hline(70.0, W_DASH - 70.0, yXMid), W_DASH)
    glyphs["–"] = (pen.hline(78.0, W_EN_DASH - 78.0, yXMid), W_EN_DASH)
    glyphs["—"] = (pen.hline(86.0, W_EM_DASH - 86.0, yXMid), W_EM_DASH)

    glyphs["/"] = (pen.line([
        (W_SLASH - 70.0, yTop + 55.0),
        (70.0,           yBase),
    ]), W_SLASH)

    glyphs["\\"] = (pen.line([
        (70.0,           yTop + 55.0),
        (W_SLASH - 70.0, yBase),
    ]), W_SLASH)

    par_rx = 92.0
    par_ry = ((yBase - yTop) / 2.0) - 8.0
    glyphs["("] = (
        pen.ellipse_arc(
            W_PAREN - 84.0, yCapMid, par_rx, par_ry,
            110.0, 250.0, steps=220
        ),
        W_PAREN
    )

    glyphs[")"] = (
        pen.ellipse_arc(
            84.0, yCapMid, par_rx, par_ry,
            290.0, 70.0, steps=220
        ),
        W_PAREN
    )

    bxL = 104.0
    arm = 92.0
    by0 = yTop + 40.0
    by1 = yBase
    glyphs["["] = (pen.union(
        pen.vline(bxL, by0, by1),
        pen.hline(bxL, bxL + arm, by0),
        pen.hline(bxL, bxL + arm, by1),
    ), W_BRACKET)

    bxR = W_BRACKET - 104.0
    glyphs["]"] = (pen.union(
        pen.vline(bxR, by0, by1),
        pen.hline(bxR - arm, bxR, by0),
        pen.hline(bxR - arm, bxR, by1),
    ), W_BRACKET)

    glyphs["…"] = (pen.union(
        pen.dot(110.0, period_y, dot_r),
        pen.dot(W_ELLIPSIS / 2.0, period_y, dot_r),
        pen.dot(W_ELLIPSIS - 110.0, period_y, dot_r),
    ), W_ELLIPSIS)

    glyphs["‘"] = (left_single_curly_quote(W_CURLY_SINGLE / 2.0, yTop), W_CURLY_SINGLE)
    glyphs["’"] = (right_single_curly_quote(W_CURLY_SINGLE / 2.0, yTop), W_CURLY_SINGLE)

    glyphs["“"] = (pen.union(
        left_single_curly_quote(118.0, yTop),
        left_single_curly_quote(242.0, yTop),
    ), W_CURLY_DOUBLE)

    glyphs["”"] = (pen.union(
        right_single_curly_quote(118.0, yTop),
        right_single_curly_quote(242.0, yTop),
    ), W_CURLY_DOUBLE)

    return glyphs


# ----------------------------
# Main
# ----------------------------

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
    punct = build_punctuation(m, pen)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    default_chars = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789"
        " !\"'(),-./:;?[]\\"
        "‘’“”…–—"
    )
    chars = args.chars or default_chars

    count = 0
    for ch in chars:
        if ch in upper:
            g, w = upper[ch]
        elif ch in lower:
            g, w = lower[ch]
        elif ch in digits:
            g, w = digits[ch]
        elif ch in punct:
            g, w = punct[ch]
        else:
            continue

        fname = codepoint_filename(ch)
        write_svg(out / fname, w, m, g)
        count += 1

    print(f"Wrote {count} SVGs to: {out}")


if __name__ == "__main__":
    main()