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
    def __init__(self, stroke: float, resolution: int = 48):
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

    def arc(self, cx: float, cy: float, r: float, deg0: float, deg1: float, steps: int = 120) -> Geom:
        # degrees: 0=right, 90=down (SVG-ish)
        def pt(deg: float) -> Tuple[float, float]:
            t = math.radians(deg)
            return (cx + math.cos(t) * r, cy + math.sin(t) * r)

        # handle wrap
        d0 = deg0 % 360.0
        d1 = deg1 % 360.0
        if d1 <= d0:
            d1 += 360.0
        angs = [d0 + (d1 - d0) * (i / (steps - 1)) for i in range(steps)]
        pts = [pt(a) for a in angs]
        return self.line(pts)

    def circle_stroke(self, cx: float, cy: float, r: float) -> Geom:
        # boundary buffered -> monoline circle
        boundary = Point(cx, cy).buffer(r, resolution=self.res).boundary
        return self._fix(boundary.buffer(self.r, cap_style=1, join_style=1, resolution=self.res))

    def ellipse_stroke(self, cx: float, cy: float, rx: float, ry: float) -> Geom:
        base = Point(cx, cy).buffer(1.0, resolution=self.res)
        ell = affinity.scale(base, xfact=rx, yfact=ry, origin=(cx, cy))
        boundary = ell.boundary
        return self._fix(boundary.buffer(self.r, cap_style=1, join_style=1, resolution=self.res))

    def wedge(self, cx: float, cy: float, r_outer: float, deg0: float, deg1: float) -> Polygon:
        # triangle wedge from center to two rays
        def pt(deg: float) -> Tuple[float, float]:
            t = math.radians(deg)
            return (cx + math.cos(t) * r_outer, cy + math.sin(t) * r_outer)
        p0 = pt(deg0)
        p1 = pt(deg1)
        return Polygon([(cx, cy), p0, p1])

    def sine_spine(self, x_center: float, y0: float, y1: float, amp: float, samples: int = 220) -> List[Tuple[float, float]]:
        # smoothstep y for flatter endpoints; cosine x for S-like
        def smoothstep(t: float) -> float:
            return t * t * (3.0 - 2.0 * t)

        pts: List[Tuple[float, float]] = []
        for i in range(samples):
            t = i / (samples - 1)
            y = y0 + smoothstep(t) * (y1 - y0)
            x = x_center + amp * math.cos(2.0 * math.pi * t)
            pts.append((x, y))
        return pts


# -------------------------
# SVG output (filled outlines)
# -------------------------

def fmt(x: float) -> str:
    return f"{x:.3f}"


def codepoint_filename(s: str) -> str:
    cps = [f"U{ord(ch):04X}" for ch in s]
    return "_".join(cps) + ".svg"


def geom_to_svg_path(g: Geom) -> str:
    # y is already SVG-y-down, so no flipping needed.
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

    glyphs: Dict[str, Tuple[Geom, float]] = {}

    # A (FIXED): stems + TRUE semicircle top + bar
    yArch = 260.0
    rArch = (xR - xL) / 2.0  # 220
    yBar = 450.0
    A = pen.union(
        pen.vline(xL, yBase, yArch),
        pen.arc(cx, yArch, rArch, 180.0, 360.0, steps=160),
        pen.vline(xR, yArch, yBase),
        pen.hline(xL, xR, yBar),
    )
    glyphs["A"] = (A, W)

    # B (your improved version: two identical bowls)
    bowl_r = (yMid - yTop) / 2.0
    xFlat = 375.0
    xBend = xFlat + bowl_r
    B = pen.union(
        pen.vline(xL, yTop, yBase),
        # top bowl
        pen.hline(xL, xFlat, yTop),
        pen.arc(xFlat, yTop + bowl_r, bowl_r, 270.0, 90.0, steps=120),  # right half
        pen.hline(xFlat, xL, yMid),
        # bottom bowl
        pen.hline(xL, xFlat, yMid),
        pen.arc(xFlat, yMid + bowl_r, bowl_r, 270.0, 90.0, steps=120),
        pen.hline(xFlat, xL, yBase),
    )
    glyphs["B"] = (B, W)

    # C
    o_r = 310.0
    o_cx, o_cy = cx, yMid
    C_ring = pen.circle_stroke(o_cx, o_cy, o_r)
    C_gap = pen.wedge(o_cx, o_cy, o_r + pen.stroke * 2.2, 350.0, 10.0)  # right-side gap
    C = C_ring.difference(C_gap)
    glyphs["C"] = (pen._fix(C), W)

    # D
    dxFlat = 320.0
    dxRX = xR - dxFlat
    dxRY = (yBase - yTop) / 2.0
    # Approx: stem + ellipse-ish right half made by scaling a circle arc
    D = pen.union(
        pen.vline(xL, yTop, yBase),
        pen.hline(xL, dxFlat, yTop),
        # two big arcs approximating the bowl
        pen.arc(dxFlat, yMid, dxRY, 270.0, 90.0, steps=140),
        pen.hline(dxFlat, xL, yBase),
    )
    # (D is still a sketch; keeping it simple)
    glyphs["D"] = (D, W)

    # E
    E = pen.union(
        pen.vline(xL, yTop, yBase),
        pen.hline(xL, xR, yTop),
        pen.hline(xL, xR - 50, yMid),
        pen.hline(xL, xR, yBase),
    )
    glyphs["E"] = (E, W)

    # F
    F = pen.union(
        pen.vline(xL, yTop, yBase),
        pen.hline(xL, xR, yTop),
        pen.hline(xL, xR - 70, yMid),
    )
    glyphs["F"] = (F, W)

    # G (FIXED): ring with small upper-right cut + bar that overlaps into stroke
    G_ring = pen.circle_stroke(o_cx, o_cy, o_r)
    # cut only upper-right (not the whole right side)
    G_gap = pen.wedge(o_cx, o_cy, o_r + pen.stroke * 2.2, 305.0, 350.0)
    G_outer = pen._fix(G_ring.difference(G_gap))
    # bar extends slightly past the inner edge so union merges
    G_bar = pen.hline(o_cx + o_r * 0.05, o_cx + o_r * 0.92, o_cy)
    G = pen.union(G_outer, G_bar)
    glyphs["G"] = (G, W)

    # H
    H = pen.union(
        pen.vline(xL, yTop, yBase),
        pen.vline(xR, yTop, yBase),
        pen.hline(xL, xR, yMid),
    )
    glyphs["H"] = (H, W)

    # I
    glyphs["I"] = (pen.vline(cx, yTop, yBase), W)

    # J (rough)
    J = pen.union(
        pen.vline(xR - 120, yTop, yBase - 130),
        pen.arc(xL + 150, yBase - 130, 150.0, 0.0, 180.0, steps=90),
    )
    glyphs["J"] = (J, W)

    # K
    K = pen.union(
        pen.vline(xL, yTop, yBase),
        pen.line([(xL, yMid), (xR, yTop)]),
        pen.line([(xL, yMid), (xR, yBase)]),
    )
    glyphs["K"] = (K, W)

    # L
    glyphs["L"] = (pen.union(pen.vline(xL, yTop, yBase), pen.hline(xL, xR, yBase)), W)

    # M
    M = pen.union(
        pen.vline(xL, yTop, yBase),
        pen.vline(xR, yTop, yBase),
        pen.line([(xL, yTop), (cx, yMid), (xR, yTop)]),
    )
    glyphs["M"] = (M, W)

    # N
    N = pen.union(
        pen.vline(xL, yTop, yBase),
        pen.vline(xR, yTop, yBase),
        pen.line([(xL, yTop), (xR, yBase)]),
    )
    glyphs["N"] = (N, W)

    # O
    glyphs["O"] = (pen.circle_stroke(o_cx, o_cy, o_r), W)

    # P
    P = pen.union(
        pen.vline(xL, yTop, yBase),
        pen.hline(xL, xFlat, yTop),
        pen.arc(xFlat, yTop + bowl_r, bowl_r, 270.0, 90.0, steps=120),
        pen.hline(xFlat, xL, yMid),
    )
    glyphs["P"] = (P, W)

    # Q
    Q = pen.union(
        pen.circle_stroke(o_cx, o_cy, o_r),
        pen.line([(cx + 80, yMid + 170), (xR, yBase)]),
    )
    glyphs["Q"] = (Q, W)

    # R
    R = pen.union(P, pen.line([(xFlat, yMid), (xR, yBase)]))
    glyphs["R"] = (R, W)

    # S (FIXED): smooth S spine (sine) outlined
    S_spine = pen.sine_spine(cx, yTop + 140, yBase - 140, amp=(xR - xL) * 0.42, samples=260)
    S = pen.line(S_spine)
    glyphs["S"] = (S, W)

    # T
    T = pen.union(pen.hline(xL, xR, yTop), pen.vline(cx, yTop, yBase))
    glyphs["T"] = (T, W)

    # U
    u_end = 560.0
    U = pen.union(
        pen.vline(xL, yTop, u_end),
        pen.arc(cx, u_end, (xR - xL) / 2.0, 180.0, 360.0, steps=140),
        pen.vline(xR, u_end, yTop),
    )
    glyphs["U"] = (U, W)

    # V
    glyphs["V"] = (pen.line([(xL, yTop), (cx, yBase), (xR, yTop)]), W)

    # W
    glyphs["W"] = (pen.line([(xL, yTop), (xL + 120, yBase), (cx, yMid), (xR - 120, yBase), (xR, yTop)]), W)

    # X
    glyphs["X"] = (pen.union(pen.line([(xL, yTop), (xR, yBase)]), pen.line([(xR, yTop), (xL, yBase)])), W)

    # Y
    glyphs["Y"] = (pen.union(pen.line([(xL, yTop), (cx, yMid), (xR, yTop)]), pen.vline(cx, yMid, yBase)), W)

    # Z
    glyphs["Z"] = (pen.union(pen.hline(xL, xR, yTop), pen.line([(xR, yTop), (xL, yBase)]), pen.hline(xL, xR, yBase)), W)

    # Fill the rest (CAVEAT: still sketchy placeholders)
    for ch in "HIJKLMNOPQRSTUVWXYZ":
        if ch not in glyphs:
            glyphs[ch] = (Polygon(), W)

    return glyphs


def build_lowercase(m: Metrics, pen: Mono) -> Dict[str, Tuple[Geom, float]]:
    W = m.LC_W
    xL, xR = 140.0, W - 140.0
    cx = W / 2.0

    yBase, yXTop, yMid = m.BASE, m.X_TOP, m.X_MID
    yAsc = m.CAP_TOP

    glyphs: Dict[str, Tuple[Geom, float]] = {}

    # a (FIXED): oval bowl + right stem that protrudes + inner bar (very close to your sample)
    bowl_cx = cx - 10
    bowl_cy = yMid
    rx = 235.0
    ry = 260.0  # puts top near x-height
    a_bowl = pen.ellipse_stroke(bowl_cx, bowl_cy, rx, ry)
    a_stem_x = bowl_cx + rx * 0.92
    a_stem = pen.vline(a_stem_x, yXTop - 80, yBase)  # protrude above bowl
    a_bar = pen.hline(bowl_cx - 40, a_stem_x + 10, yMid)  # connect into stem area
    a = pen.union(a_bowl, a_stem, a_bar)
    glyphs["a"] = (a, W)

    # b
    bowl_rx, bowl_ry = 190.0, 220.0
    b = pen.union(
        pen.vline(xL, yAsc, yBase),
        pen.hline(xL, cx, yXTop + 60),
        pen.arc(cx, yMid, bowl_ry, 270.0, 90.0, steps=120),
        pen.hline(cx, xL, yBase),
    )
    glyphs["b"] = (b, W)

    # c
    c_ring = pen.circle_stroke(cx, yMid, 220.0)
    c_gap = pen.wedge(cx, yMid, 220.0 + pen.stroke * 2.2, 350.0, 10.0)
    glyphs["c"] = (pen._fix(c_ring.difference(c_gap)), W)

    # d
    d = pen.union(
        pen.vline(xR, yAsc, yBase),
        pen.hline(xR, cx, yXTop + 60),
        pen.arc(cx, yMid, bowl_ry, 90.0, 270.0, steps=120),
        pen.hline(cx, xR, yBase),
    )
    glyphs["d"] = (d, W)

    # e (rough)
    e = pen.union(
        pen._fix(c_ring.difference(c_gap)),
        pen.hline(cx - 90, cx + 120, yMid),
    )
    glyphs["e"] = (e, W)

    # f (FIXED): tall stem + one right crossbar + slight top hook
    fx = cx - 90
    f_stem = pen.vline(fx, yAsc, yBase)
    f_bar_y = yXTop + 55
    f_bar = pen.hline(fx, fx + 260, f_bar_y)
    # top hook (tiny arc)
    f_hook = pen.arc(fx + 40, yAsc + 40, 40.0, 180.0, 270.0, steps=40)
    glyphs["f"] = (pen.union(f_stem, f_bar, f_hook), W)

    # g (placeholder-ish)
    g = pen.union(pen.circle_stroke(cx, yMid, 220.0), pen.line([(xR - 80, yMid + 30), (cx, m.DESC)]))
    glyphs["g"] = (g, W)

    # h
    h = pen.union(
        pen.vline(xL, yAsc, yBase),
        pen.line([(xL, yXTop), (cx, yXTop - 40), (xR, yXTop), (xR, yBase)]),
    )
    glyphs["h"] = (h, W)

    # i
    i = pen.union(pen.vline(cx, yXTop, yBase), Point(cx, yXTop - 70).buffer(pen.r * 0.65, resolution=pen.res))
    glyphs["i"] = (i, W)

    # j
    j = pen.union(
        pen.vline(cx, yXTop, m.DESC),
        pen.arc(cx - 120, m.DESC - 60, 120.0, 0.0, 180.0, steps=60),
        Point(cx, yXTop - 70).buffer(pen.r * 0.65, resolution=pen.res),
    )
    glyphs["j"] = (j, W)

    # k
    k = pen.union(
        pen.vline(xL, yAsc, yBase),
        pen.line([(xL, yMid), (xR, yXTop)]),
        pen.line([(xL, yMid), (xR, yBase)]),
    )
    glyphs["k"] = (k, W)

    # l
    glyphs["l"] = (pen.vline(cx, yAsc, yBase), W)

    # m
    m_ = pen.union(
        pen.vline(xL, yXTop, yBase),
        pen.line([(xL, yXTop), (cx - 60, yXTop - 40), (cx, yXTop), (cx, yBase)]),
        pen.line([(cx, yXTop), (cx + 120, yXTop - 40), (xR, yXTop), (xR, yBase)]),
    )
    glyphs["m"] = (m_, W)

    # n
    n = pen.union(
        pen.vline(xL, yXTop, yBase),
        pen.line([(xL, yXTop), (cx, yXTop - 40), (xR, yXTop), (xR, yBase)]),
    )
    glyphs["n"] = (n, W)

    # o
    glyphs["o"] = (pen.circle_stroke(cx, yMid, 220.0), W)

    # p/q/r placeholders
    glyphs["p"] = (pen.union(pen.vline(xL, yXTop, m.DESC), pen.circle_stroke(cx, yMid, 220.0)), W)
    glyphs["q"] = (pen.union(pen.vline(xR, yXTop, m.DESC), pen.circle_stroke(cx, yMid, 220.0)), W)
    glyphs["r"] = (pen.union(pen.vline(xL, yXTop, yBase), pen.arc(xL + 110, yXTop + 40, 110.0, 180.0, 270.0, steps=60)), W)

    # s (FIXED): smooth S spine at x-height scale
    s_spine = pen.sine_spine(cx, yXTop + 85, yBase - 85, amp=(xR - xL) * 0.38, samples=240)
    glyphs["s"] = (pen.line(s_spine), W)

    # t (FIXED): NOT a plus sign — tall stem with a mainly-right crossbar near x-height
    tx = cx
    t_top = yXTop - 160
    t_bar_y = yXTop + 10
    t_stem = pen.vline(tx, t_top, yBase)
    t_bar = pen.hline(tx - 40, tx + 240, t_bar_y)
    glyphs["t"] = (pen.union(t_stem, t_bar), W)

    # u
    u_end = yBase - 160
    u = pen.union(
        pen.vline(xL, yXTop, u_end),
        pen.arc(cx, u_end, (xR - xL) / 2.0, 180.0, 360.0, steps=120),
        pen.vline(xR, u_end, yXTop),
    )
    glyphs["u"] = (u, W)

    # v/w/x/y/z
    glyphs["v"] = (pen.line([(xL, yXTop), (cx, yBase), (xR, yXTop)]), W)
    glyphs["w"] = (pen.line([(xL, yXTop), (xL + 90, yBase), (cx, yMid), (xR - 90, yBase), (xR, yXTop)]), W)
    glyphs["x"] = (pen.union(pen.line([(xL, yXTop), (xR, yBase)]), pen.line([(xR, yXTop), (xL, yBase)])), W)
    glyphs["y"] = (pen.union(pen.line([(xL, yXTop), (cx, yBase), (xR, yXTop)]), pen.vline(cx, yBase, m.DESC)), W)
    glyphs["z"] = (pen.union(pen.hline(xL, xR, yXTop), pen.line([(xR, yXTop), (xL, yBase)]), pen.hline(xL, xR, yBase)), W)

    # Fill missing letters as empty sketches for now (keeps pipeline stable)
    for ch in "abcdefghijklmnopqrstuvwxyz":
        glyphs.setdefault(ch, (Polygon(), W))

    return glyphs


def build_ligatures(m: Metrics, pen: Mono, lower: Dict[str, Tuple[Geom, float]]) -> Dict[str, Tuple[Geom, float]]:
    adv = m.LC_W
    topbar_y = m.CAP_TOP + 120

    def shift(g: Geom, dx: float) -> Geom:
        return affinity.translate(g, xoff=dx, yoff=0.0)

    ligs: Dict[str, Tuple[Geom, float]] = {}

    # hij
    hij = pen.union(
        shift(lower["h"][0], 0),
        shift(lower["i"][0], adv),
        shift(lower["j"][0], adv * 2),
        pen.hline(80, adv * 3 - 80, topbar_y),
    )
    ligs["hij"] = (hij, adv * 3)

    # ik
    ik = pen.union(shift(lower["i"][0], 0), shift(lower["k"][0], adv))
    ligs["ik"] = (ik, adv * 2)

    # sch
    sch = pen.union(
        shift(lower["s"][0], 0),
        shift(lower["c"][0], adv),
        shift(lower["h"][0], adv * 2),
        pen.hline(80, adv * 3 - 80, topbar_y),
    )
    ligs["sch"] = (sch, adv * 3)

    return ligs


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("sketches"))
    ap.add_argument("--stroke", type=float, default=90.0)
    ap.add_argument("--resolution", type=int, default=48)
    ap.add_argument("--include-ligatures", action="store_true")
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

    if args.include_ligatures:
        ligs = build_ligatures(m, pen, lower)
        for name, (g, w) in ligs.items():
            fname = "liga_" + codepoint_filename(name)
            write_svg(out / fname, w, m, g)
            preview.append((f"liga:{name}", fname))

    write_preview_html(out, preview)
    print(f"Wrote {len(preview)} SVGs to: {out}")
    print(f"Open: {out / 'preview.html'}")


if __name__ == "__main__":
    main()
