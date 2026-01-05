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


def _frange(a: float, b: float, steps: int) -> List[float]:
    if steps <= 1:
        return [a]
    return [a + (b - a) * (i / (steps - 1)) for i in range(steps)]


def _arc_points(cx: float, cy: float, r: float, a0: float, a1: float, steps: int = 48) -> List[Tuple[float, float]]:
    angs = _frange(a0, a1, steps)
    return [(cx + math.cos(t) * r, cy + math.sin(t) * r) for t in angs]


def _ensure_valid(g: Geom) -> Geom:
    try:
        gg = g.buffer(0)
        return gg if not gg.is_empty else g
    except Exception:
        return g


def geom_to_svg_path(g: Geom, view_h: float, y_shift: float) -> str:
    """
    Convert shapely Polygon/MultiPolygon to a single SVG path string.
    We output y-down coordinates (SVG) by mapping:
      y_svg = view_h - (y + y_shift)
    """
    if g.is_empty:
        return ""

    if isinstance(g, Polygon):
        polys = [g]
    else:
        polys = [p for p in g.geoms if isinstance(p, Polygon)]

    def fmt_pt(x: float, y: float) -> str:
        y2 = view_h - (y + y_shift)
        return f"{x:.3f},{y2:.3f}"

    parts: List[str] = []
    for p in polys:
        ext = list(p.exterior.coords)
        if len(ext) >= 2:
            parts.append("M " + " L ".join(fmt_pt(x, y) for x, y in ext) + " Z")
        for hole in p.interiors:
            h = list(hole.coords)
            if len(h) >= 2:
                parts.append("M " + " L ".join(fmt_pt(x, y) for x, y in h) + " Z")

    return " ".join(parts)


@dataclass
class Metrics:
    upm: int = 1000
    ascent: float = 780.0
    descent: float = -220.0
    xh: float = 520.0
    cap: float = 740.0

    @property
    def em_h(self) -> float:
        return self.ascent - self.descent


class MonoBuilder:
    def __init__(self, m: Metrics, stroke: float, resolution: int = 32):
        self.m = m
        self.stroke = float(stroke)
        self.r = self.stroke / 2.0
        self.resolution = resolution

    def stroke_path(self, pts: List[Tuple[float, float]]) -> Geom:
        ls = LineString(pts)
        return ls.buffer(self.r, cap_style=1, join_style=1, resolution=self.resolution)

    def vline(self, x: float, y0: float, y1: float) -> Geom:
        return self.stroke_path([(x, y0), (x, y1)])

    def hline(self, x0: float, x1: float, y: float) -> Geom:
        return self.stroke_path([(x0, y), (x1, y)])

    def ring(self, cx: float, cy: float, r: float) -> Geom:
        boundary = Point(cx, cy).buffer(r, resolution=self.resolution).boundary
        return boundary.buffer(self.r, cap_style=1, join_style=1, resolution=self.resolution)

    def dot(self, cx: float, cy: float, radius: float) -> Geom:
        return Point(cx, cy).buffer(radius, resolution=self.resolution)

    def arc_stroke(self, cx: float, cy: float, r: float, a0: float, a1: float, steps: int = 48) -> Geom:
        pts = _arc_points(cx, cy, r, a0, a1, steps=steps)
        return self.stroke_path(pts)

    def union(self, *parts: Geom) -> Geom:
        ps = [p for p in parts if p is not None and not p.is_empty]
        if not ps:
            return Polygon()
        return _ensure_valid(unary_union(ps))

    def bowl_with_stem(self, stem_x: float, stem_y0: float, stem_y1: float, bowl_cx: float, bowl_cy: float, bowl_r: float) -> Geom:
        return self.union(self.vline(stem_x, stem_y0, stem_y1), self.ring(bowl_cx, bowl_cy, bowl_r))


def build_lowercase(builder: MonoBuilder) -> Dict[str, Tuple[Geom, float]]:
    m = builder.m
    xh = m.xh
    asc = m.ascent
    desc = m.descent

    adv = 600.0
    lsb = 80.0
    rsb = 80.0

    cx = adv * 0.50
    bowl_r = xh * 0.34
    bowl_cy = xh * 0.36

    glyphs: Dict[str, Tuple[Geom, float]] = {}

    a_ring = builder.ring(cx - 20, bowl_cy, bowl_r)
    a_stem = builder.vline(cx + bowl_r * 0.65, 0, xh)
    a_bridge = builder.hline(cx - 10, cx + bowl_r * 0.65, bowl_cy + bowl_r * 0.35)
    glyphs["a"] = (builder.union(a_ring, a_stem, a_bridge), adv)

    b_stem_x = lsb + builder.r
    b = builder.bowl_with_stem(b_stem_x, 0, asc, cx + 10, xh * 0.35, bowl_r)
    glyphs["b"] = (b, adv)

    c = builder.arc_stroke(cx, bowl_cy, bowl_r, math.radians(45), math.radians(315), steps=64)
    glyphs["c"] = (c, adv)

    d_stem_x = adv - rsb - builder.r
    d = builder.union(
        builder.vline(d_stem_x, 0, asc),
        builder.ring(cx - 10, bowl_cy, bowl_r),
        builder.hline(cx - 10, d_stem_x, bowl_cy + bowl_r * 0.35),
    )
    glyphs["d"] = (d, adv)

    e = builder.union(
        builder.arc_stroke(cx, bowl_cy, bowl_r, math.radians(20), math.radians(340), steps=72),
        builder.hline(cx - bowl_r * 0.55, cx + bowl_r * 0.55, bowl_cy),
    )
    glyphs["e"] = (e, adv)

    f_x = lsb + builder.r
    f = builder.union(
        builder.vline(f_x, 0, asc),
        builder.hline(f_x - 10, f_x + bowl_r * 0.85, xh * 0.75),
        builder.hline(f_x - 10, f_x + bowl_r * 0.55, xh * 0.35),
    )
    glyphs["f"] = (f, adv)

    g_o = builder.ring(cx - 10, bowl_cy, bowl_r)
    g_tail = builder.stroke_path([(cx + bowl_r * 0.65, bowl_cy - 10), (cx + bowl_r * 0.65, desc * 0.25), (cx - 10, desc * 0.25)])
    glyphs["g"] = (builder.union(g_o, g_tail), adv)

    h_x = lsb + builder.r
    h_arch = builder.stroke_path([(h_x, xh), (cx, xh), (cx, 0)])
    glyphs["h"] = (builder.union(builder.vline(h_x, 0, asc), h_arch), adv)

    i_x = cx
    i = builder.union(builder.vline(i_x, 0, xh), builder.dot(i_x, xh + 120, radius=builder.r * 0.65))
    glyphs["i"] = (i, adv)

    j_x = cx
    j = builder.union(
        builder.vline(j_x, desc * 0.75, xh),
        builder.arc_stroke(j_x - 40, desc * 0.75, 40, math.radians(0), math.radians(180), steps=24),
        builder.dot(j_x, xh + 120, radius=builder.r * 0.65),
    )
    glyphs["j"] = (j, adv)

    k_x = lsb + builder.r
    k = builder.union(
        builder.vline(k_x, 0, asc),
        builder.stroke_path([(k_x, xh * 0.52), (adv - rsb, xh)]),
        builder.stroke_path([(k_x, xh * 0.52), (adv - rsb, 0)]),
    )
    glyphs["k"] = (k, adv)

    l_x = lsb + builder.r
    glyphs["l"] = (builder.vline(l_x, 0, asc), adv)

    m0 = lsb + builder.r
    m1 = cx - 20
    m2 = adv - rsb - builder.r
    m_geom = builder.union(
        builder.vline(m0, 0, xh),
        builder.stroke_path([(m0, xh), (m1, xh), (m1, 0)]),
        builder.stroke_path([(m1, xh), (m2, xh), (m2, 0)]),
    )
    glyphs["m"] = (m_geom, adv)

    n0 = lsb + builder.r
    n1 = adv - rsb - builder.r
    n_geom = builder.union(
        builder.vline(n0, 0, xh),
        builder.stroke_path([(n0, xh), (n1, xh), (n1, 0)]),
    )
    glyphs["n"] = (n_geom, adv)

    glyphs["o"] = (builder.ring(cx, bowl_cy, bowl_r), adv)

    p_x = lsb + builder.r
    p = builder.union(
        builder.vline(p_x, desc * 0.75, xh),
        builder.ring(cx + 10, xh * 0.62, bowl_r),
        builder.hline(cx + 10, p_x, xh * 0.62 + bowl_r * 0.35),
    )
    glyphs["p"] = (p, adv)

    q_x = adv - rsb - builder.r
    q = builder.union(
        builder.ring(cx - 10, xh * 0.62, bowl_r),
        builder.vline(q_x, desc * 0.75, xh),
        builder.hline(cx - 10, q_x, xh * 0.62 + bowl_r * 0.35),
    )
    glyphs["q"] = (q, adv)

    r_x = lsb + builder.r
    r = builder.union(
        builder.vline(r_x, 0, xh),
        builder.stroke_path([(r_x, xh * 0.75), (cx + 40, xh * 0.75)]),
        builder.stroke_path([(cx + 40, xh * 0.75), (cx + 40, xh * 0.25)]),
    )
    glyphs["r"] = (r, adv)

    s = builder.union(
        builder.arc_stroke(cx, bowl_cy + 70, bowl_r * 0.9, math.radians(200), math.radians(20), steps=40),
        builder.arc_stroke(cx, bowl_cy - 70, bowl_r * 0.9, math.radians(20), math.radians(200), steps=40),
        builder.hline(cx - 10, cx + 10, bowl_cy),
    )
    glyphs["s"] = (s, adv)

    t_x = cx - 80
    t = builder.union(
        builder.vline(t_x, 0, asc),
        builder.hline(t_x - 120, t_x + 200, xh),
    )
    glyphs["t"] = (t, adv)

    u0 = lsb + builder.r
    u1 = adv - rsb - builder.r
    u = builder.union(
        builder.vline(u0, xh, 0),
        builder.vline(u1, 0, xh),
        builder.arc_stroke(cx, 0, (u1 - u0) * 0.42, math.radians(180), math.radians(360), steps=48),
    )
    glyphs["u"] = (u, adv)

    v = builder.union(builder.stroke_path([(lsb, xh), (cx, 0), (adv - rsb, xh)]))
    glyphs["v"] = (v, adv)

    w = builder.union(builder.stroke_path([(lsb, xh), (adv * 0.35, 0), (adv * 0.5, xh * 0.45), (adv * 0.65, 0), (adv - rsb, xh)]))
    glyphs["w"] = (w, adv)

    x = builder.union(
        builder.stroke_path([(lsb, 0), (adv - rsb, xh)]),
        builder.stroke_path([(lsb, xh), (adv - rsb, 0)]),
    )
    glyphs["x"] = (x, adv)

    y = builder.union(
        builder.stroke_path([(lsb, xh), (cx, 0), (adv - rsb, xh)]),
        builder.stroke_path([(cx, 0), (cx, desc * 0.75)]),
    )
    glyphs["y"] = (y, adv)

    z = builder.union(
        builder.hline(lsb, adv - rsb, xh),
        builder.stroke_path([(adv - rsb, xh), (lsb, 0)]),
        builder.hline(lsb, adv - rsb, 0),
    )
    glyphs["z"] = (z, adv)

    return glyphs


def build_uppercase(builder: MonoBuilder) -> Dict[str, Tuple[Geom, float]]:
    m = builder.m
    cap = m.cap
    adv = 700.0
    lsb = 90.0
    rsb = 90.0
    cx = adv * 0.5

    glyphs: Dict[str, Tuple[Geom, float]] = {}

    A = builder.union(
        builder.vline(lsb, 0, cap),
        builder.vline(adv - rsb, 0, cap),
        builder.arc_stroke(cx, cap, (adv - lsb - rsb) * 0.5, math.radians(180), math.radians(360), steps=64),
    )
    glyphs["A"] = (A, adv)

    B = builder.union(
        builder.vline(lsb, 0, cap),
        builder.ring(cx + 40, cap * 0.70, cap * 0.22),
        builder.ring(cx + 40, cap * 0.30, cap * 0.22),
        builder.hline(lsb, cx + 40, cap * 0.70 + cap * 0.08),
        builder.hline(lsb, cx + 40, cap * 0.30 + cap * 0.08),
    )
    glyphs["B"] = (B, adv)

    C = builder.arc_stroke(cx, cap * 0.52, cap * 0.42, math.radians(45), math.radians(315), steps=80)
    glyphs["C"] = (C, adv)

    D = builder.union(
        builder.vline(lsb, 0, cap),
        builder.arc_stroke(lsb + (adv - lsb - rsb) * 0.55, cap * 0.52, cap * 0.42, math.radians(-90), math.radians(90), steps=80),
        builder.hline(lsb, adv - rsb, cap),
        builder.hline(lsb, adv - rsb, 0),
    )
    glyphs["D"] = (D, adv)

    E = builder.union(
        builder.vline(lsb, 0, cap),
        builder.hline(lsb, adv - rsb, cap),
        builder.hline(lsb, adv - rsb * 1.2, cap * 0.52),
        builder.hline(lsb, adv - rsb, 0),
    )
    glyphs["E"] = (E, adv)

    F = builder.union(
        builder.vline(lsb, 0, cap),
        builder.hline(lsb, adv - rsb, cap),
        builder.hline(lsb, adv - rsb * 1.2, cap * 0.52),
    )
    glyphs["F"] = (F, adv)

    G = builder.union(C, builder.hline(cx, adv - rsb, cap * 0.38))
    glyphs["G"] = (G, adv)

    H = builder.union(builder.vline(lsb, 0, cap), builder.vline(adv - rsb, 0, cap), builder.hline(lsb, adv - rsb, cap * 0.52))
    glyphs["H"] = (H, adv)

    I = builder.vline(cx, 0, cap)
    glyphs["I"] = (I, adv)

    J = builder.union(
        builder.vline(cx, cap * 0.2, cap),
        builder.arc_stroke(cx - 120, cap * 0.2, 120, math.radians(0), math.radians(180), steps=40),
    )
    glyphs["J"] = (J, adv)

    K = builder.union(
        builder.vline(lsb, 0, cap),
        builder.stroke_path([(lsb, cap * 0.55), (adv - rsb, cap)]),
        builder.stroke_path([(lsb, cap * 0.55), (adv - rsb, 0)]),
    )
    glyphs["K"] = (K, adv)

    L = builder.union(builder.vline(lsb, 0, cap), builder.hline(lsb, adv - rsb, 0))
    glyphs["L"] = (L, adv)

    M = builder.union(
        builder.vline(lsb, 0, cap),
        builder.vline(adv - rsb, 0, cap),
        builder.stroke_path([(lsb, cap), (cx, cap * 0.45), (adv - rsb, cap)]),
    )
    glyphs["M"] = (M, adv)

    N = builder.union(
        builder.vline(lsb, 0, cap),
        builder.vline(adv - rsb, 0, cap),
        builder.stroke_path([(lsb, cap), (adv - rsb, 0)]),
    )
    glyphs["N"] = (N, adv)

    O = builder.ring(cx, cap * 0.52, cap * 0.42)
    glyphs["O"] = (O, adv)

    P = builder.union(
        builder.vline(lsb, 0, cap),
        builder.ring(cx + 40, cap * 0.72, cap * 0.22),
        builder.hline(lsb, cx + 40, cap * 0.72 + cap * 0.08),
    )
    glyphs["P"] = (P, adv)

    Q = builder.union(O, builder.stroke_path([(cx + 80, cap * 0.25), (adv - rsb, 0)]))
    glyphs["Q"] = (Q, adv)

    R = builder.union(P, builder.stroke_path([(cx + 40, cap * 0.5), (adv - rsb, 0)]))
    glyphs["R"] = (R, adv)

    S = builder.union(
        builder.arc_stroke(cx, cap * 0.70, cap * 0.36, math.radians(200), math.radians(20), steps=60),
        builder.arc_stroke(cx, cap * 0.30, cap * 0.36, math.radians(20), math.radians(200), steps=60),
    )
    glyphs["S"] = (S, adv)

    T = builder.union(builder.hline(lsb, adv - rsb, cap), builder.vline(cx, 0, cap))
    glyphs["T"] = (T, adv)

    U = builder.union(
        builder.vline(lsb, cap, cap * 0.2),
        builder.vline(adv - rsb, cap * 0.2, cap),
        builder.arc_stroke(cx, cap * 0.2, (adv - lsb - rsb) * 0.42, math.radians(180), math.radians(360), steps=64),
    )
    glyphs["U"] = (U, adv)

    V = builder.stroke_path([(lsb, cap), (cx, 0), (adv - rsb, cap)])
    glyphs["V"] = (V, adv)

    W = builder.stroke_path([(lsb, cap), (adv * 0.35, 0), (cx, cap * 0.45), (adv * 0.65, 0), (adv - rsb, cap)])
    glyphs["W"] = (W, adv)

    X = builder.union(builder.stroke_path([(lsb, 0), (adv - rsb, cap)]), builder.stroke_path([(lsb, cap), (adv - rsb, 0)]))
    glyphs["X"] = (X, adv)

    Y = builder.union(builder.stroke_path([(lsb, cap), (cx, cap * 0.52), (adv - rsb, cap)]), builder.vline(cx, 0, cap * 0.52))
    glyphs["Y"] = (Y, adv)

    Z = builder.union(
        builder.hline(lsb, adv - rsb, cap),
        builder.stroke_path([(adv - rsb, cap), (lsb, 0)]),
        builder.hline(lsb, adv - rsb, 0),
    )
    glyphs["Z"] = (Z, adv)

    return glyphs


def build_ligatures(builder: MonoBuilder, lower: Dict[str, Tuple[Geom, float]]) -> Dict[str, Tuple[Geom, float]]:
    m = builder.m
    adv = 600.0
    xh = m.xh
    topbar_y = m.ascent * 0.70

    def sh(g: Geom, dx: float) -> Geom:
        return _ensure_valid(affinity.translate(g, xoff=dx, yoff=0))

    def strip_dots(geom: Geom) -> Geom:
        from shapely.geometry import box
        kill = box(-1e6, xh + 70, 1e6, xh + 220)
        return _ensure_valid(geom.difference(kill))

    ligs: Dict[str, Tuple[Geom, float]] = {}

    # hij
    h = strip_dots(lower["h"][0])
    i = strip_dots(lower["i"][0])
    j = strip_dots(lower["j"][0])
    hij_geom = builder.union(
        sh(h, 0),
        sh(i, adv),
        sh(j, adv * 2),
        builder.hline(80, adv * 3 - 80, topbar_y),
    )
    ligs["hij"] = (hij_geom, adv * 3)

    # ik
    i2 = strip_dots(lower["i"][0])
    k = lower["k"][0]
    ik_geom = builder.union(sh(i2, 0), sh(k, adv))
    ligs["ik"] = (ik_geom, adv * 2)

    # sch
    s = lower["s"][0]
    c = lower["c"][0]
    h2 = strip_dots(lower["h"][0])
    sch_geom = builder.union(
        sh(s, 0),
        sh(c, adv),
        sh(h2, adv * 2),
        builder.hline(80, adv * 3 - 80, topbar_y),
    )
    ligs["sch"] = (sch_geom, adv * 3)

    return ligs


def codepoint_filename(s: str) -> str:
    # Single char: U0041.svg ; Multi-char (ligature sequence): U0068_U0069_U006A.svg
    cps = [f"U{ord(ch):04X}" for ch in s]
    return "_".join(cps) + ".svg"


def write_svg(out_path: Path, geom: Geom, adv: float, metrics: Metrics) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    view_w = adv
    view_h = metrics.em_h
    y_shift = -metrics.descent

    d = geom_to_svg_path(geom, view_h=view_h, y_shift=y_shift)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {view_w:.3f} {view_h:.3f}">
  <path d="{d}" fill="black" fill-rule="evenodd"/>
</svg>
"""
    out_path.write_text(svg, encoding="utf-8")


def write_preview_html(out_dir: Path, items: List[Tuple[str, Path]]) -> None:
    rows = []
    for label, p in sorted(items, key=lambda t: t[1].name):
        rows.append(f"""
<div class="cell">
  <div class="label">{label}</div>
  <div class="file">{p.name}</div>
  <img src="{p.name}" alt="{label}"/>
</div>
""".strip())

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Glyph Preview</title>
<style>
body{{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:16px}}
.cell{{border:1px solid #ddd;border-radius:10px;padding:10px}}
.label{{font-size:13px;font-weight:600;margin-bottom:4px}}
.file{{font-size:11px;opacity:.7;margin-bottom:8px}}
img{{width:100%;height:auto;display:block;background:#fff}}
</style>
</head>
<body>
<h1>Generated glyph SVGs</h1>
<div class="grid">
{chr(10).join(rows)}
</div>
</body>
</html>
"""
    (out_dir / "preview.html").write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("sketches"), help="output directory (default: sketches)")
    ap.add_argument("--stroke", type=float, default=90.0, help="stroke thickness in font units")
    ap.add_argument("--resolution", type=int, default=32, help="curve resolution (higher = smoother)")
    args = ap.parse_args()

    metrics = Metrics()
    b = MonoBuilder(metrics, stroke=args.stroke, resolution=args.resolution)

    lower = build_lowercase(b)
    upper = build_uppercase(b)
    ligs = build_ligatures(b, lower)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    preview_items: List[Tuple[str, Path]] = []

    # letters
    for ch, (g, adv) in {**lower, **upper}.items():
        fname = codepoint_filename(ch)
        p = out / fname
        write_svg(p, g, adv, metrics)
        preview_items.append((ch, p))

    # ligatures (sequence-based filename)
    for name, (g, adv) in ligs.items():
        seq_fname = codepoint_filename(name)  # name is like "hij" -> U0068_U0069_U006A.svg
        p = out / f"liga_{seq_fname}"
        write_svg(p, g, adv, metrics)
        preview_items.append((f"liga:{name}", p))

    write_preview_html(out, preview_items)
    print(f"Wrote {len(preview_items)} SVGs to: {out}")
    print(f"Open: {out / 'preview.html'}")


if __name__ == "__main__":
    main()
