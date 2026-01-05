#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional


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
    XH: float = 520.0  # x-height in units (distance from baseline upwards)

    @property
    def CAP_MID(self) -> float:
        return (self.CAP_TOP + self.BASE) / 2.0

    @property
    def X_TOP(self) -> float:
        # y at top of x-height
        return self.BASE - self.XH

    @property
    def X_MID(self) -> float:
        return (self.X_TOP + self.BASE) / 2.0

    @property
    def DESC(self) -> float:
        return float(self.H)


@dataclass
class Glyph:
    width: float
    # multiple stroked path segments (combined as separate <path> elements)
    paths: List[str]
    # filled circles (for dots)
    circles: List[Tuple[float, float, float]]  # (cx, cy, r)


# -------------------------
# SVG helpers
# -------------------------

def codepoint_filename(s: str) -> str:
    cps = [f"U{ord(ch):04X}" for ch in s]
    return "_".join(cps) + ".svg"


def pt(cx: float, cy: float, r: float, deg: float) -> Tuple[float, float]:
    # angle deg, 0° = +x, 90° = +y (SVG y-down)
    t = math.radians(deg)
    return (cx + r * math.cos(t), cy + r * math.sin(t))


def fmt(x: float) -> str:
    return f"{x:.3f}"


def svg_path_for_circle(cx: float, cy: float, r: float) -> str:
    # path-circle (useful if you want everything as <path>)
    # we’ll keep circles as <circle> elements, but this is here if you need it.
    x0 = cx + r
    y0 = cy
    x1 = cx - r
    y1 = cy
    return (
        f"M {fmt(x0)} {fmt(y0)} "
        f"A {fmt(r)} {fmt(r)} 0 1 1 {fmt(x1)} {fmt(y1)} "
        f"A {fmt(r)} {fmt(r)} 0 1 1 {fmt(x0)} {fmt(y0)}"
    )


def write_svg(path: Path, m: Metrics, g: Glyph, stroke: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    parts: List[str] = []
    for d in g.paths:
        parts.append(
            f'<path d="{d}" fill="none" stroke="black" '
            f'stroke-width="{fmt(stroke)}" stroke-linecap="round" stroke-linejoin="round"/>'
        )

    for (cx, cy, r) in g.circles:
        parts.append(f'<circle cx="{fmt(cx)}" cy="{fmt(cy)}" r="{fmt(r)}" fill="black"/>')

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {fmt(g.width)} {m.H}">\n'
        + "\n".join(f"  {p}" for p in parts)
        + "\n</svg>\n"
    )
    path.write_text(svg, encoding="utf-8")


def write_preview_html(out_dir: Path, items: List[Tuple[str, str]]) -> None:
    # items: (label, filename)
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
# Uppercase glyphs (A–Z)
# -------------------------

def build_uppercase(m: Metrics, stroke: float) -> Dict[str, Glyph]:
    W = m.CAP_W
    xL = 130.0
    xR = W - 130.0
    cx = W / 2.0

    yTop = m.CAP_TOP
    yBase = m.BASE
    yMid = m.CAP_MID

    # Used for B bowls (top+bottom halves)
    bowl_r = (yMid - yTop) / 2.0  # 185
    xFlat = 375.0
    xBend = xFlat + bowl_r        # ~560

    # Round letter bowls (O/C/G)
    o_r = 310.0
    o_cx = cx
    o_cy = yMid
    gap_deg = 35.0

    # For D / P / R (elliptical half-bowl)
    dxFlat = 320.0
    dxRX = xR - dxFlat  # ~250
    dxRY = (yBase - yTop) / 2.0  # 370

    # For U bottom arch
    u_arch_y = yBase - (xR - xL) / 2.0  # endpoints y so that a semicircle-like Q hits baseline
    # But we’ll explicitly set to a pleasing value:
    u_end_y = 560.0

    glyphs: Dict[str, Glyph] = {}

    # A (improved): two stems + rounded top arch + mid crossbar
    yArch = 260.0
    yBarA = 450.0
    A_paths = [
        f"M {fmt(xL)} {fmt(yBase)} V {fmt(yArch)}",
        f"M {fmt(xR)} {fmt(yBase)} V {fmt(yArch)}",
        f"M {fmt(xL)} {fmt(yArch)} Q {fmt(cx)} {fmt(yTop)} {fmt(xR)} {fmt(yArch)}",
        f"M {fmt(xL)} {fmt(yBarA)} H {fmt(xR)}",
    ]
    glyphs["A"] = Glyph(W, A_paths, [])

    # B (improved): stem + two identical D-bowls
    B_paths = [
        f"M {fmt(xL)} {fmt(yTop)} V {fmt(yBase)}",
        f"M {fmt(xL)} {fmt(yTop)} H {fmt(xFlat)} "
        f"A {fmt(bowl_r)} {fmt(bowl_r)} 0 0 1 {fmt(xBend)} {fmt(yTop + bowl_r)} "
        f"A {fmt(bowl_r)} {fmt(bowl_r)} 0 0 1 {fmt(xFlat)} {fmt(yMid)} "
        f"H {fmt(xL)}",
        f"M {fmt(xL)} {fmt(yMid)} H {fmt(xFlat)} "
        f"A {fmt(bowl_r)} {fmt(bowl_r)} 0 0 1 {fmt(xBend)} {fmt(yMid + bowl_r)} "
        f"A {fmt(bowl_r)} {fmt(bowl_r)} 0 0 1 {fmt(xFlat)} {fmt(yBase)} "
        f"H {fmt(xL)}",
    ]
    glyphs["B"] = Glyph(W, B_paths, [])

    # C (open circle, long way around)
    sx, sy = pt(o_cx, o_cy, o_r, -gap_deg)
    ex, ey = pt(o_cx, o_cy, o_r, +gap_deg)
    C_paths = [f"M {fmt(sx)} {fmt(sy)} A {fmt(o_r)} {fmt(o_r)} 0 1 0 {fmt(ex)} {fmt(ey)}"]
    glyphs["C"] = Glyph(W, C_paths, [])

    # D (stem + right half-bowl, ellipse-ish via two arcs)
    D_paths = [
        f"M {fmt(xL)} {fmt(yTop)} V {fmt(yBase)}",
        f"M {fmt(xL)} {fmt(yTop)} H {fmt(dxFlat)} "
        f"A {fmt(dxRX)} {fmt(dxRY)} 0 0 1 {fmt(xR)} {fmt(yMid)} "
        f"A {fmt(dxRX)} {fmt(dxRY)} 0 0 1 {fmt(dxFlat)} {fmt(yBase)} "
        f"H {fmt(xL)}",
    ]
    glyphs["D"] = Glyph(W, D_paths, [])

    # E
    E_paths = [
        f"M {fmt(xL)} {fmt(yTop)} V {fmt(yBase)}",
        f"M {fmt(xL)} {fmt(yTop)} H {fmt(xR)}",
        f"M {fmt(xL)} {fmt(yMid)} H {fmt(xR - 50)}",
        f"M {fmt(xL)} {fmt(yBase)} H {fmt(xR)}",
    ]
    glyphs["E"] = Glyph(W, E_paths, [])

    # F
    F_paths = [
        f"M {fmt(xL)} {fmt(yTop)} V {fmt(yBase)}",
        f"M {fmt(xL)} {fmt(yTop)} H {fmt(xR)}",
        f"M {fmt(xL)} {fmt(yMid)} H {fmt(xR - 70)}",
    ]
    glyphs["F"] = Glyph(W, F_paths, [])

    # G (C + inner bar)
    G_paths = C_paths + [f"M {fmt(cx)} {fmt(yMid + 110)} H {fmt(xR)}"]
    glyphs["G"] = Glyph(W, G_paths, [])

    # H
    H_paths = [
        f"M {fmt(xL)} {fmt(yTop)} V {fmt(yBase)}",
        f"M {fmt(xR)} {fmt(yTop)} V {fmt(yBase)}",
        f"M {fmt(xL)} {fmt(yMid)} H {fmt(xR)}",
    ]
    glyphs["H"] = Glyph(W, H_paths, [])

    # I
    glyphs["I"] = Glyph(W, [f"M {fmt(cx)} {fmt(yTop)} V {fmt(yBase)}"], [])

    # J (simple hooked J)
    jx = xR - 120
    jy_hook = yBase - 130
    J_paths = [
        f"M {fmt(jx)} {fmt(yTop)} V {fmt(jy_hook)} "
        f"Q {fmt(jx)} {fmt(yBase)} {fmt(xL + 150)} {fmt(yBase)}"
    ]
    glyphs["J"] = Glyph(W, J_paths, [])

    # K
    K_paths = [
        f"M {fmt(xL)} {fmt(yTop)} V {fmt(yBase)}",
        f"M {fmt(xL)} {fmt(yMid)} L {fmt(xR)} {fmt(yTop)}",
        f"M {fmt(xL)} {fmt(yMid)} L {fmt(xR)} {fmt(yBase)}",
    ]
    glyphs["K"] = Glyph(W, K_paths, [])

    # L
    glyphs["L"] = Glyph(W, [f"M {fmt(xL)} {fmt(yTop)} V {fmt(yBase)}",
                           f"M {fmt(xL)} {fmt(yBase)} H {fmt(xR)}"], [])

    # M
    M_paths = [
        f"M {fmt(xL)} {fmt(yBase)} V {fmt(yTop)}",
        f"M {fmt(xR)} {fmt(yBase)} V {fmt(yTop)}",
        f"M {fmt(xL)} {fmt(yTop)} L {fmt(cx)} {fmt(yMid)} L {fmt(xR)} {fmt(yTop)}",
    ]
    glyphs["M"] = Glyph(W, M_paths, [])

    # N
    N_paths = [
        f"M {fmt(xL)} {fmt(yBase)} V {fmt(yTop)}",
        f"M {fmt(xR)} {fmt(yBase)} V {fmt(yTop)}",
        f"M {fmt(xL)} {fmt(yTop)} L {fmt(xR)} {fmt(yBase)}",
    ]
    glyphs["N"] = Glyph(W, N_paths, [])

    # O (full circle)
    ox0 = o_cx + o_r
    oy0 = o_cy
    ox1 = o_cx - o_r
    oy1 = o_cy
    O_paths = [
        f"M {fmt(ox0)} {fmt(oy0)} "
        f"A {fmt(o_r)} {fmt(o_r)} 0 1 1 {fmt(ox1)} {fmt(oy1)} "
        f"A {fmt(o_r)} {fmt(o_r)} 0 1 1 {fmt(ox0)} {fmt(oy0)}"
    ]
    glyphs["O"] = Glyph(W, O_paths, [])

    # P (stem + top bowl like B’s top)
    P_paths = [
        f"M {fmt(xL)} {fmt(yTop)} V {fmt(yBase)}",
        f"M {fmt(xL)} {fmt(yTop)} H {fmt(xFlat)} "
        f"A {fmt(bowl_r)} {fmt(bowl_r)} 0 0 1 {fmt(xBend)} {fmt(yTop + bowl_r)} "
        f"A {fmt(bowl_r)} {fmt(bowl_r)} 0 0 1 {fmt(xFlat)} {fmt(yMid)} "
        f"H {fmt(xL)}",
    ]
    glyphs["P"] = Glyph(W, P_paths, [])

    # Q (O + tail)
    Q_paths = O_paths + [f"M {fmt(cx + 80)} {fmt(yMid + 170)} L {fmt(xR)} {fmt(yBase)}"]
    glyphs["Q"] = Glyph(W, Q_paths, [])

    # R (P + diagonal leg)
    R_paths = P_paths + [f"M {fmt(xFlat)} {fmt(yMid)} L {fmt(xR)} {fmt(yBase)}"]
    glyphs["R"] = Glyph(W, R_paths, [])

    # S (smooth-ish S using quadratic segments)
    S_paths = [(
        f"M {fmt(xR)} {fmt(yTop + 140)} "
        f"Q {fmt(cx)} {fmt(yTop)} {fmt(xL)} {fmt(yTop + 140)} "
        f"Q {fmt(xL - 40)} {fmt(yMid - 60)} {fmt(cx)} {fmt(yMid)} "
        f"Q {fmt(xR + 40)} {fmt(yMid + 60)} {fmt(xR)} {fmt(yBase - 140)} "
        f"Q {fmt(cx)} {fmt(yBase)} {fmt(xL)} {fmt(yBase - 140)}"
    )]
    glyphs["S"] = Glyph(W, S_paths, [])

    # T
    glyphs["T"] = Glyph(W, [f"M {fmt(xL)} {fmt(yTop)} H {fmt(xR)}",
                           f"M {fmt(cx)} {fmt(yTop)} V {fmt(yBase)}"], [])

    # U (stems + bottom arch)
    U_paths = [
        f"M {fmt(xL)} {fmt(yTop)} V {fmt(u_end_y)}",
        f"M {fmt(xR)} {fmt(yTop)} V {fmt(u_end_y)}",
        f"M {fmt(xL)} {fmt(u_end_y)} Q {fmt(cx)} {fmt(yBase)} {fmt(xR)} {fmt(u_end_y)}",
    ]
    glyphs["U"] = Glyph(W, U_paths, [])

    # V
    glyphs["V"] = Glyph(W, [f"M {fmt(xL)} {fmt(yTop)} L {fmt(cx)} {fmt(yBase)} L {fmt(xR)} {fmt(yTop)}"], [])

    # W
    glyphs["W"] = Glyph(W, [f"M {fmt(xL)} {fmt(yTop)} L {fmt(xL + 120)} {fmt(yBase)} "
                           f"L {fmt(cx)} {fmt(yMid)} "
                           f"L {fmt(xR - 120)} {fmt(yBase)} L {fmt(xR)} {fmt(yTop)}"], [])

    # X
    glyphs["X"] = Glyph(W, [f"M {fmt(xL)} {fmt(yTop)} L {fmt(xR)} {fmt(yBase)}",
                           f"M {fmt(xR)} {fmt(yTop)} L {fmt(xL)} {fmt(yBase)}"], [])

    # Y
    glyphs["Y"] = Glyph(W, [f"M {fmt(xL)} {fmt(yTop)} L {fmt(cx)} {fmt(yMid)} L {fmt(xR)} {fmt(yTop)}",
                           f"M {fmt(cx)} {fmt(yMid)} V {fmt(yBase)}"], [])

    # Z
    glyphs["Z"] = Glyph(W, [f"M {fmt(xL)} {fmt(yTop)} H {fmt(xR)}",
                           f"M {fmt(xR)} {fmt(yTop)} L {fmt(xL)} {fmt(yBase)}",
                           f"M {fmt(xL)} {fmt(yBase)} H {fmt(xR)}"], [])

    return glyphs


# -------------------------
# Lowercase glyphs (a–z)
# -------------------------

def build_lowercase(m: Metrics, stroke: float) -> Dict[str, Glyph]:
    W = m.LC_W
    xL = 120.0
    xR = W - 120.0
    cx = W / 2.0

    yBase = m.BASE
    yXTop = m.X_TOP
    yXMid = m.X_MID
    yAscTop = m.CAP_TOP  # ascenders up to same cap-top in this sketchy model

    # Lowercase bowl
    o_r = 220.0
    o_cx = cx
    o_cy = yXMid
    gap_deg = 35.0

    # For “d-bowl” style attachments
    bowl_rx = 190.0
    bowl_ry = 220.0
    # some handy x positions
    stemL = xL
    stemR = xR

    glyphs: Dict[str, Glyph] = {}

    # a (single-storey): bowl + right stem + short join
    a_paths = [
        # bowl (full circle)
        f"M {fmt(o_cx + o_r)} {fmt(o_cy)} "
        f"A {fmt(o_r)} {fmt(o_r)} 0 1 1 {fmt(o_cx - o_r)} {fmt(o_cy)} "
        f"A {fmt(o_r)} {fmt(o_r)} 0 1 1 {fmt(o_cx + o_r)} {fmt(o_cy)}",
        # right stem
        f"M {fmt(stemR)} {fmt(yXTop + 60)} V {fmt(yBase)}",
        # join
        f"M {fmt(o_cx + 30)} {fmt(yXMid)} H {fmt(stemR)}",
    ]
    glyphs["a"] = Glyph(W, a_paths, [])

    # b: tall left stem + bowl
    b_paths = [
        f"M {fmt(stemL)} {fmt(yAscTop)} V {fmt(yBase)}",
        f"M {fmt(stemL)} {fmt(yXTop + 60)} H {fmt(o_cx)} "
        f"A {fmt(bowl_rx)} {fmt(bowl_ry)} 0 0 1 {fmt(stemR)} {fmt(yXMid)} "
        f"A {fmt(bowl_rx)} {fmt(bowl_ry)} 0 0 1 {fmt(o_cx)} {fmt(yBase)} "
        f"H {fmt(stemL)}",
    ]
    glyphs["b"] = Glyph(W, b_paths, [])

    # c (open bowl)
    sx, sy = pt(o_cx, o_cy, o_r, -gap_deg)
    ex, ey = pt(o_cx, o_cy, o_r, +gap_deg)
    glyphs["c"] = Glyph(W, [f"M {fmt(sx)} {fmt(sy)} A {fmt(o_r)} {fmt(o_r)} 0 1 0 {fmt(ex)} {fmt(ey)}"], [])

    # d: bowl + tall right stem
    d_paths = [
        f"M {fmt(stemR)} {fmt(yAscTop)} V {fmt(yBase)}",
        f"M {fmt(stemR)} {fmt(yXTop + 60)} H {fmt(o_cx)} "
        f"A {fmt(bowl_rx)} {fmt(bowl_ry)} 0 0 0 {fmt(stemL)} {fmt(yXMid)} "
        f"A {fmt(bowl_rx)} {fmt(bowl_ry)} 0 0 0 {fmt(o_cx)} {fmt(yBase)} "
        f"H {fmt(stemR)}",
    ]
    glyphs["d"] = Glyph(W, d_paths, [])

    # e: open bowl + bar
    e_paths = [
        f"M {fmt(sx)} {fmt(sy)} A {fmt(o_r)} {fmt(o_r)} 0 1 0 {fmt(ex)} {fmt(ey)}",
        f"M {fmt(o_cx - 90)} {fmt(yXMid)} H {fmt(o_cx + 120)}",
    ]
    glyphs["e"] = Glyph(W, e_paths, [])

    # f: tall stem + crossbar
    fx = cx - 70
    f_paths = [
        f"M {fmt(fx)} {fmt(yAscTop)} V {fmt(yBase)}",
        f"M {fmt(fx - 110)} {fmt(yXTop + 40)} H {fmt(fx + 210)}",
        f"M {fmt(fx - 60)} {fmt(yXMid - 40)} H {fmt(fx + 120)}",
    ]
    glyphs["f"] = Glyph(W, f_paths, [])

    # g: bowl + descender tail
    g_paths = [
        f"M {fmt(o_cx + o_r)} {fmt(o_cy)} "
        f"A {fmt(o_r)} {fmt(o_r)} 0 1 1 {fmt(o_cx - o_r)} {fmt(o_cy)} "
        f"A {fmt(o_r)} {fmt(o_r)} 0 1 1 {fmt(o_cx + o_r)} {fmt(o_cy)}",
        f"M {fmt(stemR - 40)} {fmt(yXMid + 30)} Q {fmt(stemR + 40)} {fmt(yBase + 80)} {fmt(cx)} {fmt(m.DESC)}",
    ]
    glyphs["g"] = Glyph(W, g_paths, [])

    # h: tall stem + hump to right
    h_paths = [
        f"M {fmt(stemL)} {fmt(yAscTop)} V {fmt(yBase)}",
        f"M {fmt(stemL)} {fmt(yXTop)} Q {fmt(cx)} {fmt(yXTop - 40)} {fmt(stemR)} {fmt(yXTop)} V {fmt(yBase)}",
    ]
    glyphs["h"] = Glyph(W, h_paths, [])

    # i: stem + dot
    ix = cx
    dot_r = stroke * 0.32
    i_paths = [f"M {fmt(ix)} {fmt(yXTop)} V {fmt(yBase)}"]
    i_circles = [(ix, yXTop - 70, dot_r)]
    glyphs["i"] = Glyph(W, i_paths, i_circles)

    # j: stem desc + dot
    jx = cx
    j_paths = [f"M {fmt(jx)} {fmt(yXTop)} V {fmt(m.DESC)} Q {fmt(jx)} {fmt(m.DESC)} {fmt(jx - 120)} {fmt(m.DESC - 60)}"]
    j_circles = [(jx, yXTop - 70, dot_r)]
    glyphs["j"] = Glyph(W, j_paths, j_circles)

    # k: tall stem + diagonals
    k_paths = [
        f"M {fmt(stemL)} {fmt(yAscTop)} V {fmt(yBase)}",
        f"M {fmt(stemL)} {fmt(yXMid)} L {fmt(stemR)} {fmt(yXTop)}",
        f"M {fmt(stemL)} {fmt(yXMid)} L {fmt(stemR)} {fmt(yBase)}",
    ]
    glyphs["k"] = Glyph(W, k_paths, [])

    # l: tall stem
    glyphs["l"] = Glyph(W, [f"M {fmt(cx)} {fmt(yAscTop)} V {fmt(yBase)}"], [])

    # m: left stem + two humps
    m_paths = [
        f"M {fmt(stemL)} {fmt(yBase)} V {fmt(yXTop)}",
        f"M {fmt(stemL)} {fmt(yXTop)} Q {fmt(cx - 60)} {fmt(yXTop - 40)} {fmt(cx)} {fmt(yXTop)} V {fmt(yBase)}",
        f"M {fmt(cx)} {fmt(yXTop)} Q {fmt(cx + 120)} {fmt(yXTop - 40)} {fmt(stemR)} {fmt(yXTop)} V {fmt(yBase)}",
    ]
    glyphs["m"] = Glyph(W, m_paths, [])

    # n: left stem + one hump
    n_paths = [
        f"M {fmt(stemL)} {fmt(yBase)} V {fmt(yXTop)}",
        f"M {fmt(stemL)} {fmt(yXTop)} Q {fmt(cx)} {fmt(yXTop - 40)} {fmt(stemR)} {fmt(yXTop)} V {fmt(yBase)}",
    ]
    glyphs["n"] = Glyph(W, n_paths, [])

    # o: circle
    o_paths = [
        f"M {fmt(o_cx + o_r)} {fmt(o_cy)} "
        f"A {fmt(o_r)} {fmt(o_r)} 0 1 1 {fmt(o_cx - o_r)} {fmt(o_cy)} "
        f"A {fmt(o_r)} {fmt(o_r)} 0 1 1 {fmt(o_cx + o_r)} {fmt(o_cy)}"
    ]
    glyphs["o"] = Glyph(W, o_paths, [])

    # p: desc stem + bowl
    p_paths = [
        f"M {fmt(stemL)} {fmt(yXTop)} V {fmt(m.DESC)}",
        f"M {fmt(stemL)} {fmt(yXTop + 60)} H {fmt(o_cx)} "
        f"A {fmt(bowl_rx)} {fmt(bowl_ry)} 0 0 1 {fmt(stemR)} {fmt(yXMid)} "
        f"A {fmt(bowl_rx)} {fmt(bowl_ry)} 0 0 1 {fmt(o_cx)} {fmt(yBase)} "
        f"H {fmt(stemL)}",
    ]
    glyphs["p"] = Glyph(W, p_paths, [])

    # q: desc stem right + bowl
    q_paths = [
        f"M {fmt(stemR)} {fmt(yXTop)} V {fmt(m.DESC)}",
        f"M {fmt(stemR)} {fmt(yXTop + 60)} H {fmt(o_cx)} "
        f"A {fmt(bowl_rx)} {fmt(bowl_ry)} 0 0 0 {fmt(stemL)} {fmt(yXMid)} "
        f"A {fmt(bowl_rx)} {fmt(bowl_ry)} 0 0 0 {fmt(o_cx)} {fmt(yBase)} "
        f"H {fmt(stemR)}",
    ]
    glyphs["q"] = Glyph(W, q_paths, [])

    # r: short stem + shoulder
    r_paths = [
        f"M {fmt(stemL)} {fmt(yBase)} V {fmt(yXTop)}",
        f"M {fmt(stemL)} {fmt(yXTop)} Q {fmt(cx)} {fmt(yXTop - 30)} {fmt(cx + 80)} {fmt(yXTop)}",
    ]
    glyphs["r"] = Glyph(W, r_paths, [])

    # s: smaller S
    s_paths = [(
        f"M {fmt(stemR)} {fmt(yXTop + 90)} "
        f"Q {fmt(cx)} {fmt(yXTop)} {fmt(stemL)} {fmt(yXTop + 90)} "
        f"Q {fmt(stemL - 30)} {fmt(yXMid - 40)} {fmt(cx)} {fmt(yXMid)} "
        f"Q {fmt(stemR + 30)} {fmt(yXMid + 40)} {fmt(stemR)} {fmt(yBase - 90)} "
        f"Q {fmt(cx)} {fmt(yBase)} {fmt(stemL)} {fmt(yBase - 90)}"
    )]
    glyphs["s"] = Glyph(W, s_paths, [])

    # t: tall-ish stem + crossbar at x-top
    tx = cx - 60
    t_paths = [
        f"M {fmt(tx)} {fmt(yAscTop)} V {fmt(yBase)}",
        f"M {fmt(tx - 140)} {fmt(yXTop)} H {fmt(tx + 220)}",
    ]
    glyphs["t"] = Glyph(W, t_paths, [])

    # u: two stems + bottom arch
    u_end_y = yBase - 160
    u_paths = [
        f"M {fmt(stemL)} {fmt(yXTop)} V {fmt(u_end_y)}",
        f"M {fmt(stemR)} {fmt(yXTop)} V {fmt(u_end_y)}",
        f"M {fmt(stemL)} {fmt(u_end_y)} Q {fmt(cx)} {fmt(yBase)} {fmt(stemR)} {fmt(u_end_y)}",
    ]
    glyphs["u"] = Glyph(W, u_paths, [])

    # v
    glyphs["v"] = Glyph(W, [f"M {fmt(stemL)} {fmt(yXTop)} L {fmt(cx)} {fmt(yBase)} L {fmt(stemR)} {fmt(yXTop)}"], [])

    # w
    glyphs["w"] = Glyph(W, [f"M {fmt(stemL)} {fmt(yXTop)} L {fmt(stemL + 90)} {fmt(yBase)} "
                           f"L {fmt(cx)} {fmt(yXMid)} "
                           f"L {fmt(stemR - 90)} {fmt(yBase)} L {fmt(stemR)} {fmt(yXTop)}"], [])

    # x
    glyphs["x"] = Glyph(W, [f"M {fmt(stemL)} {fmt(yXTop)} L {fmt(stemR)} {fmt(yBase)}",
                           f"M {fmt(stemR)} {fmt(yXTop)} L {fmt(stemL)} {fmt(yBase)}"], [])

    # y: v + descender
    y_paths = [
        f"M {fmt(stemL)} {fmt(yXTop)} L {fmt(cx)} {fmt(yBase)} L {fmt(stemR)} {fmt(yXTop)}",
        f"M {fmt(cx)} {fmt(yBase)} V {fmt(m.DESC)}",
    ]
    glyphs["y"] = Glyph(W, y_paths, [])

    # z
    glyphs["z"] = Glyph(W, [f"M {fmt(stemL)} {fmt(yXTop)} H {fmt(stemR)}",
                           f"M {fmt(stemR)} {fmt(yXTop)} L {fmt(stemL)} {fmt(yBase)}",
                           f"M {fmt(stemL)} {fmt(yBase)} H {fmt(stemR)}"], [])

    return glyphs


# -------------------------
# Optional ligatures (hij, ik, sch)
# -------------------------

def build_ligatures(m: Metrics, stroke: float, lower: Dict[str, Glyph]) -> Dict[str, Glyph]:
    # Simple: place lowercase glyph sketches next to each other and add a joining bar (like your sample).
    # These are still “sketches” (stroked paths); no boolean-union here.
    adv = m.LC_W
    topbar_y = m.CAP_TOP + 120

    def shifted(g: Glyph, dx: float) -> Glyph:
        def shift_d(d: str) -> str:
            # naive numeric shift for "M x y", "L x y", "H x", "V y", "Q x1 y1 x y", "A rx ry ... x y"
            # We only shift x coordinates; y stays.
            out = []
            toks = d.replace(",", " ").split()
            i = 0
            cmd = None
            # This is intentionally lightweight; it works for the patterns used above.
            while i < len(toks):
                t = toks[i]
                if t.isalpha():
                    cmd = t
                    out.append(t)
                    i += 1
                    continue
                # numeric
                if cmd in ("V",):
                    out.append(t)
                    i += 1
                elif cmd in ("H",):
                    x = float(t) + dx
                    out.append(fmt(x))
                    i += 1
                elif cmd in ("M", "L"):
                    x = float(toks[i]) + dx
                    y = float(toks[i + 1])
                    out.extend([fmt(x), fmt(y)])
                    i += 2
                elif cmd in ("Q",):
                    x1 = float(toks[i]) + dx
                    y1 = float(toks[i + 1])
                    x = float(toks[i + 2]) + dx
                    y = float(toks[i + 3])
                    out.extend([fmt(x1), fmt(y1), fmt(x), fmt(y)])
                    i += 4
                elif cmd in ("A",):
                    # A rx ry rot laf sf x y
                    rx = toks[i]; ry = toks[i + 1]; rot = toks[i + 2]
                    laf = toks[i + 3]; sf = toks[i + 4]
                    x = float(toks[i + 5]) + dx
                    y = float(toks[i + 6])
                    out.extend([rx, ry, rot, laf, sf, fmt(x), fmt(y)])
                    i += 7
                else:
                    # fallback: shift alternating x,y pairs
                    x = float(toks[i]) + dx
                    y = float(toks[i + 1])
                    out.extend([fmt(x), fmt(y)])
                    i += 2
            return " ".join(out)

        return Glyph(
            width=g.width,
            paths=[shift_d(d) for d in g.paths],
            circles=[(cx + dx, cy, r) for (cx, cy, r) in g.circles],
        )

    ligs: Dict[str, Glyph] = {}

    # hij
    h = shifted(lower["h"], 0.0)
    i = shifted(lower["i"], adv)
    j = shifted(lower["j"], adv * 2)
    hij_paths = h.paths + i.paths + j.paths + [f"M {fmt(80)} {fmt(topbar_y)} H {fmt(adv*3 - 80)}"]
    hij_circles = h.circles + i.circles + j.circles
    ligs["hij"] = Glyph(width=adv * 3, paths=hij_paths, circles=hij_circles)

    # ik
    i2 = shifted(lower["i"], 0.0)
    k = shifted(lower["k"], adv)
    ligs["ik"] = Glyph(width=adv * 2, paths=(i2.paths + k.paths), circles=(i2.circles + k.circles))

    # sch
    s = shifted(lower["s"], 0.0)
    c = shifted(lower["c"], adv)
    h2 = shifted(lower["h"], adv * 2)
    sch_paths = s.paths + c.paths + h2.paths + [f"M {fmt(80)} {fmt(topbar_y)} H {fmt(adv*3 - 80)}"]
    sch_circles = s.circles + c.circles + h2.circles
    ligs["sch"] = Glyph(width=adv * 3, paths=sch_paths, circles=sch_circles)

    return ligs


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("sketches"), help="output directory (default: sketches)")
    ap.add_argument("--stroke", type=float, default=90.0, help="stroke thickness (SVG stroke-width)")
    ap.add_argument("--include-ligatures", action="store_true", help="also output hij/ik/sch ligature sketches")
    args = ap.parse_args()

    m = Metrics()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    upper = build_uppercase(m, args.stroke)
    lower = build_lowercase(m, args.stroke)

    preview_items: List[Tuple[str, str]] = []

    # A–Z
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        g = upper[ch]
        fname = codepoint_filename(ch)
        write_svg(out / fname, m, g, args.stroke)
        preview_items.append((ch, fname))

    # a–z
    for ch in "abcdefghijklmnopqrstuvwxyz":
        g = lower[ch]
        fname = codepoint_filename(ch)
        write_svg(out / fname, m, g, args.stroke)
        preview_items.append((ch, fname))

    if args.include_ligatures:
        ligs = build_ligatures(m, args.stroke, lower)
        for name, g in ligs.items():
            fname = "liga_" + codepoint_filename(name)
            write_svg(out / fname, m, g, args.stroke)
            preview_items.append((f"liga:{name}", fname))

    write_preview_html(out, preview_items)

    print(f"Wrote {len(preview_items)} SVGs to: {out}")
    print(f"Open: {out / 'preview.html'}")


if __name__ == "__main__":
    main()
