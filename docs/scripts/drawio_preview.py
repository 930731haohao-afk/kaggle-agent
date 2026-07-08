"""docs/scripts/drawio_preview.py — 把 drawio_gen 的 (nodes, edges) 規格用 matplotlib
渲成 PNG 預覽,讓人不必開 draw.io 也能先看圖。渲染僅供預覽;可編輯的原生檔仍是 .drawio。

`uv run python3 docs/scripts/drawio_preview.py` → docs/tree_search_architecture.png
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from drawio_gen import _tree_search_diagram


def _wrap(line: str, max_units: float) -> list[str]:
    """把一行依「框寬容納單位數」換行;CJK 字算 1.0、ASCII 算 0.55 單位。"""
    out, cur, used = [], "", 0.0
    for ch in line:
        u = 1.0 if ord(ch) > 0x2E7F else 0.55
        if used + u > max_units and cur:
            out.append(cur)
            cur, used = "", 0.0
        cur += ch
        used += u
    if cur:
        out.append(cur)
    return out or [""]


def _wrapped_label(n, fontsize) -> str:
    cap = max(6, (n.w - 12) / (fontsize * 1.05))   # 框內可容納的單位數
    lines = []
    for raw in n.label.split("\n"):
        lines.extend(_wrap(raw, cap))
    return "\n".join(lines)

plt.rcParams["font.family"] = "Noto Sans CJK JP"  # 含繁中字符集
plt.rcParams["axes.unicode_minus"] = False

# style key -> (fill, stroke)
COLORS = {
    "start": ("#d5e8d4", "#82b366"), "end": ("#d5e8d4", "#82b366"),
    "process": ("#dae8fc", "#6c8ebf"), "accent": ("#ffe6cc", "#d79b00"),
    "decision": ("#fff2cc", "#d6b656"), "note": ("#fffbe6", "#d6b656"),
    "data": ("#f5f5f5", "#666666"),
}


def _fontsize(n):
    if "fontSize=14" in n.extra_style:
        return 12
    if n.style == "note":
        return 8.0
    return 8.5


def _draw_node(ax, n):
    fill, stroke = COLORS.get(n.style, COLORS["process"])
    cx, cy = n.x + n.w / 2, n.y + n.h / 2
    if n.style == "decision":
        pts = [(cx, n.y), (n.x + n.w, cy), (cx, n.y + n.h), (n.x, cy)]
        ax.add_patch(Polygon(pts, closed=True, facecolor=fill, edgecolor=stroke, lw=1.4))
    elif n.style in ("start", "end"):
        ax.add_patch(FancyBboxPatch((n.x, n.y), n.w, n.h,
                                    boxstyle="round,pad=0,rounding_size=10",
                                    facecolor=fill, edgecolor=stroke, lw=1.4))
    else:
        ax.add_patch(Rectangle((n.x, n.y), n.w, n.h, facecolor=fill, edgecolor=stroke, lw=1.4))
    align = "left" if n.style == "note" else "center"
    fs = _fontsize(n)
    tx = n.x + 6 if align == "left" else cx
    ty = n.y + 6 if n.style == "note" else cy
    va = "top" if n.style == "note" else "center"
    weight = "bold" if "fontStyle=1" in n.extra_style else "normal"
    ax.text(tx, ty, _wrapped_label(n, fs), ha=align, va=va, fontsize=fs,
            color="#222222", weight=weight, zorder=5, linespacing=1.35)


def _ortho_pts(a, b):
    """正交(Manhattan)走線:依主方向從框的邊中點出發、經一個轉折,進到對向邊中點。"""
    acx, acy = a.x + a.w / 2, a.y + a.h / 2
    bcx, bcy = b.x + b.w / 2, b.y + b.h / 2
    dx, dy = bcx - acx, bcy - acy
    if abs(dy) >= abs(dx):                       # 垂直為主:上/下出入
        if dy > 0:
            p0, p1 = (acx, a.y + a.h), (bcx, b.y)
        else:
            p0, p1 = (acx, a.y), (bcx, b.y + b.h)
        midy = (p0[1] + p1[1]) / 2
        return [p0, (p0[0], midy), (p1[0], midy), p1]
    else:                                        # 水平為主:左/右出入
        if dx > 0:
            p0, p1 = (a.x + a.w, acy), (b.x, bcy)
        else:
            p0, p1 = (a.x, acy), (b.x + b.w, bcy)
        midx = (p0[0] + p1[0]) / 2
        return [p0, (midx, p0[1]), (midx, p1[1]), p1]


def _draw_edge(ax, nmap, e):
    pts = _ortho_pts(nmap[e.source], nmap[e.target])
    ls = (0, (4, 3)) if e.dashed else "solid"
    codes = [Path.MOVETO] + [Path.LINETO] * (len(pts) - 1)
    ax.add_patch(PathPatch(Path(pts, codes), fill=False, lw=1.2,
                           edgecolor="#555555", linestyle=ls, zorder=3,
                           joinstyle="miter"))
    # 箭頭:最後一段
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=12,
                                 lw=0, color="#555555", shrinkA=0, shrinkB=0, zorder=4))
    if e.label:
        mid = pts[len(pts) // 2]
        ax.text(mid[0], mid[1], e.label, ha="center", va="center", fontsize=7.5,
                color="#444444", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.92))


def render(nodes, edges, out_path):
    nmap = {n.id: n for n in nodes}
    xs = [n.x for n in nodes] + [n.x + n.w for n in nodes]
    ys = [n.y for n in nodes] + [n.y + n.h for n in nodes]
    pad = 20
    fig_w = (max(xs) - min(xs) + 2 * pad) / 96
    fig_h = (max(ys) - min(ys) + 2 * pad) / 96
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    for e in edges:
        _draw_edge(ax, nmap, e)
    for n in nodes:
        _draw_node(ax, n)
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.invert_yaxis()          # draw.io 的 y 向下
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.normpath(os.path.join(here, "..", "tree_search_architecture.png"))
    nodes, edges = _tree_search_diagram()
    render(nodes, edges, out)
    print(f"wrote {out}")
