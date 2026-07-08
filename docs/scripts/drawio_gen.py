"""docs/scripts/drawio_gen.py — 極簡、可重用的 draw.io (diagrams.net) 流程圖產生器。

draw.io 的原生檔就是 mxGraph XML,uncompressed 也能直接開。本模組把「節點 + 連線」
的 Python 規格轉成合法的 .drawio 檔——不需要 draw.io plugin、不需要瀏覽器/JS,純 Python
以 `uv run` 執行即可。產出的 .drawio 檔在 draw.io 桌面版與 app.diagrams.net 皆可直接開啟編輯。

用法(程式內)::

    from drawio_gen import Node, Edge, build_drawio, write_drawio
    nodes = [Node("a", "開始", 40, 40, style="start"),
             Node("b", "處理", 40, 140, style="process")]
    edges = [Edge("a", "b", "下一步")]
    write_drawio("out.drawio", nodes, edges, title="範例")

也可當腳本跑:直接執行本檔會產出 docs/tree_search_architecture.drawio(見檔尾 __main__)。

style 預設:start/end(圓角綠)、process(藍框)、decision(黃菱形)、data(灰平行四邊形)、
group(淺色群組框,當背景分層用)、note(黃便利貼)。可用 `extra_style` 疊加任意 mxGraph 樣式。
"""
from __future__ import annotations

import html
import os
from dataclasses import dataclass, field

# --- 樣式預設(mxGraph style 字串)-------------------------------------------------
STYLES = {
    "start":    "rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;",
    "end":      "rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;",
    "process":  "rounded=0;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;",
    "decision": "rhombus;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;",
    "data":     "shape=parallelogram;perimeter=parallelogramPerimeter;whiteSpace=wrap;"
                "html=1;fillColor=#f5f5f5;strokeColor=#666666;",
    "group":    "rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#b3b3b3;"
                "dashed=1;verticalAlign=top;fontStyle=2;fontColor=#666666;",
    "note":     "shape=note;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;"
                "align=left;verticalAlign=top;",
    "accent":   "rounded=0;whiteSpace=wrap;html=1;fillColor=#ffe6cc;strokeColor=#d79b00;",
}
DEFAULT_STYLE = STYLES["process"]


@dataclass
class Node:
    id: str
    label: str
    x: float
    y: float
    w: float = 160
    h: float = 50
    style: str = "process"           # STYLES 的 key,或直接給完整 mxGraph style 字串
    extra_style: str = ""            # 疊加樣式(如 "fontSize=11;")
    parent: str = "1"                # mxGraph 父層 id(用 group 分層時填 group 的 id)

    def resolved_style(self) -> str:
        base = STYLES.get(self.style, self.style)
        return base + self.extra_style


@dataclass
class Edge:
    source: str
    target: str
    label: str = ""
    style: str = ("edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;"
                  "endArrow=block;endFill=1;")
    extra_style: str = ""
    dashed: bool = False

    def resolved_style(self) -> str:
        s = self.style + self.extra_style
        if self.dashed:
            s += "dashed=1;"
        return s


def _esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def build_drawio(nodes: list[Node], edges: list[Edge], *, title: str = "Diagram") -> str:
    """把 nodes/edges 組成一份完整、uncompressed 的 .drawio XML 字串。"""
    cells: list[str] = []
    # mxGraph 必備的兩個根 cell
    cells.append('<mxCell id="0" />')
    cells.append('<mxCell id="1" parent="0" />')

    for n in nodes:
        cells.append(
            f'<mxCell id="{_esc(n.id)}" value="{_esc(n.label)}" '
            f'style="{_esc(n.resolved_style())}" vertex="1" parent="{_esc(n.parent)}">'
            f'<mxGeometry x="{n.x}" y="{n.y}" width="{n.w}" height="{n.h}" as="geometry" />'
            f'</mxCell>'
        )

    for i, e in enumerate(edges):
        cells.append(
            f'<mxCell id="e{i}" value="{_esc(e.label)}" '
            f'style="{_esc(e.resolved_style())}" edge="1" parent="1" '
            f'source="{_esc(e.source)}" target="{_esc(e.target)}">'
            f'<mxGeometry relative="1" as="geometry" />'
            f'</mxCell>'
        )

    body = "\n        ".join(cells)
    return (
        '<mxfile host="drawio_gen.py">\n'
        f'  <diagram id="d0" name="{_esc(title)}">\n'
        '    <mxGraphModel dx="1200" dy="800" grid="1" gridSize="10" guides="1" '
        'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
        'pageWidth="1169" pageHeight="826" math="0" shadow="0">\n'
        '      <root>\n'
        f'        {body}\n'
        '      </root>\n'
        '    </mxGraphModel>\n'
        '  </diagram>\n'
        '</mxfile>\n'
    )


def write_drawio(path: str, nodes: list[Node], edges: list[Edge], *, title: str = "Diagram") -> str:
    xml = build_drawio(nodes, edges, title=title)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)
    return path


# ---------------------------------------------------------------------------
# __main__:產出本專案樹搜尋架構流程圖
# ---------------------------------------------------------------------------
def _tree_search_diagram():
    """本專案 harness(v1→v3)樹搜尋迴圈的流程圖規格。"""
    COL = 360          # 主流程欄位 x
    W = 240
    nodes = [
        Node("t", "Kaggle-Agent 樹搜尋架構(harness v1→v3;確定性、GBDT 組態空間)",
             40, 20, 900, 40, style="note",
             extra_style="fontSize=14;fontStyle=1;align=center;verticalAlign=middle;"),

        Node("seed", "driver 手動種下第一代世系\n(root 的數個子節點=不同起始方向)",
             COL, 90, W, 56, style="start"),
        Node("phase", "相位機 update_phase\nexploit → explore_burst → stopped",
             COL, 176, W, 56, style="accent"),
        Node("stop", "should_stop ?\n(預算 60 節點 / burst 後無改善)",
             COL, 262, W, 60, style="decision"),
        Node("select", "select_next_parent\n=active 世系中分數最佳、仍有展開額度的節點",
             COL, 352, W, 60, style="process"),
        Node("propose", "提出變異 propose_child\n種子 / boundary-push / 重組",
             COL, 442, W, 60, style="process"),
        Node("dedup", "config_hash 去重?\n(同父連兩次重複 → 燒 failed 佔位)",
             COL, 532, W, 60, style="decision"),
        Node("eval", "評分 evaluate(config)\nsolo:K-fold→OOF 快取  |  blend:OOF 權重搜尋(Dirichlet k=800+座標上升)",
             COL - 70, 622, W + 140, 60, style="process"),
        Node("add", "add_node → 比對全域最佳\n贏:streak=0 | 沒贏:streak+1",
             COL, 712, W, 56, style="process"),
        Node("plateau", "streak ≥ 門檻(3;離散指標5)?\n→ 世系 plateaued",
             COL, 798, W, 60, style="decision"),

        # 右側分支
        Node("backtrack", "回溯:改選次佳的\n非-plateaued 世系",
             COL + 330, 798, 200, 60, style="process"),
        Node("burst", "全部 plateaued →\n強制 explore_burst\n(注入 5–8 條長射程新世系\n+ burst 種子健全閘)",
             COL + 330, 262, 200, 90, style="accent"),
        Node("done", "輸出全域最佳解\n→ OOF 逐位重現閘門\n→ 產生 submission",
             COL + 330, 90, 200, 80, style="end"),

        # 左側註記:各能力來自哪一層
        Node("layers",
             "分層來源:\n"
             "• v1 harness.py:節點/樹、select_next_parent、plateau 回溯\n"
             "• v2:solo/blend kind、OOF 快取、eval_blend 權重搜尋、dedup、metric-aware plateau、suggest_priors\n"
             "• v3:預算相位機、dedup 燒預算、boundary-push、座標上升、成本護欄、子行程逾時、resume 契約\n"
             "• v4:外部想法注入 hook(J-3:write-only=no-op)",
             30, 352, 300, 210, style="note", extra_style="fontSize=11;"),
    ]
    edges = [
        Edge("seed", "phase"),
        Edge("phase", "stop"),
        Edge("stop", "select", "否(繼續)"),
        Edge("stop", "done", "是(停止)"),
        Edge("select", "propose"),
        Edge("propose", "dedup"),
        Edge("dedup", "propose", "重複→重提", dashed=True),
        Edge("dedup", "eval", "新組態"),
        Edge("eval", "add"),
        Edge("add", "plateau"),
        Edge("plateau", "phase", "否→回相位機"),
        Edge("plateau", "backtrack", "是"),
        Edge("backtrack", "phase"),
        Edge("phase", "burst", "全 plateaued", dashed=True),
        Edge("burst", "select"),
    ]
    return nodes, edges


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "tree_search_architecture.drawio")
    nodes, edges = _tree_search_diagram()
    p = write_drawio(out, nodes, edges, title="Tree Search Architecture")
    print(f"wrote {os.path.normpath(p)}  ({len(nodes)} nodes, {len(edges)} edges)")
