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
                  "endArrow=block;endFill=1;jettySize=auto;")
    extra_style: str = ""
    dashed: bool = False
    points: list = field(default=None)   # 明確轉折點 [(x,y), ...],走側通道用

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
        if e.points:
            pts = "".join(f'<mxPoint x="{px}" y="{py}" />' for px, py in e.points)
            geom = (f'<mxGeometry relative="1" as="geometry">'
                    f'<Array as="points">{pts}</Array></mxGeometry>')
        else:
            geom = '<mxGeometry relative="1" as="geometry" />'
        cells.append(
            f'<mxCell id="e{i}" value="{_esc(e.label)}" '
            f'style="{_esc(e.resolved_style())}" edge="1" parent="1" '
            f'source="{_esc(e.source)}" target="{_esc(e.target)}">'
            f'{geom}</mxCell>'
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
    """本專案 harness(v1→v3)樹搜尋迴圈的流程圖規格。版面:中央主迴圈,
    左側走回饋通道(loop-back / dup-retry),右側放結束與 explore 分支,底部整條放分層註記。"""
    CX = 630                       # 主軸中心 x
    W = 320                        # 一般節點寬
    XP = CX - W / 2                # 一般節點左緣 = 470
    DW = 300                       # 決策(菱形)寬
    XD = CX - DW / 2               # 決策左緣 = 480
    RX = 1000                      # 右欄左緣
    nodes = [
        Node("t", "Kaggle-Agent 樹搜尋架構(harness v1→v3;確定性、GBDT 組態空間)",
             40, 20, 1200, 44, style="note",
             extra_style="fontSize=14;fontStyle=1;align=center;verticalAlign=middle;fillColor=#eef3fb;strokeColor=#6c8ebf;"),

        Node("seed", "driver 手動種下第一代世系\n(root 的數個子節點 = 不同起始方向)",
             XP, 100, W, 56, style="start"),
        Node("phase", "相位機 update_phase\nexploit → explore_burst → stopped",
             XP, 196, W, 56, style="accent"),
        Node("stop", "should_stop ?\n預算 60 節點 / burst 後無改善",
             XD, 292, DW, 90, style="decision"),
        Node("select", "select_next_parent\nactive 世系中分數最佳、仍有展開額度的節點\n(全 plateaued 時自動 reopen 次佳世系)",
             XP, 418, W, 72, style="process"),
        Node("propose", "提出變異 propose_child\n種子 / boundary-push / 重組",
             XP, 528, W, 60, style="process"),
        Node("dedup", "config_hash 去重 ?\n同父連兩次重複 → 燒 failed 佔位",
             XD, 624, DW, 90, style="decision"),
        Node("eval", "評分 evaluate(config)\nsolo:K-fold → OOF 快取   |   blend:OOF 權重搜尋(Dirichlet k=800 + 座標上升)",
             370, 752, 520, 64, style="process"),
        Node("add", "add_node → 比對全域最佳\n贏:streak=0    |    沒贏:streak+1",
             XP, 852, W, 60, style="process"),
        Node("plateau", "streak ≥ 門檻(3;離散指標 5)?\n是 → 世系 plateaued(下輪 select 改選次佳)",
             XD - 30, 948, DW + 60, 96, style="decision"),

        # 右欄:結束 + explore 分支
        Node("done", "輸出全域最佳解\n→ OOF 逐位重現閘門\n→ 產生 submission",
             RX, 100, 240, 90, style="end"),
        Node("burst", "全部 plateaued →\n強制 explore_burst\n注入 5–8 條長射程新世系\n+ burst 種子健全閘",
             RX, 300, 240, 120, style="accent"),

        # 底部整條:各能力來自哪一層(單行不換行)
        Node("layers",
             "分層來源(harness 疊加式,舊層原封不動供重現):\n"
             "• v1 harness.py:節點/樹結構、select_next_parent 選點、plateau 連續不進步回溯\n"
             "• v2:solo/blend kind、OOF 快取、eval_blend 權重搜尋、child dedup、metric-aware plateau、suggest_priors\n"
             "• v3:預算相位機、dedup 燒預算、boundary-push、座標上升精修、成本護欄、子行程硬逾時、burst 健全閘、resume 契約\n"
             "• v4:外部想法注入 hook（J-3 發現:tree['priors'] write-only = 對搜尋 no-op）",
             60, 1090, 1180, 150, style="note", extra_style="fontSize=12;"),
    ]
    edges = [
        # 主迴圈(相鄰,直落)
        Edge("seed", "phase"),
        Edge("phase", "stop"),
        Edge("stop", "select", "否(繼續)"),
        Edge("select", "propose"),
        Edge("propose", "dedup"),
        Edge("dedup", "eval", "新組態"),
        Edge("eval", "add"),
        Edge("add", "plateau"),
        # 回饋:左通道
        Edge("plateau", "phase", "回相位機", points=[(410, 996), (410, 224)]),
        Edge("dedup", "propose", "重複→重提", dashed=True, points=[(430, 669), (430, 558)]),
        # 結束 + explore:右側
        Edge("stop", "done", "是(停止)", points=[(880, 337), (880, 145)]),
        Edge("phase", "burst", "全 plateaued", dashed=True, points=[(940, 224), (940, 360)]),
        Edge("burst", "select", points=[(910, 360), (910, 454)]),
    ]
    return nodes, edges


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "tree_search_architecture.drawio")
    nodes, edges = _tree_search_diagram()
    p = write_drawio(out, nodes, edges, title="Tree Search Architecture")
    print(f"wrote {os.path.normpath(p)}  ({len(nodes)} nodes, {len(edges)} edges)")
