"""Phase J stage-5:plateau 觸發的外部想法注入(把 idea_bank 真正接進搜尋)。

當樹搜尋 plateau(階段4推不動)時,把「與本場相符、且還沒試過」的 [EXT] idea_bank 條目
翻譯成搜尋能評估的候選 config,讓搜尋去試。每次注入最多 K 條、每場搜尋最多 M 次(M 由
driver 控管)。每個注入候選帶 `[EXT-NN]` provenance,供消融歸因。

這就是讓「階段5」從裝飾品變成真的會動的接線:idea_bank 被**真的讀取並轉成候選**,而不是
寫進沒人讀的 tree['priors'](J-3 發現的死欄位)。只有「有登錄翻譯器 + 還沒試過」的想法會被注入。

設計上刻意 comp-agnostic:translator 只吃 (ext 條目, 目前最佳解成員, 上下文),回傳候選 config;
新增想法只要在 TRANSLATORS 加一條。
"""
import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_hv4():
    spec = importlib.util.spec_from_file_location(
        "harness_v4", os.path.join(_HERE, "harness_v4.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------
# 想法 -> 候選 翻譯器(可擴充登錄表)。每個 translator:
#   (ext_entry, best_members, ctx) -> candidate_cfg | None
# best_members = 目前全域最佳 blend 的成員 node-id 清單。
# ---------------------------------------------------------------------------
def _rank_space_blend(ext, best_members, ctx):
    """EXT-12 rank averaging(Kaggle Ensembling Guide)-> 把目前最佳 blend 的成員改在
    RANK 空間評估(先把各成員 OOF 轉成名次再權重搜尋),而非機率空間。對輸出尺度不同的
    成員更穩,是文獻常見的集成技法。"""
    if not best_members or len(best_members) < 2:
        return None
    members = sorted({int(m) for m in best_members})
    return {"kind": "blend", "members": members,
            "weight_search": "dirichlet", "space": "rank"}


TRANSLATORS = {
    "EXT-12": _rank_space_blend,
    # 15 場鋪開時在此加更多 translator(頻率編碼、對抗驗證等)
}


def _cfg_sig(cfg):
    """候選去重簽名:同結構的候選只算一次(避免注入搜尋已試過的)。"""
    if cfg.get("kind") == "blend":
        return ("blend", cfg.get("space", "prob"), tuple(sorted(cfg.get("members", []))))
    return ("solo", cfg.get("model"), tuple(sorted((cfg.get("params") or {}).items())))


def plateau_inject(comp_meta, best_members, tried_sigs=(), *, k=3,
                   idea_bank_path=None, respect_dedup=True, hv4=None):
    """回傳最多 k 個 (cfg, desc, ext_id):與 comp_meta 相符、有翻譯器、且候選簽名不在
    tried_sigs 的外部想法。

    - comp_meta:{"metric":..., "tags":[...]} 之類,交給 harness_v4.suggest_ext_priors 比對。
    - best_members:目前全域最佳 blend 的成員 node-id(translator 據此生候選)。
    - tried_sigs:搜尋已評估過的候選簽名集合(用 tried_signatures(tree) 產)。
    - respect_dedup:True 時沿用 idea_bank 的 [INT] 去重(不注入已被內部經驗覆蓋的想法)。
    """
    hv4 = hv4 or _load_hv4()
    tried = set(tried_sigs)
    out = []
    for e in hv4.suggest_ext_priors(comp_meta, idea_bank_path=idea_bank_path,
                                    respect_dedup=respect_dedup):
        eid = e.get("source")
        translator = TRANSLATORS.get(eid)
        if translator is None:
            continue  # 這條想法還沒有翻譯器,跳過(誠實:不硬翻)
        cfg = translator(e, best_members, {})
        if cfg is None:
            continue
        sig = _cfg_sig(cfg)
        if sig in tried:
            continue  # 搜尋已試過這個候選,注入等於多餘,跳過
        tried.add(sig)
        desc = (f"[EXT-INJECT {eid}] plateau-triggered external-idea candidate: {eid} "
                f"-> {cfg.get('space', 'prob')}-space blend of {len(cfg.get('members', []))} "
                f"best-node members [provenance=EXT]")
        out.append((cfg, desc, eid))
        if len(out) >= k:
            break
    return out


def tried_signatures(tree):
    """從樹的所有節點 config 產出「已試候選簽名」集合,供 plateau_inject 去重。"""
    sigs = set()
    for n in tree.get("nodes", []):
        cfg = n.get("config")
        if isinstance(cfg, dict):
            try:
                sigs.add(_cfg_sig(cfg))
            except Exception:
                pass
    return sigs
