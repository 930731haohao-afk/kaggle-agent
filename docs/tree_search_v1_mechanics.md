# How the v1 Tree Search Works (harness / driver / lineages / mutations)

A mechanics reference for the Stage-4 candidate tree search, at the **v1** level
(`tree_search/harness.py` + a per-competition driver such as `tree_search/run_s3e9.py`).
v2/v3/v4 layer more features on top but the core loop below is unchanged. All line
references are to `tree_search/run_s3e9.py` (the s3e9 = Concrete Strength driver) and
`tree_search/harness.py`.

---

## 1. Three roles

| Role | File | Per-competition? | Responsibility |
|---|---|---|---|
| **Harness** | `harness.py` | No — written once, reused | The generic search engine: tree/node data structure, **which node to expand** (`select_next_parent`), plateau/backtrack bookkeeping, recording nodes (`add_root`/`add_node`). Knows nothing about any competition. |
| **Driver** | `run_<comp>.py` | Yes — one per comp | Supplies the competition-specific parts and runs the loop: root config, lineage seeds, mutation queues, and `propose_child` (**what change** to try). |
| **Evaluator** | `eval_<comp>.py` | Yes — one per comp | `evaluate(config) -> {score, ...}`: trains the models with that competition's data/metric and returns the CV score. |

**Division of labor in one line:** the harness decides *which* node to expand and *when*
to stop/backtrack; the driver decides *what* mutation each expansion tries; the evaluator
says *how good* the resulting config is.

---

## 2. The tree

- A **node** = one complete, evaluated **config** (`{model, params, features:{drop:[...]}, postprocess}`) plus its CV score. In v1 a node is a single model config; a config is *not* code (contrast with ERA, whose nodes are LLM-generated programs).
- There is exactly **one root** — the starting config, added once via `add_root` (`run_s3e9.py:302`). For s3e9 the root is a hand-regularized single LGB (num_leaves=15, depth=5, L1=2, L2=4), the "fair single-model reference point."
- A **lineage** = the direct child-of-root a node descends from — i.e. one *search direction* / candidate subtree. The root's first-generation children each start their own lineage.

```
                         root  (reference config, id=0)
        ┌───────────┬───────────┬───────────┬───────────┐
      CAT         XGB        ROBUST        REG         FEAT      ← lineage seeds = "heads"
   (→CatBoost) (→XGBoost) (→huber loss) (+regularize) (drop features)
```

---

## 3. The driver's three hand-authored ingredients

### 3a. Root config
The single starting config (`ROOT_CONFIG`, `run_s3e9.py:58`), evaluated and added as the root.

### 3b. Lineage seeds ("heads") — `LINEAGE_SEEDS` (`run_s3e9.py:158`)
A handful of authored functions, each producing one first-generation child of the root
that **opens a distinct search direction (a hypothesis)**. For s3e9:

| Head | Opening bet | Seed config |
|---|---|---|
| **CAT** | switch model → CatBoost | `model=cat`, depth6, l2_leaf_reg=6 |
| **XGB** | switch model → XGBoost | `model=xgb`, max_depth4, L1=2/L2=4 |
| **ROBUST** | switch the loss (still LGB) | `objective=huber`, alpha=10 |
| **REG** | push regularization / cut capacity (still LGB) | num_leaves=10, max_depth=4 |
| **FEAT** | drop features | drops 6 weak/redundant features |

Note: a lineage is a *branch of the search*, not a feature set. Four of the five heads
hold `features.drop = []` (full feature set, fixed) and vary the model; only **FEAT**
mutates features. Which lineage touches what is a driver design choice, not a property of
lineages.

### 3c. Per-lineage mutation queues (`run_s3e9.py:171`)
Each lineage has an **ordered list of "next changes to try"**. Each entry is a function
`fn(parent_config) -> (child_config, description)` that applies **one specific change** to
the parent (via `_bump` for hyperparameters, `_feat` for features). Example:

```python
CAT_QUEUE = [
    lambda c: (_bump(c, l2_leaf_reg=9.0),         "..."),   # entry 0
    lambda c: (_bump(c, depth=5),                 "..."),   # entry 1
    lambda c: (_bump(c, bagging_temperature=1.0), "..."),   # entry 2
    lambda c: (_bump(c, learning_rate=0.015, iterations=6000), "..."),  # entry 3
    lambda c: (_bump(c, random_strength=2.0),     "..."),   # entry 4
]
```
Queue lengths: CAT 5, XGB 5, ROBUST 4, REG 4, FEAT 4. So a lineage can tune **as many
distinct knobs as its queue lists** (not 3), then falls back to seed-variation.

---

## 4. How a child is proposed (the core loop)

Main loop (`run_s3e9.py:340`):
```python
parent_id, lineage_id = harness.select_next_parent(tree)          # harness: WHICH node
child_cfg, mutation   = propose_child(tree, parent_id, lineage_id) # driver:  WHAT change
r = ev.evaluate(child_cfg)                                         # evaluator: score
harness.add_node(tree, parent_id, mutation, child_cfg, r["score"], ...)  # harness: record
```

Two independent decisions produce each child:

1. **Which node to mutate** — `select_next_parent` returns the **best-scoring node in the active lineage that still has expansion budget** (`< MAX_CHILDREN_PER_NODE = 3` children).
2. **What change to apply** — `propose_child` (`run_s3e9.py:277`) uses
   `idx = lineage_size − 1` to pick the next unused queue entry and applies it to the
   chosen parent's config:
   ```python
   idx = harness.lineage_size(tree, lineage_id) - 1
   if idx < len(queue):
       child_cfg, desc = queue[idx](parent_node["config"])   # authored mutation
   else:
       child_cfg, desc = fallback_mutation(...)              # queue empty → seed variation
   ```

`_bump` (`run_s3e9.py:165`) is a **delta on the parent**: it deep-copies the parent config
and overrides only the named params; **all other params are inherited unchanged**.

> `idx` (queue position, a running count) is **not** the tree depth. It only says *which
> change* is next in authoring order; the node's *depth* is decided separately by
> `select_next_parent` (section 6).

The mutation operator is therefore **hand-authored driver code** — pre-written queues of
config transforms — not an LLM writing code in the loop and not anything the harness
generates. "The agent is the mutation proposer" means the agent wrote these queues ahead
of time; the loop mechanically walks them.

---

## 5. Cumulative vs. isolated changes

Because each mutation is a delta on **the chosen parent** (not on the seed), whether
changes accumulate depends on the path:

- **Down a chain** (parent = a previously-mutated node) → changes **accumulate**: a deep
  node carries all its ancestors' changes plus its own.
- **Across siblings** (parent = the same earlier node) → each child is *that node + one
  isolated change*, independent of its siblings.

Only the **winning path** accumulates; sibling dead-ends do not contribute their change to
later nodes.

---

## 6. When does the tree go one layer deeper?

The new node's depth = **depth(the parent `select_next_parent` chose) + 1**. Since that
parent is the *best node with budget*, the tree goes deeper for exactly two reasons:

1. **A deeper child became the new best** (improvement pulls the search down a level).
2. **The current-best node filled up** (3 children → no budget → the search drops to the
   next-best node with budget, usually one of its children).

It **stays on the same layer (adds a sibling)** when the best node is an earlier node that
still has `< 3` children.

Separately, **plateau/backtrack**: if a lineage's active node accumulates
`PLATEAU_STREAK = 3` consecutive children that fail to beat the global best, the lineage is
frozen and `select_next_parent` switches to the next-best *non-plateaued* lineage. If every
lineage is plateaued/exhausted, plateau flags are cleared once (a "reopen") so the search
can continue toward the node budget. (This is switching direction, not going deeper.)

---

## 7. Worked example — the CAT lineage on s3e9

RMSE (lower is better); seed ≈ 12.075. **Scores below are illustrative** to show the
mechanics; the configs and queue are the real ones.

| Step | Parent chosen (best w/ budget) | idx | Queue change | New node | Differs from seed by | RMSE |
|---|---|---|---|---|---|---|
| seed | — | — | — | **seed** | CatBoost depth6, l2=6 | 12.075 |
| 1 | seed (d0) | 0 | l2_leaf_reg 6→9 | **c1** | l2=9 | 12.062 ✓ |
| 2 | c1 (d1) | 1 | depth 6→5 | **c2** | l2=9, depth=5 | 12.058 ✓ |
| 3 | c2 (d2) | 2 | bagging_temperature=1.0 | **c3** | l2=9, depth=5, bag_temp | 12.060 ✗ |
| 4 | c2 (d2) | 3 | lr 0.03→0.015, iter→6000 | **c4** | l2=9, depth=5, lr=0.015 | 12.055 ✓ |
| 5 | c4 (d3) | 4 | random_strength=2.0 | **c5** | l2=9, depth=5, lr=0.015, rand_str | 12.056 ✗ |
| 6 | c4 (d3) | 5 → queue empty | fallback: same config, new seed | **c6** | = c4, different seed | 12.054 ✓ |

Resulting tree:
```
seed (12.075)                              depth 0
 └ c1 (12.062)  [l2=9]                      depth 1
     └ c2 (12.058)  [l2=9, depth=5]         depth 2
        ├ c3 (12.060)  [+bag_temp]   ✗      depth 3   ← sibling, lost
        ├ c4 (12.055)  [+lr/iter]    ✓ best depth 3
        │   └ c5 (12.056) [+rand_str] ✗     depth 4
        │       └ c6 (12.054) [seed-var] ✓  depth 5
```

What this one lineage shows:
- **Tuned 5 different parameters** (l2_leaf_reg, depth, bagging_temperature, lr/iterations, random_strength), not 3.
- **Accumulation down the winning chain** seed→c1→c2→c4: best node **c4 differs from the seed in 3 params at once**.
- **c3, c5 are sibling dead-ends** — c4 does not carry c3's bagging_temperature (its parent is c2, not c3). Only the winning path accumulates.
- **Depth reached 5**, well past 3 — `MAX_CHILDREN_PER_NODE=3` only capped children-per-node, pushing growth downward.

---

## 8. Common misconceptions (quick corrections)

| Misconception | Reality |
|---|---|
| Each `LINEAGE_SEEDS` function makes a root | No — one root; each seed is a *first-gen child* of that root and the **head of a lineage**. |
| A lineage = a feature set | No — a lineage is a *branch/search direction*. Every config carries a feature set, but only the FEAT lineage *changes* it here. |
| `MAX_CHILDREN_PER_NODE=3` means only 3 params per lineage | No — it caps *children per node* (branching), forcing depth. A lineage tunes as many knobs as its queue lists (+ fallback) and can be many nodes deep. |
| Queue index = tree depth | No — `idx` is a running count of "which authored change is next"; tree depth is set by `select_next_parent`. |
| The mutation is generated automatically / by an LLM in the loop | No (v1) — mutations are hand-authored config transforms in the queues; the agent wrote them ahead of time. |

---

## 9. Key constants (`harness.py`)

- `MAX_CHILDREN_PER_NODE = 3` — max children per node (branching factor).
- `PLATEAU_STREAK = 3` — consecutive non-improving children before a lineage is frozen.
- Selection rule: active lineage's best-scoring node with remaining child budget; on
  plateau, backtrack to the next-best non-plateaued lineage; reopen all if every lineage is
  frozen.
