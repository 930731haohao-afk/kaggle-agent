# Firing-class AIDE lanes (3-lane coverage for sep-2022 and s5e1)

Pinned conditions: 20 steps, exec.timeout=1800, isolated root bench-comps/, 4h cap, in-process lane mutex.

- `07-31 11:44` [1/1] START playground-series-s5e1
- `07-31 12:07` [1/1] DONE playground-series-s5e1: rc=0 (23 min) -- submission + journal under aideml-runs/playground-series-s5e1/
- `07-31 12:07` FIRING-CLASS AIDE LANES COMPLETE -- next: NVIDIA lanes (agent sessions), then my-agent s5e1 full-pipeline lane, then submit everything for LB scores
- `07-31 12:12` === my-agent full-pipeline firing-class lanes starting ===
- `07-31 12:12` START my-agent/tabular-playground-series-sep-2022 (pin 104792e)
- `07-31 12:12` knowledge pinned to 104792e; backup in /tmp/knowledge-backup-d4YW
- `07-31 12:40` knowledge restored from /tmp/knowledge-backup-d4YW; lane-written versions saved under /home/tjyen/ai_agents/kaggle/competitions/tabular-playground-series-sep-2022.v5arms-20260731/lane_knowledge_diff/
- `07-31 12:40` DONE my-agent/tabular-playground-series-sep-2022: submission written (28 min, rc=0)
- `07-31 12:40` START my-agent/playground-series-s5e1 (pin 9868d9e)
- `07-31 12:40` knowledge pinned to 9868d9e; backup in /tmp/knowledge-backup-UrY0
- `07-31 12:47` knowledge restored from /tmp/knowledge-backup-UrY0; lane-written versions saved under /home/tjyen/ai_agents/kaggle/competitions/playground-series-s5e1.v5arms-20260731/lane_knowledge_diff/
- `07-31 12:47` DONE my-agent/playground-series-s5e1: NO SUBMISSION (6 min, rc=1) — needs a human look
- `07-31 12:47` MY-AGENT FIRING-CLASS LANES COMPLETE — next: NVIDIA lanes, then submit everything
- `07-31 13:42` usage-limit reset reached; relaunching my-agent/s5e1 lane
- `07-31 13:42` === my-agent full-pipeline firing-class lanes starting ===
- `07-31 13:42` START my-agent/playground-series-s5e1 (pin 9868d9e)
- `07-31 13:42` knowledge pinned to 9868d9e; backup in /tmp/knowledge-backup-EgVa
- `07-31 14:24` knowledge restored from /tmp/knowledge-backup-EgVa; lane-written versions saved under /home/tjyen/ai_agents/kaggle/competitions/playground-series-s5e1.v5arms-20260731/lane_knowledge_diff/
- `07-31 14:24` DONE my-agent/playground-series-s5e1: submission written (41 min, rc=0)
- `07-31 14:24` MY-AGENT FIRING-CLASS LANES COMPLETE — next: NVIDIA lanes, then submit everything
- `07-31 14:26` === NVIDIA firing-class lanes starting ===
- `07-31 14:26` START nvidia/tabular-playground-series-sep-2022
- `07-31 14:30` DONE nvidia/tabular-playground-series-sep-2022: submission written (3 min, rc=0)
- `07-31 14:30` START nvidia/playground-series-s5e1
- `07-31 14:35` DONE nvidia/playground-series-s5e1: submission written (5 min, rc=0)
- `07-31 14:35` NVIDIA FIRING-CLASS LANES COMPLETE — submit everything and build the 3-way table
- `07-31 14:40` SCORED nvidia/playground-series-s5e1: pub 0.05726 / priv 0.14594 (pub matches source kernel's claimed LB exactly)
- `07-31 14:40` ALL 6 LANES SCORED — 3-way tables final:
  - s5e1 (MAPE, priv): my-agent 0.12752 > NVIDIA 0.14594 > AIDE 0.17917. Paired test: M>A SIG; M-vs-N SIGN-FLIP (pub says N by .045, priv says M by .018); N-vs-A undecidable (mov .0352 > gap .0332).
  - sep-2022 (SMAPE, priv): NVIDIA 5.48351 > AIDE 9.72095 > my-agent 28.54859 — all pairwise SIG.
  - pub->priv degradation: s5e1 N +155% / A +42% / M +25%; sep-2022 A +68% / N +15% / M +1%.
