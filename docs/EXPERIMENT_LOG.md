# Huang Wei-Hao Weekly Progress Report — Week 2 (Experiment Log)

## 7/6

- Following the plan's goals, built an initial kaggle report skill so that once the agent finishes a competition it can autonomously produce a report covering:
  1. Competition purpose (what) / importance (why)
  2. The five workflow components (data spec, model spec, training spec, inference procedure, evaluation metric)
  3. Rubric checklist
  4. Score breakthrough points
  5. Experiment trajectory
  6. Reproduction instructions

- Had Claude run under four separate strategies:
  1. Without the kaggle agent skill
  2. With the kaggle agent skill
  3. kaggle agent skill + linear self-iteration (self improvement)
  4. kaggle agent skill + tree search
  Each ran all ten competitions and we compared score changes (result: for all four strategies, scores were monotonically non-decreasing in every competition; tree search beat linear iteration 9 wins and 1 tie out of 10; for s3e16, the best solution found by tree search was actually submitted to Kaggle, and it beat the skill-only version on both the Public and Private leaderboards, proving the local advantage is not overfitting).

## 7/7

- Refined the competition report structure, adding:
  1. Table of contents (every report and the preface PDF automatically gets a table of contents + page numbers)
  2. Explanation of technical terms
  3. Strategies adopted
  4. Conclusion

- Told Claude Code not to invent its own terminology and to improve readability (invented or hard-to-understand terms such as "ablation," "caliber," and "probe" were all changed to plain language; the "Stage N" cells in tables were simplified to bold numbers; "exp N" was changed to "the Nth experiment").

- Consolidated the frequently repeated structures, methods, and terms appearing across the competition reports into a single preface report to reduce repeated writing. Also mandated that after finishing all competitions a summary report be written, summarizing the score changes for each competition after running each strategy.

- Added s4e11, s5e10, s6e1, s6e2 as compensation for six other competitions that could not be entered (no "accept the rules" button) (still running; s4e1 and s4e11 are done; the score ladder still holds in Season 4 but with smaller increments than Season 3, and we found a negative case where an experience-library prior transferred to a different metric fails).

- Added a fifth strategy (still running):
  5. kaggle agent skill + tree search + externally introduced literature
  (concern: whether introducing external literature will consume too large a context window)

- Under the after-hours instruction to "not rest overnight and keep producing output," Claude autonomously completed (committed that day):
  1. First landing of the fifth strategy: external idea bank (idea_bank) with 24 real-provenance techniques (9f263a8); tree search v4 injection hooks ([INT] internal experience + [EXT] external ideas merged into one pool with dedup, 2daf362) + recombination mutation (0330b70)
  2. Complete cross-season unit s5e10: four stages 0.056074→0.055968, reproduction gate passed bit-by-bit, submission closed so recorded as CV-only (bb1f96b); mega-blend won this competition (opposite of s4e11)
  3. Complete cross-season unit s6e1: R2 four stages 0.786414→0.787183, gate reproduced "at the bit level" (7f3f505); the interaction-term prior was rejected here but confirmed on s5e10 — the second instance of the same prior yielding opposite verdicts across competitions
  4. s6e2 (the last cross-season competition) running in the background
  5. Along the way fixed two engineering bugs and wrote them back to memory: the pgrep self-match in the overnight-completion watcher, and LightGBM cross-process non-determinism (which broke the reproduction gate; now fixed by pinning the deterministic flags to achieve bit-level reproducibility)

## 7/8

- The overnight autonomous run continued into the early morning (8 commits between 02:44–04:09 that day, 15 in total counting the previous night); had Claude do:
  1. Finished the last cross-season competition s6e2 (heart disease, AUC): four stages 0.95498→0.955529, OOF gate reproduced at bit level (max|dOOF|=0), submission closed so recorded as CV-only. **All five cross-season competitions (s4e1/s4e11/s5e10/s6e1/s6e2) complete**, with the four-stage ladder holding in every competition
  2. Extended the cross-competition comparison table to 15 competitions (modified the benchmark generator to include the 5 cross-season competitions; the SUMMARY added a "cross-season generalization" five-competition table)
  3. Delivery-layer output: a results deck (a condensed version for the professor), filled in the previously blank top-level README (the entry point to the whole deliverable), and a tech-report research draft (synthesizing the whole project in academic style)
  4. Did an attribution comparison for the fifth strategy (external idea injection) and arrived at an **important honest finding**: under the current architecture, external injection is a "no-op" for the search — the priors were never actually read by the search (all along it was the driver's hand-written seeds doing the guiding). So to truly measure the value of external injection, we must first "wire it up" so the search reads the priors (this is a directional question to decide: wire it up vs. accept this honest negative result)
  5. Added statistical rigor: ran bootstrap confidence intervals on the five competitions — the end-to-end improvement of the whole ladder is statistically significant across all five, but individual single steps (especially the tree-search step) fall within noise in the smaller competitions; honestly recalibrated "the ladder holds in every competition" to "trust the cumulative gain, don't over-claim a single 0.0x% step"

- All deliverables passed number-traceability validation + produced PDFs, and git is clean. In the early morning, having reached the natural boundary of "data ready + safe + high value," Claude switched to lightweight monitoring and did not force through low-value busywork (the rest either needs a decision or carries arm64 package-build risk)

- During the day (with the user back and supervising) **fully integrated the Stage 5 (external injection) empirical results into the 15-competition report**: expanded from only editing the §2.3 "strategies adopted" line to filling in §3.2 experiment summary table + §4 trajectory + §5 performance comparison + §7 reproduction instructions; iterated per feedback throughout (experiment numbering changed to a continuation sequence rather than S5, null does not enter the turning points, added feature counts/raw OOF/execution timestamps, removed the invented "ERA second pillar" wording, aligned §2.3 layout with Preface 5.1/5.2). All 15 competitions passed verify + PDFs regenerated.

- During the day (interactive) added an investigation: assessing **the feasibility of the optional extension "statistical rigor → 10 competitions of S3"** — the conclusion is a "is it worth it" trade-off question, clarify first then decide:
  1. **Root cause**: these 10 S3 competitions were the earliest weekend batch, run on the old pipeline **before instrumentation was added** — at the time there was no plan to do bootstrap, so the intermediate stage2/3 blend OOFs were **computed then discarded, not saved** (only the cross-season batch specifically saved `blend_stage*.npz`, used tree v3 to store the champion's exact weights, and fixed LGBM determinism). The disk only kept the tree-search **member** OOFs (`solo_*.npz`).
  2. **Two paths + time estimates**:
     - **Cheap version (read cache and recombine only, no retraining)**: the CIs naturally match the committed numbers, zero re-baseline/determinism/collision risk; but **coverage is uneven** — stage4 can be exactly rebuilt for all 10 competitions (the weights are hidden in `config.result.weights`), while stage2/3 vary per competition (need to reverse-engineer experiments.json; for some competitions such as s3e20 the stage2 tree has no corresponding node). **~half a day to a day, partial coverage.**
     - **Option A (retrain + re-baseline the whole batch)**: full coverage, but the new OOF ≠ committed → have to change every number (or leave contradictions) + overnight compute + may flip results and break "no competition regressed" + collide with the experiment session. **~1.5–2 days, high risk.**
  3. **Leaning**: the results are largely predictable and the gap is already covered by the honest scope statement in `statistical_rigor.md` → **leaning toward not doing the full set**; if we do supplement, only do the cheap version's **focused version (3→4 tree-search significance, picking competitions that can be cleanly rebuilt)**. **Pending decision.**

- Starting in the evening (user off work, autonomous execution authorized) **autonomously completed the full plan-compliance sweep + gap-filling** (3→2→1, throughout with independent subagents + my item-by-item verification):
  - **Read the 4-page plan itself and audited it clause by clause** (docs/plan_compliance_audit.md): the main line fully meets targets; identified the remaining gaps = the report-validation trio (§5) + the LLM version of idea recombination + MLflow. Finalized **S3 no retraining** (already meets the plan's reproducibility standard; bit-level determinism is beyond scope and CV-only makes it inapplicable).
  - **③ MLflow**: export_to_mlflow.py imports experiments into sqlite (284 runs / 38 competitions) + reproducibility.md (reproducibility layering + structural mapping).
  - **② Report-validation trio (§5)**: rubric built + 3 independent LLM scorings **all 15 competitions passed (19.5/20)**; the consistency check caught **5 genuine report↔code inconsistencies** (s5e10 "no failed node" misstatement, s3e11 weight ranking, s3e3 quick-eval member pool, s3e9 clip, s6e1 wording) all verified + corrected; report-only reproduction of 2 competitions both "partial" (process reproducible, bit-level limited by unrecorded hyperparameters in the early batch).
  - **① LLM-driven recombination mutation**: built the LLM recombination operator (**the first LLM-in-loop mutation**, filling the ERA core gap) + s6e1 pilot test = **redundant honest null** (champion = member-pool NNLS convex optimum bit-for-bit, error correlation 0.995, nearly duplicate); restored the previously mistakenly deleted 5.2.
  - All 15 competitions passed verify, all PDFs regenerated, one commit per unit. commits: 930eab9→dda5acb→fb488ac→b46c314→71338d1→fa2cd56.

---

### to do list

- [x] Restructured the summary report (common content pulled into the preface, summary trimmed, 5c8bda6)
- [x] Beautified each section's major headings (heading hierarchy v3, deep-blue bold + thin underline + left-end accent segment)
- [x] Finished the three cross-season competitions s5e10 / s6e1 / s6e2 (all done, all five cross-season competitions complete)
- [x] Extended the summary comparison table to 15 competitions + added the "cross-season generalization" chapter
- [x] Landed the fifth strategy (external literature injection): built the idea bank ✓ → hooked into tree search ✓ → attribution comparison ✓ (finding: under the current architecture the priors were never read by the search, injection is a no-op, it is the driver's hand-written seeds doing the guiding)
- [x] **Directional question resolved**: the recombination (5.2) LLM version has been built + tested = honest null (structural redundancy, see recombine_findings); both mechanisms of the second pillar (injection + recombination) are built, both empirically show no systematic gain, mechanism explanation attached, no further wiring
- [x] Packaged the Skill: three skills placed under version control + install instructions (commit f063f16; .claude/skills/README.md contains the division-of-labor table / install steps / dependencies / credential injection)
- [x] Delivery-layer deployable wrap-up — adopted a **lightweight alternative** (not Docker): setup.sh one-click build + self-check, REPRODUCE.md consolidating reproduction steps/limitations/trade-offs, README hookup; uv sync uses --inexact so it does not remove torch outside the lock file. Zero arm64 risk. (Docker image itself, after evaluation, not done; rationale in REPRODUCE.md)
- [x] Detailed comparison against the Aygün paper: docs/aygun_comparison.md (+PDF) — item-by-item comparison of method / benchmark / results / rigor, honestly noting that this project is not LLM-code-mutation, the second pillar is a no-op, CV-only, and pointing out that this project's 10 S3 competitions are a subset of ERA's 16, with statistical rigor actually more conservative than ERA's main result
- [x] Statistical-rigor extension to 10 S3 competitions — **finalized as not done**: S3 already meets the plan's reproducibility standard (fixed seed + uv + logging); bit-level determinism is beyond scope, and S3 being entirely CV-only means non-submission makes it inapplicable; retained as reasonable progress + honest record layering (see reproducibility.md / plan_compliance_audit)
- [x] **Full plan-compliance sweep + gap-filling (3→2→1, autonomous 2026-07-08)**: ③MLflow ✅ ②report-validation trio ✅ ①LLM recombination (honest null) ✅; main line + two pillars + report validation all wrapped up, see plan_compliance_audit

## 7/15

- Had Claude run the **NVIDIA official Kaggle agent** (the `nvidia-kaggle-skill` plugin) in a workspace kept isolated from our own agent, and reproduce the strongest public kernel on all **15 competitions** (the 10 Season-3 benchmark + the 5 cross-season ones), then compare its scores head-to-head against our own from-scratch agent on the same data and the same CV.

- The point of the comparison: our agent **builds** a solution from scratch (GBDT + tree search + prior injection); the NVIDIA agent only **reproduces the best public kernel** — it has no original-solution engine.

- **Core finding (the recency effect):**
  1. On **mature competitions** (Season 3, 2023, lots of public solutions): NVIDIA is competitive — clearly wins 4, our agent 3, 3 ties.
  2. On the **newer cross-season competitions** (S4–S6, 2024–26, thinner public corpus): **NVIDIA's edge collapses — it clearly wins only 1 of 5 (s4e1), and our agent is level-or-ahead on the other 4** (s4e11 .9402 vs .9399, s5e10 .05597 vs .05609, s6e1 R2 .7872 vs .7863, s6e2 .95553 vs .9554).
  3. So: the newer and less picked-over a competition is, the less "copy the public best" buys you — direct evidence that the two agents are complementary, not competing.

- Made Claude **stress-test two doubtful results instead of trusting them:**
  1. **s3e3**: I suspected NVIDIA's win came from a target leak (its WOE encoder was fit before the CV split). Had Claude re-run it fold-safe — the blend stayed at 0.8758, so it's a **genuine win, not a leak** (corrected our earlier flag).
  2. **s3e19**: the two numbers had been measured on different CVs. Had Claude put both agents on one identical **hard TimeSeriesSplit** ruler → our agent 10.02 beats NVIDIA's disaggregation 13.72, which also **experimentally proves our agent was right to reject ratio-disaggregation** on this competition.

- Produced a per-competition comparison webpage (scores + a method-by-method comparison table for both the Season-3 and the cross-season batches) plus reproducible scripts under `~/ai_agents/nvidia-kaggle-runs/` (each competition: reproduce.py + result.json + submission.csv).

- Honest caveats recorded: a few cross-season NVIDIA numbers are only the **GBM core** of the public kernel (s6e1's public best is a neural net we didn't reproduce; s5e10's external dataset 403'd so we substituted fold-safe target encoding), so NVIDIA could be a hair higher on those — but all still within tie range.

---

### to do list

- [x] Ran the NVIDIA official Kaggle agent on all 15 competitions and did the head-to-head comparison vs our own agent (result: NVIDIA competitive on mature S3, edge collapses on newer cross-season — wins only 1 of 5)
- [x] Stress-tested s3e3 (de-leaked → genuine win) and s3e19 (unified hard CV → our agent wins 10.02 vs 13.72)
- [x] Comparison webpage + reproducible scripts delivered
- [ ] (optional) Reproduce the neural-net version of the cross-season competitions where we only did the GBM core (s6e1, s5e10) to make the NVIDIA side fully complete
- [ ] (optional) Write the NVIDIA-vs-our-agent comparison up as a formal report section
