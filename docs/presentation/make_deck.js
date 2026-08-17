const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5

const NAVY = "0F2A4A", INK = "1C1E21", MUT = "5A6472", ICE = "CADCFC", GOLD = "E2B13C",
      BLUE = "2B6CB0", LT = "F4F7FB", WHITE = "FFFFFF", PALE = "A8BDD4";
const W = 13.33, H = 7.5;

function head(slide, num, title) {
  slide.background = { color: WHITE };
  slide.addText(title, { x: 0.55, y: 0.28, w: 11.5, h: 0.75, fontFace: "Cambria",
    fontSize: 30, bold: true, color: NAVY, margin: 0 });
  slide.addText(num, { x: 12.35, y: 0.28, w: 0.6, h: 0.75, fontFace: "Calibri",
    fontSize: 16, color: PALE, align: "right", margin: 0 });
}

// ---------- 1 · Title ----------
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText("AI Agents for Data Visualization\nand Kaggle Competitions", {
    x: 0.8, y: 1.9, w: 11.7, h: 2.2, fontFace: "Cambria", fontSize: 44, bold: true, color: WHITE, margin: 0 });
  s.addText("An end-to-end LLM agent, benchmarked against two frozen yardsticks on 20 Kaggle competitions", {
    x: 0.8, y: 4.15, w: 11.0, h: 0.7, fontFace: "Calibri", fontSize: 20, color: ICE, margin: 0 });
  s.addText("Wei-Hao Huang  ·  Academia Sinica ISS Summer Internship 2026  ·  Report v4 (2026-08-17)", {
    x: 0.8, y: 6.3, w: 11.0, h: 0.5, fontFace: "Calibri", fontSize: 14, color: PALE, margin: 0 });
  s.addNotes("Four minutes: what I built, how it works, what it scored, what surprised me, and what I would tell the next person. (10s)");
}

// ---------- 2 · Motivation ----------
{
  const s = pres.addSlide();
  head(s, "2", "Motivation — why these tasks, why an agent");
  s.addText("The challenge", { x: 0.55, y: 1.2, w: 5.9, h: 0.4, fontFace: "Calibri", fontSize: 18, bold: true, color: NAVY, margin: 0 });
  s.addText([
    { text: "Kaggle demands the full expert loop — validation design, features, model selection, ensembling.", options: { bullet: true, breakLine: true } },
    { text: "Top entries differ at the 4th–5th decimal of the metric; intuition cannot rank them.", options: { bullet: true, breakLine: true } },
    { text: "Local cross-validation misleads: it contradicts the real leaderboard in 9 of 13 competitions we measured.", options: { bullet: true, breakLine: true } },
    { text: "EDA and diagnostic visualization are essential but repetitive for every new dataset.", options: { bullet: true } },
  ], { x: 0.55, y: 1.65, w: 5.9, h: 3.6, fontFace: "Calibri", fontSize: 15, color: INK, paraSpaceAfter: 10, margin: 0 });

  s.addText("Why an AI agent fits", { x: 6.9, y: 1.2, w: 5.9, h: 0.4, fontFace: "Calibri", fontSize: 18, bold: true, color: NAVY, margin: 0 });
  s.addText([
    { text: "Tireless systematic search: 22–80 candidate configurations per competition, all logged.", options: { bullet: true, breakLine: true } },
    { text: "Holds one disciplined protocol across 20 competitions — humans drift, agents don't.", options: { bullet: true, breakLine: true } },
    { text: "Emits EDA figures and per-competition reports automatically; every claim traces to an artifact.", options: { bullet: true } },
  ], { x: 6.9, y: 1.65, w: 5.9, h: 3.0, fontFace: "Calibri", fontSize: 15, color: INK, paraSpaceAfter: 10, margin: 0 });

  s.addShape(pres.ShapeType.roundRect, { x: 6.9, y: 5.0, w: 5.9, h: 1.7, fill: { color: LT }, line: { color: PALE, width: 1 }, rectRadius: 0.08 });
  s.addText([
    { text: "9 / 13", options: { fontSize: 40, bold: true, color: BLUE, breakLine: true } },
    { text: "competitions where local CV and the real leaderboard disagree — the core reason honest measurement needs apparatus", options: { fontSize: 12.5, color: MUT } },
  ], { x: 7.15, y: 5.15, w: 5.4, h: 1.45, fontFace: "Calibri", align: "left", margin: 0 });
  s.addNotes("Kaggle is the full expert loop, and the margins are at the fourth decimal. The obvious compass — your own cross-validation — points the wrong way in 9 of 13 cases we measured. An agent fits because it is tireless, holds one protocol across 20 competitions, and documents everything it does. (35s)");
}

// ---------- 3 · Workflow ----------
{
  const s = pres.addSlide();
  head(s, "3", "Workflow — an LLM reasons at every stage");
  s.addImage({ path: "fig1_architecture.png", x: 0.55, y: 1.35, w: 7.6, h: 3.84 });
  s.addText("Six-stage pipeline: problem dossier → EDA → feature engineering → modeling → evaluation → submission.", {
    x: 0.55, y: 5.35, w: 7.6, h: 0.6, fontFace: "Calibri", fontSize: 12.5, italic: true, color: MUT, margin: 0 });
  s.addText([
    { text: "LLM decides at every stage; Auto-ML tools (LightGBM / XGBoost / CatBoost / Optuna) do the heavy lifting via the shell.", options: { bullet: true, breakLine: true } },
    { text: "Experience library: cross-competition priors, every entry evidence-backed, written back after each run.", options: { bullet: true, breakLine: true } },
    { text: "Tree search over candidate configurations is the optimisation loop — ensembles are first-class nodes with OOF caching.", options: { bullet: true, breakLine: true } },
    { text: "v5 layer: a problem dossier classifies the task and injects external data upstream, before any modeling.", options: { bullet: true } },
  ], { x: 8.5, y: 1.45, w: 4.3, h: 5.0, fontFace: "Calibri", fontSize: 14, color: INK, paraSpaceAfter: 12, margin: 0 });
  s.addNotes("The workflow: data ingestion, a problem dossier, EDA, features, modeling, evaluation, submission. The LLM reasons at every stage and drives Auto-ML tools through the shell. Two things make it more than a script: an evidence-backed experience library shared across competitions, and a tree search where ensembles are first-class nodes. (40s)");
}

// ---------- 4 · Results: Kaggle ----------
{
  const s = pres.addSlide();
  head(s, "4", "Results — three agents, real leaderboards");
  s.addChart(pres.ChartType.bar, [{
    name: "Decidable-duel win rate",
    labels: ["my-agent", "NVIDIA", "AIDE"],
    values: [57, 48, 44],
  }], {
    x: 0.55, y: 1.35, w: 5.7, h: 3.1, barDir: "bar",
    chartColors: [BLUE, PALE, PALE],
    showTitle: true, title: "Decidable-duel win rate, 20 comps, pre-v5 baseline (%)",
    titleFontSize: 13, titleColor: NAVY, titleFontFace: "Calibri",
    showValue: true, dataLabelPosition: "outEnd", dataLabelColor: INK, dataLabelFontSize: 12, dataLabelFontFace: "Calibri",
    showLegend: false, catAxisLabelColor: INK, catAxisLabelFontSize: 12, catAxisLabelFontFace: "Calibri",
    valAxisLabelColor: MUT, valAxisLabelFontSize: 10, valAxisMaxVal: 70, valAxisMinVal: 0,
    valGridLine: { color: "E4E9F0", size: 0.5 }, catGridLine: { style: "none" },
    chartColorsOpacity: 100,
  });
  s.addText("27% of all duels are undecidable at the measurable error scale; verdicts come from a paired test built on Kaggle's own public/private split. Median percentile rank: my-agent 72.3, NVIDIA 68.0, AIDE 68.7. Standings are the pre-v5 baseline lanes; the Stage-0.5 re-run replaces them once scored.", {
    x: 0.55, y: 4.6, w: 5.7, h: 1.1, fontFace: "Calibri", fontSize: 12.5, color: MUT, margin: 0 });

  s.addText("Mid-complexity three-way — my-agent sweeps all three", { x: 6.9, y: 1.35, w: 5.9, h: 0.4, fontFace: "Calibri", fontSize: 16, bold: true, color: NAVY, margin: 0 });
  s.addTable([
    [{ text: "Competition", options: { bold: true, color: NAVY } }, { text: "Metric", options: { bold: true, color: NAVY } },
     { text: "my-agent", options: { bold: true, color: NAVY } }, { text: "NVIDIA", options: { bold: true, color: NAVY } }, { text: "AIDE", options: { bold: true, color: NAVY } }],
    ["us-patent (NLP)", "Pearson r ↑", { text: "0.86658", options: { bold: true, color: BLUE } }, "0.86160", "0.65821"],
    ["ventilator (time series)", "MAE ↓", { text: "0.16525", options: { bold: true, color: BLUE } }, "0.40093", "0.34591"],
    ["SIIM-ISIC (imaging)", "AUC ↑", { text: "0.93529", options: { bold: true, color: BLUE } }, "0.89822", "0.88579"],
  ], { x: 6.9, y: 1.85, w: 5.9, colW: [1.9, 1.3, 1.0, 0.9, 0.9], fontFace: "Calibri", fontSize: 11.5, color: INK,
       border: { type: "solid", color: "DFE5EC", pt: 0.5 }, fill: { color: WHITE }, rowH: 0.34, valign: "middle", margin: 0.04 });
  s.addShape(pres.ShapeType.roundRect, { x: 6.9, y: 4.15, w: 5.9, h: 2.35, fill: { color: LT }, line: { color: PALE, width: 1 }, rectRadius: 0.08 });
  s.addText([
    { text: "Final-architecture re-run — in progress", options: { fontSize: 15, bold: true, color: NAVY, breakLine: true } },
    { text: "All 20 competitions re-run under one frozen architecture, in credential-free sandboxes with per-lane transcript audits. 16/20 lanes run (one pending audit review), 1 running, 3 queued.", options: { fontSize: 12.5, color: INK, breakLine: true } },
    { text: "Leaderboard scores: TBD until the credentialed scoring step.", options: { fontSize: 12.5, bold: true, color: "8A6D1A" } },
  ], { x: 7.15, y: 4.3, w: 5.4, h: 2.05, fontFace: "Calibri", paraSpaceAfter: 8, margin: 0 });
  s.addNotes("Everything is scored on the real leaderboard, and a paired significance test decides which gaps are real. My agent wins 57 percent of decidable duels; 27 percent cannot be decided at all. On the three harder competitions outside tabular data, it takes the best score on all three. Right now all 20 competitions are re-running under the frozen final architecture with transcript audits; those leaderboard numbers are still TBD. (45s)");
}

// ---------- 5 · Results: visualization output ----------
{
  const s = pres.addSlide();
  head(s, "5", "Results — the agent's visualization output");
  s.addImage({ path: "eda_target.png", x: 0.55, y: 1.45, w: 5.95, h: 4.25 });
  s.addImage({ path: "eda_temporal.png", x: 6.85, y: 1.45, w: 5.95, h: 4.25 });
  s.addText("Agent-generated EDA, produced automatically before modeling (energy-prosumer example): target distributions by segment (left) and temporal consumption/production patterns (right). The same pipeline emits 22 per-competition ML-spec reports and its own architecture flowcharts.", {
    x: 0.55, y: 5.9, w: 12.25, h: 0.9, fontFace: "Calibri", fontSize: 13, color: MUT, margin: 0 });
  s.addNotes("The visualization half: before any modeling, the agent draws its own diagnostics — distributions, temporal patterns, drift checks — and it writes 22 per-competition specification reports. Every figure you see in the report and the poster was produced by the pipeline itself. (25s)");
}

// ---------- 6 · Advantages & difficulties ----------
{
  const s = pres.addSlide();
  head(s, "6", "What I expected vs. what I experienced");
  const rows = [
    ["The agent would quickly beat most humans.", "Median percentile 72.3 — above average; wins concentrate where no mature public solutions exist."],
    ["One run and one score decide a winner.", "Score noise + run variance leave 27% of duels undecidable; a paired error model became mandatory."],
    ["The agent follows the rules by default.", "A transcript audit caught a lane reading its own competition's recorded findings — the clean re-run scored worse, so the advantage was real."],
    ["Fairness is mostly a mindset.", "It is infrastructure: isolated data roots, credential-free sandboxes, per-lane audits — expensive, but without them numbers aren't comparable."],
  ];
  s.addText("Expected", { x: 0.55, y: 1.2, w: 5.6, h: 0.4, fontFace: "Calibri", fontSize: 17, bold: true, color: MUT, margin: 0 });
  s.addText("Experienced", { x: 6.6, y: 1.2, w: 6.1, h: 0.4, fontFace: "Calibri", fontSize: 17, bold: true, color: NAVY, margin: 0 });
  let y = 1.7;
  for (const [a, b] of rows) {
    s.addShape(pres.ShapeType.roundRect, { x: 0.55, y, w: 5.6, h: 1.22, fill: { color: LT }, line: { color: "E4E9F0", width: 0.75 }, rectRadius: 0.06 });
    s.addText(a, { x: 0.75, y: y + 0.08, w: 5.2, h: 1.06, fontFace: "Calibri", fontSize: 13, color: MUT, italic: true, margin: 0, valign: "middle" });
    s.addShape(pres.ShapeType.roundRect, { x: 6.6, y, w: 6.15, h: 1.22, fill: { color: WHITE }, line: { color: PALE, width: 1 }, rectRadius: 0.06 });
    s.addText(b, { x: 6.8, y: y + 0.08, w: 5.75, h: 1.06, fontFace: "Calibri", fontSize: 13, color: INK, margin: 0, valign: "middle" });
    y += 1.38;
  }
  s.addNotes("What surprised me. I expected the agent to beat most humans quickly — it lands above average, and its edge concentrates where no public solutions exist. I expected one score to decide a winner — a quarter of duels are actually undecidable. And I expected rule-following by default — the transcript audit proved otherwise once, and proved itself once by a false positive. (40s)");
}

// ---------- 7 · Discussion ----------
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText("Advice — if you use an AI agent for this", { x: 0.8, y: 0.55, w: 11.7, h: 0.8, fontFace: "Cambria", fontSize: 30, bold: true, color: WHITE, margin: 0 });
  s.addText([
    { text: "State the error model with every claim. Without an error scale a ranking is noise: 27% of our duels are undecidable, and the winner flips when the evaluation set changes.", options: { bullet: true, breakLine: true } },
    { text: "Never trust local CV for between-agent claims — score on the real leaderboard.", options: { bullet: true, breakLine: true } },
    { text: "Freeze your yardsticks. Improving a baseline mid-study turns the benchmark into an optimisation of the baseline.", options: { bullet: true, breakLine: true } },
    { text: "Isolate mechanically, then audit the transcript — reading what the agent actually did catches the leaks you didn't anticipate, in both directions.", options: { bullet: true, breakLine: true } },
    { text: "Put knowledge injection upstream: at the blend stage it was worth ~10⁻⁵; moved into problem identification it lifted the canonical failure case from last to first.", options: { bullet: true } },
  ], { x: 0.8, y: 1.7, w: 11.7, h: 3.9, fontFace: "Calibri", fontSize: 16.5, color: ICE, paraSpaceAfter: 14, margin: 0 });
  s.addText("Report v4 · Engineering Log · Case Studies · 22 ML-spec reports  —  github.com/930731haohao-afk/kaggle-agent", {
    x: 0.8, y: 6.45, w: 11.7, h: 0.5, fontFace: "Calibri", fontSize: 13, color: PALE, margin: 0 });
  s.addNotes("If you do this yourself: state an error model with every claim, never trust local CV between agents, freeze your yardsticks, isolate mechanically and then read the transcript, and put knowledge injection upstream. Everything is in report v4 and the repository. Thank you. (25s)");
}

pres.writeFile({ fileName: "slides_4min.pptx" }).then(() => console.log("written"));
