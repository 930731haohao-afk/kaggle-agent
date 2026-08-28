// ============================================================================
// An End-to-End AI Agent for Kaggle Competitions --- 4-minute talk.
// PPTX generator, kept in lockstep with slides_4min.tex (beamer) and with the
// poster: title plus five body slides in the poster's order, same on-screen
// text, same numbers, same speaker notes, then a questions slide.
//
// Budgeted seconds live at the end of every note; title plus the five body
// slides sum to under 240, and questions adds about a minute.
// ============================================================================
const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in -- set before any addSlide()

// ---- Palette (identical RGB to the poster and the beamer preamble) ---------
const NAVY  = "003C64"; // navy      RGB(0,60,100) -- data marks only
const INK   = "111111"; // ink       RGB(17,17,17)
const MUT   = "5A5F66"; // mut       RGB(90,95,102)
const RULE  = "C8CDD2"; // ruleline  RGB(200,205,210)
const OTHER = "B0BAC4"; // barother  RGB(176,186,196)
const TRACK = "EEF0F3"; // bartrack  RGB(238,240,243)
const WHITE = "FFFFFF";

const F = "Arial"; // matches the deck's \usepackage{helvet}

// ---- Slide title: black ink, bold, flush left, no page number --------------
function head(slide, title) {
  slide.background = { color: WHITE };
  slide.addText(title, {
    x: 0.72, y: 0.28, w: 12.0, h: 0.60,
    fontFace: F, fontSize: 26, bold: true, color: INK, margin: 0, valign: "middle",
  });
}

// ---- List item from styled runs --------------------------------------------
// The beamer deck marks its list items with a middot rather than an itemize,
// and pptxgenjs starts a new paragraph at every run carrying bullet:true --
// which would split a mixed-style item into one bullet per run. So the marker
// is part of the first run's text, exactly as in the .tex.
// ---- Sub-heading on its own line, body indented underneath (mirrors \ritem)
function ritem(slide, y, headText, bodyText) {
  slide.addText([
    { text: "\u2022   ", options: { bold: true } },
    { text: headText, options: { bold: true } },
  ], {
    x: 0.95, y: y, w: 11.6, h: 0.32,
    fontFace: F, fontSize: 16, color: INK, margin: 0, valign: "middle",
  });
  slide.addText(bodyText, {
    x: 1.35, y: y + 0.34, w: 11.2, h: 0.80,
    fontFace: F, fontSize: 16, color: INK, margin: 0,
  });
}

// ---- Second level: en-dash marker, bold lead, body indented below ---------
function sitem(slide, y, lead, body) {
  const leadRuns = Array.isArray(lead)
    ? lead.map((r) => ({ ...r, options: { bold: true, ...r.options } }))
    : [{ text: lead, options: { bold: true } }];
  slide.addText([
    { text: "\u2013   " },
    ...leadRuns,
  ], {
    x: 1.35, y: y, w: 11.2, h: 0.32,
    fontFace: F, fontSize: 16, color: INK, margin: 0, valign: "middle",
  });
  slide.addText(body, {
    x: 1.80, y: y + 0.32, w: 10.7, h: 0.60,
    fontFace: F, fontSize: 16, color: INK, margin: 0,
  });
}

function item(runs) {
  const out = runs.map((r) => ({ text: r.text, options: { ...(r.options || {}) } }));
  return [{ text: "\u2022   ", options: { bold: true } }, ...out];
}



// ---- Table 1: ordinary competitions, same rows and same bold marking as the
// poster's table. best: 0 = Sinica, 1 = NVIDIA, 2 = AIDE.
const SCORES = [
  ["s3e1", "RMSE \u2193", "0.55275", "0.55550", "0.55552", 0],
  ["s3e3", "AUC \u2191", "0.87157", "0.89537", "0.86863", 1],
  ["s3e5", "QWK \u2191", "0.57223", "0.56974", "0.58138", 2],
  ["s3e7", "AUC \u2191", "0.90441", "0.90458", "0.90109", 1],
  ["s3e9", "RMSE \u2193", "12.3319", "12.2291", "12.3067", 1],
  ["s3e11", "RMSLE \u2193", "0.29368", "0.29624", "0.29445", 0],
  ["s3e14", "MAE \u2193", "333.186", "330.716", "332.316", 1],
  ["s3e16", "MAE \u2193", "1.33822", "1.34001", "1.34224", 0],
  ["s3e19", "SMAPE \u2193", "49.4088", "48.2656", "48.3413", 1],
  ["s4e11", "Acc. \u2191", "0.94107", "0.94076", "0.94056", 0],
  ["s5e1", "MAPE \u2193", "0.08401", "0.14594", "0.17917", 0],
  ["s5e10", "RMSE \u2193", "0.05576", "0.05588", "0.05579", 0],
  ["s6e1", "Error \u2193", "8.71516", "8.73239", "8.72191", 0],
  ["s6e2", "AUC \u2191", "0.95505", "0.95508", "0.95516", 2],
  ["afsis", "MCRMSE \u2193", "0.48917", "0.69855", "0.49861", 0],
  ["cat-in-the-dat", "AUC \u2191", "0.80243", "0.77084", "0.80217", 0],
  ["conway", "MAE \u2193", "0.10414", "0.11189", "0.11065", 0],
  ["tps-aug-2022", "AUC \u2191", "0.58930", "0.58730", "0.59106", 2],
  ["tps-jan-2022", "SMAPE \u2193", "6.29895", "4.63651", "10.6664", 1],
  ["tps-sep-2022", "SMAPE \u2193", "28.4961", "5.48351", "9.72095", 1],
];

// ================================================================== TITLE ---
{
  const s = pres.addSlide();
  s.background = { color: WHITE };

  s.addText("An End-to-End AI Agent for Kaggle Competitions", {
    x: 0.90, y: 2.35, w: 11.5, h: 1.90,
    fontFace: F, fontSize: 40, bold: true, color: INK, margin: 0,
    lineSpacingMultiple: 1.1, valign: "middle",
  });
  s.addText("Wei-Hao Huang, National Chengchi University", {
    x: 0.90, y: 5.80, w: 11.8, h: 0.40,
    fontFace: F, fontSize: 14, color: INK, margin: 0, valign: "middle",
  });
  s.addText(
    "PI: Dr. Tso-Jung Yen, Institute of Statistical Science, Academia Sinica", {
      x: 0.90, y: 6.25, w: 11.8, h: 0.40,
      fontFace: F, fontSize: 13, color: MUT, margin: 0, valign: "middle",
    });

  s.addNotes(
    "Sinica is an end-to-end agent for Kaggle competitions: it takes a raw " +
    "dataset and returns a submission, with no person in the loop. I " +
    "benchmarked it against two frozen reference agents on twenty-three " +
    "competitions. (18s)"
  );
}

// ============================================================= 1 MOTIVATION --
{
  const s = pres.addSlide();
  head(s, "1.Motivation");

  const sub = {
    x: 0.72, w: 11.9, h: 0.32,
    fontFace: F, fontSize: 17, bold: true, color: INK, margin: 0, valign: "middle",
  };
  s.addText("\u2022   Challenges of Kaggle competitions", { ...sub, y: 1.05 });
  sitem(s, 1.48, "A competition requires the expert loop",
    "This includes validation design, feature extraction, model selection, and " +
    "ensemble building. The top entries separate by a very small margin.");
  sitem(s, 2.38, "Data preparation is repetitive",
    "The same cleaning, split design and drift are done on every dataset.");

  s.addText("\u2022   Why an AI agent", { ...sub, y: 3.20 });
  sitem(s, 3.63, [
    { text: "A competition is a " },
    { text: "scorable", options: { italic: true } },
    { text: " task" },
  ],
    "Every competitor delivers a number, so it can be optimized automatically.");
  sitem(s, 4.53, "A competition requires tireless, systematic search",
    "Every experiment is recorded and should be reproducible.");
  sitem(s, 5.28, "Recent success in agents for Kaggle competitions",
    "Notable cases include tree search over code [1], and tree search fed " +
    "injected research ideas [2].");

  s.addNotes(
    "Competitions demand the whole expert loop, and the top of the " +
    "leaderboard separates at the fourth or fifth decimal, so tuning skill " +
    "distinguishes nobody. Before any of that, every dataset needs the same " +
    "repetitive preparation. What an agent brings is tireless, logged search: " +
    "between thirty-seven and two hundred and twenty-four candidate " +
    "configurations per competition, and the same protocol run twenty times " +
    "without drifting. (32s)"
  );
}

// ============================================================== 2 WORKFLOW ---
{
  const s = pres.addSlide();
  head(s, "2.Workflow");

  s.addText("From raw data to a scored submission", {
    x: 0.72, y: 1.05, w: 11.9, h: 0.34,
    fontFace: F, fontSize: 19, bold: true, color: INK, margin: 0, valign: "middle",
  });

  // Landscape architecture figure, rasterized from fig1_arch_slide.pdf at
  // 300 dpi; native 561.11 x 246.99 pt, aspect 2.272. Width follows from the
  // height so the picture is not squeezed; x centers it on the 13.33in slide.
  s.addImage({ path: "fig1_arch_slide_hi-1.png", x: 0.635, y: 1.55, w: 12.06, h: 5.31 });

  s.addText([
    { text: "The LLM reasons and decides at every stage; the problem dossier " +
            "classifies the task and injects external data " },
    { text: "upstream", options: { italic: true } },
    { text: ", before any modeling. EDA then tests the dossier's hypotheses, " +
            "feature engineering and modeling follow, and evaluation returns " +
            "the score the tree search uses to pick the next candidate." },
  ], {
    x: 0.72, y: 6.95, w: 11.9, h: 0.50,
    fontFace: F, fontSize: 13, color: MUT, margin: 0,
  });

  s.addNotes(
    "The pipeline reads top to bottom. An LLM reasons and decides at every " +
    "stage, gradient boosting does the fitting, and a tree search over " +
    "complete candidate solutions is the optimization loop. The one design " +
    "decision that mattered is where outside knowledge enters. My first " +
    "version put it downstream, at the blend — measured, it was worth ten to " +
    "the minus five. So it moved upstream, into a dossier stage that " +
    "classifies the task before any modeling. The lanes hold no Kaggle " +
    "credentials, so nothing inside a run can touch a leaderboard. (40s)"
  );
}

// =============================================================== 3 RESULTS ---
{
  const s = pres.addSlide();
  head(s, "3.Results");

  // -------- left half: the private-leaderboard table --------
  s.addText("Table 1: ordinary competitions", {
    x: 0.72, y: 1.05, w: 7.40, h: 0.32,
    fontFace: F, fontSize: 14, bold: true, color: INK, margin: 0, valign: "middle",
  });

  const cell = { fontFace: F, fontSize: 11, color: INK, margin: [1, 3, 1, 3] };
  const rows = [[
    { text: "Competition", options: { ...cell, bold: true } },
    { text: "Metric", options: { ...cell, bold: true } },
    { text: "Sinica", options: { ...cell, bold: true, align: "right" } },
    { text: "NVIDIA", options: { ...cell, bold: true, align: "right" } },
    { text: "AIDE", options: { ...cell, bold: true, align: "right" } },
  ]];
  SCORES.forEach((r) => {
    rows.push([
      { text: r[0], options: { ...cell } },
      { text: r[1], options: { ...cell } },
      { text: r[2], options: { ...cell, align: "right", bold: r[5] === 0 } },
      { text: r[3], options: { ...cell, align: "right", bold: r[5] === 1 } },
      { text: r[4], options: { ...cell, align: "right", bold: r[5] === 2 } },
    ]);
  });
  rows.push([
    { text: "Best private score", options: { ...cell, bold: true } },
    { text: "", options: { ...cell } },
    { text: "10 / 20", options: { ...cell, align: "right", bold: true } },
    { text: "7 / 20", options: { ...cell, align: "right" } },
    { text: "3 / 20", options: { ...cell, align: "right" } },
  ]);
  s.addTable(rows, {
    x: 0.72, y: 1.40, w: 7.40, colW: [1.95, 1.55, 1.32, 1.32, 1.26],
    rowH: 0.245, border: { type: "none" }, autoPage: false,
  });
  s.addText([
    { text: "Bold", options: { bold: true } },
    { text: " = best of the three; reference agents AIDE [1], NVIDIA [3]." },
  ], {
    x: 0.72, y: 7.03, w: 7.40, h: 0.30,
    fontFace: F, fontSize: 11, color: MUT, margin: 0, valign: "middle",
  });

  // -------- right half: Table 2, the three special competitions --------
  s.addText("Table 2: special competitions", {
    x: 8.35, y: 1.05, w: 4.55, h: 0.32,
    fontFace: F, fontSize: 14, bold: true, color: INK, margin: 0, valign: "middle",
  });

  const mcell = { fontFace: F, fontSize: 10, color: INK, margin: [1, 3, 1, 3] };
  const mrows = [[
    { text: "Competition", options: { ...mcell, bold: true } },
    { text: "Metric", options: { ...mcell, bold: true } },
    { text: "Sinica", options: { ...mcell, bold: true, align: "right" } },
    { text: "NVIDIA", options: { ...mcell, bold: true, align: "right" } },
    { text: "AIDE", options: { ...mcell, bold: true, align: "right" } },
  ]];
  [
    ["us-patent", "Pearson r \u2191", "0.86658", "0.86160", "0.65821"],
    ["ventilator", "MAE \u2193", "0.16525", "0.40093", "0.34591"],
    ["SIIM-ISIC", "AUC \u2191", "0.93529", "0.89822", "0.88579"],
  ].forEach((r) => {
    mrows.push([
      { text: r[0], options: { ...mcell } },
      { text: r[1], options: { ...mcell } },
      { text: r[2], options: { ...mcell, align: "right", bold: true } },
      { text: r[3], options: { ...mcell, align: "right" } },
      { text: r[4], options: { ...mcell, align: "right" } },
    ]);
  });
  s.addTable(mrows, {
    x: 8.35, y: 1.40, w: 4.55, colW: [1.30, 1.02, 0.78, 0.78, 0.67],
    rowH: 0.26, border: { type: "none" }, autoPage: false,
  });
  s.addText("Graded by MLE-bench [4].", {
    x: 8.35, y: 2.62, w: 4.55, h: 0.30,
    fontFace: F, fontSize: 11, color: MUT, margin: 0, valign: "middle",
  });

  s.addNotes(
    "All 23 competitions are scored: the 20 ordinary ones on the real " +
    "private leaderboard, the 3 special ones offline with MLE-bench. Against " +
    "the two reference agents, Sinica takes the best score on 13 of the 23 " +
    "and leads the pairwise duels judged against the split noise — sixteen " +
    "wins to eight of its decided duels, sixty-six point seven percent, " +
    "against forty-two point three for each reference agent. (45s)"
  );
}

// ==================== 4 DISADVANTAGES OF THE AGENT-BASED APPROACH ---
{
  const s = pres.addSlide();
  head(s, "4.Disadvantages of the agent-based approach");

  ritem(s, 1.10, "Strong at coding, weak at judging",
    "The agent efficiently works through whatever problem it is given. " +
    "However, it does not ask whether a change is worth making. Therefore, a " +
    "poor design choice shows up and fixing the bugs caused by it is costly.");
  ritem(s, 2.80, "The main question gets buried in detail",
    "AI-written reports pile up technical detail and drift away from what the " +
    "reader wants to know.");
  ritem(s, 4.05, "AI writing habits are difficult to remove",
    "Habits such as redundant headings and invented terms come back in every " +
    "draft; verifying and modifying them by hand is exhausting.");

  s.addNotes(
    "Where the approach falls short. The agent codes efficiently, but it has " +
    "no design judgment: it never asks whether a change is worth making, so a " +
    "poor design choice surfaces late and the bugs it causes are costly to " +
    "fix. Heavy AI use also pulls a write-up toward technical detail and away " +
    "from the question the reader came for. And its writing habits — " +
    "redundant headings, invented terms — come back in every draft; verifying " +
    "and fixing them by hand is exhausting. (42s)"
  );
}

// ================= 5 PRACTICAL ADVICE FOR BUILDING AI AGENTS ---
{
  const s = pres.addSlide();
  head(s, "5.Practical advice for building AI agents for data science");

  ritem(s, 1.08, "Justify the design before building",
    "Settle on a solid design before the runs start; a redesign costs not " +
    "only debugging but also a full re-run of every earlier competition.");
  ritem(s, 2.28, "Put external knowledge before modeling",
    "Injecting external data at the final blending step changed scores by " +
    "almost nothing. Therefore, we moved the injection to before modeling, " +
    "and scores improved clearly.");
  ritem(s, 3.63, "Build isolation into the environment",
    "Set up the sandbox to prevent any access to the leaderboard or another " +
    "run's files.");
  ritem(s, 4.73, "Iteration can only target the local validation score",
    "The local validation score is the only target the agent can iterate on. " +
    "As a result, the local validation score can look outstanding while the " +
    "real leaderboard score is mediocre.");

  s.addNotes(
    "Four pieces of advice for anyone building one of these. Justify the " +
    "design before building — a redesign costs debugging plus a full re-run " +
    "of every earlier competition. Put external knowledge before modeling — " +
    "at the blending stage it bought nothing; before modeling it clearly " +
    "helped. Build isolation into the environment, so a run simply cannot " +
    "reach the leaderboard or another run's files. And remember iteration " +
    "can only target the local validation score, which can look outstanding " +
    "while the real leaderboard score is mediocre. (43s)"
  );
}

// ============================================================= REFERENCES ---
{
  const s = pres.addSlide();
  head(s, "6.References");

  s.addText([
    { text: "[1]  Z. Jiang, D. Schmidt, D. Srikanth, D. Xu, I. Kaplan, D. Jacenko, and Y. Wu, AIDE: AI-driven exploration in the space of code, " },
    { text: "arXiv:2502.13138", options: { italic: true } },
    { text: ".\n" },
    { text: "[2]  E. Ayg\u00fcn " },
    { text: "et al", options: { italic: true } },
    { text: "., An AI system to help scientists write expert-level empirical software, " },
    { text: "Nature", options: { italic: true } },
    { text: " " },
    { text: "654", options: { bold: true } },
    { text: ", 909 (2026).\n" },
    { text: "[3]  NVIDIA Corporation, computer code NVIDIA Kaggle Plugin, Santa Clara, CA, 2026, https://github.com/NVIDIA/nvidia-kaggle.\n" },
    { text: "[4]  J. S. Chan " },
    { text: "et al", options: { italic: true } },
    { text: "., MLE-bench: Evaluating machine learning agents on machine learning engineering, in " },
    { text: "Proceedings of the 13th International Conference on Learning Representations", options: { italic: true } },
    { text: ", 2025: https://openreview.net/forum?id=6s5uXNWGIh." },
  ], {
    x: 0.72, y: 1.30, w: 11.9, h: 3.20,
    fontFace: F, fontSize: 13, color: INK, margin: 0, valign: "top", lineSpacingMultiple: 1.35,
  });

  s.addNotes(
    "Four systems this work sits next to: AIDE and the Nature paper for the " +
    "search-based agents, the NVIDIA plugin as the reproduction baseline, and " +
    "MLE-bench for how the three harder competitions are graded."
  );
}

// ============================================================== QUESTIONS ---
{
  const s = pres.addSlide();
  s.background = { color: WHITE };

  s.addText("Questions", {
    x: 0.90, y: 3.10, w: 11.5, h: 1.00,
    fontFace: F, fontSize: 40, bold: true, color: INK, margin: 0,
    align: "center", valign: "middle",
  });


  s.addNotes(
    "Open for questions. The three I expect: how a duel is decided against the " +
    "noise floor, how the tree search spends its budget, and how the lanes are " +
    "isolated and audited. (about 60s of questions)"
  );
}

pres.writeFile({ fileName: "slides_4min.pptx" })
  .then((f) => console.log("wrote", f));
