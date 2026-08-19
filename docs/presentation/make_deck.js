// ============================================================================
// An End-to-End AI Agent for Kaggle Competitions --- 3.5-minute talk.
// PPTX generator, kept in lockstep with slides_4min.tex (beamer) and with the
// poster: six slides, same order as the poster's blocks, same on-screen text,
// same numbers, same speaker notes.
//
// Budgeted seconds live at the end of every note and sum to 210.
// ============================================================================
const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in -- set before any addSlide()

// ---- Palette (identical RGB to the poster and the beamer preamble) ---------
const NAVY  = "003C64"; // navy     RGB(0,60,100)
const INK   = "111111"; // ink      RGB(17,17,17)
const MUT   = "5A5F66"; // mut      RGB(90,95,102)
const ICE   = "CADCFC"; // ice      RGB(202,220,252)
const PALE  = "A8BDD4"; // pale     RGB(168,189,212)
const OTHER = "B0BAC4"; // barother RGB(176,186,196)
const TRACK = "EEF0F3"; // bartrack RGB(238,240,243)
const WHITE = "FFFFFF";

const F = "Arial"; // matches the deck's \usepackage{helvet}

// ---- Slide title: navy, bold, flush left, no page number (as in beamer) ----
function head(slide, title) {
  slide.background = { color: WHITE };
  slide.addText(title, {
    x: 0.72, y: 0.28, w: 12.0, h: 0.60,
    fontFace: F, fontSize: 26, bold: true, color: NAVY, margin: 0, valign: "middle",
  });
}

// ---- List item from styled runs --------------------------------------------
// The beamer deck marks its list items with a middot rather than an itemize,
// and pptxgenjs starts a new paragraph at every run carrying bullet:true --
// which would split a mixed-style item into one bullet per run. So the marker
// is part of the first run's text, exactly as in the .tex.
function item(runs) {
  const out = runs.map((r) => ({ text: r.text, options: { ...(r.options || {}) } }));
  out[0] = { ...out[0], text: "\u00b7   " + out[0].text };
  return out;
}

// ---- Head-to-head bar, same geometry as the poster's \countbar -------------
function countbar(slide, y, label, color, frac, value, bold) {
  slide.addText(label, {
    x: 0.72, y: y, w: 2.0, h: 0.46,
    fontFace: F, fontSize: 17, bold: !!bold, color: INK, margin: 0, valign: "middle",
  });
  slide.addShape(pres.ShapeType.rect, {
    x: 2.85, y: y + 0.05, w: 5.60, h: 0.36,
    fill: { color: TRACK }, line: { color: TRACK, width: 0 },
  });
  slide.addShape(pres.ShapeType.rect, {
    x: 2.85, y: y + 0.05, w: 5.60 * frac, h: 0.36,
    fill: { color: color }, line: { color: color, width: 0 },
  });
  slide.addText(value, {
    x: 8.65, y: y, w: 1.6, h: 0.46,
    fontFace: F, fontSize: 17, color: INK, margin: 0, valign: "middle",
  });
}

// ================================================================== TITLE ---
{
  const s = pres.addSlide();
  s.background = { color: NAVY };

  s.addText("An End-to-End AI Agent\nfor Kaggle Competitions", {
    x: 0.90, y: 1.55, w: 11.5, h: 1.90,
    fontFace: F, fontSize: 40, bold: true, color: WHITE, margin: 0,
    lineSpacingMultiple: 1.1, valign: "middle",
  });
  s.addText(
    "An end-to-end LLM agent benchmarked against two frozen yardsticks " +
    "on 20 Kaggle competitions", {
      x: 0.90, y: 3.65, w: 11.0, h: 0.50,
      fontFace: F, fontSize: 18, color: ICE, margin: 0, valign: "middle",
    });
  s.addText(
    "Wei-Hao Huang, National Chengchi University   ·   " +
    "PI: Dr. Tso-Jung Yen, Institute of Statistical Science, Academia Sinica", {
      x: 0.90, y: 6.10, w: 11.8, h: 0.44,
      fontFace: F, fontSize: 13, color: PALE, margin: 0, valign: "middle",
    });

  s.addNotes(
    "This is an end-to-end agent for Kaggle competitions: it takes a raw " +
    "dataset and returns a submission, with no person in the loop. I " +
    "benchmarked it against two frozen reference agents on twenty " +
    "competitions, scored on the real leaderboards. (18s)"
  );
}

// ============================================================= 1 MOTIVATION --
{
  const s = pres.addSlide();
  head(s, "1 · Motivation");

  s.addText("Why these tasks are hard", {
    x: 0.72, y: 1.15, w: 11.9, h: 0.40,
    fontFace: F, fontSize: 19, bold: true, color: INK, margin: 0, valign: "middle",
  });
  s.addText(item([
    { text: "A competition demands the whole expert loop — validation design, " +
            "features, model selection, ensembling — and the top entries " +
            "separate at the " },
    { text: "fourth or fifth decimal.", options: { bold: true } },
  ]), {
    x: 0.95, y: 1.62, w: 11.6, h: 0.85,
    fontFace: F, fontSize: 17, color: INK, margin: 0,
  });
  s.addText(item([
    { text: "At that scale " },
    { text: "local cross-validation is an unreliable compass", options: { bold: true } },
    { text: ": a pipeline's own score routinely disagrees with the leaderboard." },
  ]), {
    x: 0.95, y: 2.52, w: 11.6, h: 0.55,
    fontFace: F, fontSize: 17, color: INK, margin: 0,
  });

  s.addText("Why an AI agent", {
    x: 0.72, y: 3.45, w: 11.9, h: 0.40,
    fontFace: F, fontSize: 19, bold: true, color: INK, margin: 0, valign: "middle",
  });
  s.addText(item([
    { text: "Tireless, systematic search: " },
    { text: "37–224", options: { bold: true } },
    { text: " candidate configurations per competition, every experiment logged." },
  ]), {
    x: 0.95, y: 3.92, w: 11.6, h: 0.55,
    fontFace: F, fontSize: 17, color: INK, margin: 0,
  });
  s.addText(item([
    { text: "The same disciplined protocol on 20 competitions in a row — a " +
            "consistency a person cannot hold." },
  ]), {
    x: 0.95, y: 4.52, w: 11.6, h: 0.55,
    fontFace: F, fontSize: 17, color: INK, margin: 0,
  });

  s.addNotes(
    "Competitions demand the whole expert loop, and the top of the " +
    "leaderboard separates at the fourth or fifth decimal, so tuning skill " +
    "distinguishes nobody — and at that scale a pipeline's own " +
    "cross-validation routinely disagrees with the leaderboard. What an agent " +
    "brings is tireless, logged search: between thirty-seven and two hundred " +
    "and twenty-four candidate configurations per competition, and the same " +
    "protocol run twenty times without drifting. (32s)"
  );
}

// ============================================================== 2 WORKFLOW ---
{
  const s = pres.addSlide();
  head(s, "2 · Workflow");

  // The architecture figure, rasterised from the same TikZ source the poster
  // and the report use (fig1_arch.pdf -> fig1_arch_hi-1.png at 300 dpi).
  s.addImage({ path: "fig1_arch_hi-1.png", x: 0.72, y: 1.05, w: 6.95, h: 6.19 });

  s.addText([
    { text: "The LLM reasons and decides at every stage", options: { bold: true } },
    { text: "; gradient boosting does the fitting; a tree search over complete " +
            "candidate solutions is the optimisation loop." },
  ], {
    x: 8.05, y: 1.60, w: 4.55, h: 1.60,
    fontFace: F, fontSize: 16, color: INK, margin: 0,
  });
  s.addText([
    { text: "The problem dossier", options: { bold: true } },
    { text: " classifies the task and injects external data " },
    { text: "upstream", options: { italic: true } },
    { text: ", before any modelling." },
  ], {
    x: 8.05, y: 3.35, w: 4.55, h: 1.20,
    fontFace: F, fontSize: 16, color: INK, margin: 0,
  });
  s.addText(
    "Lanes hold no Kaggle credentials; submissions are scored post-run on the " +
    "real leaderboard.", {
      x: 8.05, y: 4.70, w: 4.55, h: 1.10,
      fontFace: F, fontSize: 15, color: MUT, margin: 0,
    });

  s.addNotes(
    "The pipeline reads top to bottom. An LLM reasons and decides at every " +
    "stage, gradient boosting does the fitting, and a tree search over " +
    "complete candidate solutions is the optimisation loop. The one design " +
    "decision that mattered is where outside knowledge enters. My first " +
    "version put it downstream, at the blend — measured, it was worth ten to " +
    "the minus five. So it moved upstream, into a dossier stage that " +
    "classifies the task before any modelling. The lanes hold no Kaggle " +
    "credentials, so nothing inside a run can touch a leaderboard. (40s)"
  );
}

// =============================================================== 3 RESULTS ---
{
  const s = pres.addSlide();
  head(s, "3 · Results");

  s.addText(
    "All 20 lanes scored on the real leaderboard; each of the 60 duels judged " +
    "against the score noise measured on Kaggle's own public/private split — " +
    "38 decided, 22 too close to call.", {
      x: 0.72, y: 1.12, w: 11.9, h: 0.80,
      fontFace: F, fontSize: 17, color: INK, margin: 0,
    });

  s.addText([
    { text: "my-agent leads: 16 wins / 8 losses (66.7% of its decided duels)",
      options: { bold: true } },
    { text: " against 42.3% for each frozen yardstick." },
  ], {
    x: 0.72, y: 2.05, w: 11.9, h: 0.55,
    fontFace: F, fontSize: 17, color: INK, margin: 0,
  });

  s.addText("Head-to-head best private score", {
    x: 0.72, y: 2.85, w: 11.9, h: 0.40,
    fontFace: F, fontSize: 17, bold: true, color: INK, margin: 0, valign: "middle",
  });
  countbar(s, 3.40, "my-agent", NAVY,  0.50, "10 / 20", true);
  countbar(s, 4.00, "NVIDIA",   OTHER, 0.35, "7 / 20");
  countbar(s, 4.60, "AIDE",     OTHER, 0.15, "3 / 20");

  s.addText(
    "Beyond tabular: on three mid-complexity competitions (NLP, physiological " +
    "time series, medical imaging) my-agent takes the three-way best on all " +
    "three.", {
      x: 0.72, y: 5.45, w: 11.9, h: 0.70,
      fontFace: F, fontSize: 15, color: MUT, margin: 0,
    });

  s.addNotes(
    "Here is the reading. All twenty lanes are scored on the real " +
    "leaderboard, and every one of the sixty duels is judged against the " +
    "score noise measured on Kaggle's own split of the test set: thirty-eight " +
    "decide, twenty-two the data refuses to call. My agent leads — sixteen " +
    "wins to eight of its decided duels, sixty-six point seven percent, " +
    "against forty-two point three for each yardstick. It takes the best " +
    "private score on ten of the twenty competitions, and on three harder " +
    "competitions outside tabular data it takes the best of the three every " +
    "time. (45s)"
  );
}

// ============================================================== 4 LESSONS ---
{
  const s = pres.addSlide();
  head(s, "4 · Lessons learned");

  const B = { x: 0.95, w: 11.6, fontFace: F, fontSize: 16, color: INK, margin: 0 };

  s.addText(item([
    { text: "Put knowledge injection upstream.", options: { bold: true } },
    { text: " At the blend stage it measured near-nothing; moved into problem " +
            "identification it flags exactly the competitions that need external data." },
  ]), { ...B, y: 1.20, h: 0.85 });

  s.addText(item([
    { text: "Local CV is your only feedback.", options: { bold: true } },
    { text: " A competition returns no score until submission, so an error in " +
            "problem-level judgment leaves CV looking healthy while the " +
            "leaderboard says otherwise." },
  ]), { ...B, y: 2.20, h: 0.85 });

  s.addText(item([
    { text: "Version the agent itself.", options: { bold: true } },
    { text: " An architecture change invalidates every competition that ran " +
            "before it — start the benchmark over rather than mixing versions " +
            "in one table." },
  ]), { ...B, y: 3.20, h: 0.85 });

  s.addText(item([
    { text: "Isolate mechanically then audit transcripts", options: { bold: true } },
    { text: " — and audit the auditor: ours exempted every command starting " +
            "with uv, so the one command it existed to catch was never examined." },
  ]), { ...B, y: 4.20, h: 0.85 });

  s.addNotes(
    "Four things I would tell anyone doing this. Put knowledge injection " +
    "upstream — at the blend stage it bought nothing. While you iterate, local " +
    "cross-validation is the only feedback you get, and it can look healthy " +
    "while the leaderboard says otherwise. Version the agent, not just the " +
    "code: an architecture change invalidates every run before it. And isolate " +
    "mechanically, then read the transcripts — including your auditor's: mine " +
    "was exempting the commands every lane actually issues, so the one command " +
    "it existed to catch was never examined. (43s)"
  );
}

// ================================================= 5 WHERE IT FALLS SHORT ---
{
  const s = pres.addSlide();
  head(s, "5 · AI's blind spots");

  const B = { x: 0.95, w: 11.6, fontFace: F, fontSize: 16, color: INK, margin: 0 };

  s.addText(item([
    { text: "Engineering judgment is the gap.", options: { bold: true } },
    { text: " Nothing in the loop can tell at design time whether an " +
            "architecture will hold; the flaws surface once it runs, and the " +
            "debt grows with every competition added." },
  ]), { ...B, y: 1.20, h: 0.85 });

  s.addText(item([
    { text: "Pipelines log success for missing work.", options: { bold: true } },
    { text: " Our dossier emitted operators the evaluator had never " +
            "implemented — no warning, and the ledger recorded it as handled." },
  ]), { ...B, y: 2.20, h: 0.85 });

  s.addText(item([
    { text: "AI writing habits survive review", options: { bold: true } },
    { text: " — redundant suffixes, invented terminology, detail in place of " +
            "the question the reader came for." },
  ]), { ...B, y: 3.20, h: 0.85 });

  s.addText("github.com/930731haohao-afk/kaggle-agent", {
    x: 0.95, y: 6.40, w: 11.6, h: 0.40,
    fontFace: F, fontSize: 13, color: MUT, margin: 0, valign: "middle",
  });

  s.addNotes(
    "Where working this way still falls short. Engineering judgment is the " +
    "gap, not technique: nothing in the loop can tell at design time whether " +
    "an architecture will hold, and by the time the flaws surface the debt is " +
    "already there. A pipeline will happily log success for work nothing " +
    "performed — ours emitted instructions the evaluator had never " +
    "implemented, and the ledger recorded them as handled. And AI writing habits " +
    "survive review — redundant suffixes, invented terminology, and detail in " +
    "place of the question you actually came for. Thank you. (37s)"
  );
}

pres.writeFile({ fileName: "slides_4min.pptx" })
  .then((f) => console.log("wrote", f));
