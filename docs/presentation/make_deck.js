// ============================================================================
// An End-to-End AI Agent for Kaggle Competitions --- 3.5-minute conference talk.
// PPTX generator, kept in lockstep with slides_4min.tex (beamer).
//
// Same six slides, same order, same on-screen text, same numbers, same figures,
// same speaker notes. The beamer figures are drawn in TikZ, so they are
// reproduced here with native pptx shapes rather than rasters -- every label
// stays legible from a seat and the file has no external image dependency.
//
// Budgeted seconds live at the end of every note and sum to 207.
// ============================================================================
const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in -- set before any addSlide()

// ---- Palette (identical RGB to the beamer preamble) ------------------------
const NAVY  = "0F2A4A"; // navy   RGB(15,42,74)
const INK   = "1C1E21"; // ink    RGB(28,30,33)
const MUT   = "5A6472"; // mut    RGB(90,100,114)
const ICE   = "CADCFC"; // ice    RGB(202,220,252)
const BLUE  = "2B6CB0"; // barblue RGB(43,108,176)
const PALE  = "A8BDD4"; // pale   RGB(168,189,212)
const LT    = "F4F7FB"; // lt     RGB(244,247,251)
const GOLD  = "8A6D1A"; // gold   RGB(138,109,26)
const PLUM  = "7C4F96"; // plum   RGB(124,79,150)
const PLUMF = "F1EBF5"; // plum!12 fill
const WHITE = "FFFFFF";

const F = "Arial"; // matches the deck's \usepackage{helvet}

// ---- Frame title: navy, bold, flush left, no page number (as in beamer) ----
function head(slide, title) {
  slide.background = { color: WHITE };
  slide.addText(title, {
    x: 0.72, y: 0.30, w: 12.0, h: 0.60,
    fontFace: F, fontSize: 26, bold: true, color: NAVY, margin: 0, valign: "middle",
  });
}

// ---- Win-rate bar (beamer \ratebar) ----------------------------------------
// label | 62mm track with a coloured fill and white inset text | value
function ratebar(slide, y, label, color, frac, inside, value) {
  slide.addText(label, {
    x: 0.72, y: y, w: 1.55, h: 0.52,
    fontFace: F, fontSize: 18, color: INK, margin: 0, valign: "middle",
  });
  slide.addShape(pres.ShapeType.rect, {
    x: 2.40, y: y, w: 5.40, h: 0.52,
    fill: { color: LT }, line: { color: LT, width: 0 },
  });
  slide.addShape(pres.ShapeType.rect, {
    x: 2.40, y: y, w: 5.40 * frac, h: 0.52,
    fill: { color: color }, line: { color: color, width: 0 },
  });
  slide.addText(inside, {
    x: 2.55, y: y, w: 5.40 * frac - 0.20, h: 0.52,
    fontFace: F, fontSize: 13, bold: true, color: WHITE, margin: 0, valign: "middle",
  });
  slide.addText(value, {
    x: 8.05, y: y - 0.03, w: 1.8, h: 0.58,
    fontFace: F, fontSize: 24, bold: true, color: NAVY, margin: 0, valign: "middle",
  });
}

// ---- Score bar (beamer \scorebar) ------------------------------------------
function scorebar(slide, y, label, color, frac, value) {
  slide.addText(label, {
    x: 0.72, y: y, w: 1.55, h: 0.50,
    fontFace: F, fontSize: 18, color: INK, margin: 0, valign: "middle",
  });
  slide.addShape(pres.ShapeType.rect, {
    x: 2.40, y: y, w: 4.60, h: 0.50,
    fill: { color: LT }, line: { color: LT, width: 0 },
  });
  slide.addShape(pres.ShapeType.rect, {
    x: 2.40, y: y, w: 4.60 * frac, h: 0.50,
    fill: { color: color }, line: { color: color, width: 0 },
  });
  slide.addText(value, {
    x: 7.20, y: y, w: 2.0, h: 0.50,
    fontFace: F, fontSize: 17, bold: true, color: NAVY, margin: 0, valign: "middle",
  });
}

// ---- A square bracket standing in for TikZ's \decorate brace ---------------
// side: "down" draws the ticks downward (bracket sits above its caption).
function hBracket(slide, x1, x2, y, color, depth) {
  slide.addShape(pres.ShapeType.line, {
    x: x1, y: y, w: x2 - x1, h: 0, line: { color: color, width: 1 },
  });
  slide.addShape(pres.ShapeType.line, {
    x: x1, y: y, w: 0, h: depth, line: { color: color, width: 1 },
  });
  slide.addShape(pres.ShapeType.line, {
    x: x2, y: y, w: 0, h: depth, line: { color: color, width: 1 },
  });
}

// reach > 0 opens the bracket to the right, reach < 0 to the left. Widths are
// always emitted positive -- a negative cx is invalid OOXML.
function vBracket(slide, x, y1, y2, color, reach) {
  const tx = reach < 0 ? x + reach : x;
  const tw = Math.abs(reach);
  slide.addShape(pres.ShapeType.line, {
    x: x, y: y1, w: 0, h: y2 - y1, line: { color: color, width: 1 },
  });
  slide.addShape(pres.ShapeType.line, {
    x: tx, y: y1, w: tw, h: 0, line: { color: color, width: 1 },
  });
  slide.addShape(pres.ShapeType.line, {
    x: tx, y: y2, w: tw, h: 0, line: { color: color, width: 1 },
  });
}

// ============================================================ 1. THE QUESTION
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText("An End-to-End AI Agent\nfor Kaggle Competitions", {
    x: 0.85, y: 1.35, w: 11.8, h: 2.15,
    fontFace: F, fontSize: 44, bold: true, color: WHITE, lineSpacingMultiple: 1.25, margin: 0,
  });
  s.addText("Where should external knowledge enter an agent’s pipeline?", {
    x: 0.85, y: 3.85, w: 11.8, h: 0.65,
    fontFace: F, fontSize: 28, italic: true, color: ICE, margin: 0,
  });
  s.addText("Wei-Hao Huang   ·   Institute of Statistical Science, Academia Sinica", {
    x: 0.85, y: 5.55, w: 11.8, h: 0.45,
    fontFace: F, fontSize: 17, color: PALE, margin: 0,
  });
  s.addNotes(
    "A Kaggle competition is a scorable task: one submission, one hidden score. " +
    "The judgment calls come before the modelling, which makes it a sharp test of " +
    "machine judgment. One design question dominated the build — where should " +
    "external knowledge enter the pipeline? (17s)"
  );
}

// ====================================================== 2. THE ONE DECISION --
{
  const s = pres.addSlide();
  head(s, "The design decision: knowledge enters upstream");

  const CHIPS = ["Dossier", "EDA", "Features", "Models", "Evaluation", "Submission"];
  const CW = 1.55, CH = 0.62, PITCH = 1.98, CX0 = 0.72, CY = 3.05;

  CHIPS.forEach((name, i) => {
    const x = CX0 + i * PITCH;
    const isInj = i === 0;
    s.addShape(pres.ShapeType.roundRect, {
      x: x, y: CY, w: CW, h: CH, rectRadius: 0.06,
      fill: { color: isInj ? PLUMF : LT },
      line: { color: isInj ? PLUM : PALE, width: isInj ? 1.5 : 1 },
    });
    s.addText(name, {
      x: x, y: CY, w: CW, h: CH,
      fontFace: F, fontSize: 15, color: INK, align: "center", valign: "middle", margin: 0,
    });
    if (i < CHIPS.length - 1) {
      s.addShape(pres.ShapeType.line, {
        x: x + CW, y: CY + CH / 2, w: PITCH - CW, h: 0,
        line: { color: MUT, width: 1, endArrowType: "triangle" },
      });
    }
  });

  // --- upstream injection (the decision) ------------------------------------
  s.addShape(pres.ShapeType.line, {
    x: CX0 + CW / 2, y: 2.22, w: 0, h: 0.78,
    line: { color: PLUM, width: 2.25, endArrowType: "triangle" },
  });
  s.addText([
    { text: "Upstream:", options: { bold: true } },
    { text: " outside data enters here" },
  ], {
    x: 1.75, y: 1.88, w: 6.0, h: 0.35,
    fontFace: F, fontSize: 16, color: PLUM, margin: 0, valign: "middle",
  });

  // --- the judgment bracket --------------------------------------------------
  hBracket(s, CX0, CX0 + PITCH + CW, 3.92, GOLD, 0.08);
  s.addText([
    { text: "decisions taken " },
    { text: "before", options: { italic: true } },
    { text: " any model is fit" },
  ], {
    x: CX0, y: 4.06, w: 5.2, h: 0.35,
    fontFace: F, fontSize: 16, color: GOLD, margin: 0, valign: "middle",
  });

  // --- downstream injection (the abandoned option) --------------------------
  s.addShape(pres.ShapeType.line, {
    x: CX0 + 3 * PITCH + CW / 2, y: 3.72, w: 0, h: 0.62,
    line: { color: MUT, width: 1.25, dashType: "dash", beginArrowType: "triangle" },
  });
  s.addText([
    { text: "at the blend: measured ~10" },
    { text: "−5", options: { superscript: true } },
    { text: ",", options: { breakLine: true } },
    { text: "abandoned" },
  ], {
    x: 7.95, y: 4.18, w: 5.0, h: 0.70,
    fontFace: F, fontSize: 16, color: MUT, margin: 0, valign: "top",
  });

  // --- the claim --------------------------------------------------------------
  s.addText([
    { text: "Flags " },
    { text: "exactly", options: { bold: true } },
    { text: " the competitions that need outside data.    " },
    { text: "Every", options: { bold: true } },
    { text: " controlled arm beat its baseline." },
  ], {
    x: 0.72, y: 5.55, w: 12.0, h: 0.38,
    fontFace: F, fontSize: 18, color: INK, margin: 0, valign: "middle",
  });
  s.addText("No local signal I tested picks the form.", {
    x: 0.72, y: 5.95, w: 12.0, h: 0.38,
    fontFace: F, fontSize: 18, color: MUT, margin: 0, valign: "middle",
  });

  s.addNotes(
    "The conventional answer is downstream: extra ideas among the candidates you " +
    "blend. I built that, measured it: worth about ten to the minus five. So this " +
    "architecture injects upstream. Before any modelling, a dossier reads the " +
    "problem statement — does this task need outside data — and emits " +
    "whitelisted, leakage-guarded joins. The rest is conventional: the language " +
    "model decides at every stage, gradient boosting fits, tree search is the loop. " +
    "In a blind sweep it flagged exactly the competitions that need it and stayed " +
    "silent on the rest, and every controlled arm " +
    "beat their baselines. Nothing local I tested tells me which form to use. (37s)"
  );
}

// ================================================= 3. BUILD THE INSTRUMENT --
{
  const s = pres.addSlide();
  head(s, "Local validation cannot arbitrate — so build the instrument");

  s.addText([
    { text: "One lane: out-of-fold SMAPE " },
    { text: "6.706", options: { bold: true } },
    { text: " → real leaderboard " },
    { text: "49.40878", options: { bold: true, color: BLUE } },
  ], {
    x: 0.72, y: 1.10, w: 12.0, h: 0.40,
    fontFace: F, fontSize: 18, color: INK, margin: 0, valign: "middle",
  });

  // ---- left panel: the shared drift, and the gap that survives it ----------
  const XP = 1.95, XV = 4.95;               // public half / private half
  const yA1 = 2.70, yA2 = 3.32;             // upper (barblue) series
  const yB1 = 3.16, yB2 = 3.73;             // lower (mut) series

  s.addText("both agents drift the same way", {
    x: 0.90, y: 1.92, w: 5.1, h: 0.32,
    fontFace: F, fontSize: 13, color: GOLD, align: "center", margin: 0, valign: "middle",
  });

  [XP, XV].forEach((x) => {
    s.addShape(pres.ShapeType.line, {
      x: x, y: 2.42, w: 0, h: 1.95, line: { color: PALE, width: 1 },
    });
  });

  s.addShape(pres.ShapeType.line, {
    x: XP, y: yA1, w: XV - XP, h: yA2 - yA1, line: { color: BLUE, width: 1.75 },
  });
  s.addShape(pres.ShapeType.line, {
    x: XP, y: yB1, w: XV - XP, h: yB2 - yB1, line: { color: MUT, width: 1.75 },
  });
  [[XP, yA1, BLUE], [XV, yA2, BLUE], [XP, yB1, MUT], [XV, yB2, MUT]].forEach(([x, y, c]) => {
    s.addShape(pres.ShapeType.ellipse, {
      x: x - 0.055, y: y - 0.055, w: 0.11, h: 0.11,
      fill: { color: c }, line: { color: c, width: 0 },
    });
  });

  vBracket(s, XP - 0.13, yA1, yB1, NAVY, -0.07);
  s.addText("gap", {
    x: 1.00, y: (yA1 + yB1) / 2 - 0.15, w: 0.68, h: 0.30,
    fontFace: F, fontSize: 13, color: NAVY, align: "right", margin: 0, valign: "middle",
  });
  vBracket(s, XV + 0.13, yA2, yB2, NAVY, 0.07);
  s.addText("gap", {
    x: 5.25, y: (yA2 + yB2) / 2 - 0.15, w: 0.60, h: 0.30,
    fontFace: F, fontSize: 13, color: NAVY, margin: 0, valign: "middle",
  });

  s.addText("public half", {
    x: XP - 0.80, y: 4.40, w: 1.60, h: 0.28,
    fontFace: F, fontSize: 13, color: MUT, align: "center", margin: 0, valign: "middle",
  });
  s.addText("private half", {
    x: XV - 0.80, y: 4.40, w: 1.60, h: 0.28,
    fontFace: F, fontSize: 13, color: MUT, align: "center", margin: 0, valign: "middle",
  });
  s.addText("The shift cancels. The gap does not.", {
    x: 0.72, y: 4.86, w: 5.5, h: 0.32,
    fontFace: F, fontSize: 15, color: INK, align: "center", margin: 0, valign: "middle",
  });

  // ---- right panel: 60 duels, 38 decided ----------------------------------
  const GX = 8.05, GY = 2.32, CELL = 0.36, SQ = 0.30;
  for (let i = 0; i < 60; i++) {
    const r = Math.floor(i / 10), c = i % 10;
    const x = GX + c * CELL, y = GY + r * CELL;
    if (i < 38) {
      s.addShape(pres.ShapeType.roundRect, {
        x: x, y: y, w: SQ, h: SQ, rectRadius: 0.02,
        fill: { color: BLUE }, line: { color: BLUE, width: 0 },
      });
    } else {
      s.addShape(pres.ShapeType.roundRect, {
        x: x, y: y, w: SQ, h: SQ, rectRadius: 0.02,
        fill: { color: WHITE }, line: { color: PALE, width: 1 },
      });
    }
  }
  s.addText([
    { text: "60 duels: " },
    { text: "38 decided", options: { bold: true, color: BLUE } },
    { text: ", 22 undecidable — not ties." },
  ], {
    x: 6.90, y: 4.86, w: 5.9, h: 0.32,
    fontFace: F, fontSize: 15, color: INK, align: "center", margin: 0, valign: "middle",
  });

  s.addText([
    { text: "Verdict: the ordering replicates on both halves, " },
    { text: "and", options: { italic: true } },
    { text: " the gap exceeds its own movement." },
  ], {
    x: 0.72, y: 5.72, w: 12.0, h: 0.38,
    fontFace: F, fontSize: 17, color: MUT, margin: 0, valign: "middle",
  });

  s.addNotes(
    "Those are my own numbers, and between agents local validation cannot " +
    "arbitrate: one lane scored six point seven out of fold, forty-nine on the " +
    "leaderboard. So every verdict is a real submission. Kaggle scores each " +
    "submission on two halves of the test set. My first threshold used one " +
    "agent’s drift between halves and called most duels undecidable — but " +
    "that drift is shared: it moves all three agents together. Take the gap between " +
    "two agents on both halves and the common drift cancels. A verdict needs both " +
    "halves to agree, and the gap to beat its own movement. Sixty duels: " +
    "thirty-eight decided; twenty-two are not ties, but readings the test refuses " +
    "to make. (45s)"
  );
}

// ============================================================= 4. THE READING
{
  const s = pres.addSlide();
  head(s, "What the frozen three-way run measured");

  s.addText(
    "20 competitions   ·   one architecture, frozen before the run" +
    "   ·   no my-agent lane could reach a leaderboard", {
      x: 0.72, y: 1.15, w: 12.0, h: 0.35,
      fontFace: F, fontSize: 15, color: MUT, margin: 0, valign: "middle",
    });

  ratebar(s, 2.20, "my-agent", BLUE, 0.667, "16–8",  "66.7%");
  ratebar(s, 3.05, "NVIDIA",   MUT,  0.423, "11–15", "42.3%");
  ratebar(s, 3.90, "AIDE",     MUT,  0.423, "11–15", "42.3%");

  s.addText("Wins–losses over each agent's own decided duels: 24 for my-agent, 26 for each yardstick (38 decided duels = 76 agent-side outcomes).", {
    x: 0.72, y: 5.10, w: 12.0, h: 0.38,
    fontFace: F, fontSize: 18, color: INK, margin: 0, valign: "middle",
  });
  s.addText([
    { text: "Best private score: " },
    { text: "10 of 20", options: { bold: true } },
    { text: " (NVIDIA 7, AIDE 3)." },
  ], {
    x: 0.72, y: 5.55, w: 12.0, h: 0.38,
    fontFace: F, fontSize: 18, color: INK, margin: 0, valign: "middle",
  });

  s.addNotes(
    "Here is the reading. Twenty competitions, three agents, one architecture " +
    "frozen before the run, and no lane could reach a leaderboard. The two " +
    "yardsticks are frozen too: AIDE searches from zero, the NVIDIA agent " +
    "reproduces the highest-voted public kernel. I fixed nothing in either. " +
    "Mine wins sixteen of its twenty-four decided duels, sixty-six point seven " +
    "percent, against forty-two point three for each. That number measures the " +
    "whole architecture; the injection decision is measured separately, by the " +
    "controlled arms. (38s)"
  );
}

// ====================================================== 5. WHERE IT LOSES ---
{
  const s = pres.addSlide();
  head(s, "Where it loses is where the design predicts");

  s.addText("Worst loss — SMAPE, lower is better. Bars show how far behind the best score each agent is.", {
    x: 0.72, y: 1.15, w: 12.0, h: 0.35,
    fontFace: F, fontSize: 15, color: MUT, margin: 0, valign: "middle",
  });

  scorebar(s, 1.70, "NVIDIA",   BLUE, 0.000, "5.48351");
  scorebar(s, 2.48, "AIDE",     MUT,  0.184, "9.72095");
  scorebar(s, 3.26, "my-agent", GOLD, 1.000, "28.49605");

  s.addText("My two worst losses are the only competitions here with a mature public solution.", {
    x: 0.72, y: 4.10, w: 12.0, h: 0.38,
    fontFace: F, fontSize: 18, color: INK, margin: 0, valign: "middle",
  });
  s.addText("Reproduction dominates there; search wins the rest.", {
    x: 0.72, y: 4.50, w: 12.0, h: 0.38,
    fontFace: F, fontSize: 17, color: MUT, margin: 0, valign: "middle",
  });

  // ---- the standing is fragile --------------------------------------------
  const YB = 5.30, YC = 5.95;
  s.addText("72.7%", {
    x: 1.35, y: YB, w: 2.5, h: 0.55,
    fontFace: F, fontSize: 22, color: MUT, align: "center", margin: 0, valign: "middle",
  });
  s.addText("66.7%", {
    x: 5.05, y: YB - 0.10, w: 3.2, h: 0.75,
    fontFace: F, fontSize: 42, bold: true, color: NAVY, align: "center", margin: 0, valign: "middle",
  });
  s.addText("68.0%", {
    x: 9.45, y: YB, w: 2.5, h: 0.55,
    fontFace: F, fontSize: 22, color: MUT, align: "center", margin: 0, valign: "middle",
  });
  s.addShape(pres.ShapeType.line, {
    x: 4.05, y: YB + 0.28, w: 0.85, h: 0,
    line: { color: MUT, width: 1, beginArrowType: "triangle" },
  });
  s.addShape(pres.ShapeType.line, {
    x: 8.40, y: YB + 0.28, w: 0.85, h: 0,
    line: { color: MUT, width: 1, endArrowType: "triangle" },
  });
  s.addText("drop one\ncompetition", {
    x: 1.35, y: YC, w: 2.5, h: 0.60,
    fontFace: F, fontSize: 13, color: MUT, align: "center", margin: 0, valign: "top",
  });
  s.addText("exact decimal", {
    x: 5.05, y: YC, w: 3.2, h: 0.30,
    fontFace: F, fontSize: 13, color: NAVY, align: "center", margin: 0, valign: "top",
  });
  s.addText("one duel on its\nvariance floor", {
    x: 9.45, y: YC, w: 2.5, h: 0.60,
    fontFace: F, fontSize: 13, color: MUT, align: "center", margin: 0, valign: "top",
  });

  s.addNotes(
    "Where it loses is the interesting half. My two worst losses are the only two " +
    "competitions with a mature public solution, and there the reproduction agent " +
    "beats me decisively — twenty-eight against five and nine. That is the " +
    "mechanism the design predicts: reproduction wins where a strong public solution " +
    "exists, search wins where none does. The standing is also fragile: drop one " +
    "competition and it reads seventy-two point seven, and one duel sits exactly " +
    "on its noise floor. A ranking belongs to the method and the " +
    "evaluation set together — mine included. (37s)"
  );
}

// ============================================================== 6. TAKE AWAY
{
  const s = pres.addSlide();
  head(s, "Take away");

  const LINES = [
    "State the error model, or the ranking is your own noise.",
    "Freeze the yardsticks. Then read the transcripts, not just the scores.",
  ];
  LINES.forEach((line, i) => {
    const y = 2.20 + i * 0.95;
    s.addShape(pres.ShapeType.rect, {
      x: 0.80, y: y + 0.10, w: 0.15, h: 0.20,
      fill: { color: GOLD }, line: { color: GOLD, width: 0 },
    });
    s.addText(line, {
      x: 1.15, y: y, w: 11.5, h: 0.42,
      fontFace: F, fontSize: 22, color: INK, margin: 0, valign: "middle",
    });
  });

  s.addShape(pres.ShapeType.rect, {
    x: 0.80, y: 4.25, w: 11.75, h: 1.15,
    fill: { color: NAVY }, line: { color: NAVY, width: 0 },
  });
  s.addText(
    "Where knowledge enters is an architectural choice — and the controlled arms measure which placement wins.", {
      x: 1.05, y: 4.25, w: 11.25, h: 1.15,
      fontFace: F, fontSize: 22, color: WHITE, margin: 0, valign: "middle",
      lineSpacingMultiple: 1.18,
    });

  s.addText("github.com/930731haohao-afk/kaggle-agent", {
    x: 0.80, y: 5.62, w: 11.75, h: 0.30,
    fontFace: F, fontSize: 12, color: MUT, margin: 0, valign: "middle",
  });

  s.addNotes(
    "Where knowledge enters is an architectural choice with a measurable answer. " +
    "State an error model, or a ranking is your own noise. And freeze your " +
    "yardsticks, then read the transcripts — mine caught a blind spot in my own " +
    "auditor, which was skipping the exact command it existed to catch. Corrected, a " +
    "re-run over all twenty transcripts finds no network access. Thank you. (25s)"
  );
}

pres.writeFile({ fileName: "slides_4min.pptx" })
  .then((f) => console.log("written: " + f));
