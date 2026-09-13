const pptxgen = require("pptxgenjs");
const path = require("path");

const SUB = "C:/DriveSync/Creative/hackathons/ai-infra-summit-2026/submission";
const IMG = (n) => path.join(SUB, "img", n);
const C = { bg: "161B22", panel: "222A35", panel2: "2B3440", yellow: "E8B923", green: "4ADE80",
            white: "FFFFFF", mute: "9AA7B5", red: "F87171" };
const H = "Arial", B = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.title = "Two-Arm Table Setter";

function bg(s) { s.background = { color: C.bg }; }
function title(s, t, sub) {
  s.addText(t, { x: 0.5, y: 0.3, w: 9, h: 0.6, fontFace: H, fontSize: 30, bold: true, color: C.white, margin: 0, isTextBox: true });
  if (sub) s.addText(sub, { x: 0.5, y: 0.88, w: 9, h: 0.35, fontFace: B, fontSize: 14, color: C.mute, margin: 0, isTextBox: true });
}
function circle(s, x, y, label, d = 0.42) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: C.yellow }, line: { color: C.yellow } });
  s.addText(label, { x, y, w: d, h: d, align: "center", valign: "middle", fontFace: H, fontSize: 14, bold: true, color: C.bg, margin: 0, isTextBox: true });
}

// 1 — cover
{
  const s = pres.addSlide(); bg(s);
  s.addImage({ path: path.join(SUB, "cover.png"), x: 0, y: 0, w: 10, h: 5.625 });
  s.addText("AI Infra Summit Hackathon · Intel online track · Bimanual VLA manipulation", {
    x: 0.45, y: 0.25, w: 9, h: 0.35, fontFace: B, fontSize: 13, color: C.mute, margin: 0, isTextBox: true });
  s.addNotes("Two robot arms that set a dinner table when you tell them to. They look at the table with a camera, understand the command with a vision-language model, hand things to each other, and fix their own mistakes. Everything runs on a five-year-old Intel laptop with OpenVINO: no graphics card, no cloud.");
}

// 2 — the idea + headline numbers
{
  const s = pres.addSlide(); bg(s);
  title(s, "One command. Two hands. A set table.", "Tested the way Intel asked: random tables, success checked against simulator truth");
  const lines = [
    { text: "Type what you want", options: { bold: true, color: C.white, breakLine: true } },
    { text: "\"put the red thing top left of the plate and the grey utensil on the right\"", options: { italic: true, color: C.mute, breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "The robot works out the rest", options: { bold: true, color: C.white, breakLine: true } },
    { text: "which item is meant, which arm, when to pass an item across, what to do first — and checks every step with its camera", options: { color: C.mute } },
  ];
  s.addText(lines, { x: 0.5, y: 1.5, w: 4.1, h: 3.4, fontFace: B, fontSize: 16, valign: "top", margin: 0, isTextBox: true });
  const stats = [["30/30", "tables set correctly"], ["29/30", "harder tables: positions, sizes, colours"],
                 ["95%", "trained ACT policy on unseen tables"], ["~4 s", "to understand a command (was 30 s)"]];
  stats.forEach(([n, l], i) => {
    const x = 5.0 + (i % 2) * 2.3, y = 1.45 + Math.floor(i / 2) * 1.75;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 2.1, h: 1.55, fill: { color: C.panel }, line: { color: C.panel }, rectRadius: 0.08 });
    s.addText(n, { x, y: y + 0.12, w: 2.1, h: 0.75, align: "center", fontFace: H, fontSize: 36, bold: true, color: C.yellow, margin: 0, isTextBox: true });
    s.addText(l, { x: x + 0.12, y: y + 0.88, w: 1.86, h: 0.55, align: "center", valign: "top", fontFace: B, fontSize: 12, color: C.white, margin: 0, isTextBox: true });
  });
  s.addNotes("The robot sets the table correctly on 30 out of 30 random tables, and 29 out of 30 when we make it harder by moving things further and changing their sizes and colours. Success is always judged against where objects really are in the simulator, not against what the robot thinks.");
}

// 2b — why it matters (business value)
{
  const s = pres.addSlide(); bg(s);
  title(s, "Why it matters", "Service robots for homes, hospitals, hotels and restaurants");
  const cards = [
    ["Anyone can instruct it", "Plain-language instructions. No programming, no app, no fixed commands."],
    ["Two hands, real tasks", "Passing, holding and placing — the everyday handling that one-armed robots can't do."],
    ["Checks its own work", "A bump or a slip is noticed and fixed, so a job gets finished without a person watching."],
    ["Cheap to run and to teach", "Runs on an ordinary Intel laptop with OpenVINO. Skills teach themselves from auto-recorded demos — no hand-collected data."],
  ];
  cards.forEach(([h, d], i) => {
    const x = 0.5 + (i % 2) * 4.6, y = 1.45 + Math.floor(i / 2) * 1.95;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 4.4, h: 1.75, fill: { color: C.panel }, line: { color: C.panel }, rectRadius: 0.08 });
    circle(s, x + 0.25, y + 0.28, String(i + 1), 0.4);
    s.addText(h, { x: x + 0.8, y: y + 0.28, w: 3.4, h: 0.4, fontFace: H, fontSize: 16, bold: true, color: C.white, margin: 0, isTextBox: true });
    s.addText(d, { x: x + 0.8, y: y + 0.72, w: 3.4, h: 0.9, fontFace: B, fontSize: 13, color: C.mute, valign: "top", margin: 0, isTextBox: true });
  });
  s.addNotes("Why does this matter? Service robots in homes, hospitals, hotels and restaurants need exactly this combination: anyone can tell them what to do in plain language, they can use two hands, and they check their own work. And it's cheap: it runs on an ordinary laptop, and the skills teach themselves from demonstrations the robot records on its own.");
}

// 3 — how it works
{
  const s = pres.addSlide(); bg(s);
  title(s, "Observe, understand, plan, act, look again");
  const steps = [["Observe", "overhead colour + depth camera finds every object"],
                 ["Understand", "Qwen3-VL-4B on OpenVINO reads the command and the picture"],
                 ["Plan", "which arm, hand-off or set-down swap, what first"],
                 ["Act", "two SO-101 arms; a trained ACT policy for the mug"],
                 ["Look again", "anything knocked out of place is fixed first"]];
  steps.forEach(([t, d], i) => {
    const x = 0.5 + i * 1.82;
    circle(s, x, 1.25, String(i + 1));
    s.addText(t, { x, y: 1.78, w: 1.7, h: 0.35, fontFace: H, fontSize: 15, bold: true, color: C.white, margin: 0, isTextBox: true });
    s.addText(d, { x, y: 2.12, w: 1.65, h: 0.9, fontFace: B, fontSize: 12, color: C.mute, valign: "top", margin: 0, isTextBox: true });
  });
  s.addImage({ path: IMG("eyes.png"), x: 0.5, y: 3.2, w: 2.8, h: 2.1 });
  s.addText([
    { text: "What the robot sees", options: { bold: true, color: C.white, fontSize: 16, breakLine: true } },
    { text: "Objects are found from the height map and their shape, never from the simulator. On 20 random tables every object was found, 0.5 mm average error.", options: { color: C.mute, fontSize: 13, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    { text: "Held items are tracked from the arm's own joint angles, like a real robot.", options: { color: C.mute, fontSize: 13 } },
  ], { x: 3.6, y: 3.25, w: 5.9, h: 2.0, fontFace: B, valign: "top", margin: 0, isTextBox: true });
  s.addNotes("Five steps, in a loop. The camera image is the only way the robot learns where things are. The vision-language model turns the command into goals, the planner decides who does what, the arms act, and after every item the robot looks again.");
}

// 4 — teamwork + recovery
{
  const s = pres.addSlide(); bg(s);
  title(s, "Teamwork, and it fixes its own mistakes");
  s.addImage({ path: IMG("handoff_crop.png"), x: 0.5, y: 1.3, w: 4.4, h: 2.76 });
  s.addText([
    { text: "Hand-off", options: { bold: true, color: C.white, fontSize: 16, breakLine: true } },
    { text: "The fork starts on arm B's side but belongs on arm A's. B carries it in, A moves up at the same time, lines up 1.5 mm away, grips, then B lets go.", options: { color: C.mute, fontSize: 13 } },
  ], { x: 0.5, y: 4.15, w: 4.4, h: 1.3, fontFace: B, valign: "top", margin: 0, isTextBox: true });
  s.addImage({ path: IMG("recovery_bumped.png"), x: 5.2, y: 1.3, w: 2.1, h: 1.58 });
  s.addImage({ path: IMG("recovery_fixed.png"), x: 7.4, y: 1.3, w: 2.1, h: 1.58 });
  s.addText("knocked", { x: 5.2, y: 2.92, w: 2.1, h: 0.3, align: "center", fontFace: B, fontSize: 12, color: C.red, margin: 0, isTextBox: true });
  s.addText("seen and fixed", { x: 7.4, y: 2.92, w: 2.1, h: 0.3, align: "center", fontFace: B, fontSize: 12, color: C.green, margin: 0, isTextBox: true });
  s.addText([
    { text: "Recovery", options: { bold: true, color: C.white, fontSize: 16, breakLine: true } },
    { text: "Mid-task, the plate is knocked 6 cm. After the next item the robot looks at everything it already placed, sees the plate has moved, and puts it back before carrying on.", options: { color: C.mute, fontSize: 13 } },
  ], { x: 5.2, y: 3.35, w: 4.3, h: 1.6, fontFace: B, valign: "top", margin: 0, isTextBox: true });
  s.addNotes("Cutlery often starts on the wrong arm's side, so the arms pass it across: one holds an end, the other takes the other end. And the robot checks its own work. Here we knock the plate out of place halfway through: it notices and fixes it before carrying on.");
}

// 5 — training story
{
  const s = pres.addSlide(); bg(s);
  title(s, "Our own teacher trained the policy", "LeRobot ACT on a laptop CPU · 1,219 demonstrations recorded automatically · success on unseen tables");
  s.addChart(pres.charts.BAR, [{ name: "Success", labels: ["v1", "v2", "v3", "v4", "v5", "v6", "v7"], values: [5, 10, 10, 0, 10, 20, 95] }], {
    x: 0.4, y: 1.35, w: 5.3, h: 3.9, barDir: "col", chartColors: [C.yellow],
    showValue: true, dataLabelPosition: "outEnd", dataLabelColor: C.white, dataLabelFontSize: 11, dataLabelFormatCode: "0\"%\"",
    catAxisLabelColor: C.mute, valAxisLabelColor: C.mute, valAxisMaxVal: 100, valAxisMinVal: 0,
    valGridLine: { color: "2E3743", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
    showTitle: true, title: "Mug placed within 1.5 cm", titleColor: C.white, titleFontSize: 13,
  });
  const lessons = [["A teacher must be consistent", "picking the best of 12 grip angles looked random to the student — one fixed rule"],
                   ["Targets, not small moves", "tiny per-step errors added up to centimetres"],
                   ["Think in the mug's frame", "\"grip beside the mug\" becomes the same move every time"]];
  lessons.forEach(([h, d], i) => {
    const y = 1.45 + i * 1.3;
    circle(s, 6.0, y, String(i + 1), 0.38);
    s.addText(h, { x: 6.55, y: y - 0.02, w: 3.1, h: 0.35, fontFace: H, fontSize: 14, bold: true, color: C.white, margin: 0, isTextBox: true });
    s.addText(d, { x: 6.55, y: y + 0.33, w: 3.1, h: 0.75, fontFace: B, fontSize: 12, color: C.mute, valign: "top", margin: 0, isTextBox: true });
  });
  s.addNotes("Instead of recording examples by hand, our scripted skills acted as the teacher and recorded over a thousand perfect demonstrations automatically, keeping only the successes. The first six versions failed, and each failure taught us something. Version seven places the mug correctly 95 percent of the time on tables it has never seen.");
}

// 6 — Intel optimisation
{
  const s = pres.addSlide(); bg(s);
  title(s, "OpenVINO: faster, same accuracy", "Intel Core i7-1165G7 + Iris Xe iGPU, 16 GB · OpenVINO 2026.3");
  s.addChart(pres.charts.BAR, [{ name: "ms", labels: ["INT8 weights", "OpenVINO INT8", "OpenVINO FP16", "OpenVINO FP32", "PyTorch FP32"], values: [3.53, 3.68, 1.88, 1.73, 4.34] }], {
    x: 0.4, y: 1.35, w: 5.4, h: 3.9, barDir: "bar", chartColors: [C.yellow],
    showValue: true, dataLabelPosition: "outEnd", dataLabelColor: C.white, dataLabelFontSize: 11, dataLabelFormatCode: "0.00\" ms\"",
    catAxisLabelColor: C.white, valAxisLabelColor: C.mute, valGridLine: { color: "2E3743", size: 0.5 }, catGridLine: { style: "none" },
    showLegend: false, showTitle: true, title: "ACT policy, CPU, ms per call (lower is better)", titleColor: C.white, titleFontSize: 13,
  });
  const calls = [["2.5×", "faster policy on CPU with OpenVINO — task success unchanged at every precision (9–10/10)"],
                 ["30 s → 4 s", "to read a command: 480 px picture and short answers instead of JSON"],
                 ["2×", "faster on the Iris Xe iGPU for large pictures (7.5 s vs 14.9 s)"]];
  calls.forEach(([n, d], i) => {
    const y = 1.4 + i * 1.3;
    s.addText(n, { x: 6.1, y, w: 3.5, h: 0.5, fontFace: H, fontSize: 26, bold: true, color: C.yellow, margin: 0, isTextBox: true });
    s.addText(d, { x: 6.1, y: y + 0.5, w: 3.5, h: 0.7, fontFace: B, fontSize: 12, color: C.mute, valign: "top", margin: 0, isTextBox: true });
  });
  s.addNotes("OpenVINO makes our trained policy two and a half times faster on the laptop CPU, and the robot still succeeds just as often. For a model this small, INT8 and the integrated GPU don't pay off, and we say so. The vision-language model went from thirty seconds to four seconds per command, and the integrated GPU halves the time for large pictures.");
}

// 7 — honest + try it
{
  const s = pres.addSlide(); bg(s);
  title(s, "Open, reproducible, honest");
  s.addText([
    { text: "What we'd do next", options: { bold: true, color: C.white, fontSize: 16, breakLine: true } },
    { text: "Run the benchmark on a Core Ultra with an NPU (ours is an 11th-gen i7)", options: { bullet: true, color: C.mute, breakLine: true } },
    { text: "Train policies for the hand-off too; today most skills are the scripted teacher", options: { bullet: true, color: C.mute, breakLine: true } },
    { text: "Add the drawer and pouring from Intel's example task", options: { bullet: true, color: C.mute } },
  ], { x: 0.5, y: 1.3, w: 4.4, h: 2.6, fontFace: B, fontSize: 13, valign: "top", paraSpaceAfter: 6, margin: 0, isTextBox: true });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 5.2, y: 1.3, w: 4.3, h: 2.1, fill: { color: C.panel }, line: { color: C.panel }, rectRadius: 0.08 });
  s.addText([
    { text: "Try it", options: { bold: true, color: C.white, fontFace: H, fontSize: 16, breakLine: true } },
    { text: "python brain/run.py \"set the table\"", options: { fontFace: "Courier New", fontSize: 11, color: C.yellow, breakLine: true } },
    { text: "python brain/evaluate.py --seeds 10 --hard", options: { fontFace: "Courier New", fontSize: 11, color: C.yellow, breakLine: true } },
    { text: "python bench/benchmark.py", options: { fontFace: "Courier New", fontSize: 11, color: C.yellow, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    { text: "github.com/PegBitStudio/two-arm-table-setter", options: { color: C.white, fontFace: B, fontSize: 13 } },
  ], { x: 5.45, y: 1.45, w: 3.9, h: 1.85, valign: "top", paraSpaceAfter: 4, margin: 0, isTextBox: true });
  s.addText("Every scene is seeded and every number here is reproducible from one command. MIT licence. Built with MuJoCo, LeRobot, OpenVINO and Qwen3-VL.", {
    x: 0.5, y: 3.95, w: 9, h: 0.6, fontFace: B, fontSize: 13, color: C.mute, margin: 0, isTextBox: true });
  s.addNotes("Everything is open source and seeded, so every number in this deck can be reproduced with one command. We're upfront about what's missing: a Core Ultra and NPU run, more learned skills, and the drawer and pouring. Thank you.");
}

pres.writeFile({ fileName: path.join(SUB, "slides.pptx") }).then((f) => console.log("saved", f));
