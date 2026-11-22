// Builds submission/SiteSafe_Presentation.pptx (10-minute talk + Q&A), styled like the app.
//   NODE_PATH=<folder with pptxgenjs> node scripts/make_presentation.js
// Every number comes from results/ (see docs/report.md); screenshots from scripts/take_screenshots.py.
const path = require('path');
const fs = require('fs');
const pptxgen = require('pptxgenjs');

const ROOT = path.resolve(__dirname, '..');
const SHOT = (f) => path.join(ROOT, 'submission', 'screenshots', f);
const FIG = (f) => path.join(ROOT, 'submission', 'figures', f);
const LIVE = 'https://aftaab9.github.io/sitesafe/';
const REPO = 'https://github.com/Aftaab9/sitesafe';

// App palette ("control-room charcoal")
const C = {
  bg: '0B0D10', surface: '13171C', raised: '1C2229', line: '2C343E', ink: 'E8ECEF', muted: '8B98A5',
  hivis: 'FFD400', hazard: 'FF7A1A', ok: '2FD67B', danger: 'FF5B5D', unsure: '5BC0FF',
};
const HEAD = 'Arial';
const BODY = 'Calibri';

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE'; // 13.33 x 7.5 in
pres.author = 'SiteSafe';
pres.title = 'SiteSafe: PPE compliance with privacy';
pres.theme = { headFontFace: HEAD, bodyFontFace: BODY };

pres.defineSlideMaster({
  title: 'CONTENT',
  background: { color: C.bg },
  objects: [
    { text: { text: 'SiteSafe · MAIB AI 221 Project 6', options: { x: 0.6, y: 7.02, w: 6, h: 0.3, fontFace: BODY, fontSize: 10, color: C.muted } } },
  ],
  slideNumber: { x: 12.3, y: 7.02, w: 0.5, h: 0.3, fontFace: BODY, fontSize: 10, color: C.muted, align: 'right' },
});
pres.defineSlideMaster({ title: 'DARK', background: { color: C.bg } });

function title(slide, text, sub) {
  slide.addText(text, { x: 0.6, y: 0.35, w: 12.1, h: 0.8, fontFace: HEAD, fontSize: 32, bold: true, color: C.ink, margin: 0, isTextBox: true });
  if (sub) slide.addText(sub, { x: 0.6, y: 1.1, w: 12.1, h: 0.45, fontFace: BODY, fontSize: 16, color: C.muted, margin: 0, isTextBox: true });
}
function card(slide, x, y, w, h, fill = C.surface) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: C.line, width: 1 }, rectRadius: 0.08 });
}
function stat(slide, x, y, w, value, label, color = C.hivis) {
  card(slide, x, y, w, 1.55);
  // Shrink long values so they stay on one line (bold Arial is about 0.62 em per character).
  const size = Math.min(34, Math.floor(((w - 0.5) * 72) / (value.length * 0.62)));
  slide.addText(value, { x: x + 0.25, y: y + 0.15, w: w - 0.5, h: 0.8, fontFace: HEAD, fontSize: size, bold: true, color, margin: 0, valign: 'middle', isTextBox: true });
  slide.addText(label, { x: x + 0.25, y: y + 0.92, w: w - 0.5, h: 0.55, fontFace: BODY, fontSize: 13, color: C.muted, margin: 0, valign: 'top', isTextBox: true });
}
function bullets(slide, items, x, y, w, h, size = 16) {
  slide.addText(items.map((t, i) => ({ text: t, options: { bullet: { indent: 18 }, breakLine: i < items.length - 1 } })),
    { x, y, w, h, fontFace: BODY, fontSize: size, color: C.ink, valign: 'top', paraSpaceAfter: 8, margin: 0, isTextBox: true });
}
function shot(slide, file, x, y, w, h) {
  slide.addShape(pres.shapes.RECTANGLE, { x: x - 0.03, y: y - 0.03, w: w + 0.06, h: h + 0.06, fill: { color: C.line }, line: { color: C.line } });
  slide.addImage({ path: file, x, y, w, h, sizing: { type: 'cover', w, h } });
}
function figure(slide, file, x, y, w, h) {
  card(slide, x, y, w, h, 'FFFFFF');
  slide.addImage({ path: file, x: x + 0.12, y: y + 0.12, w: w - 0.24, h: h - 0.24, sizing: { type: 'contain', w: w - 0.24, h: h - 0.24 } });
}
function tag(slide, text, x, y, color) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 0.18, h: 0.18, fill: { color }, line: { color }, rectRadius: 0.04 });
  slide.addText(text, { x: x + 0.28, y: y - 0.06, w: 3.6, h: 0.3, fontFace: BODY, fontSize: 13, color: C.ink, margin: 0, isTextBox: true });
}

// 1 · Title ------------------------------------------------------------------
let s = pres.addSlide({ masterName: 'DARK' });
s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 13.33, h: 0.16, fill: { color: C.hivis }, line: { color: C.hivis } });
s.addText('SITESAFE · PROJECT 6 · CONSTRUCTION', { x: 0.7, y: 1.0, w: 6, h: 0.35, fontFace: BODY, fontSize: 13, color: C.muted, charSpacing: 2, margin: 0, isTextBox: true });
s.addText([{ text: 'See who is missing PPE.', options: { breakLine: true, color: C.ink } }, { text: 'Without seeing who they are.', options: { color: C.hivis } }],
  { x: 0.7, y: 1.5, w: 6.4, h: 1.9, fontFace: HEAD, fontSize: 33, bold: true, margin: 0, valign: 'top', isTextBox: true });
s.addText('On-device PPE checks for site safety officers: photos, short clips and live video. Faces blurred, nothing uploaded.',
  { x: 0.7, y: 3.55, w: 5.9, h: 1.0, fontFace: BODY, fontSize: 16, color: C.muted, margin: 0, valign: 'top', isTextBox: true });
s.addText([{ text: 'Live app  ', options: { color: C.muted } }, { text: LIVE, options: { color: C.hivis, hyperlink: { url: LIVE } } }],
  { x: 0.7, y: 5.2, w: 6.2, h: 0.35, fontFace: BODY, fontSize: 14, margin: 0, isTextBox: true });
s.addText([{ text: 'Code  ', options: { color: C.muted } }, { text: REPO, options: { color: C.ink, hyperlink: { url: REPO } } }],
  { x: 0.7, y: 5.6, w: 6.2, h: 0.35, fontFace: BODY, fontSize: 14, margin: 0, isTextBox: true });
s.addText('MAIB AI 221 Computer Vision with Applications · S P Jain School of Global Management', { x: 0.7, y: 6.6, w: 7, h: 0.3, fontFace: BODY, fontSize: 11, color: C.muted, margin: 0, isTextBox: true });
shot(s, SHOT('02_photo_compliant_crew.png'), 7.2, 1.3, 5.5, 5.0);
s.addNotes('[0:00-0:30] Hello. SiteSafe answers one question for a construction safety officer: who on this site is missing their PPE right now? It does that from a photo, a short clip or a live camera, and it does it without identifying anyone: faces are blurred, and no image ever leaves the phone or laptop. Everything you will see today runs live at the address on the slide.');

// 2 · Problem, user, cost of errors ---------------------------------------------
s = pres.addSlide({ masterName: 'CONTENT' });
title(s, 'The user, the decision and what an error costs', 'A site safety officer on a walk-round: which workers do I go and speak to now?');
stat(s, 0.6, 1.9, 3.85, 'Missed violation', 'A worker without a helmet shown as fine: the costly error, can end in a head injury', C.danger);
stat(s, 4.74, 1.9, 3.85, 'False alarm', 'A compliant worker flagged: costs the officer a short walk', C.hazard);
stat(s, 8.88, 1.9, 3.85, 'Not sure', 'A third state: hard cases go back to the officer instead of a guessed green', C.unsure);
card(s, 0.6, 3.75, 12.13, 2.95);
s.addText('Design choices that follow', { x: 0.9, y: 3.95, w: 11.5, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: C.ink, margin: 0, isTextBox: true });
bullets(s, [
  'Recall on the five "no_" classes (visibly missing PPE) matters more than overall accuracy',
  'Per-class "not sure" thresholds chosen on validation, so uncertain workers turn blue, not green',
  'Aggregate statistics only: a site compliance %, counts by PPE type, anonymous worker numbers',
  'Not a surveillance tool: no identities, no stored images, workers told what the check is for',
], 0.9, 4.45, 11.5, 2.2, 16);
s.addNotes('[0:30-1:15] The user is the site safety officer. The decision is narrow on purpose: which workers should I go and talk to now. The two errors are not equal. A missed "no helmet" can end in a head injury; a false alarm costs a short walk. So I optimise recall on the five "no_" classes, and I added a third state, not sure, so the model hands hard cases back to the person instead of guessing green. And because cameras on workers are easily abused, the app reports aggregates only and never identifies anyone.');

// 3 · Data ------------------------------------------------------------------------
s = pres.addSlide({ masterName: 'CONTENT' });
title(s, 'Data: Construction-PPE plus 30 real photos of my own', 'Ultralytics Construction-PPE, CC BY 4.0 · 1,000 images · 12 classes, five of them "no_"');
figure(s, FIG('fig_class_counts.png'), 0.6, 1.8, 6.6, 4.9);
stat(s, 7.5, 1.8, 2.5, '700 / 200 / 100', 'seeded train / val / test split; test used once');
stat(s, 10.23, 1.8, 2.5, '30 photos', 'my own Wikimedia test set, 91 workers');
card(s, 7.5, 3.6, 5.23, 3.1);
bullets(s, [
  'Test split is never used for training or tuning',
  'Own photos: CC0, public domain, CC BY, CC BY-SA, with authors recorded; 12 at night or in tunnels',
  'About 570 of the 800 training images look AI-generated, and "no_boots" mostly means canvas trainers: the main reason real photos are harder',
], 7.75, 3.85, 4.8, 2.8, 14);
s.addNotes('[1:15-2:00] The data is Ultralytics Construction-PPE: a thousand images, twelve classes, including five "no_" classes for visibly missing items. It has no test split, so I carved a seeded, stratified 100-image test split out of the training set and never touched it until the end. The rare classes are no_goggle and forklift. The brief asked for 30 images of my own: I collected 30 licensed Wikimedia Commons photos, 12 at night, and labelled the 91 workers by hand. That set exposed the main weakness: most training images look AI-generated, so real photos are harder.');

// 4 · Approach and architecture ------------------------------------------------------
s = pres.addSlide({ masterName: 'CONTENT' });
title(s, 'Approach: one pipeline, entirely in the browser', 'Static site on GitHub Pages (approved platform) · ONNX Runtime Web with WebGPU or WebAssembly');
const steps = [
  ['Frame', 'photo, clip frame (2–5 fps) or camera'],
  ['YOLO11n', 'fine-tuned, ONNX, 10.7 MB'],
  ['Matching', 'PPE box inside the person + body band'],
  ['Site profile', 'which items are required here'],
  ['Decision', 'green · red with missing items · blue "not sure"'],
  ['Output', 'faces blurred (YuNet), report, dashboard'],
];
steps.forEach(([h, d], i) => {
  const x = 0.6 + i * 2.05;
  card(s, x, 2.0, 1.85, 2.2, i === 4 ? C.raised : C.surface);
  s.addText(String(i + 1), { x: x + 0.2, y: 2.15, w: 0.5, h: 0.45, fontFace: HEAD, fontSize: 20, bold: true, color: C.hivis, margin: 0, isTextBox: true });
  s.addText(h, { x: x + 0.2, y: 2.65, w: 1.55, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: C.ink, margin: 0, isTextBox: true });
  s.addText(d, { x: x + 0.2, y: 3.1, w: 1.5, h: 1.0, fontFace: BODY, fontSize: 12, color: C.muted, margin: 0, valign: 'top', isTextBox: true });
  if (i < steps.length - 1) s.addText('›', { x: x + 1.83, y: 2.75, w: 0.25, h: 0.5, fontFace: HEAD, fontSize: 24, color: C.hivis, margin: 0, align: 'center', isTextBox: true });
});
card(s, 0.6, 4.55, 5.9, 2.15);
s.addText('Clips', { x: 0.85, y: 4.7, w: 5, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: C.ink, margin: 0, isTextBox: true });
bullets(s, ['IoU tracker gives each worker an anonymous number', 'A violation needs 3 of the last 5 frames (smoothing)', 'Danger zones and forklift proximity per worker'], 0.85, 5.15, 5.5, 1.5, 14);
card(s, 6.83, 4.55, 5.9, 2.15);
s.addText('Privacy rule', { x: 7.08, y: 4.7, w: 5, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: C.ink, margin: 0, isTextBox: true });
bullets(s, ['Detect on the original frame, held in memory', 'Blur faces only in what is shown or exported', 'No server: images never leave the device'], 7.08, 5.15, 5.5, 1.5, 14);
s.addNotes('[2:00-3:00] The pipeline runs entirely in the browser. A frame goes through my fine-tuned YOLO11n, exported to ONNX at 10.7 megabytes. Then the matching step: plain IoU fails because a helmet box is tiny next to a person box, so I use intersection over the PPE box plus a body band check, helmet near the top, boots near the bottom. A site profile decides which items are required, and per-class thresholds decide green, red or not sure. For clips, a simple IoU tracker keeps anonymous worker numbers and a violation must persist for three of five frames. Faces are blurred only in the output. The site is static on GitHub Pages, so there is no server that could see an image.');

// 5 · Baselines vs main model ---------------------------------------------------------
s = pres.addSlide({ masterName: 'CONTENT' });
title(s, 'Baselines vs my model (test split, 230 people)', 'Same person–PPE matching code for every method, so the comparison is at the level the officer sees');
s.addChart(pres.charts.BAR, [
  { name: 'Accuracy of decided people', labels: ['COCO person + colour rules', 'YOLO-World zero-shot', 'LFM3-Omni 1.7B (local VLM)', 'Fine-tuned YOLO11n'], values: [70.5, 87.5, 91.4, 94.9] },
  { name: 'Violators shown green', labels: ['COCO person + colour rules', 'YOLO-World zero-shot', 'LFM3-Omni 1.7B (local VLM)', 'Fine-tuned YOLO11n'], values: [20.0, 0, 13.7, 4.2] },
], {
  x: 0.6, y: 1.8, w: 7.6, h: 4.9, barDir: 'bar', barGrouping: 'clustered', chartColors: [C.ok, C.danger],
  showValue: true, dataLabelPosition: 'outEnd', dataLabelFontSize: 11, dataLabelColor: C.ink, dataLabelFormatCode: '0.0"%"',
  catAxisLabelColor: C.ink, catAxisLabelFontSize: 12, catAxisLabelFontFace: '+mn-lt', valAxisHidden: true, valAxisMaxVal: 110,
  valGridLine: { style: 'none' }, catGridLine: { style: 'none' }, showLegend: true, legendPos: 'b', legendColor: C.ink, legendFontSize: 12, legendFontFace: '+mn-lt',
  plotArea: { fill: { color: C.bg } },
});
card(s, 8.55, 1.8, 4.18, 4.9);
bullets(s, [
  'Colour rules: 70.5 % correct, 20 % of violators green',
  'YOLO-World cannot see absence: "no hard hat" prompts have recall 0, so it is unsure about 93 % of people',
  'LFM3-Omni finds more people on real photos but rarely says "no hi-vis": 14 % of violators green, about 30 s per photo',
  'Fine-tuned YOLO11n: 94.9 % correct, 4.2 % green, 16.1 % sent to "not sure", 0.1 s',
], 8.8, 2.05, 3.75, 4.5, 13);
s.addNotes('[3:00-4:00] I compared against three methods that need no PPE training, all fed through the same matching code. COCO people plus colour rules gets 70 percent right and shows one in five violators green. YOLO-World, zero-shot, knows what a hard hat is but cannot see absence; it is unsure about 93 percent of people. I also tried LFM3-Omni, a 1.7 billion parameter vision-language model, locally in Ollama: it finds people very well, but it almost always says hi-vis is worn, so 14 percent of violators go green, and it takes about 30 seconds a photo. My fine-tuned model is right 95 percent of the time on decided people, shows 4 percent of violators green, and takes a tenth of a second. Note the YOLO-World accuracy is over only the 7 percent of people it decides.');

// 6 · Generative AI experiment --------------------------------------------------------
s = pres.addSlide({ masterName: 'CONTENT' });
title(s, 'Generative AI: diffusion-made hard cases', 'SD-Turbo image-to-image at strength 0.3: night, rain, fog and dust versions with the original labels · CLIP + detector quality filter');
figure(s, FIG('fig_synthetic_with_without.png'), 0.6, 1.8, 7.7, 3.2);
stat(s, 0.6, 5.2, 2.45, '−0.3 pts', 'on real test images (CI −3.0 to +2.7)', C.muted);
stat(s, 3.22, 5.2, 2.45, '+14.5 pts', 'on weather versions (CI 8.6 to 20.9)', C.ok);
stat(s, 5.84, 5.2, 2.45, '35 → 75 %', 'no_boots recall, weather versions', C.hivis);
card(s, 8.6, 1.8, 4.13, 4.95);
bullets(s, [
  '339 of 400 diffusion images passed the quality filter',
  'Classical filters (control) gave +10.2 pts: most of the gain is from showing hard conditions at all',
  'Diffusion vs filters: +1.5 pts "no_" recall (not significant), +2.0 pts over all classes',
  'Goggle inpainting did not help no_goggle',
  'Honest result: no gain on ordinary real photos',
], 8.85, 2.05, 3.7, 4.6, 13);
s.addNotes('[4:00-5:15] The generative component targets the cases the data lacks: night, rain, fog, dust. SD-Turbo re-renders training images at low strength, so the boxes stay valid, and a CLIP plus detector filter kept 339 of 400. I retrained with the same seed and compared with paired bootstrap intervals. On the 100 real test photos: no measurable change, minus 0.3 points. On weather versions of those photos: plus 14.5 points of "no_" recall, and no_boots from 35 to 75 percent. A classical-filter control got plus 10.2, so most of the gain comes from showing hard conditions at all; diffusion adds a small, non-significant edge. Inpainting goggles off did not help. I report that plainly.');

// 7 · Demo: photo ---------------------------------------------------------------------
s = pres.addSlide({ masterName: 'CONTENT' });
title(s, 'Live demo 1: a photo check', 'Real Wikimedia Commons photos, never used for training');
shot(s, SHOT('03_photo_missing_helmets.png'), 0.6, 1.75, 8.1, 4.56);
card(s, 9.0, 1.75, 3.73, 4.95);
tag(s, 'Compliant', 9.3, 2.1, C.ok);
tag(s, 'PPE missing, with the items', 9.3, 2.5, C.danger);
tag(s, 'Not sure: check manually', 9.3, 2.9, C.unsure);
bullets(s, ['Compliance % and counts by PPE type', 'Hover a worker to see the matched boxes', 'Privacy slider, danger zones, Explain heat-map', 'PDF and CSV incident report, made on the device'], 9.3, 3.45, 3.3, 3.1, 13);
s.addNotes('[5:15-6:00] Demo. I open the photo page and pick "Hi-vis, no helmets" from the real-photo gallery: a group on a station platform in hi-vis jackets but no helmets. Every person gets a red dashed box listing what is missing, faces are blurred, and the panel shows zero percent compliance with counts by item. I hover a worker to show the matched boxes, drag the privacy slider, and export the PDF incident summary. All of it runs on this laptop in about a third of a second.');

// 8 · Demo: clip, zones -------------------------------------------------------------
s = pres.addSlide({ masterName: 'CONTENT' });
title(s, 'Live demo 2: a clip, danger zones and the failure case', '12-second clip analysed in about 9 s in the browser · 30 s at 3 fps in 13.2 s on two CPU threads');
shot(s, SHOT('10_clip_timeline.png'), 0.6, 1.75, 6.0, 3.38);
shot(s, SHOT('05_danger_zone.png'), 6.73, 1.75, 6.0, 3.38);
s.addText('Anonymous worker log, violation timeline and forklift alerts', { x: 0.6, y: 5.25, w: 6, h: 0.35, fontFace: BODY, fontSize: 13, color: C.muted, margin: 0, isTextBox: true });
s.addText('A drawn danger zone flags people whose feet are inside it', { x: 6.73, y: 5.25, w: 6, h: 0.35, fontFace: BODY, fontSize: 13, color: C.muted, margin: 0, isTextBox: true });
card(s, 0.6, 5.75, 12.13, 0.95, C.raised);
s.addText('Third case, a failure: "Night paving" misses a bent-over worker at the image edge and one in the paver\'s glare. The app says so on screen.',
  { x: 0.85, y: 5.85, w: 11.7, h: 0.75, fontFace: BODY, fontSize: 14, color: C.ink, margin: 0, valign: 'middle', isTextBox: true });
s.addNotes('[6:00-6:45] Second demo: a 12-second clip. Frames are sampled at 3 per second, each worker keeps an anonymous number, and the timeline shows when violations happen; this warehouse clip also raises forklift-proximity alerts. Then I draw a danger zone on a photo and the people standing in it turn into alerts. Third, a failure on purpose: at night the model misses a bent-over worker cut off by the edge and one in the paver lights. The app marks this as a known failure, because the result is a prompt for the officer, not a verdict.');

// 9 · Evaluation highlights ------------------------------------------------------------
s = pres.addSlide({ masterName: 'CONTENT' });
title(s, 'Evaluation highlights', 'Held-out test split, 100 images · 100-epoch YOLO11n trained on a free Colab T4');
stat(s, 0.6, 1.8, 2.9, '0.801', 'mAP@0.5 over 12 classes (0.593 at 0.5:0.95)');
stat(s, 3.67, 1.8, 2.9, '0.61', 'mean recall of the five "no_" classes; weakest no_goggle 0.37', C.hazard);
stat(s, 0.6, 3.55, 2.9, '0.93 / 0.94', 'precision / recall of "PPE missing" on decided people', C.ok);
stat(s, 3.67, 3.55, 2.9, '0.086 → 0.053', 'calibration error (ECE) with temperature 0.67', C.unsure);
card(s, 0.6, 5.3, 5.97, 1.4);
s.addText('Speed: 0.1 s model, about 0.3 s whole photo in the browser; 147 ms in Python on 2 CPU threads; int8 is 3.3 MB but lost 1.8 mAP points, so float ships.',
  { x: 0.85, y: 5.4, w: 5.5, h: 1.2, fontFace: BODY, fontSize: 13, color: C.ink, margin: 0, valign: 'middle', isTextBox: true });
figure(s, FIG('fig_confusion_matrix.png'), 6.9, 1.8, 5.83, 4.9);
s.addNotes('[6:45-7:45] The numbers. mAP at 0.5 is 0.80. The "no_" classes average 0.61 recall; no_goggle is weakest at 0.37 with only 35 test boxes. When the model calls PPE missing it is right 93 percent of the time and catches 94 percent of decided violators. Temperature scaling cut calibration error from 0.086 to 0.053, and that drives the not-sure state. The confusion matrix shows the expected look-alikes: boots against no_boots and helmet against no_helmet. Speed is far inside the five-second limit: a tenth of a second for the model.');

// 10 · Fairness and real photos ---------------------------------------------------------
s = pres.addSlide({ masterName: 'CONTENT' });
title(s, 'Bias check and real-world photos', 'Person recall by subgroup with 95 % bootstrap intervals; skin tone labelled by eye from head-and-shoulder crops');
figure(s, FIG('fig_subgroups.png'), 0.6, 1.8, 7.0, 4.9);
card(s, 7.9, 1.8, 4.83, 4.9);
bullets(s, [
  'Distance and darkness matter most: far 0.86 vs near 1.00; dark 0.74 vs normal 0.96',
  'Skin tone: violators caught 13 of 15 (lighter), 34 of 35 (medium), 8 of 8 (darker). No sign of a gap, but only 12 darker-skinned workers',
  'Clothing: dark clothes 0.87, hi-vis 1.00',
  'My 30 real photos: 84 % of decided right, 24 % sent to check, 2 of 19 violators green',
], 8.15, 2.05, 4.4, 4.6, 13);
s.addNotes('[7:45-8:30] The bias check. Distance and darkness are the big effects: far workers 86 percent detected against 100 for near ones, dark scenes 74 against 96. Automatic skin-tone grouping failed, so I labelled the test workers by eye; violators were caught 13 of 15, 34 of 35 and 8 of 8 across lighter, medium and darker skin, so no sign of a gap, but the darker group is only 12 people. On my own real photos the model is weaker: 84 percent of decided workers right, a quarter sent to check.');

// 11 · Failure cases ----------------------------------------------------------------------
s = pres.addSlide({ masterName: 'CONTENT' });
title(s, 'Where it fails, and what did not fix it', 'Failure gallery in the app; most failures are small, distant or dark');
shot(s, SHOT('07_failure_case.png'), 0.6, 1.75, 6.6, 3.71);
card(s, 7.5, 1.75, 5.23, 4.95);
bullets(s, [
  'Distant and crouching workers; night glare; people cut off by the edge',
  'Look-alikes: boots vs no_boots, helmet vs no_helmet',
  'Real hi-vis jackets: 2 of 11 workers without hi-vis shown wearing it',
  'Extra real data with pseudo-labels made the model doubt (41 % "not sure"); a Kaggle run with Hard Hat Workers and SH17 fixed false vests but called only 4 of 12 missing helmets. Neither shipped',
], 7.75, 2.0, 4.8, 4.6, 13);
card(s, 0.6, 5.65, 6.6, 1.05, C.raised);
s.addText('Bent-over workers score just under 0.25 as people, so the apps keep people from 0.20 as "check manually".',
  { x: 0.85, y: 5.72, w: 6.2, h: 0.9, fontFace: BODY, fontSize: 13, color: C.ink, margin: 0, valign: 'middle', isTextBox: true });
s.addNotes('[8:30-9:10] Failures. Most missed people are small, distant, bent over or in the dark. The look-alike pairs confuse it. I tried twice to fix the real-photo gap with more data. Pseudo-labelled extra photos taught the model that unlabelled PPE is background, so it sent 41 percent of workers to "not sure". A second run on Kaggle with fully labelled helmet data fixed the false vests but called only a third of missing helmets. Neither beat the original, so neither shipped. One small fix did ship: people are kept from a confidence of 0.20, so bent-over workers show up as "check manually".');

// 12 · Responsible AI -------------------------------------------------------------------
s = pres.addSlide({ masterName: 'CONTENT' });
title(s, 'Responsible AI by design', 'Model card, privacy, bias check and a human in the loop');
const rai = [
  ['Blur the output, not the input', 'Blurring faces before detection would cut goggles recall from 0.98 to 0.02 and mAP from 0.80 to 0.70'],
  ['Retain no data', 'No server; the session log lives in memory and is gone when the tab closes'],
  ['Aggregate only', 'Compliance % and counts; worker numbers are anonymous and reset every analysis'],
  ['Explain to workers', 'The live page asks the officer to say what the check is for before starting'],
  ['Human in the loop', '"Not sure" state, per-class thresholds, failure notes, and a written summary that is fact-checked against the detector'],
  ['Not for discipline', 'Every report says it supports the officer and must not be used against individuals'],
];
rai.forEach(([h, d], i) => {
  const x = 0.6 + (i % 3) * 4.08, y = 1.8 + Math.floor(i / 3) * 2.5;
  card(s, x, y, 3.88, 2.3);
  s.addText(h, { x: x + 0.25, y: y + 0.2, w: 3.4, h: 0.5, fontFace: HEAD, fontSize: 16, bold: true, color: C.hivis, margin: 0, isTextBox: true });
  s.addText(d, { x: x + 0.25, y: y + 0.75, w: 3.4, h: 1.4, fontFace: BODY, fontSize: 13, color: C.ink, margin: 0, valign: 'top', isTextBox: true });
});
s.addNotes('[9:10-9:40] Responsible AI is built in, not added on. The face-blur study turned a privacy instinct into a rule: if I blurred faces before detection, goggles recall would fall from 98 to 2 percent, so the app detects on the original in memory and blurs only what it shows. Nothing is stored, results are aggregate, the officer is asked to explain the check to workers, and uncertain cases go back to a person.');

// 13 · Limitations and next steps ------------------------------------------------------
s = pres.addSlide({ masterName: 'CONTENT' });
title(s, 'Limitations and next steps');
card(s, 0.6, 1.4, 5.97, 5.3);
s.addText('Limitations', { x: 0.85, y: 1.6, w: 5.5, h: 0.45, fontFace: HEAD, fontSize: 20, bold: true, color: C.danger, margin: 0, isTextBox: true });
bullets(s, [
  'Small, partly synthetic training set: distance and night are weak',
  'Own test set is small: 91 workers, 19 violators',
  'Smoothing study uses simulated clips; forklift proximity has no depth',
  'Skin-tone groups labelled by one person; darker group is small',
], 0.85, 2.25, 5.5, 4.3, 18);
card(s, 6.76, 1.4, 5.97, 5.3);
s.addText('Next steps', { x: 7.01, y: 1.6, w: 5.5, h: 0.45, fontFace: HEAD, fontSize: 20, bold: true, color: C.ok, margin: 0, isTextBox: true });
bullets(s, [
  'Hand-label a few hundred real site photos (hi-vis jackets, work boots, night)',
  'Use the vision-language model only as a second opinion: green when both agree halved violators shown green (4.2 → 2.1 %)',
  'Real labelled video for the tracker and smoothing',
  'A ControlNet fairness set with fixed poses and varied skin tones',
], 7.01, 2.25, 5.5, 4.3, 18);
s.addNotes('[9:40-10:00] The limits are mainly data: a small, partly synthetic training set, and a small real test set. The next step is labelled real photos, not a bigger model. The vision-language model is worth keeping as a second opinion: green only when both agree halved the violators shown green on the test split. Thank you; I am happy to take questions.');

// 14 · Close ----------------------------------------------------------------------------
s = pres.addSlide({ masterName: 'DARK' });
s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 13.33, h: 0.16, fill: { color: C.hivis }, line: { color: C.hivis } });
s.addText('Thank you. Questions?', { x: 0.7, y: 1.6, w: 11.9, h: 1.0, fontFace: HEAD, fontSize: 40, bold: true, color: C.ink, margin: 0, isTextBox: true });
s.addText([{ text: 'Live app (GitHub Pages)', options: { color: C.muted, breakLine: true } }, { text: LIVE, options: { color: C.hivis, hyperlink: { url: LIVE } } }],
  { x: 0.7, y: 3.0, w: 6, h: 0.9, fontFace: BODY, fontSize: 18, margin: 0, isTextBox: true });
s.addText([{ text: 'Mirror', options: { color: C.muted, breakLine: true } }, { text: 'https://sitesafe-phi.vercel.app', options: { color: C.ink } }],
  { x: 0.7, y: 4.05, w: 6, h: 0.9, fontFace: BODY, fontSize: 18, margin: 0, isTextBox: true });
s.addText([{ text: 'Code, model, results', options: { color: C.muted, breakLine: true } }, { text: REPO, options: { color: C.ink, hyperlink: { url: REPO } } }],
  { x: 0.7, y: 5.1, w: 6, h: 0.9, fontFace: BODY, fontSize: 18, margin: 0, isTextBox: true });
shot(s, SHOT('01_landing.png'), 7.2, 2.9, 5.5, 3.09);
s.addNotes('Q&A (5 minutes). Likely questions: why a static site instead of Streamlit (privacy: images never leave the device; a Streamlit version is in the repo); why diffusion did not help on real photos (they are daylight, where the data was not aimed); how the "not sure" thresholds were chosen (on the validation split, per class, for 80 % precision); what the vision-language model adds (a second opinion only).');

pres.writeFile({ fileName: path.join(ROOT, 'submission', 'SiteSafe_Presentation.pptx') }).then((f) => console.log('wrote', f));
