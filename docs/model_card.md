# Model card: SiteSafe PPE detector

**Model.** YOLO11n (2.6 M parameters) fine-tuned from COCO weights; ONNX opset 17, 640 × 640 input, 10.7 MB (float32). Face detection: YuNet (MIT). Version 1.0, November 2026. Code licence AGPL-3.0.

**Intended use.** Helping a construction site safety officer find workers who are missing required PPE (helmet, hi-vis vest, gloves, boots, goggles) in photos, 10 to 30 second clips and live camera video, and reporting aggregate compliance for a site or shift. The output is a prompt to look, with a "check manually" state for uncertain cases.

**Out of scope.** Monitoring, scoring or disciplining individual workers. Identifying people (there is no face recognition; faces are blurred). Signing off a site as safe or replacing an inspection. Covert use, or use without telling workers what the check is for. Measuring real distances (forklift proximity has no depth).

**Data.** Construction-PPE (Ultralytics, CC BY 4.0): 1,000 images, 12 classes including five "no_" classes for visibly missing PPE. Seeded, stratified split: 700 train, 200 validation (model selection, thresholds, calibration), 100 test (final numbers only). Mostly daylight scenes; night, bad weather and very distant workers are under-represented.

**Metrics (held-out test split, 100 images, 230 people).**

| | |
|---|---|
| mAP@0.5 / mAP@0.5:0.95 | 0.796 / 0.584 |
| Recall: no_boots / no_vest / no_gloves / no_helmet / no_goggle | 0.74 / 0.70 / 0.66 / 0.48 / 0.31 |
| Person detection recall | 0.948 |
| Compliance accuracy (decided people) | 0.954 (colour-rules baseline 0.705) |
| "PPE missing": precision / recall (decided) | 1.00 / 0.88 |
| Violators shown compliant | 8.4 % |
| Sent to "check manually" | 18.8 % |
| Calibration error (ECE), raw → temperature-scaled | 0.090 → 0.059 |
| Blurring faces before detection | mAP 0.796 → 0.677, goggles recall 0.96 → 0.02 (so the app detects first, then blurs the output) |

**Subgroups (person detection recall, 95 % bootstrap interval).** Near 1.00; mid 0.97 (0.91–1.00); far 0.88 (0.81–0.94). Normal light 0.97 (0.94–0.99); dark 0.74 (0.53–0.89, only 19 people). Hi-vis clothing 1.00; dark clothing 0.90 (0.77–1.00). Violators caught or sent to check: near 0.98, far 0.70. Skin-tone results are withheld: the automatic colour-based grouping was unreliable, and the tags must be checked by hand first.

**Known failures.** Small, distant workers (most missed people and most violators shown green). Dark scenes. Workers cut off by the image edge or heavily overlapping. Look-alike pairs: boots vs no_boots, helmet vs no_helmet, gloves vs no_gloves. Rare classes (no_goggle) have few test examples and wide uncertainty.

**Privacy and human oversight.** Inference runs in the user's browser; no image is uploaded or stored. Faces are blurred before display or export. Results are aggregate, and worker numbers are anonymous and reset per analysis. Low-confidence decisions go to "check manually". Reports state that the tool supports and does not replace the safety officer.

**Generative components.** Diffusion-generated night, rain, fog and dust versions of training images (SD-Turbo, offline, label-preserving) are tested as extra training data; the effect is reported in the technical report with confidence intervals. An optional in-browser SmolVLM-256M summary is checked claim by claim against the detector, and claims it cannot verify are corrected or removed.
