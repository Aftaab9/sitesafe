# Model card: SiteSafe PPE detector

**Model.** YOLO11n (2.6 M parameters) fine-tuned from COCO weights; ONNX opset 17, 640 × 640 input, 10.7 MB (float32). Face detection: YuNet (MIT). Version 1.0, November 2026. Code licence AGPL-3.0.

**Intended use.** Helping a construction site safety officer find workers who are missing required PPE (helmet, hi-vis vest, gloves, boots, goggles) in photos, 10 to 30 second clips and live camera video, and reporting aggregate compliance for a site or shift. The output is a prompt to look, with a "check manually" state for uncertain cases.

**Out of scope.** Monitoring, scoring or disciplining individual workers. Identifying people (there is no face recognition; faces are blurred). Signing off a site as safe or replacing an inspection. Covert use, or use without telling workers what the check is for. Measuring real distances (forklift proximity has no depth).

**Data.** Construction-PPE (Ultralytics, CC BY 4.0): 1,000 images, 12 classes including five "no_" classes for visibly missing PPE. Seeded, stratified split: 700 train, 200 validation (model selection, thresholds, calibration), 100 test (final numbers only). Mostly daylight scenes; night, bad weather and very distant workers are under-represented.

**Metrics (held-out test split, 100 images, 230 people; 100-epoch model trained on a free Colab T4).**

| | |
|---|---|
| mAP@0.5 / mAP@0.5:0.95 | 0.801 / 0.593 |
| Recall: no_boots / no_vest / no_gloves / no_helmet / no_goggle | 0.77 / 0.70 / 0.70 / 0.51 / 0.37 |
| Person detection recall | 0.939 |
| Compliance accuracy (decided people) | 0.949 (colour-rules baseline 0.705) |
| "PPE missing": precision / recall (decided) | 0.93 / 0.94 |
| Violators shown compliant | 4.2 % (4 of 95) |
| Sent to "check manually" | 16.1 % |
| Calibration error (ECE), raw → temperature-scaled | 0.086 → 0.053 |
| Blurring faces before detection | mAP 0.801 → 0.700, goggles recall 0.98 → 0.02 (so the app detects first, then blurs the output) |
| My own 30 real photos (91 workers) | accuracy 0.84 of decided, 24 % "check manually", 2 of 19 violators shown compliant |

**Subgroups (person detection recall, 95 % bootstrap interval).** Near 1.00; mid 0.97 (0.91–1.00); far 0.86 (0.77–0.93). Normal light 0.96 (0.92–0.98); dark 0.74 (0.53–0.89, only 19 people). Hi-vis clothing 1.00; dark clothing 0.87 (0.74–0.97). Violators caught or sent to check: near 1.00, far 0.75. Skin-tone results are withheld: the automatic colour-based grouping was unreliable, and the tags must be checked by hand first.

**Known failures.** Small, distant workers (most missed people and most violators shown green). Dark scenes. On real photos outside the dataset: hi-vis jackets missed or vests seen that are not there (18 % of workers without hi-vis shown as wearing it), and some work boots called "no boots". Workers cut off by the image edge or heavily overlapping. Look-alike pairs: boots vs no_boots, helmet vs no_helmet, gloves vs no_gloves. Rare classes (no_goggle) have few test examples and wide uncertainty. A version fine-tuned on 3,465 extra real photos with pseudo-labels showed no worker without hi-vis as wearing it, but sent 41 % of workers on my photos to "check manually" and rarely called a missing helmet, so it is not shipped.

**Privacy and human oversight.** Inference runs in the user's browser; no image is uploaded or stored. Faces are blurred before display or export. Results are aggregate, and worker numbers are anonymous and reset per analysis. Low-confidence decisions go to "check manually". Reports state that the tool supports and does not replace the safety officer.

**Generative components.** Diffusion-generated night, rain, fog and dust versions of training images (SD-Turbo, offline, label-preserving) are tested as extra training data: no change on real daylight photos (−0.3 points of "no_" recall), +14.5 points on weather versions of the test photos, about 1.5 to 2 points better than cheap image filters. An optional in-browser SmolVLM-256M summary is checked claim by claim against the detector, and claims it cannot verify are corrected or removed.
