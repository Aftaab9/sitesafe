# Model card: SiteSafe PPE detector

**Model.** YOLO11n (2.6 M parameters) fine-tuned from COCO weights for 100 epochs on a free Colab T4; ONNX opset 17, 640 × 640 input, 10.7 MB (float32). Faces: YuNet (MIT). Version 1.0, November 2026, AGPL-3.0. Live app: https://aftaab9.github.io/sitesafe/ (mirror https://sitesafe-phi.vercel.app); code: https://github.com/Aftaab9/sitesafe.

**Intended use.** Helping a construction-site safety officer find workers who are visibly missing required PPE (helmet, hi-vis vest, gloves, boots, goggles) in photos, 10 to 30 second clips and live camera video, and reporting aggregate compliance for a site or shift. The output is a prompt to look, with a "check manually" state for uncertain cases.

**Out-of-scope use.** Monitoring, scoring or disciplining individual workers. Identifying people (no face recognition; faces are blurred). Signing off a site as safe or replacing an inspection. Covert use. Measuring real distances (forklift proximity has no depth).

**Data.** Construction-PPE (Ultralytics, CC BY 4.0): 1,000 images, 12 classes including five "no_" classes. Seeded split: 700 train, 200 validation (selection, thresholds, calibration), 100 test (final numbers only). About 570 of the 800 training images look AI-generated; night, bad weather and distant workers are rare. Own test set: 30 Wikimedia Commons photos (CC0, public domain, CC BY, CC BY-SA), 91 workers labelled by hand.

**Metrics (test split, 100 images, 230 people).**

| | |
|---|---|
| mAP@0.5 / mAP@0.5:0.95 | 0.801 / 0.593 |
| Recall no_boots / no_vest / no_gloves / no_helmet / no_goggle | 0.77 / 0.70 / 0.70 / 0.51 / 0.37 |
| Compliance accuracy, decided people (colour-rules baseline) | 0.949 (0.705) |
| "PPE missing" precision / recall | 0.93 / 0.94 |
| Violators shown compliant / sent to "check manually" | 4.2 % / 16.1 % |
| Calibration error (ECE), raw → temperature-scaled | 0.086 → 0.053 |
| Own 30 real photos | 0.84 of decided correct, 24 % to check, 2 of 19 violators shown compliant |
| Speed | 0.1 s model, about 0.3 s whole photo in the browser; 30 s clip at 3 fps in 13 s on CPU |

**Subgroup results (person recall, 95 % interval).** Near 1.00, far 0.86 (0.77–0.93). Normal light 0.96, dark 0.74 (0.53–0.89, 19 people). Hi-vis clothing 1.00, dark clothing 0.87. Skin tone, labelled by eye: lighter 0.98, medium 1.00, darker 1.00 (12 people only); violators caught or sent to check 13 of 15, 34 of 35 and 8 of 8. No sign of worse results on darker skin, but the group is small. The weak group is "skin not visible" (far or turned away): 0.84, violators caught 0.75.

**Known failures.** Small, distant and crouching workers; dark scenes; workers cut off by the image edge or overlapping. Look-alike pairs: boots vs no_boots, helmet vs no_helmet. On real photos: hi-vis jackets missed or vests seen that are not there (2 of 11 workers without hi-vis shown wearing it). Blurring faces before detection would cut goggles recall from 0.98 to 0.02, so the app detects first and blurs only the output. Diffusion training data (SD-Turbo) did not change "no_" recall on real photos (−0.3 points) but raised it by 14.5 points on weather versions. A small vision-language model (LFM3-Omni, optional, local) finds more people but shows 14 % of violators compliant; it is offered only as a second opinion.

**Privacy and oversight.** Everything runs in the user's browser; no image is uploaded or stored. Results are aggregate, worker numbers are anonymous and reset per analysis, and low-confidence decisions go to the officer.
