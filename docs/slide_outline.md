# Presentation outline (10 minutes, 11 slides)

Numbers are from the held-out test split (100 real images, 230 people) unless stated. Keep them identical to the report.

1. **SiteSafe: see who is missing PPE without seeing who they are** (0:30). Title, live URL as a QR code, one hero image with boxes and blurred faces.

2. **The user and the decision** (0:45). Site safety officer on a walk-round. Decision: who needs to be stopped and kitted out. Cost of errors: a missed "no helmet" can mean a head injury; a false alarm costs a minute. Therefore: no_ recall first, and a "not sure" state instead of a wrong green.

3. **Data** (0:45). Construction-PPE, CC BY 4.0, 1,000 images, 12 classes, five of them "no_". No test split in the dataset, so a seeded, stratified 100-image test split carved from train. Bar chart of class counts: no_goggle (35 test boxes) and forklift (37) are the rarest.

4. **Method** (1:00). Pipeline diagram: frame → YOLO11n (ONNX, 10.7 MB) → person–PPE matching (intersection over the PPE box + position bands) → site profile → compliant / missing / not sure → tracker and 3-of-5 smoothing for clips → face blur on the output only → report. All in the browser.

5. **Baselines vs my model** (1:00). Table: person-level accuracy 70.5 % (COCO + colour rules), zero-shot YOLO-World abstains on 93 % of people, fine-tuned 95.4 %. Precision for "PPE missing" 0.60 vs 1.00. Violators shown green 20 % vs 8 %.

6. **Generative AI: synthetic hard cases** (1:15). Before/after images (night, rain, fog, dust). Label-preserving img2img, quality filter (kept 165 of 200 locally). Result table: no_ recall with vs without, with 95 % intervals, on real test images and on weather versions of them. Say plainly what it did and did not change.

7. **Evaluation highlights** (1:00). mAP@0.5 0.796; no_ mean recall 0.58; weakest no_goggle 0.31, no_helmet 0.48. Confusion: boots vs no_boots, helmet vs no_helmet. Calibration: ECE 0.090 → 0.059 with temperature 0.67. 18.8 % of people sent to "check manually".

8. **Privacy: does blurring hurt?** (0:45). Blurring faces before detection: goggles recall 0.96 → 0.02, mAP 0.80 → 0.68. Blurring heads: helmet recall 0.76 → 0.08. Decision: detect on the original in memory, blur only the output. Images never leave the device.

9. **Fairness and failures** (1:00). Subgroup recall with intervals: dark scenes 0.74 vs normal light 0.97; far workers 0.88 vs near 1.00. Skin-tone results withheld until tags are hand-checked (automatic grouping failed). Three failure images with one line each.

10. **Live demo** (1:30). Sample photo → hover a worker → privacy slider → danger zone → clip timeline → PDF. Phone live view if time allows.

11. **Limits and responsible use** (0:30). Distance, crowds, night. Proximity without depth. Not for individual monitoring or discipline. Speed: about 115 ms per photo with WebGPU, 156 ms on two CPU threads; a 30 s clip at 3 fps in about 14 s.
