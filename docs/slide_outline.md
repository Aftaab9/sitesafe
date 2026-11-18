# Presentation outline (10 minutes, 11 slides)

Numbers are from the held-out test split (100 real images, 230 people) unless stated. Keep them identical to the report.

1. **SiteSafe: see who is missing PPE without seeing who they are** (0:30). Title, live URL (https://sitesafe-phi.vercel.app) as a QR code, one hero image with boxes and blurred faces.

2. **The user and the decision** (0:45). Site safety officer on a walk-round. Decision: who needs to be stopped and kitted out. Cost of errors: a missed "no helmet" can mean a head injury; a false alarm costs a minute. Therefore: no_ recall first, and a "not sure" state instead of a wrong green.

3. **Data** (0:45). Construction-PPE, CC BY 4.0, 1,000 images, 12 classes, five of them "no_". No test split in the dataset, so a seeded, stratified 100-image test split carved from train. Bar chart of class counts: no_goggle (35 test boxes) and forklift (37) are the rarest. About 570 of 800 training images look AI-generated: the reason for the extra real-photo data.

4. **Method** (1:00). Pipeline diagram: frame → YOLO11n (ONNX, 10.7 MB) → person–PPE matching (intersection over the PPE box + position bands) → site profile → compliant / missing / not sure → tracker and 3-of-5 smoothing for clips → face blur on the output only → report. All in the browser.

5. **Baselines vs my model** (1:00). Table: person-level accuracy 70.5 % (COCO + colour rules), zero-shot YOLO-World abstains on 93 % of people, fine-tuned 94.9 %. Precision for "PPE missing" 0.60 vs 0.93. Violators shown green 20 % vs 4 %.

6. **Generative AI: synthetic hard cases** (1:15). Before/after images (night, rain, fog, dust). Label-preserving img2img, quality filter (kept 339 of 400 on Colab). Result: real test −0.3 points (−3.0 to +2.7); weather versions +14.5 (8.6 to 20.9), no_boots 35 → 75 %; classical filters +10.2; diffusion vs filters +1.5 (not significant) / +2.0 all classes. Inpainting no help. Say plainly what it did and did not change.

7. **Evaluation highlights** (1:00). mAP@0.5 0.801; no_ mean recall 0.61; weakest no_goggle 0.37, no_helmet 0.51. Confusion: boots vs no_boots, helmet vs no_helmet. Calibration: ECE 0.086 → 0.053 with temperature 0.67. 16.1 % of people sent to "check manually".

8. **Privacy: does blurring hurt?** (0:45). Blurring faces before detection: goggles recall 0.98 → 0.02, mAP 0.80 → 0.70. Strict face-area blur: helmet recall 0.77 → 0.58. Decision: detect on the original in memory, blur only the output. Images never leave the device.

9. **Fairness and failures** (1:00). Subgroup recall with intervals: dark scenes 0.74 vs normal light 0.96; far workers 0.86 vs near 1.00. My own 30 real photos: 84 % of decided correct, 24 % to check. Extra-real-data model: fewer wrong greens but 41 % to check, so not shipped. Skin-tone results withheld until tags are hand-checked (automatic grouping failed). Three failure images with one line each.

10. **Live demo** (1:30). Sample photo → hover a worker → privacy slider → danger zone → clip timeline → PDF. Phone live view if time allows.

11. **Limits and responsible use** (0:30). Distance, crowds, night. Proximity without depth. Not for individual monitoring or discipline. Speed: about 115 ms per photo with WebGPU, 147 ms on two CPU threads; a 30 s clip at 3 fps in about 13 s.
