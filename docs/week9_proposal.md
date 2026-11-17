# Week 9 proposal: SiteSafe, PPE compliance with privacy

MAIB AI 221 Computer Vision with Applications. Project 6 of the brief.

## Problem and user

The user is a site safety officer on a construction site. During a walk-round they have to judge, for many workers at once, whether each one is wearing the protective equipment the area requires. They cannot watch everyone, and photographing workers to review later creates a privacy problem of its own.

SiteSafe takes a site photo, a 10 to 30 second clip or a live camera view and marks each person as compliant, missing PPE (with the missing item named) or "not sure, check manually". It reports a compliance percentage and counts by PPE type, and it blurs faces. The officer keeps the decision; the tool tells them where to look.

The two errors do not cost the same. A missed "no helmet" can end in a head injury, so it is the error I most want to avoid. A false alarm costs the officer a minute. For that reason I will report recall on the "no_" classes separately, and the app will prefer "not sure" over a wrong green.

## Data and licence

Construction-PPE by Ultralytics, licence CC BY 4.0: 1,000 images, 12 classes (helmet, gloves, vest, boots, goggles, Person, forklift, and no_helmet, no_goggle, no_gloves, no_boots, no_vest). It has train (800) and val (200) but no test split, so I carve a fixed, seeded, stratified 100-image test split out of train and never train on it. I will also collect 30 images of my own from public, permitted sources, label them, and tag them by lighting, distance, clothing colour and apparent skin-tone group for the bias check.

## Method

The main model is YOLO11n fine-tuned on the dataset and exported to ONNX (about 10 MB). PPE boxes are matched to person boxes by overlap and position. Clips use frame sampling, a simple tracker with anonymous worker numbers, and temporal smoothing. Faces are detected with YuNet and blurred.

I compare against two baselines: a COCO-pretrained YOLO11n person detector with HSV colour rules for helmet and vest, and a zero-shot open-vocabulary detector (YOLO-World) prompted with the class names.

The generative AI component is a diffusion model (SD-Turbo, image to image at low strength) that turns real labelled training images into night, rain, fog and dust versions. The geometry does not change, so the labels stay valid. I train with and without the synthetic images, same seed and epochs, and measure the change in recall for each "no_" class on real test images.

## Metrics

mAP@0.5 and mAP@0.5:0.95 per class, with the focus on recall for the "no_" classes. Person-level compliance accuracy, precision and recall. With versus without synthetic data. Effect of face blurring on detection. Calibration (ECE) and the share of people sent to "not sure". Subgroup recall with bootstrap confidence intervals. Time per image and per clip on the deployed app. A gallery of at least ten real failures.

## Deployment plan

All models are open and run inside the app. Training uses only Colab free tier and my laptop. No paid service is used anywhere.

The main app is a static web app (React) that runs the ONNX model in the user's browser with ONNX Runtime Web, so images never leave the device. The same model also runs in a Streamlit app on Streamlit Community Cloud, which is on the approved list and is my fallback.

**One point for your approval.** I would like to host the static build on Vercel's free Hobby plan (no card, no serverless functions, static files only). Vercel is not on the approved list, and it is not on the excluded list either. The identical build also deploys to Cloudflare Pages and GitHub Pages, which are approved, with no code change. If you prefer, I will submit the Cloudflare Pages URL as the official one. Please let me know which you want.

## Risks

Accuracy on the rare classes (no_goggle has 35 boxes in my test split) is the main risk, and the reason for the generative experiment and the "not sure" state. Speed is not a risk: the model answers in well under a second on a laptop CPU.
