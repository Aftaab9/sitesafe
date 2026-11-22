# SiteSafe: PPE Compliance with Privacy

MAIB AI 221 Computer Vision with Applications, Project 6. Technical report.

Live app: https://aftaab9.github.io/sitesafe/ (mirror: https://sitesafe-phi.vercel.app) · Code: https://github.com/Aftaab9/sitesafe

## 1. Problem

On a construction site, the safety officer has to notice that a worker has taken off a hard hat, or that the new labourer on the scaffold has no gloves. They cannot watch every work area at once, and the obvious fix, cameras that record workers, creates a second problem: identifiable images of people at work, which workers rightly distrust.

I built SiteSafe for that officer. It takes a site photo, a short clip of 10 to 30 seconds, or a live camera view, and marks each person in one of three ways: compliant, missing required PPE (and which item), or "not sure, check manually". It reports a compliance percentage and counts by PPE type, flags people inside a danger zone or near a forklift, and blurs faces. In the main app nothing leaves the user's device.

The decision it supports: which workers should the officer go and speak to now. A missed "no helmet" is the expensive mistake, because it can end in a head injury; a false alarm costs a short walk. So I put recall on the "no_" classes ahead of overall accuracy, and added a "not sure" state so the model can hand a hard case back instead of guessing green.

## 2. Data

I used Construction-PPE, published by Ultralytics under CC BY 4.0. It has 1,000 photographs of construction and warehouse scenes with boxes for twelve classes: helmet, gloves, vest, boots, goggles, Person, forklift, and five "absence" classes, no_helmet, no_goggle, no_gloves, no_boots and no_vest. The "no_" classes make the dataset fit this problem: a detector that only knows "helmet" cannot tell a bare head from a head it failed to see.

The dataset ships with 800 training and 200 validation images and no test split. Reporting final numbers on the validation images would be optimistic, because I use them to pick the best epoch and to set thresholds. I therefore carved a fixed, seeded, stratified test split of 100 images out of the training set, topping up whichever class was furthest below its share so every class is present. That leaves 700 training, 200 validation and 100 test images; the test images were used once, for the final numbers, and the Colab notebook checks it rebuilds exactly the same split.

The class balance is uneven in the way that matters. In the test split there are 230 labelled people and 241 boots, but only 35 no_goggle boxes and 37 forklifts. Most photos are daylight scenes; very few are at night or in rain, fog or dust, the gap the generative part targets.

The brief also asks for 30 images of my own from public, permitted sources. A script took them from Wikimedia Commons, keeping only CC0, public-domain, CC BY and CC BY-SA files with author and licence (`data/own_test/sources.csv`). From 113 candidates I chose 30 that differ from the training data: 12 at night or in tunnels, distant workers, crowds, hi-vis jackets instead of vests, and people with no PPE. I labelled the 91 workers in them by eye at person level: helmet, hi-vis and boots each worn, missing or not visible. This set exposed a weakness (section 5): about 570 of the 800 training images look AI-generated, and "no_boots" mostly means canvas trainers.

![Figure 1. Labelled boxes per class in Construction-PPE.](submission/figures/fig_class_counts.png)

## 3. Methods

### 3.1 Baselines

I compared my model with two baselines that need no PPE training. The first is a COCO-trained YOLO11n that finds people, plus HSV colour rules: the head region is checked for typical hard-hat colours and the torso for high-visibility yellow-green or orange. The second is zero-shot: YOLO-World, an open-vocabulary detector, prompted with a phrase per class, such as "hard hat" or "bare head without hard hat". Both baselines feed the same person–PPE matching code as my model, so the comparison is at the level the officer sees: is this person compliant?

### 3.2 The detector

The main model is YOLO11n (2.6 million parameters), fine-tuned from COCO weights on the 700 training images at 640 pixels with a fixed seed. A first 30-epoch run on my laptop CPU took seventeen hours; the final models were trained for 100 epochs on a free Colab T4 GPU in about twenty minutes each, from a notebook that resumes after a disconnect. I also trained YOLO11s for comparison. The best checkpoint was exported to ONNX (opset 17, static input, no NMS in the graph) at 10.7 MB. I do the non-maximum suppression in my own code in both Python and TypeScript, so the browser and the Streamlit app make identical decisions.

### 3.3 Matching PPE to people

The matching step assigns each PPE or "no_" box to at most one person. Plain IoU does not work, because a helmet box is tiny compared with a person box, so even a perfect match has an IoU near zero. I use intersection over the PPE box's own area instead. A box must be at least half inside a person, and its centre must fall in the band of the body where that item is worn: helmet and goggles in the top 30 percent, vest across the torso, boots in the bottom 20 percent. The band check is skipped for people who are wider than tall, such as someone crouching. When a box overlaps two people, it goes to the one whose centre is closer.

If both "helmet" and "no_helmet" are matched to one person, the more confident box wins. If the two are within 0.10 of each other, or the winner is below its class threshold, the item is "unsure". A person is compliant only if every item required by the site profile is positively seen; non-compliant if a required item is missing; otherwise "check manually". Profiles are configurable: a general area needs helmet and vest, full PPE adds gloves and boots, and cutting or grinding adds goggles. Twelve fixture cases (a helmet lying at someone's feet, two people side by side, a close conflict) run against both the Python and the TypeScript implementation, so the two apps cannot drift apart.

### 3.4 Calibration and the "not sure" state

On the validation split I marked every detection as correct or wrong and fitted a single temperature to the confidence scores. It came out at 0.67, so the raw scores were slightly under-confident, and it reduced the expected calibration error on the test split from 0.086 to 0.053. I then set a per-class threshold: the lowest confidence at which the class reaches 80 percent precision on validation. For most classes this is 0.30; for no_goggle it is 0.70, so a no_goggle detection must be strong before it can turn a person red.

### 3.5 Clips, danger zones and privacy

Clips are sampled at 2, 3 or 5 frames per second. A ByteTrack-style tracker links people across frames with a constant-velocity prediction, matching confident detections first and using weak ones only to keep tracks alive. Each track gets an anonymous number, and no appearance features are stored, so the tracker cannot recognise someone who leaves and comes back. A violation is raised only when an item has been missing in three of the last five sampled frames.

Danger zones are polygons drawn by the user. A person is inside when their foot point, the bottom centre of the box, is inside. Forklift proximity has no depth, so I measure the gap between boxes in units of the person's height, and present it as a prompt to look rather than a measurement.

Faces are found with YuNet (0.2 MB) and blurred by shrinking each region to a few pixels and enlarging it before a Gaussian blur, which destroys detail rather than hiding it. In strict mode, the face area of anyone whose face was not found is blurred too. My first version of that area covered the whole head, which hid the hard hat, so I moved it below the top of the box. The main app is static: the model runs in the browser with ONNX Runtime Web, on WebGPU or multi-threaded WebAssembly, inside a Web Worker. There is no server to receive an image. The Streamlit fallback processes uploads in memory and keeps nothing.

![Figure 2. Photo check on a real Wikimedia Commons photo: four compliant workers, faces blurred, credit under the photo.](submission/screenshots/02_photo_compliant_crew.png)

![Figure 3. The privacy slider: original frame on one side, blurred output on the other.](submission/screenshots/04_face_blur_before_after.png)

### 3.6 The generative AI component

The data has almost no photographs at night or in rain, fog or dust, and few of missing goggles: exactly where a missed violation is most likely. I used SD-Turbo, an open diffusion model, in image-to-image mode at a low strength of 0.3. Each image first gets a cheap classical hint (darkening, streaks, haze or a sand tint), which the diffusion model re-renders as a photograph. At this strength the scene does not move, so the original boxes stay valid with no hand-labelling. A quality filter keeps an image only if the real-data model still finds at least half the labelled people, CLIP judges it the same scene, and CLIP agrees the condition was applied. On Colab 339 of 400 images passed; most rejects were fog or dust that had erased distant workers.

The experiment trains a second model on the real images plus the kept synthetic ones, with the same seed, epochs and settings, and compares recall per class on the real test images, with a paired bootstrap over images for 95 percent intervals. The Colab notebook adds an arm trained with the classical filters only, which asks whether diffusion adds anything over a few lines of OpenCV, and an arm in which goggles are removed by inpainting and relabelled as no_goggle.

The second generative piece is in the app. An optional button loads SmolVLM-256M in the browser, gives it the face-blurred photo and asks for fixed JSON: number of workers, number without each item, whether a forklift is present, and one sentence about the scene. Every number and yes/no claim is checked against the detector. Matching claims are kept, wrong ones are replaced by the detector's value and marked as corrected, and a scene sentence that restates counts or PPE facts is dropped. The language model may describe; it is never the source of a safety fact.

![Figure 4. Weather Lab in the app: diffusion-made night, rain, fog and dust versions scored by both detectors.](submission/screenshots/12_weather_lab.png)

## 4. Results

All numbers in this section are on the 100 held-out test images unless I say otherwise.

### 4.1 Detection

The fine-tuned YOLO11n reaches a mAP@0.5 of 0.801 and a mAP@0.5:0.95 of 0.593 across the twelve classes (0.765 and 0.540 on validation). The easy classes are where one would expect: Person (recall 0.92), goggles (0.97), vest (0.88). The "no_" classes are harder, with a mean recall of 0.61: no_boots 0.77, no_vest 0.70, no_gloves 0.70, no_helmet 0.51 and no_goggle 0.37. Precision on the "no_" classes is between 0.70 and 0.95, so when the model says something is missing it is usually right; the problem is what it does not say. YOLO11s reached 0.829 mAP@0.5 and 0.646 "no_" recall, but at 38 MB and 2.4 times the CPU time, and it found fewer people on my own photos (75 against 86 percent), so the small model stays.

The confusion matrix shows where the errors go. The largest confusion is between boots and no_boots (fourteen no_boots called boots, seven the other way), then vest and no_vest, gloves and no_gloves, and helmet and no_helmet in both directions. These pairs share shape and position and differ in a detail a few pixels wide. Most remaining errors are small boxes not detected at all: 32 boots, 22 no_helmet and 14 no_goggle.

![Figure 5. Training curves of the 100-epoch models, with and without diffusion data.](submission/figures/fig_training_curves.png)

![Figure 6. Normalised confusion matrix on the test split.](submission/figures/fig_confusion_matrix.png)

### 4.2 Person-level compliance and the baselines

Of the 230 labelled people, 223 can be judged from the labels; 95 of them are non-compliant under the general profile. The fine-tuned model detects 93.9 percent of people. It decides 177 of the 223 and sends the rest to "check manually" (16.1 percent) or misses them (4.5 percent). Among decided people it is right 94.9 percent of the time, with a precision of 0.93 and a recall of 0.94 for "PPE missing". Of all 95 violators, 88 percent were flagged or handed to the officer, and 4 (4.2 percent) were shown green, the figure I care most about.

The colour-rules baseline detects fewer people (86.5 percent). Its accuracy on decided people is 70.5 percent, its precision for "PPE missing" 0.60, and it shows 20 percent of violators as green. Colour alone cannot separate a hard hat from a pale wall behind a bare head, or a vest from orange brickwork.

The zero-shot baseline finds people very well (95 percent) and hard hats reasonably (0.69), but almost never finds a vest from the prompt "safety vest" (0.05), and it cannot detect absence at all: "bare head without hard hat" and "person without safety vest" have recall zero. Through the matching code it is unsure about 93 percent of people. An open-vocabulary model knows what things are, not what is missing. I also tested LFM3-Omni, a 1.7-billion-parameter vision-language model run locally in Ollama, asked for every person with a box and whether they wear a hard hat and hi-vis. On my 30 photos it found 99 percent of workers against 86, called 92 percent of missing helmets and never said "not sure", but it showed 6 of 19 violators green against 2, because it almost always answers that hi-vis is worn. On the test split it was right on 91 percent against 95, showed 14 percent of violators green against 4, and took about 30 seconds per photo against 0.1. Used as a second opinion, with green only when both agree, it halved violators shown green on the test split (4.2 to 2.1 percent). The app offers both modes. An open YOLOv8s trained by others on about 13,000 real construction photos was no better than my model on my own photos (section 5).

![Figure 7. Person-level results of the baselines and the fine-tuned model on the test split.](submission/figures/fig_baselines.png)

### 4.3 Calibration

After temperature scaling the reliability curve (`results/eval/reliability.png`) sits close to the diagonal. The thresholds send 16.1 percent of people to "check manually": that costs the officer some time, and it is where the borderline cases go instead of into a wrong green.

![Figure 8. Reliability diagram before and after temperature scaling.](submission/figures/fig_reliability.png)

### 4.4 The generative experiment

All arms were trained on Colab with the same seed and 100 epochs. On the 100 real test photos the diffusion data made no measurable difference: pooled "no_" recall changed by −0.3 points (95 percent interval −3.0 to +2.7). Those photos are almost all clear daylight, which is not where the data was aimed. On weather versions of the same photos, pooled "no_" recall rose from 34.5 to 48.9 percent (+14.5 points, interval 8.6 to 20.9), and no_boots from 35 to 75 percent. The classical-filter control gained +10.2 points (6.0 to 14.8). Compared directly, diffusion beat the filters by 1.5 points of "no_" recall (interval −2.5 to +5.8, not significant) and 2.0 points over all classes (0.4 to 3.9), mostly on boots. So most of the gain comes from showing the model hard conditions at all, and diffusion adds a small extra; the stress test also flatters both, because the same pipeline made the training images. Goggle inpainting did not help the class it targeted (no_goggle −8.6 points, interval −20.6 to +3.1). On my own real photos both models caught 79 percent of violators, but the diffusion model called fewer missing helmets (58 against 82 percent) and showed one more violator green, so I found no real-world gain.

The verified narrative was evaluated on 40 face-blurred test photos, scoring each claim against the labels. Only 14 of 40 answers were usable JSON, and most invented a list of workers with names and ages ("JOHN, 25"), a privacy failure as well as a factual one. Of its claims, 68 percent were wrong. After verification every published number comes from the detector, and the error on the same claims fell to 32 percent, the detector's own error on exact counts in crowded scenes. The verifier is what makes it usable, and it stays an optional button.

![Figure 9. Recall of the "no_" classes with and without diffusion data, on real test images and on their weather versions.](submission/figures/fig_synthetic_with_without.png)

### 4.5 Face blurring and detection

I evaluated the same model on the test images as they are, with YuNet faces blurred first, and with the strict-mode face area of every person blurred as well. Blurring faces first cut mAP@0.5 from 0.801 to 0.700. The damage was where one would predict: goggles recall fell from 0.98 to 0.02 and no_goggle from 0.37 to zero, because goggles sit on the face. Strict mode was worse still: helmet recall fell from 0.77 to 0.58 and no_helmet from 0.51 to 0.25.

So the app detects on the original frame, held only in memory, and blurs what it displays or exports. The original never leaves the device, so this ordering costs nothing in privacy.

### 4.6 Temporal smoothing

There is no labelled PPE video, so I simulated 30 clips: a slow pan and zoom over labelled test photos, 10 seconds at 3 frames per second, with brightness flicker, noise, motion blur and JPEG compression. The crop is known, so every frame has ground truth; tracks with no decidable labelled person were left out. Neither setting raised a false alert, so precision was 1.0. Smoothing (three of the last five frames) cut status flips per track from 5.9 to 2.3 per hundred track-frames, so the boxes are much steadier, at the price of per-frame recall, which fell from 53 to 46 percent. A simulated clip is a still photograph, so its mistakes repeat on every frame; real video should give smoothing more flicker to remove.

![Figure 10. Clip check: anonymous worker log, timeline and forklift alerts.](submission/screenshots/10_clip_timeline.png)

### 4.7 Speed

End to end in Python, with face detection and matching, a photo takes about 147 ms on two laptop CPU threads, and a 30-second clip at 3 frames per second about 13 seconds. In the browser, WebGPU on the laptop's integrated graphics runs the model in about 115 ms and a whole photo in 250 to 350 ms; the WebAssembly fallback took 407 ms for the model while training occupied the CPU. The first WebGPU run compiles shaders and took almost five seconds, so the app warms up behind the loading bar. All of this is well inside the five-second budget. The app is a static site on GitHub Pages, an approved platform, with a Vercel mirror. A static site keeps every image on the device; a Streamlit version of the pipeline is in the repository as a fallback. GitHub Pages cannot send the cross-origin isolation headers, so WebAssembly runs on one thread there; with WebGPU a photo took 325 ms.

Int8 quantisation shrank the model from 10.7 to 3.3 MB, cost 1.8 points of mAP@0.5 and changed "no_" recall by +1.6 points; its CPU speed-up was inconsistent across machines and WebGPU has no int8 fast path, so the app ships the float model.

## 5. Error analysis and fairness

The failure gallery (`results/failures/`, also on the model card page) shows twelve real failures with a sentence each. On the test split four violators were shown green, five compliant workers were flagged, and ten workers were missed. Almost all are small, distant workers; the false alarms are mostly a helmet the model could not see on a far or turned-away head. The weakness is distance, and darkness makes it worse.

The subgroup audit, with bootstrap intervals over people tagged by lighting, distance and clothing colour, says the same. Near workers were detected 100 percent of the time and far workers 86 percent (77 to 93); among far violators 75 percent were caught, against 100 percent of near ones. Dark scenes had a detection recall of 74 percent (53 to 89) against 96 percent in normal light, from only 19 people. Dark clothing gave 87 percent, hi-vis 100 percent.

My own 30 real photos were harder than the test split. With the same calibrated thresholds as the app, the model decided 49 of 75 judgeable workers and was right on 84 percent of them, sent 24 percent to "check manually", caught 79 percent of the 19 violators and showed 2 of them green. It also showed 18 percent of workers without hi-vis as wearing it, and called 8 percent of workers in boots "no boots". The off-the-shelf YOLOv8s was similar (85 percent of decided, 84 percent of violators caught, 3 shown green), which suggests the limit is real-photo training data rather than my model. Bent-over workers score just under 0.25 as a person, so both apps keep people from 0.20 as "check manually"; this finds 2 more of my 91 workers.

Two attempts to fix this with more real photos failed. Fine-tuning on 3,465 extra photos whose missing labels came from that YOLOv8s left the model doubting: on my photos 41 percent of workers went to "check manually" and only 2 of 12 missing helmets were called, because unlabelled PPE had been taught as background. A second run on Kaggle added Hard Hat Workers and SH17 and painted uncertain regions grey; it cut no-vest-shown-as-vest from 18 to 8 percent but called only 4 of 12 missing helmets. The original model stays in the app.

Automatic skin-tone grouping from face-pixel colour (Individual Typology Angle) failed: it put 124 people in the darkest group and one in the lightest, because hi-vis colour casts and small shaded faces break a colour estimate. I therefore labelled the 230 test-split workers by eye from head-and-shoulder crops: 62 lighter, 74 medium, 12 darker and 82 not visible. Violators flagged or sent to "check manually" were 13 of 15 for lighter skin, 34 of 35 for medium and 8 of 8 for darker, so there is no sign that the model is worse on darker skin; but the darker group is small, one person did the labelling, and many dataset images look AI-generated. The weak group is "not visible", far or turned-away workers, with 75 percent caught: the distance effect again.

![Figure 11. Person recall by subgroup with 95 % intervals.](submission/figures/fig_subgroups.png)

![Figure 12. A known failure shown in the app: two workers missed at night.](submission/screenshots/07_failure_case.png)

## 6. Responsible AI

First, aggregate statistics only. Reports give counts and a percentage; worker numbers are anonymous and reset with every analysis, and the tracker stores no appearance. Second, retain no data. The browser app has no back end, the session log lives in memory and is gone when the tab closes, and the Streamlit app keeps nothing after the request. Third, explain the purpose. The live page asks the officer to tell people in view what the check is for, and every report says it must not be used to discipline individuals. Fourth, a human in the loop by design, not as a disclaimer: the "not sure" state, the per-class thresholds and the failure note on the demo gallery all push hard cases back to the person who can walk over and look.

The model card lists out-of-scope uses: individual monitoring, discipline, identification and covert surveillance.

## 7. Limitations

The training data is small, mostly daylight and partly synthetic-looking, and the model inherits that: distance and darkness are the dominant failures, and real hi-vis jackets and work boots are under-represented. My own test set has only 91 workers and 19 violators, labelled by me, so its numbers carry wide uncertainty. The test split has only 35 no_goggle boxes. The smoothing study uses simulated clips, forklift proximity has no depth, and the skin-tone groups were labelled by one person, with only 12 darker-skinned workers. Compliance ground truth on the test split is derived from the dataset labels with the same matching rules as the model, so a labelling mistake becomes a ground-truth mistake.

## 8. Conclusion

A 10 MB detector, fine-tuned in twenty minutes on a free GPU, decides whether a worker is wearing the required PPE correctly 95 percent of the time when it decides at all, hands about one in six people back to the officer, and shows 4 of 95 violators as compliant. It runs in a third of a second in the browser, sending nothing anywhere. Baselines without PPE training reach 70 percent accuracy or abstain almost entirely. The diffusion experiment gave a clear result: no change on ordinary photos, a large gain under simulated bad weather, and only a small edge over cheap filters. The face-blur study set a design rule: blur the output, never the input. On real photos from outside the dataset the model is weaker, and more real training data, not a bigger model, is the next step.

## References

Jocher, G., Qiu, J. and Chaurasia, A. (2024) *Ultralytics YOLO11*. Available at: https://github.com/ultralytics/ultralytics (AGPL-3.0).

Ultralytics (2025) *Construction-PPE dataset*. Available at: https://docs.ultralytics.com/datasets/detect/construction-ppe (CC BY 4.0).

Wu, W., Peng, H. and Yu, S. (2023) 'YuNet: a tiny millisecond-level face detector', *Machine Intelligence Research*, 20, pp. 656–665.

Zhang, Y. et al. (2022) 'ByteTrack: multi-object tracking by associating every detection box', *ECCV 2022*.

Cheng, T. et al. (2024) 'YOLO-World: real-time open-vocabulary object detection', *CVPR 2024*.

Sauer, A. et al. (2024) 'Adversarial diffusion distillation', *ECCV 2024* (SD-Turbo).

Radford, A. et al. (2021) 'Learning transferable visual models from natural language supervision', *ICML 2021* (CLIP).

Marafioti, A. et al. (2025) 'SmolVLM: redefining small and efficient multimodal models', arXiv:2504.05299.

Guo, C. et al. (2017) 'On calibration of modern neural networks', *ICML 2017*.

Chardon, A., Cretois, I. and Hourseau, C. (1991) 'Skin colour typology and suntanning pathways', *International Journal of Cosmetic Science*, 13(4), pp. 191–208 (Individual Typology Angle).

Liquid AI (2026) *LFM3-Omni* (1.7B, LFM Open License v1.0). Run locally with Ollama (`lfm3-omni`).

Ahmad, H. M. and Rahimi, A. (2024) 'SH17: a dataset for human safety and personal protective equipment detection', arXiv:2407.04590. Roboflow export: https://huggingface.co/datasets/fathansanum/SH-17-Dataset (CC BY 4.0).

Voxel51 (2024) *Hard Hat Workers* (hard-hat-detection). Available at: https://huggingface.co/datasets/Voxel51/hard-hat-detection (CC0).

Monk, E. (2019) *Monk Skin Tone Scale*. Available at: https://skintone.google.

Muhammad, M. B. and Yeasin, M. (2020) 'Eigen-CAM: class activation map using principal components', *IJCNN 2020*.

Mitchell, M. et al. (2019) 'Model cards for model reporting', *FAccT 2019*.
