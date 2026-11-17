# SiteSafe: PPE Compliance with Privacy

MAIB AI 221 Computer Vision with Applications, Project 6. Technical report.

## 1. Problem

On a construction site, the safety officer is the person who has to notice that a worker has taken off a hard hat to wipe their forehead, or that the new labourer on the scaffold has no gloves. They do this by walking round and looking. They cannot watch every work area at once, and the obvious technical fix, cameras that record workers, creates a second problem: a stream of identifiable images of people at work, which workers rightly distrust and which data protection law treats seriously.

I built SiteSafe for that officer. It takes a site photo, a short clip of 10 to 30 seconds, or a live camera view, and marks each person in one of three ways: compliant, missing required PPE (and which item), or "not sure, check manually". It adds up a compliance percentage and counts by PPE type, flags people standing inside a danger zone or close to a forklift, and produces a short incident report. Faces are blurred, and in the main version of the app nothing leaves the user's device.

The decision the tool supports is narrow on purpose: which workers should the officer go and speak to now. It does not decide anything about a person. That shapes how I think about errors. A missed "no helmet" is the expensive mistake, because it can end in a head injury, and the officer had no reason to look. A false alarm costs the officer a short walk and a glance. So throughout the project I put recall on the "no_" classes (visibly missing PPE) ahead of overall accuracy, and I designed a third output state, "not sure", so that the model can hand a difficult case back to the human instead of guessing green.

## 2. Data

I used Construction-PPE, published by Ultralytics under CC BY 4.0. It has 1,000 photographs of construction and warehouse scenes with boxes for twelve classes: helmet, gloves, vest, boots, goggles, Person, forklift, and five "absence" classes, no_helmet, no_goggle, no_gloves, no_boots and no_vest. The "no_" classes are what make the dataset fit this problem: a detector that only knows "helmet" cannot tell a bare head from a head it failed to see, while a detector with "no_helmet" can be positively wrong or right about absence.

The dataset ships with 800 training and 200 validation images and no test split. Reporting final numbers on the validation images would be optimistic, because I use them to pick the best epoch and to set thresholds. I therefore carved a fixed test split of 100 images out of the training set with a seeded, stratified procedure: at every step the class furthest below its proportional share is topped up with a random image that contains it, which guarantees that every class, including the rare ones, is present. The result is 700 training, 200 validation and 100 test images. The test images were used once, for the final numbers. The split lists are committed to the repository, and the Colab notebook checks that it rebuilds exactly the same split before training.

The class balance is uneven in the way that matters. In the test split there are 230 labelled people and 241 boots, but only 35 no_goggle boxes and 37 forklifts. The "no_" boxes are also small: the median no_goggle box covers well under one percent of the image. Most photos are daylight scenes; very few are at night or in rain, fog or dust, which is the gap the generative part of the project targets.

The brief also asks for 30 images of my own from public, permitted sources. I built the workflow for this rather than the set itself at the time of writing: `scripts/prelabel.py` runs the trained model over the images, writes pre-labels in CVAT's YOLO format and as Label Studio predictions, and re-imports the corrected labels. Because correcting a model's own boxes can quietly copy its mistakes into the ground truth, the guide (`docs/own_test_set.md`) asks for a second pass with all pre-labels hidden. Every evaluation script accepts this set as `--split own`, so its results drop into the same tables.

## 3. Methods

### 3.1 Baselines

I compared my model with two baselines that need no PPE training.

The first is a pre-trained detector plus classical rules. A COCO-trained YOLO11n finds people. Inside each person box, an HSV colour rule looks at the head region for typical hard-hat colours (white, yellow, orange, red, blue, green) and at the torso for high-visibility yellow-green or orange. If enough pixels match, the rule reports a helmet or a vest; otherwise it reports its absence.

The second is zero-shot: YOLO-World, an open-vocabulary detector, prompted with a phrase per class, such as "hard hat", "safety vest" or "bare head without hard hat".

Both baselines feed the same person–PPE matching code as my model, so the comparison is at the level the officer sees: is this person compliant?

### 3.2 The detector

The main model is YOLO11n, the smallest model in the current Ultralytics family (2.6 million parameters), fine-tuned from its COCO weights on the 700 training images at 640 pixels with a fixed seed. The model shipped in the app was trained on my laptop CPU for 30 epochs, which took about seventeen hours. The validation mAP was still rising slowly at the end, so I prepared a Colab notebook that repeats the run for 100 epochs with early stopping on a free T4 GPU, in under half an hour, together with the YOLO11s comparison. The best checkpoint was exported to ONNX (opset 17, static 640 by 640 input, no NMS inside the graph) at 10.7 MB. I do the non-maximum suppression in my own code in both Python and TypeScript, so the browser and the Streamlit app make identical decisions.

### 3.3 Matching PPE to people

A detector outputs boxes; the officer needs a judgement per person. The matching step assigns each PPE or "no_" box to at most one person. Plain IoU does not work here, because a helmet box is tiny compared with a person box, so the IoU of a perfect match is close to zero. I use intersection over the PPE box's own area instead: the share of the helmet that lies inside the person. A box must be at least half inside a person, and its centre must fall in the band of the body where that item is worn: helmet and goggles in the top 30 percent, vest across the torso, gloves around the arms, boots in the bottom 20 percent, each with a tolerance. The band check is skipped for people whose box is wider than it is tall, such as someone crouching. When a box overlaps two people, it goes to the one whose horizontal centre is closer.

If both "helmet" and "no_helmet" are matched to one person, the more confident box wins; if the two confidences are within 0.10 of each other, or the winner is below its class threshold, the item becomes "unsure". A person is compliant only if every item required by the site profile is positively seen. If a required item is missing, they are non-compliant. If a required item is unsure or simply not visible, they go to "check manually". The site profile is configurable: a general area needs helmet and vest, full PPE adds gloves and boots, and cutting or grinding adds goggles. The algorithm has twelve fixture cases, such as a helmet lying at someone's feet, two people side by side and a close conflict. The same fixture file is run against both the Python and the TypeScript implementation, so the two apps cannot silently diverge.

### 3.4 Calibration and the "not sure" state

On the validation split I marked every detection as correct or wrong and fitted a single temperature to the confidence scores. The temperature came out at 0.67, which means the raw scores were slightly under-confident. It reduced the expected calibration error on the test split from 0.090 to 0.059. I then chose a per-class threshold: the lowest confidence at which that class reaches 80 percent precision on validation. For most classes this is 0.30. For no_goggle it is 0.70, so a no_goggle detection has to be strong before it can turn a person red. Below threshold, the item is "unsure".

### 3.5 Clips: sampling, tracking and smoothing

A clip is sampled at 2, 3 or 5 frames per second. A tracker in the style of ByteTrack links people across frames. It predicts each track with a constant-velocity model and matches confident detections first, then uses low-confidence detections only to keep existing tracks alive. Each track gets an anonymous number: Worker 1, Worker 2. The tracker stores no appearance features, so it cannot recognise a person who leaves and comes back, and that is intentional. A violation is raised for a track only when an item has been "missing" in at least three of the last five sampled frames, which suppresses single-frame flicker.

### 3.6 Danger zones

The user draws polygons on the image; corners snap to the image edges and to each other. A person counts as inside a zone when their foot point, the bottom centre of the box, is inside it, because that is where they stand. Proximity to a forklift has no depth information, so I measure the gap between the two boxes in units of the person's own height. "Within half a person-height" means roughly the same thing near the camera and far from it. I present it as a prompt to look, not a measurement.

### 3.7 Privacy

Faces are found with YuNet, a 0.2 MB face detector from the OpenCV model zoo. Each face is blurred by shrinking the region to a few pixels and enlarging it again before a Gaussian blur, which destroys the detail rather than hiding it behind a reversible filter. In strict mode, the face area of every person whose face was not found is blurred as well. I first placed that area over the whole head, and then noticed it hid the hard hat, which is the very thing the officer needs to see; it now sits below the top of the box.

The main app is a static web application. The model runs inside the browser with ONNX Runtime Web, on WebGPU where available and multi-threaded WebAssembly otherwise, in a Web Worker so the interface never freezes. There is no server code to receive an image. The fallback Streamlit app processes uploads in memory for one request and keeps nothing.

### 3.8 The generative AI component

The data has almost no photographs at night, in rain, in fog or in dust, and few examples of missing goggles. Those are exactly the conditions in which a missed violation is most likely. My generative component tries to fill that gap without hand-labelling.

I used SD-Turbo, an open diffusion model, in image-to-image mode at a low strength of 0.3. Each selected training image first gets a cheap classical hint (darkening and a blue cast for night, streaks for rain, a haze gradient for fog, a sand tint for dust), and the diffusion model then re-renders it so that it looks like a photograph taken in that condition rather than a filter. At this strength the layout of the scene does not move, so the original boxes remain valid. A quality filter then keeps an image only if three things hold. The real-data model must still find at least half the labelled people. CLIP must judge the image to be the same scene as the original. CLIP must also agree that the condition was actually applied. Locally, 165 of 200 generated images passed. The rejected sheet showed what one would hope: mostly fog and dust that had erased distant workers. A known weakness of the first test is that it discards exactly the images the base model already struggles with, so I made it lenient.

The experiment trains a second model on the 700 real images plus the kept synthetic images, with the same seed, epochs and settings, and compares recall per class on the real test images, with a paired bootstrap over test images for a 95 percent interval. The Colab notebook adds a third arm trained with the classical filters only, to ask whether the diffusion model adds anything over a few lines of OpenCV, and a fourth arm in which goggles are removed from faces by inpainting and relabelled as no_goggle.

The second generative piece is in the app. An optional button loads SmolVLM-256M, a small open vision-language model, in the browser. It is given the face-blurred photo and asked for a fixed JSON description: number of workers, number without each item, whether a forklift is present, and one sentence about the scene. Every number and yes/no claim is then checked against the detector. Matching claims are kept, disagreeing claims are replaced by the detector's value and marked as corrected, and a scene sentence that tries to restate counts or PPE facts is dropped, because free text cannot be checked word by word. This is the "verify the assistant" pattern: the language model is allowed to describe, and never allowed to be the source of a safety fact.

## 4. Results

All numbers in this section are on the 100 held-out test images unless I say otherwise.

### 4.1 Detection

The fine-tuned YOLO11n reaches a mAP@0.5 of 0.796 and a mAP@0.5:0.95 of 0.584 across the twelve classes, against 0.753 and 0.533 on validation. The easy classes are where one would expect: Person (recall 0.93), goggles (0.96), vest (0.87). The "no_" classes are harder, with a mean recall of 0.58: no_boots 0.74, no_vest 0.70, no_gloves 0.66, no_helmet 0.48 and no_goggle 0.31. Precision on the "no_" classes is high, between 0.79 and 0.93, so when the model says something is missing it is usually right; the problem is what it does not say.

The confusion matrix shows where the errors go. The largest single confusion is between boots and no_boots: sixteen no_boots boxes were called boots. Next come helmet and no_helmet, confused in both directions (six and four), then vest and no_vest, and gloves and no_gloves. These pairs share shape and position and differ only in a detail that may be a few pixels wide, such as a dark boot against a dark trouser, or a bare hand against a brown glove. Most remaining errors are background: small "no_" boxes that were not detected at all.

### 4.2 Person-level compliance and the baselines

Of the 230 labelled people, 223 can be judged from the labels. The other seven have a required item outside the frame. Of those 223, 95 are non-compliant under the general profile.

The fine-tuned model detects 94.8 percent of people. It decides 173 of the 223 and sends the rest to "check manually" (18.8 percent) or misses them (3.6 percent). Among decided people it is right 95.4 percent of the time. It never called a compliant person non-compliant (precision 1.00), and it found 88 percent of the violators it decided on. Counting the whole group of 95 violators, 85 percent were either flagged or handed to the officer, and 8.4 percent were shown green. That last figure is the one I care most about, and the failure gallery shows where it comes from.

The colour-rules baseline detects fewer people (86.5 percent) because the COCO person detector misses some workers in odd poses. Its accuracy on decided people is 70.5 percent, its precision for "PPE missing" is 0.60, and it shows 20 percent of violators as green. The rules fail on predictable cases: white walls behind a bare head look like a white helmet, and orange brickwork looks like an orange vest.

The zero-shot baseline is instructive in a different way. It finds people very well (95 percent), finds hard hats and gloves reasonably (recall 0.69 and 0.67), but almost never finds a vest from the prompt "safety vest" (0.05). It cannot detect absence at all: "bare head without hard hat", "face without goggles" and "person without safety vest" all have recall zero, while "shoes" fires on every boot (recall 1.00, precision 0.26). Fed through the matching code, it ends up unsure about 93 percent of people. An open-vocabulary model knows what things are, not what is missing, and that is the whole task here.

### 4.3 Calibration

The reliability diagram (`results/eval/reliability.png`) shows raw scores that are mostly under-confident in the middle range; after temperature scaling the curve sits close to the diagonal. The practical effect in the app is the "check manually" share of 18.8 percent. That is high enough to cost the officer some time, and it is where most of the borderline cases end up instead of in a wrong green.

### 4.4 The generative experiment

_[Synthetic-data results: filled in from `results/synth/experiment_diffusion.csv` once the comparison run has finished. See section 4.4 notes in the repository.]_

### 4.5 Face blurring and detection

I evaluated the same model on three versions of the test images: original, with YuNet faces blurred first, and with faces and the whole head area of every person blurred. Blurring faces before detection cut mAP@0.5 from 0.796 to 0.677. The damage was concentrated where one would predict: goggles recall fell from 0.96 to 0.02 and no_goggle from 0.31 to 0.03, because goggles sit on the face. Helmet recall was unchanged, but no_helmet recall dropped from 0.48 to 0.40. Blurring whole heads was far worse: helmet recall fell to 0.08.

This settled the design question. The app detects on the original frame, held only in memory, and blurs what it displays or exports. In the browser version the original never leaves the device, so this ordering costs nothing in privacy and keeps the accuracy. It also showed that my first strict-mode blur region was wrong, because it covered the helmet, and I moved it.

### 4.6 Temporal smoothing

There is no labelled PPE video in the dataset, so I simulated clips: a slow pan and zoom over 30 labelled test photos, at 3 frames per second for 10 seconds. Each frame got brightness flicker, sensor noise, occasional motion blur and JPEG compression. Because the crop is known, the labels move with it exactly and every frame has ground truth. Smoothing (three of the last five frames) halved the number of status flips per track, from 5.5 to 2.4 per hundred track-frames, so the boxes on screen are steadier. It did not raise violation precision in these clips, and it cost some recall. The reason is instructive: in a simulated clip the scene is the same still photograph, so most of the model's false alarms are systematic. They repeat on every frame, and smoothing cannot remove them. Smoothing removes transient flicker, which is what makes a live view feel nervous, not consistent misjudgements. Real video, with people turning and moving, should have more transient errors than this simulation, so I see this as a lower bound on its usefulness rather than the final word.

### 4.7 Speed

The ONNX model alone takes 44 ms per image on four laptop CPU threads and 73 ms on two. End to end in Python, including pre-processing, NMS, matching and face detection, a photo takes about 156 ms on two threads. A 30-second clip sampled at 3 frames per second takes about 14 seconds. These figures were measured while a training run occupied the same CPU, so they are pessimistic. In the browser, with WebGPU on the laptop's integrated graphics, the model takes about 115 ms and the whole photo analysis 250 to 350 ms. The first WebGPU run compiles its shaders and took almost five seconds, so the app now does a warm-up pass behind the loading bar. All of this is well inside the brief's limit of about five seconds per image.

Int8 quantisation shrank the model from 10.7 MB to 3.3 MB and cut CPU time by about a quarter. It cost only 0.8 points of mAP@0.5 but 5.3 points of mean "no_" recall, which is the wrong place to lose accuracy, so the app ships the float model.

## 5. Error analysis and fairness

The failure gallery (`results/failures/`, also on the app's model card page) has twelve real failures with a sentence each. On the whole test split, eight violators were shown green and eight workers were not detected at all; the gallery shows six of each. Of the six violators shown green, four are small, distant workers whose missing helmet or vest occupies a handful of pixels; the other two are a worker cut off by the image edge and one in an unusual pose. All six undetected workers are small, and three of them are in dark scenes. There were no false alarms on decided people. The pattern is clear: the model's weakness is distance, and darkness makes it worse.

The subgroup audit says the same thing with intervals. I tagged every labelled person automatically by lighting (image brightness), distance (person height as a share of the image) and clothing colour, then computed person-detection recall and "violators flagged or sent to check" per group with a bootstrap. Near workers were detected 100 percent of the time, mid-distance 97 percent and far workers 88 percent (95 percent interval 81 to 94). Among far violators, only 70 percent were caught (55 to 83), against 98 percent of near ones. Dark scenes had a detection recall of 74 percent (53 to 89), against 97 percent in normal light, but there are only 19 people in dark scenes, so the interval is wide. Clothing colour made less difference: dark clothing 90 percent detection recall, hi-vis 100 percent.

The skin-tone part of the audit did not work, and I want to be direct about it. I estimated an apparent skin-tone group from the colour of face pixels using the Individual Typology Angle, with three coarse groups in the spirit of the Monk scale. The automatic tags put 124 people in the darkest group and one in the lightest, which is plainly not what the photos show. Hi-vis colour casts and small, shaded faces break a colour-based estimate. I have withheld the skin-tone results from the app and this report rather than present numbers I know are wrong. The tagging sheet (`data/splits/test_tags.csv`) has a "verified" column for hand correction, and the audit script reports skin-tone groups only once tags have been checked by eye. A controlled set generated with ControlNet (same pose, same PPE, varied skin tone) is the natural next step, and the generation pipeline already exists.

## 6. Responsible AI

Several design decisions follow from the fact that this tool looks at workers.

First, aggregate statistics only. The report and the dashboard give counts and a percentage; worker numbers are anonymous and reset with every analysis; the tracker stores no appearance, so it cannot follow a person across clips. Second, retain no data. The browser app has no back end, the session log lives in memory and is gone when the tab closes, and the Streamlit app keeps nothing after the request. Third, explain the purpose. The live camera page asks the officer to tell people in view what the check is for before starting, and every report says it supports the officer and must not be used to discipline individuals. Fourth, a human in the loop by design, not as a disclaimer: the "not sure" state, the per-class thresholds and the failure note on the demo gallery all push hard cases back to the person who can walk over and look.

The model card (`docs/model_card.md`) lists intended and out-of-scope uses. The out-of-scope list includes individual performance monitoring, disciplinary use, identification and covert surveillance.

## 7. Limitations

The training data is small, mostly daylight, and mostly photographed from human height at moderate distance; the model inherits that. Distance is the dominant failure. The shipped model was trained for 30 epochs on a CPU; the 100-epoch GPU run in the Colab notebook should be somewhat better and will replace it. The test split has only 35 no_goggle boxes, so per-class numbers for the rare classes carry wide intervals. The temporal-smoothing study uses simulated clips. The forklift proximity check has no depth. The skin-tone audit is not yet valid, as explained above. Finally, compliance ground truth is derived from the dataset labels with the same matching rules as the model. A labelling mistake therefore becomes a ground-truth mistake, and I hand-checked only a sample of the per-person sheet.

## 8. Conclusion

A 10 MB detector, fine-tuned for less than a day on a laptop, decides whether a worker is wearing the required PPE correctly 95 percent of the time when it decides at all. It hands about one in five people back to the officer, and shows fewer than one in ten violators as compliant. It does this in a third of a second inside a browser, without sending a single pixel anywhere. Two baselines without PPE training reach 70 percent and near-total abstention, which shows that the fine-tuning, and the "no_" classes in particular, carry the result. The face-blur study turned a privacy instinct into a measured design rule: blur the output, never the input. The remaining weaknesses, distant workers and darkness, are visible in the failure gallery and the subgroup table, and they are the right targets for more data, synthetic or real.

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

Monk, E. (2019) *Monk Skin Tone Scale*. Available at: https://skintone.google.

Muhammad, M. B. and Yeasin, M. (2020) 'Eigen-CAM: class activation map using principal components', *IJCNN 2020*.

Mitchell, M. et al. (2019) 'Model cards for model reporting', *FAccT 2019*.
