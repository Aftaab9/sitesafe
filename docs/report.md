# SiteSafe: PPE Compliance with Privacy

MAIB AI 221 Computer Vision with Applications, Project 6. Technical report.

## 1. Problem

On a construction site, the safety officer is the person who has to notice that a worker has taken off a hard hat to wipe their forehead, or that the new labourer on the scaffold has no gloves. They cannot watch every work area at once, and the obvious fix, cameras that record workers, creates a second problem: identifiable images of people at work, which workers rightly distrust.

I built SiteSafe for that officer. It takes a site photo, a short clip of 10 to 30 seconds, or a live camera view, and marks each person in one of three ways: compliant, missing required PPE (and which item), or "not sure, check manually". It reports a compliance percentage and counts by PPE type, flags people inside a danger zone or near a forklift, and blurs faces. In the main app nothing leaves the user's device.

The decision it supports is narrow on purpose: which workers should the officer go and speak to now. A missed "no helmet" is the expensive mistake, because it can end in a head injury; a false alarm costs a short walk. So I put recall on the "no_" classes (visibly missing PPE) ahead of overall accuracy, and added a third state, "not sure", so the model can hand a difficult case back to the human instead of guessing green.

## 2. Data

I used Construction-PPE, published by Ultralytics under CC BY 4.0. It has 1,000 photographs of construction and warehouse scenes with boxes for twelve classes: helmet, gloves, vest, boots, goggles, Person, forklift, and five "absence" classes, no_helmet, no_goggle, no_gloves, no_boots and no_vest. The "no_" classes make the dataset fit this problem: a detector that only knows "helmet" cannot tell a bare head from a head it failed to see.

The dataset ships with 800 training and 200 validation images and no test split. Reporting final numbers on the validation images would be optimistic, because I use them to pick the best epoch and to set thresholds. I therefore carved a fixed, seeded, stratified test split of 100 images out of the training set, topping up whichever class was furthest below its share so every class is present. That leaves 700 training, 200 validation and 100 test images; the test images were used once, for the final numbers, and the Colab notebook checks it rebuilds exactly the same split.

The class balance is uneven in the way that matters. In the test split there are 230 labelled people and 241 boots, but only 35 no_goggle boxes and 37 forklifts. Most photos are daylight scenes; very few are at night or in rain, fog or dust, the gap the generative part targets.

The brief also asks for 30 images of my own from public, permitted sources. I took them from Wikimedia Commons with a script that keeps only CC0, public-domain, CC BY and CC BY-SA files and records author and licence (`data/own_test/sources.csv`). From 113 candidates I chose 30 that differ from the training data: 12 at night or in tunnels, distant workers, crowds, hi-vis jackets instead of vests, and people with no PPE. I labelled the 91 workers in them by eye at person level: helmet, hi-vis and boots each worn, missing or not visible. This set exposed a real weakness: the base model sent 44 percent of these workers to "check manually", and it showed a quarter of workers without hi-vis as wearing it. The cause is in the training data. About 570 of the 800 Construction-PPE images look AI-generated, and its "no_boots" examples are almost all canvas trainers.

## 3. Methods

### 3.1 Baselines

I compared my model with two baselines that need no PPE training. The first is a COCO-trained YOLO11n that finds people, plus HSV colour rules: the head region is checked for typical hard-hat colours and the torso for high-visibility yellow-green or orange. The second is zero-shot: YOLO-World, an open-vocabulary detector, prompted with a phrase per class, such as "hard hat" or "bare head without hard hat". Both baselines feed the same person–PPE matching code as my model, so the comparison is at the level the officer sees: is this person compliant?

### 3.2 The detector

The main model is YOLO11n (2.6 million parameters), fine-tuned from COCO weights on the 700 training images at 640 pixels with a fixed seed. The model in the app was trained on my laptop CPU for 30 epochs, which took about seventeen hours; validation mAP was still rising slowly at the end. A Colab notebook repeats the run for 100 epochs with early stopping on a free T4 GPU, together with a YOLO11s comparison. The best checkpoint was exported to ONNX (opset 17, static input, no NMS in the graph) at 10.7 MB. I do the non-maximum suppression in my own code in both Python and TypeScript, so the browser and the Streamlit app make identical decisions.

### 3.3 Matching PPE to people

The matching step assigns each PPE or "no_" box to at most one person. Plain IoU does not work, because a helmet box is tiny compared with a person box, so even a perfect match has an IoU near zero. I use intersection over the PPE box's own area instead. A box must be at least half inside a person, and its centre must fall in the band of the body where that item is worn: helmet and goggles in the top 30 percent, vest across the torso, boots in the bottom 20 percent. The band check is skipped for people who are wider than tall, such as someone crouching. When a box overlaps two people, it goes to the one whose centre is closer.

If both "helmet" and "no_helmet" are matched to one person, the more confident box wins. If the two are within 0.10 of each other, or the winner is below its class threshold, the item is "unsure". A person is compliant only if every item required by the site profile is positively seen; non-compliant if a required item is missing; otherwise "check manually". Profiles are configurable: a general area needs helmet and vest, full PPE adds gloves and boots, and cutting or grinding adds goggles. Twelve fixture cases (a helmet lying at someone's feet, two people side by side, a close conflict) run against both the Python and the TypeScript implementation, so the two apps cannot drift apart.

### 3.4 Calibration and the "not sure" state

On the validation split I marked every detection as correct or wrong and fitted a single temperature to the confidence scores. It came out at 0.67, so the raw scores were slightly under-confident, and it reduced the expected calibration error on the test split from 0.090 to 0.059. I then set a per-class threshold: the lowest confidence at which the class reaches 80 percent precision on validation. For most classes this is 0.30; for no_goggle it is 0.70, so a no_goggle detection must be strong before it can turn a person red.

### 3.5 Clips, danger zones and privacy

Clips are sampled at 2, 3 or 5 frames per second. A ByteTrack-style tracker links people across frames with a constant-velocity prediction, matching confident detections first and using weak ones only to keep tracks alive. Each track gets an anonymous number, and no appearance features are stored, so the tracker cannot recognise someone who leaves and comes back. A violation is raised only when an item has been missing in three of the last five sampled frames.

Danger zones are polygons drawn by the user. A person is inside when their foot point, the bottom centre of the box, is inside. Forklift proximity has no depth, so I measure the gap between boxes in units of the person's height, and present it as a prompt to look rather than a measurement.

Faces are found with YuNet (0.2 MB) and blurred by shrinking each region to a few pixels and enlarging it before a Gaussian blur, which destroys detail rather than hiding it. In strict mode, the face area of anyone whose face was not found is blurred too. My first version of that area covered the whole head, which hid the hard hat, so I moved it below the top of the box. The main app is static: the model runs in the browser with ONNX Runtime Web, on WebGPU or multi-threaded WebAssembly, inside a Web Worker. There is no server to receive an image. The Streamlit fallback processes uploads in memory and keeps nothing.

### 3.6 The generative AI component

The data has almost no photographs at night or in rain, fog or dust, and few of missing goggles: exactly where a missed violation is most likely. I used SD-Turbo, an open diffusion model, in image-to-image mode at a low strength of 0.3. Each selected training image first gets a cheap classical hint (darkening for night, streaks for rain, haze for fog, a sand tint for dust), and the diffusion model then re-renders it so it looks like a photograph taken in that condition. At this strength the scene does not move, so the original boxes stay valid with no hand-labelling. A quality filter keeps an image only if the real-data model still finds at least half the labelled people, CLIP judges it the same scene, and CLIP agrees the condition was applied. Locally 165 of 200 images passed; the rejects were mostly fog and dust that had erased distant workers. The first test discards exactly the images the base model already struggles with, so I kept it lenient.

The experiment trains a second model on the real images plus the kept synthetic ones, with the same seed, epochs and settings, and compares recall per class on the real test images, with a paired bootstrap over images for 95 percent intervals. The Colab notebook adds an arm trained with the classical filters only, which asks whether diffusion adds anything over a few lines of OpenCV, and an arm in which goggles are removed by inpainting and relabelled as no_goggle.

The second generative piece is in the app. An optional button loads SmolVLM-256M in the browser, gives it the face-blurred photo and asks for fixed JSON: number of workers, number without each item, whether a forklift is present, and one sentence about the scene. Every number and yes/no claim is checked against the detector. Matching claims are kept, wrong ones are replaced by the detector's value and marked as corrected, and a scene sentence that restates counts or PPE facts is dropped. The language model may describe; it is never the source of a safety fact.

## 4. Results

All numbers in this section are on the 100 held-out test images unless I say otherwise.

### 4.1 Detection

The fine-tuned YOLO11n reaches a mAP@0.5 of 0.796 and a mAP@0.5:0.95 of 0.584 across the twelve classes, against 0.753 and 0.533 on validation. The easy classes are where one would expect: Person (recall 0.93), goggles (0.96), vest (0.87). The "no_" classes are harder, with a mean recall of 0.58: no_boots 0.74, no_vest 0.70, no_gloves 0.66, no_helmet 0.48 and no_goggle 0.31. Precision on the "no_" classes is high, between 0.79 and 0.93, so when the model says something is missing it is usually right; the problem is what it does not say.

The confusion matrix shows where the errors go. The largest single confusion is between boots and no_boots: sixteen no_boots boxes were called boots. Next come helmet and no_helmet, confused in both directions (six and four), then vest and no_vest, and gloves and no_gloves. These pairs share shape and position and differ only in a detail that may be a few pixels wide, such as a dark boot against a dark trouser, or a bare hand against a brown glove. Most remaining errors are background: small "no_" boxes that were not detected at all.

### 4.2 Person-level compliance and the baselines

Of the 230 labelled people, 223 can be judged from the labels. The other seven have a required item outside the frame. Of those 223, 95 are non-compliant under the general profile.

The fine-tuned model detects 94.8 percent of people. It decides 173 of the 223 and sends the rest to "check manually" (18.8 percent) or misses them (3.6 percent). Among decided people it is right 95.4 percent of the time. It never called a compliant person non-compliant (precision 1.00), and it found 88 percent of the violators it decided on. Counting the whole group of 95 violators, 85 percent were either flagged or handed to the officer, and 8.4 percent were shown green. That last figure is the one I care most about, and the failure gallery shows where it comes from.

The colour-rules baseline detects fewer people (86.5 percent). Its accuracy on decided people is 70.5 percent, its precision for "PPE missing" 0.60, and it shows 20 percent of violators as green. Colour alone cannot separate a hard hat from a pale wall behind a bare head, or a vest from orange brickwork.

The zero-shot baseline is instructive in a different way. It finds people very well (95 percent), finds hard hats and gloves reasonably (recall 0.69 and 0.67), but almost never finds a vest from the prompt "safety vest" (0.05). It cannot detect absence at all: "bare head without hard hat", "face without goggles" and "person without safety vest" all have recall zero, while "shoes" fires on every boot (recall 1.00, precision 0.26). Fed through the matching code, it ends up unsure about 93 percent of people. An open-vocabulary model knows what things are, not what is missing, and that is the whole task here.

### 4.3 Calibration

After temperature scaling the reliability curve (`results/eval/reliability.png`) sits close to the diagonal. In practice the thresholds send 18.8 percent of people to "check manually": that costs the officer some time, and it is where the borderline cases go instead of into a wrong green.

### 4.4 The generative experiment

The synthetic-data model was trained exactly like the base model, with 165 diffusion images added. On the 100 real test photos it made no measurable difference: pooled recall of the "no_" classes was 0.637 for both models, with a 95 percent interval for the change of −2.9 to +3.0 points, and mAP@0.5 moved from 0.796 to 0.786. Those photos are almost all clear daylight, which is not where the synthetic data was aimed. On weather versions of the same 100 test photos, made with the same pipeline, pooled "no_" recall rose from 34.5 to 42.8 percent (+8.3 points, interval 3.9 to 13.1) and no_boots from 32 to 61 percent. That test flatters the method, because the generator also made the training images. On my own 30 real photos, a third of them at night, the picture changed. Non-compliant workers caught or sent to "check manually" rose from 74 to 89 percent, and at night from 3 of 6 to 5 of 6. Missing helmets correctly called rose from 2 of 10 to 5 of 12, false person boxes fell from 18 to 7, and daytime results were unchanged. With 19 violators in the set these are indications rather than proof. The honest summary is that diffusion data did no harm on ordinary photos and helped where it was meant to, on a sample too small to be sure.

The verified narrative was evaluated on 40 face-blurred test photos, scoring each claim against the labels. The small model was a poor witness. Only 14 of 40 answers were usable JSON, and most invented a list of workers with names and ages ("JOHN, 25"), a privacy failure as well as a factual one. Of its claims, 68 percent were wrong. After verification every published number comes from the detector, and the error on the same claims fell to 32 percent, the detector's own error on exact counts in crowded scenes. Each answer took about 46 seconds on the laptop CPU. A 256-million-parameter model can describe a scene but cannot be trusted with a fact; the verifier is what makes it usable, and it stays an optional button.

### 4.5 Face blurring and detection

I evaluated the same model on three versions of the test images: original, with YuNet faces blurred first, and with faces and the whole head area of every person blurred. Blurring faces before detection cut mAP@0.5 from 0.796 to 0.677. The damage was concentrated where one would predict: goggles recall fell from 0.96 to 0.02 and no_goggle from 0.31 to 0.03, because goggles sit on the face. Helmet recall was unchanged, but no_helmet recall dropped from 0.48 to 0.40. Blurring whole heads was far worse: helmet recall fell to 0.08.

So the app detects on the original frame, held only in memory, and blurs what it displays or exports. The original never leaves the device, so this ordering costs nothing in privacy.

### 4.6 Temporal smoothing

There is no labelled PPE video, so I simulated 30 clips: a slow pan and zoom over labelled test photos, 10 seconds at 3 frames per second, with brightness flicker, noise, motion blur and JPEG compression. The crop is known, so every frame has ground truth; tracks with no decidable labelled person were left out of the scoring. Smoothing (three of the last five frames) raised the precision of violation alerts from 96.9 to 98.2 percent and halved the number of status flips per track, from 5.5 to 2.4 per hundred track-frames, so the boxes on screen are much steadier. The price was recall per frame, which fell from 47.7 to 42.5 percent. Part of that is delay: a new violation needs three sampled frames, about a second, before it is shown. Per-frame recall is low in both settings because the clips zoom across crowded photos with distant workers. The precision gain is modest because a simulated clip is a still photograph, so most mistakes repeat on every frame and only transient flicker can be removed; real video should contain more of it.

### 4.7 Speed

The ONNX model takes 44 ms per image on four laptop CPU threads. End to end in Python, with face detection and matching, a photo takes about 156 ms on two threads, and a 30-second clip at 3 frames per second about 14 seconds, measured while training occupied the same CPU. In the browser, WebGPU on the laptop's integrated graphics runs the model in about 115 ms and a whole photo in 250 to 350 ms; the WebAssembly fallback took 407 ms for the model. The first WebGPU run compiles shaders and took almost five seconds, so the app warms up behind the loading bar. All of this is well inside the five-second budget.

Int8 quantisation shrank the model from 10.7 MB to 3.3 MB and cut CPU time by about a quarter. It cost only 0.8 points of mAP@0.5 but 5.3 points of mean "no_" recall, which is the wrong place to lose accuracy, so the app ships the float model.

## 5. Error analysis and fairness

The failure gallery (`results/failures/`, also on the model card page) shows twelve real failures with a sentence each. On the test split eight violators were shown green and eight workers were missed. Of the six green violators in the gallery, four are small, distant workers; the others are cut off by the image edge or in an unusual pose. All six missed workers are small, three of them in dark scenes. The weakness is distance, and darkness makes it worse.

The subgroup audit, with bootstrap intervals over people tagged by lighting, distance and clothing colour, says the same. Near workers were detected 100 percent of the time, mid-distance 97 percent and far workers 88 percent (95 percent interval 81 to 94). Among far violators, only 70 percent were caught (55 to 83), against 98 percent of near ones. Dark scenes had a detection recall of 74 percent (53 to 89), against 97 percent in normal light, but there are only 19 people in dark scenes, so the interval is wide. Clothing colour made less difference: dark clothing 90 percent detection recall, hi-vis 100 percent.

The skin-tone part of the audit did not work. I estimated a coarse, Monk-style group from face-pixel colour (Individual Typology Angle); the tags put 124 people in the darkest group and one in the lightest, which is plainly wrong, because hi-vis colour casts and small shaded faces break a colour estimate. I withheld these results rather than report numbers I know are false. The tag sheet has a "verified" column, and skin-tone groups are reported only after hand checking; a ControlNet-generated set with fixed pose and varied skin tone is the natural next step.

## 6. Responsible AI

First, aggregate statistics only. The report and the dashboard give counts and a percentage; worker numbers are anonymous and reset with every analysis; the tracker stores no appearance, so it cannot follow a person across clips. Second, retain no data. The browser app has no back end, the session log lives in memory and is gone when the tab closes, and the Streamlit app keeps nothing after the request. Third, explain the purpose. The live camera page asks the officer to tell people in view what the check is for before starting, and every report says it supports the officer and must not be used to discipline individuals. Fourth, a human in the loop by design, not as a disclaimer: the "not sure" state, the per-class thresholds and the failure note on the demo gallery all push hard cases back to the person who can walk over and look.

The model card lists out-of-scope uses: individual monitoring, discipline, identification and covert surveillance.

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
