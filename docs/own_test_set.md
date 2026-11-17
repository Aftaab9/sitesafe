# My own 30-image test set: how to collect and label it

The brief asks for a small test set of my own, 30 images from public, permitted sources. It is the only data in the project that does not come from the Construction-PPE distribution, so it is the most honest check of how the model behaves on new sites.

## 1. Collect

Use only sources whose licence allows reuse, and record each one.

- Wikimedia Commons (search "construction workers", "construction site night", "road works rain"). Most files are CC BY or CC BY-SA.
- Pexels and Unsplash (their own free licences; note the photographer).
- Flickr, filtered to "Creative Commons" or "No known copyright restrictions".
- Photos I take myself, with the consent of the people in them.

Aim for a spread, because the bias check needs it. A workable target for 30 images:

| What | Target |
|---|---|
| Night or dusk | at least 6 |
| Rain, fog or dust | at least 5 |
| Distant workers (person under a quarter of the image height) | at least 8 |
| Dark or non hi-vis clothing | at least 6 |
| Visible range of skin tones | as balanced as the sources allow |
| At least one worker with PPE missing | at least 15 |
| Forklift or other vehicle | at least 4 |

Save them to `data/own_test/images/`. The folder is ignored by git; the images are not redistributed, only their sources are listed.

## 2. Pre-label

```bash
python scripts/prelabel.py
```

This runs the trained model on every image (confidence 0.15, deliberately low) and writes:

- `data/own_test/cvat_yolo.zip` for CVAT (create a task with the images, then Actions, Upload annotations, format "YOLO 1.1"),
- `data/own_test/label_studio.json` and `label_studio_config.xml` for Label Studio,
- `data/own_test/sources.csv` to fill in (file, URL, licence, author),
- `data/own_test/tags.csv` to fill in (lighting, distance, clothing, skin tone group, notes).

## 3. Correct by hand

The pre-labels are the model's guesses. Starting from them saves time, and it also risks copying the model's mistakes into the ground truth. To limit that:

1. First pass: fix, delete and add boxes with the pre-labels visible.
2. Second pass: hide all boxes, look at each person, and ask "what PPE can I see, what is visibly missing?" Then show the boxes and compare.
3. Label a `no_` class only when the body part is visible and the item is clearly absent. If the head or the feet are out of view, label nothing for that item.

Class order must stay: `helmet, gloves, vest, boots, goggles, Person, no_helmet, no_goggle, no_gloves, no_boots, forklift, no_vest`.

## 4. Re-import and evaluate

Export from CVAT as "YOLO 1.1" (or from Label Studio as "YOLO"), then:

```bash
python scripts/prelabel.py --import-zip path/to/export.zip
python scripts/evaluate.py --weights models/sitesafe_base.pt --split own --tag base
python scripts/compliance_eval.py --tag base --split own
python scripts/synth_experiment.py
python scripts/build_metrics.py
```

`synth_experiment.py` picks the set up automatically and adds a "my own images" block to the with-versus-without table.

## 5. In the report

State the number of images per condition, the licences, and that the labels were made by correcting model pre-labels in two passes. With 30 images the confidence intervals will be wide; say so.
