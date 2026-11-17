"""Generates the two Colab notebooks from one readable source, so they never go stale.

    python notebooks/make_notebooks.py

The notebooks are thin: every real step calls a script in scripts/, the same code used locally.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent


def nb(cells):
    out = []
    for kind, src in cells:
        cell = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
        if kind == "code":
            cell.update({"outputs": [], "execution_count": None})
        out.append(cell)
    return {"cells": out, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}}


SETUP = [
    ("code", '''
#@title 1. Configuration: every path and hyperparameter in one place
REPO_URL   = "https://github.com/Aftaab9/sitesafe.git"   # public repo with the scripts
DRIVE_DIR  = "/content/drive/MyDrive/sitesafe"            # checkpoints, plots and results survive a disconnect
SEED       = 42                                           # fixed in sitesafe/paths.py, repeated here for the record
MODEL      = "yolo11n.pt"
IMGSZ      = 640
EPOCHS     = 100
PATIENCE   = 20                                           # early stopping
BATCH      = 16
TRY_YOLO11S = True                                        # also train the larger model and report the trade-off
'''),
    ("code", '''
#@title 2. Runtime check and time estimate
import subprocess, torch
assert torch.cuda.is_available(), "No GPU: Runtime > Change runtime type > T4 GPU"
print(torch.cuda.get_device_name(0))
print(subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv"], capture_output=True, text=True).stdout)
print("Expected on a free T4: data 2 min, YOLO11n 100 epochs about 20-25 min, evaluation and export 3 min.")
'''),
    ("code", '''
#@title 3. Mount Drive, get the code, install pinned requirements
from google.colab import drive
drive.mount("/content/drive")
import os
os.makedirs(DRIVE_DIR, exist_ok=True)
if not os.path.exists("/content/sitesafe"):
    !git clone -q {REPO_URL} /content/sitesafe
%cd /content/sitesafe
!pip -q install -r requirements-train.txt
# runs/ and results/ live on Drive so a disconnect loses nothing
for d in ("runs", "results", "models"):
    os.makedirs(f"{DRIVE_DIR}/{d}", exist_ok=True)
    if not os.path.islink(d):
        !rm -rf {d} && ln -s {DRIVE_DIR}/{d} {d}
'''),
    ("markdown", '''
## 4. Data

Construction-PPE (Ultralytics, **CC BY 4.0**): 800 train and 200 val images, 12 classes. The dataset has no test
split, so `make_split.py` carves a fixed, seeded, stratified 100-image **test** split out of train. Val is used
for model selection and thresholds only; test is touched once, for the final numbers.
'''),
    ("code", '''
!python scripts/download_data.py
!python scripts/make_split.py
!python scripts/dataset_stats.py
from IPython.display import Image, display
display(Image("results/data/class_counts.png", width=760)); display(Image("results/data/samples_grid.jpg", width=900))
'''),
]

TRAIN = [("markdown", '''
# SiteSafe: PPE compliance with privacy
### Notebook 1 of 2: train, evaluate, export

**MAIB AI 221 Computer Vision with Applications, Project 6.**
Fine-tunes YOLO11n on Construction-PPE to detect helmets, vests, gloves, boots, goggles, people, forklifts and the
five "no_" classes (visibly missing PPE), then exports the model to ONNX for the browser app.

**Contents**
1. Configuration  2. Runtime check  3. Setup  4. Data  5. Training (with resume)  6. Evaluation on the held-out test split
7. Person-level compliance and calibration  8. ONNX export and int8 quantisation  9. YOLO11s trade-off  10. Results table

Everything is free tier: Colab T4, open data, open models. "Run all" works end to end, and can be re-run after a
disconnect: finished steps are skipped and training resumes from the last checkpoint on Drive.
''')] + SETUP + [
    ("markdown", '''
## 5. Training

Same seed, image size and schedule for every run so they are comparable. Each run appends one row
(configuration, metrics, minutes) to `results/runs.csv`. If the runtime disconnects, run this cell again:
it resumes from `runs/base/weights/last.pt`.
'''),
    ("code", '''
import os
resume = "--resume" if os.path.exists("runs/base/weights/last.pt") and not os.path.exists("runs/base/results.png") else ""
if not os.path.exists("runs/base/results.png"):
    !python scripts/train.py --name base --model {MODEL} --epochs {EPOCHS} --patience {PATIENCE} --imgsz {IMGSZ} --batch {BATCH} --workers 2 --cache ram {resume}
display(Image("runs/base/results.png", width=900))
'''),
    ("markdown", "## 6. Evaluation on the held-out test split\nPer-class precision, recall and AP, the confusion matrix and the precision-recall curves. The five `no_` classes are the ones that matter for safety and are the rarest."),
    ("code", '''
!python scripts/evaluate.py --weights runs/base/weights/best.pt --split test --tag base
import pandas as pd
display(pd.read_csv("results/eval/base_test.csv"))
for f in ("confusion_matrix_normalized.png", "BoxPR_curve.png", "val_batch0_pred.jpg"):
    p = f"results/eval/plots/base_test/{f}"
    if os.path.exists(p): display(Image(p, width=820))
'''),
    ("markdown", "## 7. Person-level compliance and calibration\nPPE boxes are matched to people (intersection over PPE area plus a position check). Confidence is calibrated on val; per-class thresholds send low-confidence decisions to \"check manually\"."),
    ("code", '''
!python scripts/export_onnx.py --weights runs/base/weights/best.pt --name sitesafe_base
!python scripts/calibrate.py
!python scripts/compliance_eval.py --tag base --split test
display(Image("results/eval/reliability.png", width=460))
'''),
    ("markdown", "## 8. Int8 quantisation\nStatic quantisation calibrated on training images; the accuracy cost is measured on the test split."),
    ("code", '''
!python scripts/quantize.py --name sitesafe_base
import json; print(json.dumps(json.load(open("results/eval/quantisation.json")), indent=2))
'''),
    ("markdown", "## 9. YOLO11s: is the bigger model worth it?\nSame data and schedule. Reported: accuracy gain, file size and CPU time per image."),
    ("code", '''
if TRY_YOLO11S:
    if not os.path.exists("runs/base_s/results.png"):
        !python scripts/train.py --name base_s --model yolo11s.pt --epochs {EPOCHS} --patience {PATIENCE} --imgsz {IMGSZ} --batch {BATCH} --workers 2
    !python scripts/evaluate.py --weights runs/base_s/weights/best.pt --split test --tag base_s
    !python scripts/export_onnx.py --weights runs/base_s/weights/best.pt --name sitesafe_base_s
'''),
    ("markdown", "## 10. Results table\nEvery run so far. The files in `models/` and `results/` are on Drive; download `models/sitesafe_base.onnx` and `sitesafe_base.pt` into the repo's `models/` folder, then run `python scripts/build_web_assets.py`."),
    ("code", 'display(pd.read_csv("results/runs.csv"))\n!ls -la models/'),
]

SYNTH = [("markdown", '''
# SiteSafe: PPE compliance with privacy
### Notebook 2 of 2: generative AI experiment (synthetic hard cases)

**Question.** Rare, hard cases (night, rain, fog, dust, missing goggles) are almost absent from the training data.
Does adding diffusion-generated versions of real training images raise recall for the `no_` classes on **real**
test images?

**Method.** SD-Turbo image-to-image at low strength re-renders real labelled training images in four conditions.
Geometry is kept, so the original boxes stay valid with no hand-labelling. A quality filter drops damaged images.
Two models are trained with the same seed, epochs and settings: real only, and real + synthetic. A third arm uses
cheap classical filters instead of diffusion, to test whether the diffusion model adds anything.

**Contents** 1-4. Setup and data  5. Generate  6. Before / after examples  7. Quality filter  8. Train the arms
9. Result on real test images with confidence intervals  10. Goggle removal by inpainting (optional ablation)

Run notebook 1 first: this one needs `models/sitesafe_base.onnx` and `runs/base` on Drive.
''')] + SETUP + [
    ("code", "N_SYNTH = 400      #@param {type:'integer'}\nSTRENGTH = 0.3     #@param {type:'number'}"),
    ("markdown", "## 5. Generate the synthetic images\nAbout 0.3 s per image on a T4. Resumable: finished images are skipped."),
    ("code", '''
!python scripts/synth_generate.py --n {N_SYNTH} --mode diffusion --strength {STRENGTH}
!python scripts/synth_generate.py --n {N_SYNTH} --mode classical
'''),
    ("markdown", "## 6. Before / after examples"),
    ("code", '''
import cv2, glob, matplotlib.pyplot as plt, pandas as pd
m = pd.read_csv("data/synthetic/diffusion/manifest.csv").groupby("condition").head(2)
fig, ax = plt.subplots(len(m), 2, figsize=(10, 3.2 * len(m)))
for i, r in enumerate(m.itertuples()):
    ax[i, 0].imshow(cv2.cvtColor(cv2.imread(r.source), cv2.COLOR_BGR2RGB)); ax[i, 0].set_title("real"); ax[i, 0].axis("off")
    ax[i, 1].imshow(cv2.cvtColor(cv2.imread(f"data/synthetic/diffusion/images/train/{r.file}"), cv2.COLOR_BGR2RGB)); ax[i, 1].set_title(f"synthetic: {r.condition}"); ax[i, 1].axis("off")
plt.tight_layout(); plt.savefig("results/synth/before_after.png", dpi=120); plt.show()
'''),
    ("markdown", "## 7. Quality filter\nKeep an image only if the base model still finds half of the labelled people, CLIP says the scene is the same, and CLIP says the condition was applied. Look at the rejected sheet: that is the manual spot-check."),
    ("code", '''
!python scripts/synth_filter.py --mode diffusion
!python scripts/synth_filter.py --mode classical
display(pd.read_csv("results/synth/filter_summary_diffusion.csv"))
display(Image("results/synth/kept_diffusion.jpg", width=900)); display(Image("results/synth/rejected_diffusion.jpg", width=900))
'''),
    ("markdown", "## 8. Train the arms\nIdentical settings to the base run in notebook 1. Only the training list differs."),
    ("code", '''
for name, data in (("synth", "ppe_synth_diffusion.yaml"), ("classical", "ppe_synth_classical.yaml")):
    if not os.path.exists(f"runs/{name}/results.png"):
        resume = "--resume" if os.path.exists(f"runs/{name}/weights/last.pt") else ""
        !python scripts/train.py --name {name} --data {data} --model {MODEL} --epochs {EPOCHS} --patience {PATIENCE} --imgsz {IMGSZ} --batch {BATCH} --workers 2 {resume}
    !python scripts/export_onnx.py --weights runs/{name}/weights/best.pt --name sitesafe_{name}
'''),
    ("markdown", "## 9. Result: recall on real test images, without vs with synthetic data\nPaired bootstrap over test images (2,000 resamples). An interval that includes zero means no demonstrated effect."),
    ("code", '''
!python scripts/synth_generate.py --images data/splits/test.txt --out data/synthetic/weatherlab_eval --strength {STRENGTH}
!python scripts/synth_experiment.py --a models/sitesafe_base.onnx --b models/sitesafe_synth.onnx --tag diffusion
!python scripts/synth_experiment.py --a models/sitesafe_base.onnx --b models/sitesafe_classical.onnx --tag classical
df = pd.read_csv("results/synth/experiment_diffusion.csv"); display(df[df["class"].str.startswith(("no_", "ALL"))])
display(pd.read_csv("results/synth/experiment_classical.csv").query("`class`.str.startswith('ALL')"))
'''),
    ("markdown", '''
## 10. Optional ablation: remove goggles by inpainting
`scripts/synth_inpaint.py` inpaints the goggles box of real training images (Stable Diffusion inpainting, prompt
"bare eyes, no glasses") and relabels `goggles` to `no_goggle`. It keeps a result only if the base model no longer
detects goggles there. Train a fourth arm on real + inpainted images to separate the effect of weather from the
effect of rebalancing the rarest class.
'''),
    ("code", '''
RUN_INPAINT = False  #@param {type:"boolean"}
if RUN_INPAINT:
    !python scripts/synth_inpaint.py --n 200
    !python scripts/train.py --name inpaint --data ppe_synth_inpaint.yaml --model {MODEL} --epochs {EPOCHS} --patience {PATIENCE} --batch {BATCH} --workers 2
    !python scripts/export_onnx.py --weights runs/inpaint/weights/best.pt --name sitesafe_inpaint
    !python scripts/synth_experiment.py --a models/sitesafe_base.onnx --b models/sitesafe_inpaint.onnx --tag inpaint
'''),
]

(HERE / "01_train_evaluate_export.ipynb").write_text(json.dumps(nb(TRAIN), indent=1), encoding="utf-8")
(HERE / "02_synthetic_data_experiment.ipynb").write_text(json.dumps(nb(SYNTH), indent=1), encoding="utf-8")
print("wrote 2 notebooks")
