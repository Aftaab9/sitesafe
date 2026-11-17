"""Generates colab/SiteSafe_Colab_Full_Training.ipynb (one notebook for all GPU work).

    python colab/make_colab_notebook.py
"""
import json
from pathlib import Path

CELLS = [
("markdown", """
# SiteSafe: full GPU training on Colab (free T4)

**MAIB AI 221, Project 6.** This one notebook runs every step that is slow on a laptop:

| Step | What | Approx. time on a T4 |
|---|---|---|
| A | Fine-tune YOLO11n on real data, 100 epochs, early stopping | 25 min |
| B | Diffusion (SD-Turbo) night / rain / fog / dust versions of 400 training images + classical-filter control set + quality filter | 15 min |
| C | Train real + diffusion, and real + classical filters (same seed and epochs as A) | 2 x 30 min |
| D | Goggle removal by inpainting, relabel goggles to no_goggle, train that arm (optional) | 45 min |
| E | YOLO11s for the speed / accuracy trade-off (optional) | 40 min |
| F | Evaluate everything on the held-out test split, with-vs-without experiment, calibration, quantisation | 15 min |
| G | Zip all models and results for download | 1 min |

**How to run**
1. Runtime > Change runtime type > **T4 GPU**.
2. Upload `sitesafe_code.zip` to **My Drive / sitesafe /** (create the folder), or upload it when cell 2 asks.
3. Runtime > **Run all**. Keep the tab open.
4. If Colab disconnects, reconnect and **Run all** again. Every finished step is skipped and training resumes from the last checkpoint, because all outputs are saved to Drive.
5. At the end, download **My Drive / sitesafe / sitesafe_colab_results.zip** and send it back. It contains the models (`.onnx`, `.pt`) and every metric.

Everything here is free: open data (CC BY 4.0), open models, free Colab GPU. No API keys.
"""),
("code", """
#@title 1. Settings (edit here only)
EPOCHS      = 100    #@param {type:"integer"}
PATIENCE    = 20     #@param {type:"integer"}
BATCH       = 16     #@param {type:"integer"}
N_SYNTH     = 400    #@param {type:"integer"}
STRENGTH    = 0.30   #@param {type:"number"}
RUN_CLASSICAL_ARM = True   #@param {type:"boolean"}
RUN_INPAINT_ARM   = True   #@param {type:"boolean"}
RUN_YOLO11S       = True   #@param {type:"boolean"}
RUN_VLM_STUDY     = True   #@param {type:"boolean"}
DRIVE = "/content/drive/MyDrive/sitesafe"
"""),
("code", """
#@title 2. GPU check, Drive, code bundle, packages
import os, subprocess, shutil, zipfile, json, time
import torch
assert torch.cuda.is_available(), "No GPU. Runtime > Change runtime type > T4 GPU, then Run all again."
print("GPU:", torch.cuda.get_device_name(0))
from google.colab import drive
drive.mount("/content/drive")
os.makedirs(DRIVE, exist_ok=True)
bundle = f"{DRIVE}/sitesafe_code.zip"
if not os.path.exists(bundle):
    from google.colab import files
    print("Upload sitesafe_code.zip")
    up = files.upload()
    shutil.move(next(iter(up)), bundle)
W = "/content/sitesafe"
os.makedirs(W, exist_ok=True)
with zipfile.ZipFile(bundle) as z:
    z.extractall(W)
os.chdir(W)
# Persist outputs on Drive so a disconnect loses nothing.
for d in ("runs", "results", "data/synthetic"):
    target = f"{DRIVE}/{d}"
    os.makedirs(target, exist_ok=True)
    if os.path.islink(d): continue
    if os.path.exists(d): shutil.rmtree(d)
    os.makedirs(os.path.dirname(d) or ".", exist_ok=True)
    os.symlink(target, d)
os.makedirs(f"{DRIVE}/models", exist_ok=True)
for f in os.listdir("models"):
    if not os.path.exists(f"{DRIVE}/models/{f}"): shutil.copy(f"models/{f}", f"{DRIVE}/models/{f}")
shutil.rmtree("models"); os.symlink(f"{DRIVE}/models", "models")
!pip -q install ultralytics==8.4.215 onnx onnxruntime onnxslim diffusers transformers accelerate safetensors imageio-ffmpeg
def done(path): return os.path.exists(path)
def sh(cmd):
    print(">", cmd); t = time.time()
    r = subprocess.run(cmd, shell=True)
    print(f"  ({(time.time()-t)/60:.1f} min)")
    if r.returncode: raise RuntimeError(f"failed: {cmd}")
def try_sh(cmd):
    # optional studies: a failure is reported but does not stop the notebook
    try: sh(cmd)
    except RuntimeError as e: print("WARNING (continuing):", e)
"""),
("code", """
#@title 3. Data: download Construction-PPE (CC BY 4.0) and rebuild the identical split
sh("python scripts/download_data.py")
before = {n: open(f"data/splits/{n}.txt").read() for n in ("train", "val", "test")}
sh("python scripts/make_split.py")
for n, txt in before.items():
    assert open(f"data/splits/{n}.txt").read() == txt, f"{n} split differs from the laptop split!"
print("Split identical to the one used locally: 700 train / 200 val / 100 test.")
"""),
("code", """
#@title A. Train YOLO11n on real data
def train(name, data, model="yolo11n.pt"):
    if done(f"runs/{name}/results.png") or done(f"runs/{name}/weights/best.pt") and not done(f"runs/{name}/weights/last.pt"):
        print(f"{name}: already trained"); return
    resume = "--resume" if done(f"runs/{name}/weights/last.pt") else ""
    sh(f"python scripts/train.py --name {name} --data {data} --model {model} --epochs {EPOCHS} --patience {PATIENCE} "
       f"--batch {BATCH} --workers 2 --device 0 {resume}")
def export(name):
    if not done(f"models/sitesafe_{name}.onnx") or os.path.getmtime(f"models/sitesafe_{name}.onnx") < os.path.getmtime(f"runs/{name}/weights/best.pt"):
        sh(f"python scripts/export_onnx.py --weights runs/{name}/weights/best.pt --name sitesafe_{name}")
train("base", "ppe.yaml"); export("base")
from IPython.display import Image, display
display(Image("runs/base/results.png", width=900))
"""),
("code", """
#@title B. Generate synthetic images (diffusion and classical control) and filter them
if not done("results/synth/filter_diffusion.csv"):
    sh(f"python scripts/synth_generate.py --n {N_SYNTH} --mode diffusion --strength {STRENGTH}")
    sh("python scripts/synth_filter.py --mode diffusion")
if RUN_CLASSICAL_ARM and not done("results/synth/filter_classical.csv"):
    sh(f"python scripts/synth_generate.py --n {N_SYNTH} --mode classical")
    sh("python scripts/synth_filter.py --mode classical")
# Weather versions of the 100 TEST images (stress test) and of the 5 Weather Lab images (web demo)
if len(os.listdir("data/synthetic/weatherlab_eval/images/train") if done("data/synthetic/weatherlab_eval/images/train") else []) < 100:
    sh(f"python scripts/synth_generate.py --images data/splits/test.txt --out data/synthetic/weatherlab_eval --strength {STRENGTH}")
if len(os.listdir("data/synthetic/weatherlab/images/train") if done("data/synthetic/weatherlab/images/train") else []) < 20:
    sh(f"python scripts/synth_generate.py --images data/splits/weatherlab.txt --all-conditions --out data/synthetic/weatherlab --strength {STRENGTH}")
import pandas as pd
display(pd.read_csv("results/synth/filter_summary_diffusion.csv"))
display(Image("results/synth/kept_diffusion.jpg", width=900))
"""),
("code", """
#@title C. Train the synthetic arms (same seed, epochs and settings as A)
# The yaml files contain absolute paths; rebuild them for this machine.
sh("python scripts/synth_filter.py --mode diffusion") if not done("data/splits/ppe_synth_diffusion.yaml") else None
train("synth", "ppe_synth_diffusion.yaml"); export("synth")
if RUN_CLASSICAL_ARM:
    sh("python scripts/synth_filter.py --mode classical") if not done("data/splits/ppe_synth_classical.yaml") else None
    train("classical", "ppe_synth_classical.yaml"); export("classical")
"""),
("code", """
#@title D. (optional) Goggle removal by inpainting -> extra no_goggle examples
if RUN_INPAINT_ARM:
    if not done("data/splits/ppe_synth_inpaint.yaml"):
        sh("python scripts/synth_inpaint.py --n 200")
    train("inpaint", "ppe_synth_inpaint.yaml"); export("inpaint")
"""),
("code", """
#@title E. (optional) YOLO11s: is the bigger model worth it?
if RUN_YOLO11S:
    train("base_s", "ppe.yaml", model="yolo11s.pt"); export("base_s")
"""),
("code", """
#@title F. Evaluate everything on the held-out test split
arms = ["base", "synth"] + (["classical"] if RUN_CLASSICAL_ARM else []) + (["inpaint"] if RUN_INPAINT_ARM else []) + (["base_s"] if RUN_YOLO11S else [])
shutil.copy("runs/base/weights/best.pt", "models/sitesafe_base.pt")
for a in arms:
    sh(f"python scripts/evaluate.py --weights runs/{a}/weights/best.pt --split test --tag {a}")
sh("python scripts/calibrate.py --model models/sitesafe_base.onnx")
for a in arms:
    sh(f"python scripts/compliance_eval.py --model models/sitesafe_{a}.onnx --tag {a} --split test")
sh("python scripts/synth_experiment.py --a models/sitesafe_base.onnx --b models/sitesafe_synth.onnx --tag diffusion")
for a in ("classical", "inpaint"):
    if a in arms: sh(f"python scripts/synth_experiment.py --a models/sitesafe_base.onnx --b models/sitesafe_{a}.onnx --tag {a}")
try_sh("python scripts/quantize.py --name sitesafe_base")
try_sh("python scripts/blur_study.py --weights models/sitesafe_base.pt")
try_sh("python scripts/smoothing_study.py --model models/sitesafe_base.onnx")
try_sh("python scripts/failure_gallery.py --tag base")
if RUN_VLM_STUDY:
    try_sh("python scripts/vlm_hallucination.py --n 60")
df = pd.read_csv("results/synth/experiment_diffusion.csv")
display(df[df["class"].str.startswith(("no_", "ALL"))])
display(pd.read_csv("results/runs.csv"))
"""),
("code", """
#@title G. Pack the results for download
out = f"{DRIVE}/sitesafe_colab_results.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for root in ("models", "results"):
        for dp, _, fs in os.walk(root, followlinks=True):
            for f in fs:
                z.write(os.path.join(dp, f))
    for dp, _, fs in os.walk("runs", followlinks=True):
        for f in fs:
            if f in ("results.csv", "results.png", "args.yaml", "confusion_matrix_normalized.png", "BoxPR_curve.png") or (f == "best.pt"):
                z.write(os.path.join(dp, f))
    for d in ("data/synthetic/weatherlab", "data/synthetic/diffusion"):
        for dp, _, fs in os.walk(d, followlinks=True):
            for f in fs:
                if f.endswith((".jpg", ".csv")) and ("weatherlab" in d or len(z.namelist()) < 4000):
                    z.write(os.path.join(dp, f))
    for f in os.listdir("data/splits"):
        if f.endswith(".txt") and not f.endswith("_abs.txt"):
            z.write(f"data/splits/{f}")
print(f"Done. Download {out} ({os.path.getsize(out)/1e6:.0f} MB) from Google Drive and send it back.")
from google.colab import files
try: files.download(out)
except Exception as e: print("Browser download did not start; get it from Drive.", e)
"""),
]


def main():
    cells = []
    for kind, src in CELLS:
        c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
        if kind == "code":
            c.update(outputs=[], execution_count=None)
        cells.append(c)
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}}
    out = Path(__file__).parent / "SiteSafe_Colab_Full_Training.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
