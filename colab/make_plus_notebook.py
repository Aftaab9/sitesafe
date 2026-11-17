"""Generates colab/SiteSafe_Colab_Plus_Training.ipynb: the improved model trained on extra real data.

    python colab/make_plus_notebook.py
"""
import json
from pathlib import Path

CELLS = [
("markdown", """
# SiteSafe: improved model with extra real photos (free T4)

The first model was trained only on Construction-PPE, where about 570 of 800 training images look
AI-generated and "no_boots" almost always means canvas trainers. On real photos it missed hi-vis
jackets, called work boots "no_boots" and saw vests that were not there.

This notebook adds three open datasets of **real** photographs (all CC BY 4.0, downloaded from
Hugging Face inside Colab), maps them to our 12 classes, keeps one copy of each Roboflow-augmented
photo, fills each dataset's missing classes with confident pseudo-labels from the first model
(never on top of a human label), and fine-tunes YOLO11n on everything.

| Step | Time on a T4 |
|---|---|
| 1-3 Setup, Construction-PPE, extra datasets | about 10 min |
| 4 Train `plus` (80 epochs, early stopping) | about 1.5 to 2 h |
| 5 Evaluate on the test split and on my 30 own real photos, export | 5 min |

**How to run.** Runtime → Change runtime type → T4 GPU. Upload `sitesafe_plus_code.zip` to
**My Drive / sitesafe /** (or when cell 2 asks). Runtime → Run all. When it finishes, send back
**My Drive / sitesafe / sitesafe_plus_results.zip**. A disconnect is safe: Run all again and it resumes.
"""),
("code", """
#@title 1. Settings
EPOCHS = 80     #@param {type:"integer"}
PATIENCE = 20   #@param {type:"integer"}
BATCH = 16      #@param {type:"integer"}
DRIVE = "/content/drive/MyDrive/sitesafe"
"""),
("code", """
#@title 2. GPU, Drive, code bundle, packages
import os, subprocess, shutil, zipfile, time, json
import torch
assert torch.cuda.is_available(), "No GPU. Runtime > Change runtime type > T4 GPU, then Run all again."
print("GPU:", torch.cuda.get_device_name(0))
from google.colab import drive
drive.mount("/content/drive")
os.makedirs(DRIVE, exist_ok=True)
bundle = f"{DRIVE}/sitesafe_plus_code.zip"
if not os.path.exists(bundle):
    from google.colab import files
    print("Upload sitesafe_plus_code.zip")
    up = files.upload()
    shutil.move(next(iter(up)), bundle)
W = "/content/sitesafe_plus"
os.makedirs(W, exist_ok=True)
with zipfile.ZipFile(bundle) as z:
    z.extractall(W)
os.chdir(W)
RUNS = f"{DRIVE}/runs_plus"
os.makedirs(RUNS, exist_ok=True)
if not os.path.islink("runs"):
    shutil.rmtree("runs", ignore_errors=True); os.symlink(RUNS, "runs")
!pip -q install ultralytics==8.4.215 onnx onnxruntime onnxslim huggingface_hub
def sh(cmd):
    print(">", cmd); t = time.time()
    r = subprocess.run(cmd, shell=True)
    print(f"  ({(time.time()-t)/60:.1f} min)")
    if r.returncode: raise RuntimeError(f"failed: {cmd}")
"""),
("code", """
#@title 3. Data: Construction-PPE (identical split) + three extra real-photo datasets
sh("python scripts/download_data.py")
before = {n: open(f"data/splits/{n}.txt").read() for n in ("train", "val", "test")}
sh("python scripts/make_split.py")
for n, txt in before.items():
    assert open(f"data/splits/{n}.txt").read() == txt, f"{n} split differs from the laptop split!"
sh("python scripts/prepare_extra_data.py")
print(open("data/extra/stats.json").read())
"""),
("code", """
#@title 4. Train YOLO11n on Construction-PPE + extra real photos
done = os.path.exists("runs/plus/results.png")
if not done:
    resume = "--resume" if os.path.exists("runs/plus/weights/last.pt") else ""
    sh(f"python scripts/train.py --name plus --data ppe_plus.yaml --model yolo11n.pt --epochs {EPOCHS} --patience {PATIENCE} "
       f"--batch {BATCH} --workers 2 --device 0 --cache disk {resume}")
from IPython.display import Image, display
display(Image("runs/plus/results.png", width=900))
"""),
("code", """
#@title 5. Evaluate (test split + my 30 own real photos), export, pack results
sh("python scripts/export_onnx.py --weights runs/plus/weights/best.pt --name sitesafe_plus")
sh("python scripts/evaluate.py --weights runs/plus/weights/best.pt --split test --tag plus")
sh("python scripts/compliance_eval.py --model models/sitesafe_plus.onnx --tag plus --split test")
sh("python scripts/own_eval.py --models models/sitesafe_base.onnx models/sitesafe_plus.onnx")
out = f"{DRIVE}/sitesafe_plus_results.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in ("models/sitesafe_plus.onnx", "data/extra/stats.json", "results/runs.csv"):
        if os.path.exists(f): z.write(f)
    z.write("runs/plus/weights/best.pt", "models/sitesafe_plus.pt")
    for dp, _, fs in os.walk("results"):
        for f in fs: z.write(os.path.join(dp, f))
    for f in ("results.csv", "results.png", "args.yaml"):
        if os.path.exists(f"runs/plus/{f}"): z.write(f"runs/plus/{f}", f"runs/plus/{f}")
print(f"Done. Send back {out} ({os.path.getsize(out)/1e6:.0f} MB).")
print(open("results/eval/own_eval.json").read())
from google.colab import files
try: files.download(out)
except Exception as e: print("Download from Drive instead.", e)
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
    out = Path(__file__).parent / "SiteSafe_Colab_Plus_Training.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
