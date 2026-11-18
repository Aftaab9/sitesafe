"""Generates colab/SiteSafe_Colab_V2_Training.ipynb: second attempt at a model that works on real photos.

    python colab/make_v2_notebook.py
"""
import json
from pathlib import Path

CELLS = [
("markdown", """
# SiteSafe v2: more real photos, without the missing-label trap (free T4)

The "plus" model learned to doubt everything because its extra photos were only partly labelled:
every helmet or boot nobody labelled counted as background. This run uses data where the important
classes are labelled by people (Hard Hat Workers, CC0; SH17, CC BY 4.0; plus the three earlier sets),
fills other classes only where the teacher is confident, and **paints uncertain regions grey** so they
teach nothing. It fine-tunes the current app model, then scores the result on the test split and on
my 30 own real photos and prints a ship / don't-ship verdict.

| Step | Time on a T4 |
|---|---|
| 1-3 Setup, data download and preparation | about 25 min (first run only, then cached on Drive) |
| 4 Train `v2` (40 epochs) | about 2 h |
| 5 Evaluate, pack, download | about 5 min |

**How to run.** Runtime → Change runtime type → **T4 GPU**. Put `sitesafe_v2_code.zip` in
**My Drive / sitesafe /** (or upload it when cell 2 asks). Then **Runtime → Run all**.

**Nothing is lost if you are away.** Prepared data, every training checkpoint and the final
`sitesafe_v2_results.zip` are written to **My Drive / sitesafe /** as they are made. When the run ends
the zip is also downloaded to your computer automatically, as long as this browser tab is still open.
If Colab disconnects part-way, open the notebook again and **Run all**: it skips finished steps and
resumes training from the last epoch.
"""),
("code", """
#@title 1. Settings
EPOCHS = 40     #@param {type:"integer"}
PATIENCE = 12   #@param {type:"integer"}
BATCH = 32      #@param {type:"integer"}
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
bundle = f"{DRIVE}/sitesafe_v2_code.zip"
if not os.path.exists(bundle):
    from google.colab import files
    print("Upload sitesafe_v2_code.zip")
    up = files.upload()
    shutil.move(next(iter(up)), bundle)
W = "/content/sitesafe_v2"
os.makedirs(W, exist_ok=True)
with zipfile.ZipFile(bundle) as z:
    z.extractall(W)
os.chdir(W)
RUNS = f"{DRIVE}/runs_v2"          # checkpoints go straight to Drive, every epoch
os.makedirs(RUNS, exist_ok=True)
if not os.path.islink("runs"):
    shutil.rmtree("runs", ignore_errors=True); os.symlink(RUNS, "runs")
!pip -q install ultralytics==8.4.215 onnx onnxruntime onnxslim huggingface_hub
def sh(cmd):
    print(">", cmd, flush=True); t = time.time()
    r = subprocess.run(cmd, shell=True)
    print(f"  ({(time.time()-t)/60:.1f} min)", flush=True)
    if r.returncode: raise RuntimeError(f"failed: {cmd}")
"""),
("code", """
#@title 3. Data: Construction-PPE (identical split) + five real-photo datasets (cached on Drive)
sh("python scripts/download_data.py")
before = {n: open(f"data/splits/{n}.txt").read() for n in ("train", "val", "test")}
sh("python scripts/make_split.py")
for n, txt in before.items():
    assert open(f"data/splits/{n}.txt").read() == txt, f"{n} split differs from the laptop split!"
cache = f"{DRIVE}/sitesafe_v2_data.zip"
if os.path.exists(cache):
    print("Restoring prepared data from Drive ...")
    with zipfile.ZipFile(cache) as z:
        z.extractall(W)
else:
    sh("python scripts/prepare_v2_data.py")
    print("Saving prepared data to Drive (so a disconnect never repeats this step) ...")
    tmp = "/content/sitesafe_v2_data.zip"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for dp, _, fs in os.walk("data/extra_v2"):
            for f in fs:
                z.write(os.path.join(dp, f))
        for f in os.listdir("data/splits"):
            if f.startswith("v2_") or f == "ppe_v2.yaml":
                z.write(f"data/splits/{f}")
    shutil.copy(tmp, cache)
print(open("data/extra_v2/stats.json").read())
"""),
("code", """
#@title 4. Fine-tune the current app model on all of it
if not os.path.exists("runs/v2/results.png"):
    resume = "--resume" if os.path.exists("runs/v2/weights/last.pt") else ""
    sh(f"python scripts/train.py --name v2 --data ppe_v2.yaml --model models/sitesafe_base.pt --epochs {EPOCHS} "
       f"--patience {PATIENCE} --batch {BATCH} --workers 2 --device 0 --cache disk {resume}")
from IPython.display import Image, display
display(Image("runs/v2/results.png", width=900))
"""),
("code", """
#@title 5. Evaluate, verdict, save the results zip to Drive and download it
errors = []
def step(cmd):
    try: sh(cmd)
    except Exception as e: errors.append(str(e)); print("  !!", e)
step("python scripts/export_onnx.py --weights runs/v2/weights/best.pt --name sitesafe_v2")
step("python scripts/evaluate.py --weights runs/v2/weights/best.pt --split test --tag v2")
step("python scripts/calibrate.py --model models/sitesafe_v2.onnx --tag v2")
step("python scripts/compliance_eval.py --model models/sitesafe_v2.onnx --tag v2 --split test")
step("python scripts/own_eval.py --models models/sitesafe_base.onnx models/sitesafe_v2.onnx")

verdict = "not evaluated"
try:
    o = json.load(open("results/eval/own_eval.json"))
    b, v = o["sitesafe_base"], o["sitesafe_v2"]
    keys = [("helmet_M_shown_M", "missing helmets called", +1), ("unsure_share", "sent to check manually", -1),
            ("nc_shown_green", "violators shown green", -1), ("vest_M_shown_W", "no-vest shown as vest", -1),
            ("boots_W_shown_W", "boots recognised", +1), ("accuracy", "right, of decided", +1)]
    print(f"{'own 30 photos':28s}{'current':>10s}{'v2':>10s}")
    for k, label, _ in keys:
        print(f"{label:28s}{b[k]:>10.2f}{v[k]:>10.2f}")
    ok = (v["helmet_M_shown_M"] >= b["helmet_M_shown_M"] - 0.1 and v["unsure_share"] <= b["unsure_share"] + 0.03
          and v["nc_shown_green"] <= b["nc_shown_green"] + 0.06)
    verdict = "SHIP: v2 is at least as good on the checks that matter" if ok else "DON'T SHIP: v2 is worse on a key check"
except Exception as e:
    errors.append(f"verdict: {e}")
print("\\nVERDICT:", verdict)

out = f"{DRIVE}/sitesafe_v2_results.zip"
tmp = "/content/sitesafe_v2_results.zip"
with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in ("models/sitesafe_v2.onnx", "data/extra_v2/stats.json", "results/runs.csv"):
        if os.path.exists(f): z.write(f)
    if os.path.exists("runs/v2/weights/best.pt"):
        z.write("runs/v2/weights/best.pt", "models/sitesafe_v2.pt")
    for dp, _, fs in os.walk("results"):
        for f in fs: z.write(os.path.join(dp, f))
    for f in ("results.csv", "results.png", "args.yaml"):
        if os.path.exists(f"runs/v2/{f}"): z.write(f"runs/v2/{f}", f"runs/v2/{f}")
    z.writestr("verdict.txt", verdict + "\\n" + "\\n".join(errors))
shutil.copy(tmp, out)
print(f"Saved to Drive: {out} ({os.path.getsize(out)/1e6:.0f} MB)")
if errors: print("Some steps failed (details in verdict.txt):", errors)
from google.colab import files
try:
    files.download(tmp)
    print("Download started. If nothing arrives, take it from My Drive / sitesafe /.")
except Exception as e:
    print("Automatic download not possible (tab closed?). Take it from My Drive / sitesafe /.", e)
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
    out = Path(__file__).parent / "SiteSafe_Colab_V2_Training.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
