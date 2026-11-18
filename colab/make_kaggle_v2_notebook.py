"""Generates colab/SiteSafe_Kaggle_V2_Training.ipynb: the v2 run for Kaggle's free GPU (when Colab has none).

    python colab/make_kaggle_v2_notebook.py

Same steps as SiteSafe_Colab_V2_Training.ipynb. Differences: the code bundle comes from a Kaggle
dataset instead of Google Drive, the work happens in /tmp, and only the results are written to
/kaggle/working, which Kaggle keeps as the notebook's output after a background run.
"""
import json
from pathlib import Path

CELLS = [
("markdown", """
# SiteSafe v2 on Kaggle (free GPU)

Same run as the Colab v2 notebook: five real-photo datasets, uncertain regions painted grey so they
never count as background, a fine-tune of the current app model, then a ship / don't-ship verdict
on the test split and my 30 own photos. About 2.5 hours on a T4.

**One-time setup**
1. kaggle.com → Settings → verify your phone number (needed for GPU and internet).
2. Datasets → New dataset → upload `sitesafe_v2_code.zip` (keep it as a zip), any title, Create.
3. Code → New notebook → File → Import notebook → this file.
4. Right-hand panel: **Add input** → your dataset; **Accelerator → GPU T4 x2** (not P100: the
   current PyTorch no longer supports it); **Internet → On**.

**Run it in the background (browser can be closed)**
Save Version (top right) → **Save & Run All (Commit)** → Save. Close the tab if you like.
When it finishes (Kaggle e-mails you), open the notebook → **Output** → download
`sitesafe_v2_results.zip`.
"""),
("code", """
# 1. Settings
EPOCHS = 40
PATIENCE = 12
BATCH = 32
"""),
("code", """
# 2. GPU, code bundle, packages
import os, subprocess, shutil, zipfile, time, json, glob
import torch
assert torch.cuda.is_available(), "No GPU: right-hand panel > Accelerator > GPU T4 x2"
print("GPU:", torch.cuda.get_device_name(0))
found = glob.glob("/kaggle/input/**/sitesafe_v2_code.zip", recursive=True)
W = "/tmp/sitesafe_v2"            # big data stays out of the saved output
os.makedirs(W, exist_ok=True)
if found:
    with zipfile.ZipFile(found[0]) as z:
        z.extractall(W)
else:  # Kaggle sometimes unpacks uploaded zips: copy the folder instead
    marker = glob.glob("/kaggle/input/**/scripts/prepare_v2_data.py", recursive=True)
    assert marker, "Add the dataset with sitesafe_v2_code.zip as an input (right-hand panel > Add input)"
    shutil.copytree(os.path.dirname(os.path.dirname(marker[0])), W, dirs_exist_ok=True)
os.chdir(W)
OUT = "/kaggle/working"
os.makedirs(f"{OUT}/runs_v2", exist_ok=True)  # checkpoints are written straight into the saved output
if not os.path.islink("runs"):
    shutil.rmtree("runs", ignore_errors=True); os.symlink(f"{OUT}/runs_v2", "runs")
subprocess.run("pip -q install ultralytics==8.4.215 onnx onnxruntime onnxslim huggingface_hub", shell=True, check=True)
def sh(cmd):
    print(">", cmd, flush=True); t = time.time()
    r = subprocess.run(cmd, shell=True)
    print(f"  ({(time.time()-t)/60:.1f} min)", flush=True)
    if r.returncode: raise RuntimeError(f"failed: {cmd}")
"""),
("code", """
# 3. Data: Construction-PPE (identical split) + five real-photo datasets
sh("python scripts/download_data.py")
before = {n: open(f"data/splits/{n}.txt").read() for n in ("train", "val", "test")}
sh("python scripts/make_split.py")
for n, txt in before.items():
    assert open(f"data/splits/{n}.txt").read() == txt, f"{n} split differs from the laptop split!"
sh("python scripts/prepare_v2_data.py")
print(open("data/extra_v2/stats.json").read())
"""),
("code", """
# 4. Fine-tune the current app model on all of it
if not os.path.exists("runs/v2/results.png"):
    resume = "--resume" if os.path.exists("runs/v2/weights/last.pt") else ""
    sh(f"python scripts/train.py --name v2 --data ppe_v2.yaml --model models/sitesafe_base.pt --epochs {EPOCHS} "
       f"--patience {PATIENCE} --batch {BATCH} --workers 4 --device 0 --cache disk {resume}")
"""),
("code", """
# 5. Evaluate, verdict, results zip in the saved output
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
    keys = [("helmet_M_shown_M", "missing helmets called"), ("unsure_share", "sent to check manually"),
            ("nc_shown_green", "violators shown green"), ("vest_M_shown_W", "no-vest shown as vest"),
            ("boots_W_shown_W", "boots recognised"), ("accuracy", "right, of decided")]
    print(f"{'own 30 photos':28s}{'current':>10s}{'v2':>10s}")
    for k, label in keys:
        print(f"{label:28s}{b[k]:>10.2f}{v[k]:>10.2f}")
    ok = (v["helmet_M_shown_M"] >= b["helmet_M_shown_M"] - 0.1 and v["unsure_share"] <= b["unsure_share"] + 0.03
          and v["nc_shown_green"] <= b["nc_shown_green"] + 0.06)
    verdict = "SHIP: v2 is at least as good on the checks that matter" if ok else "DON'T SHIP: v2 is worse on a key check"
except Exception as e:
    errors.append(f"verdict: {e}")
print("\\nVERDICT:", verdict)

out = f"{OUT}/sitesafe_v2_results.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in ("models/sitesafe_v2.onnx", "data/extra_v2/stats.json", "results/runs.csv"):
        if os.path.exists(f): z.write(f)
    if os.path.exists("runs/v2/weights/best.pt"):
        z.write("runs/v2/weights/best.pt", "models/sitesafe_v2.pt")
    for dp, _, fs in os.walk("results"):
        for f in fs: z.write(os.path.join(dp, f))
    for f in ("results.csv", "results.png", "args.yaml"):
        if os.path.exists(f"runs/v2/{f}"): z.write(f"runs/v2/{f}", f"runs/v2/{f}")
    z.writestr("verdict.txt", verdict + "\\n" + "\\n".join(errors))
print(f"Saved {out} ({os.path.getsize(out)/1e6:.0f} MB): download it from the notebook's Output tab.")
if errors: print("Some steps failed (details in verdict.txt):", errors)
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
          "metadata": {"kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}}
    out = Path(__file__).parent / "SiteSafe_Kaggle_V2_Training.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
