"""Detection evaluation on a held-out split: per-class AP, precision, recall, confusion matrix.

    python scripts/evaluate.py --weights runs/base/weights/best.pt --split test --tag base
    python scripts/evaluate.py --weights models/sitesafe_base.onnx --split test --tag base_onnx

Writes results/eval/<tag>_<split>.csv (one row per class), a confusion matrix PNG and JSON summary.
The no_* classes are summarised separately because their recall is what matters for safety.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sitesafe.classes import CLASS_NAMES, NO_CLASSES  # noqa: E402
from sitesafe.paths import RESULTS, SPLITS  # noqa: E402


def evaluate(weights: str, split: str, tag: str, data: str = "ppe.yaml", imgsz: int = 640, plots: bool = True) -> dict:
    from ultralytics import YOLO
    out = RESULTS / "eval"
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(weights, task="detect")
    tag_split = split
    if split == "own":  # my own 30 images: point the yaml's test entry at them
        from sitesafe.paths import ROOT
        lst = SPLITS / "own_abs.txt"
        lst.write_text("\n".join((ROOT / ln).as_posix() for ln in (SPLITS / "own.txt").read_text().splitlines() if ln.strip()) + "\n")
        text = (SPLITS / data).read_text().splitlines()
        (SPLITS / "ppe_own.yaml").write_text("\n".join(f"test: {lst.as_posix()}" if ln.startswith("test:") else ln for ln in text) + "\n")
        data, split = "ppe_own.yaml", "test"
    m = model.val(data=str(SPLITS / data), split=split, imgsz=imgsz, plots=plots, verbose=False,
                  project=str(out / "plots"), name=f"{tag}_{tag_split}", exist_ok=True, batch=8, workers=0)
    rows = []
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        rows.append({"class": CLASS_NAMES[int(c)], "precision": round(float(p), 4), "recall": round(float(r), 4),
                     "ap50": round(float(ap50), 4), "ap50_95": round(float(ap), 4)})
    df = pd.DataFrame(rows)
    split = tag_split
    df.to_csv(out / f"{tag}_{split}.csv", index=False)
    no = df[df["class"].isin([CLASS_NAMES[i] for i in NO_CLASSES])]
    speed = {k: round(float(v), 2) for k, v in m.speed.items()}
    summary = {"tag": tag, "split": split, "weights": weights,
               "map50": round(float(m.box.map50), 4), "map50_95": round(float(m.box.map), 4),
               "precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4),
               "no_class_mean_recall": round(float(no["recall"].mean()), 4),
               "no_class_mean_ap50": round(float(no["ap50"].mean()), 4),
               "speed_ms": speed}
    (out / f"{tag}_{split}.json").write_text(json.dumps(summary, indent=2))
    if plots and getattr(m, "confusion_matrix", None) is not None:
        import numpy as np
        np.savetxt(out / f"{tag}_{split}_confusion.csv", m.confusion_matrix.matrix, delimiter=",", fmt="%d")
    print(df.to_string(index=False))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", default="ppe.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    evaluate(a.weights, a.split, a.tag, a.data, a.imgsz)
