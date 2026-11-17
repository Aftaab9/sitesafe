"""Calibration and the "not sure" thresholds. Fitted on VAL, reported on TEST.

    python scripts/calibrate.py --model models/sitesafe_base.onnx

1. Every detection on val is marked correct (matches a same-class label at IoU >= 0.5) or wrong.
2. Reliability diagram and expected calibration error (ECE, 15 bins) before and after temperature
   scaling of the confidence (one scalar T fitted on val by minimising the negative log-likelihood).
3. Per-class "unsure" threshold: the lowest confidence at which that class reaches the target
   precision on val (clamped to [0.30, 0.70]). A deciding PPE box below its threshold sends the
   person to "check manually" instead of green or red.

Writes results/eval/calibration.json, results/eval/reliability.png and web/src/lib/thresholds.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sitesafe.classes import CLASS_NAMES, ITEM_OF_CLASS  # noqa: E402
from sitesafe.detector import Detector  # noqa: E402
from sitesafe.evalutil import load_split, match_boxes, read_gt, read_rgb  # noqa: E402
from sitesafe.paths import RESULTS, ROOT  # noqa: E402

TARGET_PRECISION = 0.80
BINS = 15


def collect(det: Detector, split: str, conf: float = 0.1):
    confs, correct, cls = [], [], []
    for p in load_split(split):
        rgb = read_rgb(p)
        h, w = rgb.shape[:2]
        gt = read_gt(p, w, h)
        dets = det(rgb, conf)
        matched = set(match_boxes(gt, dets).values())
        for i, d in enumerate(dets):
            confs.append(d.conf); correct.append(i in matched); cls.append(d.cls)
    return np.array(confs), np.array(correct, float), np.array(cls)


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = BINS):
    edges = np.linspace(0, 1, bins + 1)
    total, e, pts = len(conf), 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            acc, c = correct[m].mean(), conf[m].mean()
            e += m.sum() / total * abs(acc - c)
            pts.append((c, acc, int(m.sum())))
    return float(e), pts


def scale(conf: np.ndarray, T: float) -> np.ndarray:
    c = np.clip(conf, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-np.log(c / (1 - c)) / T))


def fit_temperature(conf: np.ndarray, correct: np.ndarray) -> float:
    best_t, best = 1.0, 1e18
    for T in np.linspace(0.3, 4.0, 371):
        p = np.clip(scale(conf, T), 1e-6, 1 - 1e-6)
        nll = -(correct * np.log(p) + (1 - correct) * np.log(1 - p)).mean()
        if nll < best:
            best_t, best = float(T), nll
    return best_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/sitesafe_base.onnx")
    a = ap.parse_args()
    det = Detector(a.model)
    out = RESULTS / "eval"
    out.mkdir(parents=True, exist_ok=True)

    vc, vy, vk = collect(det, "val")
    tc, ty, _ = collect(det, "test")
    T = fit_temperature(vc, vy)
    e_val_before, _ = ece(vc, vy)
    e_val_after, _ = ece(scale(vc, T), vy)
    e_before, pts_before = ece(tc, ty)
    e_after, pts_after = ece(scale(tc, T), ty)

    thresholds = {}
    for c in ITEM_OF_CLASS:  # only PPE and no_PPE classes decide an item
        m = vk == c
        thr = 0.45
        if m.sum() >= 20:
            for t in np.arange(0.30, 0.701, 0.05):
                sel = m & (vc >= t)
                if sel.sum() >= 10 and vy[sel].mean() >= TARGET_PRECISION:
                    thr = float(round(t, 2))
                    break
            else:
                thr = 0.70
        thresholds[str(c)] = thr

    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], "--", color="#888", label="perfect calibration")
    ax.plot([p[0] for p in pts_before], [p[1] for p in pts_before], "o-", color="#c0392b", label=f"raw (ECE {e_before:.3f})")
    ax.plot([p[0] for p in pts_after], [p[1] for p in pts_after], "s-", color="#1f6f43", label=f"temperature T={T:.2f} (ECE {e_after:.3f})")
    ax.set_xlabel("confidence"); ax.set_ylabel("share of detections that are correct")
    ax.set_title("Reliability diagram, test split"); ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out / "reliability.png", dpi=150); plt.close()

    result = {"model": a.model, "temperature": round(T, 3), "target_precision": TARGET_PRECISION,
              "ece_val_before": round(e_val_before, 4), "ece_val_after": round(e_val_after, 4),
              "ece_test_before": round(e_before, 4), "ece_test_after": round(e_after, 4),
              "class_thresholds": thresholds,
              "class_thresholds_named": {CLASS_NAMES[int(k)]: v for k, v in thresholds.items()},
              "n_val_detections": int(len(vc)), "n_test_detections": int(len(tc))}
    (out / "calibration.json").write_text(json.dumps(result, indent=2))
    (ROOT / "web" / "src" / "lib").mkdir(parents=True, exist_ok=True)  # absent in the Colab code bundle
    (ROOT / "web" / "src" / "lib" / "thresholds.json").write_text(json.dumps({
        "note": "Written by scripts/calibrate.py from the validation split. Do not edit by hand.",
        "class_thresholds": thresholds, "temperature": round(T, 3),
        "ece_before": round(e_before, 4), "ece_after": round(e_after, 4)}, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
