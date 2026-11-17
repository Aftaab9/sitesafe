"""Baseline 2 (zero-shot): open-vocabulary YOLO-World prompted with the PPE class names, no fine-tuning.

    python scripts/baseline_zeroshot.py --split test

Reports (a) detection metrics per class on the same split and (b) person-level compliance with the
same matching code as the fine-tuned model. The "no_*" classes are described in words; that an
open-vocabulary model handles "absence of an object" poorly is exactly what this baseline shows.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sitesafe.classes import CLASS_NAMES, NO_CLASSES, SITE_PROFILES  # noqa: E402
from sitesafe.evalutil import class_counts, compliance_metrics, load_split, person_rows, recall_table  # noqa: E402
from sitesafe.matching import Det, MatchConfig, match_people  # noqa: E402
from sitesafe.paths import RESULTS  # noqa: E402

PROMPTS = ["hard hat", "work gloves", "safety vest", "work boots", "safety goggles", "person",
           "bare head without hard hat", "face without goggles", "bare hands", "shoes", "forklift",
           "person without safety vest"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--weights", default="yolov8s-worldv2.pt")
    ap.add_argument("--conf", type=float, default=0.1)
    a = ap.parse_args()

    from ultralytics import YOLOWorld
    model = YOLOWorld(a.weights)
    model.set_classes(PROMPTS)

    def pred(rgb: np.ndarray) -> list[Det]:
        r = model.predict(rgb[..., ::-1], conf=a.conf, verbose=False, imgsz=640)[0]
        return [Det([float(v) for v in b], int(c), float(s)) for b, c, s in
                zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())]

    images = load_split(a.split)
    out = RESULTS / "eval"
    out.mkdir(parents=True, exist_ok=True)

    counts = class_counts(images, pred, conf=a.conf)
    rec = recall_table(counts)
    gt = sum(r[1] for r in counts); tp = sum(r[2] for r in counts); npred = sum(r[3] for r in counts)
    det_rows = [{"class": CLASS_NAMES[c], "prompt": PROMPTS[c], "gt": int(gt[c]), "recall": round(rec[CLASS_NAMES[c]], 4),
                 "precision": round(float(tp[c] / npred[c]), 4) if npred[c] else None} for c in range(len(CLASS_NAMES))]
    pd.DataFrame(det_rows).to_csv(out / f"baseline_zeroshot_{a.split}.csv", index=False)
    print(pd.DataFrame(det_rows).to_string(index=False))

    required = SITE_PROFILES["general"]
    # Open-vocabulary scores are much lower than a fine-tuned detector's, so the "unsure" thresholds are
    # lowered for this baseline; with the default 0.45 it would abstain on almost everyone.
    zs_cfg = MatchConfig(conf_unsure=a.conf, margin_unsure=0.0, person_unsure=a.conf)
    rows = person_rows(images, lambda rgb: match_people(pred(rgb), required, zs_cfg))
    pd.DataFrame(rows).to_csv(out / f"people_baseline_zeroshot_{a.split}.csv", index=False)
    m = {"tag": "baseline_zeroshot", "split": a.split, "weights": a.weights, "conf": a.conf,
         "no_class_mean_recall": round(float(np.nanmean([rec[CLASS_NAMES[c]] for c in NO_CLASSES])), 4),
         **compliance_metrics(rows)}
    (out / f"compliance_baseline_zeroshot_{a.split}.json").write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
