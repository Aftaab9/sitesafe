"""Baseline 1 (pre-trained + classical rules): COCO YOLO11n finds people, HSV colour rules decide PPE.

No PPE training at all. For every COCO "person" box:
* helmet: the head region (top 20 % of the box, middle 60 % of its width) is checked for typical
  hard-hat colours (white, yellow, orange, red, blue, green). Enough coloured pixels -> helmet.
* vest: the torso region (25-65 % of the height) is checked for high-visibility yellow-green and
  orange. Enough hi-vis pixels -> vest.

The rules output helmet / no_helmet / vest / no_vest pseudo-boxes, so the SAME person-level
compliance code scores the baseline and the fine-tuned model.

    python scripts/baseline_rules.py --split test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sitesafe.classes import SITE_PROFILES  # noqa: E402
from sitesafe.evalutil import compliance_metrics, load_split, person_rows  # noqa: E402
from sitesafe.matching import Det, match_people  # noqa: E402
from sitesafe.paths import RESULTS  # noqa: E402

HELMET_FRAC = 0.22   # share of head-region pixels that must look like a hard hat
VEST_FRAC = 0.14     # share of torso pixels that must be hi-vis


def helmet_mask(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    white = (s < 45) & (v > 185)
    vivid = (s > 110) & (v > 110) & ((h < 35) | (h > 165) | ((h > 95) & (h < 130)) | ((h > 40) & (h < 85)))
    return white | vivid


def vest_mask(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    yellow_green = (h >= 22) & (h <= 48) & (s > 95) & (v > 130)
    orange = (h >= 4) & (h <= 20) & (s > 140) & (v > 150)
    return yellow_green | orange


class RulesBaseline:
    def __init__(self, weights: str = "yolo11n.pt", conf: float = 0.35):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf

    def __call__(self, rgb: np.ndarray) -> list[Det]:
        res = self.model.predict(rgb[..., ::-1], classes=[0], conf=self.conf, verbose=False, imgsz=640)[0]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        H, W = rgb.shape[:2]
        dets: list[Det] = []
        for box, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = [float(v) for v in box]
            w, h = x2 - x1, y2 - y1
            dets.append(Det([x1, y1, x2, y2], 5, float(c)))
            if h < 24 or w < 10:
                continue
            head = [x1 + 0.2 * w, y1, x2 - 0.2 * w, y1 + 0.2 * h]
            torso = [x1 + 0.15 * w, y1 + 0.25 * h, x2 - 0.15 * w, y1 + 0.65 * h]
            for region, mask_fn, thr, pos, neg in ((head, helmet_mask, HELMET_FRAC, 0, 6), (torso, vest_mask, VEST_FRAC, 2, 11)):
                rx1, ry1, rx2, ry2 = [int(max(0, v)) for v in region]
                rx2, ry2 = min(W, rx2), min(H, ry2)
                if rx2 - rx1 < 3 or ry2 - ry1 < 3:
                    continue
                frac = float(mask_fn(hsv[ry1:ry2, rx1:rx2]).mean())
                # confidence grows with the distance from the decision threshold
                conf = float(np.clip(0.5 + 0.5 * abs(frac - thr) / max(thr, 1 - thr), 0.5, 0.99))
                dets.append(Det(region, pos if frac >= thr else neg, conf))
        return dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    base = RulesBaseline()
    required = SITE_PROFILES["general"]
    rows = person_rows(load_split(a.split), lambda rgb: match_people(base(rgb), required))
    out = RESULTS / "eval"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"people_baseline_rules_{a.split}.csv", index=False)
    m = {"tag": "baseline_rules", "split": a.split, "profile": "general", **compliance_metrics(rows)}
    (out / f"compliance_baseline_rules_{a.split}.json").write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
