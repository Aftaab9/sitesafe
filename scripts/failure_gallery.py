"""Failure gallery: at least 10 real failures on the test split, each with a one-line explanation.

    python scripts/failure_gallery.py --tag base

Reads the per-person sheet written by compliance_eval.py, picks the worst and most varied errors
(violator shown green, compliant worker flagged, worker not found), crops each with context, blurs
faces, draws the labelled box (white) and writes a short cause chosen by simple measurements
(size, brightness, overlap with other people). Explanations are automatic first guesses and are
meant to be edited by hand in results/failures/failures.json.
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
from sitesafe.classes import PERSON  # noqa: E402
from sitesafe.evalutil import load_split, read_gt  # noqa: E402
from sitesafe.matching import iou  # noqa: E402
from sitesafe.paths import MODELS, RESULTS, WEB_PUBLIC  # noqa: E402
from sitesafe.privacy import FaceDetector, blur_regions  # noqa: E402

KIND_TEXT = {
    "green": "A worker with PPE missing ({missing}) was shown as compliant",
    "false_alarm": "A compliant worker was flagged for missing {pred_missing}",
    "missed": "A worker was not detected at all",
}


def cause(row, others, lum) -> str:
    reasons = []
    if row["height_frac"] < 0.25:
        reasons.append("the worker is small and far from the camera")
    if lum < 80:
        reasons.append("the scene is dark")
    overlap = max([iou(row["box"], o) for o in others] or [0])
    if overlap > 0.15:
        reasons.append("the worker overlaps another person, so PPE boxes can be matched to the wrong one")
    b = row["box"]
    if b[0] < 4 or b[1] < 4 or b[2] > row["img_w"] - 4 or b[3] > row["img_h"] - 4:
        reasons.append("the worker is cut off by the image edge")
    if not reasons:
        reasons.append("unusual pose, PPE colour or partial occlusion")
    return "; ".join(reasons[:2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    df = pd.read_csv(RESULTS / "eval" / f"people_{a.tag}_test.csv").fillna("")
    df["box"] = df["box"].apply(json.loads)
    df = df[df["gt_status"].isin(["compliant", "non_compliant"])]
    green = df[(df.gt_status == "non_compliant") & (df.pred_status == "compliant")].assign(kind="green")
    alarm = df[(df.gt_status == "compliant") & (df.pred_status == "non_compliant")].assign(kind="false_alarm")
    missed = df[~df.detected].assign(kind="missed")
    # interleave kinds, one per image first, so the gallery is varied
    pools = [green.sort_values("pred_conf", ascending=False), alarm.sort_values("pred_conf", ascending=False),
             missed.sort_values("height_frac", ascending=False)]
    picks, seen_img = [], set()
    while len(picks) < a.n and any(len(p) for p in pools):
        for i, p in enumerate(pools):
            fresh = p[~p.image.isin(seen_img)]
            src = fresh if len(fresh) else p
            if len(src) and len(picks) < a.n:
                r = src.iloc[0]
                picks.append(r); seen_img.add(r.image)
                pools[i] = p.drop(r.name)

    paths = {p.name: p for p in load_split("test")}
    faces = FaceDetector(str(MODELS / "face_detection_yunet_2023mar.onnx"), 0.5)
    out_dir = RESULTS / "failures"; web_dir = WEB_PUBLIC / "failures"
    for d in (out_dir, web_dir):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("f*.jpg"):
            old.unlink()
    items = []
    for i, r in enumerate(picks, 1):
        bgr = cv2.imread(str(paths[r.image]))
        H, W = bgr.shape[:2]
        gt_p = [g.box for g in read_gt(paths[r.image], W, H) if g.cls == PERSON]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # faces only: blurring the whole head would hide the evidence (helmet or no helmet)
        rgb = blur_regions(rgb, faces(rgb), expand=0.05)
        lum = float(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).mean())
        x1, y1, x2, y2 = r.box
        others = [b for b in gt_p if abs(b[0] - x1) > 2 or abs(b[1] - y1) > 2]
        bw, bh = x2 - x1, y2 - y1
        cx1, cy1 = int(max(0, x1 - 0.8 * bw)), int(max(0, y1 - 0.25 * bh))
        cx2, cy2 = int(min(W, x2 + 0.8 * bw)), int(min(H, y2 + 0.2 * bh))
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        col = {"green": (60, 60, 255), "false_alarm": (0, 170, 255), "missed": (255, 255, 255)}[r.kind]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, max(2, W // 400))
        crop = img[cy1:cy2, cx1:cx2]
        s = 520 / max(crop.shape[:2])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))), interpolation=cv2.INTER_AREA)
        name = f"f{i:02d}.jpg"
        cv2.imwrite(str(out_dir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 82])
        cv2.imwrite(str(web_dir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 78])
        text = KIND_TEXT[r.kind].format(missing=r.gt_missing.replace("|", ", ") or "PPE", pred_missing=r.pred_missing.replace("|", ", ") or "PPE")
        items.append({"image": f"failures/{name}", "source": r.image, "kind": r.kind,
                      "note": f"{text}: {cause(r, others, lum)}."})
    summary = {"violator_shown_green": int(len(green)), "false_alarms": int(len(alarm)), "workers_not_detected": int(len(missed)),
               "decidable_people": int(len(df))}
    (out_dir / "failures.json").write_text(json.dumps({"summary": summary, "items": items}, indent=2))
    print(json.dumps(summary, indent=2))
    for it in items:
        print(" -", it["note"])


if __name__ == "__main__":
    main()
