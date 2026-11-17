"""Temporal smoothing study: violation precision / recall per frame, with vs without smoothing.

    python scripts/smoothing_study.py --model models/sitesafe_base.onnx

There is no labelled PPE video in the dataset, so clips are SIMULATED: a smooth pan and zoom over a
labelled test photo, with per-frame brightness flicker, sensor noise, motion blur and JPEG artefacts.
The crop is known for every frame, so the labels move with it exactly and each frame has ground truth.
Tracks that cannot be matched to a decidable labelled person are ignored (neither TP nor FP).
What this captures: detector flicker under small appearance changes. What it does not: real motion,
people entering and leaving, occlusion over time. That limit is stated in the report.

For every frame and every tracked worker: is a violation reported, and is it true?
  raw       = the single-frame decision
  smoothed  = missing in at least 3 of the last 5 sampled frames
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sitesafe.classes import SITE_PROFILES  # noqa: E402
from sitesafe.detector import Detector  # noqa: E402
from sitesafe.evalutil import load_split, read_gt  # noqa: E402
from sitesafe.matching import Det, iou, match_people  # noqa: E402
from sitesafe.paths import RESULTS  # noqa: E402
from sitesafe.tracking import Tracker, TrackerConfig  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_web_assets import ken_burns  # noqa: E402
from compliance_eval import load_match_config  # noqa: E402

SIZE = (960, 540)
REQUIRED = SITE_PROFILES["general"]


def degrade(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    f = frame.astype(np.float32) * rng.uniform(0.75, 1.2) + rng.normal(0, 7, frame.shape)
    f = np.clip(f, 0, 255).astype(np.uint8)
    if rng.random() < 0.35:
        k = int(rng.integers(3, 8))
        kernel = np.zeros((k, k), np.float32); kernel[k // 2] = 1 / k
        f = cv2.filter2D(f, -1, kernel)
    ok, enc = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 80))])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def gt_in_frame(gt, crop):
    x, y, cw, ch = crop
    sx, sy = SIZE[0] / cw, SIZE[1] / ch
    out = []
    for g in gt:
        b = [(g.box[0] - x) * sx, (g.box[1] - y) * sy, (g.box[2] - x) * sx, (g.box[3] - y) * sy]
        c = [min(max(b[0], 0), SIZE[0]), min(max(b[1], 0), SIZE[1]), min(max(b[2], 0), SIZE[0]), min(max(b[3], 0), SIZE[1])]
        full = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
        if (c[2] - c[0]) * (c[3] - c[1]) / full >= 0.6:  # keep labels that are mostly inside the frame
            out.append(Det(c, g.cls, 1.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/sitesafe_base.onnx")
    ap.add_argument("--clips", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--fps", type=int, default=3)
    a = ap.parse_args()

    det = Detector(a.model)
    cfg = load_match_config()
    rng = np.random.default_rng(42)
    images = [p for p in load_split("test")]
    rng.shuffle(images)
    stats = {k: {"tp": 0, "fp": 0, "fn": 0} for k in ("raw", "smoothed")}
    flips_raw = flips_smooth = tracked = id_switches = 0
    used = 0
    for ci, p in enumerate(images):
        if used >= a.clips:
            break
        bgr = cv2.imread(str(p))
        gt_full = read_gt(p, bgr.shape[1], bgr.shape[0])
        if sum(g.cls == 5 for g in gt_full) < 1:
            continue
        used += 1
        tracker = Tracker(TrackerConfig(window=5, min_hits=3))
        last_gt_of_track: dict[int, int] = {}
        prev_raw: dict[int, bool] = {}
        prev_smooth: dict[int, bool] = {}
        for frame, crop in ken_burns(bgr, a.seconds, a.fps, SIZE, seed=ci):
            frame = degrade(frame, rng)
            gt_people = match_people(gt_in_frame(gt_full, crop), REQUIRED, cfg)
            people = match_people(det(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 0.25), REQUIRED, cfg)
            seen = tracker.update(people)
            matched_gt = set()
            for t in seen:
                best, bi = None, 0.5
                for gi, g in enumerate(gt_people):
                    v = iou(t.box, g.box)
                    if v >= bi and gi not in matched_gt:
                        best, bi = gi, v
                raw_v = t.last.status == "non_compliant"
                smooth_v = tracker.smoothed_status(t, REQUIRED) == "non_compliant"
                if t.id in prev_raw:
                    flips_raw += raw_v != prev_raw[t.id]; flips_smooth += smooth_v != prev_smooth[t.id]; tracked += 1
                prev_raw[t.id], prev_smooth[t.id] = raw_v, smooth_v
                if best is None:
                    # No usable label: the person is mostly outside this crop, or not labelled as decidable.
                    # Such tracks are left out of precision and recall instead of being counted as false alarms.
                    truth = None
                else:
                    matched_gt.add(best)
                    if t.id in last_gt_of_track and last_gt_of_track[t.id] != best:
                        id_switches += 1
                    last_gt_of_track[t.id] = best
                    truth = gt_people[best].status
                for key, v in (("raw", raw_v), ("smoothed", smooth_v)):
                    if v and truth == "non_compliant":
                        stats[key]["tp"] += 1
                    elif v and truth == "compliant":
                        stats[key]["fp"] += 1
                    elif not v and truth == "non_compliant":
                        stats[key]["fn"] += 1
            for gi, g in enumerate(gt_people):  # violators the tracker did not see at all in this frame
                if gi not in matched_gt and g.status == "non_compliant":
                    stats["raw"]["fn"] += 1; stats["smoothed"]["fn"] += 1

    out = {"clips": used, "seconds": a.seconds, "fps": a.fps, "window": 5, "min_hits": 3, "id_switches": id_switches,
           "status_flips_per_100_track_frames_raw": round(100 * flips_raw / max(1, tracked), 2),
           "status_flips_per_100_track_frames_smoothed": round(100 * flips_smooth / max(1, tracked), 2)}
    for key, s in stats.items():
        out[key] = {**s, "precision": round(s["tp"] / max(1, s["tp"] + s["fp"]), 4), "recall": round(s["tp"] / max(1, s["tp"] + s["fn"]), 4)}
    (RESULTS / "eval").mkdir(parents=True, exist_ok=True)
    (RESULTS / "eval" / "smoothing_study.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
