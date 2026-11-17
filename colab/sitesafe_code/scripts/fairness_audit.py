"""Bias check: person-detection recall and non-compliance recall by subgroup, with bootstrap intervals.

    python scripts/fairness_audit.py --tags            # 1) write automatic pre-tags to data/splits/test_tags.csv
    (open the CSV, correct the tags by hand, set verified=1)
    python scripts/fairness_audit.py                   # 2) compute subgroup results

Subgroups (per labelled person):
  lighting   dark / normal / bright      mean luminance of the image
  distance   near / mid / far            person height as a share of the image height
  clothing   hi-vis / dark / light / other   dominant torso colour
  skin tone  lighter / medium / darker / not visible   coarse three-level grouping in the spirit of the
             Monk scale, from the Individual Typology Angle (ITA) of skin pixels in the face box.

IMPORTANT. The automatic skin-tone and clothing tags are only a starting point: ITA is unreliable
under coloured light, shadow and small faces. They must be checked by eye before any claim is made,
and the output records how many tags were verified. Groups are apparent groups for an audit of the
MODEL; nothing is stored about any person and the tags never leave the repo's evaluation files.
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
from sitesafe.evalutil import bootstrap_ci, load_split, read_gt, read_rgb  # noqa: E402
from sitesafe.paths import MODELS, RESULTS, SPLITS  # noqa: E402
from sitesafe.privacy import FaceDetector  # noqa: E402

TAGS = SPLITS / "test_tags.csv"


def skin_group(rgb: np.ndarray, face) -> str:
    x1, y1, x2, y2 = [int(v) for v in face]
    w, h = x2 - x1, y2 - y1
    if w < 14 or h < 14:
        return "not visible"
    crop = rgb[max(0, y1 + h // 4):y1 + 3 * h // 4, max(0, x1 + w // 5):x2 - w // 5]  # cheeks / nose band
    if crop.size == 0:
        return "not visible"
    ycc = cv2.cvtColor(crop, cv2.COLOR_RGB2YCrCb).reshape(-1, 3)
    skin = (ycc[:, 1] > 135) & (ycc[:, 1] < 180) & (ycc[:, 2] > 85) & (ycc[:, 2] < 135)
    if skin.sum() < 20:
        return "not visible"
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).reshape(-1, 3)[skin].astype(np.float32)
    L = np.median(lab[:, 0]) * 100 / 255
    b = np.median(lab[:, 2]) - 128
    ita = np.degrees(np.arctan2(L - 50, b if abs(b) > 1e-3 else 1e-3))
    return "lighter" if ita > 41 else "medium" if ita > 10 else "darker"


def clothing_group(rgb: np.ndarray, box) -> str:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    crop = rgb[int(y1 + 0.25 * h):int(y1 + 0.65 * h), int(x1 + 0.2 * w):int(x2 - 0.2 * w)]
    if crop.size < 30:
        return "other"
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV).reshape(-1, 3)
    hh, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    hivis = (((hh >= 22) & (hh <= 48)) | ((hh >= 4) & (hh <= 20))) & (s > 110) & (v > 130)
    if hivis.mean() > 0.30:
        return "hi-vis"
    if (v < 70).mean() > 0.5:
        return "dark"
    if ((v > 170) & (s < 60)).mean() > 0.5:
        return "light"
    return "other"


def write_tags():
    faces = FaceDetector(str(MODELS / "face_detection_yunet_2023mar.onnx"), 0.5)
    rows = []
    for p in load_split("test"):
        rgb = read_rgb(p)
        h, w = rgb.shape[:2]
        lum = float(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).mean())
        lighting = "dark" if lum < 85 else "bright" if lum > 160 else "normal"
        fboxes = faces(rgb)
        for k, g in enumerate([g for g in read_gt(p, w, h) if g.cls == PERSON]):
            hf = (g.box[3] - g.box[1]) / h
            face = next((f for f in fboxes if g.box[0] <= (f[0] + f[2]) / 2 <= g.box[2] and g.box[1] <= (f[1] + f[3]) / 2 <= g.box[1] + 0.4 * (g.box[3] - g.box[1])), None)
            rows.append({"image": p.name, "person": k, "x1": round(g.box[0]), "y1": round(g.box[1]), "x2": round(g.box[2]), "y2": round(g.box[3]),
                         "lighting": lighting, "distance": "near" if hf > 0.6 else "mid" if hf > 0.3 else "far",
                         "clothing": clothing_group(rgb, g.box), "skin_tone": skin_group(rgb, face) if face else "not visible",
                         "verified": 0})
    if TAGS.exists():  # never overwrite hand corrections
        old = pd.read_csv(TAGS)
        ver = old[old["verified"] == 1].set_index(["image", "person"])
        for r in rows:
            key = (r["image"], r["person"])
            if key in ver.index:
                r.update({c: ver.loc[key, c] for c in ("lighting", "distance", "clothing", "skin_tone")}); r["verified"] = 1
    pd.DataFrame(rows).to_csv(TAGS, index=False)
    print(f"wrote {len(rows)} person tags to {TAGS}")
    print(pd.DataFrame(rows)[["lighting", "distance", "clothing", "skin_tone"]].apply(lambda c: c.value_counts()).fillna(0).astype(int))


def audit(tag: str):
    people = pd.read_csv(RESULTS / "eval" / f"people_{tag}_test.csv")
    people = people[people["gt_status"] != "no_gt_person"].copy()
    tags = pd.read_csv(TAGS)
    people["x1"] = people["box"].apply(lambda s: round(json.loads(s)[0]))
    people["y1"] = people["box"].apply(lambda s: round(json.loads(s)[1]))
    df = people.merge(tags, on=["image", "x1", "y1"], how="left")
    out_rows = []
    for dim in ("lighting", "distance", "clothing", "skin_tone"):
        for group, g in df.groupby(dim):
            det = g["detected"].astype(int).to_numpy()
            rec, lo, hi = bootstrap_ci(det, np.ones_like(det))
            nc = g[g["gt_status"] == "non_compliant"]
            caught = nc["pred_status"].isin(["non_compliant", "unsure"]).astype(int).to_numpy()
            nrec, nlo, nhi = bootstrap_ci(caught, np.ones_like(caught)) if len(nc) else (float("nan"),) * 3
            out_rows.append({"dimension": dim, "group": group, "n_people": len(g), "person_recall": round(rec, 4), "lo": round(lo, 4), "hi": round(hi, 4),
                             "n_non_compliant": len(nc), "nc_flagged_or_unsure": None if np.isnan(nrec) else round(nrec, 4),
                             "nc_lo": None if np.isnan(nlo) else round(nlo, 4), "nc_hi": None if np.isnan(nhi) else round(nhi, 4),
                             "small_group": len(g) < 30})
    res = pd.DataFrame(out_rows)
    res.to_csv(RESULTS / "eval" / "fairness_audit.csv", index=False)
    meta = {"n_people": int(len(df)), "tags_verified_by_hand": int(tags["verified"].sum()), "tags_total": int(len(tags))}
    (RESULTS / "eval" / "fairness_meta.json").write_text(json.dumps(meta, indent=2))
    print(res.to_string(index=False)); print(meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", action="store_true", help="(re)generate automatic pre-tags")
    ap.add_argument("--tag", default="base")
    a = ap.parse_args()
    if a.tags or not TAGS.exists():
        write_tags()
    if not a.tags:
        audit(a.tag)
