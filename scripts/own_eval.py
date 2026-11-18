"""Evaluate models on my own 30 real images (Wikimedia Commons, see data/own_test/sources.csv).

    python scripts/own_eval.py --models models/sitesafe_base.onnx models/sitesafe_plus.onnx

Ground truth is PERSON-LEVEL (data/own_test/people_labels.json), labelled by hand while looking
at each image: for every worker, helmet / vest / boots are W (worn), M (visibly missing) or N (not
visible / cannot tell). "Hi-vis" counts any high-visibility garment (vest, jacket, coverall),
because that is what a site rule requires. People marked "ignore" (tiny, hidden, in a cab) are
neither scored nor counted as false detections.

Reported per model:
  person recall          share of scored workers the model detects (IoU >= 0.3 with my box)
  per item               for workers who wear the item: % shown worn / missing / unsure
                         for workers missing the item: % shown missing / worn / unsure
  compliance (general)   person-level decision for helmet + vest, same metrics as the test split
  false person boxes     detected people that match no labelled person
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sitesafe.classes import SITE_PROFILES  # noqa: E402
from sitesafe.detector import Detector  # noqa: E402
from sitesafe.matching import MatchConfig, iou, match_people  # noqa: E402
from sitesafe.paths import OWN, RESULTS  # noqa: E402

STATE = {"present": "W", "missing": "M", "unsure": "?", "not_visible": "?"}


def gt_status(p) -> str:
    if "M" in (p["helmet"], p["vest"]):
        return "non_compliant"
    if p["helmet"] == "W" and p["vest"] == "W":
        return "compliant"
    return "undecidable"


def make_detector(model: str):
    if model == "external":
        from sitesafe.external import ExternalDetector
        return ExternalDetector()
    return Detector(model)


def evaluate(model: str, labels: dict, conf: float = 0.25, cfg: MatchConfig | None = None):
    det = make_detector(model)
    cfg = cfg or MatchConfig()
    rows, fp = [], 0
    for name, info in labels.items():
        rgb = cv2.cvtColor(cv2.imread(str(OWN / "images" / name)), cv2.COLOR_BGR2RGB)
        dets = det(rgb, conf)
        items_people = match_people(dets, SITE_PROFILES["cutting_grinding"], cfg)  # item states for all five items
        general = match_people(dets, SITE_PROFILES["general"], cfg)                 # the decision the officer sees
        used = set()
        for g in info["people"]:
            best, bi = None, 0.3
            for k, q in enumerate(items_people):
                if k in used:
                    continue
                v = iou(q.box, g["box"])
                if v >= bi:
                    best, bi = k, v
            if best is not None:
                used.add(best)
            if g["ignore"]:
                continue
            q = items_people[best] if best is not None else None
            gen = next((x for x in general if x.det_index == q.det_index), None) if q else None
            rows.append({"image": name, "detected": q is not None, "gt_status": gt_status(g),
                         "pred_status": gen.status if gen else "not_detected",
                         **{f"gt_{i}": g[i] for i in ("helmet", "vest", "boots")},
                         **{f"pred_{i}": (STATE[q.items[i].state] if q else "-") for i in ("helmet", "vest", "boots")}})
        fp += sum(1 for k in range(len(items_people)) if k not in used)
    return pd.DataFrame(rows), fp


def summary(df: pd.DataFrame, fp: int) -> dict:
    out = {"scored_people": len(df), "person_recall": round(df.detected.mean(), 3), "false_person_boxes": fp}
    det = df[df.detected]
    for item in ("helmet", "vest", "boots"):
        for truth in ("W", "M"):
            sub = det[det[f"gt_{item}"] == truth]
            if len(sub):
                out[f"{item}_{truth}_n"] = len(sub)
                for p in ("W", "M", "?"):
                    out[f"{item}_{truth}_shown_{p}"] = round((sub[f"pred_{item}"] == p).mean(), 3)
    dec = df[df.gt_status.isin(["compliant", "non_compliant"])]
    decided = dec[dec.pred_status.isin(["compliant", "non_compliant"])]
    nc = dec[dec.gt_status == "non_compliant"]
    tp = ((decided.gt_status == "non_compliant") & (decided.pred_status == "non_compliant")).sum()
    fpn = ((decided.gt_status == "compliant") & (decided.pred_status == "non_compliant")).sum()
    out.update({"decidable_people": len(dec), "decided": len(decided),
                "accuracy": round((decided.gt_status == decided.pred_status).mean(), 3) if len(decided) else None,
                "precision_nc": round(tp / (tp + fpn), 3) if tp + fpn else None,
                "nc_people": len(nc),
                "nc_flagged_or_unsure": round(nc.pred_status.isin(["non_compliant", "unsure"]).mean(), 3) if len(nc) else None,
                "nc_shown_green": round((nc.pred_status == "compliant").mean(), 3) if len(nc) else None,
                "compliant_shown_red": round(((dec.gt_status == "compliant") & (dec.pred_status == "non_compliant")).sum()
                                             / max(1, (dec.gt_status == "compliant").sum()), 3),
                "unsure_share": round((dec.pred_status == "unsure").mean(), 3)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["models/sitesafe_base.onnx"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--raw", action="store_true", help="flat 0.45 unsure threshold instead of the calibrated ones")
    a = ap.parse_args()
    labels = json.loads((OWN / "people_labels.json").read_text())
    res = {}
    for m in a.models:
        if m != "external" and not Path(m).exists():
            print("missing", m)
            continue
        tag = "external_ppe_yolov8s" if m == "external" else Path(m).stem
        # Same "unsure" thresholds the app uses: each model's own calibration from the validation split.
        cfg = None
        if m != "external" and not a.raw:
            from compliance_eval import load_match_config
            cfg = load_match_config(tag.removeprefix("sitesafe_"))
        df, fp = evaluate(m, labels, a.conf, cfg)
        (RESULTS / "eval").mkdir(parents=True, exist_ok=True)
        df.to_csv(RESULTS / "eval" / f"own_people_{tag}.csv", index=False)
        res[tag] = summary(df, fp)
        night = {"c041.jpg", "c050.jpg", "c051.jpg", "c091.jpg", "c105.jpg", "c106.jpg", "c107.jpg", "c109.jpg",
                 "c110.jpg", "c116.jpg", "c118.jpg", "c119.jpg"}
        for part, sel in (("night", df.image.isin(night)), ("day", ~df.image.isin(night))):
            sub = summary(df[sel], 0)
            res[f"{tag}:{part}"] = {k: sub[k] for k in ("scored_people", "person_recall", "accuracy", "decided",
                                                        "nc_flagged_or_unsure", "unsure_share")}
    (RESULTS / "eval" / "own_eval.json").write_text(json.dumps(res, indent=2))
    print(pd.DataFrame(res).to_string())


if __name__ == "__main__":
    main()
