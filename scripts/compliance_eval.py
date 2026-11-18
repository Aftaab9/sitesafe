"""Person-level compliance evaluation of the fine-tuned detector on a held-out split.

    python scripts/compliance_eval.py --model models/sitesafe_base.onnx --tag base --split test

Ground truth per person is derived from the dataset labels with the same matching rules
(labels have confidence 1). A sample of these derived labels should be hand-checked; the per-person
CSV written here (results/eval/people_<tag>_<split>.csv) is the sheet to check against.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sitesafe.classes import SITE_PROFILES  # noqa: E402
from sitesafe.detector import Detector  # noqa: E402
from sitesafe.evalutil import compliance_metrics, load_split, person_rows  # noqa: E402
from sitesafe.matching import MatchConfig, match_people  # noqa: E402
from sitesafe.paths import RESULTS  # noqa: E402


def load_match_config(tag: str = "base") -> MatchConfig:
    """Per-class thresholds from this model's calibration (calibration_<tag>.json), else the base one."""
    f = RESULTS / "eval" / f"calibration_{tag}.json"
    if tag == "base" or not f.exists():
        f = RESULTS / "eval" / "calibration.json"
    if f.exists():
        th = json.loads(f.read_text()).get("class_thresholds", {})
        return MatchConfig(class_thresholds={int(k): float(v) for k, v in th.items()})
    return MatchConfig()


def run(model: str, tag: str, split: str, profile: str = "general", conf: float = 0.25, images=None) -> dict:
    if model == "lfm":  # LFM3-Omni through a local Ollama
        from sitesafe.lfm import LfmDetector
        det = LfmDetector()
    else:
        det = Detector(model)
    cfg = load_match_config(tag)
    required = SITE_PROFILES[profile]
    rows = person_rows(images or load_split(split), lambda rgb: match_people(det(rgb, conf), required, cfg), profile, cfg)
    out = RESULTS / "eval"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"people_{tag}_{split}.csv", index=False)
    m = {"tag": tag, "split": split, "profile": profile, "conf": conf, **compliance_metrics(rows)}
    (out / f"compliance_{tag}_{split}.json").write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/sitesafe_base.onnx")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--split", default="test")
    ap.add_argument("--profile", default="general")
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()
    run(a.model, a.tag, a.split, a.profile, a.conf)
