"""Generative AI part 1c: does the synthetic data help? Recall on REAL test images, without vs with.

    python scripts/synth_experiment.py --a models/sitesafe_base.onnx --b models/sitesafe_synth.onnx

Both models were trained with the same seed, epochs and settings; the only difference is the extra
synthetic images. Recall is measured at confidence 0.25 and IoU 0.5 for every class, with a paired
bootstrap over test images (2,000 resamples) for the 95 % interval of the change. Also reports the
same comparison on (a) the diffusion "weather" versions of held-out images, if present, and (b) my own
30 images, if they are labelled.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sitesafe.classes import CLASS_NAMES, NO_CLASSES  # noqa: E402
from sitesafe.detector import Detector  # noqa: E402
from sitesafe.evalutil import bootstrap_recall_delta, class_counts, load_split  # noqa: E402
from sitesafe.paths import OWN, RESULTS, SYNTH  # noqa: E402


def compare(images, det_a, det_b, label: str) -> pd.DataFrame:
    ra = class_counts(images, lambda rgb: det_a(rgb, 0.25))
    rb = class_counts(images, lambda rgb: det_b(rgb, 0.25))
    rows = []
    for c in range(len(CLASS_NAMES)):
        r = bootstrap_recall_delta(ra, rb, [c])
        rows.append({"set": label, "class": CLASS_NAMES[c], **r})
    r = bootstrap_recall_delta(ra, rb, NO_CLASSES)
    rows.append({"set": label, "class": "ALL no_* classes", **r})
    r = bootstrap_recall_delta(ra, rb, list(range(len(CLASS_NAMES))))
    rows.append({"set": label, "class": "ALL classes", **r})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="models/sitesafe_base.onnx")
    ap.add_argument("--b", default="models/sitesafe_synth.onnx")
    ap.add_argument("--tag", default="diffusion")
    a = ap.parse_args()
    det_a, det_b = Detector(a.a), Detector(a.b)
    frames = [compare(load_split("test"), det_a, det_b, "real test (100 images)")]

    wl = sorted((SYNTH / "weatherlab_eval" / "images" / "train").glob("*.jpg"))
    if wl:
        frames.append(compare(wl, det_a, det_b, f"synthetic weather versions of test images ({len(wl)})"))
    own = sorted((OWN / "images").glob("*.*"))
    if own and any((OWN / "labels").glob("*.txt")):
        frames.append(compare(own, det_a, det_b, f"my own images ({len(own)})"))

    df = pd.concat(frames, ignore_index=True)
    for col in ("a", "b", "delta", "lo", "hi"):
        df[col] = df[col].round(4)
    df = df.rename(columns={"a": "recall_without", "b": "recall_with"})
    out = RESULTS / "synth"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"experiment_{a.tag}.csv", index=False)
    (out / f"experiment_{a.tag}.json").write_text(json.dumps(df.to_dict(orient="records"), indent=2))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
