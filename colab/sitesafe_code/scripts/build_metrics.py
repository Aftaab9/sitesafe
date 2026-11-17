"""Collect every evaluation output into one file used by the web model card and the report.

    python scripts/build_metrics.py

Writes web/public/metrics.json and results/summary.md. Sections whose inputs are missing are left
out, so the page always shows exactly what has actually been measured.
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sitesafe.paths import RESULTS, WEB_PUBLIC  # noqa: E402

E = RESULTS / "eval"


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def detection(tag: str, split: str, label: str):
    s, c = load(E / f"{tag}_{split}.json"), E / f"{tag}_{split}.csv"
    if not s or not c.exists():
        return None
    return {"label": label, "split": split, "map50": s["map50"], "map50_95": s["map50_95"],
            "no_class_mean_recall": s["no_class_mean_recall"], "per_class": pd.read_csv(c).to_dict(orient="records")}


def main():
    m: dict = {"updated": datetime.date.today().isoformat()}
    runs = pd.read_csv(RESULTS / "runs.csv") if (RESULTS / "runs.csv").exists() else None
    if runs is not None:
        r = runs.iloc[0]
        m["training"] = {"note": f"The shipped model was trained for {int(r.epochs)} epochs at {int(r.imgsz)} px on a laptop CPU "
                                 f"(seed {int(r.seed)}). The Colab notebooks repeat the run for 100 epochs on a free T4 GPU."}

    det = {}
    for tag, label in (("base", "Real photos only"), ("synth", "Real + synthetic")):
        d = detection(tag, "test", label)
        if d:
            det[tag] = d
    if det:
        m["detection"] = det

    comp = []
    for tag, label in (("base", "Fine-tuned YOLO11n (this model)"), ("synth", "Fine-tuned YOLO11n, real + synthetic"),
                       ("baseline_rules", "Baseline: COCO YOLO11n + colour rules"),
                       ("baseline_zeroshot", "Baseline: zero-shot YOLO-World")):
        c = load(E / f"compliance_{tag}_test.json")
        if c:
            comp.append({"label": label, "n": c["gt_decidable"], "accuracy": c["accuracy"], "precision_nc": c["precision_nc"],
                         "recall_nc": c["recall_nc"], "unsure_share": c["unsure_share"], "nc_shown_green": c["nc_shown_green"],
                         "person_recall": c["person_recall"], "decided": c["decided"]})
    if comp:
        m["compliance"] = comp

    ex = RESULTS / "synth" / "experiment_diffusion.csv"
    if ex.exists():
        df = pd.read_csv(ex)
        real = df[df["set"].str.startswith("real test")]
        rows = real[real["class"].str.startswith(("no_", "ALL"))]
        filt = RESULTS / "synth" / "filter_diffusion.csv"
        kept = int(pd.read_csv(filt)["keep"].sum()) if filt.exists() else None
        m["synthetic"] = {
            "note": (f"{kept} diffusion-generated images (night, rain, fog, dust) were added to 700 real training images. Same seed and epochs for both "
                     "models. Intervals are from a paired bootstrap over the 100 real test images; an interval that includes zero means "
                     "the change is not demonstrated. The real test photos are almost all clear daylight, so this measures whether the synthetic "
                     "data harms or helps ordinary conditions; robustness in bad weather is measured separately on generated versions."),
            "rows": [{"class": r["class"], "base": r.recall_without, "synth": r.recall_with, "delta": r.delta, "lo": r.lo, "hi": r.hi}
                     for _, r in rows.iterrows()]}
        stress = df[df["set"].str.startswith("synthetic weather")]
        if len(stress):
            s = stress[stress["class"].str.startswith("ALL")]
            m["synthetic"]["stress"] = [{"class": r["class"], "base": r.recall_without, "synth": r.recall_with, "delta": r.delta,
                                         "lo": r.lo, "hi": r.hi} for _, r in s.iterrows()]

    q = load(E / "quantisation.json")
    if q:
        drop = q["map50_fp32"] - q["map50_int8"]
        m["quantisation"] = {"fp32_mb": q["fp32_mb"], "int8_mb": q["int8_mb"], "map50_fp32": q["map50_fp32"], "map50_int8": q["map50_int8"],
                             "shipped": ("The app ships the float32 model: " + (f"int8 costs {100 * drop:.1f} points of mAP@0.5 for a "
                                         f"{q['fp32_mb'] - q['int8_mb']:.1f} MB saving, and WebGPU runs float models faster than quantised ones."
                                         if drop > 0.01 else "the int8 model is within one point of mAP@0.5, but WebGPU has no fast path for quantised operators."))}

    b = load(E / "blur_study.json")
    if b:
        o, f = b["original"], b["faces"]
        hurt = (o["map50"] - f["map50"]) > 0.005 or (o["helmet_recall"] or 0) - (f["helmet_recall"] or 0) > 0.01
        m["blur"] = {"map50_original": o["map50"], "map50_blurred": f["map50"], "map50_strict": b["strict"]["map50"],
                     "helmet_recall_original": o["helmet_recall"], "helmet_recall_blurred": f["helmet_recall"],
                     "goggles_recall_original": o["goggles_recall"], "goggles_recall_blurred": f["goggles_recall"],
                     "decision": ("Blurring before detection costs accuracy, so the app detects on the original frame held in memory and blurs only what is displayed or exported."
                                  if hurt else "Blurring faces first changes detection very little, but the app still detects on the original frame in memory and blurs only the output: "
                                  "it costs nothing in privacy (the frame never leaves the device) and avoids the larger loss seen when whole head regions are blurred.")}

    c = load(E / "calibration.json")
    cb = load(E / "compliance_base_test.json")
    if c:
        m["calibration"] = {"ece_before": c["ece_test_before"], "ece_after": c["ece_test_after"], "temperature": c["temperature"],
                            "unsure_share": cb["unsure_share"] if cb else None}

    fa = E / "fairness_audit.csv"
    if fa.exists():
        df = pd.read_csv(fa)
        meta = load(E / "fairness_meta.json") or {}
        m["subgroups"] = {
            "note": (f"Subgroup tags are automatic estimates ({meta.get('tags_verified_by_hand', 0)} of {meta.get('tags_total', 0)} checked by hand so far); "
                     "skin-tone results are withheld until the tags are checked by hand, because the automatic colour-based grouping proved unreliable "
                     "(it put almost every visible face in one group). Groups under 30 people have wide intervals and support no firm conclusion. "
                     "“no_” recall counts a non-compliant worker as caught when the model says missing PPE or check manually."),
            "rows": [{"group": f"{r.dimension}: {r.group}", "n": int(r.n_people), "person_recall": r.person_recall, "lo": r.lo, "hi": r.hi,
                      "no_recall": None if pd.isna(r.nc_flagged_or_unsure) else r.nc_flagged_or_unsure} for r in df.itertuples()
                     # Automatic skin-tone tags are not shown until they have been checked by hand.
                     if r.dimension != "skin_tone" or meta.get("tags_verified_by_hand", 0) > 0]}

    s = load(E / "smoothing_study.json")
    if s:
        m["smoothing"] = {"precision_raw": s["raw"]["precision"], "precision_smoothed": s["smoothed"]["precision"],
                          "recall_raw": s["raw"]["recall"], "recall_smoothed": s["smoothed"]["recall"], "clips": s["clips"],
                          "flips_raw": s["status_flips_per_100_track_frames_raw"], "flips_smoothed": s["status_flips_per_100_track_frames_smoothed"]}

    speed = []
    ex_b = load(E / "export_sitesafe_base.json")
    if ex_b:
        speed += [{"label": "ONNX Runtime, laptop CPU, 2 threads (model only)", "ms": ex_b["cpu_ms_2_threads"]},
                  {"label": "ONNX Runtime, laptop CPU, 4 threads (model only)", "ms": ex_b["cpu_ms_4_threads"]}]
    sp = load(E / "speed_browser.json")
    if sp:
        speed += sp
    if speed:
        m["speed"] = speed

    fl = load(RESULTS / "failures" / "failures.json")
    if fl:
        m["failures"] = [{"image": i["image"], "note": i["note"]} for i in fl["items"]]

    v = load(RESULTS / "vlm" / "hallucination.json")
    if v:
        m["vlm"] = v

    WEB_PUBLIC.mkdir(parents=True, exist_ok=True)
    (WEB_PUBLIC / "metrics.json").write_text(json.dumps(m, indent=1))
    (RESULTS / "metrics.json").write_text(json.dumps(m, indent=1))
    print("sections:", [k for k in m if k != "updated"])


if __name__ == "__main__":
    main()
