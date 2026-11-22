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
        m["training"] = {"note": (f"The shipped models were trained for {int(r.epochs)} epochs at {int(r.imgsz)} px on a free Colab T4 GPU "
                                  f"(about {float(r.train_minutes):.0f} minutes each, seed {int(r.seed)}). An earlier 30-epoch run on a laptop CPU "
                                  "took about 17 hours and is kept for comparison in results/laptop30/.")}

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
        no_drop = q["no_recall_fp32"] - q["no_recall_int8"]
        m["quantisation"] = {"fp32_mb": q["fp32_mb"], "int8_mb": q["int8_mb"], "map50_fp32": q["map50_fp32"], "map50_int8": q["map50_int8"],
                             "no_recall_fp32": q["no_recall_fp32"], "no_recall_int8": q["no_recall_int8"],
                             "shipped": (f"Int8 is {q['fp32_mb'] / q['int8_mb']:.1f} times smaller; it changes mAP@0.5 by {-100 * drop:+.1f} points and mean "
                                         f"“no_” recall by {-100 * no_drop:+.1f} points. Its CPU speed-up was not consistent across machines, and WebGPU "
                                         "has no fast path for int8, so the app ships the float32 model (10.7 MB, which still loads quickly).")}

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
            "note": ("Lighting, distance and clothing tags are automatic; skin tone was labelled by eye from head-and-shoulder crops "
                     "(lighter / medium / darker, or not visible when the face is turned away, tiny or blurred), because the automatic "
                     "colour-based grouping proved unreliable. Groups under 30 people have wide intervals and support no firm conclusion. "
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
    sc = load(E / "speed_cpu.json")
    if sc:
        speed += [{"label": "Python + ONNX Runtime, laptop CPU, 2 threads, whole photo (detector, face blur, matching)", "ms": sc["threads_2"]["mean_ms"]},
                  {"label": "Same, 4 threads", "ms": sc["threads_4"]["mean_ms"]},
                  {"label": "30-second clip at 3 frames per second, 2 threads (total)", "ms": 1000 * sc["clip_30s_at_3fps_seconds_2_threads"]}]
    sp = load(E / "speed_browser.json")
    if sp:
        speed += sp
    if speed:
        m["speed"] = speed

    fl = load(RESULTS / "failures" / "failures.json")
    if fl:
        m["failures"] = [{"image": i["image"], "note": i["note"]} for i in fl["items"]]

    o = load(E / "own_eval.json")
    if o:
        labels = {"sitesafe_base": "My model, real photos only", "sitesafe_synth": "My model + synthetic data",
                  "sitesafe_plus": "My model + extra real photos", "external_ppe_yolov8s": "Off-the-shelf YOLOv8s (13k real photos)",
                  "lfm3_omni": "LFM3-Omni 1.7B vision-language model (local)"}
        rows = []
        for tag, label in labels.items():
            r = o.get(tag)
            if not r:
                continue
            rows.append({"label": label, "person_recall": r["person_recall"], "accuracy": r["accuracy"], "decided": r["decided"],
                         "decidable": r["decidable_people"], "nc_caught": r["nc_flagged_or_unsure"], "nc_green": r["nc_shown_green"],
                         "unsure": r["unsure_share"], "no_vest_called": r.get("vest_M_shown_M"), "no_vest_shown_vest": r.get("vest_M_shown_W"),
                         "boots_shown_missing": r.get("boots_W_shown_M"), "no_helmet_called": r.get("helmet_M_shown_M"),
                         "false_person_boxes": r["false_person_boxes"],
                         "night_nc_caught": (o.get(f"{tag}:night") or {}).get("nc_flagged_or_unsure")})
        m["own"] = {"note": ("30 real photos from Wikimedia Commons (CC0, public domain, CC BY, CC BY-SA), 12 of them at night or in tunnels, "
                             "with 91 workers labelled by hand at person level (helmet, hi-vis, boots: worn, missing or not visible). "
                             "Only 19 workers are non-compliant (6 at night), so differences of a few people are within noise."),
                    "rows": rows}

    v = load(RESULTS / "vlm" / "hallucination.json")
    if v:
        m["vlm"] = v

    WEB_PUBLIC.mkdir(parents=True, exist_ok=True)
    (WEB_PUBLIC / "metrics.json").write_text(json.dumps(m, indent=1))
    (RESULTS / "metrics.json").write_text(json.dumps(m, indent=1))
    print("sections:", [k for k in m if k != "updated"])


if __name__ == "__main__":
    main()
