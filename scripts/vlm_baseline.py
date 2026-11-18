"""Can a small general vision-language model do the PPE check? LFM3-Omni (1.7B) vs my detector.

    ollama pull lfm3-omni        (Liquid AI, LFM Open License; runs locally, nothing leaves the laptop)
    python scripts/vlm_baseline.py                  # own 30 photos + test split
    python scripts/vlm_baseline.py --sets own --limit 5

Each labelled worker is cropped (with a margin) and the model is asked, with a JSON schema that only
allows yes / no / unsure, whether the person wears a hard hat and a hi-vis vest or jacket. The model is
given the ground-truth person boxes, so it is judged on recognising PPE only, which flatters it:
my detector must also find the person itself. Both are scored with the same code (own_eval.summary)
on the same people. Answers are cached in results/vlm_baseline/responses.jsonl, so a run can resume.

Writes results/vlm_baseline/vlm_baseline.json and per-person CSVs.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import own_eval  # noqa: E402
from build_web_assets import read_gt  # noqa: E402
from compliance_eval import load_match_config  # noqa: E402
from sitesafe.classes import SITE_PROFILES  # noqa: E402
from sitesafe.matching import match_people  # noqa: E402
from sitesafe.paths import OWN, RESULTS, ROOT, SPLITS  # noqa: E402

OUT = RESULTS / "vlm_baseline"
URL = "http://localhost:11434/api/generate"
SCHEMA = {"type": "object", "properties": {"hard_hat": {"enum": ["yes", "no", "unsure"]},
                                           "hi_vis": {"enum": ["yes", "no", "unsure"]}},
          "required": ["hard_hat", "hi_vis"]}
PROMPT = ("The image shows one worker on a construction site. Answer about this person only. "
          "hard_hat: is the person wearing a hard hat (safety helmet) on their head? "
          "hi_vis: is the person wearing a high-visibility vest or jacket (fluorescent yellow, green or orange, "
          "often with reflective stripes)? Answer yes or no; answer unsure only if that part of the body cannot be seen.")
MAX_CROP = 768
ANS = {"yes": "W", "no": "M", "unsure": "?"}
STATE = {"present": "W", "missing": "M", "not_visible": "N", "unsure": "N"}


def test_labels() -> dict:
    """Test split in the own_eval label format: person boxes with helmet / vest / boots worn, missing or not visible."""
    labels = {}
    for ln in (SPLITS / "test.txt").read_text().splitlines():
        if not ln.strip():
            continue
        p = ROOT / ln
        h, w = cv2.imread(str(p)).shape[:2]
        gt = read_gt(p, w, h)
        people = match_people([d for d in gt], SITE_PROFILES["cutting_grinding"])
        labels[str(p)] = {"w": w, "h": h, "people": [
            {"box": [round(v) for v in q.box], "ignore": False,
             **{i: STATE[q.items[i].state] for i in ("helmet", "vest", "boots")}} for q in people]}
    return labels


def ask(model: str, crop) -> tuple[dict, float]:
    b64 = base64.b64encode(cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])[1]).decode()
    body = json.dumps({"model": model, "prompt": PROMPT, "images": [b64], "stream": False, "format": SCHEMA,
                       "options": {"temperature": 0, "num_ctx": 4096}}).encode()
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=600))
    try:
        ans = json.loads(r["response"])
    except json.JSONDecodeError:
        ans = {}
    return ans, time.time() - t


def run_vlm(model: str, set_name: str, labels: dict, cache: dict, cache_f, limit: int | None) -> pd.DataFrame:
    rows, n_img = [], 0
    for name, info in labels.items():
        if limit and n_img >= limit:
            break
        n_img += 1
        path = Path(name) if Path(name).is_absolute() else OWN / "images" / name
        im = cv2.imread(str(path))
        H, W = im.shape[:2]
        for k, g in enumerate(info["people"]):
            if g["ignore"]:
                continue
            key = f"{model}|{set_name}|{Path(name).name}|{k}"
            if key not in cache:
                x1, y1, x2, y2 = g["box"]
                mw, mh = (x2 - x1) * 0.15, (y2 - y1) * 0.10
                crop = im[max(0, int(y1 - mh)):min(H, int(y2 + mh)), max(0, int(x1 - mw)):min(W, int(x2 + mw))]
                f = min(1.0, MAX_CROP / max(crop.shape[:2]))  # big crops cost many image tokens (12 s vs 1.3 s)
                if f < 1:
                    crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
                ans, sec = ask(model, crop)
                cache[key] = {"ans": ans, "sec": round(sec, 2)}
                cache_f.write(json.dumps({"key": key, **cache[key]}) + "\n"); cache_f.flush()
            a = cache[key]["ans"]
            ph, pv = ANS.get(a.get("hard_hat"), "?"), ANS.get(a.get("hi_vis"), "?")
            status = "non_compliant" if "M" in (ph, pv) else "compliant" if ph == pv == "W" else "unsure"
            rows.append({"image": Path(name).name, "detected": True, "gt_status": own_eval.gt_status(g), "pred_status": status,
                         **{f"gt_{i}": g[i] for i in ("helmet", "vest", "boots")},
                         "pred_helmet": ph, "pred_vest": pv, "pred_boots": "-", "sec": cache[key]["sec"]})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lfm3-omni")
    ap.add_argument("--sets", nargs="+", default=["own", "test"])
    ap.add_argument("--limit", type=int, default=None, help="images per set (testing)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cache_path = OUT / "responses.jsonl"
    cache = {}
    if cache_path.exists():
        for ln in cache_path.read_text(encoding="utf-8").splitlines():
            d = json.loads(ln); cache[d.pop("key")] = d
    cfg = load_match_config()
    res = {"model": a.model, "prompt": PROMPT,
           "note": ("The language model is given the true person boxes and judges helmet and hi-vis on a crop; "
                    "the detector has to find each person itself. Same people, same scoring code.")}
    with cache_path.open("a", encoding="utf-8") as cf:
        for s in a.sets:
            labels = json.loads((OWN / "people_labels.json").read_text()) if s == "own" else test_labels()
            if a.limit:
                labels = dict(list(labels.items())[:a.limit])
            vdf = run_vlm(a.model, s, labels, cache, cf, a.limit)
            ddf, fp = own_eval.evaluate("models/sitesafe_base.onnx", labels, 0.25, cfg)
            vdf.to_csv(OUT / f"people_vlm_{s}.csv", index=False)
            ddf.to_csv(OUT / f"people_detector_{s}.csv", index=False)
            vs, ds = own_eval.summary(vdf.drop(columns="sec"), 0), own_eval.summary(ddf, fp)
            vs["sec_per_person"] = round(vdf.sec.median(), 2)
            res[s] = {"vlm": vs, "detector": ds}
            keys = ["scored_people", "person_recall", "decided", "accuracy", "unsure_share", "nc_flagged_or_unsure",
                    "nc_shown_green", "compliant_shown_red", "helmet_M_shown_M", "helmet_W_shown_W", "vest_M_shown_M",
                    "vest_M_shown_W", "vest_W_shown_W"]
            print(f"\n== {s} ==")
            print(pd.DataFrame({"LFM3-Omni (true boxes)": {k: vs.get(k) for k in keys},
                                "my detector": {k: ds.get(k) for k in keys}}).to_string())
            print("median seconds per person (LFM3-Omni):", vs["sec_per_person"])
    (OUT / "vlm_baseline.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
