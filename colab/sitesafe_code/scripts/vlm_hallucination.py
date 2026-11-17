"""Generative AI part 2: hallucination rate of the incident narrative, before and after verification.

    python scripts/vlm_hallucination.py --n 60

The same small open VLM as in the browser (SmolVLM-256M-Instruct) is asked for the same fixed JSON
on face-blurred test images. Each claim is then scored against the LABELS (ground truth):
  before verification = share of the VLM's own claims that are wrong
  after verification  = share of claims in the final text that are wrong, once every claim has been
                        cross-checked against the DETECTOR and corrected (the detector is not perfect,
                        so this is not zero; it is the detector's own error rate on those claims)
Claims: number of workers, number without helmet, number without vest, forklift present.
Also reports JSON validity and seconds per image on this CPU.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sitesafe.classes import FORKLIFT, PERSON, SITE_PROFILES  # noqa: E402
from sitesafe.detector import Detector  # noqa: E402
from sitesafe.evalutil import load_split, read_gt, read_rgb  # noqa: E402
from sitesafe.matching import match_people, summarise  # noqa: E402
from sitesafe.paths import MODELS, RESULTS  # noqa: E402
from sitesafe.privacy import FaceDetector, blur_regions, privacy_boxes  # noqa: E402

MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"
PROMPT = ('You are a construction site safety assistant. Look at the photo and answer ONLY with JSON in exactly this form: '
          '{"workers": <number of people>, "missing_helmet": <people without a hard hat>, "missing_vest": <people without a high-visibility vest>, '
          '"missing_gloves": <people without gloves>, "missing_boots": <people without safety boots>, "missing_goggles": <people without safety goggles>, '
          '"forklift_present": <true or false>, "scene": "<one short factual sentence about the place and conditions>"}')
REQUIRED = SITE_PROFILES["general"]
KEYS = ["workers", "missing_helmet", "missing_vest", "forklift_present"]


def parse(raw: str):
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return None
    for cand in (m.group(0), re.sub(r",\s*}", "}", m.group(0)).replace("'", '"')):
        try:
            o = json.loads(cand)
            num = lambda v: max(0, int(float(v))) if str(v).replace(".", "", 1).isdigit() else 0  # noqa: E731
            return {"workers": num(o.get("workers", 0)), "missing_helmet": num(o.get("missing_helmet", 0)),
                    "missing_vest": num(o.get("missing_vest", 0)),
                    "forklift_present": str(o.get("forklift_present", "")).lower() in ("true", "yes", "1")}
        except Exception:
            continue
    return None


def facts(summary: dict) -> dict:
    return {"workers": summary["persons"], "missing_helmet": summary["missing_by_item"]["helmet"],
            "missing_vest": summary["missing_by_item"]["vest"], "forklift_present": summary["forklifts"] > 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    a = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    det = Detector(str(MODELS / "sitesafe_base.onnx"))
    faces = FaceDetector(str(MODELS / "face_detection_yunet_2023mar.onnx"))
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)

    rows = []
    for i, p in enumerate(load_split("test")[: a.n]):
        rgb = read_rgb(p, 1024)
        h, w = rgb.shape[:2]
        gt = read_gt(p, w, h)
        truth = facts(summarise(match_people(gt, REQUIRED), gt))
        dets = det(rgb, 0.25)
        people = match_people(dets, REQUIRED)
        detector = facts(summarise(people, dets))
        blurred = blur_regions(rgb, privacy_boxes(faces(rgb), [q.box for q in people], True))  # the VLM only sees blurred faces
        t0 = time.time()
        inputs = proc(text=text, images=[Image.fromarray(blurred)], return_tensors="pt")
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False, repetition_penalty=1.1)
        raw = proc.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        secs = time.time() - t0
        claim = parse(raw)
        row = {"image": p.name, "seconds": round(secs, 2), "json_valid": claim is not None, "raw": raw[:300]}
        for k in KEYS:
            row[f"truth_{k}"] = truth[k]
            row[f"vlm_{k}"] = claim[k] if claim else None
            row[f"det_{k}"] = detector[k]
            row[f"vlm_wrong_{k}"] = (claim[k] != truth[k]) if claim else None
            # after verification the published value is always the detector's
            row[f"final_wrong_{k}"] = detector[k] != truth[k]
            row[f"caught_{k}"] = (claim[k] != detector[k]) if claim else None
        rows.append(row)
        print(f"[{i + 1}/{a.n}] {secs:.1f}s valid={claim is not None} {raw[:90]!r}", flush=True)

    df = pd.DataFrame(rows)
    out = RESULTS / "vlm"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "hallucination_rows.csv", index=False)
    valid = df[df.json_valid]
    summary = {"model": MODEL, "images": len(df), "json_valid_share": round(float(df.json_valid.mean()), 4),
               "seconds_per_image_cpu": round(float(df.seconds.mean()), 2), "per_claim": {}}
    before_w = after_w = total = 0
    for k in KEYS:
        b = float(valid[f"vlm_wrong_{k}"].mean()) if len(valid) else float("nan")
        af = float(valid[f"final_wrong_{k}"].mean()) if len(valid) else float("nan")
        summary["per_claim"][k] = {"hallucination_before": round(b, 4), "error_after_verification": round(af, 4),
                                   "claims_changed_by_verifier": round(float(valid[f"caught_{k}"].mean()), 4) if len(valid) else None}
        before_w += valid[f"vlm_wrong_{k}"].sum(); after_w += valid[f"final_wrong_{k}"].sum(); total += len(valid)
    summary["hallucination_rate_before"] = round(float(before_w / total), 4) if total else None
    summary["error_rate_after_verification"] = round(float(after_w / total), 4) if total else None
    (out / "hallucination.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
