"""Generative AI part 1b: quality filter for the synthetic images, then build the training yaml.

    python scripts/synth_filter.py --mode diffusion

A synthetic image is kept only if
1. the BASE detector (trained on real data only) still finds at least half of the labelled people
   at a low confidence of 0.10. If the people are gone, the image is too damaged for its labels;
2. its CLIP image embedding is still close to the original (cosine >= 0.60): the scene is the same;
3. CLIP agrees that the condition was actually applied (the condition prompt scores higher on the
   synthetic image than on the original). This drops images where diffusion changed nothing.

Known weakness, stated in the report: test 1 removes exactly the hardest images, the ones the base
model cannot yet handle, so the filter is deliberately lenient (0.10 confidence, half the people).

Writes data/splits/synth_<mode>.txt, ppe_synth_<mode>.yaml, results/synth/filter_<mode>.csv and a
contact sheet (kept and rejected examples) for the manual spot-check.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sitesafe.classes import PERSON  # noqa: E402
from sitesafe.detector import Detector  # noqa: E402
from sitesafe.evalutil import match_boxes, read_gt  # noqa: E402
from sitesafe.paths import MODELS, RESULTS, ROOT, SPLITS, SYNTH  # noqa: E402

CONDITION_TEXT = {"night": "a photo taken at night", "rain": "a photo taken in heavy rain",
                  "fog": "a photo taken in thick fog", "dust": "a photo taken in a dust storm"}


def clip_setup():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def embed_images(bgrs):
        with torch.no_grad():
            x = proc(images=[cv2.cvtColor(b, cv2.COLOR_BGR2RGB) for b in bgrs], return_tensors="pt")
            f = model.get_image_features(**x)
            f = f.pooler_output if hasattr(f, "pooler_output") else f
            return torch.nn.functional.normalize(f, dim=-1)

    def embed_text(texts):
        with torch.no_grad():
            x = proc(text=texts, return_tensors="pt", padding=True)
            f = model.get_text_features(**x)
            f = f.pooler_output if hasattr(f, "pooler_output") else f
            return torch.nn.functional.normalize(f, dim=-1)

    return embed_images, embed_text


def contact_sheet(rows, dest: Path, n: int = 12):
    tiles = []
    for r in rows[:n]:
        a = cv2.resize(cv2.imread(str(ROOT / r["source"])), (320, 220))
        b = cv2.resize(cv2.imread(r["path"]), (320, 220))
        cv2.putText(b, f"{r['condition']} sim={r['clip_sim']:.2f} people={r['person_recall']:.2f}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.hstack([a, b]))
    if tiles:
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
        cv2.imwrite(str(dest), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="diffusion")
    ap.add_argument("--min-person-recall", type=float, default=0.5)
    ap.add_argument("--min-clip-sim", type=float, default=0.60)
    ap.add_argument("--no-clip", action="store_true")
    a = ap.parse_args()

    root = SYNTH / a.mode
    manifest = list(csv.DictReader((root / "manifest.csv").open()))
    det = Detector(str(MODELS / "sitesafe_base.onnx"))
    embed_images = embed_text = None
    if not a.no_clip:
        embed_images, embed_text = clip_setup()
        text = {k: embed_text([v])[0] for k, v in CONDITION_TEXT.items()}

    rows = []
    for i, m in enumerate(manifest):
        path = root / "images" / "train" / m["file"]
        if not path.exists():
            continue
        syn = cv2.imread(str(path))
        h, w = syn.shape[:2]
        gt = [g for g in read_gt(path, w, h)]
        persons = [g for g in gt if g.cls == PERSON]
        dets = det(cv2.cvtColor(syn, cv2.COLOR_BGR2RGB), 0.10)
        found = match_boxes(persons, [d for d in dets if d.cls == PERSON])
        pr = len(found) / len(persons) if persons else 1.0
        sim, gain = 1.0, 1.0
        if embed_images:
            orig = cv2.resize(cv2.imread(str(ROOT / m["source"])), (w, h))
            e = embed_images([orig, syn])
            sim = float((e[0] * e[1]).sum())
            t = text[m["condition"]]
            gain = float((e[1] * t).sum() - (e[0] * t).sum())
        keep = pr >= a.min_person_recall and sim >= a.min_clip_sim and gain > 0
        rows.append({"file": m["file"], "path": str(path), "source": m["source"], "condition": m["condition"],
                     "persons": len(persons), "person_recall": round(pr, 3), "clip_sim": round(sim, 3),
                     "condition_gain": round(gain, 4), "keep": keep})
        if i % 25 == 0:
            print(f"[{i + 1}/{len(manifest)}] kept so far {sum(r['keep'] for r in rows)}", flush=True)

    out = RESULTS / "synth"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.drop(columns=["path"]).to_csv(out / f"filter_{a.mode}.csv", index=False)
    kept = [r for r in rows if r["keep"]]
    rejected = [r for r in rows if not r["keep"]]
    contact_sheet(kept, out / f"kept_{a.mode}.jpg")
    contact_sheet(rejected, out / f"rejected_{a.mode}.jpg")

    lst = SPLITS / f"synth_{a.mode}_abs.txt"
    lst.write_text("\n".join(Path(r["path"]).as_posix() for r in kept) + "\n")
    (SPLITS / f"synth_{a.mode}.txt").write_text("\n".join(Path(r["path"]).relative_to(ROOT).as_posix() for r in kept) + "\n")
    from make_split import write_yaml
    write_yaml(f"ppe_synth_{a.mode}.yaml", SPLITS / "train_abs.txt", [lst])
    summary = df.groupby("condition")["keep"].agg(["sum", "count"]).rename(columns={"sum": "kept", "count": "generated"})
    summary.to_csv(out / f"filter_summary_{a.mode}.csv")
    print(summary)
    print(f"kept {len(kept)} of {len(rows)} -> {lst}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
