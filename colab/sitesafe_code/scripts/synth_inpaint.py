"""Generative AI ablation: remove goggles by inpainting and relabel goggles -> no_goggle.

    python scripts/synth_inpaint.py --n 200        (needs a GPU: run in Colab notebook 2)

no_goggle is the rarest safety class. For each real training image with a `goggles` box, the box
(slightly enlarged) is inpainted with Stable Diffusion inpainting using the prompt "bare eyes, no
glasses", inside a square crop around the head so the face has enough pixels. The label is changed
from goggles (4) to no_goggle (7); every other label is untouched.

Quality filter: the result is kept only if the base detector no longer finds goggles in that box
(confidence >= 0.25, IoU >= 0.3). Not run locally (no GPU); results come from Colab.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sitesafe.detector import Detector  # noqa: E402
from sitesafe.evalutil import label_of  # noqa: E402
from sitesafe.matching import iou  # noqa: E402
from sitesafe.paths import MODELS, ROOT, SEED, SPLITS, SYNTH  # noqa: E402

GOGGLES, NO_GOGGLE = 4, 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-inpainting")
    ap.add_argument("--steps", type=int, default=25)
    a = ap.parse_args()

    import torch
    from diffusers import AutoPipelineForInpainting
    from PIL import Image
    assert torch.cuda.is_available(), "inpainting needs a GPU (use the Colab notebook)"
    pipe = AutoPipelineForInpainting.from_pretrained(a.model, torch_dtype=torch.float16, safety_checker=None).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    det = Detector(str(MODELS / "sitesafe_base.onnx"))

    out = SYNTH / "inpaint"
    (out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "train").mkdir(parents=True, exist_ok=True)
    rnd = random.Random(SEED)
    train = [ROOT / ln for ln in (SPLITS / "train.txt").read_text().splitlines() if ln.strip()]
    rnd.shuffle(train)
    kept = []
    for p in train:
        if len(kept) >= a.n:
            break
        lab = label_of(p)
        rows = [ln.split() for ln in lab.read_text().splitlines() if ln.strip()] if lab.exists() else []
        if not any(r[0] == str(GOGGLES) for r in rows):
            continue
        bgr = cv2.imread(str(p))
        H, W = bgr.shape[:2]
        changed = []
        for r in rows:
            if r[0] != str(GOGGLES):
                continue
            cx, cy, bw, bh = float(r[1]) * W, float(r[2]) * H, float(r[3]) * W, float(r[4]) * H
            side = int(max(96, 4 * max(bw, bh)))
            x0, y0 = int(np.clip(cx - side / 2, 0, max(0, W - side))), int(np.clip(cy - side / 2, 0, max(0, H - side)))
            crop = bgr[y0:y0 + side, x0:x0 + side]
            ch, cw = crop.shape[:2]
            mask = np.zeros((ch, cw), np.uint8)
            mx1, my1 = int(cx - x0 - 0.65 * bw), int(cy - y0 - 0.75 * bh)
            mx2, my2 = int(cx - x0 + 0.65 * bw), int(cy - y0 + 0.75 * bh)
            cv2.rectangle(mask, (max(0, mx1), max(0, my1)), (min(cw, mx2), min(ch, my2)), 255, -1)
            res = pipe(prompt="close-up photo of a construction worker's face, bare eyes, no glasses, no goggles, realistic skin",
                       negative_prompt="glasses, goggles, sunglasses, visor, deformed",
                       image=Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize((512, 512)),
                       mask_image=Image.fromarray(mask).resize((512, 512)), num_inference_steps=a.steps, guidance_scale=7.0,
                       generator=torch.Generator("cuda").manual_seed(SEED + len(kept))).images[0]
            new = cv2.resize(cv2.cvtColor(np.array(res), cv2.COLOR_RGB2BGR), (cw, ch))
            soft = cv2.GaussianBlur(mask, (0, 0), max(1.0, 0.08 * side))[..., None] / 255.0
            bgr[y0:y0 + ch, x0:x0 + cw] = (new * soft + crop * (1 - soft)).astype(np.uint8)
            changed.append((r, [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]))
        dets = [d for d in det(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), 0.25) if d.cls == GOGGLES]
        ok = all(not any(iou(d.box, box) >= 0.3 for d in dets) for _, box in changed)
        if not ok:
            continue
        for r, _ in changed:
            r[0] = str(NO_GOGGLE)
        name = f"{p.stem}_nogoggle"
        cv2.imwrite(str(out / "images" / "train" / f"{name}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / "labels" / "train" / f"{name}.txt").write_text("\n".join(" ".join(r) for r in rows) + "\n")
        kept.append(out / "images" / "train" / f"{name}.jpg")
        if len(kept) % 20 == 0:
            print(f"kept {len(kept)}", flush=True)

    lst = SPLITS / "synth_inpaint_abs.txt"
    lst.write_text("\n".join(k.as_posix() for k in kept) + "\n")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_split import write_yaml
    write_yaml("ppe_synth_inpaint.yaml", SPLITS / "train_abs.txt", [lst])
    print(f"kept {len(kept)} inpainted images -> {lst}")


if __name__ == "__main__":
    main()
