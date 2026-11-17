"""Generative AI part 1a: label-preserving weather / lighting synthesis with a diffusion model.

Each selected real training image is turned into ONE hard-condition version (night, rain, fog or
dust). Two stages:

1. A cheap classical "hint" (darkening, haze, streaks) pushes the image toward the condition.
2. SD-Turbo img2img at LOW strength re-renders it so the result looks like a photograph taken in
   that condition rather than a filter. Low strength keeps the geometry, so the original YOLO boxes
   stay valid and no hand-labelling is needed.

`--mode classical` stops after stage 1. That gives a "cheap augmentation" control arm, so the
experiment can answer: does the diffusion model add anything over a few lines of OpenCV?

Runs on a Colab T4 (about 0.3 s per image) or, slowly, on a laptop CPU (about 10 s per image).

    python scripts/synth_generate.py --n 280 --mode diffusion
    python scripts/synth_generate.py --n 280 --mode classical
    python scripts/synth_generate.py --images web_samples.txt --all-conditions --out data/synthetic/weatherlab
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sitesafe.paths import ROOT, SEED, SPLITS, SYNTH  # noqa: E402

CONDITIONS = ["night", "rain", "fog", "dust"]
PROMPTS = {
    "night": "construction site at night, dark sky, dim artificial floodlights, workers, realistic photo",
    "rain": "construction site in heavy rain, wet ground, overcast, rain streaks, workers, realistic photo",
    "fog": "construction site in dense fog, mist, low visibility, workers, realistic photo",
    "dust": "construction site in a dust storm, airborne dust haze, sandy air, workers, realistic photo",
}


def label_of(img: Path) -> Path:
    return Path(str(img).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")).with_suffix(".txt")


def classical_hint(bgr: np.ndarray, cond: str, rng: np.random.Generator) -> np.ndarray:
    img = bgr.astype(np.float32) / 255.0
    h, w = img.shape[:2]
    if cond == "night":
        img = np.power(img, 1.5) * 0.38
        img *= np.array([1.15, 0.95, 0.85], np.float32)  # BGR: push toward blue
        img += rng.normal(0, 0.02, img.shape).astype(np.float32)
    elif cond == "rain":
        grey = img.mean(2, keepdims=True)
        img = (0.55 * img + 0.45 * grey) * 0.72
        streaks = np.zeros((h, w), np.float32)
        for _ in range(int(h * w / 900)):
            x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            ln = int(rng.integers(h // 40, h // 14))
            cv2.line(streaks, (x, y), (x + ln // 4, y + ln), float(rng.uniform(0.25, 0.6)), 1)
        streaks = cv2.GaussianBlur(streaks, (3, 3), 0)
        img = img + streaks[..., None] * 0.35
    elif cond == "fog":
        grad = np.linspace(0.72, 0.45, h, dtype=np.float32)[:, None, None]  # thicker toward the horizon
        img = img * (1 - grad) + 0.88 * grad
    elif cond == "dust":
        sand = np.array([0.43, 0.63, 0.77], np.float32)  # BGR
        a = 0.48
        img = cv2.GaussianBlur(img, (0, 0), 1.2) * (1 - a) + sand * a
        img += rng.normal(0, 0.015, img.shape).astype(np.float32)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def fit_size(w: int, h: int, long_side: int) -> tuple[int, int]:
    r = long_side / max(w, h)
    return max(64, int(round(w * r / 8)) * 8), max(64, int(round(h * r / 8)) * 8)


def load_pipe(model: str, threads: int):
    import torch
    from diffusers import AutoPipelineForImage2Image
    cuda = torch.cuda.is_available()
    if not cuda and threads:
        torch.set_num_threads(threads)
    # The fp16 weight files are half the download; on CPU they are cast up to float32 after loading.
    pipe = AutoPipelineForImage2Image.from_pretrained(model, torch_dtype=torch.float16, variant="fp16",
                                                      safety_checker=None)
    pipe = pipe.to("cuda") if cuda else pipe.to("cpu", torch.float32)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=280, help="number of train images to transform")
    ap.add_argument("--mode", choices=["diffusion", "classical"], default="diffusion")
    ap.add_argument("--model", default="stabilityai/sd-turbo")
    ap.add_argument("--strength", type=float, default=0.3)
    ap.add_argument("--steps", type=int, default=4, help="scheduler steps; strength*steps UNet calls are run")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--images", default=None, help="optional txt list of images instead of sampling train")
    ap.add_argument("--all-conditions", action="store_true", help="make every condition for every image")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out) if args.out else SYNTH / args.mode
    (out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "train").mkdir(parents=True, exist_ok=True)

    rnd = random.Random(SEED)
    if args.images:
        imgs = [Path(ln) if Path(ln).is_absolute() else ROOT / ln
                for ln in Path(args.images).read_text().splitlines() if ln.strip()]
    else:
        train = [ROOT / ln for ln in (SPLITS / "train.txt").read_text().splitlines() if ln.strip()]
        imgs = rnd.sample(train, min(args.n, len(train)))
    jobs = []
    for i, p in enumerate(imgs):
        for cond in (CONDITIONS if args.all_conditions else [CONDITIONS[i % len(CONDITIONS)]]):
            jobs.append((p, cond))

    pipe = load_pipe(args.model, args.threads) if args.mode == "diffusion" else None
    if pipe is not None:
        import torch
        from PIL import Image
    rng = np.random.default_rng(SEED)
    manifest = out / "manifest.csv"
    done = set()
    if manifest.exists():
        done = {r["file"] for r in csv.DictReader(manifest.open())}
    mf = manifest.open("a", newline="")
    writer = csv.DictWriter(mf, fieldnames=["file", "source", "condition", "mode", "strength", "seconds"])
    if not done:
        writer.writeheader()

    t_all = time.time()
    for k, (src, cond) in enumerate(jobs):
        name = f"{src.stem}_{cond}.jpg"
        if name in done:  # resume after a disconnect
            continue
        t0 = time.time()
        bgr = cv2.imread(str(src))
        w, h = fit_size(bgr.shape[1], bgr.shape[0], args.size)
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        hint = classical_hint(bgr, cond, rng)
        if pipe is not None:
            init = Image.fromarray(cv2.cvtColor(hint, cv2.COLOR_BGR2RGB))
            gen = torch.Generator(device=pipe.device).manual_seed(SEED + k)
            res = pipe(prompt=PROMPTS[cond], image=init, strength=args.strength,
                       num_inference_steps=args.steps, guidance_scale=0.0, generator=gen).images[0]
            result = cv2.cvtColor(np.array(res), cv2.COLOR_RGB2BGR)
        else:
            result = hint
        cv2.imwrite(str(out / "images" / "train" / name), result, [cv2.IMWRITE_JPEG_QUALITY, 92])
        lab = label_of(src)
        if lab.exists():
            shutil.copy(lab, out / "labels" / "train" / (Path(name).stem + ".txt"))
        writer.writerow({"file": name, "source": src.relative_to(ROOT).as_posix() if src.is_relative_to(ROOT) else str(src),
                         "condition": cond, "mode": args.mode, "strength": args.strength,
                         "seconds": round(time.time() - t0, 2)})
        mf.flush()
        if k % 10 == 0:
            print(f"[{k + 1}/{len(jobs)}] {name}  {time.time() - t0:.1f}s  total {(time.time() - t_all) / 60:.1f} min", flush=True)
    mf.close()
    print(f"done: {len(jobs)} images in {out}")


if __name__ == "__main__":
    main()
