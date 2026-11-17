"""Package everything the static web app needs into web/public/.

    python scripts/build_web_assets.py

* models/    detector ONNX (base, and synth if trained) + YuNet
* samples/   curated gallery from the held-out TEST split (never trained on), with ground truth,
             thumbnails, one deliberate failure case, the pre-blurred hero image
* heatmaps/  EigenCAM overlays for the "Explain" toggle
* weather/   diffusion-generated night / rain / fog / dust versions for the Weather Lab
* clips/     sample clips (simulated camera moves over still test images)
* icons      PWA icons

All images come from Construction-PPE (CC BY 4.0, Ultralytics). The script is idempotent.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sitesafe.classes import CLASS_NAMES, FORKLIFT, NO_CLASSES, PERSON, SITE_PROFILES  # noqa: E402
from sitesafe.detector import Detector, letterbox  # noqa: E402
from sitesafe.matching import Det, iou, match_people  # noqa: E402
from sitesafe.paths import MODELS, ROOT, SPLITS, SYNTH, WEB_PUBLIC  # noqa: E402
from sitesafe.privacy import FaceDetector, blur_regions, privacy_boxes  # noqa: E402

REQUIRED = SITE_PROFILES["general"]
MAX_SIDE = 1280
N_SAMPLES = 9
WEATHER_LIST = SPLITS / "weatherlab.txt"


def label_of(img: Path) -> Path:
    return Path(str(img).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")).with_suffix(".txt")


def read_gt(img: Path, w: int, h: int) -> list[Det]:
    out = []
    p = label_of(img)
    if p.exists():
        for ln in p.read_text().splitlines():
            a = ln.split()
            if len(a) >= 5:
                c, cx, cy, bw, bh = int(a[0]), *map(float, a[1:5])
                out.append(Det([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], c, 1.0))
    return out


def load_resized(path: Path, max_side: int = MAX_SIDE) -> np.ndarray:
    bgr = cv2.imread(str(path))
    h, w = bgr.shape[:2]
    r = min(1.0, max_side / max(h, w))
    if r < 1:
        bgr = cv2.resize(bgr, (round(w * r), round(h * r)), interpolation=cv2.INTER_AREA)
    return bgr


def save_with_thumb(bgr: np.ndarray, dest: Path, quality: int = 84):
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    h, w = bgr.shape[:2]
    tw = 320
    thumb = cv2.resize(bgr, (tw, round(h * tw / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dest.with_name(dest.stem + "_thumb.jpg")), thumb, [cv2.IMWRITE_JPEG_QUALITY, 78])


def gt_json(gt: list[Det]) -> list[dict]:
    return [{"box": [round(v, 1) for v in d.box], "cls": d.cls, "conf": 1} for d in gt]


def compare_people(gt_people, pred_people):
    """Per-image person-level comparison: returns (n decidable, n wrong, n missed, n unsure)."""
    wrong = missed = unsure = n = 0
    for g in gt_people:
        if g.status == "unsure":
            continue
        n += 1
        best, bi = None, 0.5
        for p in pred_people:
            v = iou(p.box, g.box)
            if v >= bi:
                best, bi = p, v
        if best is None:
            missed += 1
        elif best.status == "unsure":
            unsure += 1
        elif best.status != g.status:
            wrong += 1
    return n, wrong, missed, unsure


def eigencam(pt_path: Path, bgr: np.ndarray) -> np.ndarray:
    """EigenCAM: first principal component of the neck feature maps that feed the detection head.

    Class-agnostic and gradient-free, so it shows what the detector as a whole looked at.
    Returns a BGR heat-map on black, same size as the image (for mix-blend-mode: screen).
    """
    import torch
    from ultralytics import YOLO
    net = YOLO(str(pt_path)).model.eval()
    feats = {}
    head = net.model[-1]
    hooks = [net.model[i].register_forward_hook(lambda m, i_, o, k=i: feats.__setitem__(k, o.detach())) for i in head.f[:2]]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    canvas, r, (dx, dy) = letterbox(rgb, 640)
    x = torch.from_numpy(canvas).permute(2, 0, 1)[None].float() / 255
    with torch.no_grad():
        net(x)
    for hk in hooks:
        hk.remove()
    cam = np.zeros((640, 640), np.float32)
    for a in feats.values():
        c, h, w = a.shape[1:]
        m = a[0].reshape(c, h * w).T.numpy()
        m = m - m.mean(0, keepdims=True)
        _, _, vt = np.linalg.svd(m, full_matrices=False)
        proj = (m @ vt[0]).reshape(h, w)
        if proj.sum() < 0 and abs(proj.min()) > abs(proj.max()):
            proj = -proj
        proj = np.maximum(proj, 0)
        proj = proj / (proj.max() + 1e-9)
        cam += cv2.resize(proj, (640, 640), interpolation=cv2.INTER_CUBIC)
    cam = cam / (cam.max() + 1e-9)
    H, W = bgr.shape[:2]
    cam = cam[dy:dy + round(H * r), dx:dx + round(W * r)]
    cam = cv2.resize(cam, (W, H), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return (heat.astype(np.float32) * (cam[..., None] ** 0.8)).astype(np.uint8)


def ken_burns(bgr: np.ndarray, seconds: float, fps: int, size=(960, 540), seed: int = 0):
    """Yield frames of a smooth pan + zoom over a still image. Also returns the crop per frame."""
    rng = np.random.default_rng(seed)
    H, W = bgr.shape[:2]
    ar = size[0] / size[1]
    def crop_at(zoom, cx, cy):
        cw = min(W, H * ar) / zoom
        ch = cw / ar
        x = float(np.clip(cx * W - cw / 2, 0, W - cw)); y = float(np.clip(cy * H - ch / 2, 0, H - ch))
        return x, y, cw, ch
    z0, z1 = 1.0, float(rng.uniform(1.25, 1.5))
    c0 = (float(rng.uniform(0.35, 0.65)), float(rng.uniform(0.4, 0.6)))
    c1 = (float(rng.uniform(0.3, 0.7)), float(rng.uniform(0.35, 0.65)))
    n = int(seconds * fps)
    for i in range(n):
        t = i / max(1, n - 1)
        e = 0.5 - 0.5 * np.cos(np.pi * t)
        x, y, cw, ch = crop_at(z0 + (z1 - z0) * e, c0[0] + (c1[0] - c0[0]) * e, c0[1] + (c1[1] - c0[1]) * e)
        M = np.array([[size[0] / cw, 0, -x * size[0] / cw], [0, size[1] / ch, -y * size[1] / ch]], np.float32)
        yield cv2.warpAffine(bgr, M, size, flags=cv2.INTER_LINEAR), (x, y, cw, ch)


def write_mp4(frames, dest: Path, fps: int, size=(960, 540)):
    import imageio_ffmpeg
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-profile:v", "baseline", "-crf", "26", "-movflags", "+faststart", str(dest)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    first = None
    for f, _ in frames:
        if first is None:
            first = f.copy()
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()
    return first


def make_icons():
    for s in (192, 512):
        img = np.zeros((s, s, 3), np.uint8)
        img[:] = (16, 13, 11)
        c = s // 2
        cv2.ellipse(img, (c, int(s * 0.62)), (int(s * 0.31), int(s * 0.31)), 0, 180, 360, (0, 212, 255), -1, cv2.LINE_AA)
        cv2.rectangle(img, (int(s * 0.13), int(s * 0.62)), (int(s * 0.87), int(s * 0.72)), (26, 122, 255), -1, cv2.LINE_AA)
        cv2.rectangle(img, (int(c - s * 0.045), int(s * 0.29)), (int(c + s * 0.045), int(s * 0.6)), (0, 150, 190), -1, cv2.LINE_AA)
        cv2.imwrite(str(WEB_PUBLIC / f"icon-{s}.png"), img)


def main():
    pub = WEB_PUBLIC
    for d in ("models", "samples", "heatmaps", "weather", "clips"):
        (pub / d).mkdir(parents=True, exist_ok=True)

    # ---- models ----
    shutil.copy(MODELS / "sitesafe_base.onnx", pub / "models" / "sitesafe_base.onnx")
    has_synth = (MODELS / "sitesafe_synth.onnx").exists()
    if has_synth:
        shutil.copy(MODELS / "sitesafe_synth.onnx", pub / "models" / "sitesafe_synth.onnx")
    shutil.copy(MODELS / "face_detection_yunet_2023mar.onnx", pub / "models" / "yunet.onnx")
    make_icons()

    det = Detector(str(MODELS / "sitesafe_base.onnx"))
    faces = FaceDetector(str(MODELS / "face_detection_yunet_2023mar.onnx"))
    test = [ROOT / ln for ln in (SPLITS / "test.txt").read_text().splitlines() if ln.strip()]

    # ---- score every test image so the gallery can be curated automatically ----
    rows = []
    for p in test:
        bgr = load_resized(p)
        h, w = bgr.shape[:2]
        gt = read_gt(p, w, h)
        gt_people = match_people(gt, REQUIRED)
        pred = det(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        n, wrong, missed, unsure = compare_people(gt_people, match_people(pred, REQUIRED))
        cls = {d.cls for d in gt}
        rows.append({"path": p, "n": n, "wrong": wrong, "missed": missed, "unsure": unsure, "cls": cls,
                     "bad": sum(g.status == "non_compliant" for g in gt_people),
                     "good": sum(g.status == "compliant" for g in gt_people), "aspect": w / h})

    used: set[Path] = set()

    def pick(cond, key, title, tags):
        cands = [r for r in rows if r["path"] not in used and cond(r)]
        if not cands:
            return None
        r = max(cands, key=key)
        used.add(r["path"])
        return {**r, "title": title, "tags": tags, "failure": False, "note": ""}

    clean = lambda r: r["wrong"] == 0 and r["missed"] == 0 and r["n"] >= 1  # noqa: E731
    wide = lambda r: 1.2 <= r["aspect"] <= 1.9  # noqa: E731
    picks = [
        pick(lambda r: clean(r) and wide(r) and r["bad"] >= 1 and r["good"] >= 1, lambda r: (min(r["n"], 5), -r["unsure"]), "Mixed crew", ["mixed"]),
        pick(lambda r: clean(r) and r["bad"] == 0 and r["good"] >= 2, lambda r: (r["good"], -r["unsure"]), "All compliant", ["compliant"]),
        pick(lambda r: clean(r) and r["bad"] >= 2, lambda r: (r["bad"], -r["unsure"]), "Several violations", ["violations"]),
        pick(lambda r: clean(r) and FORKLIFT in r["cls"] and r["n"] >= 1, lambda r: (r["n"], -r["unsure"]), "Forklift on site", ["forklift"]),
        pick(lambda r: clean(r) and 7 in r["cls"], lambda r: (r["n"], -r["unsure"]), "No goggles", ["goggles"]),
        pick(lambda r: clean(r) and 4 in r["cls"], lambda r: (r["n"], -r["unsure"]), "Goggles worn", ["goggles"]),
        pick(lambda r: clean(r) and r["n"] >= 3, lambda r: (r["n"], -r["unsure"]), "Busy scene", ["crowd"]),
        pick(lambda r: clean(r) and r["n"] == 1 and r["bad"] == 1, lambda r: -r["unsure"], "Single worker", ["single"]),
    ]
    picks = [p for p in picks if p][: N_SAMPLES - 1]
    worst = max((r for r in rows if r["path"] not in used and r["n"] >= 2), key=lambda r: (r["wrong"] + r["missed"], r["n"]))
    parts = []
    if worst["missed"]:
        parts.append(f"the model does not find {worst['missed']} of the {worst['n']} labelled workers")
    if worst["wrong"]:
        parts.append(f"it gives the wrong status to {worst['wrong']}")
    if worst["unsure"]:
        parts.append(f"it sends {worst['unsure']} to “check manually”")
    picks.append({**worst, "title": "Where it fails", "tags": ["failure"], "failure": True,
                  "note": ("On this photo " + ", and ".join(parts) + ". Crowded, overlapping or distant workers are the model's weak spot. "
                           "This is why the result is a prompt for the officer, not a verdict.")})
    used.add(worst["path"])

    # ---- write samples, heat-maps ----
    for old in (pub / "samples").glob("s*.jpg"):
        old.unlink()
    samples = []
    pt = MODELS / "sitesafe_base.pt"
    for i, r in enumerate(picks, 1):
        sid = f"s{i:02d}"
        bgr = load_resized(r["path"])
        h, w = bgr.shape[:2]
        save_with_thumb(bgr, pub / "samples" / f"{sid}.jpg")
        heat = None
        if pt.exists():
            hm = eigencam(pt, bgr)
            cv2.imwrite(str(pub / "heatmaps" / f"{sid}.jpg"), hm, [cv2.IMWRITE_JPEG_QUALITY, 70])
            heat = f"heatmaps/{sid}.jpg"
        samples.append({"id": sid, "file": f"samples/{sid}.jpg", "title": r["title"], "tags": r["tags"], "failure": r["failure"],
                        "note": r["note"], "heatmap": heat, "w": w, "h": h, "gt": gt_json(read_gt(r["path"], w, h)),
                        "source": r["path"].name})

    # ---- hero image: pre-blurred, with ground-truth statuses ----
    hero_src = picks[0]["path"]
    bgr = load_resized(hero_src, 1100)
    h, w = bgr.shape[:2]
    gt = read_gt(hero_src, w, h)
    people = match_people(gt, REQUIRED)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    blurred = blur_regions(rgb, privacy_boxes(faces(rgb), [p.box for p in people], strict=True))
    cv2.imwrite(str(pub / "samples" / "hero.jpg"), cv2.cvtColor(blurred, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
    (pub / "samples" / "hero.json").write_text(json.dumps({
        "file": "samples/hero.jpg", "w": w, "h": h,
        "people": [{"box": [round(v) for v in p.box], "status": p.status, "missing": p.missing} for p in people]}))

    # ---- weather lab ----
    if not WEATHER_LIST.exists():
        cands = [r for r in rows if r["path"] not in used and clean(r) and wide(r) and r["n"] >= 2]
        cands.sort(key=lambda r: (-min(r["n"], 4), r["unsure"]))
        WEATHER_LIST.write_text("\n".join(r["path"].relative_to(ROOT).as_posix() for r in cands[:5]) + "\n")
    weather = []
    wl_dir = SYNTH / "weatherlab" / "images" / "train"
    for i, ln in enumerate([x for x in WEATHER_LIST.read_text().splitlines() if x.strip()], 1):
        src = ROOT / ln
        variants = {c: wl_dir / f"{src.stem}_{c}.jpg" for c in ("night", "rain", "fog", "dust")}
        if not all(v.exists() for v in variants.values()):
            continue
        wid = f"w{i:02d}"
        ref = cv2.imread(str(variants["night"]))
        h, w = ref.shape[:2]
        orig = cv2.resize(cv2.imread(str(src)), (w, h), interpolation=cv2.INTER_AREA)
        save_with_thumb(orig, pub / "weather" / f"{wid}_original.jpg")
        out = {"original": f"weather/{wid}_original.jpg"}
        for c, v in variants.items():
            shutil.copy(v, pub / "weather" / f"{wid}_{c}.jpg")
            out[c] = f"weather/{wid}_{c}.jpg"
        weather.append({"id": wid, "title": src.stem[:24], "w": w, "h": h, "variants": out, "gt": gt_json(read_gt(src, w, h))})

    # ---- sample clips ----
    clips = []
    clip_src = [p for p in picks if not p["failure"] and p["aspect"] >= 1.2][:3]
    for i, r in enumerate(clip_src, 1):
        cid = f"c{i:02d}"
        bgr = cv2.imread(str(r["path"]))
        first = write_mp4(ken_burns(bgr, 12, 24, seed=i), pub / "clips" / f"{cid}.mp4", 24)
        cv2.imwrite(str(pub / "clips" / f"{cid}.jpg"), cv2.resize(first, (320, 180)), [cv2.IMWRITE_JPEG_QUALITY, 78])
        clips.append({"id": cid, "file": f"clips/{cid}.mp4", "title": f"{r['title']} (camera move)", "poster": f"clips/{cid}.jpg",
                      "note": "Simulated pan and zoom over a still image from the held-out test split."})

    (pub / "samples" / "catalogue.json").write_text(json.dumps({
        "licence": "Images: Construction-PPE dataset by Ultralytics, CC BY 4.0. Weather versions generated with SD-Turbo.",
        "samples": samples, "weather": weather, "clips": clips, "models": {"synth": has_synth}}))
    print(f"samples: {len(samples)} (failure case: {worst['path'].name}), weather: {len(weather)}, clips: {len(clips)}, synth model: {has_synth}")
    print(f"NO_CLASSES for reference: {[CLASS_NAMES[i] for i in NO_CLASSES]}, person id {PERSON}")


if __name__ == "__main__":
    main()
