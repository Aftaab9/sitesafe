"""Merge extra open PPE datasets (all CC BY 4.0, real photographs) into our 12-class format.

    python scripts/prepare_extra_data.py            # download (if needed), remap, de-duplicate, pseudo-label
    -> data/extra/<name>/images|labels, data/splits/extra_train_abs.txt, data/splits/ppe_plus.yaml

Why: about 570 of the 800 Construction-PPE training images look AI-generated, "no_boots" there
almost always means canvas trainers, and hi-vis jackets are rare. On real photos the model then
misses hi-vis jackets, calls work boots "no_boots" and sees vests that are not there.

Sources
  HB1204/PPE_Detection                         Boots, Gloves, Goggles, Hardhat, NO-Gloves, NO-Goggles,
                                               NO-Hardhat, NO-Safety Vest, Person, Safety Vest
  VincentGOURBIN/ppe-detection                 helmet, no-helmet, vest, no-vest, person
  keremberke/construction-safety-object-detection  hardhat, no-hardhat, safety vest, no-safety vest,
                                               person, gloves, safety shoes (+ unrelated classes, dropped)

Roboflow exports contain up to three augmented copies of each photo (`name_jpg.rf.<hash>.jpg`);
only one copy per source photo is kept, so no photo is counted three times.

Missing classes. Each dataset labels only some of our 12 classes. The teacher is an off-the-shelf
YOLOv8s trained on about 13,000 real construction photos (killuminati1/construction-ppe-yolov8,
Apache-2.0); on my 30 real test photos it judged PPE better than my first model. An unlabelled helmet in a training
image teaches the model that helmets are background, so the classes a dataset does not label are
filled in with the current model's detections (pseudo-labels) at a high confidence: 0.5 for worn
items and person, 0.75 for "no_" classes, 0.85 for forklift. A pseudo-label is dropped when it overlaps a
human label of the same item (worn or missing), so the teacher can never overrule a human.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sitesafe.classes import CLASS_NAMES, ITEM_OF_CLASS  # noqa: E402
from sitesafe.matching import Det, iou  # noqa: E402
from sitesafe.paths import DATA, MODELS, SPLITS  # noqa: E402

RAW = DATA / "extra_raw"
OUT = DATA / "extra"

SOURCES = {
    "hb1204": {"repo": "HB1204/PPE_Detection", "format": "yolo",
               "map": {0: 3, 1: 1, 2: 4, 3: 0, 4: 8, 5: 7, 6: 6, 7: 11, 9: 5, 10: 2},
               "labelled": {0, 1, 2, 3, 4, 5, 6, 7, 8, 11}},
    "vincent": {"repo": "VincentGOURBIN/ppe-detection", "format": "yolo",
                "map": {0: 0, 1: 6, 2: 11, 3: 5, 4: 2},
                "labelled": {0, 2, 5, 6, 11}},
    "kerem_cs": {"repo": "keremberke/construction-safety-object-detection", "format": "coco_zip",
                 "map_names": {"hardhat": 0, "no-hardhat": 6, "safety vest": 2, "no-safety vest": 11, "person": 5,
                               "gloves": 1, "safety shoes": 3},
                 "labelled": {0, 1, 2, 3, 5, 6, 11}},
}
POS_THR, NEG_THR, FORKLIFT_THR = 0.5, 0.75, 0.85  # forklift: the teacher mistook gym machines for forklifts at 0.5
NO_IDS = {6, 7, 8, 9, 11}


def download():
    from huggingface_hub import snapshot_download
    for name, s in SOURCES.items():
        snapshot_download(s["repo"], repo_type="dataset", local_dir=str(RAW / name), max_workers=8)


def source_key(p: Path) -> str:
    """Roboflow names augmented copies `<source>_<ext>.rf.<hash>.jpg`; the part before .rf. is the photo."""
    return re.split(r"\.rf\.", p.name)[0]


def yolo_items(root: Path, split: str):
    for img in sorted((root / split / "images").glob("*")):
        lab = root / split / "labels" / (img.stem + ".txt")
        rows = []
        if lab.exists():
            for ln in lab.read_text().splitlines():
                a = ln.split()
                if len(a) == 5:
                    rows.append((int(a[0]), *map(float, a[1:])))
                elif len(a) > 5:  # polygon: convert to its bounding box
                    xs, ys = list(map(float, a[1::2])), list(map(float, a[2::2]))
                    rows.append((int(a[0]), (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)))
        yield img, rows


def coco_items(root: Path, split: str, extract_to: Path):
    z = root / "data" / f"{split}.zip"
    if not z.exists():
        return
    d = extract_to / split
    if not d.exists():
        with zipfile.ZipFile(z) as f:
            f.extractall(d)
    ann = next(d.rglob("_annotations.coco.json"))
    coco = json.loads(ann.read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    for im in coco["images"]:
        p = ann.parent / im["file_name"]
        W, H = im["width"], im["height"]
        rows = []
        for a in by_img.get(im["id"], []):
            x, y, w, h = a["bbox"]
            rows.append((cats[a["category_id"]], (x + w / 2) / W, (y + h / 2) / H, w / W, h / H))
        yield p, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--teacher", default="external",
                    help="'external' = killuminati1/construction-ppe-yolov8 (Apache-2.0, real photos), or a path to our ONNX")
    ap.add_argument("--only", nargs="*", default=None, help="process only these sources (for testing)")
    a = ap.parse_args()
    if not a.no_download:
        download()
    if a.teacher == "external":
        from sitesafe.external import ExternalDetector
        teacher = ExternalDetector()  # has no forklift class, so no forklift pseudo-labels are added
    else:
        from sitesafe.detector import Detector
        teacher = Detector(a.teacher)

    stats, train_lists = {}, []
    for name, s in SOURCES.items():
        if a.only and name not in a.only:
            continue
        root = RAW / name
        out_img, out_lab = OUT / name / "images", OUT / name / "labels"
        if (OUT / name).exists():
            shutil.rmtree(OUT / name)
        out_img.mkdir(parents=True); out_lab.mkdir(parents=True)
        seen, counts, pseudo = set(), Counter(), Counter()
        kept = []
        for split in ("train", "valid", "test"):
            items = yolo_items(root, split) if s["format"] == "yolo" else coco_items(root, split, OUT / f"_{name}_unzip")
            for img, rows in items:
                key = source_key(img)
                if key in seen:
                    continue
                seen.add(key)
                bgr = cv2.imread(str(img))
                if bgr is None:
                    continue
                h, w = bgr.shape[:2]
                human = []
                for r in rows:
                    c = s["map"].get(r[0]) if "map" in s else s["map_names"].get(r[0])
                    if c is None:
                        continue
                    cx, cy, bw, bh = r[1:]
                    human.append(Det([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], c, 1.0))
                if not human:
                    continue
                extra = []
                for d in teacher(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), POS_THR):
                    # Person is filled in everywhere: some sets label only a few of the people in a photo.
                    # A teacher person box overlapping a human person box is dropped below as a duplicate.
                    if d.cls in s["labelled"] and d.cls != 5:
                        continue
                    thr = FORKLIFT_THR if d.cls == 10 else NEG_THR if d.cls in NO_IDS else POS_THR
                    if d.conf < thr:
                        continue
                    item = ITEM_OF_CLASS.get(d.cls, (None,))[0]
                    clash = any(iou(d.box, g.box) > 0.3 and (g.cls == d.cls or (item and ITEM_OF_CLASS.get(g.cls, (None,))[0] == item))
                                for g in human)
                    if not clash:
                        extra.append(d)
                        pseudo[CLASS_NAMES[d.cls]] += 1
                stem = f"{name}_{len(kept):05d}"
                cv2.imwrite(str(out_img / f"{stem}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                lines = []
                for d in human + extra:
                    x1, y1, x2, y2 = d.box
                    lines.append(f"{d.cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
                    counts[CLASS_NAMES[d.cls]] += d in human
                (out_lab / f"{stem}.txt").write_text("\n".join(lines) + "\n")
                kept.append((out_img / f"{stem}.jpg").as_posix())
        shutil.rmtree(OUT / f"_{name}_unzip", ignore_errors=True)
        lst = SPLITS / f"extra_{name}_abs.txt"
        lst.write_text("\n".join(kept) + "\n")
        train_lists.append(lst)
        stats[name] = {"images": len(kept), "human_boxes": dict(counts), "pseudo_boxes": dict(pseudo)}
        print(name, json.dumps(stats[name]))

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_split import write_yaml
    if a.only:
        return
    write_yaml("ppe_plus.yaml", SPLITS / "train_abs.txt", train_lists)
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    print("wrote data/splits/ppe_plus.yaml")


if __name__ == "__main__":
    main()
