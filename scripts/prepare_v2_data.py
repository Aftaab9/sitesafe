"""Second attempt at extra real-photo data, built so that missing labels cannot teach "PPE is background".

    python scripts/prepare_v2_data.py                 # download, convert, pseudo-label, mask
    python scripts/prepare_v2_data.py --limit 40      # quick test on 40 images per source
    -> data/extra_v2/<name>/images|labels, data/extra_v2/stats.json, data/splits/ppe_v2.yaml

What went wrong the first time. The "plus" model was trained on three Roboflow datasets whose missing
classes were filled with confident pseudo-labels only. Every helmet, vest or boot the teacher was not
sure about stayed unlabelled, i.e. a training example of "this is background". The model learned to
doubt: on my own photos it sent 41 % of workers to "check manually" and called 2 of 12 missing helmets.

What this script does differently
  1. Two new sources that are labelled completely by people for the classes taken from them:
       Voxel51/hard-hat-detection (Hard Hat Workers, CC0, 5,000 real site photos): every helmet and
         every bare head -> helmet / no_helmet. Many small, distant workers.
       fathansanum/SH-17-Dataset (SH17, CC BY 4.0, 8,099 real photos of people at work): helmet, head,
         hands, gloves, safety vest, person. A head with no helmet on it -> no_helmet; a hand with no
         glove on it -> no_gloves.
     The three earlier sources are kept, processed with the same new rules.
  2. Classes a source does not label are filled from the teacher (YOLOv8s trained on 13,000 real
     construction photos) when it is confident (0.5 worn / person, 0.6 missing).
  3. Ignore regions. Wherever the teacher or my first model sees an unlabelled item with medium
     confidence (0.25 up to the pseudo-label threshold), the region is painted grey, the same grey
     YOLO uses for padding. The model is then never told that a probable helmet or boot is background.
     If such a region would cover much of a labelled box, the whole image is dropped instead.
  4. A pseudo-label never overrides a human label of the same item.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_extra_data import SOURCES as OLD_SOURCES, coco_items, source_key, yolo_items  # noqa: E402
from sitesafe.classes import CLASS_NAMES, ITEM_OF_CLASS  # noqa: E402
from sitesafe.matching import Det, area, intersection, iou  # noqa: E402
from sitesafe.paths import DATA, MODELS, SPLITS  # noqa: E402

RAW = DATA / "extra_raw"
OUT = DATA / "extra_v2"

SOURCES = dict(OLD_SOURCES)
SOURCES["hardhat"] = {"repo": "Voxel51/hard-hat-detection", "format": "fiftyone",
                      "map_names": {"helmet": 0, "head": 6, "person": 5},
                      "labelled": {0, 6}}  # person is labelled on only a few images: filled in like elsewhere
SOURCES["sh17"] = {"repo": "fathansanum/SH-17-Dataset", "format": "sh17", "labelled": {0, 1, 2, 5, 6, 8}}
# SH17 category ids in this Roboflow export (names are just "0".."16"; checked by drawing examples):
SH17 = {"person": 1, "helmet": 3, "hands": 4, "head": 5, "vest": 9, "gloves": 17}

POS_THR, NEG_THR = 0.5, 0.6
LOW = 0.25           # medium-confidence detections of unlabelled items become grey ignore regions
NO_IDS = {6, 7, 8, 9, 11}
GREY = 114
MINOR = {1, 4, 7, 8}  # gloves, goggles, no_goggle, no_gloves


def item(c: int):
    return ITEM_OF_CLASS.get(c, (None,))[0]


def download(names):
    from huggingface_hub import snapshot_download
    for name in names:
        snapshot_download(SOURCES[name]["repo"], repo_type="dataset", local_dir=str(RAW / name), max_workers=8)


def fiftyone_items(root: Path, limit: int | None):
    samples = json.loads((root / "samples.json").read_text())["samples"]
    for s in samples[:limit]:
        rows = []
        for d in s["ground_truth"]["detections"]:
            x, y, w, h = d["bounding_box"]  # relative top-left x, y, width, height
            rows.append((d["label"], x + w / 2, y + h / 2, w, h))
        yield root / s["filepath"], rows


def ioa(inner, outer) -> float:
    a = area(inner)
    return intersection(inner, outer) / a if a > 0 else 0.0


def sh17_items(root: Path, limit: int | None):
    """Yield (image, rows) with rows already in OUR class ids, derived classes included."""
    base = root / "SH17-dataset-1"
    n = 0
    for split in ("train", "valid", "test"):
        ann = base / split / "_annotations.coco.json"
        if not ann.exists():
            continue
        coco = json.loads(ann.read_text())
        by_img = {}
        for a in coco["annotations"]:
            by_img.setdefault(a["image_id"], []).append(a)
        for im in coco["images"]:
            if limit and n >= limit:
                return
            n += 1
            W, H = im["width"], im["height"]
            boxes = {}
            for a in by_img.get(im["id"], []):
                x, y, w, h = a["bbox"]
                boxes.setdefault(a["category_id"], []).append([x, y, x + w, y + h])
            helmets, gloves = boxes.get(SH17["helmet"], []), boxes.get(SH17["gloves"], [])
            out = [(5, b) for b in boxes.get(SH17["person"], [])]
            out += [(0, b) for b in helmets] + [(1, b) for b in gloves] + [(2, b) for b in boxes.get(SH17["vest"], [])]
            for hd in boxes.get(SH17["head"], []):
                # head boxes include the helmet when one is worn; look a little above the head too
                up = [hd[0], hd[1] - 0.5 * (hd[3] - hd[1]), hd[2], hd[3]]
                if not any(ioa(hm, up) > 0.3 for hm in helmets):
                    out.append((6, hd))
            for hn in boxes.get(SH17["hands"], []):
                if not any(ioa(g, hn) > 0.3 or ioa(hn, g) > 0.3 for g in gloves):
                    out.append((8, hn))
            rows = [(c, (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H, (b[2] - b[0]) / W, (b[3] - b[1]) / H) for c, b in out]
            yield base / split / im["file_name"], rows


class PtDetector:
    """Ultralytics .pt model that already uses our 12 classes (my first model); runs on the GPU in Colab."""
    def __init__(self, path: Path):
        from ultralytics import YOLO
        self.model = YOLO(str(path))

    def __call__(self, rgb, conf_thr=0.25, iou_thr=0.5):
        r = self.model.predict(rgb[..., ::-1], conf=conf_thr, iou=iou_thr, imgsz=640, verbose=False)[0]
        return [Det([float(v) for v in b], int(c), float(s))
                for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())]


def label_image(bgr, human, labelled, teacher, second, stats):
    """Returns (image, labels) or None if the image has to be dropped."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = bgr.shape[:2]
    t_dets, s_dets = teacher(rgb, LOW), second(rgb, LOW)

    def known(d):  # same item already labelled here (human or pseudo), or the same class
        return any(iou(d.box, g.box) > 0.3 and (g.cls == d.cls or (item(d.cls) and item(g.cls) == item(d.cls)))
                   for g in human + pseudo)

    pseudo = []
    for d in t_dets:
        if d.cls == 10 or (d.cls in labelled and d.cls != 5):
            continue
        thr = NEG_THR if d.cls in NO_IDS else POS_THR
        if d.conf >= thr and not known(d):
            pseudo.append(d)
            stats["pseudo"][CLASS_NAMES[d.cls]] += 1

    ignore = []
    for d in t_dets + s_dets:
        if d.cls in (5, 10) or d.cls in labelled:  # people are never painted out; forklifts are left alone
            continue
        if known(d):
            continue
        ignore.append((d.cls, d.box))
    labels = human + pseudo
    kept_ignore = []
    for c, b in ignore:
        if any(intersection(b, g.box) > 0.4 * area(g.box) and g.cls != 5 for g in labels):
            # Painting it out would hide a labelled object. Goggles and gloves sit on labelled heads and
            # hands all the time and matter least, so for them the region is simply left as it is;
            # for helmet, vest and boots the image is dropped.
            if c in MINOR:
                stats["ignore_skipped"] += 1
                continue
            stats["dropped_overlap"] += 1
            return None
        kept_ignore.append(b)
    ignore = kept_ignore
    if ignore:
        mask = np.zeros((h, w), bool)
        for b in ignore:
            x1, y1, x2, y2 = [int(round(v)) for v in b]
            mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = True
        if mask.mean() > 0.2:
            stats["dropped_area"] += 1
            return None
        bgr = bgr.copy()
        bgr[mask] = GREY
        stats["ignore_regions"] += len(ignore)
    return bgr, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="images per source (testing)")
    ap.add_argument("--second", default=str(MODELS / "sitesafe_base.pt"), help="my first model, second opinion for ignore regions")
    a = ap.parse_args()
    names = [n for n in SOURCES if not a.only or n in a.only]
    if not a.no_download:
        download(names)
    from sitesafe.external import ExternalDetector
    teacher, second = ExternalDetector(), PtDetector(Path(a.second))

    stats_all, lists = {}, []
    for name in names:
        s = SOURCES[name]
        root = RAW / name
        out_img, out_lab = OUT / name / "images", OUT / name / "labels"
        shutil.rmtree(OUT / name, ignore_errors=True)
        out_img.mkdir(parents=True); out_lab.mkdir(parents=True)
        st = {"seen": 0, "dropped_overlap": 0, "dropped_area": 0, "ignore_regions": 0, "ignore_skipped": 0,
              "human": Counter(), "pseudo": Counter()}
        kept, seen = [], set()
        if s["format"] == "fiftyone":
            gen = fiftyone_items(root, a.limit)
        elif s["format"] == "sh17":
            gen = sh17_items(root, a.limit)
        else:
            def gen_old():
                n = 0
                for split in ("train", "valid", "test"):
                    it = yolo_items(root, split) if s["format"] == "yolo" else coco_items(root, split, OUT / f"_{name}_unzip")
                    for img, rows in it:
                        key = source_key(img)
                        if key in seen:
                            continue
                        seen.add(key)
                        if a.limit and n >= a.limit:
                            return
                        n += 1
                        yield img, rows
            gen = gen_old()
        for img, rows in gen:
            st["seen"] += 1
            bgr = cv2.imread(str(img))
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            human = []
            for r in rows:
                c = r[0] if isinstance(r[0], int) and s["format"] == "sh17" else \
                    (s["map"].get(r[0]) if "map" in s else s["map_names"].get(r[0]))
                if c is None:
                    continue
                cx, cy, bw, bh = r[1:]
                if bw <= 0 or bh <= 0:
                    continue
                human.append(Det([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], c, 1.0))
            if not human:
                continue
            res = label_image(bgr, human, s["labelled"], teacher, second, st)
            if res is None:
                continue
            bgr2, labels = res
            for d in human:
                st["human"][CLASS_NAMES[d.cls]] += 1
            stem = f"{name}_{len(kept):05d}"
            cv2.imwrite(str(out_img / f"{stem}.jpg"), bgr2, [cv2.IMWRITE_JPEG_QUALITY, 92])
            lines = []
            for d in labels:
                x1, y1, x2, y2 = [min(max(v, 0.0), lim) for v, lim in zip(d.box, (w, h, w, h))]
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                lines.append(f"{d.cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            (out_lab / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            kept.append((out_img / f"{stem}.jpg").as_posix())
        shutil.rmtree(OUT / f"_{name}_unzip", ignore_errors=True)
        lst = SPLITS / f"v2_{name}_abs.txt"
        lst.write_text("\n".join(kept) + "\n")
        lists.append(lst)
        st["kept"] = len(kept)
        st["human"], st["pseudo"] = dict(st["human"]), dict(st["pseudo"])
        stats_all[name] = st
        print(name, json.dumps(st), flush=True)

    (OUT / "stats.json").write_text(json.dumps(stats_all, indent=2))
    from make_split import write_yaml
    # The original 700 photos are listed twice so they keep their weight next to ~15,000 extra images.
    write_yaml("ppe_v2.yaml", SPLITS / "train_abs.txt", [SPLITS / "train_abs.txt"] + lists)
    print("wrote data/splits/ppe_v2.yaml")


if __name__ == "__main__":
    main()
