"""Face-blur impact study: does blurring faces BEFORE detection hurt the detector?

    python scripts/blur_study.py --weights models/sitesafe_base.pt

Three versions of the test split are evaluated with the same weights:
  original     - untouched images
  faces        - YuNet face boxes blurred (what a "blur first, detect second" pipeline would see)
  strict       - faces plus the head region of every labelled person blurred (worst case)

The result decides the pipeline order. If blurring first costs helmet / goggles recall, the app must
detect on the original frame in memory and blur only what is displayed. Writes results/eval/blur_study.json.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sitesafe.classes import CLASS_NAMES, PERSON  # noqa: E402
from sitesafe.evalutil import label_of, load_split, read_gt  # noqa: E402
from sitesafe.paths import DATA, MODELS, RESULTS  # noqa: E402
from sitesafe.privacy import FaceDetector, blur_regions, head_region  # noqa: E402


def build_variant(name: str, images, faces: FaceDetector) -> Path:
    root = DATA / "blur_study" / name
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    n_faces = 0
    for p in images:
        bgr = cv2.imread(str(p))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if name != "original":
            boxes = faces(rgb)
            n_faces += len(boxes)
            if name == "strict":
                h, w = rgb.shape[:2]
                boxes += [head_region(g.box) for g in read_gt(p, w, h) if g.cls == PERSON]
            rgb = blur_regions(rgb, boxes)
        cv2.imwrite(str(root / "images" / p.name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        shutil.copy(label_of(p), root / "labels" / (p.stem + ".txt"))
    (root / "data.yaml").write_text(
        f"path: {root.as_posix()}\ntrain: images\nval: images\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES)))
    return root, n_faces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(MODELS / "sitesafe_base.pt"))
    a = ap.parse_args()
    from ultralytics import YOLO
    images = load_split("test")
    faces = FaceDetector(str(MODELS / "face_detection_yunet_2023mar.onnx"))
    out = {}
    for name in ("original", "faces", "strict"):
        root, n_faces = build_variant(name, images, faces)
        m = YOLO(a.weights).val(data=str(root / "data.yaml"), split="val", imgsz=640, plots=False, verbose=False, workers=0, batch=8)
        per = {CLASS_NAMES[int(c)]: {"recall": round(float(m.box.class_result(i)[1]), 4), "ap50": round(float(m.box.class_result(i)[2]), 4)}
               for i, c in enumerate(m.box.ap_class_index)}
        out[name] = {"map50": round(float(m.box.map50), 4), "map50_95": round(float(m.box.map), 4), "faces_blurred": n_faces,
                     "helmet_recall": per.get("helmet", {}).get("recall"), "no_helmet_recall": per.get("no_helmet", {}).get("recall"),
                     "goggles_recall": per.get("goggles", {}).get("recall"), "no_goggle_recall": per.get("no_goggle", {}).get("recall"),
                     "person_recall": per.get("Person", {}).get("recall"), "per_class": per}
        print(name, {k: v for k, v in out[name].items() if k != "per_class"})
    shutil.rmtree(DATA / "blur_study", ignore_errors=True)  # blurred copies are not kept
    (RESULTS / "eval").mkdir(parents=True, exist_ok=True)
    (RESULTS / "eval" / "blur_study.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
