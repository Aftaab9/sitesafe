"""My own 30-image test set: pre-label with the trained model, correct by hand, re-import.

Workflow
  1. Put the images in data/own_test/images/ and fill in data/own_test/sources.csv
     (file, source URL, licence, photographer) - only public, permitted sources.
  2. python scripts/prelabel.py
        writes YOLO labels to data/own_test/prelabels/ plus two import bundles:
        data/own_test/cvat_yolo.zip          -> CVAT: Create task, then Upload annotations, format "YOLO 1.1"
        data/own_test/label_studio.json      -> Label Studio: Import (predictions shown as pre-annotations)
        and data/own_test/tags.csv to fill in (lighting, distance, clothing, skin tone group).
  3. Correct every box by hand in CVAT or Label Studio. Export as "YOLO 1.1" (CVAT) or "YOLO" (Label Studio).
  4. python scripts/prelabel.py --import-zip path/to/export.zip
        copies the corrected labels to data/own_test/labels/ and checks them.
  5. python scripts/compliance_eval.py --split own   and   python scripts/evaluate.py --split own ...

The pre-labels are the model's own guesses: they must be corrected, never trusted. To limit
anchoring on the model's mistakes, low-confidence boxes are included (conf >= 0.15) so missed
objects are easier to notice, and the checklist in docs/own_test_set.md asks for a second pass with
pre-labels hidden.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sitesafe.classes import CLASS_NAMES  # noqa: E402
from sitesafe.detector import Detector  # noqa: E402
from sitesafe.paths import MODELS, OWN, ROOT, SPLITS  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def images():
    return sorted(p for p in (OWN / "images").glob("*") if p.suffix.lower() in EXTS)


def prelabel(conf: float):
    det = Detector(str(MODELS / "sitesafe_base.onnx"))
    pre = OWN / "prelabels"
    pre.mkdir(parents=True, exist_ok=True)
    ls_tasks, tag_rows = [], []
    imgs = images()
    if not imgs:
        print(f"No images in {OWN / 'images'}. Add your 30 images first.")
        return
    for p in imgs:
        rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        dets = det(rgb, conf)
        lines, results = [], []
        for i, d in enumerate(dets):
            x1, y1, x2, y2 = d.box
            lines.append(f"{d.cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            results.append({"id": f"{p.stem}_{i}", "type": "rectanglelabels", "from_name": "label", "to_name": "image",
                            "original_width": w, "original_height": h, "score": round(d.conf, 3),
                            "value": {"x": 100 * x1 / w, "y": 100 * y1 / h, "width": 100 * (x2 - x1) / w,
                                      "height": 100 * (y2 - y1) / h, "rotation": 0, "rectanglelabels": [CLASS_NAMES[d.cls]]}})
        (pre / f"{p.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        ls_tasks.append({"data": {"image": f"/data/local-files/?d=own_test/images/{p.name}"},
                         "predictions": [{"model_version": "sitesafe_base", "result": results}]})
        tag_rows.append({"image": p.name, "lighting": "", "distance": "", "clothing": "", "skin_tone": "", "notes": ""})

    with zipfile.ZipFile(OWN / "cvat_yolo.zip", "w") as z:  # CVAT "YOLO 1.1" layout
        z.writestr("obj.names", "\n".join(CLASS_NAMES) + "\n")
        z.writestr("obj.data", f"classes = {len(CLASS_NAMES)}\ntrain = data/train.txt\nnames = data/obj.names\nbackup = backup/\n")
        z.writestr("train.txt", "\n".join(f"data/obj_train_data/{p.name}" for p in imgs) + "\n")
        for p in imgs:
            z.write(pre / f"{p.stem}.txt", f"obj_train_data/{p.stem}.txt")
    (OWN / "label_studio.json").write_text(json.dumps(ls_tasks, indent=1))
    (OWN / "label_studio_config.xml").write_text(
        '<View>\n  <Image name="image" value="$image"/>\n  <RectangleLabels name="label" toName="image">\n'
        + "".join(f'    <Label value="{n}"/>\n' for n in CLASS_NAMES) + "  </RectangleLabels>\n</View>\n")
    if not (OWN / "tags.csv").exists():
        pd.DataFrame(tag_rows).to_csv(OWN / "tags.csv", index=False)
    if not (OWN / "sources.csv").exists():
        pd.DataFrame([{"file": p.name, "source_url": "", "licence": "", "author": ""} for p in imgs]).to_csv(OWN / "sources.csv", index=False)
    print(f"pre-labelled {len(imgs)} images -> {pre}\nCVAT bundle: {OWN / 'cvat_yolo.zip'}\nLabel Studio tasks: {OWN / 'label_studio.json'}")


def import_zip(zpath: Path):
    labels = OWN / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    stems = {p.stem for p in images()}
    n = 0
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            q = Path(name)
            if q.suffix == ".txt" and q.stem in stems:
                (labels / q.name).write_bytes(z.read(name)); n += 1
    bad = []
    for f in labels.glob("*.txt"):
        for ln in f.read_text().splitlines():
            a = ln.split()
            if len(a) != 5 or not (0 <= int(a[0]) < len(CLASS_NAMES)) or any(not (0 <= float(v) <= 1) for v in a[1:]):
                bad.append(f.name); break
    lst = [p for p in images() if (labels / f"{p.stem}.txt").exists()]
    (SPLITS / "own.txt").write_text("\n".join(p.relative_to(ROOT).as_posix() for p in lst) + "\n")
    print(f"imported {n} label files; {len(lst)} of {len(stems)} images are labelled; invalid files: {bad or 'none'}")
    print("data/splits/own.txt written: use --split own in the evaluation scripts.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--import-zip", default=None)
    a = ap.parse_args()
    (OWN / "images").mkdir(parents=True, exist_ok=True)
    if a.import_zip:
        import_zip(Path(a.import_zip))
    else:
        prelabel(a.conf)
