"""Step 1: download and unzip Construction-PPE (CC BY 4.0), then fix the path in data.yaml.

    python scripts/download_data.py

Also downloads the YuNet face detector (MIT) into models/.
"""
from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sitesafe.paths import DATA, DATASET, MODELS, RAW  # noqa: E402

DATA_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")


def fetch(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"already present: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")

    def hook(n, bs, total):
        if total > 0 and n % 200 == 0:
            print(f"\r  {min(100, n * bs * 100 // total)}%", end="")

    urllib.request.urlretrieve(url, dest, hook)
    print(f"\r  done ({dest.stat().st_size / 1e6:.1f} MB)")


def main():
    zip_path = RAW / "construction-ppe.zip"
    fetch(DATA_URL, zip_path)
    if not (DATASET / "images" / "train").exists():
        print("unzipping ...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DATA)
    # Fix the `path:` line: Ultralytics needs an absolute path or it looks in its own datasets dir.
    yaml_path = DATASET / "data.yaml"
    lines = yaml_path.read_text(encoding="utf-8").splitlines()
    lines = [f"path: {DATASET.as_posix()}" if ln.startswith("path:") else ln for ln in lines]
    if not any(ln.startswith("path:") for ln in lines):
        lines.insert(0, f"path: {DATASET.as_posix()}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    n_train = len(list((DATASET / "images" / "train").glob("*")))
    n_val = len(list((DATASET / "images" / "val").glob("*")))
    print(f"dataset ready: {n_train} train, {n_val} val images at {DATASET}")
    fetch(YUNET_URL, MODELS / "face_detection_yunet_2023mar.onnx")


if __name__ == "__main__":
    main()
