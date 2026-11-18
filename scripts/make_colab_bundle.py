"""Build colab/sitesafe_code.zip: everything Colab needs, nothing large.

    python scripts/make_colab_bundle.py          # --plus / --v2 for the extra-data runs

Contains the sitesafe/ package, scripts/, requirements, the committed split lists (so Colab trains
on exactly the same 700 / 200 / 100 split) and the YuNet face model. Data is downloaded in Colab.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    plus = "--plus" in sys.argv
    v2 = "--v2" in sys.argv
    out = ROOT / "colab" / ("sitesafe_v2_code.zip" if v2 else "sitesafe_plus_code.zip" if plus else "sitesafe_code.zip")
    out.parent.mkdir(exist_ok=True)
    files = []
    for pattern in ("sitesafe/*.py", "scripts/*.py", "tests/*.py", "tests/fixtures/*.json"):
        files += sorted(ROOT.glob(pattern))
    files += [ROOT / "requirements.txt", ROOT / "requirements-train.txt", ROOT / "pyproject.toml",
              ROOT / "models" / "face_detection_yunet_2023mar.onnx"]
    for name in ("train.txt", "val.txt", "test.txt", "weatherlab.txt"):
        files.append(ROOT / "data" / "splits" / name)
    if v2:  # v2 fine-tunes the app model (needs its .pt) and scores it with the app's calibration
        files += [ROOT / "models" / "sitesafe_base.pt", ROOT / "results" / "eval" / "calibration.json"]
    if plus or v2:  # the improved-model run also scores my own 30 real photos and needs the first model
        files += [ROOT / "models" / "sitesafe_base.onnx", ROOT / "data" / "own_test" / "people_labels.json",
                  ROOT / "data" / "own_test" / "sources.csv"]
        files += sorted((ROOT / "data" / "own_test" / "images").glob("*.jpg"))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if f.exists():
                z.write(f, f.relative_to(ROOT).as_posix())
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {len(files)} files)")


if __name__ == "__main__":
    main()
