"""Add a "real site photos" gallery to the web app: licensed Wikimedia Commons photos.

    python scripts/build_real_samples.py      (after build_web_assets.py, which writes catalogue.json)

The photos are listed in data/own_test/demo.txt. Most come from my own 30-photo test set, picked
where every worker the model decided on is right according to my labels (so the demo is honest
about what works); the gallery card still shows "not sure" people as they are. Each sample carries
its author and licence, shown under the photo in the app (CC BY / CC BY-SA require attribution).
These photos have person-level labels only, so they are not used by the Inspector Challenge.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_web_assets import eigencam, load_resized, save_with_thumb  # noqa: E402
from sitesafe.paths import DATA, MODELS, WEB_PUBLIC  # noqa: E402

OWN = DATA / "own_test"


def main():
    meta = {}
    for f in ("sources.csv", "candidates.csv"):
        p = OWN / f
        if p.exists():
            for r in csv.DictReader(p.open(encoding="utf-8")):
                meta.setdefault(r["file"], r)
    pub = WEB_PUBLIC
    cat_path = pub / "samples" / "catalogue.json"
    cat = json.loads(cat_path.read_text(encoding="utf-8"))
    for old in (pub / "samples").glob("r*.jpg"):
        old.unlink()
    for old in (pub / "heatmaps").glob("r*.jpg"):
        old.unlink()
    pt = MODELS / "sitesafe_base.pt"
    real = []
    for i, line in enumerate([x for x in (OWN / "demo.txt").read_text().splitlines() if x.strip() and not x.startswith("#")], 1):
        name, title = [s.strip() for s in line.split("|", 1)]
        src = OWN / "images" / name
        if not src.exists():
            src = OWN / "candidates" / name
        m = meta[name]
        sid = f"r{i:02d}"
        bgr = load_resized(src)
        h, w = bgr.shape[:2]
        save_with_thumb(bgr, pub / "samples" / f"{sid}.jpg")
        heat = None
        if pt.exists():
            cv2.imwrite(str(pub / "heatmaps" / f"{sid}.jpg"), eigencam(pt, bgr), [cv2.IMWRITE_JPEG_QUALITY, 70])
            heat = f"heatmaps/{sid}.jpg"
        author = m["author"].strip() or "unknown author"
        real.append({"id": sid, "file": f"samples/{sid}.jpg", "title": title, "tags": ["real"], "failure": False, "note": "",
                     "heatmap": heat, "w": w, "h": h, "gt": [],
                     "credit": {"title": m["title"].removeprefix("File:").rsplit(".", 1)[0], "author": author[:80],
                                "licence": m["licence"], "licence_url": m["licence_url"], "page": m["page"]}})
        print(sid, name, title, "|", m["licence"], "|", author[:40])
    cat["real"] = real
    cat["real_licence"] = "Real site photos: Wikimedia Commons, each under its own free licence (shown with the photo)."
    cat_path.write_text(json.dumps(cat), encoding="utf-8")
    print(f"real samples: {len(real)}")


if __name__ == "__main__":
    main()
