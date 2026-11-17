"""Step 3: dataset statistics and sample visualisations for the report.

Writes results/data/class_counts.csv, class_counts.png, box_sizes.png and samples_grid.png.
"""
from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sitesafe.classes import CLASS_NAMES, NO_CLASSES  # noqa: E402
from sitesafe.paths import RESULTS, ROOT, SEED, SPLITS  # noqa: E402

OUT = RESULTS / "data"


def label_of(img: Path) -> Path:
    return Path(str(img).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")).with_suffix(".txt")


def read_split(name: str) -> list[Path]:
    return [ROOT / ln for ln in (SPLITS / f"{name}.txt").read_text().splitlines() if ln.strip()]


def read_labels(img: Path):
    p = label_of(img)
    rows = []
    if p.exists():
        for ln in p.read_text().splitlines():
            parts = ln.split()
            if len(parts) >= 5:
                rows.append((int(parts[0]), *map(float, parts[1:5])))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    counts, areas = {}, []
    for split in ("train", "val", "test"):
        c = Counter()
        for img in read_split(split):
            for cls, _, _, w, h in read_labels(img):
                c[cls] += 1
                areas.append((CLASS_NAMES[cls], w * h))
        counts[split] = c
    df = pd.DataFrame({s: [counts[s][i] for i in range(len(CLASS_NAMES))] for s in counts}, index=CLASS_NAMES)
    df["images"] = ""
    df.to_csv(OUT / "class_counts.csv")
    print(df)

    ax = df[["train", "val", "test"]].plot.bar(figsize=(10, 4), color=["#333333", "#888888", "#f5b700"])
    ax.set_ylabel("boxes"); ax.set_title("Boxes per class in each split")
    plt.tight_layout(); plt.savefig(OUT / "class_counts.png", dpi=150); plt.close()

    a = pd.DataFrame(areas, columns=["cls", "area"])
    med = a.groupby("cls")["area"].median().reindex(CLASS_NAMES)
    ax = (med * 100).plot.bar(figsize=(10, 4), color=["#c0392b" if i in NO_CLASSES else "#555555" for i in range(len(CLASS_NAMES))])
    ax.set_ylabel("median box area (% of image)"); ax.set_title("Median box size per class (red = no_* classes)")
    plt.tight_layout(); plt.savefig(OUT / "box_sizes.png", dpi=150); plt.close()

    rng = random.Random(SEED)
    picks = rng.sample(read_split("train"), 8)
    tiles = []
    for img_path in picks:
        im = Image.open(img_path).convert("RGB")
        W, H = im.size
        d = ImageDraw.Draw(im)
        for cls, cx, cy, w, h in read_labels(img_path):
            col = (220, 40, 40) if cls in NO_CLASSES else (60, 200, 90) if cls != 5 else (245, 183, 0)
            box = [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]
            d.rectangle(box, outline=col, width=max(2, W // 300))
            d.text((box[0] + 3, box[1] + 3), CLASS_NAMES[cls], fill=col)
        im.thumbnail((480, 480))
        tile = Image.new("RGB", (480, 480), (20, 20, 20))
        tile.paste(im, ((480 - im.width) // 2, (480 - im.height) // 2))
        tiles.append(tile)
    grid = Image.new("RGB", (480 * 4, 480 * 2))
    for i, t in enumerate(tiles):
        grid.paste(t, ((i % 4) * 480, (i // 4) * 480))
    grid.save(OUT / "samples_grid.jpg", quality=88)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
