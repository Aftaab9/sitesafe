"""Collect candidate images for the 30-image own test set from Wikimedia Commons.

    python scripts/fetch_own_images.py --n 90
    python scripts/fetch_own_images.py --select data/own_test/selected.txt   # keep only the chosen ones

Only files whose licence is public domain, CC0, CC BY or CC BY-SA are kept. For every image the
source page, licence, author and credit line are written to data/own_test/sources.csv, which the
report and README cite. Images are fetched as 1280-pixel thumbnails. The client identifies itself
with a generic project name as Wikimedia's API policy asks; no personal contact data is sent.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sitesafe.paths import OWN  # noqa: E402

API = "https://commons.wikimedia.org/w/api.php"
UA = "SiteSafeCourseProject/1.0 (educational computer-vision coursework; python-urllib)"
OK_LICENCES = re.compile(r"^(cc0|public domain|pd|cc[- ]by(-sa)?[- ]?\d)", re.I)

QUERIES = [
    "construction workers hard hats site",
    "construction worker safety vest",
    "construction site workers helmet",
    "road works workers high visibility",
    "building site labourers",
    "construction workers night",
    "workers scaffolding helmet",
    "construction workers rain",
    "warehouse forklift worker",
    "bricklayer construction site",
    "construction workers india site",
    "construction workers africa site",
    "steel fixers construction",
    "railway track workers orange",
    "construction workers concrete pouring",
]


def get(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception:
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"Commons API failed: {params}")


def strip_html(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def search(query: str, limit: int = 40) -> list[dict]:
    d = get({"action": "query", "generator": "search", "gsrsearch": f"{query} filetype:bitmap", "gsrnamespace": 6,
             "gsrlimit": limit, "prop": "imageinfo", "iiprop": "url|size|extmetadata|mime", "iiurlwidth": 1280})
    out = []
    for page in (d.get("query", {}).get("pages", {}) or {}).values():
        ii = (page.get("imageinfo") or [{}])[0]
        meta = ii.get("extmetadata", {})
        lic = strip_html(meta.get("LicenseShortName", {}).get("value", ""))
        if ii.get("mime") not in ("image/jpeg", "image/png") or not OK_LICENCES.match(lic):
            continue
        if min(ii.get("width", 0), ii.get("height", 0)) < 600:
            continue
        out.append({"title": page["title"], "page": ii.get("descriptionurl", ""), "thumb": ii.get("thumburl") or ii.get("url"),
                    "licence": lic, "licence_url": meta.get("LicenseUrl", {}).get("value", ""),
                    "author": strip_html(meta.get("Artist", {}).get("value", ""))[:120],
                    "credit": strip_html(meta.get("Credit", {}).get("value", ""))[:160]})
    return out


def download(url: str, dest: Path):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        dest.write_bytes(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=90, help="candidates to download")
    ap.add_argument("--select", default=None, help="text file with chosen file names; others are removed")
    a = ap.parse_args()
    cand_dir = OWN / "candidates"
    img_dir = OWN / "images"
    if a.select:
        chosen = [ln.strip() for ln in Path(a.select).read_text().splitlines() if ln.strip()]
        rows = {r["file"]: r for r in csv.DictReader((OWN / "candidates.csv").open(encoding="utf-8"))}
        img_dir.mkdir(parents=True, exist_ok=True)
        with (OWN / "sources.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(next(iter(rows.values())).keys()))
            w.writeheader()
            for name in chosen:
                (img_dir / name).write_bytes((cand_dir / name).read_bytes())
                w.writerow(rows[name])
        print(f"kept {len(chosen)} images in {img_dir}")
        return

    cand_dir.mkdir(parents=True, exist_ok=True)
    seen, rows = set(), []
    for q in QUERIES:
        for r in search(q):
            if r["title"] in seen:
                continue
            seen.add(r["title"])
            rows.append(r)
        time.sleep(0.5)
    print(f"{len(rows)} licensed candidates found")
    out = []
    for i, r in enumerate(rows[: a.n]):
        name = f"c{i:03d}.jpg"
        try:
            download(r["thumb"], cand_dir / name)
        except Exception as e:  # noqa: BLE001
            print("skip", r["title"], e)
            continue
        out.append({"file": name, **r})
        time.sleep(0.3)
    with (OWN / "candidates.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"downloaded {len(out)} candidates to {cand_dir}")


if __name__ == "__main__":
    main()
