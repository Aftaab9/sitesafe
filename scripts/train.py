"""Train YOLO11 on Construction-PPE. Same script locally and on Colab.

    python scripts/train.py --name base                       # real data only
    python scripts/train.py --name synth --data ppe_synth.yaml  # real + synthetic
    python scripts/train.py --name base_s --model yolo11s.pt

Every run appends one row (config, metrics, time) to results/runs.csv, so report tables
can be regenerated. `--resume` continues an interrupted run from its last checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sitesafe.paths import RESULTS, RUNS, SEED, SPLITS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default="ppe.yaml", help="yaml inside data/splits/")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--cache", default="ram")
    ap.add_argument("--project", default=str(RUNS))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    run_dir = Path(args.project) / args.name
    last = run_dir / "weights" / "last.pt"
    t0 = time.time()
    if args.resume and last.exists():
        print(f"resuming from {last}")
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        model.train(data=str(SPLITS / args.data), epochs=args.epochs, patience=args.patience,
                    imgsz=args.imgsz, batch=args.batch, workers=args.workers, device=args.device,
                    seed=SEED, deterministic=True, cache=args.cache, project=args.project,
                    name=args.name, exist_ok=True, plots=True)
    minutes = (time.time() - t0) / 60

    best = YOLO(str(run_dir / "weights" / "best.pt"))
    m = best.val(data=str(SPLITS / args.data), split="val", imgsz=args.imgsz, plots=False, verbose=False)
    row = {"name": args.name, "model": args.model, "data": args.data, "epochs": args.epochs,
           "imgsz": args.imgsz, "batch": args.batch, "seed": SEED, "train_minutes": round(minutes, 1),
           "val_map50": round(float(m.box.map50), 4), "val_map50_95": round(float(m.box.map), 4),
           "val_precision": round(float(m.box.mp), 4), "val_recall": round(float(m.box.mr), 4)}
    RESULTS.mkdir(exist_ok=True)
    csv_path = RESULTS / "runs.csv"
    new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
