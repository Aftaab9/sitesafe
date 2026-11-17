"""Speed on CPU with ONNX Runtime, end to end (pre-processing, model, NMS, matching, face blur).

    python scripts/speed_bench.py

This is what the Streamlit app does per image. Threads 1 and 2 approximate a free cloud CPU.
Browser timings (WebGPU and WASM, laptop and phone) are read from the app itself, which prints
the time under every analysed photo; record them in results/eval/speed_browser.json as
[{"label": "...", "ms": 123}, ...] and they appear on the model card.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sitesafe.classes import SITE_PROFILES  # noqa: E402
from sitesafe.detector import Detector  # noqa: E402
from sitesafe.evalutil import load_split, read_rgb  # noqa: E402
from sitesafe.matching import match_people  # noqa: E402
from sitesafe.paths import MODELS, RESULTS  # noqa: E402
from sitesafe.privacy import FaceDetector, blur_regions, privacy_boxes  # noqa: E402


def main():
    images = [read_rgb(p, 1600) for p in load_split("test")[:30]]
    faces = FaceDetector(str(MODELS / "face_detection_yunet_2023mar.onnx"))
    out = {}
    for threads in (1, 2, 4):
        t0 = time.perf_counter()
        det = Detector(str(MODELS / "sitesafe_base.onnx"), threads=threads)
        det(images[0])
        cold = (time.perf_counter() - t0) * 1000
        times = []
        for rgb in images:
            t = time.perf_counter()
            dets = det(rgb)
            people = match_people(dets, SITE_PROFILES["general"])
            blur_regions(rgb, privacy_boxes(faces(rgb), [p.box for p in people], True))
            times.append((time.perf_counter() - t) * 1000)
        out[f"threads_{threads}"] = {"cold_start_ms": round(cold), "mean_ms": round(float(np.mean(times)), 1),
                                     "p95_ms": round(float(np.percentile(times, 95)), 1)}
    clip_frames = 30 * 3  # a 30 second clip sampled at 3 fps
    out["clip_30s_at_3fps_seconds_2_threads"] = round(out["threads_2"]["mean_ms"] * clip_frames / 1000, 1)
    (RESULTS / "eval").mkdir(parents=True, exist_ok=True)
    (RESULTS / "eval" / "speed_cpu.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
