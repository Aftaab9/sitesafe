"""Export a trained checkpoint to ONNX for ONNX Runtime (Python and Web) and check it.

    python scripts/export_onnx.py --weights runs/base/weights/best.pt --name sitesafe_base

Opset 17 is supported by onnxruntime-web 1.22 (WASM and WebGPU). The export is static 640x640,
batch 1, no NMS in the graph: NMS is done in our own code so Python and the browser behave the same.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sitesafe.paths import MODELS, RESULTS  # noqa: E402


def cpu_speed(path: Path, threads: int = 2, runs: int = 30) -> float:
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    s = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 3, 640, 640).astype(np.float32)
    name = s.get_inputs()[0].name
    for _ in range(3):
        s.run(None, {name: x})
    t = time.perf_counter()
    for _ in range(runs):
        s.run(None, {name: x})
    return (time.perf_counter() - t) / runs * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()

    from ultralytics import YOLO
    MODELS.mkdir(exist_ok=True)
    tmp = MODELS / f"_{a.name}.pt"
    shutil.copy(a.weights, tmp)  # export next to a copy so the run folder stays clean
    exported = Path(YOLO(str(tmp)).export(format="onnx", opset=a.opset, imgsz=640, simplify=True, dynamic=False, half=False))
    out = MODELS / f"{a.name}.onnx"
    shutil.move(str(exported), out)
    pt_out = MODELS / f"{a.name}.pt"
    shutil.move(str(tmp), pt_out)

    import onnxruntime as ort
    s = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    shape = s.get_outputs()[0].shape
    info = {"name": a.name, "file": out.name, "size_mb": round(out.stat().st_size / 1e6, 2), "opset": a.opset,
            "input": s.get_inputs()[0].shape, "output": shape,
            "cpu_ms_2_threads": round(cpu_speed(out, 2), 1), "cpu_ms_4_threads": round(cpu_speed(out, 4), 1)}
    (RESULTS / "eval").mkdir(parents=True, exist_ok=True)
    (RESULTS / "eval" / f"export_{a.name}.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    assert out.stat().st_size < 100e6, "model must stay under 100 MB"


if __name__ == "__main__":
    main()
