"""Int8 quantisation of the exported ONNX model, with the accuracy cost measured.

    python scripts/quantize.py --name sitesafe_base

Static (calibrated) QDQ quantisation with per-channel weights. The last layers of the detection
head are left in float: they output box coordinates in pixels (0..640) next to class scores
(0..1), and squeezing both ranges through 8 bits is what usually destroys YOLO accuracy.
Calibration uses TRAIN images only. Writes results/eval/quantisation.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import onnx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sitesafe.detector import letterbox  # noqa: E402
from sitesafe.paths import MODELS, RESULTS, ROOT, SEED, SPLITS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="sitesafe_base")
    ap.add_argument("--calib", type=int, default=120)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()

    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    src = MODELS / f"{a.name}.onnx"
    pre = MODELS / f"_{a.name}_pre.onnx"
    dst = MODELS / f"{a.name}_int8.onnx"
    quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)

    model = onnx.load(str(pre))
    input_name = model.graph.input[0].name
    # Nodes of the final Detect module (highest-numbered /model.N/) that come after its conv towers.
    head = sorted({n.name.split("/")[1] for n in model.graph.node if n.name.startswith("/model.")},
                  key=lambda s: int(s.split(".")[1]))[-1]
    exclude = [n.name for n in model.graph.node
               if n.name.startswith(f"/{head}/") and (n.op_type not in ("Conv",) or "/dfl/" in n.name or "cv" not in n.name)]
    exclude += [n.name for n in model.graph.node if n.op_type in ("Sigmoid", "Mul", "Concat", "Softmax") and n.name.startswith(f"/{head}/")]

    rnd = random.Random(SEED)
    train = [ROOT / ln for ln in (SPLITS / "train.txt").read_text().splitlines() if ln.strip()]
    calib = rnd.sample(train, a.calib)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(calib)

        def get_next(self):
            p = next(self.it, None)
            if p is None:
                return None
            rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            canvas, _, _ = letterbox(rgb, 640)
            return {input_name: canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0}

    quantize_static(str(pre), str(dst), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax, nodes_to_exclude=sorted(set(exclude)),
                    extra_options={"ActivationSymmetric": False, "WeightSymmetric": True})
    pre.unlink(missing_ok=True)

    from evaluate import evaluate
    from export_onnx import cpu_speed
    fp = evaluate(str(src), a.split, f"{a.name}_fp32_onnx", plots=False)
    q = evaluate(str(dst), a.split, f"{a.name}_int8_onnx", plots=False)
    out = {"name": a.name, "split": a.split, "calibration_images": a.calib, "excluded_head_nodes": len(set(exclude)),
           "fp32_mb": round(src.stat().st_size / 1e6, 2), "int8_mb": round(dst.stat().st_size / 1e6, 2),
           "map50_fp32": fp["map50"], "map50_int8": q["map50"], "map50_95_fp32": fp["map50_95"], "map50_95_int8": q["map50_95"],
           "no_recall_fp32": fp["no_class_mean_recall"], "no_recall_int8": q["no_class_mean_recall"],
           "cpu_ms_fp32_4t": round(cpu_speed(src, 4), 1), "cpu_ms_int8_4t": round(cpu_speed(dst, 4), 1)}
    (RESULTS / "eval" / "quantisation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
