"""ONNX Runtime inference for the fine-tuned YOLO11 detector (no PyTorch needed at run time).

Used by the Streamlit app and by the evaluation scripts that must measure exactly what
is deployed. The pre/post-processing matches web/src/lib/yolo.ts step for step.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .classes import NUM_CLASSES
from .matching import Det


def letterbox(img: np.ndarray, size: int = 640) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Resize keeping aspect ratio, pad with grey (114) to size x size. img is HxWx3 uint8 RGB."""
    import cv2
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas, r, (dx, dy)


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> List[int]:
    order = scores.argsort()[::-1]
    keep = []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        ious = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[ious <= thr]
    return keep


def decode(output: np.ndarray, ratio: float, pad: Tuple[float, float], orig_hw: Tuple[int, int],
           conf_thr: float = 0.25, iou_thr: float = 0.5, num_classes: int = NUM_CLASSES) -> List[Det]:
    """Decode a raw YOLO11 head output of shape (1, 4 + nc, N) into detections in original pixels."""
    pred = output[0].T  # (N, 4 + nc)
    scores = pred[:, 4:4 + num_classes]
    cls = scores.argmax(1)
    conf = scores.max(1)
    mask = conf >= conf_thr
    pred, cls, conf = pred[mask], cls[mask], conf[mask]
    if not len(pred):
        return []
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / ratio
    H, W = orig_hw
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
    dets: List[Det] = []
    for c in np.unique(cls):  # class-aware NMS
        idx = np.where(cls == c)[0]
        for k in nms(boxes[idx], conf[idx], iou_thr):
            i = idx[k]
            dets.append(Det(box=[float(v) for v in boxes[i]], cls=int(c), conf=float(conf[i])))
    dets.sort(key=lambda d: -d.conf)
    return dets


class Detector:
    def __init__(self, model_path: str, size: int = 640, threads: int = 0):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.size = size

    def __call__(self, rgb: np.ndarray, conf_thr: float = 0.25, iou_thr: float = 0.5) -> List[Det]:
        canvas, r, pad = letterbox(rgb, self.size)
        x = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        out = self.session.run(None, {self.input_name: x})[0]
        return decode(out, r, pad, rgb.shape[:2], conf_thr, iou_thr)
