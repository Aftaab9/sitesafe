"""Face detection (YuNet) and blurring.

Design decision, supported by the face-blur impact study (scripts/blur_study.py):
the detector always runs on the ORIGINAL frame held in memory, and blurring is applied
only to what is displayed or exported. Nothing is written to disk.

Two layers of protection:
1. YuNet face boxes are blurred.
2. "Strict" mode also blurs the head region of every detected person where YuNet found no
   face (far away, turned away, hard hat shadow). A missed face then still gets blurred.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np


class FaceDetector:
    def __init__(self, model_path: str, score_thr: float = 0.6):
        import cv2
        self.det = cv2.FaceDetectorYN.create(model_path, "", (320, 320), score_thr, 0.3, 5000)

    def __call__(self, rgb: np.ndarray) -> List[List[float]]:
        import cv2
        h, w = rgb.shape[:2]
        # YuNet finds small faces better on a larger input; cap the long side at 1280 for speed.
        s = min(1.0, 1280 / max(h, w))
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if s < 1.0:
            img = cv2.resize(img, (int(w * s), int(h * s)))
        self.det.setInputSize((img.shape[1], img.shape[0]))
        _, faces = self.det.detect(img)
        out = []
        if faces is not None:
            for f in faces:
                x, y, fw, fh = f[:4] / s
                out.append([float(x), float(y), float(x + fw), float(y + fh)])
        return out


def head_region(person: Sequence[float]) -> List[float]:
    """Where the FACE normally is in a person box: just below the top, so a hard hat stays visible.

    Blurring the whole head would hide the helmet, which is the thing the officer needs to see.
    """
    x1, y1, x2, y2 = person
    w, h = x2 - x1, y2 - y1
    hh = min(0.14 * h, 0.7 * w)
    cx = (x1 + x2) / 2
    hw = min(w, max(0.34 * w, 0.9 * hh)) / 2
    top = y1 + 0.06 * h
    return [cx - hw, top, cx + hw, top + hh]


def blur_regions(rgb: np.ndarray, boxes: Sequence[Sequence[float]], expand: float = 0.15) -> np.ndarray:
    """Return a copy with every box heavily Gaussian-blurred. Kernel scales with the box size."""
    import cv2
    out = rgb.copy()
    H, W = out.shape[:2]
    for b in boxes:
        bw, bh = b[2] - b[0], b[3] - b[1]
        x1 = int(max(0, b[0] - expand * bw)); y1 = int(max(0, b[1] - expand * bh))
        x2 = int(min(W, b[2] + expand * bw)); y2 = int(min(H, b[3] + expand * bh))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        k = max(9, int(max(x2 - x1, y2 - y1) * 0.6) | 1)
        roi = out[y1:y2, x1:x2]
        # Down-up sampling first makes the blur non-invertible in practice, then Gaussian smooths it.
        small = cv2.resize(roi, (max(1, (x2 - x1) // 12), max(1, (y2 - y1) // 12)), interpolation=cv2.INTER_AREA)
        roi = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
        out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    return out


def privacy_boxes(faces: Sequence[Sequence[float]], persons: Sequence[Sequence[float]], strict: bool = True):
    """Face boxes plus (strict mode) head regions of persons with no detected face."""
    boxes = [list(f) for f in faces]
    if strict:
        for p in persons:
            hr = head_region(p)
            covered = any(f[0] < hr[2] and f[2] > hr[0] and f[1] < hr[3] and f[3] > hr[1] for f in faces)
            if not covered:
                boxes.append(hr)
    return boxes
