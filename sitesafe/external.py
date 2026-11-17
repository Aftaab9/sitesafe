"""An off-the-shelf PPE detector from Hugging Face, wrapped to speak our 12 classes.

killuminati1/construction-ppe-yolov8 (Apache-2.0): YOLOv8s trained on about 13,000 real construction
photos with 19 classes. It cannot be the delivered model (the brief asks for a YOLO11n I fine-tune
myself, and v8 weights do not load into YOLO11), so it is used in two other roles:
  1. an "off-the-shelf pre-trained" baseline, scored with exactly the same code as my model;
  2. a stronger teacher for pseudo-labelling classes that the extra datasets do not label.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .matching import Det
from .paths import MODELS

REPO = "killuminati1/construction-ppe-yolov8"
NAME_TO_OURS = {"Hard_hat": 0, "Glove": 1, "Vest": 2, "Boots": 3, "Glass": 4, "Worker": 5,
                "No-Helmet": 6, "No-Glass": 7, "No-Glove": 8, "No-Boots": 9, "No-Vest": 11}


def weights_path() -> Path:
    p = MODELS / "external_killuminati_ppe_yolov8s.pt"
    if not p.exists():
        from huggingface_hub import hf_hub_download
        src = hf_hub_download(REPO, "best.pt")
        p.write_bytes(Path(src).read_bytes())
    return p


class ExternalDetector:
    def __init__(self, imgsz: int = 640):
        from ultralytics import YOLO
        self.model = YOLO(str(weights_path()))
        self.map = {i: NAME_TO_OURS[n] for i, n in self.model.names.items() if n in NAME_TO_OURS}
        self.imgsz = imgsz

    def __call__(self, rgb: np.ndarray, conf_thr: float = 0.25, iou_thr: float = 0.5) -> List[Det]:
        r = self.model.predict(rgb[..., ::-1], conf=conf_thr, iou=iou_thr, imgsz=self.imgsz, verbose=False)[0]
        out = []
        for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            if int(c) in self.map:
                out.append(Det([float(v) for v in b], self.map[int(c)], float(s)))
        out.sort(key=lambda d: -d.conf)
        return out
