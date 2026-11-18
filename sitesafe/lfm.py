"""LFM3-Omni (Liquid AI, 1.7B vision-language model) as a whole-image PPE "detector", via a local Ollama.

The model is asked for every person with a box (0-1000 normalised, the convention it follows) and
whether they wear a hard hat and a hi-vis vest. Its answers are turned into ordinary detections
(Person, helmet / no_helmet, vest / no_vest) placed in the body bands the matching code expects, so the
same matching, profiles, scoring and drawing code applies. Gloves, boots and goggles are not asked.
Same logic as web/src/lib/ollama.ts.
"""
from __future__ import annotations

import base64
import json
import urllib.request
from typing import List

import cv2
import numpy as np

from .matching import Det

URL = "http://localhost:11434/api/generate"
SCHEMA = {"type": "object", "properties": {"workers": {"type": "array", "items": {"type": "object", "properties": {
    "box": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
    "hard_hat": {"enum": ["yes", "no", "unsure"]}, "hi_vis": {"enum": ["yes", "no", "unsure"]}},
    "required": ["box", "hard_hat", "hi_vis"]}}}, "required": ["workers"]}
PROMPT = ("Find every person in this construction-site photo. For each person give a bounding box [x1, y1, x2, y2] "
          "as numbers from 0 to 1000 relative to the image width and height, and say whether they wear a hard hat "
          "and a high-visibility vest or jacket (yes, no or unsure).")
OPTIONS = {"temperature": 0, "num_ctx": 4096}


def workers_to_dets(workers: list, w: int, h: int, conf: float = 0.9) -> List[Det]:
    out = []
    for wk in workers:
        b = wk.get("box") or []
        if len(b) != 4:
            continue
        x1, y1, x2, y2 = [float(np.clip(v, 0, 1000)) / 1000 * s for v, s in zip(b, (w, h, w, h))]
        x1, x2, y1, y2 = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        pw, ph = x2 - x1, y2 - y1
        out.append(Det([x1, y1, x2, y2], 5, conf))
        head = [x1 + 0.25 * pw, y1, x2 - 0.25 * pw, y1 + 0.15 * ph]
        torso = [x1 + 0.15 * pw, y1 + 0.25 * ph, x2 - 0.15 * pw, y1 + 0.55 * ph]
        if wk.get("hard_hat") in ("yes", "no"):
            out.append(Det(head, 0 if wk["hard_hat"] == "yes" else 6, conf))
        if wk.get("hi_vis") in ("yes", "no"):
            out.append(Det(torso, 2 if wk["hi_vis"] == "yes" else 11, conf))
    return out


class LfmDetector:
    def __init__(self, model: str = "lfm3-omni", max_side: int = 1024):
        self.model, self.max_side = model, max_side
        self.last_seconds = 0.0

    def __call__(self, rgb: np.ndarray, conf_thr: float = 0.25, iou_thr: float = 0.5) -> List[Det]:
        h, w = rgb.shape[:2]
        s = min(1.0, self.max_side / max(h, w))
        small = cv2.resize(rgb, (int(w * s), int(h * s))) if s < 1 else rgb
        b64 = base64.b64encode(cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])[1]).decode()
        body = json.dumps({"model": self.model, "prompt": PROMPT, "images": [b64], "stream": False,
                           "format": SCHEMA, "options": OPTIONS}).encode()
        req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=900))
        self.last_seconds = r.get("total_duration", 0) / 1e9
        try:
            workers = json.loads(r["response"]).get("workers", [])
        except (json.JSONDecodeError, AttributeError):
            workers = []
        return workers_to_dets(workers, w, h)
