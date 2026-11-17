"""Evaluation helpers shared by all study scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .classes import CLASS_NAMES, NO_CLASSES, SITE_PROFILES
from .matching import Det, MatchConfig, PersonResult, iou, match_people
from .paths import ROOT, SPLITS

PredFn = Callable[[np.ndarray], List[Det]]  # RGB image -> detections


def load_split(name: str) -> List[Path]:
    f = SPLITS / f"{name}.txt"
    return [ROOT / ln for ln in f.read_text().splitlines() if ln.strip()]


def label_of(img: Path) -> Path:
    return Path(str(img).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")).with_suffix(".txt")


def read_gt(img: Path, w: int, h: int) -> List[Det]:
    out: List[Det] = []
    p = label_of(img)
    if p.exists():
        for ln in p.read_text().splitlines():
            a = ln.split()
            if len(a) >= 5:
                c = int(a[0]); cx, cy, bw, bh = map(float, a[1:5])
                out.append(Det([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], c, 1.0))
    return out


def read_rgb(path: Path, max_side: Optional[int] = None) -> np.ndarray:
    import cv2
    bgr = cv2.imread(str(path))
    if max_side:
        h, w = bgr.shape[:2]
        r = min(1.0, max_side / max(h, w))
        if r < 1:
            bgr = cv2.resize(bgr, (round(w * r), round(h * r)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def match_boxes(gt: Sequence[Det], dets: Sequence[Det], iou_thr: float = 0.5) -> Dict[int, int]:
    """Greedy same-class matching, most confident detections first. Returns gt index -> det index."""
    used_g: set = set()
    out: Dict[int, int] = {}
    for di in sorted(range(len(dets)), key=lambda i: -dets[i].conf):
        d = dets[di]
        best, bi = -1, iou_thr
        for gi, g in enumerate(gt):
            if gi in used_g or g.cls != d.cls:
                continue
            v = iou(d.box, g.box)
            if v >= bi:
                best, bi = gi, v
        if best >= 0:
            used_g.add(best)
            out[best] = di
    return out


def class_counts(images: Iterable[Path], pred_fn: PredFn, conf: float = 0.25, iou_thr: float = 0.5):
    """Per image and class: (n_gt, n_found, n_pred). Used for recall with bootstrap intervals."""
    rows = []
    nc = len(CLASS_NAMES)
    for p in images:
        rgb = read_rgb(p)
        h, w = rgb.shape[:2]
        gt = read_gt(p, w, h)
        dets = [d for d in pred_fn(rgb) if d.conf >= conf]
        m = match_boxes(gt, dets, iou_thr)
        n_gt = np.zeros(nc, int); n_tp = np.zeros(nc, int); n_pred = np.zeros(nc, int)
        for gi, g in enumerate(gt):
            n_gt[g.cls] += 1
            if gi in m:
                n_tp[g.cls] += 1
        for d in dets:
            n_pred[d.cls] += 1
        rows.append((p.name, n_gt, n_tp, n_pred))
    return rows


def recall_table(rows) -> Dict[str, float]:
    gt = sum(r[1] for r in rows); tp = sum(r[2] for r in rows)
    return {CLASS_NAMES[c]: (float(tp[c] / gt[c]) if gt[c] else float("nan")) for c in range(len(CLASS_NAMES))}


def bootstrap_recall_delta(rows_a, rows_b, cls: Sequence[int], n: int = 2000, seed: int = 42):
    """Paired bootstrap over images of recall_b - recall_a for the given classes (pooled)."""
    rng = np.random.default_rng(seed)
    ga = np.array([r[1][list(cls)].sum() for r in rows_a]); ta = np.array([r[2][list(cls)].sum() for r in rows_a])
    tb = np.array([r[2][list(cls)].sum() for r in rows_b])
    k = len(ga)
    deltas = []
    for _ in range(n):
        idx = rng.integers(0, k, k)
        g = ga[idx].sum()
        if g:
            deltas.append((tb[idx].sum() - ta[idx].sum()) / g)
    g = ga.sum()
    return {"a": float(ta.sum() / g) if g else float("nan"), "b": float(tb.sum() / g) if g else float("nan"),
            "delta": float((tb.sum() - ta.sum()) / g) if g else float("nan"),
            "lo": float(np.percentile(deltas, 2.5)) if deltas else float("nan"),
            "hi": float(np.percentile(deltas, 97.5)) if deltas else float("nan"), "n_gt": int(g)}


def bootstrap_ci(successes: Sequence[int], totals: Sequence[int], n: int = 2000, seed: int = 42):
    """Bootstrap interval for a pooled proportion where units (images / people) are resampled."""
    s = np.asarray(successes, float); t = np.asarray(totals, float)
    if t.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(s), len(s))
        if t[idx].sum():
            vals.append(s[idx].sum() / t[idx].sum())
    return float(s.sum() / t.sum()), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def person_rows(images: Iterable[Path], people_fn: Callable[[np.ndarray], List[PersonResult]],
                profile: str = "general", cfg: MatchConfig = MatchConfig()) -> List[dict]:
    """One row per GROUND-TRUTH person: what the labels say and what the method said."""
    required = SITE_PROFILES[profile]
    rows = []
    for p in images:
        rgb = read_rgb(p)
        h, w = rgb.shape[:2]
        gt = read_gt(p, w, h)
        gt_people = match_people(gt, required, cfg)
        pred_people = people_fn(rgb)
        used = set()
        for g in gt_people:
            best, bi = None, 0.5
            for k, q in enumerate(pred_people):
                if k in used:
                    continue
                v = iou(q.box, g.box)
                if v >= bi:
                    best, bi = k, v
            if best is not None:
                used.add(best)
            q = pred_people[best] if best is not None else None
            rows.append({"image": p.name, "box": [round(v, 1) for v in g.box], "img_w": w, "img_h": h,
                         "gt_status": g.status, "gt_missing": "|".join(g.missing),
                         "detected": q is not None, "pred_status": q.status if q else "not_detected",
                         "pred_missing": "|".join(q.missing) if q else "", "pred_conf": round(q.conf, 3) if q else 0.0,
                         "height_frac": round((g.box[3] - g.box[1]) / h, 3)})
        for k, q in enumerate(pred_people):
            if k not in used:
                rows.append({"image": p.name, "box": [round(v, 1) for v in q.box], "img_w": w, "img_h": h,
                             "gt_status": "no_gt_person", "gt_missing": "", "detected": True, "pred_status": q.status,
                             "pred_missing": "|".join(q.missing), "pred_conf": round(q.conf, 3),
                             "height_frac": round((q.box[3] - q.box[1]) / h, 3)})
    return rows


def compliance_metrics(rows: Sequence[dict]) -> dict:
    """Person-level metrics. "Decided" = the method said compliant or non_compliant (not unsure)."""
    real = [r for r in rows if r["gt_status"] in ("compliant", "non_compliant")]
    all_gt = [r for r in rows if r["gt_status"] != "no_gt_person"]
    decided = [r for r in real if r["pred_status"] in ("compliant", "non_compliant")]
    tp = sum(r["gt_status"] == "non_compliant" and r["pred_status"] == "non_compliant" for r in decided)
    fp = sum(r["gt_status"] == "compliant" and r["pred_status"] == "non_compliant" for r in decided)
    fn = sum(r["gt_status"] == "non_compliant" and r["pred_status"] == "compliant" for r in decided)
    tn = sum(r["gt_status"] == "compliant" and r["pred_status"] == "compliant" for r in decided)
    gt_nc = [r for r in real if r["gt_status"] == "non_compliant"]
    div = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
    return {
        "gt_people": len(all_gt), "gt_decidable": len(real), "gt_non_compliant": len(gt_nc),
        "person_recall": div(sum(r["detected"] for r in all_gt), len(all_gt)),
        "decided": len(decided),
        "accuracy": div(tp + tn, len(decided)),
        "precision_nc": div(tp, tp + fp),
        "recall_nc": div(tp, tp + fn),                       # among decided people
        "recall_nc_strict": div(tp, len(gt_nc)),             # unsure / undetected count as misses
        "nc_flagged_or_unsure": div(sum(r["pred_status"] in ("non_compliant", "unsure") for r in gt_nc), len(gt_nc)),
        "nc_shown_green": div(fn, len(gt_nc)),               # the costly error: a violator shown as compliant
        "unsure_share": div(sum(r["pred_status"] == "unsure" for r in real), len(real)),
        "not_detected_share": div(sum(not r["detected"] for r in real), len(real)),
        "false_person_boxes": sum(r["gt_status"] == "no_gt_person" for r in rows),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def no_class_ids() -> List[int]:
    return list(NO_CLASSES)
