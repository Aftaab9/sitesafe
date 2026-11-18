"""Person-PPE matching.

Each PPE (or no_PPE) box is assigned to at most one Person box. PPE boxes are much
smaller than person boxes, so plain IoU is close to zero even for a perfect match.
We therefore use intersection-over-PPE-area (IoA): the share of the PPE box that
lies inside the person box. A position check then makes sure the item sits where
that item is worn (helmet near the top, boots near the bottom, and so on).

The same algorithm is implemented in TypeScript in web/src/lib/matching.ts and both
are tested against the same fixture file (tests/fixtures/matching_cases.json).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from .classes import (DEFAULT_PROFILE, FORKLIFT, FORKLIFT_MIN_CONF, ITEM_BAND, ITEM_OF_CLASS,
                      PERSON, PPE_ITEMS, SITE_PROFILES)

Box = Sequence[float]  # x1, y1, x2, y2


@dataclass
class Det:
    box: Box
    cls: int
    conf: float = 1.0


@dataclass
class ItemStatus:
    state: str  # present | missing | unsure | not_visible
    conf: float = 0.0
    det_index: Optional[int] = None  # index into the detection list of the deciding box


@dataclass
class PersonResult:
    det_index: int
    box: Box
    conf: float
    items: Dict[str, ItemStatus] = field(default_factory=dict)
    status: str = "unsure"  # compliant | non_compliant | unsure
    missing: List[str] = field(default_factory=list)
    unsure_items: List[str] = field(default_factory=list)
    matched: List[int] = field(default_factory=list)  # all detection indices assigned to this person


@dataclass
class MatchConfig:
    ioa_min: float = 0.5          # min share of the PPE box inside the person box
    band_tol: float = 0.12        # tolerance on the vertical band, fraction of person height
    conf_unsure: float = 0.45     # a deciding box below this confidence gives "unsure"
    margin_unsure: float = 0.10   # worn-vs-missing confidence gap below this gives "unsure"
    person_unsure: float = 0.40   # person box below this confidence gives an unsure person
    # Optional per-class "unsure" thresholds chosen on validation (class id -> threshold)
    class_thresholds: Optional[Dict[int, float]] = None


def area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a: Box, b: Box) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def iou(a: Box, b: Box) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def ioa(ppe: Box, person: Box) -> float:
    """Intersection over the PPE box's own area."""
    a = area(ppe)
    return intersection(ppe, person) / a if a > 0 else 0.0


def in_band(item: str, ppe: Box, person: Box, tol: float) -> bool:
    """Is the PPE box centre inside the vertical band where this item is worn?

    Skipped for people who are not upright (box wider than tall), e.g. crouching or lying,
    because the head is then not necessarily at the top of the box.
    """
    pw, ph = person[2] - person[0], person[3] - person[1]
    if ph <= 0 or pw > ph:
        return True
    cy = ((ppe[1] + ppe[3]) / 2 - person[1]) / ph
    lo, hi = ITEM_BAND[item]
    return (lo - tol) <= cy <= (hi + tol)


def assign_ppe(dets: Sequence[Det], cfg: MatchConfig = MatchConfig()) -> Dict[int, List[int]]:
    """Assign each PPE / no_PPE detection to at most one person. Returns person_idx -> [det idx]."""
    persons = [i for i, d in enumerate(dets) if d.cls == PERSON]
    out: Dict[int, List[int]] = {p: [] for p in persons}
    for j, d in enumerate(dets):
        if d.cls not in ITEM_OF_CLASS:
            continue
        item, _ = ITEM_OF_CLASS[d.cls]
        best, best_score = None, 0.0
        for p in persons:
            pb = dets[p].box
            score = ioa(d.box, pb)
            if score < cfg.ioa_min or not in_band(item, d.box, pb, cfg.band_tol):
                continue
            # Tie-break overlapping people: prefer the person whose horizontal centre is closest
            # and, for equal IoA, the smaller (nearer-fitting) person box.
            pcx = (pb[0] + pb[2]) / 2
            dcx = (d.box[0] + d.box[2]) / 2
            pw = max(pb[2] - pb[0], 1e-6)
            score = score - 0.25 * min(abs(dcx - pcx) / pw, 1.0) - 1e-9 * area(pb)
            if best is None or score > best_score:
                best, best_score = p, score
        if best is not None:
            out[best].append(j)
    return out


def _threshold(cls: int, cfg: MatchConfig) -> float:
    if cfg.class_thresholds and cls in cfg.class_thresholds:
        return cfg.class_thresholds[cls]
    return cfg.conf_unsure


def decide_item(pos: Optional[tuple], neg: Optional[tuple], cfg: MatchConfig) -> ItemStatus:
    """pos / neg are (conf, det_index, cls) of the strongest worn / missing box, or None."""
    if pos is None and neg is None:
        return ItemStatus("not_visible")
    if pos is not None and neg is not None:
        win, lose = (pos, neg) if pos[0] >= neg[0] else (neg, pos)
        state = "present" if win is pos else "missing"
        if win[0] - lose[0] < cfg.margin_unsure or win[0] < _threshold(win[2], cfg):
            state = "unsure"
        return ItemStatus(state, win[0], win[1])
    win = pos if pos is not None else neg
    state = "present" if pos is not None else "missing"
    if win[0] < _threshold(win[2], cfg):
        state = "unsure"
    return ItemStatus(state, win[0], win[1])


def match_people(dets: Sequence[Det], required: Optional[Iterable[str]] = None,
                 cfg: MatchConfig = MatchConfig()) -> List[PersonResult]:
    """Full person-level compliance decision for one image."""
    required = list(required) if required is not None else SITE_PROFILES[DEFAULT_PROFILE]
    assigned = assign_ppe(dets, cfg)
    results: List[PersonResult] = []
    for p, idxs in assigned.items():
        res = PersonResult(det_index=p, box=list(dets[p].box), conf=dets[p].conf, matched=list(idxs))
        for item in PPE_ITEMS:
            pos = neg = None
            for j in idxs:
                it, worn = ITEM_OF_CLASS[dets[j].cls]
                if it != item:
                    continue
                cand = (dets[j].conf, j, dets[j].cls)
                if worn and (pos is None or cand[0] > pos[0]):
                    pos = cand
                if not worn and (neg is None or cand[0] > neg[0]):
                    neg = cand
            res.items[item] = decide_item(pos, neg, cfg)
        res.missing = [i for i in required if res.items[i].state == "missing"]
        res.unsure_items = [i for i in required if res.items[i].state == "unsure"]
        not_visible = [i for i in required if res.items[i].state == "not_visible"]
        res.unsure_items += not_visible
        if res.missing and res.conf >= cfg.person_unsure:
            res.status = "non_compliant"
        elif res.missing or res.unsure_items or res.conf < cfg.person_unsure:
            # Green is never shown without positive evidence for every required item. A required
            # item that is out of view or undetected sends the person to "check manually", because
            # a missed "no helmet" costs far more than an extra manual check.
            res.status = "unsure"
        else:
            res.status = "compliant"
        results.append(res)
    results.sort(key=lambda r: (r.box[0], r.box[1]))
    return results


def summarise(people: Sequence[PersonResult], dets: Sequence[Det]) -> dict:
    """Aggregate, anonymous statistics for one image. No identities, only counts."""
    n = len(people)
    compliant = sum(p.status == "compliant" for p in people)
    non = sum(p.status == "non_compliant" for p in people)
    unsure = sum(p.status == "unsure" for p in people)
    decided = compliant + non
    missing_counts = {i: 0 for i in PPE_ITEMS}
    for p in people:
        for i in p.missing:
            missing_counts[i] += 1
    return {
        "persons": n,
        "compliant": compliant,
        "non_compliant": non,
        "unsure": unsure,
        "compliance_pct": round(100.0 * compliant / decided, 1) if decided else None,
        "missing_by_item": missing_counts,
        "forklifts": sum(d.cls == FORKLIFT and d.conf >= FORKLIFT_MIN_CONF for d in dets),
    }
