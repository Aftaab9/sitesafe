"""Lightweight ByteTrack-style tracker and temporal smoothing for clips.

* Association uses IoU between a constant-velocity prediction of each track and the
  new person boxes, in two stages like ByteTrack: confident detections first, then the
  low-confidence ones are used only to keep existing tracks alive.
* Each track gets an anonymous id (Worker 1, Worker 2, ...). No appearance features are
  stored, so the tracker cannot re-identify a person across clips. That is deliberate.
* Temporal smoothing: an item is reported as a violation only if it was "missing" in at
  least `k` of the last `n` sampled frames. This removes single-frame flicker.

Mirrored in web/src/lib/tracking.ts.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence

from .matching import PersonResult, iou
from .classes import PPE_ITEMS


@dataclass
class Track:
    id: int
    box: List[float]
    vel: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    hits: int = 1
    misses: int = 0
    history: Dict[str, Deque[str]] = field(default_factory=dict)
    status_history: Deque[str] = field(default_factory=deque)
    last: Optional[PersonResult] = None

    def predict(self) -> List[float]:
        return [b + v for b, v in zip(self.box, self.vel)]


@dataclass
class TrackerConfig:
    high_conf: float = 0.5
    iou_high: float = 0.3
    iou_low: float = 0.5
    max_misses: int = 8       # sampled frames a track survives without a detection
    window: int = 5           # n: smoothing window in sampled frames
    min_hits: int = 3         # k: frames within the window needed to raise a violation
    vel_alpha: float = 0.6    # velocity smoothing factor


class Tracker:
    def __init__(self, cfg: TrackerConfig = TrackerConfig()):
        self.cfg = cfg
        self.tracks: List[Track] = []
        self._next_id = 1

    def _associate(self, tracks: List[Track], people: List[PersonResult], thr: float):
        """Greedy IoU matching, best pairs first. Returns (pairs, unmatched_tracks, unmatched_people)."""
        cands = []
        for ti, t in enumerate(tracks):
            pred = t.predict()
            for pi, p in enumerate(people):
                v = iou(pred, p.box)
                if v >= thr:
                    cands.append((v, ti, pi))
        cands.sort(reverse=True)
        used_t, used_p, pairs = set(), set(), []
        for v, ti, pi in cands:
            if ti in used_t or pi in used_p:
                continue
            used_t.add(ti)
            used_p.add(pi)
            pairs.append((ti, pi))
        return (pairs,
                [t for i, t in enumerate(tracks) if i not in used_t],
                [p for i, p in enumerate(people) if i not in used_p])

    def _update_track(self, t: Track, p: PersonResult):
        a = self.cfg.vel_alpha
        new_vel = [nb - ob for nb, ob in zip(p.box, t.box)]
        t.vel = [a * nv + (1 - a) * ov for nv, ov in zip(new_vel, t.vel)]
        t.box = list(p.box)
        t.hits += 1
        t.misses = 0
        t.last = p
        for item in PPE_ITEMS:
            h = t.history.setdefault(item, deque(maxlen=self.cfg.window))
            h.append(p.items[item].state if item in p.items else "not_visible")
        if t.status_history.maxlen != self.cfg.window:
            t.status_history = deque(t.status_history, maxlen=self.cfg.window)
        t.status_history.append(p.status)

    def update(self, people: Sequence[PersonResult]) -> List[Track]:
        """Feed the matched people of one sampled frame. Returns the tracks seen in this frame."""
        cfg = self.cfg
        high = [p for p in people if p.conf >= cfg.high_conf]
        low = [p for p in people if p.conf < cfg.high_conf]

        pairs, rest_tracks, rest_high = self._associate(self.tracks, high, cfg.iou_high)
        seen: List[Track] = []
        for ti, pi in pairs:
            self._update_track(self.tracks[ti], high[pi])
            seen.append(self.tracks[ti])

        pairs2, lost_tracks, _ = self._associate(rest_tracks, low, cfg.iou_low)
        for ti, pi in pairs2:
            self._update_track(rest_tracks[ti], low[pi])
            seen.append(rest_tracks[ti])
        matched2 = {id(rest_tracks[ti]) for ti, _ in pairs2}
        for t in rest_tracks:
            if id(t) not in matched2:
                t.misses += 1
                t.box = t.predict()

        for p in rest_high:  # only confident detections may start a new track
            t = Track(id=self._next_id, box=list(p.box))
            self._next_id += 1
            t.hits = 0
            self._update_track(t, p)
            self.tracks.append(t)
            seen.append(t)

        self.tracks = [t for t in self.tracks if t.misses <= cfg.max_misses]
        return seen

    def smoothed_missing(self, t: Track, required: Sequence[str]) -> List[str]:
        """Items that have been missing in at least k of the last n sampled frames."""
        out = []
        for item in required:
            h = t.history.get(item)
            if h and sum(s == "missing" for s in h) >= self.cfg.min_hits:
                out.append(item)
        return out

    def smoothed_status(self, t: Track, required: Sequence[str]) -> str:
        if self.smoothed_missing(t, required):
            return "non_compliant"
        h = list(t.status_history)
        if not h:
            return "unsure"
        if sum(s == "compliant" for s in h) >= max(1, (len(h) + 1) // 2):
            return "compliant"
        return "unsure"

    @property
    def total_workers(self) -> int:
        return self._next_id - 1
