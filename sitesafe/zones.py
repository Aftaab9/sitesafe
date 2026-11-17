"""Danger zones: user-drawn polygons and forklift proximity.

A person is "in" a zone when their foot point (bottom-centre of the person box) is
inside the polygon. The foot point is the best single-camera proxy for where a person
stands on the ground plane.

Forklift proximity has no depth information, so distance is measured in the image and
expressed in units of the person's own height. That makes it roughly scale-invariant:
"within 1.0 person-heights of a forklift" means about the same thing near and far.
Mirrored in web/src/lib/zones.ts.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

Point = Tuple[float, float]


def foot_point(box: Sequence[float]) -> Point:
    return ((box[0] + box[2]) / 2.0, box[3])


def point_in_polygon(pt: Point, poly: Sequence[Point]) -> bool:
    """Ray casting. Points exactly on an edge count as inside for horizontal-ray hits."""
    if len(poly) < 3:
        return False
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def box_gap(a: Sequence[float], b: Sequence[float]) -> float:
    """Shortest distance between two axis-aligned boxes (0 if they overlap)."""
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return (dx * dx + dy * dy) ** 0.5


def near_forklift(person: Sequence[float], forklift: Sequence[float], max_heights: float = 0.5) -> bool:
    h = max(person[3] - person[1], 1e-6)
    return box_gap(person, forklift) / h <= max_heights


def zone_events(person_boxes: Sequence[Sequence[float]], forklift_boxes: Sequence[Sequence[float]],
                zones: Sequence[Sequence[Point]], max_heights: float = 0.5) -> List[dict]:
    """One entry per person: indices of zones entered and whether a forklift is close."""
    out = []
    for box in person_boxes:
        fp = foot_point(box)
        out.append({
            "zones": [zi for zi, z in enumerate(zones) if point_in_polygon(fp, z)],
            "near_forklift": any(near_forklift(box, f, max_heights) for f in forklift_boxes),
        })
    return out
