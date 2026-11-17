from sitesafe.matching import ItemStatus, PersonResult
from sitesafe.tracking import Tracker, TrackerConfig
from sitesafe.zones import box_gap, foot_point, near_forklift, point_in_polygon, zone_events


def person(x, helmet="present", conf=0.9, y=100):
    items = {i: ItemStatus("present", 0.9) for i in ("helmet", "goggles", "vest", "gloves", "boots")}
    items["helmet"] = ItemStatus(helmet, 0.9)
    status = "non_compliant" if helmet == "missing" else "compliant"
    return PersonResult(det_index=0, box=[x, y, x + 100, y + 300], conf=conf, items=items, status=status,
                        missing=["helmet"] if helmet == "missing" else [])


def test_ids_stay_stable_while_walking():
    tr = Tracker()
    ids = []
    for f in range(10):
        seen = tr.update([person(100 + 15 * f), person(600 - 15 * f)])
        ids.append(sorted(t.id for t in seen))
    assert all(i == [1, 2] for i in ids)
    assert tr.total_workers == 2


def test_track_survives_short_occlusion():
    tr = Tracker()
    for f in range(4):
        tr.update([person(100 + 10 * f)])
    for _ in range(3):
        tr.update([])  # occluded
    seen = tr.update([person(100 + 10 * 7)])
    assert [t.id for t in seen] == [1]


def test_track_dropped_after_long_absence_and_new_id_issued():
    tr = Tracker(TrackerConfig(max_misses=2))
    tr.update([person(100)])
    for _ in range(4):
        tr.update([])
    seen = tr.update([person(100)])
    assert [t.id for t in seen] == [2]


def test_low_confidence_box_cannot_start_a_track_but_keeps_one_alive():
    tr = Tracker()
    assert tr.update([person(100, conf=0.3)]) == []
    tr.update([person(100, conf=0.9)])
    seen = tr.update([person(104, conf=0.3)])
    assert [t.id for t in seen] == [1]


def test_single_frame_flicker_is_not_a_violation():
    tr = Tracker(TrackerConfig(window=5, min_hits=3))
    states = ["present", "present", "missing", "present", "present"]
    for s in states:
        t = tr.update([person(100, helmet=s)])[0]
    assert tr.smoothed_missing(t, ["helmet"]) == []
    assert tr.smoothed_status(t, ["helmet"]) == "compliant"


def test_persistent_violation_is_raised():
    tr = Tracker(TrackerConfig(window=5, min_hits=3))
    for s in ["present", "missing", "missing", "present", "missing"]:
        t = tr.update([person(100, helmet=s)])[0]
    assert tr.smoothed_missing(t, ["helmet"]) == ["helmet"]
    assert tr.smoothed_status(t, ["helmet"]) == "non_compliant"


def test_point_in_polygon_square_and_concave():
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon((5, 5), sq) and not point_in_polygon((15, 5), sq)
    l_shape = [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)]
    assert point_in_polygon((2, 8), l_shape) and not point_in_polygon((8, 8), l_shape)
    assert not point_in_polygon((1, 1), [(0, 0), (5, 5)])  # not a polygon


def test_zone_uses_feet_not_head():
    zone = [(0, 350), (400, 350), (400, 500), (0, 500)]
    standing_in = [100, 100, 200, 400]   # feet at y=400, inside
    head_only_in = [100, 300, 200, 700]  # head overlaps the zone, feet at 700, outside
    ev = zone_events([standing_in, head_only_in], [], [zone])
    assert ev[0]["zones"] == [0] and ev[1]["zones"] == []
    assert foot_point(standing_in) == (150, 400)


def test_forklift_proximity_scales_with_person_height():
    forklift = [300, 100, 500, 400]
    near = [180, 100, 280, 400]    # gap 20 px, height 300 -> 0.07 heights
    far = [0, 100, 60, 280]        # gap 240 px, height 180 -> 1.33 heights
    assert box_gap(near, forklift) == 20
    assert near_forklift(near, forklift, 0.5) and not near_forklift(far, forklift, 0.5)
    assert box_gap([0, 0, 10, 10], [5, 5, 20, 20]) == 0
