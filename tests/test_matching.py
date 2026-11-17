import json
from pathlib import Path

import pytest

from sitesafe.matching import Det, MatchConfig, assign_ppe, ioa, iou, match_people, summarise

CASES = json.loads((Path(__file__).parent / "fixtures" / "matching_cases.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_fixture_cases(case):
    """The same fixture file is run against the TypeScript port (web/src/lib/matching.test.ts)."""
    dets = [Det(**d) for d in case["dets"]]
    people = match_people(dets, case["required"])
    assert [{"status": p.status, "missing": p.missing} for p in people] == case["expected"]


def test_ioa_vs_iou_for_small_ppe():
    person = [0, 0, 100, 400]
    helmet = [30, 0, 70, 40]
    assert ioa(helmet, person) == pytest.approx(1.0)
    assert iou(helmet, person) < 0.05  # why plain IoU cannot be used for matching


def test_each_ppe_box_assigned_once():
    dets = [Det([0, 0, 200, 400], 5, .9), Det([100, 0, 300, 400], 5, .9), Det([120, 0, 180, 50], 0, .9)]
    assigned = assign_ppe(dets)
    assert sum(len(v) for v in assigned.values()) == 1


def test_partial_overlap_below_threshold_not_assigned():
    dets = [Det([0, 0, 100, 400], 5, .9), Det([80, 0, 180, 60], 0, .9)]  # only 20% inside
    assert assign_ppe(dets)[0] == []


def test_per_class_thresholds_override_default():
    dets = [Det([100, 100, 300, 600], 5, .9), Det([160, 100, 240, 170], 6, .5)]
    assert match_people(dets, ["helmet"])[0].status == "non_compliant"
    strict = MatchConfig(class_thresholds={6: 0.7})
    assert match_people(dets, ["helmet"], strict)[0].status == "unsure"


def test_summary_is_aggregate_only():
    dets = [Det([100, 100, 300, 600], 5, .9), Det([160, 100, 240, 170], 6, .8),
            Det([120, 220, 280, 400], 2, .9), Det([500, 100, 700, 600], 5, .9),
            Det([560, 100, 640, 170], 0, .9), Det([520, 220, 680, 400], 2, .9),
            Det([800, 300, 1100, 600], 10, .9)]
    s = summarise(match_people(dets, ["helmet", "vest"]), dets)
    assert s["persons"] == 2 and s["compliant"] == 1 and s["non_compliant"] == 1
    assert s["compliance_pct"] == 50.0 and s["missing_by_item"]["helmet"] == 1 and s["forklifts"] == 1
    assert set(s) == {"persons", "compliant", "non_compliant", "unsure", "compliance_pct",
                      "missing_by_item", "forklifts"}


def test_unsure_people_excluded_from_compliance_pct():
    dets = [Det([100, 100, 300, 600], 5, .9)]
    s = summarise(match_people(dets, ["helmet"]), dets)
    assert s["unsure"] == 1 and s["compliance_pct"] is None
