// The TypeScript ports are tested against the SAME fixture file as the Python originals,
// so the web app and the Streamlit app cannot silently disagree.
import { describe, expect, it } from 'vitest';
import cases from '../../../tests/fixtures/matching_cases.json';
import type { Item } from './classes';
import { ioa, iou, matchPeople, summarise, type Box, type Det, type ItemStatus, type PersonResult } from './matching';
import { closeJson, parseNarrative, verifyNarrative } from './narrative';
import { Tracker } from './tracking';
import { decode, letterboxParams, nms } from './yolo';
import { boxGap, nearForklift, pointInPolygon, zoneEvents } from './zones';

describe('matching (shared fixtures)', () => {
  for (const c of cases as { name: string; required: Item[]; dets: Det[]; expected: { status: string; missing: string[] }[] }[]) {
    it(c.name, () => {
      const people = matchPeople(c.dets, c.required);
      expect(people.map((p) => ({ status: p.status, missing: p.missing }))).toEqual(c.expected);
    });
  }
  it('uses intersection-over-PPE-area, not IoU', () => {
    const person: Box = [0, 0, 100, 400];
    const helmet: Box = [30, 0, 70, 40];
    expect(ioa(helmet, person)).toBeCloseTo(1);
    expect(iou(helmet, person)).toBeLessThan(0.05);
  });
  it('summary is aggregate only', () => {
    const dets: Det[] = [
      { box: [100, 100, 300, 600], cls: 5, conf: 0.9 }, { box: [160, 100, 240, 170], cls: 6, conf: 0.8 }, { box: [120, 220, 280, 400], cls: 2, conf: 0.9 },
      { box: [500, 100, 700, 600], cls: 5, conf: 0.9 }, { box: [560, 100, 640, 170], cls: 0, conf: 0.9 }, { box: [520, 220, 680, 400], cls: 2, conf: 0.9 },
      { box: [800, 300, 1100, 600], cls: 10, conf: 0.9 },
    ];
    const s = summarise(matchPeople(dets, ['helmet', 'vest']), dets);
    expect(s).toMatchObject({ persons: 2, compliant: 1, nonCompliant: 1, compliancePct: 50, forklifts: 1 });
  });
});

const person = (x: number, helmet: ItemStatus['state'] = 'present', conf = 0.9): PersonResult => {
  const it = (state: ItemStatus['state']): ItemStatus => ({ state, conf: 0.9, detIndex: null });
  return {
    detIndex: 0, box: [x, 100, x + 100, 400], conf,
    items: { helmet: it(helmet), goggles: it('present'), vest: it('present'), gloves: it('present'), boots: it('present') },
    status: helmet === 'missing' ? 'non_compliant' : 'compliant', missing: helmet === 'missing' ? ['helmet'] : [], unsureItems: [], matched: [],
  };
};

describe('tracking', () => {
  it('keeps ids stable while two workers walk', () => {
    const tr = new Tracker();
    for (let f = 0; f < 10; f++) {
      const seen = tr.update([person(100 + 15 * f), person(600 - 15 * f)]);
      expect(seen.map((t) => t.id).sort()).toEqual([1, 2]);
    }
    expect(tr.totalWorkers).toBe(2);
  });
  it('survives a short occlusion', () => {
    const tr = new Tracker();
    for (let f = 0; f < 4; f++) tr.update([person(100 + 10 * f)]);
    for (let f = 0; f < 3; f++) tr.update([]);
    expect(tr.update([person(170)]).map((t) => t.id)).toEqual([1]);
  });
  it('ignores single-frame flicker but raises a persistent violation', () => {
    const a = new Tracker();
    let t = a.update([person(100)])[0];
    for (const s of ['present', 'missing', 'present', 'present'] as const) t = a.update([person(100, s)])[0];
    expect(a.smoothedStatus(t, ['helmet'])).toBe('compliant');
    const b = new Tracker();
    for (const s of ['present', 'missing', 'missing', 'present', 'missing'] as const) t = b.update([person(100, s)])[0];
    expect(b.smoothedMissing(t, ['helmet'])).toEqual(['helmet']);
  });
  it('a low-confidence box cannot start a track', () => {
    expect(new Tracker().update([person(100, 'present', 0.3)])).toEqual([]);
  });
});

describe('zones', () => {
  it('point in polygon, including concave shapes', () => {
    const l: [number, number][] = [[0, 0], [10, 0], [10, 4], [4, 4], [4, 10], [0, 10]];
    expect(pointInPolygon([2, 8], l)).toBe(true);
    expect(pointInPolygon([8, 8], l)).toBe(false);
  });
  it('uses the feet, not the head', () => {
    const zone = { id: 1, name: 'z', points: [[0, 350], [400, 350], [400, 500], [0, 500]] as [number, number][] };
    const ev = zoneEvents([[100, 100, 200, 400], [100, 300, 200, 700]], [], [zone]);
    expect(ev[0].zones).toEqual([1]);
    expect(ev[1].zones).toEqual([]);
  });
  it('forklift proximity scales with person height', () => {
    const forklift: Box = [300, 100, 500, 400];
    expect(boxGap([180, 100, 280, 400], forklift)).toBe(20);
    expect(nearForklift([180, 100, 280, 400], forklift, 0.5)).toBe(true);
    expect(nearForklift([0, 100, 60, 280], forklift, 0.5)).toBe(false);
  });
});

describe('yolo decoding', () => {
  it('nms removes overlapping boxes of lower score', () => {
    expect(nms([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], [0.9, 0.8, 0.7], 0.5)).toEqual([0, 2]);
  });
  it('decodes a letterboxed prediction back to image pixels', () => {
    const lb = letterboxParams(1280, 720); // ratio 0.5, dy 140
    const n = 2;
    const nc = 12;
    const data = new Float32Array((4 + nc) * n);
    data[0] = 320; data[n] = 320; data[2 * n] = 100; data[3 * n] = 200; // one box, class 5 at 0.9
    data[(4 + 5) * n] = 0.9;
    const dets = decode(data, n, lb, 1280, 720);
    expect(dets).toHaveLength(1);
    expect(dets[0].cls).toBe(5);
    expect(dets[0].box.map(Math.round)).toEqual([540, 160, 740, 560]);
  });
});

describe('narrative verification', () => {
  const summary = summarise(matchPeople([
    { box: [100, 100, 300, 600], cls: 5, conf: 0.9 }, { box: [160, 100, 240, 170], cls: 6, conf: 0.8 }, { box: [120, 220, 280, 400], cls: 2, conf: 0.9 },
  ], ['helmet', 'vest']), []);
  it('keeps supported claims and corrects hallucinated counts', () => {
    const n = parseNarrative('Sure! {"workers": 3, "missing_helmet": 1, "missing_vest": 0, "missing_gloves": 0, "missing_boots": 0, "missing_goggles": 0, "forklift_present": true, "scene": "An outdoor building site in daylight."}');
    const v = verifyNarrative(n, summary, ['helmet', 'vest']);
    const verdict = Object.fromEntries(v.claims.map((c) => [c.key, c.verdict]));
    expect(verdict).toMatchObject({ workers: 'corrected', missing_helmet: 'supported', missing_vest: 'supported', forklift: 'corrected', scene: 'supported' });
    expect(v.unsupported).toBe(2);
    expect(v.text).toContain('1 worker visible.');
    expect(v.text).not.toContain('3 workers');
  });
  it('removes a scene sentence that smuggles in unchecked PPE claims', () => {
    const n = parseNarrative('{"workers": 1, "missing_helmet": 1, "missing_vest": 0, "forklift_present": false, "scene": "Two workers wearing helmets near a forklift."}');
    const v = verifyNarrative(n, summary, ['helmet', 'vest']);
    expect(v.claims.find((c) => c.key === 'scene')?.verdict).toBe('removed');
    expect(v.text).not.toContain('Two workers');
  });
  it('repairs truncated output and counts a list of invented workers by its length', () => {
    expect(closeJson('{"workers": [{"name": "JOHN", "age": "25"}, {"name": "A')).toBe('{"workers": [{"name": "JOHN", "age": "25"}, {"name": "A"}]}');
    const n = parseNarrative(' {\n "workers": [\n {"name": "JOHN", "age": "25"}, {"name":"A",');
    expect(n?.workers).toBe(2);
  });
  it('falls back to a detector-written summary when the model output is not JSON', () => {
    const v = verifyNarrative(parseNarrative('I cannot see the image.'), summary, ['helmet', 'vest']);
    expect(v.parsed).toBe(false);
    expect(v.text).toContain('1 worker without helmet.');
  });
});
