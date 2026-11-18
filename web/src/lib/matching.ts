// Person-PPE matching. Line-for-line port of sitesafe/matching.py; both run the same fixture tests.
import { FORKLIFT, FORKLIFT_MIN_CONF, ITEM_BAND, ITEM_OF_CLASS, PERSON, PPE_ITEMS, type Item } from './classes';

export type Box = [number, number, number, number]; // x1, y1, x2, y2
export interface Det { box: Box; cls: number; conf: number }
export type ItemState = 'present' | 'missing' | 'unsure' | 'not_visible';
export interface ItemStatus { state: ItemState; conf: number; detIndex: number | null }
export type PersonStatus = 'compliant' | 'non_compliant' | 'unsure';
export interface PersonResult {
  detIndex: number;
  box: Box;
  conf: number;
  items: Record<Item, ItemStatus>;
  status: PersonStatus;
  missing: Item[];
  unsureItems: Item[];
  matched: number[];
}
export interface MatchConfig {
  ioaMin: number;
  bandTol: number;
  confUnsure: number;
  marginUnsure: number;
  personUnsure: number;
  classThresholds?: Record<number, number>;
}
export const DEFAULT_MATCH: MatchConfig = {
  ioaMin: 0.5, bandTol: 0.12, confUnsure: 0.45, marginUnsure: 0.1, personUnsure: 0.4,
};

export const area = (b: Box) => Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
export function intersection(a: Box, b: Box) {
  const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
  const h = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  return w > 0 && h > 0 ? w * h : 0;
}
export function iou(a: Box, b: Box) {
  const inter = intersection(a, b);
  const union = area(a) + area(b) - inter;
  return union > 0 ? inter / union : 0;
}
/** Intersection over the PPE box's own area: PPE boxes are small, so plain IoU is useless here. */
export function ioa(ppe: Box, person: Box) {
  const a = area(ppe);
  return a > 0 ? intersection(ppe, person) / a : 0;
}

export function inBand(item: Item, ppe: Box, person: Box, tol: number) {
  const pw = person[2] - person[0];
  const ph = person[3] - person[1];
  if (ph <= 0 || pw > ph) return true; // not upright: head is not necessarily at the top
  const cy = ((ppe[1] + ppe[3]) / 2 - person[1]) / ph;
  const [lo, hi] = ITEM_BAND[item];
  return lo - tol <= cy && cy <= hi + tol;
}

export function assignPpe(dets: Det[], cfg: MatchConfig = DEFAULT_MATCH): Map<number, number[]> {
  const persons: number[] = [];
  dets.forEach((d, i) => { if (d.cls === PERSON) persons.push(i); });
  const out = new Map<number, number[]>(persons.map((p) => [p, []]));
  dets.forEach((d, j) => {
    const info = ITEM_OF_CLASS[d.cls];
    if (!info) return;
    let best = -1;
    let bestScore = 0;
    for (const p of persons) {
      const pb = dets[p].box;
      let score = ioa(d.box, pb);
      if (score < cfg.ioaMin || !inBand(info.item, d.box, pb, cfg.bandTol)) continue;
      const pcx = (pb[0] + pb[2]) / 2;
      const dcx = (d.box[0] + d.box[2]) / 2;
      const pw = Math.max(pb[2] - pb[0], 1e-6);
      score = score - 0.25 * Math.min(Math.abs(dcx - pcx) / pw, 1) - 1e-9 * area(pb);
      if (best < 0 || score > bestScore) { best = p; bestScore = score; }
    }
    if (best >= 0) out.get(best)!.push(j);
  });
  return out;
}

type Cand = { conf: number; idx: number; cls: number } | null;
const threshold = (cls: number, cfg: MatchConfig) => cfg.classThresholds?.[cls] ?? cfg.confUnsure;

export function decideItem(pos: Cand, neg: Cand, cfg: MatchConfig): ItemStatus {
  if (!pos && !neg) return { state: 'not_visible', conf: 0, detIndex: null };
  if (pos && neg) {
    const posWins = pos.conf >= neg.conf;
    const win = posWins ? pos : neg;
    const lose = posWins ? neg : pos;
    let state: ItemState = posWins ? 'present' : 'missing';
    if (win.conf - lose.conf < cfg.marginUnsure || win.conf < threshold(win.cls, cfg)) state = 'unsure';
    return { state, conf: win.conf, detIndex: win.idx };
  }
  const win = (pos ?? neg)!;
  let state: ItemState = pos ? 'present' : 'missing';
  if (win.conf < threshold(win.cls, cfg)) state = 'unsure';
  return { state, conf: win.conf, detIndex: win.idx };
}

export function matchPeople(dets: Det[], required: Item[], cfg: MatchConfig = DEFAULT_MATCH): PersonResult[] {
  const assigned = assignPpe(dets, cfg);
  const results: PersonResult[] = [];
  for (const [p, idxs] of assigned) {
    const items = {} as Record<Item, ItemStatus>;
    for (const item of PPE_ITEMS) {
      let pos: Cand = null;
      let neg: Cand = null;
      for (const j of idxs) {
        const info = ITEM_OF_CLASS[dets[j].cls];
        if (info.item !== item) continue;
        const cand = { conf: dets[j].conf, idx: j, cls: dets[j].cls };
        if (info.worn && (!pos || cand.conf > pos.conf)) pos = cand;
        if (!info.worn && (!neg || cand.conf > neg.conf)) neg = cand;
      }
      items[item] = decideItem(pos, neg, cfg);
    }
    const missing = required.filter((i) => items[i].state === 'missing');
    const unsureItems = [
      ...required.filter((i) => items[i].state === 'unsure'),
      ...required.filter((i) => items[i].state === 'not_visible'),
    ];
    const conf = dets[p].conf;
    let status: PersonStatus;
    if (missing.length && conf >= cfg.personUnsure) status = 'non_compliant';
    // Green is never shown without positive evidence for every required item.
    else if (missing.length || unsureItems.length || conf < cfg.personUnsure) status = 'unsure';
    else status = 'compliant';
    results.push({ detIndex: p, box: [...dets[p].box] as Box, conf, items, status, missing, unsureItems, matched: [...idxs] });
  }
  results.sort((a, b) => a.box[0] - b.box[0] || a.box[1] - b.box[1]);
  return results;
}

export interface Summary {
  persons: number;
  compliant: number;
  nonCompliant: number;
  unsure: number;
  compliancePct: number | null;
  missingByItem: Record<Item, number>;
  forklifts: number;
}

/** Aggregate, anonymous statistics. No identities, only counts. */
export function summarise(people: PersonResult[], dets: Det[]): Summary {
  const compliant = people.filter((p) => p.status === 'compliant').length;
  const nonCompliant = people.filter((p) => p.status === 'non_compliant').length;
  const unsure = people.filter((p) => p.status === 'unsure').length;
  const decided = compliant + nonCompliant;
  const missingByItem = { helmet: 0, goggles: 0, vest: 0, gloves: 0, boots: 0 } as Record<Item, number>;
  for (const p of people) for (const i of p.missing) missingByItem[i] += 1;
  return {
    persons: people.length,
    compliant,
    nonCompliant,
    unsure,
    compliancePct: decided ? Math.round((1000 * compliant) / decided) / 10 : null,
    missingByItem,
    forklifts: dets.filter((d) => d.cls === FORKLIFT && d.conf >= FORKLIFT_MIN_CONF).length,
  };
}
