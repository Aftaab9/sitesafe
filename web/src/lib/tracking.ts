// ByteTrack-style IoU tracker with temporal smoothing. Port of sitesafe/tracking.py.
import { PPE_ITEMS, type Item } from './classes';
import { iou, type Box, type ItemState, type PersonResult, type PersonStatus } from './matching';

export interface Track {
  id: number;
  box: Box;
  vel: Box;
  hits: number;
  misses: number;
  history: Record<Item, ItemState[]>;
  statusHistory: PersonStatus[];
  last: PersonResult | null;
}

export interface TrackerConfig {
  highConf: number;
  iouHigh: number;
  iouLow: number;
  maxMisses: number;
  window: number; // n
  minHits: number; // k
  velAlpha: number;
}
export const DEFAULT_TRACKER: TrackerConfig = {
  highConf: 0.5, iouHigh: 0.3, iouLow: 0.5, maxMisses: 8, window: 5, minHits: 3, velAlpha: 0.6,
};

const predict = (t: Track): Box => t.box.map((b, i) => b + t.vel[i]) as Box;

export class Tracker {
  tracks: Track[] = [];
  private nextId = 1;
  constructor(public cfg: TrackerConfig = DEFAULT_TRACKER) {}

  private associate(tracks: Track[], people: PersonResult[], thr: number) {
    const cands: [number, number, number][] = [];
    tracks.forEach((t, ti) => {
      const pred = predict(t);
      people.forEach((p, pi) => {
        const v = iou(pred, p.box);
        if (v >= thr) cands.push([v, ti, pi]);
      });
    });
    cands.sort((a, b) => b[0] - a[0] || b[1] - a[1] || b[2] - a[2]);
    const usedT = new Set<number>();
    const usedP = new Set<number>();
    const pairs: [number, number][] = [];
    for (const [, ti, pi] of cands) {
      if (usedT.has(ti) || usedP.has(pi)) continue;
      usedT.add(ti); usedP.add(pi); pairs.push([ti, pi]);
    }
    return {
      pairs,
      restTracks: tracks.filter((_, i) => !usedT.has(i)),
      restPeople: people.filter((_, i) => !usedP.has(i)),
    };
  }

  private updateTrack(t: Track, p: PersonResult) {
    const a = this.cfg.velAlpha;
    t.vel = t.vel.map((ov, i) => a * (p.box[i] - t.box[i]) + (1 - a) * ov) as Box;
    t.box = [...p.box] as Box;
    t.hits += 1;
    t.misses = 0;
    t.last = p;
    for (const item of PPE_ITEMS) {
      const h = t.history[item];
      h.push(p.items[item]?.state ?? 'not_visible');
      if (h.length > this.cfg.window) h.shift();
    }
    t.statusHistory.push(p.status);
    if (t.statusHistory.length > this.cfg.window) t.statusHistory.shift();
  }

  /** Feed the matched people of one sampled frame. Returns the tracks seen in this frame. */
  update(people: PersonResult[]): Track[] {
    const cfg = this.cfg;
    const high = people.filter((p) => p.conf >= cfg.highConf);
    const low = people.filter((p) => p.conf < cfg.highConf);
    const seen: Track[] = [];

    const s1 = this.associate(this.tracks, high, cfg.iouHigh);
    for (const [ti, pi] of s1.pairs) { this.updateTrack(this.tracks[ti], high[pi]); seen.push(this.tracks[ti]); }

    const s2 = this.associate(s1.restTracks, low, cfg.iouLow);
    const matched2 = new Set<Track>();
    for (const [ti, pi] of s2.pairs) {
      this.updateTrack(s1.restTracks[ti], low[pi]);
      seen.push(s1.restTracks[ti]);
      matched2.add(s1.restTracks[ti]);
    }
    for (const t of s1.restTracks) {
      if (!matched2.has(t)) { t.misses += 1; t.box = predict(t); }
    }

    for (const p of s1.restPeople) {
      const t: Track = {
        id: this.nextId++, box: [...p.box] as Box, vel: [0, 0, 0, 0], hits: 0, misses: 0,
        history: { helmet: [], goggles: [], vest: [], gloves: [], boots: [] }, statusHistory: [], last: null,
      };
      this.updateTrack(t, p);
      this.tracks.push(t);
      seen.push(t);
    }
    this.tracks = this.tracks.filter((t) => t.misses <= cfg.maxMisses);
    return seen;
  }

  smoothedMissing(t: Track, required: Item[]): Item[] {
    return required.filter((item) => t.history[item].filter((s) => s === 'missing').length >= this.cfg.minHits);
  }

  smoothedStatus(t: Track, required: Item[]): PersonStatus {
    if (this.smoothedMissing(t, required).length) return 'non_compliant';
    const h = t.statusHistory;
    if (!h.length) return 'unsure';
    const ok = h.filter((s) => s === 'compliant').length;
    return ok >= Math.max(1, Math.floor((h.length + 1) / 2)) ? 'compliant' : 'unsure';
  }

  get totalWorkers() { return this.nextId - 1; }
}
