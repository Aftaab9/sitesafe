// A clip or live session: tracker + temporal smoothing + anonymous per-worker aggregates.
import { ITEM_LABEL, SITE_PROFILES, type Item, type ProfileId } from './classes';
import type { Analysis } from './analyze';
import type { OverlayPerson } from './draw';
import type { Box, PersonStatus, Summary } from './matching';
import { DEFAULT_TRACKER, Tracker, type TrackerConfig } from './tracking';

export interface WorkerAgg {
  id: number;
  frames: number;
  missing: Set<Item>;      // smoothed violations ever raised
  rawMissingFrames: number; // frames with an un-smoothed violation (for the smoothing comparison)
  compliantFrames: number;
  everViolation: boolean;
  zone: boolean;
  forklift: boolean;
}

export interface FrameRecord {
  t: number;
  people: OverlayPerson[];
  blur: Box[];
  activeZones: number[];
  persons: number;
  nonCompliant: number;     // smoothed
  rawNonCompliant: number;  // per-frame decision, no smoothing
  alerts: number;
  ms: number;
}

export class ClipSession {
  tracker: Tracker;
  workers = new Map<number, WorkerAgg>();
  frames: FrameRecord[] = [];
  forkliftsMax = 0;
  newViolation = false; // set when a worker's first smoothed violation appears in the latest step

  constructor(public profile: ProfileId, cfg: Partial<TrackerConfig> = {}) {
    this.tracker = new Tracker({ ...DEFAULT_TRACKER, ...cfg });
  }

  step(a: Analysis, t: number, keep = true): FrameRecord {
    const required = SITE_PROFILES[this.profile].required;
    const seen = this.tracker.update(a.people);
    this.newViolation = false;
    const people: OverlayPerson[] = [];
    let nonCompliant = 0;
    let alerts = 0;
    for (const tr of seen) {
      const idx = a.people.indexOf(tr.last!);
      const ev = a.events[idx];
      const status: PersonStatus = this.tracker.smoothedStatus(tr, required);
      const missing = this.tracker.smoothedMissing(tr, required);
      let w = this.workers.get(tr.id);
      if (!w) {
        w = { id: tr.id, frames: 0, missing: new Set(), rawMissingFrames: 0, compliantFrames: 0, everViolation: false, zone: false, forklift: false };
        this.workers.set(tr.id, w);
      }
      w.frames += 1;
      if (tr.last!.status === 'non_compliant') w.rawMissingFrames += 1;
      if (status === 'compliant') w.compliantFrames += 1;
      if (status === 'non_compliant') {
        nonCompliant += 1;
        if (!w.everViolation) { w.everViolation = true; this.newViolation = true; }
        missing.forEach((m) => w!.missing.add(m));
      }
      const alert = !!ev && (ev.zones.length > 0 || ev.nearForklift);
      if (ev?.zones.length) w.zone = true;
      if (ev?.nearForklift) w.forklift = true;
      if (alert) alerts += 1;
      people.push({ box: tr.box, status, label: `W${tr.id}`, missing: missing.map((m) => ITEM_LABEL[m].toLowerCase()), alert });
    }
    this.forkliftsMax = Math.max(this.forkliftsMax, a.summary.forklifts);
    const rec: FrameRecord = {
      t, people, blur: a.blur, activeZones: [...new Set(a.events.flatMap((e) => e.zones))],
      persons: people.length, nonCompliant, rawNonCompliant: a.summary.nonCompliant, alerts, ms: a.ms.total,
    };
    if (keep) this.frames.push(rec);
    return rec;
  }

  /** Workers that were tracked long enough to count (filters one-frame ghosts). */
  stableWorkers(): WorkerAgg[] {
    const min = Math.min(this.tracker.cfg.minHits, Math.max(1, this.frames.length));
    return [...this.workers.values()].filter((w) => w.frames >= min);
  }

  workerStatus(w: WorkerAgg): PersonStatus {
    if (w.everViolation) return 'non_compliant';
    return w.compliantFrames >= w.frames / 2 ? 'compliant' : 'unsure';
  }

  summary(): Summary {
    const ws = this.stableWorkers();
    const missingByItem = { helmet: 0, goggles: 0, vest: 0, gloves: 0, boots: 0 } as Record<Item, number>;
    let compliant = 0;
    let nonCompliant = 0;
    let unsure = 0;
    for (const w of ws) {
      const s = this.workerStatus(w);
      if (s === 'compliant') compliant += 1; else if (s === 'non_compliant') nonCompliant += 1; else unsure += 1;
      w.missing.forEach((m) => { missingByItem[m] += 1; });
    }
    const decided = compliant + nonCompliant;
    return {
      persons: ws.length, compliant, nonCompliant, unsure,
      compliancePct: decided ? Math.round((1000 * compliant) / decided) / 10 : null,
      missingByItem, forklifts: this.forkliftsMax,
    };
  }

  zoneEvents() { return this.stableWorkers().filter((w) => w.zone).length; }
  forkliftEvents() { return this.stableWorkers().filter((w) => w.forklift).length; }
}
