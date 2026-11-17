// Turns raw detections into the decision the safety officer sees.
import { FORKLIFT, SITE_PROFILES, type ProfileId } from './classes';
import type { DetectResult } from './engine';
import { matchPeople, summarise, type Box, type Det, type MatchConfig, type PersonResult, type Summary, DEFAULT_MATCH } from './matching';
import { privacyBoxes } from './yunet';
import { zoneEvents, type Zone, type ZoneEvent } from './zones';
import thresholds from './thresholds.json';

export interface Analysis {
  w: number;
  h: number;
  dets: Det[];
  people: PersonResult[];
  summary: Summary;
  faces: Box[];
  blur: Box[];
  events: ZoneEvent[];
  zoneAlerts: number;
  forkliftAlerts: number;
  ms: DetectResult['ms'];
}

/** Per-class "unsure" thresholds chosen on the validation split by scripts/calibrate.py. */
export const MATCH_CONFIG: MatchConfig = {
  ...DEFAULT_MATCH,
  classThresholds: Object.fromEntries(Object.entries(thresholds.class_thresholds).map(([k, v]) => [Number(k), v as number])),
};

export function analyse(
  res: DetectResult, profile: ProfileId, zones: Zone[], opts: { strictBlur: boolean; forkliftDistance: number },
): Analysis {
  const people = matchPeople(res.dets, SITE_PROFILES[profile].required, MATCH_CONFIG);
  const forklifts = res.dets.filter((d) => d.cls === FORKLIFT).map((d) => d.box);
  const events = zoneEvents(people.map((p) => p.box), forklifts, zones, opts.forkliftDistance);
  return {
    w: res.w,
    h: res.h,
    dets: res.dets,
    people,
    summary: summarise(people, res.dets),
    faces: res.faces,
    blur: privacyBoxes(res.faces, people.map((p) => p.box), opts.strictBlur),
    events,
    zoneAlerts: events.filter((e) => e.zones.length > 0).length,
    forkliftAlerts: events.filter((e) => e.nearForklift).length,
    ms: res.ms,
  };
}
