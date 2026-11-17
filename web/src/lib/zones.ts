// Danger zones and forklift proximity. Port of sitesafe/zones.py.
import type { Box } from './matching';

export type Point = [number, number];
export interface Zone { id: number; name: string; points: Point[] }

export const footPoint = (b: Box): Point => [(b[0] + b[2]) / 2, b[3]];

export function pointInPolygon(pt: Point, poly: Point[]): boolean {
  if (poly.length < 3) return false;
  const [x, y] = pt;
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > y !== yj > y) {
      const xCross = ((xj - xi) * (y - yi)) / (yj - yi) + xi;
      if (x < xCross) inside = !inside;
    }
  }
  return inside;
}

export function boxGap(a: Box, b: Box): number {
  const dx = Math.max(a[0] - b[2], b[0] - a[2], 0);
  const dy = Math.max(a[1] - b[3], b[1] - a[3], 0);
  return Math.hypot(dx, dy);
}

/** Distance is measured in units of the person's own height, so it is roughly scale-invariant. */
export function nearForklift(person: Box, forklift: Box, maxHeights = 0.5): boolean {
  const h = Math.max(person[3] - person[1], 1e-6);
  return boxGap(person, forklift) / h <= maxHeights;
}

export interface ZoneEvent { zones: number[]; nearForklift: boolean }

export function zoneEvents(persons: Box[], forklifts: Box[], zones: Zone[], maxHeights = 0.5): ZoneEvent[] {
  return persons.map((box) => {
    const fp = footPoint(box);
    return {
      zones: zones.filter((z) => pointInPolygon(fp, z.points)).map((z) => z.id),
      nearForklift: forklifts.some((f) => nearForklift(box, f, maxHeights)),
    };
  });
}
