// YOLO11 pre/post-processing. Mirrors sitesafe/detector.py.
import { NUM_CLASSES } from './classes';
import type { Box, Det } from './matching';

export const INPUT = 640;

export interface Letterbox { ratio: number; dx: number; dy: number; w: number; h: number }

export function letterboxParams(w: number, h: number, size = INPUT): Letterbox {
  const ratio = Math.min(size / h, size / w);
  const nw = Math.round(w * ratio);
  const nh = Math.round(h * ratio);
  return { ratio, dx: Math.floor((size - nw) / 2), dy: Math.floor((size - nh) / 2), w: nw, h: nh };
}

/** RGBA pixels of the letterboxed size x size canvas -> planar float32 RGB in [0, 1]. */
export function toTensorData(rgba: Uint8ClampedArray, size = INPUT): Float32Array {
  const n = size * size;
  const out = new Float32Array(3 * n);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    out[i] = rgba[p] / 255;
    out[n + i] = rgba[p + 1] / 255;
    out[2 * n + i] = rgba[p + 2] / 255;
  }
  return out;
}

export function nms(boxes: Box[], scores: number[], thr: number): number[] {
  const order = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a]);
  const keep: number[] = [];
  const dead = new Uint8Array(boxes.length);
  for (const i of order) {
    if (dead[i]) continue;
    keep.push(i);
    const a = boxes[i];
    const areaA = (a[2] - a[0]) * (a[3] - a[1]);
    for (const j of order) {
      if (dead[j] || j === i) continue;
      const b = boxes[j];
      const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
      const h = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
      if (w <= 0 || h <= 0) continue;
      const inter = w * h;
      if (inter / (areaA + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9) > thr) dead[j] = 1;
    }
    dead[i] = 1;
  }
  return keep;
}

/** Decode a raw head output of shape (1, 4 + nc, N), stored channel-major, into original-pixel boxes. */
export function decode(
  data: Float32Array, n: number, lb: Letterbox, origW: number, origH: number,
  confThr = 0.25, iouThr = 0.5, nc = NUM_CLASSES,
): Det[] {
  const perClass: { boxes: Box[]; scores: number[] }[] = Array.from({ length: nc }, () => ({ boxes: [], scores: [] }));
  for (let i = 0; i < n; i++) {
    let best = 0;
    let cls = -1;
    for (let c = 0; c < nc; c++) {
      const s = data[(4 + c) * n + i];
      if (s > best) { best = s; cls = c; }
    }
    if (best < confThr) continue;
    const cx = data[i];
    const cy = data[n + i];
    const w = data[2 * n + i];
    const h = data[3 * n + i];
    const clampX = (v: number) => Math.min(origW, Math.max(0, (v - lb.dx) / lb.ratio));
    const clampY = (v: number) => Math.min(origH, Math.max(0, (v - lb.dy) / lb.ratio));
    perClass[cls].boxes.push([clampX(cx - w / 2), clampY(cy - h / 2), clampX(cx + w / 2), clampY(cy + h / 2)]);
    perClass[cls].scores.push(best);
  }
  const dets: Det[] = [];
  perClass.forEach((pc, cls) => {
    for (const k of nms(pc.boxes, pc.scores, iouThr)) dets.push({ box: pc.boxes[k], cls, conf: pc.scores[k] });
  });
  dets.sort((a, b) => b.conf - a.conf);
  return dets;
}
