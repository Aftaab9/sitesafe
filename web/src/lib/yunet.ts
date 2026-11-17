// YuNet (OpenCV Zoo, MIT) face detector post-processing for ONNX Runtime Web.
// The 2023mar model takes a fixed 640x640 BGR image with raw 0..255 values and returns, per stride
// (8, 16, 32), a class score, an objectness score and a box regression for every grid cell.
import { nms } from './yolo';
import type { Box } from './matching';

export const FACE_INPUT = 640;
const STRIDES = [8, 16, 32];

/** RGBA letterboxed pixels -> planar float32 BGR, 0..255. */
export function toFaceTensor(rgba: Uint8ClampedArray, size = FACE_INPUT): Float32Array {
  const n = size * size;
  const out = new Float32Array(3 * n);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    out[i] = rgba[p + 2];
    out[n + i] = rgba[p + 1];
    out[2 * n + i] = rgba[p];
  }
  return out;
}

export function decodeFaces(
  outputs: Record<string, Float32Array>, scoreThr = 0.6, nmsThr = 0.3, size = FACE_INPUT,
): { box: Box; score: number }[] {
  const boxes: Box[] = [];
  const scores: number[] = [];
  for (const s of STRIDES) {
    const cls = outputs[`cls_${s}`];
    const obj = outputs[`obj_${s}`];
    const bbox = outputs[`bbox_${s}`];
    const cols = size / s;
    const rows = size / s;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const i = r * cols + c;
        const score = Math.sqrt(Math.min(1, Math.max(0, cls[i])) * Math.min(1, Math.max(0, obj[i])));
        if (score < scoreThr) continue;
        const cx = (c + bbox[i * 4]) * s;
        const cy = (r + bbox[i * 4 + 1]) * s;
        const w = Math.exp(bbox[i * 4 + 2]) * s;
        const h = Math.exp(bbox[i * 4 + 3]) * s;
        boxes.push([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]);
        scores.push(score);
      }
    }
  }
  return nms(boxes, scores, nmsThr).map((k) => ({ box: boxes[k], score: scores[k] }));
}

/**
 * Where the face normally is in a person box: just below the top, so a hard hat stays visible.
 * Used to blur faces that the face detector missed (strict mode). Mirrors sitesafe/privacy.py.
 */
export function headRegion(p: Box): Box {
  const w = p[2] - p[0];
  const h = p[3] - p[1];
  const hh = Math.min(0.14 * h, 0.7 * w);
  const cx = (p[0] + p[2]) / 2;
  const hw = Math.min(w, Math.max(0.34 * w, 0.9 * hh)) / 2;
  const top = p[1] + 0.06 * h;
  return [cx - hw, top, cx + hw, top + hh];
}

export function privacyBoxes(faces: Box[], persons: Box[], strict: boolean): Box[] {
  const out = faces.map((f) => [...f] as Box);
  if (strict) {
    for (const p of persons) {
      const hr = headRegion(p);
      const covered = faces.some((f) => f[0] < hr[2] && f[2] > hr[0] && f[1] < hr[3] && f[3] > hr[1]);
      if (!covered) out.push(hr);
    }
  }
  return out;
}
