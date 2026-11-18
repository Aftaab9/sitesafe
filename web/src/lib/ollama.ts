// Optional second opinion from a small vision-language model running in Ollama on THIS device.
// The website itself has no server: the browser talks straight to http://localhost:11434, so the
// crops never leave the computer. Visitors without Ollama simply see that it is not available.
//
// One-time setup on the computer (so Ollama accepts requests from the website):
//   Windows:  setx OLLAMA_ORIGINS "https://sitesafe-phi.vercel.app,http://localhost:5173,http://localhost:4173"
//             then quit Ollama from the tray and start it again.
//   ollama pull lfm3-omni

import { PERSON } from './classes';
import type { Det } from './matching';

const BASE = 'http://localhost:11434';
const PREFERRED = ['lfm3-omni', 'lfm'];

export const SCHEMA = {
  type: 'object',
  properties: { hard_hat: { enum: ['yes', 'no', 'unsure'] }, hi_vis: { enum: ['yes', 'no', 'unsure'] } },
  required: ['hard_hat', 'hi_vis'],
};
// Same prompt as scripts/vlm_baseline.py, so the app matches the offline comparison.
export const PROMPT =
  'The image shows one worker on a construction site. Answer about this person only. ' +
  'hard_hat: is the person wearing a hard hat (safety helmet) on their head? ' +
  'hi_vis: is the person wearing a high-visibility vest or jacket (fluorescent yellow, green or orange, ' +
  'often with reflective stripes)? Answer yes or no; answer unsure only if that part of the body cannot be seen.';

export type Answer = 'yes' | 'no' | 'unsure';
export interface VlmAnswer { hard_hat: Answer; hi_vis: Answer; ms: number }

/** The vision model to use, or null when Ollama is not running / not reachable from this page. */
export async function findOllamaModel(): Promise<string | null> {
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 1500);
    const res = await fetch(`${BASE}/api/tags`, { signal: ctrl.signal });
    clearTimeout(t);
    if (!res.ok) return null;
    const names: string[] = ((await res.json()).models ?? []).map((m: { name: string }) => m.name);
    for (const p of PREFERRED) {
      const hit = names.find((n) => n.toLowerCase().startsWith(p));
      if (hit) return hit;
    }
    return null;
  } catch {
    return null;
  }
}

function cropJpeg(source: HTMLCanvasElement, box: number[]): string {
  const [x1, y1, x2, y2] = box;
  const mw = (x2 - x1) * 0.15, mh = (y2 - y1) * 0.1;
  const sx = Math.max(0, x1 - mw), sy = Math.max(0, y1 - mh);
  const sw = Math.min(source.width, x2 + mw) - sx, sh = Math.min(source.height, y2 + mh) - sy;
  const f = Math.min(1, 768 / Math.max(sw, sh)); // big crops cost many image tokens: 12 s instead of 1.3 s
  const c = document.createElement('canvas');
  c.width = Math.max(1, Math.round(sw * f)); c.height = Math.max(1, Math.round(sh * f));
  c.getContext('2d')!.drawImage(source, sx, sy, sw, sh, 0, 0, c.width, c.height);
  return c.toDataURL('image/jpeg', 0.92).split(',')[1];
}

/** Ask the local model about one worker (a crop of the original frame, held in memory only). */
export async function askWorker(model: string, source: HTMLCanvasElement, box: number[]): Promise<VlmAnswer> {
  const t0 = performance.now();
  const res = await fetch(`${BASE}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, prompt: PROMPT, images: [cropJpeg(source, box)], stream: false, format: SCHEMA, options: { temperature: 0, num_ctx: 4096 } }),
  });
  if (!res.ok) throw new Error(`Ollama answered ${res.status}`);
  let a: Partial<VlmAnswer> = {};
  try { a = JSON.parse((await res.json()).response); } catch { /* treated as unsure */ }
  const norm = (v: unknown): Answer => (v === 'yes' || v === 'no' ? v : 'unsure');
  return { hard_hat: norm(a.hard_hat), hi_vis: norm(a.hi_vis), ms: performance.now() - t0 };
}

// ---- Whole-photo mode: LFM3-Omni finds the workers itself (same as sitesafe/lfm.py) ----
const DETECT_SCHEMA = {
  type: 'object',
  properties: {
    workers: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          box: { type: 'array', items: { type: 'integer' }, minItems: 4, maxItems: 4 },
          hard_hat: { enum: ['yes', 'no', 'unsure'] }, hi_vis: { enum: ['yes', 'no', 'unsure'] },
        },
        required: ['box', 'hard_hat', 'hi_vis'],
      },
    },
  },
  required: ['workers'],
};
const DETECT_PROMPT =
  'Find every person in this construction-site photo. For each person give a bounding box [x1, y1, x2, y2] ' +
  'as numbers from 0 to 1000 relative to the image width and height, and say whether they wear a hard hat ' +
  'and a high-visibility vest or jacket (yes, no or unsure).';

/**
 * Ask LFM3-Omni for every worker in the photo and turn its answers into ordinary detections
 * (Person, helmet / no_helmet, vest / no_vest in the body bands the matching code expects), so the
 * rest of the app (matching, profiles, drawing, reports) works unchanged. Gloves, boots and goggles are not asked.
 */
export async function detectWorkers(model: string, source: HTMLCanvasElement): Promise<{ dets: Det[]; ms: number }> {
  const t0 = performance.now();
  const W = source.width, H = source.height;
  const f = Math.min(1, 1024 / Math.max(W, H));
  const c = document.createElement('canvas');
  c.width = Math.round(W * f); c.height = Math.round(H * f);
  c.getContext('2d')!.drawImage(source, 0, 0, c.width, c.height);
  const res = await fetch(`${BASE}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model, prompt: DETECT_PROMPT, images: [c.toDataURL('image/jpeg', 0.9).split(',')[1]], stream: false,
      format: DETECT_SCHEMA, options: { temperature: 0, num_ctx: 4096 },
    }),
  });
  if (!res.ok) throw new Error(`Ollama answered ${res.status}`);
  let workers: { box?: number[]; hard_hat?: string; hi_vis?: string }[] = [];
  try { workers = JSON.parse((await res.json()).response).workers ?? []; } catch { /* no workers */ }
  const dets: Det[] = [];
  const conf = 0.9;
  for (const wk of workers) {
    if (!wk.box || wk.box.length !== 4) continue;
    const [a, b, cc, d] = wk.box.map((v, i) => (Math.min(1000, Math.max(0, v)) / 1000) * (i % 2 === 0 ? W : H));
    const x1 = Math.min(a, cc), x2 = Math.max(a, cc), y1 = Math.min(b, d), y2 = Math.max(b, d);
    if (x2 - x1 < 4 || y2 - y1 < 4) continue;
    const pw = x2 - x1, ph = y2 - y1;
    dets.push({ box: [x1, y1, x2, y2], cls: PERSON, conf });
    if (wk.hard_hat === 'yes' || wk.hard_hat === 'no')
      dets.push({ box: [x1 + 0.25 * pw, y1, x2 - 0.25 * pw, y1 + 0.15 * ph], cls: wk.hard_hat === 'yes' ? 0 : 6, conf });
    if (wk.hi_vis === 'yes' || wk.hi_vis === 'no')
      dets.push({ box: [x1 + 0.15 * pw, y1 + 0.25 * ph, x2 - 0.15 * pw, y1 + 0.55 * ph], cls: wk.hi_vis === 'yes' ? 2 : 11, conf });
  }
  return { dets, ms: performance.now() - t0 };
}
