// Optional second opinion from a small vision-language model running in Ollama on THIS device.
// The website itself has no server: the browser talks straight to http://localhost:11434, so the
// crops never leave the computer. Visitors without Ollama simply see that it is not available.
//
// One-time setup on the computer (so Ollama accepts requests from the website):
//   Windows:  setx OLLAMA_ORIGINS "https://sitesafe-phi.vercel.app,http://localhost:5173,http://localhost:4173"
//             then quit Ollama from the tray and start it again.
//   ollama pull lfm3-omni

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
  const c = document.createElement('canvas');
  c.width = Math.max(1, Math.round(sw)); c.height = Math.max(1, Math.round(sh));
  c.getContext('2d')!.drawImage(source, sx, sy, sw, sh, 0, 0, c.width, c.height);
  return c.toDataURL('image/jpeg', 0.92).split(',')[1];
}

/** Ask the local model about one worker (a crop of the original frame, held in memory only). */
export async function askWorker(model: string, source: HTMLCanvasElement, box: number[]): Promise<VlmAnswer> {
  const t0 = performance.now();
  const res = await fetch(`${BASE}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, prompt: PROMPT, images: [cropJpeg(source, box)], stream: false, format: SCHEMA, options: { temperature: 0 } }),
  });
  if (!res.ok) throw new Error(`Ollama answered ${res.status}`);
  let a: Partial<VlmAnswer> = {};
  try { a = JSON.parse((await res.json()).response); } catch { /* treated as unsure */ }
  const norm = (v: unknown): Answer => (v === 'yes' || v === 'no' ? v : 'unsure');
  return { hard_hat: norm(a.hard_hat), hi_vis: norm(a.hi_vis), ms: performance.now() - t0 };
}
