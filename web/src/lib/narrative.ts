// Generative AI part 2: an in-browser vision-language model writes a short incident description,
// and every claim it makes is verified against the detector ("verify the assistant").
//
// The verification half of this file is pure and unit-tested (narrative.test.ts). The same rules
// are implemented in scripts/vlm_hallucination.py, which measures the hallucination rate before
// and after verification on the held-out test split.
import { ITEM_LABEL, PPE_ITEMS, type Item } from './classes';
import type { Summary } from './matching';

export const VLM_MODEL = 'HuggingFaceTB/SmolVLM-256M-Instruct';

/** The fixed JSON the model is asked to fill in. */
export interface NarrativeJson {
  workers: number;
  missing_helmet: number;
  missing_vest: number;
  missing_gloves: number;
  missing_boots: number;
  missing_goggles: number;
  forklift_present: boolean;
  scene: string;
}

export const NARRATIVE_PROMPT =
  "Count the people in this construction site photo and check their safety equipment. Reply with JSON only, numbers not lists, no names. Example of the format: {\"workers\": 3, \"missing_helmet\": 1, \"missing_vest\": 0, \"missing_gloves\": 2, \"missing_boots\": 0, \"missing_goggles\": 0, \"forklift_present\": false, \"scene\": \"Outdoor building site in daylight.\"}";

/** Close a truncated JSON text: finish an open string, drop a dangling comma or key, close brackets in order. */
export function closeJson(t: string): string {
  const stack: string[] = [];
  let inStr = false;
  let esc = false;
  for (const ch of t) {
    if (inStr) {
      if (esc) esc = false;
      else if (ch === '\\') esc = true;
      else if (ch === '"') inStr = false;
    } else if (ch === '"') inStr = true;
    else if (ch === '{' || ch === '[') stack.push(ch === '{' ? '}' : ']');
    else if ((ch === '}' || ch === ']') && stack.length) stack.pop();
  }
  let out = inStr ? t + '"' : t;
  out = out.trimEnd().replace(/[,:]\s*$/, '').replace(/,\s*"[^"]*"\s*$/, '');
  return out + stack.reverse().join('');
}

export function parseNarrative(raw: string): NarrativeJson | null {
  // Lenient: small models often truncate or wrap values in lists. A list of workers counts as its length.
  const m = raw.match(/\{[\s\S]*/);
  if (!m) return null;
  const t = m[0];
  const cands = [...(t.includes('}') ? [t.slice(0, t.lastIndexOf('}') + 1)] : []), closeJson(t)];
  let obj: Record<string, unknown> | null = null;
  for (const c of cands) {
    for (const v of [c, c.replace(/,\s*([}\]])/g, '$1').replace(/'/g, '"')]) {
      try { const o = JSON.parse(v); if (o && typeof o === 'object' && !Array.isArray(o)) { obj = o; break; } } catch { /* next */ }
    }
    if (obj) break;
  }
  if (!obj) return null;
  const num = (v: unknown) => {
    if (Array.isArray(v)) return v.length;
    const n = typeof v === 'number' ? v : parseInt(String(v), 10);
    return Number.isFinite(n) && n >= 0 ? Math.round(n) : 0;
  };
  const bool = (v: unknown) => v === true || String(v).toLowerCase() === 'true' || String(v).toLowerCase() === 'yes';
  return {
    workers: num(obj.workers),
    missing_helmet: num(obj.missing_helmet),
    missing_vest: num(obj.missing_vest),
    missing_gloves: num(obj.missing_gloves),
    missing_boots: num(obj.missing_boots),
    missing_goggles: num(obj.missing_goggles),
    forklift_present: bool(obj.forklift_present),
    scene: typeof obj.scene === 'string' ? obj.scene.slice(0, 220) : '',
  };
}

export type Verdict = 'supported' | 'corrected' | 'removed';
export interface Claim { key: string; original: string; text: string; verdict: Verdict }
export interface VerifiedNarrative {
  claims: Claim[];
  total: number;
  unsupported: number;      // claims that disagreed with the detector (hallucinations caught)
  text: string;             // final, verified summary
  parsed: boolean;
}

const plural = (n: number, s: string) => `${n} ${s}${n === 1 ? '' : 's'}`;

// Words in the free-text scene sentence that assert things only the detector may assert.
const SCENE_RISKY = /\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|helmets?|hard ?hats?|vests?|gloves?|boots?|goggles?|glasses|forklifts?|wearing|without|no ppe|all|every|none|nobody)\b/i;

/**
 * Cross-check each claim against the detector's summary.
 * - A numeric or boolean claim that matches the detector is kept ("supported").
 * - One that disagrees is replaced by the detector's value ("corrected").
 * - A free-text scene sentence that restates counts or PPE facts cannot be checked word by word,
 *   so it is dropped ("removed"); a purely descriptive one is kept but never treated as evidence.
 */
export function verifyNarrative(n: NarrativeJson | null, s: Summary, required: Item[]): VerifiedNarrative {
  const claims: Claim[] = [];
  const truth = (item: Item) => s.missingByItem[item];
  const push = (key: string, original: string, truthText: string, ok: boolean) =>
    claims.push({ key, original, text: ok ? original : truthText, verdict: ok ? 'supported' : 'corrected' });

  if (!n) {
    // Unparseable model output: fall back to a purely detector-written summary.
    return { claims: [], total: 0, unsupported: 0, parsed: false, text: detectorSummary(s, required) };
  }
  push('workers', `${plural(n.workers, 'worker')} visible.`, `${plural(s.persons, 'worker')} visible.`, n.workers === s.persons);
  for (const item of PPE_ITEMS) {
    if (!required.includes(item)) continue;
    const claimed = n[`missing_${item}` as keyof NarrativeJson] as number;
    const name = ITEM_LABEL[item].toLowerCase();
    push(`missing_${item}`,
      claimed ? `${plural(claimed, 'worker')} without ${name}.` : `No worker without ${name}.`,
      truth(item) ? `${plural(truth(item), 'worker')} without ${name}.` : `No worker confirmed without ${name}.`,
      claimed === truth(item));
  }
  push('forklift', n.forklift_present ? 'A forklift is present.' : 'No forklift in view.',
    s.forklifts ? `${plural(s.forklifts, 'forklift')} detected.` : 'No forklift detected.', n.forklift_present === s.forklifts > 0);
  if (n.scene) {
    const risky = SCENE_RISKY.test(n.scene);
    claims.push({ key: 'scene', original: n.scene, text: n.scene, verdict: risky ? 'removed' : 'supported' });
  }
  if (s.unsure > 0) {
    claims.push({ key: 'unsure', original: '', text: `${plural(s.unsure, 'worker')} could not be decided and should be checked in person.`, verdict: 'supported' });
  }
  const checked = claims.filter((c) => c.key !== 'unsure');
  return {
    claims,
    total: checked.length,
    unsupported: checked.filter((c) => c.verdict !== 'supported').length,
    parsed: true,
    text: claims.filter((c) => c.verdict !== 'removed').map((c) => c.text).join(' '),
  };
}

export function detectorSummary(s: Summary, required: Item[]): string {
  const parts = [`${plural(s.persons, 'worker')} visible.`];
  for (const item of required) if (s.missingByItem[item]) parts.push(`${plural(s.missingByItem[item], 'worker')} without ${ITEM_LABEL[item].toLowerCase()}.`);
  if (s.nonCompliant === 0 && s.persons > 0) parts.push('No confirmed PPE violation.');
  if (s.unsure) parts.push(`${plural(s.unsure, 'worker')} should be checked in person.`);
  if (s.forklifts) parts.push(`${plural(s.forklifts, 'forklift')} detected.`);
  return parts.join(' ');
}

// ---- model loading and generation (lazy: nothing is downloaded until the user asks) ----

type Progress = (info: { status: string; pct: number }) => void;
let vlm: Promise<{ processor: any; model: any; lib: any }> | null = null;

function loadVlm(onProgress: Progress) {
  vlm ??= (async () => {
    const lib = await import('@huggingface/transformers');
    const gpu = 'gpu' in navigator;
    const files = new Map<string, { loaded: number; total: number }>();
    const progress_callback = (p: any) => {
      if (p.status === 'progress' && p.file) {
        files.set(p.file, { loaded: p.loaded, total: p.total });
        let l = 0;
        let t = 0;
        for (const f of files.values()) { l += f.loaded; t += f.total; }
        onProgress({ status: 'Downloading the language model (once)', pct: t ? Math.round((100 * l) / t) : 0 });
      }
    };
    const processor = await lib.AutoProcessor.from_pretrained(VLM_MODEL, { progress_callback });
    const model = await lib.AutoModelForVision2Seq.from_pretrained(VLM_MODEL, {
      dtype: { embed_tokens: 'fp32', vision_encoder: gpu ? 'fp32' : 'q8', decoder_model_merged: gpu ? 'q4' : 'q8' } as never,
      device: gpu ? 'webgpu' : 'wasm',
      progress_callback,
    });
    return { processor, model, lib };
  })();
  vlm.catch(() => { vlm = null; });
  return vlm;
}

/** Run the VLM on a (face-blurred) canvas. Returns the raw text and how long it took. */
export async function generateNarrative(canvas: HTMLCanvasElement, onProgress: Progress): Promise<{ raw: string; ms: number }> {
  const { processor, model, lib } = await loadVlm(onProgress);
  onProgress({ status: 'Writing the summary', pct: 100 });
  const t0 = performance.now();
  const blob: Blob = await new Promise((res) => canvas.toBlob((b) => res(b!), 'image/jpeg', 0.9));
  const image = await lib.RawImage.fromBlob(blob);
  const messages = [{ role: 'user', content: [{ type: 'image' }, { type: 'text', text: NARRATIVE_PROMPT }] }];
  const text = processor.apply_chat_template(messages, { add_generation_prompt: true });
  const inputs = await processor(text, [image], { do_image_splitting: false });
  const ids = await model.generate({ ...inputs, max_new_tokens: 120, do_sample: false, repetition_penalty: 1.1 });
  const out = processor.batch_decode(ids.slice(null, [inputs.input_ids.dims.at(-1), null]), { skip_special_tokens: true });
  return { raw: out[0] ?? '', ms: performance.now() - t0 };
}
