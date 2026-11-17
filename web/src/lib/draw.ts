// Canvas rendering shared by live camera, video, PDF export and the privacy slider.
import { ITEM_LABEL } from './classes';
import type { Analysis } from './analyze';
import type { Box, PersonStatus } from './matching';
import type { Zone } from './zones';

export const STATUS_COLOR: Record<PersonStatus, string> = {
  compliant: '#2fd67b',
  non_compliant: '#ff4d4f',
  unsure: '#5bc0ff',
};
export const STATUS_LABEL: Record<PersonStatus, string> = {
  compliant: 'Compliant',
  non_compliant: 'PPE missing',
  unsure: 'Check manually',
};
// Never colour alone: each status also has its own line pattern and symbol.
export const STATUS_DASH: Record<PersonStatus, number[]> = { compliant: [], non_compliant: [10, 6], unsure: [3, 5] };
export const STATUS_SYMBOL: Record<PersonStatus, string> = { compliant: '✓', non_compliant: '✕', unsure: '?' };

let scratch: HTMLCanvasElement | null = null;

/**
 * Blur regions by squeezing each one through a tiny bitmap and scaling it back up.
 * That destroys the detail for real (it is not a reversible filter) and works in every
 * browser, including Safari, which has no canvas `filter`.
 */
export function blurRegions(ctx: CanvasRenderingContext2D, boxes: Box[], expand = 0.15) {
  const { width: W, height: H } = ctx.canvas;
  scratch ??= document.createElement('canvas');
  const sctx = scratch.getContext('2d')!;
  for (const b of boxes) {
    const bw = b[2] - b[0];
    const bh = b[3] - b[1];
    const x = Math.max(0, Math.floor(b[0] - expand * bw));
    const y = Math.max(0, Math.floor(b[1] - expand * bh));
    const w = Math.min(W - x, Math.ceil(bw * (1 + 2 * expand)));
    const h = Math.min(H - y, Math.ceil(bh * (1 + 2 * expand)));
    if (w < 2 || h < 2) continue;
    const tw = Math.max(2, Math.min(6, Math.round(w / 12)));
    const th = Math.max(2, Math.min(6, Math.round(h / 12)));
    scratch.width = tw;
    scratch.height = th;
    sctx.imageSmoothingEnabled = true;
    sctx.drawImage(ctx.canvas, x, y, w, h, 0, 0, tw, th);
    ctx.save();
    ctx.beginPath();
    ctx.ellipse(x + w / 2, y + h / 2, w / 2, h / 2, 0, 0, Math.PI * 2);
    ctx.clip();
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(scratch, 0, 0, tw, th, x, y, w, h);
    ctx.restore();
  }
}

export interface OverlayPerson {
  box: Box;
  status: PersonStatus;
  label: string;
  missing: string[];
  alert?: boolean; // in a danger zone or near a forklift
}

export function drawZones(ctx: CanvasRenderingContext2D, zones: Zone[], active: Set<number>, scale = 1) {
  for (const z of zones) {
    if (z.points.length < 2) continue;
    const hot = active.has(z.id);
    ctx.save();
    ctx.beginPath();
    z.points.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.closePath();
    ctx.fillStyle = hot ? 'rgba(255,77,79,0.22)' : 'rgba(255,122,26,0.12)';
    ctx.fill();
    ctx.lineWidth = (hot ? 4 : 2.5) * scale;
    ctx.setLineDash([12 * scale, 8 * scale]);
    ctx.strokeStyle = hot ? '#ff4d4f' : '#ff7a1a';
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.font = `600 ${13 * scale}px "JetBrains Mono Variable", monospace`;
    ctx.fillStyle = hot ? '#ff4d4f' : '#ff7a1a';
    ctx.fillText(`⚠ ${z.name}`, z.points[0][0] + 6 * scale, z.points[0][1] - 6 * scale);
    ctx.restore();
  }
}

export function drawPeople(ctx: CanvasRenderingContext2D, people: OverlayPerson[], scale = 1) {
  for (const p of people) {
    const [x1, y1, x2, y2] = p.box;
    const color = STATUS_COLOR[p.status];
    ctx.save();
    ctx.lineWidth = 3 * scale;
    ctx.strokeStyle = color;
    ctx.setLineDash(STATUS_DASH[p.status].map((d) => d * scale));
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.setLineDash([]);
    if (p.alert) {
      ctx.lineWidth = 2 * scale;
      ctx.strokeStyle = '#ff7a1a';
      ctx.strokeRect(x1 - 5 * scale, y1 - 5 * scale, x2 - x1 + 10 * scale, y2 - y1 + 10 * scale);
    }
    const text = `${STATUS_SYMBOL[p.status]} ${p.label}${p.missing.length ? ' · no ' + p.missing.join(', ') : ''}`;
    ctx.font = `600 ${13 * scale}px "JetBrains Mono Variable", monospace`;
    const tw = ctx.measureText(text).width + 12 * scale;
    const th = 20 * scale;
    const ty = y1 - th >= 0 ? y1 - th : y1;
    ctx.fillStyle = color;
    ctx.fillRect(x1 - 1.5 * scale, ty, tw, th);
    ctx.fillStyle = '#0b0d10';
    ctx.fillText(text, x1 + 5 * scale, ty + 14.5 * scale);
    ctx.restore();
  }
}

export function overlayFromAnalysis(a: Analysis, idOf: (i: number) => number = (i) => i + 1): OverlayPerson[] {
  return a.people.map((p, i) => ({
    box: p.box,
    status: p.status,
    label: `W${idOf(i)}`,
    missing: p.missing.map((m) => ITEM_LABEL[m].toLowerCase()),
    alert: a.events[i]?.zones.length > 0 || a.events[i]?.nearForklift,
  }));
}

/** Compose the shareable frame: source image, blurred faces, zones and person boxes. */
export function renderAnnotated(
  source: CanvasImageSource, a: Analysis, zones: Zone[], opts: { blur: boolean; boxes?: boolean } = { blur: true },
): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = a.w;
  canvas.height = a.h;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(source, 0, 0, a.w, a.h);
  if (opts.blur) blurRegions(ctx, a.blur);
  if (opts.boxes !== false) {
    const scale = Math.max(1, Math.max(a.w, a.h) / 900);
    const active = new Set(a.events.flatMap((e) => e.zones));
    drawZones(ctx, zones, active, scale);
    drawPeople(ctx, overlayFromAnalysis(a), scale);
  }
  return canvas;
}
