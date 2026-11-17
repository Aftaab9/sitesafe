// Incident report generation, entirely in the browser (PDF via jsPDF, CSV as text).
// Reports contain aggregate counts and anonymous worker numbers only.
import { ITEM_LABEL, PPE_ITEMS, SITE_PROFILES } from './classes';
import type { LogEntry } from './store';
import type { VerifiedNarrative } from './narrative';

export interface ReportInput {
  entry: LogEntry;
  image?: HTMLCanvasElement | null; // already face-blurred
  zones?: number;
  narrative?: VerifiedNarrative | null;
}

const fmtTime = (t: number) => new Date(t).toLocaleString();

export function entryCsv(entries: LogEntry[]): string {
  const head = ['time', 'source', 'label', 'site_profile', 'persons', 'compliant', 'non_compliant', 'unsure',
    'compliance_pct', ...PPE_ITEMS.map((i) => `missing_${i}`), 'forklifts', 'zone_events', 'forklift_proximity_events', 'ms'];
  const rows = entries.map((e) => [
    new Date(e.time).toISOString(), e.source, JSON.stringify(e.label), e.profile, e.summary.persons, e.summary.compliant,
    e.summary.nonCompliant, e.summary.unsure, e.summary.compliancePct ?? '', ...PPE_ITEMS.map((i) => e.summary.missingByItem[i]),
    e.summary.forklifts, e.zoneEvents, e.forkliftEvents, Math.round(e.ms),
  ].join(','));
  return [head.join(','), ...rows].join('\n') + '\n';
}

export async function buildPdf({ entry, image, narrative }: ReportInput): Promise<Blob> {
  const { jsPDF } = await import('jspdf');
  const doc = new jsPDF({ unit: 'mm', format: 'a4' });
  const W = 210;
  const M = 16;
  let y = 0;

  // header band with hazard stripes
  doc.setFillColor(11, 13, 16);
  doc.rect(0, 0, W, 26, 'F');
  doc.setFillColor(255, 212, 0);
  for (let x = -10; x < W + 10; x += 8) doc.triangle(x, 26, x + 4, 26, x + 7, 23, 'F'), doc.triangle(x + 4, 26, x + 7, 23, x + 3, 23, 'F');
  doc.setTextColor(255, 212, 0);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(18);
  doc.text('SiteSafe', M, 13);
  doc.setTextColor(255, 255, 255);
  doc.setFontSize(11);
  doc.setFont('helvetica', 'normal');
  doc.text('PPE compliance incident summary', M, 19);
  doc.setFontSize(9);
  doc.text(fmtTime(entry.time), W - M, 13, { align: 'right' });
  doc.text(`Profile: ${SITE_PROFILES[entry.profile].label}`, W - M, 19, { align: 'right' });
  y = 36;

  const s = entry.summary;
  doc.setTextColor(20, 24, 29);
  // headline numbers
  const cells: [string, string][] = [
    ['Compliance', s.compliancePct === null ? 'n/a' : `${s.compliancePct}%`],
    ['Workers seen', String(entry.workers?.length ?? s.persons)],
    ['PPE missing', String(s.nonCompliant)],
    ['Check manually', String(s.unsure)],
  ];
  const cw = (W - 2 * M) / cells.length;
  cells.forEach(([label, value], i) => {
    doc.setDrawColor(200, 200, 200);
    doc.rect(M + i * cw, y, cw - 3, 20);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(17);
    doc.text(value, M + i * cw + 4, y + 10);
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(8.5);
    doc.setTextColor(95, 99, 108);
    doc.text(label.toUpperCase(), M + i * cw + 4, y + 16);
    doc.setTextColor(20, 24, 29);
  });
  y += 28;

  if (image) {
    const maxW = W - 2 * M;
    const maxH = 95;
    const r = Math.min(maxW / image.width, maxH / image.height);
    const iw = image.width * r;
    const ih = image.height * r;
    doc.addImage(image.toDataURL('image/jpeg', 0.82), 'JPEG', M + (maxW - iw) / 2, y, iw, ih);
    y += ih + 3;
    doc.setFontSize(8);
    doc.setTextColor(95, 99, 108);
    doc.text('Faces are blurred on the device before this image is produced. Solid box: compliant. Dashed: PPE missing. Dotted: check manually.', M, y + 2);
    doc.setTextColor(20, 24, 29);
    y += 9;
  }

  const section = (title: string) => {
    if (y > 262) { doc.addPage(); y = 20; }
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(11);
    doc.text(title, M, y);
    doc.setDrawColor(255, 122, 26);
    doc.setLineWidth(0.6);
    doc.line(M, y + 1.5, M + 18, y + 1.5);
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(9.5);
    y += 7;
  };
  const line = (text: string, indent = 0) => {
    for (const ln of doc.splitTextToSize(text, W - 2 * M - indent) as string[]) {
      if (y > 280) { doc.addPage(); y = 20; }
      doc.text(ln, M + indent, y);
      y += 4.8;
    }
  };

  section('Violations by PPE type');
  const maxCount = Math.max(1, ...PPE_ITEMS.map((i) => s.missingByItem[i]));
  for (const item of PPE_ITEMS) {
    const n = s.missingByItem[item];
    const req = SITE_PROFILES[entry.profile].required.includes(item);
    doc.text(`${ITEM_LABEL[item]}${req ? '' : ' (not required)'}`, M, y);
    doc.setFillColor(230, 230, 230);
    doc.rect(M + 42, y - 3, 90, 3.6, 'F');
    if (n) { doc.setFillColor(200, 32, 38); doc.rect(M + 42, y - 3, (90 * n) / maxCount, 3.6, 'F'); }
    doc.text(String(n), M + 136, y);
    y += 6;
  }
  y += 3;

  section('Site hazards');
  line(`Forklifts detected: ${s.forklifts}.  Workers inside a marked danger zone: ${entry.zoneEvents}.  Workers close to a forklift: ${entry.forkliftEvents}.`);
  y += 3;

  if (entry.workers?.length) {
    section('Anonymous worker log (clip)');
    for (const w of entry.workers) {
      line(`Worker ${String(w.id).padStart(2, '0')}: ${w.status.replace('_', ' ')}${w.missing.length ? ', missing ' + w.missing.map((m) => ITEM_LABEL[m].toLowerCase()).join(', ') : ''}`);
    }
    y += 3;
  }

  if (narrative) {
    section('Assistant summary (verified against the detector)');
    for (const c of narrative.claims) {
      const mark = c.verdict === 'supported' ? '[verified]' : c.verdict === 'corrected' ? '[corrected]' : '[removed: unsupported]';
      line(`${mark} ${c.verdict === 'removed' ? c.original : c.text}`);
    }
    doc.setFontSize(8);
    doc.setTextColor(95, 99, 108);
    line('The summary is written by a small open vision-language model running on the device. Each claim is checked against the detector output; unsupported claims are removed.');
    doc.setTextColor(20, 24, 29);
    doc.setFontSize(9.5);
    y += 3;
  }

  section('How to read this report');
  doc.setTextColor(60, 64, 72);
  line('This summary supports the site safety officer and does not replace an inspection. Counts are aggregate and worker numbers are anonymous and reset for every analysis. "Check manually" means the model was not confident enough to decide. The report must not be used to monitor or discipline individual workers. No image or video was uploaded or stored to produce it.');

  doc.setFontSize(7.5);
  doc.setTextColor(140, 140, 140);
  const pages = doc.getNumberOfPages();
  for (let i = 1; i <= pages; i++) {
    doc.setPage(i);
    doc.text(`SiteSafe · generated on-device · ${entry.label} · page ${i}/${pages}`, M, 291);
  }
  return doc.output('blob');
}
