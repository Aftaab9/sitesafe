// Model card for non-experts. All numbers are read from metrics.json, which is written by
// scripts/build_metrics.py from the evaluation outputs, so the page cannot drift from the report.
import { useEffect, useState } from 'react';
import { Ban, CheckCircle2, TriangleAlert } from 'lucide-react';
import { cn, Eyebrow, PageHeader, Panel } from '../components/ui';
import { asset } from '../lib/engine';

interface ClassRow { class: string; precision: number; recall: number; ap50: number; ap50_95: number }
interface Metrics {
  updated: string;
  training?: { note: string };
  detection?: Record<string, { label: string; split: string; map50: number; map50_95: number; no_class_mean_recall: number; per_class: ClassRow[] }>;
  compliance?: { label: string; n: number; decided: number; accuracy: number | null; precision_nc: number | null; recall_nc: number | null; unsure_share: number; nc_shown_green: number }[];
  synthetic?: { note: string; rows: { class: string; base: number; synth: number; delta: number; lo: number; hi: number }[]; stress?: { class: string; base: number; synth: number; delta: number; lo: number; hi: number }[] };
  quantisation?: { fp32_mb: number; int8_mb: number; map50_fp32: number; map50_int8: number; shipped: string };
  blur?: { map50_original: number; map50_blurred: number; helmet_recall_original: number; helmet_recall_blurred: number; goggles_recall_original: number; goggles_recall_blurred: number; decision: string };
  calibration?: { ece_before: number; ece_after: number; temperature: number; unsure_share: number };
  subgroups?: { note: string; rows: { group: string; n: number; person_recall: number; lo: number; hi: number; no_recall: number | null }[] };
  smoothing?: { precision_raw: number; precision_smoothed: number; recall_raw: number; recall_smoothed: number; clips: number; flips_raw: number; flips_smoothed: number };
  vlm?: { model: string; images: number; json_valid_share: number; seconds_per_image_cpu: number; hallucination_rate_before: number | null; error_rate_after_verification: number | null; per_claim: Record<string, { hallucination_before: number; error_after_verification: number }> };

  speed?: { label: string; ms: number }[];
  own?: { note: string; rows: { label: string; person_recall: number; accuracy: number | null; decided: number; decidable: number; nc_caught: number | null; nc_green: number | null; unsure: number; no_vest_called: number | null; no_vest_shown_vest: number | null; boots_shown_missing: number | null; no_helmet_called: number | null; false_person_boxes: number; night_nc_caught: number | null }[] };
  failures?: { image: string; note: string }[];
}

const pct = (v: number | null | undefined) => (v === null || v === undefined ? '–' : `${(100 * v).toFixed(1)}%`);

function Table({ head, rows }: { head: string[]; rows: (string | number)[][] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[420px] text-left text-sm">
        <thead><tr className="text-xs text-muted">{head.map((h, i) => <th key={h} className={cn('py-1.5 pr-3 font-medium', i > 0 && 'text-right')}>{h}</th>)}</tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-t border-line">
              {r.map((c, j) => <td key={j} className={cn('py-1.5 pr-3', j > 0 ? 'num text-right' : 'font-medium')}>{c}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return <Panel className="p-5"><h2 className="text-lg font-semibold">{title}</h2><div className="mt-3 space-y-3 text-sm">{children}</div></Panel>;
}

export default function ModelCard() {
  const [m, setM] = useState<Metrics | null>(null);
  useEffect(() => { fetch(asset('metrics.json')).then((r) => (r.ok ? r.json() : null)).then(setM).catch(() => undefined); }, []);
  const det = m?.detection ? Object.values(m.detection) : [];

  return (
    <div className="mx-auto max-w-4xl space-y-4 px-4 py-6">
      <PageHeader eyebrow="Model card" title="What this model is, and what it is not">
        A plain-language summary of the detector behind SiteSafe: what it was built for, how well it works, where it fails and how it should be used.
      </PageHeader>

      <div className="grid gap-4 md:grid-cols-2 [&>*]:min-w-0">
        <Section title="Intended use">
          <ul className="space-y-2">
            {['Helping a site safety officer spot missing PPE in photos, short clips and live video.',
              'Aggregate reporting: compliance percentage and counts by PPE type for a site or shift.',
              'Toolbox talks and training, including the Inspector Challenge.'].map((t) => (
              <li key={t} className="flex gap-2"><CheckCircle2 size={17} className="mt-0.5 shrink-0 text-ok" /> {t}</li>
            ))}
          </ul>
        </Section>
        <Section title="Out of scope: do not use it for">
          <ul className="space-y-2">
            {['Monitoring, scoring or disciplining an individual worker.',
              'Identifying people. The system has no face recognition and blurs faces.',
              'Replacing a site inspection or signing off a site as safe.',
              'Covert surveillance, or use without telling the workers what it is for.'].map((t) => (
              <li key={t} className="flex gap-2"><Ban size={17} className="mt-0.5 shrink-0 text-danger" /> {t}</li>
            ))}
          </ul>
        </Section>
      </div>

      <Section title="Model and data">
        <p>
          YOLO11n object detector (2.6 M parameters, about 10 MB as ONNX) fine-tuned on <strong>Construction-PPE</strong> by Ultralytics (CC BY 4.0):
          1,000 labelled images with 12 classes (helmet, gloves, vest, boots, goggles, Person, forklift and five “no_” classes for visibly missing PPE).
          700 images were used for training, 200 for model selection and 100 were held out as a test set that the model never saw.
          Faces are found with YuNet (MIT licence) and blurred on the device.
        </p>
        {m?.training && <p className="text-muted">{m.training.note}</p>}
      </Section>

      {det.length > 0 && (
        <Section title="Detection quality on held-out photos">
          {det.map((d) => (
            <div key={d.label}>
              <Eyebrow className="mb-1">{d.label} · {d.split} split · mAP@0.5 {pct(d.map50)} · mAP@0.5:0.95 {pct(d.map50_95)} · mean recall of “no_” classes {pct(d.no_class_mean_recall)}</Eyebrow>
              <Table head={['Class', 'Precision', 'Recall', 'AP@0.5', 'AP@0.5:0.95']} rows={d.per_class.map((c) => [c.class, pct(c.precision), pct(c.recall), pct(c.ap50), pct(c.ap50_95)])} />
            </div>
          ))}
          <p className="text-muted">Recall on the “no_” classes matters most: a missed “no helmet” is the costly error. Those classes are also the rarest in the data.</p>
        </Section>
      )}

      {m?.compliance && (
        <Section title="Person-level compliance (the decision the officer sees)">
          <Table head={['Method', 'Decided / people', 'Accuracy', 'Precision (missing PPE)', 'Recall (missing PPE)', 'Sent to “not sure”', 'Violators shown green']}
            rows={m.compliance.map((c) => [c.label, `${c.decided} / ${c.n}`, pct(c.accuracy), pct(c.precision_nc), pct(c.recall_nc), pct(c.unsure_share), pct(c.nc_shown_green)])} />
          <p className="text-muted">Accuracy, precision and recall are computed on the people a method decided. A method that abstains on almost everyone can look accurate on the few it decides, so read the first column too. People sent to “not sure” are handed to the officer.</p>
        </Section>
      )}

      {m?.own && (
        <Section title="Real-world check: 30 photos I collected and labelled myself">
          <Table head={['Model', 'Workers found', 'Accuracy (decided)', 'Violators caught or sent to check', 'Same, at night', 'Violators shown green', 'No-hi-vis shown as vest', 'Boots shown as “no boots”']}
            rows={m.own.rows.map((r) => [r.label, pct(r.person_recall), `${pct(r.accuracy)} of ${r.decided}`, pct(r.nc_caught), pct(r.night_nc_caught), pct(r.nc_green), pct(r.no_vest_shown_vest), pct(r.boots_shown_missing)])} />
          <p className="text-muted">{m.own.note}</p>
        </Section>
      )}

      {m?.synthetic && (
        <Section title="Does the generative AI help? Recall on real test photos, without vs with synthetic training images">
          <Table head={['Class', 'Real only', 'Real + synthetic', 'Change', '95% interval']}
            rows={m.synthetic.rows.map((r) => [r.class, pct(r.base), pct(r.synth), `${r.delta >= 0 ? '+' : ''}${(100 * r.delta).toFixed(1)} pts`, `${(100 * r.lo).toFixed(1)} to ${(100 * r.hi).toFixed(1)}`])} />
          {m.synthetic.stress && (
            <>
              <Eyebrow className="pt-2">Stress test: the same 100 test photos re-rendered as night, rain, fog or dust</Eyebrow>
              <Table head={['Classes', 'Real only', 'Real + synthetic', 'Change', '95% interval']}
                rows={m.synthetic.stress.map((r) => [r.class, pct(r.base), pct(r.synth), `${r.delta >= 0 ? '+' : ''}${(100 * r.delta).toFixed(1)} pts`, `${(100 * r.lo).toFixed(1)} to ${(100 * r.hi).toFixed(1)}`])} />
            </>
          )}
          <p className="text-muted">{m.synthetic.note}</p>
        </Section>
      )}

      <div className="grid gap-4 md:grid-cols-2 [&>*]:min-w-0">
        {m?.calibration && (
          <Section title="Confidence and the “not sure” state">
            <p>Expected calibration error on the test split: {pct(m.calibration.ece_before)} before and {pct(m.calibration.ece_after)} after temperature scaling (T = {m.calibration.temperature.toFixed(2)}).</p>
            <p>Per-class thresholds chosen on validation send low-confidence decisions to “check manually” instead of forcing green or red. On the test split {pct(m.calibration.unsure_share)} of people end up there.</p>
          </Section>
        )}
        {m?.blur && (
          <Section title="Does face blurring hurt detection?">
            <Table head={['', 'Original', 'Blurred first']} rows={[
              ['mAP@0.5', pct(m.blur.map50_original), pct(m.blur.map50_blurred)],
              ['Helmet recall', pct(m.blur.helmet_recall_original), pct(m.blur.helmet_recall_blurred)],
              ['Goggles recall', pct(m.blur.goggles_recall_original), pct(m.blur.goggles_recall_blurred)],
            ]} />
            <p className="text-muted">{m.blur.decision}</p>
          </Section>
        )}
        {m?.quantisation && (
          <Section title="Model size and quantisation">
            <Table head={['', 'Float32', 'Int8']} rows={[['File size', `${m.quantisation.fp32_mb.toFixed(1)} MB`, `${m.quantisation.int8_mb.toFixed(1)} MB`], ['mAP@0.5', pct(m.quantisation.map50_fp32), pct(m.quantisation.map50_int8)]]} />
            <p className="text-muted">{m.quantisation.shipped}</p>
          </Section>
        )}
        {m?.smoothing && (
          <Section title="Temporal smoothing on clips">
            <Table head={['', 'Per frame', '3 of 5 frames']} rows={[
              ['Violation precision', pct(m.smoothing.precision_raw), pct(m.smoothing.precision_smoothed)],
              ['Violation recall', pct(m.smoothing.recall_raw), pct(m.smoothing.recall_smoothed)],
              ['Status flips per 100 track-frames', m.smoothing.flips_raw.toFixed(1), m.smoothing.flips_smoothed.toFixed(1)],
            ]} />
            <p className="text-muted">Measured on {m.smoothing.clips} simulated clips (camera moves over labelled test photos with frame noise).</p>
          </Section>
        )}
      </div>

      {m?.vlm && (
        <Section title="Written summary: how often does the language model make things up?">
          <Table head={['Claim', 'Wrong before verification', 'Wrong after verification']}
            rows={[...Object.entries(m.vlm.per_claim).map(([k, v]) => [k.replace('_', ' '), pct(v.hallucination_before), pct(v.error_after_verification)]),
              ['All claims', pct(m.vlm.hallucination_rate_before), pct(m.vlm.error_rate_after_verification)]]} />
          <p className="text-muted">
            {m.vlm.model.split('/')[1]} on {m.vlm.images} face-blurred test photos, scored against the labels. Only {pct(m.vlm.json_valid_share)} of its answers
            were usable JSON, and it often invented worker names and ages. After verification every published number comes from the detector, so the
            remaining error is the detector's own. About {m.vlm.seconds_per_image_cpu.toFixed(0)} s per photo on a laptop CPU, which is why the summary is an optional button.
          </p>
        </Section>
      )}

      {m?.subgroups && (
        <Section title="Bias check: results by subgroup">
          <Table head={['Subgroup', 'People', 'Person recall', '95% interval', '“no_” recall']}
            rows={m.subgroups.rows.map((r) => [r.group, r.n, pct(r.person_recall), `${(100 * r.lo).toFixed(0)} to ${(100 * r.hi).toFixed(0)}`, pct(r.no_recall)])} />
          <p className="text-muted">{m.subgroups.note}</p>
        </Section>
      )}

      {m?.speed && (
        <Section title="Speed">
          <Table head={['Where', 'Time per image']} rows={m.speed.map((s) => [s.label, `${s.ms.toFixed(0)} ms`])} />
        </Section>
      )}

      <Section title="Known failures and limits">
        <ul className="list-disc space-y-1.5 pl-5">
          <li>Small or distant workers: gloves, goggles and boots are only a few pixels wide and are often missed or sent to “not sure”.</li>
          <li>Occlusion and crowds: PPE can be matched to the wrong person when people overlap heavily.</li>
          <li>Night, heavy rain, fog and dust reduce recall. The synthetic-data experiment targets this, and it does not remove it.</li>
          <li>Unusual PPE colours and shapes (dark vests, bump caps, sunglasses instead of goggles) are under-represented in the training data.</li>
          <li>Forklift distance is estimated in the image, without depth. Treat proximity alerts as prompts to look, not as measurements.</li>
        </ul>
        {m?.failures && m.failures.length > 0 && (
          <div className="grid gap-3 pt-2 sm:grid-cols-2 [&>*]:min-w-0">
            {m.failures.map((f) => (
              <figure key={f.image} className="overflow-hidden rounded border border-line">
                <img src={asset(f.image)} alt="" loading="lazy" className="w-full" />
                <figcaption className="flex gap-2 p-2 text-xs text-muted"><TriangleAlert size={14} className="mt-0.5 shrink-0 text-hazard" /> {f.note}</figcaption>
              </figure>
            ))}
          </div>
        )}
      </Section>

      <Section title="Human in the loop">
        <p>The model never has the last word. Low-confidence people are routed to “check manually”; reports say that they support, not replace, the officer; and every figure is aggregate. If the model and your own eyes disagree, trust your eyes and, if you can, note the case so the model can be improved.</p>
      </Section>
      {m && <p className="font-mono text-[11px] text-muted">Metrics generated {m.updated}.</p>}
      {!m && <p className="text-sm text-muted">Metrics file not found. Run <code className="font-mono">python scripts/build_metrics.py</code> after evaluation.</p>}
    </div>
  );
}
