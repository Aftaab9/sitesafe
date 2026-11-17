import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  Camera, Download, Eye, FileText, ImagePlus, Loader2, PenTool, ScanFace, Sparkles, Table, Trash2, TriangleAlert, Upload,
} from 'lucide-react';
import { EngineGate } from '../components/EngineStatus';
import { DetectionSettings, ProfilePicker, SummaryPanel } from '../components/Panels';
import { WorkerCard } from '../components/Ppe';
import { Stage } from '../components/Stage';
import { Button, Chip, cn, Eyebrow, PageHeader, Panel } from '../components/ui';
import { analyse } from '../lib/analyze';
import { SITE_PROFILES } from '../lib/classes';
import { renderAnnotated } from '../lib/draw';
import { asset, engine, type DetectResult } from '../lib/engine';
import { download, fileToCanvas, loadCatalogue, urlToCanvas, type Sample } from '../lib/media';
import { generateNarrative, parseNarrative, verifyNarrative, type VerifiedNarrative } from '../lib/narrative';
import { buildPdf, entryCsv } from '../lib/report';
import { prefersReducedMotion, useLog, useSettings, type LogEntry } from '../lib/store';
import type { Zone } from '../lib/zones';

function Narrative({ state }: { state: { busy: string | null; pct: number; result: VerifiedNarrative | null; ms: number; error: string | null } }) {
  if (state.busy) {
    return (
      <div className="mt-3 text-sm text-muted" role="status">
        <div className="flex items-center gap-2"><Loader2 size={14} className="animate-spin" /> {state.busy} {state.pct < 100 ? `${state.pct}%` : ''}</div>
        <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-raised"><div className="h-full bg-hivis transition-[width]" style={{ width: `${state.pct}%` }} /></div>
      </div>
    );
  }
  if (state.error) return <p className="mt-3 text-sm text-danger">{state.error}</p>;
  if (!state.result) return null;
  const r = state.result;
  return (
    <div className="mt-3 space-y-2 text-sm">
      <p className="leading-relaxed">{r.text}</p>
      <ul className="space-y-1 border-t border-line pt-2 text-xs">
        {r.claims.filter((c) => c.key !== 'unsure').map((c) => (
          <li key={c.key} className="flex gap-2">
            <span className={cn('shrink-0 font-mono', c.verdict === 'supported' ? 'text-ok' : c.verdict === 'corrected' ? 'text-hazard' : 'text-danger')}>
              {c.verdict === 'supported' ? '✓ verified' : c.verdict === 'corrected' ? '↻ corrected' : '✕ removed'}
            </span>
            <span className="text-muted">
              {c.verdict === 'supported' ? c.text : <><s>{c.original}</s>{c.verdict === 'corrected' && <> → {c.text}</>}</>}
            </span>
          </li>
        ))}
      </ul>
      <p className="font-mono text-[11px] text-muted">
        {r.parsed
          ? `${r.unsupported} of ${r.total} model claims disagreed with the detector and were fixed · ${(state.ms / 1000).toFixed(1)} s on this device`
          : `The model did not return valid JSON, so the summary was written from the detector output alone · ${(state.ms / 1000).toFixed(1)} s`}
      </p>
    </div>
  );
}

function Workbench() {
  const st = useSettings();
  const addLog = useLog((s) => s.add);
  const [params, setParams] = useSearchParams();
  const [samples, setSamples] = useState<Sample[]>([]);
  const [source, setSource] = useState<HTMLCanvasElement | null>(null);
  const [label, setLabel] = useState('');
  const [sample, setSample] = useState<Sample | null>(null);
  const [raw, setRaw] = useState<DetectResult | null>(null);
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [zones, setZones] = useState<Zone[]>([]);
  const [drawing, setDrawing] = useState(false);
  const [compare, setCompare] = useState(false);
  const [explain, setExplain] = useState(false);
  const [hovered, setHovered] = useState<number | null>(null);
  const [animateKey, setAnimateKey] = useState(0);
  const [dragOver, setDragOver] = useState(false);
  const [nar, setNar] = useState<{ busy: string | null; pct: number; result: VerifiedNarrative | null; ms: number; error: string | null }>(
    { busy: null, pct: 0, result: null, ms: 0, error: null });
  const fileInput = useRef<HTMLInputElement>(null);
  const cameraInput = useRef<HTMLInputElement>(null);
  const logged = useRef<string | null>(null);
  const runId = useRef(0);

  useEffect(() => { loadCatalogue().then((c) => setSamples(c.samples)).catch(() => setSamples([])); }, []);

  const run = useCallback(async (canvas: HTMLCanvasElement, name: string, smp: Sample | null) => {
    const id = ++runId.current;
    setSource(canvas); setLabel(name); setSample(smp); setRaw(null); setError(null); setScanning(true);
    setHovered(null); setZones([]); setExplain(false); setNar({ busy: null, pct: 0, result: null, ms: 0, error: null });
    try {
      const minShow = new Promise((r) => setTimeout(r, prefersReducedMotion() ? 0 : 1100));
      const bitmap = await createImageBitmap(canvas);
      const [res] = await Promise.all([engine.detect(bitmap, { conf: useSettings.getState().conf }), minShow]);
      if (id !== runId.current) return;
      logged.current = null;
      setRaw(res);
      setAnimateKey((k) => k + 1);
    } catch (e) {
      if (id === runId.current) setError(e instanceof Error ? e.message : 'Analysis failed');
    } finally {
      if (id === runId.current) setScanning(false);
    }
  }, []);

  const openFile = useCallback(async (file: File | undefined) => {
    if (!file) return;
    if (!file.type.startsWith('image/')) { setError('That file is not an image. For clips, use the Clip tab.'); return; }
    try { await run(await fileToCanvas(file), file.name, null); } catch { setError('This image could not be opened.'); }
  }, [run]);

  const openSample = useCallback(async (s: Sample) => {
    try { await run(await urlToCanvas(asset(s.file)), s.title, s); } catch { setError('The sample could not be loaded.'); }
  }, [run]);

  // deep link from the landing page: #/analyze?sample=s01
  useEffect(() => {
    const id = params.get('sample');
    if (!id || !samples.length) return;
    const s = samples.find((x) => x.id === id) ?? samples[0];
    setParams({}, { replace: true });
    void openSample(s);
  }, [params, samples, openSample, setParams]);

  // Re-detect when the sensitivity changes; re-match (cheap) when the profile or zones change.
  useEffect(() => {
    if (!source || !raw) return;
    const t = setTimeout(async () => {
      const bitmap = await createImageBitmap(source);
      const res = await engine.detect(bitmap, { conf: st.conf });
      setRaw(res);
    }, 250);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [st.conf]);

  const analysis = useMemo(
    () => (raw ? analyse(raw, st.profile, zones, { strictBlur: st.strictBlur, forkliftDistance: st.forkliftDistance }) : null),
    [raw, st.profile, zones, st.strictBlur, st.forkliftDistance],
  );
  const required = SITE_PROFILES[st.profile].required;

  const entry: LogEntry | null = useMemo(() => analysis && {
    id: `${label}-${animateKey}`, time: Date.now(), source: 'photo', label, profile: st.profile, summary: analysis.summary,
    zoneEvents: analysis.zoneAlerts, forkliftEvents: analysis.forkliftAlerts, ms: analysis.ms.total,
  }, [analysis, label, animateKey, st.profile]);

  useEffect(() => {
    if (entry && logged.current !== entry.id) { logged.current = entry.id; addLog(entry); }
  }, [entry, addLog]);

  const annotated = () => (source && analysis ? renderAnnotated(source, analysis, zones, { blur: true }) : null);

  const exportPdf = async () => {
    if (!entry) return;
    download(`sitesafe-report-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-')}.pdf`,
      await buildPdf({ entry, image: annotated(), narrative: nar.result }));
  };
  const exportCsv = () => entry && download('sitesafe-incident.csv', new Blob([entryCsv([entry])], { type: 'text/csv' }));
  const exportImage = () => annotated()?.toBlob((b) => b && download('sitesafe-blurred.jpg', b), 'image/jpeg', 0.9);

  const summarise = async () => {
    if (!source || !analysis) return;
    setNar({ busy: 'Loading the language model', pct: 0, result: null, ms: 0, error: null });
    try {
      // The VLM only ever sees the face-blurred frame.
      const blurred = renderAnnotated(source, analysis, [], { blur: true, boxes: false });
      const { raw: text, ms } = await generateNarrative(blurred, (p) => setNar((n) => ({ ...n, busy: p.status, pct: p.pct })));
      setNar({ busy: null, pct: 100, ms, error: null, result: verifyNarrative(parseNarrative(text), analysis.summary, required) });
    } catch (e) {
      setNar({ busy: null, pct: 0, result: null, ms: 0, error: `The language model could not run on this device (${e instanceof Error ? e.message : 'unknown error'}). The detector results above are unaffected.` });
    }
  };

  const picker = (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => { e.preventDefault(); setDragOver(false); void openFile(e.dataTransfer.files[0]); }}
      className={cn(
        'hud-corners grid place-items-center rounded border border-dashed border-line bg-surface px-4 py-12 text-center transition-colors',
        dragOver && 'border-hivis bg-hivis/5',
      )}
    >
      <div>
        <div className="mx-auto grid h-14 w-14 place-items-center rounded-full bg-hivis text-[#0b0d10]"><ImagePlus size={26} /></div>
        <h2 className="mt-4 text-lg font-semibold">Drop a site photo here</h2>
        <p className="mx-auto mt-1 max-w-sm text-sm text-muted">It is analysed on this device. Nothing is uploaded, and nothing is kept when you close the tab.</p>
        <div className="mt-5 flex flex-wrap justify-center gap-2">
          <Button variant="primary" onClick={() => fileInput.current?.click()}><Upload size={16} /> Choose a photo</Button>
          <Button onClick={() => cameraInput.current?.click()}><Camera size={16} /> Take a photo</Button>
        </div>
      </div>
    </div>
  );

  return (
    <div className="mx-auto max-w-7xl px-4 py-6">
      <PageHeader eyebrow="Photo check" title="Who is missing PPE in this photo?">
        Each person gets a box: solid green when compliant, dashed red when required PPE is missing, dotted blue when the model is not sure.
      </PageHeader>
      <input ref={fileInput} type="file" accept="image/*" className="hidden" onChange={(e) => { void openFile(e.target.files?.[0]); e.target.value = ''; }} />
      <input ref={cameraInput} type="file" accept="image/*" capture="environment" className="hidden" onChange={(e) => { void openFile(e.target.files?.[0]); e.target.value = ''; }} />

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_360px]">
        <div className="min-w-0 space-y-4">
          {error && <p role="alert" className="flex items-center gap-2 rounded border border-danger/50 bg-danger/10 px-3 py-2 text-sm text-danger"><TriangleAlert size={16} /> {error}</p>}
          {source ? (
            <>
              <Stage
                source={source} analysis={analysis} scanning={scanning} profile={st.profile} blur={st.blurFaces}
                compare={compare} hovered={hovered} onHover={setHovered} zones={zones} onZones={setZones}
                drawing={drawing} onDrawingDone={() => setDrawing(false)} animateKey={animateKey}
                heatmap={explain && sample?.heatmap ? asset(sample.heatmap) : null}
              />
              {sample?.failure && (
                <p className="flex items-start gap-2 rounded border border-hazard/50 bg-hazard/10 px-3 py-2 text-sm">
                  <TriangleAlert size={16} className="mt-0.5 shrink-0 text-hazard" />
                  <span><strong>Known failure case.</strong> {sample.note}</span>
                </p>
              )}
              <div className="flex flex-wrap items-center gap-2">
                <Button size="sm" variant={compare ? 'primary' : 'outline'} aria-pressed={compare} onClick={() => setCompare((c) => !c)} title="Drag to compare the original with the face-blurred output">
                  <ScanFace size={15} /> Privacy slider
                </Button>
                <Button size="sm" variant={drawing ? 'primary' : 'outline'} aria-pressed={drawing} onClick={() => setDrawing((d) => !d)}>
                  <PenTool size={15} /> Draw danger zone
                </Button>
                {zones.length > 0 && <Button size="sm" variant="ghost" onClick={() => setZones([])}><Trash2 size={15} /> Clear zones ({zones.length})</Button>}
                <Button size="sm" variant={explain ? 'primary' : 'outline'} aria-pressed={explain} disabled={!sample?.heatmap} onClick={() => setExplain((x) => !x)}
                  title={sample?.heatmap ? 'Show where the detector looked (EigenCAM)' : 'Heat-maps are pre-computed for the gallery samples'}>
                  <Eye size={15} /> Explain
                </Button>
                <span className="ml-auto font-mono text-[11px] text-muted">
                  {analysis ? `${analysis.ms.total.toFixed(0)} ms · model ${analysis.ms.infer.toFixed(0)} ms · ${analysis.w}×${analysis.h}` : scanning ? 'scanning…' : ''}
                </span>
              </div>
              <div className="flex flex-wrap gap-2">
                <Button size="sm" onClick={() => fileInput.current?.click()}><Upload size={15} /> Another photo</Button>
                <Button size="sm" onClick={() => cameraInput.current?.click()}><Camera size={15} /> Take a photo</Button>
              </div>
            </>
          ) : picker}

          <section aria-labelledby="gallery-h">
            <div className="mb-2 flex items-baseline justify-between">
              <Eyebrow id="gallery-h">Sample gallery: one tap to try</Eyebrow>
              <span className="text-[11px] text-muted">Construction-PPE, CC BY 4.0</span>
            </div>
            <div className="flex gap-2 overflow-x-auto pb-2">
              {samples.map((s) => (
                <button
                  key={s.id} type="button" onClick={() => void openSample(s)}
                  className={cn('group relative h-24 w-36 shrink-0 overflow-hidden rounded border', sample?.id === s.id ? 'border-hivis' : 'border-line')}
                >
                  <img src={asset(s.file.replace('.jpg', '_thumb.jpg'))} alt={s.title} loading="lazy" className="h-full w-full object-cover transition-transform group-hover:scale-105" />
                  <span className="absolute inset-x-0 bottom-0 truncate bg-[#0b0d10]/80 px-1.5 py-0.5 text-left text-[11px] text-white">{s.title}</span>
                  {s.failure && <span className="absolute left-1 top-1 rounded bg-hazard px-1 font-mono text-[9px] font-bold uppercase text-[#0b0d10]">failure case</span>}
                </button>
              ))}
              {!samples.length && <p className="text-sm text-muted">Samples are loading…</p>}
            </div>
          </section>

          {analysis && analysis.people.length > 0 && (
            <section aria-labelledby="workers-h">
              <Eyebrow id="workers-h" className="mb-2">Workers (anonymous numbers, reset for every photo)</Eyebrow>
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                {analysis.people.map((person, i) => (
                  <WorkerCard
                    key={i} person={person} id={i + 1} required={required} active={hovered === i}
                    onHover={(on) => setHovered(on ? i : null)}
                    alert={analysis.events[i].zones.length ? 'Inside a danger zone' : analysis.events[i].nearForklift ? 'Close to a forklift' : null}
                  />
                ))}
              </div>
            </section>
          )}
        </div>

        <aside className="space-y-4">
          <SummaryPanel summary={analysis?.summary ?? null} zoneAlerts={analysis?.zoneAlerts ?? 0} forkliftAlerts={analysis?.forkliftAlerts ?? 0} required={required} />
          <Panel className="p-4"><ProfilePicker /></Panel>
          <Panel className="p-4">
            <Eyebrow>Site Report Studio</Eyebrow>
            <p className="mt-1 text-xs text-muted">Made on this device. Aggregate counts and anonymous numbers only.</p>
            <div className="mt-3 grid grid-cols-2 gap-2">
              <Button variant="primary" disabled={!analysis} onClick={() => void exportPdf()}><FileText size={16} /> PDF report</Button>
              <Button disabled={!analysis} onClick={exportCsv}><Table size={16} /> CSV</Button>
              <Button className="col-span-2" disabled={!analysis} onClick={exportImage}><Download size={16} /> Blurred image</Button>
            </div>
            <div className="mt-4 border-t border-line pt-3">
              <div className="flex items-center justify-between gap-2">
                <span className="text-sm font-medium">Written summary</span>
                <Chip>optional · ~200 MB model</Chip>
              </div>
              <p className="mt-1 text-xs text-muted">A small open language model describes the scene. Every claim is checked against the detector before you see it.</p>
              <Button className="mt-2 w-full" disabled={!analysis || !!nar.busy} onClick={() => void summarise()}><Sparkles size={16} /> Generate summary</Button>
              <Narrative state={nar} />
            </div>
          </Panel>
          <Panel className="p-4">
            <Eyebrow>Settings</Eyebrow>
            <DetectionSettings />
          </Panel>
        </aside>
      </div>
    </div>
  );
}

export default function Analyze() {
  return <EngineGate><Workbench /></EngineGate>;
}
