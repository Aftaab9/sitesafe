import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { FileText, Film, Pause, PenTool, Play, RotateCcw, Table, Trash2, TriangleAlert, Upload } from 'lucide-react';
import { EngineGate } from '../components/EngineStatus';
import { ProfilePicker, SummaryPanel } from '../components/Panels';
import { STATUS_ICON, STATUS_TEXT } from '../components/Ppe';
import { Stage } from '../components/Stage';
import { Button, cn, Eyebrow, PageHeader, Panel, Segmented } from '../components/ui';
import { analyse } from '../lib/analyze';
import { ITEM_LABEL, SITE_PROFILES } from '../lib/classes';
import { blurRegions, drawPeople, drawZones, STATUS_LABEL } from '../lib/draw';
import { asset, engine } from '../lib/engine';
import { download, loadCatalogue, toCanvas, type ClipItem } from '../lib/media';
import { buildPdf, entryCsv } from '../lib/report';
import { ClipSession, type FrameRecord } from '../lib/session';
import { useLog, useSettings, type LogEntry } from '../lib/store';
import type { Zone } from '../lib/zones';

const seek = (v: HTMLVideoElement, t: number) => new Promise<void>((resolve) => {
  const done = () => { v.removeEventListener('seeked', done); resolve(); };
  v.addEventListener('seeked', done);
  v.currentTime = t;
});

type Phase = 'pick' | 'setup' | 'processing' | 'done';

function ClipBench() {
  const st = useSettings();
  const addLog = useLog((s) => s.add);
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const cancel = useRef(false);
  const framesRef = useRef<FrameRecord[]>([]);
  const zonesRef = useRef<Zone[]>([]);
  const [clips, setClips] = useState<ClipItem[]>([]);
  const [phase, setPhase] = useState<Phase>('pick');
  const [url, setUrl] = useState<string | null>(null);
  const [name, setName] = useState('');
  const [firstFrame, setFirstFrame] = useState<HTMLCanvasElement | null>(null);
  const [duration, setDuration] = useState(0);
  const [fps, setFps] = useState<'2' | '3' | '5'>('3');
  const [smoothing, setSmoothing] = useState(true);
  const [zones, setZones] = useState<Zone[]>([]);
  const [drawing, setDrawing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [session, setSession] = useState<ClipSession | null>(null);
  const [clipMs, setClipMs] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [error, setError] = useState<string | null>(null);
  zonesRef.current = zones;

  useEffect(() => { loadCatalogue().then((c) => setClips(c.clips)).catch(() => undefined); }, []);
  useEffect(() => () => { cancel.current = true; }, []);

  const open = useCallback(async (src: string, label: string) => {
    setError(null); setSession(null); setZones([]); setPhase('pick'); setName(label); setUrl(src);
    const v = video.current!;
    v.src = src;
    try {
      await new Promise<void>((res, rej) => { v.onloadedmetadata = () => res(); v.onerror = () => rej(new Error('This video format is not supported by your browser. MP4 (H.264) works everywhere.')); });
      if (!Number.isFinite(v.duration) || v.duration <= 0) throw new Error('Could not read the clip length.');
      setDuration(v.duration);
      await seek(v, Math.min(0.1, v.duration / 2));
      setFirstFrame(toCanvas(v, v.videoWidth, v.videoHeight, 1280));
      setPhase('setup');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'The clip could not be opened.');
    }
  }, []);

  const openFile = (f?: File) => {
    if (!f) return;
    if (!f.type.startsWith('video/')) { setError('That file is not a video. For photos, use the Photo tab.'); return; }
    if (url?.startsWith('blob:')) URL.revokeObjectURL(url);
    void open(URL.createObjectURL(f), f.name);
  };

  const paint = useCallback((rec: FrameRecord | null) => {
    const v = video.current!;
    const c = canvas.current;
    if (!c || !v.videoWidth) return;
    const r = Math.min(1, 1280 / Math.max(v.videoWidth, v.videoHeight));
    const w = Math.round(v.videoWidth * r);
    const h = Math.round(v.videoHeight * r);
    if (c.width !== w) { c.width = w; c.height = h; }
    const ctx = c.getContext('2d')!;
    ctx.drawImage(v, 0, 0, w, h);
    if (!rec) return;
    if (useSettings.getState().blurFaces) blurRegions(ctx, rec.blur, 0.3);
    const scale = Math.max(1, w / 800);
    drawZones(ctx, zonesRef.current, new Set(rec.activeZones), scale);
    drawPeople(ctx, rec.people, scale);
  }, []);

  const process = async () => {
    const v = video.current!;
    cancel.current = false;
    setPhase('processing'); setProgress(0); setPlaying(false);
    v.pause();
    const s = useSettings.getState();
    const step = 1 / Number(fps);
    // Without smoothing: a window of 1 raises a violation on any single frame (the comparison arm).
    const ses = new ClipSession(s.profile, smoothing ? { window: 5, minHits: 3 } : { window: 1, minHits: 1 });
    framesRef.current = ses.frames;
    const t0 = performance.now();
    const end = Math.min(v.duration, 60);
    const work = document.createElement('canvas');
    try {
      for (let t = 0; t < end; t += step) {
        if (cancel.current) return;
        await seek(v, t);
        const r = Math.min(1, 1280 / Math.max(v.videoWidth, v.videoHeight));
        work.width = Math.round(v.videoWidth * r);
        work.height = Math.round(v.videoHeight * r);
        work.getContext('2d')!.drawImage(v, 0, 0, work.width, work.height);
        const res = await engine.detect(await createImageBitmap(work), { conf: s.conf, faces: s.blurFaces });
        const rec = ses.step(analyse(res, s.profile, zonesRef.current, { strictBlur: s.strictBlur, forkliftDistance: s.forkliftDistance }), t);
        paint(rec);
        setProgress(Math.min(100, Math.round((100 * (t + step)) / end)));
      }
      const total = performance.now() - t0;
      setClipMs(total);
      setSession(ses);
      setPhase('done');
      await seek(v, 0);
      setTime(0);
      paint(ses.frames[0] ?? null);
      const entry = makeEntry(ses, total);
      addLog(entry);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Processing failed');
      setPhase('setup');
    }
  };

  const makeEntry = useCallback((ses: ClipSession, total: number): LogEntry => ({
    id: `clip-${Date.now()}`, time: Date.now(), source: 'video', label: name, profile: ses.profile, summary: ses.summary(),
    zoneEvents: ses.zoneEvents(), forkliftEvents: ses.forkliftEvents(), ms: total,
    timeline: ses.frames.map((f) => ({ t: f.t, persons: f.persons, nonCompliant: f.nonCompliant })),
    workers: ses.stableWorkers().map((w) => ({ id: w.id, missing: [...w.missing], status: ses.workerStatus(w) })),
  }), [name]);

  const recordAt = useCallback((t: number): FrameRecord | null => {
    const fr = framesRef.current;
    if (!fr.length) return null;
    let best = fr[0];
    for (const f of fr) { if (f.t <= t + 1e-3) best = f; else break; }
    return best;
  }, []);

  // playback loop
  useEffect(() => {
    if (phase !== 'done') return;
    const v = video.current!;
    let raf = 0;
    const tick = () => {
      paint(recordAt(v.currentTime));
      setTime(v.currentTime);
      if (!v.paused && !v.ended) raf = requestAnimationFrame(tick);
      else setPlaying(false);
    };
    if (playing) { void v.play().then(() => { raf = requestAnimationFrame(tick); }).catch(() => setPlaying(false)); }
    else v.pause();
    return () => cancelAnimationFrame(raf);
  }, [playing, phase, paint, recordAt]);

  const jump = async (t: number) => {
    const v = video.current!;
    setPlaying(false);
    await seek(v, Math.max(0, Math.min(t, v.duration - 0.05)));
    setTime(v.currentTime);
    paint(recordAt(v.currentTime));
  };

  const summary = useMemo(() => session?.summary() ?? null, [session]);
  const required = SITE_PROFILES[st.profile].required;
  const frames = session?.frames ?? [];
  const rawFlagged = frames.filter((f) => f.rawNonCompliant > 0).length;
  const smoothFlagged = frames.filter((f) => f.nonCompliant > 0).length;

  const exportPdf = async () => {
    if (!session) return;
    const snap = document.createElement('canvas');
    snap.width = canvas.current!.width; snap.height = canvas.current!.height;
    snap.getContext('2d')!.drawImage(canvas.current!, 0, 0);
    download('sitesafe-clip-report.pdf', await buildPdf({ entry: makeEntry(session, clipMs), image: snap }));
  };
  const exportCsv = () => {
    if (!session) return;
    const rows = ['t_seconds,persons,non_compliant_smoothed,non_compliant_raw,zone_or_forklift_alerts,ms',
      ...session.frames.map((f) => [f.t.toFixed(2), f.persons, f.nonCompliant, f.rawNonCompliant, f.alerts, f.ms.toFixed(0)].join(','))];
    download('sitesafe-clip-timeline.csv', new Blob([entryCsv([makeEntry(session, clipMs)]) + '\n' + rows.join('\n') + '\n'], { type: 'text/csv' }));
  };

  return (
    <div className="mx-auto max-w-7xl px-4 py-6">
      <PageHeader eyebrow="Clip check" title="Check a 10 to 30 second clip">
        Frames are sampled a few times per second. Each worker keeps an anonymous number across the clip, and a violation is only raised when it persists over several frames.
      </PageHeader>
      <video ref={video} playsInline muted preload="auto" className="hidden" crossOrigin="anonymous" />
      <input ref={fileInput} type="file" accept="video/*" className="hidden" onChange={(e) => { openFile(e.target.files?.[0]); e.target.value = ''; }} />
      {error && <p role="alert" className="mb-4 flex items-center gap-2 rounded border border-danger/50 bg-danger/10 px-3 py-2 text-sm text-danger"><TriangleAlert size={16} /> {error}</p>}

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_360px]">
        <div className="min-w-0 space-y-4">
          {phase === 'pick' && (
            <Panel className="hud-corners grid place-items-center border-dashed px-4 py-12 text-center">
              <div>
                <div className="mx-auto grid h-14 w-14 place-items-center rounded-full bg-hivis text-[#0b0d10]"><Film size={26} /></div>
                <h2 className="mt-4 text-lg font-semibold">Choose a short clip</h2>
                <p className="mx-auto mt-1 max-w-sm text-sm text-muted">10 to 30 seconds works best (up to 60 s is processed). The clip stays on this device.</p>
                <Button variant="primary" className="mt-5" onClick={() => fileInput.current?.click()}><Upload size={16} /> Choose a clip</Button>
              </div>
            </Panel>
          )}

          {phase === 'setup' && firstFrame && (
            <>
              <Stage source={firstFrame} analysis={null} scanning={false} profile={st.profile} blur={false} compare={false} hovered={null}
                onHover={() => undefined} zones={zones} onZones={setZones} drawing={drawing} onDrawingDone={() => setDrawing(false)} animateKey={0} />
              <div className="flex flex-wrap items-center gap-2">
                <Button variant="primary" onClick={() => void process()}><Play size={16} /> Analyse clip ({duration.toFixed(0)} s)</Button>
                <Button variant={drawing ? 'primary' : 'outline'} aria-pressed={drawing} onClick={() => setDrawing((d) => !d)}><PenTool size={15} /> Draw danger zone</Button>
                {zones.length > 0 && <Button variant="ghost" onClick={() => setZones([])}><Trash2 size={15} /> Clear zones</Button>}
                <Button variant="ghost" onClick={() => fileInput.current?.click()}><Upload size={15} /> Other clip</Button>
              </div>
              {duration > 30 && <p className="text-xs text-muted">This clip is {duration.toFixed(0)} s long. It will work, but 10 to 30 s is the intended length{duration > 60 ? '; only the first 60 s are analysed' : ''}.</p>}
            </>
          )}

          <div className={cn('space-y-3', phase === 'processing' || phase === 'done' ? 'block' : 'hidden')}>
            <div className="hud-corners relative overflow-hidden rounded bg-black">
              <canvas ref={canvas} className="block h-auto w-full" aria-label="Clip with PPE overlay and blurred faces" />
              {phase === 'processing' && <div className="scan-line" aria-hidden />}
            </div>
            {phase === 'processing' && (
              <div role="status">
                <div className="h-3 overflow-hidden rounded-full border border-line bg-raised"><div className="h-full hazard-stripes animate-stripes" style={{ width: `${progress}%` }} /></div>
                <div className="mt-1.5 flex justify-between font-mono text-xs text-muted">
                  <span>Analysing frame by frame on this device… {progress}%</span>
                  <button type="button" className="underline" onClick={() => { cancel.current = true; setPhase('setup'); }}>Cancel</button>
                </div>
              </div>
            )}
            {phase === 'done' && session && (
              <>
                <div className="flex items-center gap-3">
                  <Button variant="primary" aria-label={playing ? 'Pause' : 'Play'} onClick={() => setPlaying((p) => !p)}>{playing ? <Pause size={16} /> : <Play size={16} />}</Button>
                  <span className="num w-24 text-xs text-muted">{time.toFixed(1)} / {duration.toFixed(1)} s</span>
                  <Button size="sm" variant="ghost" onClick={() => setPhase('setup')}><RotateCcw size={15} /> Re-run</Button>
                  <Button size="sm" variant="ghost" onClick={() => fileInput.current?.click()}><Upload size={15} /> Other clip</Button>
                </div>
                {/* timeline with violation markers */}
                <div>
                  <div className="relative h-9 rounded border border-line bg-raised">
                    {frames.map((f, i) => {
                      const next = frames[i + 1]?.t ?? duration;
                      const cls = f.nonCompliant > 0 ? 'bg-danger' : f.alerts > 0 ? 'bg-hazard' : f.persons > 0 ? 'bg-ok/50' : 'bg-transparent';
                      return (
                        <button
                          key={i} type="button" onClick={() => void jump(f.t)}
                          title={`${f.t.toFixed(1)} s: ${f.nonCompliant} with PPE missing, ${f.alerts} zone alerts`}
                          aria-label={`Jump to ${f.t.toFixed(1)} seconds${f.nonCompliant ? ', violation' : ''}`}
                          className={cn('absolute inset-y-1 rounded-[2px] hover:brightness-125', cls, f.nonCompliant > 0 && 'hazard-stripes-soft')}
                          style={{ left: `${(100 * f.t) / duration}%`, width: `${Math.max(0.6, (100 * (next - f.t)) / duration - 0.15)}%` }}
                        />
                      );
                    })}
                    <span className="pointer-events-none absolute inset-y-0 w-0.5 bg-hivis" style={{ left: `${(100 * time) / duration}%` }} />
                  </div>
                  <input type="range" min={0} max={duration} step={0.05} value={time} aria-label="Scrub clip" onChange={(e) => void jump(Number(e.target.value))}
                    className="mt-1 w-full accent-[rgb(var(--hivis))]" />
                  <div className="flex flex-wrap gap-x-4 gap-y-1 font-mono text-[11px] text-muted">
                    <span><span className="mr-1 inline-block h-2 w-3 rounded-[2px] bg-danger" />PPE missing</span>
                    <span><span className="mr-1 inline-block h-2 w-3 rounded-[2px] bg-hazard" />zone / forklift</span>
                    <span><span className="mr-1 inline-block h-2 w-3 rounded-[2px] bg-ok/50" />no violation</span>
                    <span className="ml-auto">{frames.length} frames · {(clipMs / 1000).toFixed(1)} s total · {(clipMs / Math.max(1, frames.length)).toFixed(0)} ms/frame</span>
                  </div>
                </div>
                <Panel className="p-4">
                  <Eyebrow>Anonymous worker log</Eyebrow>
                  <ul className="mt-2 grid gap-1.5 sm:grid-cols-2">
                    {session.stableWorkers().map((w) => {
                      const s = session.workerStatus(w);
                      const Icon = STATUS_ICON[s];
                      return (
                        <li key={w.id} className="flex items-center gap-2 rounded border border-line px-2.5 py-1.5 text-sm">
                          <span className="font-mono text-xs text-muted">W{String(w.id).padStart(2, '0')}</span>
                          <span className={cn('flex items-center gap-1 font-medium', STATUS_TEXT[s])}><Icon size={14} /> {STATUS_LABEL[s]}</span>
                          <span className="ml-auto truncate text-xs text-muted">
                            {[...w.missing].map((m) => ITEM_LABEL[m].toLowerCase()).join(', ')}{w.zone ? ' · zone' : ''}{w.forklift ? ' · forklift' : ''}
                          </span>
                        </li>
                      );
                    })}
                    {!session.stableWorkers().length && <li className="text-sm text-muted">No worker was tracked for long enough.</li>}
                  </ul>
                  <p className="mt-3 text-xs text-muted">
                    Temporal smoothing {smoothing ? 'on' : 'off'}: {rawFlagged} of {frames.length} frames had a single-frame violation;
                    {' '}{smoothFlagged} are reported{smoothing ? ' after requiring 3 of the last 5 frames' : ''}.
                  </p>
                </Panel>
              </>
            )}
          </div>

          {clips.length > 0 && (
            <section>
              <Eyebrow className="mb-2">Sample clips</Eyebrow>
              <div className="flex gap-2 overflow-x-auto pb-2">
                {clips.map((c) => (
                  <button key={c.id} type="button" onClick={() => void open(asset(c.file), c.title)} className="group relative h-24 w-40 shrink-0 overflow-hidden rounded border border-line" title={c.note}>
                    <img src={asset(c.poster)} alt={c.title} className="h-full w-full object-cover" />
                    <span className="absolute inset-0 grid place-items-center bg-[#0b0d10]/30 text-white opacity-90 group-hover:bg-[#0b0d10]/10"><Play size={22} /></span>
                    <span className="absolute inset-x-0 bottom-0 truncate bg-[#0b0d10]/80 px-1.5 py-0.5 text-left text-[11px] text-white">{c.title}</span>
                  </button>
                ))}
              </div>
              <p className="text-[11px] text-muted">Sample clips are simulated camera moves over still images from Construction-PPE (CC BY 4.0). Upload your own clip for real motion.</p>
            </section>
          )}
        </div>

        <aside className="space-y-4">
          <SummaryPanel summary={summary} zoneAlerts={session?.zoneEvents() ?? 0} forkliftAlerts={session?.forkliftEvents() ?? 0} required={required} title="Whole clip, per worker" />
          <Panel className="space-y-4 p-4">
            <ProfilePicker />
            <div>
              <Eyebrow className="mb-1.5">Frames analysed per second</Eyebrow>
              <Segmented label="Sampling rate" value={fps} onChange={setFps} options={[{ value: '2', label: '2 fps' }, { value: '3', label: '3 fps' }, { value: '5', label: '5 fps' }]} />
            </div>
            <div>
              <Eyebrow className="mb-1.5">Temporal smoothing</Eyebrow>
              <Segmented label="Temporal smoothing" value={smoothing ? 'on' : 'off'} onChange={(v) => setSmoothing(v === 'on')}
                options={[{ value: 'on', label: 'On (3 of 5 frames)' }, { value: 'off', label: 'Off' }]} />
              <p className="mt-1.5 text-xs text-muted">Applies on the next run. Smoothing removes single-frame false alarms.</p>
            </div>
          </Panel>
          <Panel className="p-4">
            <Eyebrow>Site Report Studio</Eyebrow>
            <div className="mt-3 grid grid-cols-2 gap-2">
              <Button variant="primary" disabled={!session} onClick={() => void exportPdf()}><FileText size={16} /> PDF report</Button>
              <Button disabled={!session} onClick={exportCsv}><Table size={16} /> CSV</Button>
            </div>
          </Panel>
        </aside>
      </div>
    </div>
  );
}

export default function Video() {
  return <EngineGate><ClipBench /></EngineGate>;
}
