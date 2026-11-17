import { useCallback, useEffect, useRef, useState } from 'react';
import { Camera, CameraOff, PenTool, Settings2, SwitchCamera, Trash2, Volume2, VolumeX, X } from 'lucide-react';
import { EngineGate } from '../components/EngineStatus';
import { DetectionSettings, ProfilePicker } from '../components/Panels';
import { Gauge } from '../components/Ppe';
import { Stage } from '../components/Stage';
import { Button, cn, PageHeader, Panel } from '../components/ui';
import { analyse } from '../lib/analyze';
import { blurRegions, drawPeople, drawZones } from '../lib/draw';
import { engine, useEngine } from '../lib/engine';
import { beep, toCanvas } from '../lib/media';
import { ClipSession, type FrameRecord } from '../lib/session';
import { useLog, useSettings } from '../lib/store';
import type { Zone } from '../lib/zones';

function LiveHud() {
  const st = useSettings();
  const backend = useEngine((s) => s.backend);
  const addLog = useLog((s) => s.add);
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const session = useRef<ClipSession | null>(null);
  const latest = useRef<FrameRecord | null>(null);
  const running = useRef(false);
  const zonesRef = useRef<Zone[]>([]);
  const started = useRef(0);
  const [on, setOn] = useState(false);
  const [facing, setFacing] = useState<'environment' | 'user'>('environment');
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState({ fps: 0, ms: 0, pct: null as number | null, workers: 0, violations: 0, unsure: 0, alerts: 0 });
  const [zones, setZones] = useState<Zone[]>([]);
  const [freeze, setFreeze] = useState<HTMLCanvasElement | null>(null);
  const [panel, setPanel] = useState(false);
  zonesRef.current = zones;

  const stop = useCallback((log = true) => {
    running.current = false;
    const v = video.current;
    (v?.srcObject as MediaStream | null)?.getTracks().forEach((t) => t.stop());
    if (v) v.srcObject = null;
    const s = session.current;
    if (log && s && s.frames.length + s.workers.size > 0 && s.stableWorkers().length) {
      addLog({
        id: `live-${started.current}`, time: Date.now(), source: 'live', label: 'Live camera session', profile: s.profile,
        summary: s.summary(), zoneEvents: s.zoneEvents(), forkliftEvents: s.forkliftEvents(), ms: latest.current?.ms ?? 0,
        workers: s.stableWorkers().map((w) => ({ id: w.id, missing: [...w.missing], status: s.workerStatus(w) })),
      });
    }
    session.current = null;
    setOn(false);
  }, [addLog]);

  const start = useCallback(async (mode: 'environment' | 'user') => {
    setError(null);
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('This browser has no camera access. Use the Photo tab instead.');
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: mode }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
      });
      const v = video.current!;
      v.srcObject = stream;
      await v.play();
      session.current = new ClipSession(useSettings.getState().profile, { window: 6, minHits: 3, maxMisses: 10 });
      latest.current = null;
      started.current = Date.now();
      running.current = true;
      setOn(true);

      // inference loop: as fast as the device allows, never queued
      let frames = 0;
      let t0 = performance.now();
      const infer = async () => {
        while (running.current) {
          if (v.readyState < 2 || document.hidden) { await new Promise((r) => setTimeout(r, 120)); continue; }
          try {
            const s = useSettings.getState();
            const bitmap = await createImageBitmap(v);
            const res = await engine.detect(bitmap, { conf: s.conf, faces: s.blurFaces });
            if (!running.current || !session.current) break;
            session.current.profile = s.profile;
            const a = analyse(res, s.profile, zonesRef.current, { strictBlur: s.strictBlur, forkliftDistance: s.forkliftDistance });
            const rec = session.current.step(a, (Date.now() - started.current) / 1000, false);
            latest.current = rec;
            if (session.current.newViolation && s.sound) beep('violation');
            frames += 1;
            const now = performance.now();
            if (now - t0 > 700) {
              const sum = session.current.summary();
              const visibleDecided = rec.people.filter((p) => p.status !== 'unsure').length;
              setStats({
                fps: (frames * 1000) / (now - t0), ms: res.ms.total,
                pct: visibleDecided ? Math.round((100 * rec.people.filter((p) => p.status === 'compliant').length) / visibleDecided) : null,
                workers: sum.persons, violations: sum.nonCompliant, unsure: rec.people.filter((p) => p.status === 'unsure').length, alerts: rec.alerts,
              });
              frames = 0; t0 = now;
            }
          } catch (e) {
            setError(e instanceof Error ? e.message : 'Detection failed');
            await new Promise((r) => setTimeout(r, 500));
          }
        }
      };
      void infer();

      // render loop: every display frame, using the latest detections
      const draw = () => {
        if (!running.current) return;
        const c = canvas.current;
        if (c && v.videoWidth) {
          if (c.width !== v.videoWidth) { c.width = v.videoWidth; c.height = v.videoHeight; }
          const ctx = c.getContext('2d')!;
          ctx.drawImage(v, 0, 0);
          const rec = latest.current;
          const s = useSettings.getState();
          if (rec) {
            // wider margin than for photos: faces move between two detections
            if (s.blurFaces) blurRegions(ctx, rec.blur, 0.35);
            const scale = Math.max(1, c.width / 800);
            drawZones(ctx, zonesRef.current, new Set(rec.activeZones), scale);
            drawPeople(ctx, rec.people, scale);
          } else if (s.blurFaces) {
            // nothing detected yet: show nothing identifiable until the first result arrives
            ctx.fillStyle = '#0b0d10';
            ctx.fillRect(0, 0, c.width, c.height);
          }
        }
        requestAnimationFrame(draw);
      };
      requestAnimationFrame(draw);
    } catch (e) {
      const name = e instanceof DOMException ? e.name : '';
      setError(name === 'NotAllowedError' ? 'Camera permission was refused. Allow it in the browser address bar, or use the Photo tab.'
        : name === 'NotFoundError' ? 'No camera was found on this device.' : e instanceof Error ? e.message : 'The camera could not start.');
      setOn(false);
    }
  }, []);

  useEffect(() => () => stop(true), [stop]);

  const flip = () => {
    const next = facing === 'environment' ? 'user' : 'environment';
    setFacing(next);
    stop(false);
    void start(next);
  };

  const markZone = () => {
    const v = video.current;
    if (!v?.videoWidth) return;
    // freeze a blurred frame to draw on
    const c = toCanvas(v, v.videoWidth, v.videoHeight, 4000);
    if (latest.current && st.blurFaces) blurRegions(c.getContext('2d')!, latest.current.blur, 0.35);
    setFreeze(c);
  };

  return (
    <div className="mx-auto max-w-7xl px-4 py-6">
      {!on && (
        <>
          <PageHeader eyebrow="Live camera" title="Point the camera at the work area">
            Detection runs on this device in real time. The video is never recorded, uploaded or stored, and faces are blurred on screen.
          </PageHeader>
          {error && <p role="alert" className="mb-4 rounded border border-danger/50 bg-danger/10 px-3 py-2 text-sm text-danger">{error}</p>}
          <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_360px]">
            <Panel className="hud-corners grid place-items-center px-4 py-16 text-center">
              <div>
                <div className="mx-auto grid h-14 w-14 place-items-center rounded-full bg-hivis text-[#0b0d10]"><Camera size={26} /></div>
                <h2 className="mt-4 text-lg font-semibold">Start the live check</h2>
                <p className="mx-auto mt-1 max-w-sm text-sm text-muted">
                  Tell the people in view what this is for before you start: a PPE check with aggregate counts only, no recording and no names.
                </p>
                <Button variant="primary" size="lg" className="mt-5" onClick={() => void start(facing)}><Camera size={18} /> Use my camera</Button>
              </div>
            </Panel>
            <div className="space-y-4">
              <Panel className="p-4"><ProfilePicker /></Panel>
              <Panel className="p-4"><DetectionSettings showSound /></Panel>
            </div>
          </div>
        </>
      )}

      <video ref={video} playsInline muted className="hidden" />

      {/* full-screen HUD */}
      <div className={cn('fixed inset-0 z-50 bg-black', on ? 'block' : 'hidden')}>
        <canvas ref={canvas} className="absolute inset-0 h-full w-full object-contain" aria-label="Live camera with PPE overlay" />
        {/* corner brackets */}
        {['left-3 top-3 border-l-2 border-t-2', 'right-3 top-3 border-r-2 border-t-2', 'bottom-3 left-3 border-b-2 border-l-2', 'bottom-3 right-3 border-b-2 border-r-2'].map((c) => (
          <span key={c} aria-hidden className={cn('pointer-events-none absolute h-8 w-8 border-hivis', c)} />
        ))}
        <div className="pointer-events-none absolute left-5 top-5 space-y-1 font-mono text-[11px] text-white [text-shadow:0_1px_2px_#000]">
          <div className="flex items-center gap-2"><span className="h-2 w-2 animate-pulse rounded-full bg-danger" /> LIVE · NOT RECORDED</div>
          <div><span className="text-hivis">{stats.fps.toFixed(1)}</span> fps · <span className="text-hivis">{stats.ms.toFixed(0)}</span> ms · {backend === 'webgpu' ? 'WebGPU' : 'WASM'}</div>
          <div>workers seen <span className="text-hivis">{stats.workers}</span> · violations <span className="text-hivis">{stats.violations}</span></div>
          {stats.alerts > 0 && <div className="animate-hazard font-bold text-[#ff7a1a]">⚠ {stats.alerts} in danger zone / near forklift</div>}
        </div>
        <div className="absolute right-4 top-4 flex gap-2">
          <button type="button" aria-label={st.sound ? 'Mute sound cues' : 'Enable sound cues'} onClick={() => st.set({ sound: !st.sound })}
            className="grid h-11 w-11 place-items-center rounded-full bg-[#0b0d10]/75 text-white">{st.sound ? <Volume2 size={19} /> : <VolumeX size={19} />}</button>
          <button type="button" aria-label="Switch camera" onClick={flip} className="grid h-11 w-11 place-items-center rounded-full bg-[#0b0d10]/75 text-white"><SwitchCamera size={19} /></button>
          <button type="button" aria-label="Settings" onClick={() => setPanel((p) => !p)} className="grid h-11 w-11 place-items-center rounded-full bg-[#0b0d10]/75 text-white"><Settings2 size={19} /></button>
        </div>
        <div className="pointer-events-none absolute bottom-5 left-5 rounded-lg bg-[#0b0d10]/75 px-3 pb-2 pt-3 text-white backdrop-blur">
          <Gauge pct={stats.pct} size={150} label="in view now" />
          {stats.unsure > 0 && <div className="mt-1 text-center font-mono text-[10px] text-[#5bc0ff]">{stats.unsure} to check manually</div>}
        </div>
        <div className="absolute bottom-5 right-4 flex flex-col items-end gap-2">
          {zones.length > 0 && <Button size="sm" className="border-0 bg-[#0b0d10]/75 text-white hover:bg-[#0b0d10]" onClick={() => setZones([])}><Trash2 size={15} /> Clear zones</Button>}
          <Button size="sm" className="border-0 bg-[#0b0d10]/75 text-white hover:bg-[#0b0d10]" onClick={markZone}><PenTool size={15} /> Mark danger zone</Button>
          <Button variant="primary" onClick={() => stop(true)}><CameraOff size={17} /> Stop</Button>
        </div>
        {error && on && <p role="alert" className="absolute inset-x-0 top-20 mx-auto w-fit rounded bg-danger px-3 py-1.5 text-sm text-white">{error}</p>}
        {panel && (
          <div className="absolute right-4 top-20 w-[min(92vw,340px)] rounded-lg border border-line bg-surface p-4 shadow-2xl">
            <div className="mb-2 flex items-center justify-between"><span className="eyebrow">Live settings</span>
              <button type="button" aria-label="Close settings" onClick={() => setPanel(false)}><X size={17} /></button></div>
            <ProfilePicker />
            <div className="mt-2"><DetectionSettings showSound /></div>
          </div>
        )}
        {freeze && (
          <div className="absolute inset-0 z-10 grid place-items-center bg-[#0b0d10]/90 p-4">
            <div className="w-full max-w-4xl">
              <div className="mb-2 flex items-center justify-between text-white">
                <span className="font-display text-lg font-semibold">Mark a danger zone on this frame</span>
                <Button size="sm" variant="primary" onClick={() => setFreeze(null)}>Done</Button>
              </div>
              <Stage source={freeze} analysis={null} scanning={false} profile={st.profile} blur={false} compare={false} hovered={null}
                onHover={() => undefined} zones={zones} onZones={setZones} drawing onDrawingDone={() => undefined} animateKey={0} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default function Live() {
  return <EngineGate><LiveHud /></EngineGate>;
}
