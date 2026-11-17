// Inspector Challenge: human vs AI on the same photos. Doubles as a small human-baseline study:
// every round is logged locally and can be exported as CSV for the report.
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Bot, Download, Play, RotateCcw, Timer, Trophy, User } from 'lucide-react';
import { EngineGate } from '../components/EngineStatus';
import { Button, cn, Eyebrow, PageHeader, Panel } from '../components/ui';
import { analyse, MATCH_CONFIG } from '../lib/analyze';
import { SITE_PROFILES } from '../lib/classes';
import { blurRegions } from '../lib/draw';
import { asset, engine } from '../lib/engine';
import { iou, matchPeople, type PersonResult } from '../lib/matching';
import { download, loadCatalogue, urlToCanvas, type Sample } from '../lib/media';
import { useSettings } from '../lib/store';

const ROUNDS = 5;
const SECONDS = 12;
const KEY = 'sitesafe-challenge-v1';
const REQUIRED = SITE_PROFILES.general.required;

interface RoundLog { sample: string; persons: number; human: number; ai: number; aiUnsure: number; seconds: number }
interface Score { date: string; human: number; ai: number; total: number; seconds: number }

function loadScores(): Score[] { try { return JSON.parse(localStorage.getItem(KEY) || '[]'); } catch { return []; } }

interface RoundData {
  sample: Sample;
  canvas: HTMLCanvasElement;   // faces already blurred
  truth: PersonResult[];       // ground truth, only decidable people
  ai: ('flag' | 'ok' | 'unsure')[];
}

async function prepare(sample: Sample): Promise<RoundData> {
  const canvas = await urlToCanvas(asset(sample.file));
  const sx = canvas.width / sample.w;
  const gt = sample.gt.map((d) => ({ ...d, conf: 1, box: d.box.map((v) => v * sx) as typeof d.box }));
  const truth = matchPeople(gt, REQUIRED, MATCH_CONFIG).filter((p) => p.status !== 'unsure');
  const res = await engine.detect(await createImageBitmap(canvas), { conf: useSettings.getState().conf });
  const a = analyse(res, 'general', [], { strictBlur: useSettings.getState().strictBlur, forkliftDistance: 0.5 });
  blurRegions(canvas.getContext('2d')!, a.blur);
  const ai = truth.map((t) => {
    let best: PersonResult | null = null;
    let bestIou = 0.5;
    for (const p of a.people) { const v = iou(p.box, t.box); if (v >= bestIou) { best = p; bestIou = v; } }
    if (!best || best.status === 'unsure') return 'unsure' as const;
    return best.status === 'non_compliant' ? 'flag' as const : 'ok' as const;
  });
  return { sample, canvas, truth, ai };
}

function Game() {
  const [pool, setPool] = useState<Sample[]>([]);
  const [phase, setPhase] = useState<'intro' | 'loading' | 'play' | 'reveal' | 'end'>('intro');
  const [round, setRound] = useState(0);
  const [data, setData] = useState<RoundData | null>(null);
  const [flags, setFlags] = useState<Set<number>>(new Set());
  const [left, setLeft] = useState(SECONDS);
  const [log, setLog] = useState<RoundLog[]>([]);
  const [scores, setScores] = useState<Score[]>(loadScores);
  const order = useRef<Sample[]>([]);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const startedAt = useRef(0);

  useEffect(() => {
    loadCatalogue().then((c) => setPool(c.samples.filter((s) => {
      const t = matchPeople(s.gt.map((d) => ({ ...d, conf: 1 })), REQUIRED, MATCH_CONFIG).filter((p) => p.status !== 'unsure');
      return t.length >= 1;
    }))).catch(() => undefined);
  }, []);

  const loadRound = useCallback(async (i: number) => {
    setPhase('loading'); setFlags(new Set()); setLeft(SECONDS);
    const d = await prepare(order.current[i]);
    setData(d); setRound(i); setPhase('play');
    startedAt.current = performance.now();
  }, []);

  const start = () => {
    order.current = [...pool].sort(() => Math.random() - 0.5).slice(0, ROUNDS);
    setLog([]);
    void loadRound(0);
  };

  useEffect(() => {
    if (!data || !canvasRef.current) return;
    const c = canvasRef.current;
    c.width = data.canvas.width; c.height = data.canvas.height;
    c.getContext('2d')!.drawImage(data.canvas, 0, 0);
  }, [data, phase]);

  const submit = useCallback(() => {
    if (!data || phase !== 'play') return;
    const seconds = Math.min(SECONDS, (performance.now() - startedAt.current) / 1000);
    let human = 0;
    let ai = 0;
    let aiUnsure = 0;
    data.truth.forEach((t, i) => {
      const bad = t.status === 'non_compliant';
      if (flags.has(i) === bad) human += 1;
      if (data.ai[i] === 'unsure') aiUnsure += 1;
      else if ((data.ai[i] === 'flag') === bad) ai += 1;
    });
    setLog((l) => [...l, { sample: data.sample.id, persons: data.truth.length, human, ai, aiUnsure, seconds }]);
    setPhase('reveal');
  }, [data, flags, phase]);

  useEffect(() => {
    if (phase !== 'play') return;
    const iv = setInterval(() => {
      const l = SECONDS - (performance.now() - startedAt.current) / 1000;
      setLeft(Math.max(0, l));
      if (l <= 0) submit();
    }, 100);
    return () => clearInterval(iv);
  }, [phase, submit]);

  const next = () => {
    if (round + 1 < order.current.length) void loadRound(round + 1);
    else {
      const total = log.reduce((s, r) => s + r.persons, 0);
      const sc: Score = {
        date: new Date().toISOString(), human: log.reduce((s, r) => s + r.human, 0), ai: log.reduce((s, r) => s + r.ai, 0),
        total, seconds: log.reduce((s, r) => s + r.seconds, 0),
      };
      const all = [sc, ...scores].sort((a, b) => b.human / b.total - a.human / a.total || a.seconds - b.seconds).slice(0, 10);
      localStorage.setItem(KEY, JSON.stringify(all));
      setScores(all);
      setPhase('end');
    }
  };

  const totals = useMemo(() => ({
    persons: log.reduce((s, r) => s + r.persons, 0), human: log.reduce((s, r) => s + r.human, 0),
    ai: log.reduce((s, r) => s + r.ai, 0), aiUnsure: log.reduce((s, r) => s + r.aiUnsure, 0),
  }), [log]);

  const exportCsv = () => {
    const rows = ['sample,persons,human_correct,ai_correct,ai_unsure,seconds_used', ...log.map((r) => [r.sample, r.persons, r.human, r.ai, r.aiUnsure, r.seconds.toFixed(1)].join(','))];
    download('sitesafe-inspector-challenge.csv', new Blob([rows.join('\n') + '\n'], { type: 'text/csv' }));
  };

  const W = data?.canvas.width ?? 1;
  const H = data?.canvas.height ?? 1;

  return (
    <div className="mx-auto max-w-5xl px-4 py-6">
      <PageHeader eyebrow="Inspector Challenge" title="Can you beat the model?">
        {ROUNDS} site photos, {SECONDS} seconds each. Tap every worker who is missing a helmet or a hi-vis vest, then see what the model said. Scores stay on this device.
      </PageHeader>

      {phase === 'intro' && (
        <div className="grid gap-5 md:grid-cols-2">
          <Panel className="hud-corners p-6">
            <h2 className="text-lg font-semibold">How it works</h2>
            <ol className="mt-3 list-decimal space-y-2 pl-5 text-sm text-muted">
              <li>Every worker has a white outline. Tap the ones you think are <strong className="text-ink">not compliant</strong> (no helmet or no vest).</li>
              <li>Press <strong className="text-ink">Lock in</strong> or let the timer run out.</li>
              <li>The answer, your result and the model's result are shown side by side.</li>
            </ol>
            <Button variant="primary" size="lg" className="mt-5" disabled={pool.length < 1} onClick={start}><Play size={18} /> Start the challenge</Button>
            {pool.length < 1 && <p className="mt-2 text-xs text-muted">Loading photos…</p>}
          </Panel>
          <Panel className="p-6">
            <div className="flex items-center gap-2"><Trophy size={18} className="text-hivis" /><h2 className="text-lg font-semibold">Leaderboard (this device)</h2></div>
            {scores.length ? (
              <ol className="mt-3 space-y-1.5 text-sm">
                {scores.map((s, i) => (
                  <li key={s.date} className="flex items-center gap-3 rounded border border-line px-3 py-1.5">
                    <span className="num w-5 text-muted">{i + 1}</span>
                    <span className="num font-semibold">{Math.round((100 * s.human) / s.total)}%</span>
                    <span className="text-xs text-muted">you</span>
                    <span className="num">{Math.round((100 * s.ai) / s.total)}%</span>
                    <span className="text-xs text-muted">model</span>
                    <span className="num ml-auto text-xs text-muted">{s.seconds.toFixed(0)} s</span>
                  </li>
                ))}
              </ol>
            ) : <p className="mt-3 text-sm text-muted">No runs yet. Be the first inspector.</p>}
          </Panel>
        </div>
      )}

      {phase === 'loading' && <Panel className="grid h-64 place-items-center text-sm text-muted">Preparing photo {round + 1}…</Panel>}

      {(phase === 'play' || phase === 'reveal') && data && (
        <div className="space-y-3">
          <div className="flex items-center gap-3">
            <Eyebrow>Photo {round + 1} / {order.current.length}</Eyebrow>
            <div className="flex flex-1 items-center gap-2">
              <Timer size={15} className={left < 4 && phase === 'play' ? 'text-danger' : 'text-muted'} />
              <div className="h-2 flex-1 overflow-hidden rounded-full bg-raised">
                <div className={cn('h-full transition-[width] duration-100', left < 4 ? 'bg-danger' : 'bg-hivis')} style={{ width: `${phase === 'play' ? (100 * left) / SECONDS : 0}%` }} />
              </div>
              <span className="num w-8 text-right text-xs text-muted">{phase === 'play' ? left.toFixed(0) : '–'}s</span>
            </div>
          </div>
          <div className="hud-corners relative overflow-hidden rounded bg-black" style={{ aspectRatio: `${W} / ${H}` }}>
            <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />
            {data.truth.map((t, i) => {
              const bad = t.status === 'non_compliant';
              const flagged = flags.has(i);
              const reveal = phase === 'reveal';
              return (
                <button
                  key={i} type="button" disabled={reveal}
                  aria-pressed={flagged} aria-label={`Worker ${i + 1}${flagged ? ', flagged' : ''}`}
                  onClick={() => setFlags((f) => { const n = new Set(f); if (n.has(i)) n.delete(i); else n.add(i); return n; })}
                  className={cn(
                    'absolute border-2 transition-colors',
                    !reveal && (flagged ? 'border-[#ff7a1a] bg-[#ff7a1a]/25' : 'border-dashed border-white/80 hover:bg-white/10'),
                    reveal && (bad ? 'border-dashed border-[#ff4d4f] bg-[#ff4d4f]/15' : 'border-[#2fd67b] bg-[#2fd67b]/10'),
                  )}
                  style={{ left: `${(100 * t.box[0]) / W}%`, top: `${(100 * t.box[1]) / H}%`, width: `${(100 * (t.box[2] - t.box[0])) / W}%`, height: `${(100 * (t.box[3] - t.box[1])) / H}%` }}
                >
                  {!reveal && flagged && <span className="absolute left-0 top-0 bg-[#ff7a1a] px-1 font-mono text-[10px] font-bold text-[#0b0d10]">FLAGGED</span>}
                  {reveal && (
                    <span className="absolute left-0 top-0 flex flex-col items-start gap-px font-mono text-[10px] font-bold text-[#0b0d10]">
                      <span className={cn('px-1', bad ? 'bg-[#ff4d4f]' : 'bg-[#2fd67b]')}>{bad ? `✕ no ${t.missing.join(', ')}` : '✓ compliant'}</span>
                      <span className="bg-white px-1">you {flagged === bad ? '✓' : '✕'} · AI {data.ai[i] === 'unsure' ? '?' : (data.ai[i] === 'flag') === bad ? '✓' : '✕'}</span>
                    </span>
                  )}
                </button>
              );
            })}
          </div>
          {phase === 'play' ? (
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted">{flags.size} flagged</span>
              <Button variant="primary" size="lg" onClick={submit}>Lock in</Button>
            </div>
          ) : (
            <div className="flex flex-wrap items-center gap-4">
              <span className="flex items-center gap-1.5 text-sm"><User size={16} /> You: <strong className="num">{log.at(-1)?.human}/{log.at(-1)?.persons}</strong></span>
              <span className="flex items-center gap-1.5 text-sm"><Bot size={16} /> Model: <strong className="num">{log.at(-1)?.ai}/{log.at(-1)?.persons}</strong>
                {(log.at(-1)?.aiUnsure ?? 0) > 0 && <span className="text-xs text-unsure">({log.at(-1)?.aiUnsure} “not sure”)</span>}</span>
              <Button variant="primary" className="ml-auto" onClick={next}>{round + 1 < order.current.length ? 'Next photo' : 'See result'}</Button>
            </div>
          )}
        </div>
      )}

      {phase === 'end' && (
        <Panel className="hud-corners p-6 text-center">
          <Trophy size={34} className="mx-auto text-hivis" />
          <h2 className="mt-3 text-2xl font-bold">
            {totals.human > totals.ai ? 'You beat the model.' : totals.human === totals.ai ? 'A draw with the model.' : 'The model wins this round.'}
          </h2>
          <div className="mx-auto mt-5 grid max-w-md grid-cols-2 gap-3">
            <div className="rounded border border-line p-4"><User className="mx-auto" /><div className="num mt-1 text-3xl font-bold">{Math.round((100 * totals.human) / totals.persons)}%</div><div className="eyebrow">you · {totals.human}/{totals.persons}</div></div>
            <div className="rounded border border-line p-4"><Bot className="mx-auto" /><div className="num mt-1 text-3xl font-bold">{Math.round((100 * totals.ai) / totals.persons)}%</div><div className="eyebrow">model · {totals.ai}/{totals.persons}</div></div>
          </div>
          <p className="mx-auto mt-4 max-w-md text-sm text-muted">
            The model said “not sure” for {totals.aiUnsure} worker{totals.aiUnsure === 1 ? '' : 's'}; those count against it here. In real use they are handed to you, which is the point: the model and the inspector work together.
          </p>
          <div className="mt-5 flex justify-center gap-2">
            <Button variant="primary" onClick={start}><RotateCcw size={16} /> Play again</Button>
            <Button onClick={exportCsv}><Download size={16} /> Export my rounds (CSV)</Button>
          </div>
        </Panel>
      )}
    </div>
  );
}

export default function Challenge() {
  return <EngineGate><Game /></EngineGate>;
}
