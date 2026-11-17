// The photo stage: blurred image, before/after privacy slider, animated lock-on boxes,
// PPE connector lines, danger-zone editor and the "explain" heat-map.
import { motion } from 'framer-motion';
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { MoveHorizontal } from 'lucide-react';
import type { Analysis } from '../lib/analyze';
import { CLASS_NAMES, SITE_PROFILES, type ProfileId } from '../lib/classes';
import { blurRegions, STATUS_COLOR, STATUS_DASH, STATUS_SYMBOL } from '../lib/draw';
import { prefersReducedMotion } from '../lib/store';
import type { Point, Zone } from '../lib/zones';
import { WorkerCard } from './Ppe';
import { cn } from './ui';

interface Props {
  source: HTMLCanvasElement;         // the image, already scaled, never leaves memory
  analysis: Analysis | null;
  scanning: boolean;
  profile: ProfileId;
  blur: boolean;
  compare: boolean;                  // show the before/after slider
  hovered: number | null;
  onHover: (i: number | null) => void;
  zones: Zone[];
  onZones: (z: Zone[]) => void;
  drawing: boolean;
  onDrawingDone: () => void;
  heatmap?: string | null;
  animateKey: number;
}

const SNAP = 14; // screen px

export function Stage(p: Props) {
  const { source, analysis } = p;
  const W = source.width;
  const H = source.height;
  const wrap = useRef<HTMLDivElement>(null);
  const blurRef = useRef<HTMLCanvasElement>(null);
  const origRef = useRef<HTMLCanvasElement>(null);
  const [k, setK] = useState(1); // image px per screen px
  const [split, setSplit] = useState(50);
  const [draft, setDraft] = useState<Point[]>([]);
  const [cursor, setCursor] = useState<Point | null>(null);
  const drag = useRef<{ zone: number; pt: number } | null>(null);
  const reduce = prefersReducedMotion();

  useLayoutEffect(() => {
    const el = wrap.current!;
    const ro = new ResizeObserver(() => setK(W / Math.max(1, el.clientWidth)));
    ro.observe(el);
    return () => ro.disconnect();
  }, [W]);

  // Paint the two layers. The original is only painted when the slider is on.
  useEffect(() => {
    const b = blurRef.current!;
    b.width = W; b.height = H;
    const ctx = b.getContext('2d')!;
    ctx.drawImage(source, 0, 0);
    if (p.blur && analysis) blurRegions(ctx, analysis.blur);
    const o = origRef.current;
    if (o) { o.width = W; o.height = H; o.getContext('2d')!.drawImage(source, 0, 0); }
  }, [source, analysis, p.blur, p.compare, W, H]);

  useEffect(() => { if (!p.drawing) { setDraft([]); setCursor(null); } }, [p.drawing]);

  const toImage = useCallback((e: { clientX: number; clientY: number }): Point => {
    const r = wrap.current!.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * W, ((e.clientY - r.top) / r.height) * H];
  }, [W, H]);

  /** Snap to image edges and to existing vertices so zones line up cleanly. */
  const snap = useCallback((pt: Point, skip?: { zone: number; pt: number }): Point => {
    const tol = SNAP * k;
    let [x, y] = pt;
    for (const z of p.zones) {
      for (let i = 0; i < z.points.length; i++) {
        if (skip && skip.zone === z.id && skip.pt === i) continue;
        const [vx, vy] = z.points[i];
        if (Math.hypot(vx - x, vy - y) < tol) return [vx, vy];
      }
    }
    if (x < tol) x = 0; else if (x > W - tol) x = W;
    if (y < tol) y = 0; else if (y > H - tol) y = H;
    return [Math.min(W, Math.max(0, x)), Math.min(H, Math.max(0, y))];
  }, [k, p.zones, W, H]);

  const closeDraft = useCallback((pts: Point[]) => {
    if (pts.length >= 3) {
      const id = Math.max(0, ...p.zones.map((z) => z.id)) + 1;
      p.onZones([...p.zones, { id, name: `Zone ${id}`, points: pts }]);
    }
    setDraft([]);
    setCursor(null);
    p.onDrawingDone();
  }, [p]);

  useEffect(() => {
    if (!p.drawing) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { setDraft([]); p.onDrawingDone(); }
      if (e.key === 'Enter') closeDraft(draft);
      if (e.key === 'Backspace') setDraft((d) => d.slice(0, -1));
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [p, draft, closeDraft]);

  const onStageClick = (e: React.MouseEvent) => {
    if (!p.drawing) return;
    const pt = snap(toImage(e));
    if (draft.length >= 3 && Math.hypot(draft[0][0] - pt[0], draft[0][1] - pt[1]) < SNAP * k * 1.3) closeDraft(draft);
    else setDraft((d) => [...d, pt]);
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (drag.current) {
      const d = drag.current;
      const pt = snap(toImage(e), d);
      p.onZones(p.zones.map((z) => (z.id === d.zone ? { ...z, points: z.points.map((q, i) => (i === d.pt ? pt : q)) } : z)));
    } else if (p.drawing) setCursor(snap(toImage(e)));
  };

  const activeZones = new Set(analysis?.events.flatMap((e) => e.zones) ?? []);
  const required = SITE_PROFILES[p.profile].required;
  const hov = p.hovered !== null && analysis ? analysis.people[p.hovered] : null;
  const sw = 2.5 * k; // stroke width that looks constant on screen

  return (
    <div
      ref={wrap}
      className={cn('hud-corners relative w-full select-none overflow-hidden rounded bg-black', p.drawing && 'cursor-crosshair')}
      style={{ aspectRatio: `${W} / ${H}`, touchAction: p.drawing || drag.current ? 'none' : undefined }}
      onClick={onStageClick}
      onDoubleClick={() => p.drawing && closeDraft(draft)}
      onPointerMove={onPointerMove}
      onPointerUp={() => { drag.current = null; }}
      onPointerLeave={() => { drag.current = null; setCursor(null); }}
    >
      <canvas ref={blurRef} className="absolute inset-0 h-full w-full" aria-label="Analysed site photo with faces blurred" />
      {p.compare && (
        <div className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }}>
          <canvas ref={origRef} className="h-full w-full" aria-hidden />
        </div>
      )}
      {p.heatmap && (
        <img src={p.heatmap} alt="" className="pointer-events-none absolute inset-0 h-full w-full object-fill opacity-70 mix-blend-screen" />
      )}

      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="absolute inset-0 h-full w-full">
        {/* danger zones */}
        {p.zones.map((z) => {
          const hot = activeZones.has(z.id);
          return (
            <g key={z.id} className={hot ? 'animate-hazard' : undefined}>
              <polygon
                points={z.points.map((q) => q.join(',')).join(' ')}
                fill={hot ? 'rgba(255,77,79,0.22)' : 'rgba(255,122,26,0.12)'}
                stroke={hot ? '#ff4d4f' : '#ff7a1a'} strokeWidth={(hot ? 4 : 2.5) * k} strokeDasharray={`${12 * k} ${8 * k}`}
              />
              {z.points.map((q, i) => (
                <circle
                  key={i} cx={q[0]} cy={q[1]} r={7 * k} fill="#0b0d10" stroke="#ff7a1a" strokeWidth={2 * k}
                  className="cursor-grab"
                  onPointerDown={(e) => { e.stopPropagation(); (e.target as Element).setPointerCapture?.(e.pointerId); drag.current = { zone: z.id, pt: i }; }}
                  onClick={(e) => e.stopPropagation()}
                />
              ))}
            </g>
          );
        })}
        {draft.length > 0 && (
          <g>
            <polyline
              points={[...draft, ...(cursor ? [cursor] : [])].map((q) => q.join(',')).join(' ')}
              fill="rgba(255,212,0,0.08)" stroke="#ffd400" strokeWidth={2.5 * k} strokeDasharray={`${8 * k} ${6 * k}`}
            />
            {draft.map((q, i) => (
              <circle key={i} cx={q[0]} cy={q[1]} r={(i === 0 && draft.length >= 3 ? 9 : 6) * k} fill={i === 0 ? '#ffd400' : '#0b0d10'} stroke="#ffd400" strokeWidth={2 * k} />
            ))}
          </g>
        )}

        {/* person boxes with a lock-on snap */}
        {analysis && !p.scanning && analysis.people.map((person, i) => {
          const [x1, y1, x2, y2] = person.box;
          const color = STATUS_COLOR[person.status];
          const c = Math.min(x2 - x1, y2 - y1) * 0.22;
          const dim = p.hovered !== null && p.hovered !== i;
          return (
            <motion.g
              key={`${p.animateKey}-${i}`}
              initial={reduce ? false : { opacity: 0, scale: 1.25 }}
              animate={{ opacity: dim ? 0.35 : 1, scale: 1 }}
              transition={{ delay: reduce ? 0 : 0.12 * i, type: 'spring', stiffness: 420, damping: 22 }}
              style={{ transformOrigin: `${(x1 + x2) / 2}px ${(y1 + y2) / 2}px`, transformBox: 'view-box' as never }}
              onMouseEnter={() => p.onHover(i)}
              onMouseLeave={() => p.onHover(null)}
              onClick={(e) => { if (!p.drawing) { e.stopPropagation(); p.onHover(p.hovered === i ? null : i); } }}
            >
              <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} fill="transparent" stroke={color} strokeWidth={sw * 0.6} strokeOpacity={0.55}
                strokeDasharray={STATUS_DASH[person.status].map((d) => d * k).join(' ')} />
              <path
                d={`M${x1} ${y1 + c}V${y1}H${x1 + c} M${x2 - c} ${y1}H${x2}V${y1 + c} M${x2} ${y2 - c}V${y2}H${x2 - c} M${x1 + c} ${y2}H${x1}V${y2 - c}`}
                fill="none" stroke={color} strokeWidth={sw * 1.5} strokeLinecap="square"
              />
              {analysis.events[i] && (analysis.events[i].zones.length > 0 || analysis.events[i].nearForklift) && (
                <rect x={x1 - 6 * k} y={y1 - 6 * k} width={x2 - x1 + 12 * k} height={y2 - y1 + 12 * k} fill="none" stroke="#ff7a1a"
                  strokeWidth={2 * k} className="animate-hazard" />
              )}
            </motion.g>
          );
        })}

        {/* hovered worker: matched PPE boxes and connector lines (the matching logic, made visible) */}
        {analysis && hov && hov.matched.map((j) => {
          const d = analysis.dets[j];
          const [x1, y1, x2, y2] = d.box;
          const neg = CLASS_NAMES[d.cls].startsWith('no_');
          const col = neg ? '#ff4d4f' : '#2fd67b';
          const ax = hov.box[0];
          const ay = hov.box[1];
          return (
            <g key={j} pointerEvents="none">
              <line x1={ax} y1={ay} x2={(x1 + x2) / 2} y2={(y1 + y2) / 2} stroke={col} strokeWidth={1.5 * k} strokeDasharray={`${4 * k} ${4 * k}`} />
              <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} fill={`${col}22`} stroke={col} strokeWidth={2 * k}
                strokeDasharray={neg ? `${6 * k} ${4 * k}` : undefined} />
              <circle cx={(x1 + x2) / 2} cy={(y1 + y2) / 2} r={3 * k} fill={col} />
            </g>
          );
        })}
      </svg>

      {/* labels as HTML so text stays crisp at any image size */}
      {analysis && !p.scanning && analysis.people.map((person, i) => (
        <motion.button
          type="button"
          key={`${p.animateKey}-tag-${i}`}
          initial={reduce ? false : { opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: reduce ? 0 : 0.12 * i + 0.15 }}
          onMouseEnter={() => p.onHover(i)}
          onMouseLeave={() => p.onHover(null)}
          onFocus={() => p.onHover(i)}
          onBlur={() => p.onHover(null)}
          onClick={(e) => e.stopPropagation()}
          className="absolute -translate-y-full whitespace-nowrap rounded-t px-1.5 py-0.5 font-mono text-[11px] font-semibold text-[#0b0d10]"
          style={{
            left: `${(person.box[0] / W) * 100}%`,
            top: `${(Math.max(person.box[1], 22 * k) / H) * 100}%`,
            background: STATUS_COLOR[person.status],
          }}
          aria-label={`Worker ${i + 1}: ${person.status.replace('_', ' ')}`}
        >
          {STATUS_SYMBOL[person.status]} W{i + 1}
          {person.missing.length > 0 && <span className="hidden sm:inline"> · no {person.missing.join(', ')}</span>}
        </motion.button>
      ))}

      {/* hovered PPE labels */}
      {analysis && hov && hov.matched.map((j) => {
        const d = analysis.dets[j];
        return (
          <span
            key={j}
            className="pointer-events-none absolute rounded bg-[#0b0d10]/85 px-1 font-mono text-[10px] text-white"
            style={{ left: `${(d.box[0] / W) * 100}%`, top: `${(d.box[3] / H) * 100}%` }}
          >
            {CLASS_NAMES[d.cls]} {(d.conf * 100).toFixed(0)}%
          </span>
        );
      })}

      {/* floating safety card (desktop) */}
      {analysis && hov && p.hovered !== null && (
        <div
          className="pointer-events-none absolute z-10 hidden md:block"
          style={{
            top: `${Math.min(55, (hov.box[1] / H) * 100)}%`,
            ...(hov.box[2] / W < 0.62
              ? { left: `calc(${(hov.box[2] / W) * 100}% + 10px)` }
              : { right: `calc(${(1 - hov.box[0] / W) * 100}% + 10px)` }),
          }}
        >
          <WorkerCard person={hov} id={p.hovered + 1} required={required} compact />
        </div>
      )}

      {p.scanning && <div className="scan-line" aria-hidden />}
      {p.scanning && <div className="absolute inset-0 bg-[#0b0d10]/25" aria-hidden />}

      {/* before / after privacy slider */}
      {p.compare && (
        <div
          role="slider"
          tabIndex={0}
          aria-label="Compare original and face-blurred image"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(split)}
          className="absolute inset-y-0 z-20 w-10 -translate-x-1/2 cursor-ew-resize touch-none"
          style={{ left: `${split}%` }}
          onClick={(e) => e.stopPropagation()}
          onKeyDown={(e) => {
            if (e.key === 'ArrowLeft') setSplit((s) => Math.max(0, s - 4));
            if (e.key === 'ArrowRight') setSplit((s) => Math.min(100, s + 4));
          }}
          onPointerDown={(e) => {
            e.stopPropagation();
            const el = e.currentTarget;
            el.setPointerCapture(e.pointerId);
            const move = (ev: PointerEvent) => {
              const r = wrap.current!.getBoundingClientRect();
              setSplit(Math.min(100, Math.max(0, ((ev.clientX - r.left) / r.width) * 100)));
            };
            const up = () => { el.removeEventListener('pointermove', move); el.removeEventListener('pointerup', up); };
            el.addEventListener('pointermove', move);
            el.addEventListener('pointerup', up);
          }}
        >
          <div className="mx-auto h-full w-0.5 bg-hivis" />
          <div className="absolute left-1/2 top-1/2 grid h-10 w-10 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full bg-hivis text-[#0b0d10] shadow-lg">
            <MoveHorizontal size={18} />
          </div>
        </div>
      )}
      {p.compare && (
        <>
          <span className="pointer-events-none absolute left-2 top-2 rounded bg-[#0b0d10]/80 px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-white">Original</span>
          <span className="pointer-events-none absolute right-2 top-2 rounded bg-hivis px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-[#0b0d10]">Faces blurred</span>
        </>
      )}
      {p.drawing && (
        <div className="pointer-events-none absolute inset-x-0 bottom-2 mx-auto w-fit rounded bg-[#0b0d10]/85 px-3 py-1.5 text-center font-mono text-[11px] text-white">
          Tap to add corners · tap the first corner (or press Enter) to close · Esc to cancel
        </div>
      )}
    </div>
  );
}
