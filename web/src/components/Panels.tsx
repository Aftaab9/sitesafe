import { motion } from 'framer-motion';
import { BadgeCheck, HelpCircle, ShieldAlert, Truck, TriangleAlert } from 'lucide-react';
import { ITEM_LABEL, PPE_ITEMS, SITE_PROFILES, type Item, type ProfileId } from '../lib/classes';
import type { Summary } from '../lib/matching';
import { prefersReducedMotion, useSettings } from '../lib/store';
import { Gauge, ITEM_ICON } from './Ppe';
import { cn, Eyebrow, Panel, Range, Segmented, Switch } from './ui';

export function ProfilePicker() {
  const { profile, set } = useSettings();
  return (
    <div>
      <Eyebrow className="mb-1.5">Site profile: which PPE is mandatory here</Eyebrow>
      <Segmented<ProfileId>
        label="Site profile"
        value={profile}
        onChange={(v) => set({ profile: v })}
        options={(Object.keys(SITE_PROFILES) as ProfileId[]).map((id) => ({ value: id, label: SITE_PROFILES[id].label, title: SITE_PROFILES[id].hint }))}
      />
      <p className="mt-1.5 text-xs text-muted">{SITE_PROFILES[profile].hint}.</p>
    </div>
  );
}

function Confetti() {
  if (prefersReducedMotion()) return null;
  const bits = Array.from({ length: 16 }, (_, i) => i);
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden>
      {bits.map((i) => (
        <motion.span
          key={i}
          className="absolute left-1/2 top-1/2 h-2 w-1.5 rounded-[1px]"
          style={{ background: ['#ffd400', '#2fd67b', '#ff7a1a'][i % 3] }}
          initial={{ x: 0, y: 0, opacity: 1, rotate: 0 }}
          animate={{ x: Math.cos((i / 16) * Math.PI * 2) * (70 + (i % 4) * 22), y: Math.sin((i / 16) * Math.PI * 2) * (46 + (i % 3) * 18) + 30, opacity: 0, rotate: 260 }}
          transition={{ duration: 1.1, ease: 'easeOut' }}
        />
      ))}
    </div>
  );
}

export function SummaryPanel({ summary, zoneAlerts, forkliftAlerts, required, title = 'This photo' }: {
  summary: Summary | null; zoneAlerts: number; forkliftAlerts: number; required: Item[]; title?: string;
}) {
  const s = summary;
  const clear = !!s && s.persons > 0 && s.nonCompliant === 0 && s.unsure === 0 && zoneAlerts === 0;
  const max = Math.max(1, ...(s ? PPE_ITEMS.map((i) => s.missingByItem[i]) : [1]));
  return (
    <Panel className="relative overflow-hidden p-4" aria-live="polite">
      <Eyebrow>{title}</Eyebrow>
      <div className="mt-2 flex items-end gap-4">
        <Gauge pct={s?.compliancePct ?? null} />
        <dl className="grid flex-1 grid-cols-1 gap-1.5 text-sm">
          <div className="flex items-center justify-between"><dt className="flex items-center gap-1.5 text-ok"><BadgeCheck size={15} /> Compliant</dt><dd className="num font-semibold">{s?.compliant ?? '–'}</dd></div>
          <div className="flex items-center justify-between"><dt className="flex items-center gap-1.5 text-danger"><ShieldAlert size={15} /> PPE missing</dt><dd className="num font-semibold">{s?.nonCompliant ?? '–'}</dd></div>
          <div className="flex items-center justify-between"><dt className="flex items-center gap-1.5 text-unsure"><HelpCircle size={15} /> Check manually</dt><dd className="num font-semibold">{s?.unsure ?? '–'}</dd></div>
        </dl>
      </div>

      {clear && (
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
          className="relative mt-4 flex items-center gap-2 rounded border border-ok/50 bg-ok/10 px-3 py-2 text-sm font-semibold text-ok"
        >
          <BadgeCheck size={18} /> Site clear: everyone in view is wearing the required PPE.
          <Confetti />
        </motion.div>
      )}
      {s && s.unsure > 0 && (
        <div className="mt-4 flex items-start gap-2 rounded border border-unsure/40 bg-unsure/10 px-3 py-2 text-sm text-unsure">
          <HelpCircle size={17} className="mt-0.5 shrink-0" />
          <span>
            {s.unsure} worker{s.unsure > 1 ? 's' : ''} could not be decided with confidence. That is expected for people who are far away,
            partly hidden or in poor light. Please take a look yourself.
          </span>
        </div>
      )}
      {s && s.persons === 0 && (
        <p className="mt-4 rounded border border-line bg-raised px-3 py-2 text-sm text-muted">No people found in this image. Try a closer or brighter photo.</p>
      )}

      <Eyebrow className="mt-5">Missing by PPE type</Eyebrow>
      <ul className="mt-2 space-y-1.5">
        {PPE_ITEMS.map((item) => {
          const Icon = ITEM_ICON[item];
          const n = s?.missingByItem[item] ?? 0;
          const req = required.includes(item);
          return (
            <li key={item} className={cn('flex items-center gap-2 text-sm', !req && 'opacity-45')}>
              <Icon size={15} className="shrink-0 text-muted" />
              <span className="w-16 shrink-0">{ITEM_LABEL[item]}</span>
              <span className="h-2 flex-1 overflow-hidden rounded-full bg-raised">
                <span className="block h-full rounded-full bg-danger transition-[width] duration-500" style={{ width: `${(100 * n) / max}%` }} />
              </span>
              <span className="num w-5 text-right font-semibold">{n}</span>
            </li>
          );
        })}
      </ul>

      {(zoneAlerts > 0 || forkliftAlerts > 0 || (s?.forklifts ?? 0) > 0) && (
        <div className="mt-4 space-y-1.5 border-t border-line pt-3 text-sm">
          {zoneAlerts > 0 && <p className="flex items-center gap-2 font-medium text-hazard"><TriangleAlert size={15} /> {zoneAlerts} worker{zoneAlerts > 1 ? 's' : ''} inside a danger zone</p>}
          {forkliftAlerts > 0 && <p className="flex items-center gap-2 font-medium text-hazard"><Truck size={15} /> {forkliftAlerts} worker{forkliftAlerts > 1 ? 's' : ''} close to a forklift</p>}
          {s && s.forklifts > 0 && forkliftAlerts === 0 && <p className="flex items-center gap-2 text-muted"><Truck size={15} /> {s.forklifts} forklift{s.forklifts > 1 ? 's' : ''} in view, nobody close</p>}
        </div>
      )}
    </Panel>
  );
}

export function DetectionSettings({ showSound = false }: { showSound?: boolean }) {
  const st = useSettings();
  return (
    <div className="divide-y divide-line">
      <Switch id="blur" label="Blur faces" hint="Applied on this device before anything is shown or exported." checked={st.blurFaces} onChange={(v) => st.set({ blurFaces: v })} />
      <Switch id="strict" label="Strict privacy" hint="Also blur the head area when no face is found (far away, turned away)." checked={st.strictBlur} onChange={(v) => st.set({ strictBlur: v })} />
      {showSound && <Switch id="sound" label="Sound cue" hint="A short tone when a new violation appears." checked={st.sound} onChange={(v) => st.set({ sound: v })} />}
      <Range label="Detection sensitivity" display={`min. confidence ${st.conf.toFixed(2)}`} min={0.1} max={0.6} step={0.05} value={st.conf}
        onChange={(e) => st.set({ conf: Number(e.target.value) })} />
      <Range label="Forklift safety distance" display={`${st.forkliftDistance.toFixed(1)} × person height`} min={0.1} max={2} step={0.1} value={st.forkliftDistance}
        onChange={(e) => st.set({ forkliftDistance: Number(e.target.value) })} />
      <Switch id="motion" label="Reduce motion" hint="Turn off scan and lock-on animations." checked={st.reducedMotion} onChange={(v) => st.set({ reducedMotion: v })} />
    </div>
  );
}
