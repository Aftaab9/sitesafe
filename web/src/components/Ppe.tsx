import { Check, Footprints, Glasses, Hand, HardHat, HelpCircle, Shirt, TriangleAlert, X, EyeOff, type LucideIcon } from 'lucide-react';
import { ITEM_LABEL, PPE_ITEMS, type Item } from '../lib/classes';
import type { ItemState, PersonResult, PersonStatus } from '../lib/matching';
import { STATUS_LABEL } from '../lib/draw';
import { cn } from './ui';

export const ITEM_ICON: Record<Item, LucideIcon> = {
  helmet: HardHat, goggles: Glasses, vest: Shirt, gloves: Hand, boots: Footprints,
};

const STATE_STYLE: Record<ItemState, { cls: string; mark: LucideIcon; text: string }> = {
  present: { cls: 'border-ok/50 text-ok bg-ok/10', mark: Check, text: 'worn' },
  missing: { cls: 'border-danger/60 text-danger bg-danger/10', mark: X, text: 'missing' },
  unsure: { cls: 'border-unsure/60 text-unsure bg-unsure/10', mark: HelpCircle, text: 'not sure' },
  not_visible: { cls: 'border-line text-muted bg-raised', mark: EyeOff, text: 'not visible' },
};

export const STATUS_TEXT: Record<PersonStatus, string> = {
  compliant: 'text-ok', non_compliant: 'text-danger', unsure: 'text-unsure',
};
export const STATUS_BORDER: Record<PersonStatus, string> = {
  compliant: 'border-ok/60', non_compliant: 'border-danger/70', unsure: 'border-unsure/60',
};
export const STATUS_ICON: Record<PersonStatus, LucideIcon> = {
  compliant: Check, non_compliant: TriangleAlert, unsure: HelpCircle,
};

/** One PPE item: icon plus an explicit tick / cross / question mark (never colour alone). */
export function ItemBadge({ item, state, required }: { item: Item; state: ItemState; required: boolean }) {
  const Icon = ITEM_ICON[item];
  const s = STATE_STYLE[state];
  const Mark = s.mark;
  return (
    <span
      title={`${ITEM_LABEL[item]}: ${s.text}${required ? '' : ' (not required here)'}`}
      className={cn('relative grid h-9 w-9 place-items-center rounded border', s.cls, !required && 'opacity-45')}
    >
      <Icon size={17} aria-hidden />
      <span className="absolute -bottom-1.5 -right-1.5 grid h-4 w-4 place-items-center rounded-full border border-current bg-surface">
        <Mark size={10} strokeWidth={3} aria-hidden />
      </span>
      <span className="sr-only">{ITEM_LABEL[item]}: {s.text}</span>
    </span>
  );
}

export function ConfidenceRing({ value, status, size = 44 }: { value: number; status: PersonStatus; size?: number }) {
  const r = size / 2 - 4;
  const c = 2 * Math.PI * r;
  return (
    <span className="relative inline-grid place-items-center" style={{ width: size, height: size }} title={`Person confidence ${(value * 100).toFixed(0)}%`}>
      <svg width={size} height={size} className={cn('-rotate-90', STATUS_TEXT[status])} aria-hidden>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgb(var(--line))" strokeWidth={3.5} />
        <circle
          cx={size / 2} cy={size / 2} r={r} fill="none" stroke="currentColor" strokeWidth={3.5}
          strokeLinecap="round" strokeDasharray={c} strokeDashoffset={c * (1 - value)}
          style={{ transition: 'stroke-dashoffset .6s ease' }}
        />
      </svg>
      <span className="num absolute text-[11px] font-semibold">{Math.round(value * 100)}</span>
    </span>
  );
}

export function WorkerCard({
  person, id, required, active, alert, onHover, compact,
}: {
  person: PersonResult; id: number; required: Item[]; active?: boolean; alert?: string | null;
  onHover?: (on: boolean) => void; compact?: boolean;
}) {
  const Icon = STATUS_ICON[person.status];
  return (
    <div
      onMouseEnter={() => onHover?.(true)}
      onMouseLeave={() => onHover?.(false)}
      onFocus={() => onHover?.(true)}
      onBlur={() => onHover?.(false)}
      tabIndex={onHover ? 0 : undefined}
      className={cn(
        'rounded-lg border bg-surface p-3 transition-shadow',
        STATUS_BORDER[person.status],
        active && 'shadow-[0_0_0_2px_rgb(var(--hivis))]',
        compact && 'w-60 shadow-xl',
      )}
    >
      <div className="flex items-center gap-3">
        <ConfidenceRing value={person.conf} status={person.status} />
        <div className="min-w-0">
          <div className="font-mono text-xs text-muted">WORKER {String(id).padStart(2, '0')}</div>
          <div className={cn('flex items-center gap-1.5 text-sm font-semibold', STATUS_TEXT[person.status])}>
            <Icon size={15} aria-hidden /> {STATUS_LABEL[person.status]}
          </div>
        </div>
      </div>
      <div className="mt-3 flex gap-2.5">
        {PPE_ITEMS.map((item) => (
          <ItemBadge key={item} item={item} state={person.items[item].state} required={required.includes(item)} />
        ))}
      </div>
      {person.missing.length > 0 && (
        <p className="mt-3 text-xs text-danger">Missing: {person.missing.map((m) => ITEM_LABEL[m].toLowerCase()).join(', ')}</p>
      )}
      {person.status === 'unsure' && (
        <p className="mt-3 text-xs text-unsure">
          Not sure about {person.unsureItems.length ? person.unsureItems.map((m) => ITEM_LABEL[m].toLowerCase()).join(', ') : 'this person'}. Please check in person.
        </p>
      )}
      {alert && <p className="mt-2 flex items-center gap-1.5 text-xs font-medium text-hazard"><TriangleAlert size={13} /> {alert}</p>}
    </div>
  );
}

/** Semi-circular compliance gauge. */
export function Gauge({ pct, size = 180, label = 'compliance' }: { pct: number | null; size?: number; label?: string }) {
  const r = size / 2 - 12;
  const c = Math.PI * r;
  const v = pct ?? 0;
  const tone = pct === null ? 'text-muted' : v >= 100 ? 'text-ok' : v >= 70 ? 'text-hivis' : 'text-danger';
  return (
    <div className="relative inline-block" style={{ width: size, height: size / 2 + 14 }} role="img" aria-label={pct === null ? 'Compliance not available' : `Compliance ${v}%`}>
      <svg width={size} height={size / 2 + 8} className={tone} aria-hidden>
        <path d={`M 12 ${size / 2} A ${r} ${r} 0 0 1 ${size - 12} ${size / 2}`} fill="none" stroke="rgb(var(--line))" strokeWidth={10} strokeLinecap="round" />
        <path
          d={`M 12 ${size / 2} A ${r} ${r} 0 0 1 ${size - 12} ${size / 2}`} fill="none" stroke="currentColor" strokeWidth={10}
          strokeLinecap="round" strokeDasharray={c} strokeDashoffset={c * (1 - v / 100)} style={{ transition: 'stroke-dashoffset .7s ease' }}
        />
      </svg>
      <div className="absolute inset-x-0 bottom-0 text-center">
        <div className="num font-display text-3xl font-bold leading-none">{pct === null ? '–' : `${Math.round(v)}%`}</div>
        <div className="eyebrow mt-1">{label}</div>
      </div>
    </div>
  );
}
