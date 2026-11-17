// Site dashboard: what a safety officer reviews at the end of a walk-round.
// Everything shown is aggregate and lives in memory only; closing the tab erases it.
import { useMemo } from 'react';
import { Link } from 'react-router-dom';
import { Camera, FileText, Film, ScanSearch, Table, Trash2 } from 'lucide-react';
import { Gauge, ITEM_ICON } from '../components/Ppe';
import { Button, Eyebrow, PageHeader, Panel } from '../components/ui';
import { ITEM_LABEL, PPE_ITEMS, SITE_PROFILES } from '../lib/classes';
import { download } from '../lib/media';
import { buildPdf, entryCsv } from '../lib/report';
import { useLog, type LogEntry } from '../lib/store';

const SOURCE_ICON = { photo: ScanSearch, video: Film, live: Camera };

function Stat({ label, value, tone }: { label: string; value: string | number; tone?: string }) {
  return (
    <Panel className="p-4">
      <div className={`num font-display text-3xl font-bold ${tone ?? ''}`}>{value}</div>
      <div className="eyebrow mt-1">{label}</div>
    </Panel>
  );
}

function Timeline({ entry }: { entry: LogEntry }) {
  const tl = entry.timeline!;
  const W = 600;
  const H = 120;
  const maxT = Math.max(1, tl.at(-1)?.t ?? 1);
  const maxY = Math.max(1, ...tl.map((p) => p.persons));
  const x = (t: number) => (t / maxT) * (W - 30) + 24;
  const y = (v: number) => H - 18 - (v / maxY) * (H - 30);
  const path = (key: 'persons' | 'nonCompliant') => tl.map((p, i) => `${i ? 'L' : 'M'}${x(p.t).toFixed(1)} ${y(p[key]).toFixed(1)}`).join(' ');
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Workers in view and workers with PPE missing over the clip">
      {[0, maxY].map((v) => (
        <g key={v}><line x1={24} x2={W - 6} y1={y(v)} y2={y(v)} stroke="rgb(var(--line))" strokeWidth={1} />
          <text x={18} y={y(v) + 3} textAnchor="end" fontSize={10} fill="rgb(var(--muted))" fontFamily="JetBrains Mono Variable">{v}</text></g>
      ))}
      <path d={`${path('nonCompliant')} L${x(maxT)} ${y(0)} L${x(0)} ${y(0)} Z`} fill="rgb(var(--danger) / 0.25)" />
      <path d={path('persons')} fill="none" stroke="rgb(var(--muted))" strokeWidth={2} strokeDasharray="5 4" />
      <path d={path('nonCompliant')} fill="none" stroke="rgb(var(--danger))" strokeWidth={2.5} />
      <text x={24} y={H - 3} fontSize={10} fill="rgb(var(--muted))" fontFamily="JetBrains Mono Variable">0 s</text>
      <text x={W - 6} y={H - 3} textAnchor="end" fontSize={10} fill="rgb(var(--muted))" fontFamily="JetBrains Mono Variable">{maxT.toFixed(0)} s</text>
    </svg>
  );
}

export default function Dashboard() {
  const { entries, clear } = useLog();
  const agg = useMemo(() => {
    const t = { persons: 0, compliant: 0, non: 0, unsure: 0, zone: 0, forklift: 0, missing: { helmet: 0, goggles: 0, vest: 0, gloves: 0, boots: 0 } };
    for (const e of entries) {
      t.persons += e.summary.persons; t.compliant += e.summary.compliant; t.non += e.summary.nonCompliant; t.unsure += e.summary.unsure;
      t.zone += e.zoneEvents; t.forklift += e.forkliftEvents;
      for (const i of PPE_ITEMS) t.missing[i] += e.summary.missingByItem[i];
    }
    const decided = t.compliant + t.non;
    return { ...t, pct: decided ? Math.round((1000 * t.compliant) / decided) / 10 : null };
  }, [entries]);
  const lastClip = entries.find((e) => e.timeline?.length);
  const max = Math.max(1, ...PPE_ITEMS.map((i) => agg.missing[i]));

  if (!entries.length) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-6">
        <PageHeader eyebrow="Site dashboard" title="Nothing checked yet in this session">
          Analyse a photo, a clip or the live camera and the totals appear here. The dashboard holds counts only and is erased when you close the tab.
        </PageHeader>
        <div className="flex flex-wrap gap-2">
          <Link to="/analyze?sample=first"><Button variant="primary"><ScanSearch size={16} /> Try a sample photo</Button></Link>
          <Link to="/video"><Button><Film size={16} /> Check a clip</Button></Link>
          <Link to="/live"><Button><Camera size={16} /> Use my camera</Button></Link>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-7xl px-4 py-6">
      <PageHeader eyebrow="Site dashboard" title="This session at a glance">
        {entries.length} check{entries.length > 1 ? 's' : ''} so far. Counts and anonymous worker numbers only; no images are kept.
      </PageHeader>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-[auto_repeat(4,1fr)]">
        <Panel className="grid place-items-center px-6 py-4"><Gauge pct={agg.pct} label="session compliance" /></Panel>
        <Stat label="worker observations" value={agg.persons} />
        <Stat label="PPE missing" value={agg.non} tone="text-danger" />
        <Stat label="check manually" value={agg.unsure} tone="text-unsure" />
        <Stat label="danger zone / forklift" value={agg.zone + agg.forklift} tone="text-hazard" />
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Panel className="p-4">
          <Eyebrow>Violations by PPE type</Eyebrow>
          <ul className="mt-3 space-y-2.5">
            {PPE_ITEMS.map((item) => {
              const Icon = ITEM_ICON[item];
              return (
                <li key={item} className="flex items-center gap-3 text-sm">
                  <Icon size={16} className="shrink-0 text-muted" />
                  <span className="w-16 shrink-0">{ITEM_LABEL[item]}</span>
                  <span className="h-4 flex-1 overflow-hidden rounded bg-raised">
                    <span className="block h-full rounded bg-danger" style={{ width: `${(100 * agg.missing[item]) / max}%` }} />
                  </span>
                  <span className="num w-6 text-right font-semibold">{agg.missing[item]}</span>
                </li>
              );
            })}
          </ul>
        </Panel>
        <Panel className="p-4">
          <Eyebrow>Violations over time{lastClip ? `: ${lastClip.label}` : ''}</Eyebrow>
          {lastClip ? (
            <>
              <div className="mt-3"><Timeline entry={lastClip} /></div>
              <p className="mt-1 font-mono text-[11px] text-muted"><span className="text-danger">━</span> PPE missing (smoothed) &nbsp; <span>┅</span> workers in view</p>
            </>
          ) : <p className="mt-3 text-sm text-muted">Analyse a clip to see how violations change over time.</p>}
        </Panel>
      </div>

      <Panel className="mt-4 overflow-x-auto">
        <div className="flex flex-wrap items-center gap-2 border-b border-line p-3">
          <Eyebrow className="mr-auto">Session log</Eyebrow>
          <Button size="sm" onClick={() => download('sitesafe-session.csv', new Blob([entryCsv(entries)], { type: 'text/csv' }))}><Table size={15} /> Export CSV</Button>
          <Button size="sm" variant="danger" onClick={clear}><Trash2 size={15} /> Erase session</Button>
        </div>
        <table className="w-full min-w-[640px] text-left text-sm">
          <thead className="text-xs text-muted">
            <tr><th className="p-3 font-medium">Time</th><th className="p-3 font-medium">Check</th><th className="p-3 font-medium">Profile</th>
              <th className="p-3 text-right font-medium">Workers</th><th className="p-3 text-right font-medium">Missing</th><th className="p-3 text-right font-medium">Unsure</th>
              <th className="p-3 text-right font-medium">Compliance</th><th className="p-3" /></tr>
          </thead>
          <tbody>
            {entries.map((e) => {
              const Icon = SOURCE_ICON[e.source];
              return (
                <tr key={e.id} className="border-t border-line">
                  <td className="num p-3 text-xs text-muted">{new Date(e.time).toLocaleTimeString()}</td>
                  <td className="max-w-[220px] truncate p-3"><Icon size={14} className="mr-1.5 inline text-muted" />{e.label}</td>
                  <td className="p-3 text-xs text-muted">{SITE_PROFILES[e.profile].label}</td>
                  <td className="num p-3 text-right">{e.summary.persons}</td>
                  <td className="num p-3 text-right text-danger">{e.summary.nonCompliant}</td>
                  <td className="num p-3 text-right text-unsure">{e.summary.unsure}</td>
                  <td className="num p-3 text-right font-semibold">{e.summary.compliancePct === null ? '–' : `${e.summary.compliancePct}%`}</td>
                  <td className="p-3 text-right">
                    <button type="button" className="text-muted hover:text-ink" aria-label={`PDF report for ${e.label}`}
                      onClick={async () => download('sitesafe-report.pdf', await buildPdf({ entry: e }))}><FileText size={16} /></button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Panel>
    </div>
  );
}
