import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Camera, CloudOff, EyeOff, Film, HardHat, Lock, ScanSearch, Swords, UserCheck, Wand2 } from 'lucide-react';
import { Button, Eyebrow, Panel } from '../components/ui';
import { STATUS_COLOR, STATUS_SYMBOL } from '../lib/draw';
import { asset } from '../lib/engine';
import type { PersonStatus } from '../lib/matching';
import { prefersReducedMotion } from '../lib/store';

interface Hero { file: string; w: number; h: number; people: { box: [number, number, number, number]; status: PersonStatus; missing: string[] }[] }

/** Looping demo: scan line, then boxes lock on one by one. Uses a pre-blurred sample image. */
function HeroDemo() {
  const [hero, setHero] = useState<Hero | null>(null);
  const [cycle, setCycle] = useState(0);
  const reduce = prefersReducedMotion();
  useEffect(() => { fetch(asset('samples/hero.json')).then((r) => (r.ok ? r.json() : null)).then(setHero).catch(() => undefined); }, []);
  useEffect(() => {
    if (reduce) return;
    const iv = setInterval(() => setCycle((c) => c + 1), 6500);
    return () => clearInterval(iv);
  }, [reduce]);
  if (!hero) return <div className="hud-corners aspect-[4/3] rounded bg-surface" />;
  const done = hero.people.filter((p) => p.status === 'compliant').length;
  const decided = hero.people.filter((p) => p.status !== 'unsure').length;
  return (
    <div className="hud-corners relative overflow-hidden rounded bg-black shadow-2xl" style={{ aspectRatio: `${hero.w} / ${hero.h}` }}>
      <img src={asset(hero.file)} alt="Construction workers on site, faces blurred, each outlined by compliance status" className="absolute inset-0 h-full w-full object-cover" />
      {!reduce && <motion.div key={`scan-${cycle}`} className="absolute inset-x-0 h-[5%] bg-gradient-to-b from-transparent to-[#ffd400]/70" initial={{ top: '-5%' }} animate={{ top: '100%' }} transition={{ duration: 1.3, ease: 'easeInOut' }} />}
      {hero.people.map((p, i) => (
        <motion.div
          key={`${cycle}-${i}`}
          className="absolute border-2"
          style={{
            left: `${(100 * p.box[0]) / hero.w}%`, top: `${(100 * p.box[1]) / hero.h}%`,
            width: `${(100 * (p.box[2] - p.box[0])) / hero.w}%`, height: `${(100 * (p.box[3] - p.box[1])) / hero.h}%`,
            borderColor: STATUS_COLOR[p.status], borderStyle: p.status === 'compliant' ? 'solid' : p.status === 'non_compliant' ? 'dashed' : 'dotted',
          }}
          initial={reduce ? false : { opacity: 0, scale: 1.3 }} animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: reduce ? 0 : 1.3 + i * 0.22, type: 'spring', stiffness: 380, damping: 20 }}
        >
          <span className="absolute -top-px left-0 -translate-y-full whitespace-nowrap px-1.5 py-px font-mono text-[10px] font-bold text-[#0b0d10] sm:text-[11px]" style={{ background: STATUS_COLOR[p.status] }}>
            {STATUS_SYMBOL[p.status]} W{i + 1}{p.missing.length ? ` · no ${p.missing.join(', ')}` : ''}
          </span>
        </motion.div>
      ))}
      <div className="absolute bottom-2 left-2 rounded bg-[#0b0d10]/85 px-2.5 py-1.5 font-mono text-[11px] text-white">
        compliance <span className="font-bold text-[#ffd400]">{decided ? Math.round((100 * done) / decided) : '–'}%</span> · {hero.people.length} workers · faces blurred
      </div>
    </div>
  );
}

const STEPS = [
  { icon: ScanSearch, title: 'Show it the site', text: 'Upload a photo, a 10 to 30 second clip, or point your camera.' },
  { icon: HardHat, title: 'It checks every worker', text: 'Helmet, vest, gloves, boots and goggles are matched to each person. Green, red or “not sure”.' },
  { icon: UserCheck, title: 'You decide', text: 'You get a compliance figure, a list of what is missing and a report. The judgement stays with you.' },
];

const MODES = [
  { to: '/video', icon: Film, title: 'Clip check', text: 'Anonymous worker tracking and a violation timeline you can scrub.' },
  { to: '/challenge', icon: Swords, title: 'Inspector Challenge', text: 'You against the model on the same photos, against the clock.' },
  { to: '/weather', icon: Wand2, title: 'Weather Lab', text: 'See generated night, rain, fog and dust versions and how the detector copes.' },
];

export default function Landing() {
  return (
    <div className="blueprint">
      <section className="mx-auto grid max-w-7xl items-center gap-8 px-4 pb-12 pt-10 lg:grid-cols-[1fr_1.1fr] lg:pt-16">
        <div>
          <Eyebrow>PPE compliance · on-device computer vision</Eyebrow>
          <h1 className="mt-3 text-4xl font-bold leading-[1.05] sm:text-5xl lg:text-6xl">
            See who is missing PPE.<br /><span className="bg-hivis px-2 text-[#0b0d10]">Without seeing who they are.</span>
          </h1>
          <p className="mt-5 max-w-lg text-lg text-muted">
            SiteSafe checks hard hats, vests, gloves, boots and goggles in site photos, clips and live video. Faces are blurred, and your images never leave this device.
          </p>
          <div className="mt-7 flex flex-wrap gap-3">
            <Link to="/analyze?sample=first"><Button variant="primary" size="lg"><ScanSearch size={19} /> Try a sample</Button></Link>
            <Link to="/live"><Button size="lg"><Camera size={19} /> Use my camera</Button></Link>
          </div>
          <p className="mt-4 font-mono text-xs text-muted">Free · no sign-up · works offline after the first visit</p>
        </div>
        <HeroDemo />
      </section>

      <div className="h-2 hazard-stripes opacity-80" aria-hidden />

      <section className="bg-bg">
        <div className="mx-auto max-w-7xl px-4 py-12">
          <Eyebrow>How it works</Eyebrow>
          <div className="mt-4 grid gap-4 md:grid-cols-3">
            {STEPS.map((s, i) => (
              <Panel key={s.title} className="p-5">
                <div className="flex items-center gap-3">
                  <span className="num grid h-9 w-9 place-items-center rounded bg-hivis text-lg font-bold text-[#0b0d10]">{i + 1}</span>
                  <s.icon size={20} className="text-muted" />
                </div>
                <h2 className="mt-4 text-lg font-semibold">{s.title}</h2>
                <p className="mt-1 text-sm text-muted">{s.text}</p>
              </Panel>
            ))}
          </div>

          <Panel className="mt-6 grid gap-6 p-6 md:grid-cols-[1.2fr_1fr] md:items-center">
            <div>
              <Eyebrow>The privacy promise</Eyebrow>
              <h2 className="mt-2 text-2xl font-bold sm:text-3xl">Your images never leave this device.</h2>
              <p className="mt-3 text-sm text-muted">
                The detector runs inside your browser. There is no server to send pictures to, no account, and nothing is saved when you close the tab. Reports contain counts and anonymous worker numbers, never names or faces.
              </p>
            </div>
            <ul className="space-y-3 text-sm">
              <li className="flex items-center gap-3"><CloudOff size={19} className="shrink-0 text-hivis" /> No upload: all processing is on-device</li>
              <li className="flex items-center gap-3"><EyeOff size={19} className="shrink-0 text-hivis" /> Faces blurred before anything is shown or exported</li>
              <li className="flex items-center gap-3"><Lock size={19} className="shrink-0 text-hivis" /> Nothing stored; aggregate statistics only</li>
              <li className="flex items-center gap-3"><UserCheck size={19} className="shrink-0 text-hivis" /> Supports the safety officer, never used to discipline individuals</li>
            </ul>
          </Panel>

          <Eyebrow className="mt-10">More to try</Eyebrow>
          <div className="mt-4 grid gap-4 md:grid-cols-3">
            {MODES.map((m) => (
              <Link key={m.to} to={m.to} className="group">
                <Panel className="h-full p-5 transition-colors group-hover:border-hivis">
                  <m.icon size={22} className="text-hivis" />
                  <h3 className="mt-3 font-semibold">{m.title}</h3>
                  <p className="mt-1 text-sm text-muted">{m.text}</p>
                </Panel>
              </Link>
            ))}
          </div>
          <p className="mt-8 text-sm text-muted">
            New here? Read <Link to="/help" className="underline decoration-hivis underline-offset-4">how it works</Link> or the{' '}
            <Link to="/model-card" className="underline decoration-hivis underline-offset-4">model card</Link> for what the model can and cannot do.
          </p>
        </div>
      </section>
    </div>
  );
}
