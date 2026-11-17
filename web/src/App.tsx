import { lazy, Suspense, useEffect, useState } from 'react';
import { NavLink, Route, Routes, useLocation } from 'react-router-dom';
import {
  BarChart3, Camera, CloudRain, Film, HardHat, HelpCircle, Menu, Moon, ScanSearch, ShieldCheck, Sun, Swords, X,
} from 'lucide-react';
import { applyTheme, useSettings } from './lib/store';
import { cn } from './components/ui';
import { EngineBadge } from './components/EngineStatus';

const Landing = lazy(() => import('./pages/Landing'));
const Analyze = lazy(() => import('./pages/Analyze'));
const Live = lazy(() => import('./pages/Live'));
const Video = lazy(() => import('./pages/Video'));
const Challenge = lazy(() => import('./pages/Challenge'));
const WeatherLab = lazy(() => import('./pages/WeatherLab'));
const Dashboard = lazy(() => import('./pages/Dashboard'));
const ModelCard = lazy(() => import('./pages/ModelCard'));
const Help = lazy(() => import('./pages/Help'));

const NAV = [
  { to: '/analyze', label: 'Photo', icon: ScanSearch },
  { to: '/live', label: 'Live', icon: Camera },
  { to: '/video', label: 'Clip', icon: Film },
  { to: '/dashboard', label: 'Dashboard', icon: BarChart3 },
  { to: '/challenge', label: 'Challenge', icon: Swords },
  { to: '/weather', label: 'Weather Lab', icon: CloudRain },
  { to: '/model-card', label: 'Model card', icon: ShieldCheck },
  { to: '/help', label: 'Help', icon: HelpCircle },
];

function Header() {
  const { theme, set } = useSettings();
  const [open, setOpen] = useState(false);
  const loc = useLocation();
  useEffect(() => { setOpen(false); }, [loc.pathname]);
  const toggle = () => {
    const next = theme === 'dark' ? 'light' : 'dark';
    set({ theme: next });
    applyTheme(next);
  };
  return (
    <header className="no-print sticky top-0 z-40 border-b border-line bg-bg/90 backdrop-blur">
      <div className="h-1 hazard-stripes" aria-hidden />
      <div className="mx-auto flex h-14 max-w-7xl items-center gap-3 px-4">
        <NavLink to="/" className="flex items-center gap-2 font-display text-lg font-bold">
          <span className="grid h-8 w-8 place-items-center rounded bg-hivis text-[#0b0d10]"><HardHat size={19} strokeWidth={2.4} /></span>
          SiteSafe
        </NavLink>
        <nav aria-label="Main" className="ml-4 hidden items-center gap-0.5 lg:flex">
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              className={({ isActive }) => cn(
                'flex items-center gap-1.5 whitespace-nowrap rounded px-2 py-1.5 text-[13px] font-medium transition-colors',
                isActive ? 'bg-raised text-ink' : 'text-muted hover:text-ink',
              )}
            >
              <n.icon size={15} /> {n.label}
            </NavLink>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-2">
          <EngineBadge />
          <button
            type="button"
            onClick={toggle}
            aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            className="grid h-9 w-9 place-items-center rounded text-muted hover:bg-raised hover:text-ink"
          >
            {theme === 'dark' ? <Sun size={17} /> : <Moon size={17} />}
          </button>
          <button
            type="button"
            onClick={() => setOpen((o) => !o)}
            aria-label="Menu"
            aria-expanded={open}
            className="grid h-9 w-9 place-items-center rounded text-muted hover:bg-raised hover:text-ink lg:hidden"
          >
            {open ? <X size={19} /> : <Menu size={19} />}
          </button>
        </div>
      </div>
      {open && (
        <nav aria-label="Mobile" className="grid grid-cols-2 gap-1 border-t border-line p-3 lg:hidden">
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              className={({ isActive }) => cn(
                'flex items-center gap-2 rounded px-3 py-3 text-sm font-medium',
                isActive ? 'bg-raised text-ink' : 'text-muted',
              )}
            >
              <n.icon size={17} /> {n.label}
            </NavLink>
          ))}
        </nav>
      )}
    </header>
  );
}

function Footer() {
  return (
    <footer className="no-print mt-16 border-t border-line">
      <div className="mx-auto max-w-7xl space-y-2 px-4 py-8 text-xs text-muted">
        <p>
          <strong className="text-ink">SiteSafe supports the safety officer. It does not replace them.</strong>{' '}
          Results are aggregate and anonymous, and must not be used to monitor or discipline individual workers.
        </p>
        <p>
          Everything runs on this device. No image, frame or face is uploaded or stored. Training data:
          Construction-PPE by Ultralytics (CC BY 4.0). Detector: YOLO11n (AGPL-3.0). Face detector: YuNet (MIT).
        </p>
        <p>MAIB AI 221 Computer Vision with Applications, S P Jain School of Global Management. Project 6.</p>
      </div>
    </footer>
  );
}

function Loading() {
  return <div className="grid min-h-[50vh] place-items-center text-sm text-muted">Loading…</div>;
}

export default function App() {
  const { theme, reducedMotion } = useSettings();
  const loc = useLocation();
  useEffect(() => { applyTheme(theme); }, [theme]);
  useEffect(() => { document.documentElement.classList.toggle('reduce-motion', reducedMotion); }, [reducedMotion]);
  useEffect(() => { window.scrollTo(0, 0); }, [loc.pathname]);
  const live = loc.pathname === '/live';
  return (
    <div className="flex min-h-dvh flex-col">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-50 focus:rounded focus:bg-hivis focus:px-3 focus:py-2 focus:text-[#0b0d10]">
        Skip to content
      </a>
      <Header />
      <main id="main" className="flex-1">
        <Suspense fallback={<Loading />}>
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/analyze" element={<Analyze />} />
            <Route path="/live" element={<Live />} />
            <Route path="/video" element={<Video />} />
            <Route path="/challenge" element={<Challenge />} />
            <Route path="/weather" element={<WeatherLab />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/model-card" element={<ModelCard />} />
            <Route path="/help" element={<Help />} />
            <Route path="*" element={<Landing />} />
          </Routes>
        </Suspense>
      </main>
      {!live && <Footer />}
    </div>
  );
}
