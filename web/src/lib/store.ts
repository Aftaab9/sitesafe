// App settings and the session log. Only aggregate numbers are ever stored: no images, no faces.
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Item, ProfileId } from './classes';
import type { Summary } from './matching';

export interface Settings {
  profile: ProfileId;
  conf: number;
  strictBlur: boolean;
  blurFaces: boolean;
  sound: boolean;
  forkliftDistance: number; // in person-heights
  theme: 'dark' | 'light';
  reducedMotion: boolean;
  /** which detector runs on every page (all run on this device) */
  detector: 'base' | 'synth' | 'plus' | 'lfm';
  set: (patch: Partial<Omit<Settings, 'set'>>) => void;
}

export const useSettings = create<Settings>()(
  persist(
    (set) => ({
      profile: 'general',
      conf: 0.25,
      strictBlur: true,
      blurFaces: true,
      sound: false,
      forkliftDistance: 0.5,
      theme: 'dark',
      reducedMotion: false,
      detector: 'base',
      set: (patch) => set(patch),
    }),
    { name: 'sitesafe-settings' },
  ),
);

export interface LogEntry {
  id: string;
  time: number;
  source: 'photo' | 'video' | 'live';
  label: string;
  profile: ProfileId;
  summary: Summary;
  zoneEvents: number;
  forkliftEvents: number;
  ms: number;
  /** video only: violations over time, one point per sampled frame */
  timeline?: { t: number; persons: number; nonCompliant: number }[];
  /** video only: violations per anonymous worker id */
  workers?: { id: number; missing: Item[]; status: string }[];
}

interface SessionLog {
  entries: LogEntry[];
  add: (e: LogEntry) => void;
  clear: () => void;
}
// Deliberately NOT persisted: the log lives in memory and disappears when the tab closes ("retain no data").
export const useLog = create<SessionLog>((set) => ({
  entries: [],
  add: (e) => set((s) => ({ entries: [e, ...s.entries].slice(0, 200) })),
  clear: () => set({ entries: [] }),
}));

export function applyTheme(theme: 'dark' | 'light') {
  document.documentElement.classList.toggle('dark', theme === 'dark');
  try { localStorage.setItem('sitesafe-theme', theme); } catch { /* ignore */ }
}

export function prefersReducedMotion(): boolean {
  return useSettings.getState().reducedMotion || window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}
