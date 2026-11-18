// Loading user media into memory. Nothing here touches the network for user files.
import { asset } from './engine';
import type { Det } from './matching';

export const MAX_SIDE = 1600;

export function toCanvas(src: CanvasImageSource, w: number, h: number, maxSide = MAX_SIDE): HTMLCanvasElement {
  const r = Math.min(1, maxSide / Math.max(w, h));
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(w * r);
  canvas.height = Math.round(h * r);
  canvas.getContext('2d')!.drawImage(src, 0, 0, canvas.width, canvas.height);
  return canvas;
}

export async function fileToCanvas(file: Blob): Promise<HTMLCanvasElement> {
  // imageOrientation: respect EXIF rotation from phone cameras
  const bmp = await createImageBitmap(file, { imageOrientation: 'from-image' });
  const c = toCanvas(bmp, bmp.width, bmp.height);
  bmp.close();
  return c;
}

export async function urlToCanvas(url: string): Promise<HTMLCanvasElement> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not load ${url}`);
  return fileToCanvas(await res.blob());
}

export interface Sample {
  id: string;
  file: string;
  title: string;
  tags: string[];
  failure: boolean;
  note: string;
  heatmap: string | null;
  w: number;
  h: number;
  gt: Det[];
  /** Real photos from Wikimedia Commons: author and licence, shown with the photo. */
  credit?: { title: string; author: string; licence: string; licence_url: string; page: string };
}
export interface WeatherItem {
  id: string;
  title: string;
  w: number;
  h: number;
  variants: Record<string, string>;
  gt: Det[];
}
export interface ClipItem { id: string; file: string; title: string; note: string; poster: string }

interface Catalogue {
  licence: string;
  samples: Sample[];
  real?: Sample[];
  weather: WeatherItem[];
  clips: ClipItem[];
  models: { synth: boolean };
}

let cataloguePromise: Promise<Catalogue> | null = null;
export function loadCatalogue(): Promise<Catalogue> {
  cataloguePromise ??= fetch(asset('samples/catalogue.json')).then((r) => {
    if (!r.ok) throw new Error('Sample catalogue missing');
    return r.json();
  });
  return cataloguePromise;
}

export function download(name: string, blob: Blob) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

let audio: AudioContext | null = null;
/** Short two-tone cue for a new violation. Off by default; toggled in settings. */
export function beep(kind: 'violation' | 'clear' = 'violation') {
  try {
    audio ??= new AudioContext();
    const t = audio.currentTime;
    const tones = kind === 'violation' ? [660, 440] : [523, 784];
    tones.forEach((f, i) => {
      const o = audio!.createOscillator();
      const g = audio!.createGain();
      o.type = 'sine';
      o.frequency.value = f;
      g.gain.setValueAtTime(0.0001, t + i * 0.13);
      g.gain.exponentialRampToValueAtTime(0.12, t + i * 0.13 + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, t + i * 0.13 + 0.12);
      o.connect(g).connect(audio!.destination);
      o.start(t + i * 0.13);
      o.stop(t + i * 0.13 + 0.14);
    });
  } catch { /* audio not available */ }
}
