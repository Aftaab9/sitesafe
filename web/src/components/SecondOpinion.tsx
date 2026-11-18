// Optional: ask a small vision-language model (LFM3-Omni in Ollama, on this device) about each worker
// the detector found, and show where the two agree. For comparison and demos; the detector stays in charge.
import { Bot, Loader2 } from 'lucide-react';
import { useEffect, useState } from 'react';
import type { Analysis } from '../lib/analyze';
import type { ItemState } from '../lib/matching';
import { askWorker, findOllamaModel, type Answer, type VlmAnswer } from '../lib/ollama';
import { Button, Chip, cn } from './ui';

const DET: Record<ItemState, Answer> = { present: 'yes', missing: 'no', unsure: 'unsure', not_visible: 'unsure' };
const MARK: Record<Answer, string> = { yes: '✓', no: '✗', unsure: '?' };
const TONE: Record<Answer, string> = { yes: 'text-ok', no: 'text-danger', unsure: 'text-info' };

function Cell({ a }: { a: Answer }) {
  return <span className={cn('font-mono font-bold', TONE[a])}>{MARK[a]}</span>;
}

export function SecondOpinion({ source, analysis }: { source: HTMLCanvasElement | null; analysis: Analysis | null }) {
  const [model, setModel] = useState<string | null | undefined>(undefined); // undefined = still checking
  const [answers, setAnswers] = useState<(VlmAnswer | null)[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { void findOllamaModel().then(setModel); }, []);
  useEffect(() => { setAnswers([]); setError(null); }, [analysis]);

  const people = analysis?.people ?? [];
  const run = async () => {
    if (!model || !source) return;
    setBusy(true); setError(null);
    const out: (VlmAnswer | null)[] = people.map(() => null);
    setAnswers([...out]);
    try {
      for (let i = 0; i < people.length; i++) {
        out[i] = await askWorker(model, source, people[i].box);
        setAnswers([...out]);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Ollama did not answer');
    } finally {
      setBusy(false);
    }
  };

  const done = answers.filter(Boolean) as VlmAnswer[];
  const agree = done.length
    ? people.slice(0, answers.length).filter((p, i) => answers[i]
      && DET[p.items.helmet.state] === answers[i]!.hard_hat && DET[p.items.vest.state] === answers[i]!.hi_vis).length
    : 0;

  return (
    <div className="mt-4 border-t border-line pt-3">
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-1.5 text-sm font-medium"><Bot size={15} /> Second opinion</span>
        <Chip>{model ? model : 'local Ollama'}</Chip>
      </div>
      <p className="mt-1 text-xs text-muted">
        A small vision-language model (LFM3-Omni, 1.7 B) running in Ollama on this computer looks at each worker the detector found.
        Crops go to localhost only.
      </p>
      {model === undefined && <p className="mt-2 text-xs text-muted">Looking for Ollama on this device…</p>}
      {model === null && (
        <details className="mt-2 text-xs text-muted">
          <summary className="cursor-pointer">Not available on this device (how to enable)</summary>
          <p className="mt-1">Install Ollama, run <code>ollama pull lfm3-omni</code>, allow this site with
            <code className="mx-1 break-all">setx OLLAMA_ORIGINS "{window.location.origin}"</code>
            (Windows) and restart Ollama. Chrome may ask to allow access to devices on your local network.</p>
        </details>
      )}
      {model && (
        <Button className="mt-2 w-full" disabled={!analysis || !people.length || busy} onClick={() => void run()}>
          {busy ? <Loader2 size={16} className="animate-spin" /> : <Bot size={16} />}
          {busy ? `Asking… ${done.length}/${people.length}` : `Ask about ${people.length} worker${people.length === 1 ? '' : 's'}`}
        </Button>
      )}
      {error && <p className="mt-2 text-xs text-danger">{error}. Is Ollama running and OLLAMA_ORIGINS set?</p>}
      {answers.length > 0 && (
        <>
          <table className="mt-3 w-full text-xs">
            <thead className="text-muted">
              <tr><th className="text-left font-normal">Worker</th><th className="font-normal" colSpan={2}>Detector</th><th className="font-normal" colSpan={2}>LFM3-Omni</th><th /></tr>
              <tr><th /><th className="font-normal">helmet</th><th className="font-normal">vest</th><th className="font-normal">helmet</th><th className="font-normal">vest</th><th /></tr>
            </thead>
            <tbody>
              {people.map((p, i) => {
                const a = answers[i];
                const dh = DET[p.items.helmet.state], dv = DET[p.items.vest.state];
                return (
                  <tr key={i} className="border-t border-line text-center">
                    <td className="py-1 text-left font-mono">W{i + 1}</td>
                    <td><Cell a={dh} /></td><td><Cell a={dv} /></td>
                    <td>{a ? <Cell a={a.hard_hat} /> : busy ? '…' : '–'}</td>
                    <td>{a ? <Cell a={a.hi_vis} /> : busy ? '…' : '–'}</td>
                    <td className="text-muted">{a ? (a.hard_hat === dh && a.hi_vis === dv ? 'agree' : 'differ') : ''}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {done.length > 0 && (
            <p className="mt-2 text-xs text-muted">
              Agree on {agree} of {done.length} · about {(done.reduce((s, a) => s + a.ms, 0) / done.length / 1000).toFixed(1)} s per worker
              (the detector takes about 0.1 s for the whole photo). Where they differ, look yourself.
            </p>
          )}
        </>
      )}
    </div>
  );
}
