import { Link } from 'react-router-dom';
import { BadgeCheck, HelpCircle, ShieldAlert } from 'lucide-react';
import { PageHeader, Panel } from '../components/ui';

function Q({ q, children }: { q: string; children: React.ReactNode }) {
  return (
    <details className="group border-b border-line py-3 last:border-0">
      <summary className="cursor-pointer list-none font-medium marker:hidden">
        <span className="mr-2 inline-block text-hivis transition-transform group-open:rotate-90">▸</span>{q}
      </summary>
      <div className="mt-2 space-y-2 pl-5 text-sm text-muted">{children}</div>
    </details>
  );
}

export default function Help() {
  return (
    <div className="mx-auto max-w-3xl space-y-4 px-4 py-6">
      <PageHeader eyebrow="Help · how it works" title="A short guide for safety officers">
        SiteSafe looks at a picture of your site and tells you, worker by worker, whether the required protective equipment is visible.
      </PageHeader>

      <Panel className="p-5">
        <h2 className="text-lg font-semibold">Reading the results</h2>
        <ul className="mt-3 space-y-3 text-sm">
          <li className="flex gap-3"><BadgeCheck className="mt-0.5 shrink-0 text-ok" size={19} /><span><strong>Green, solid box, tick: compliant.</strong> Every item required by the site profile was seen on this person.</span></li>
          <li className="flex gap-3"><ShieldAlert className="mt-0.5 shrink-0 text-danger" size={19} /><span><strong>Red, dashed box, cross: PPE missing.</strong> The model saw that a required item is not being worn. The card lists which one.</span></li>
          <li className="flex gap-3"><HelpCircle className="mt-0.5 shrink-0 text-unsure" size={19} /><span><strong>Blue, dotted box, question mark: check manually.</strong> The model is not confident, or a required item is out of view. This is not an error. It is the model handing the decision to you.</span></li>
        </ul>
        <p className="mt-3 text-sm text-muted">Status never depends on colour alone: each one has its own line style, symbol and label.</p>
      </Panel>

      <Panel className="p-5">
        <h2 className="text-lg font-semibold">Three ways to check</h2>
        <ol className="mt-3 list-decimal space-y-2 pl-5 text-sm text-muted">
          <li><Link className="text-ink underline decoration-hivis underline-offset-4" to="/analyze">Photo</Link>: upload or take a photo. Hover or tap a worker to see which PPE boxes were matched to them.</li>
          <li><Link className="text-ink underline decoration-hivis underline-offset-4" to="/video">Clip</Link>: a 10 to 30 second video. Each worker keeps an anonymous number, and a violation is only raised if it lasts for several frames.</li>
          <li><Link className="text-ink underline decoration-hivis underline-offset-4" to="/live">Live</Link>: real-time camera view with a running compliance gauge.</li>
        </ol>
      </Panel>

      <Panel className="p-5">
        <h2 className="text-lg font-semibold">Questions</h2>
        <div className="mt-2">
          <Q q="Where do my photos go?">
            <p>Nowhere. The model is downloaded once to your browser and runs on your own phone or laptop. There is no server that receives images, and nothing is saved when you close the tab. You can switch to flight mode after the first load and it still works.</p>
          </Q>
          <Q q="Why are faces blurred?">
            <p>The question is whether PPE is worn, not who is wearing it. Faces are blurred before anything is shown or exported. With “Strict privacy” on, the head area is also blurred when no face could be found, for example when someone is far away or turned around.</p>
            <p>The privacy slider on the Photo page lets you compare the original and the blurred picture on your own device.</p>
          </Q>
          <Q q="What is a site profile?">
            <p>It sets which PPE is mandatory. “General site” needs a helmet and a hi-vis vest. “Full PPE” adds gloves and boots. “Cutting / grinding” adds goggles. Choose the one that matches the work area before you read the result.</p>
          </Q>
          <Q q="How do danger zones work?">
            <p>Press “Draw danger zone” and tap the corners of the area on the picture, for example around a forklift route. Anyone whose feet are inside the zone is flagged. Separately, anyone standing close to a detected forklift is flagged; you can change that distance in the settings.</p>
            <p>The camera cannot measure real distance, so treat these alerts as a prompt to look.</p>
          </Q>
          <Q q="How accurate is it?">
            <p>It is good at people, helmets and vests in clear daylight photos, and weaker on small items such as goggles and gloves, on distant workers, and at night or in bad weather. The <Link className="text-ink underline" to="/model-card">model card</Link> has the measured numbers and a gallery of real failures.</p>
          </Q>
          <Q q="What does a mistake cost?">
            <p>A missed “no helmet” can mean a head injury, so the model is tuned to prefer “check manually” over a wrong green. A false alarm costs a minute of your time. That is the trade-off behind the “not sure” state.</p>
          </Q>
          <Q q="Can I use this to discipline someone?">
            <p>No. The tool reports aggregate figures and anonymous numbers that reset with every check. It can be wrong about any single person, it does not identify anyone, and it must not be used for individual monitoring or disciplinary action.</p>
          </Q>
          <Q q="What should I tell the workers?">
            <p>Tell them before you start: that this is a PPE check, that faces are blurred, that nothing is recorded or uploaded, and that the result is a site-level figure used to improve safety, not a record about them.</p>
          </Q>
          <Q q="It is slow or will not start. What can I do?">
            <p>The first visit downloads about 11 MB. After that it loads from your device. On older phones the live view may run at a few frames per second; photos still take well under five seconds. If the model fails to start, the page offers a “Use CPU engine” button.</p>
          </Q>
          <Q q="What is the written summary?">
            <p>An optional small language model that describes the scene in a sentence or two. Because such models can make things up, every number it states is checked against the detector, and anything that does not match is corrected or removed. It needs a one-off download of about 200 MB, so it is off until you press the button.</p>
          </Q>
        </div>
      </Panel>

      <Panel className="p-5 text-sm text-muted">
        <h2 className="text-lg font-semibold text-ink">Keyboard</h2>
        <p className="mt-2">All controls are reachable with Tab. On the privacy slider use the arrow keys. When drawing a zone: Enter closes it, Backspace removes the last corner, Esc cancels.</p>
      </Panel>
    </div>
  );
}
