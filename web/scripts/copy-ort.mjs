// Copies the ONNX Runtime Web WASM binaries into public/ort so they are served from our own origin
// (no CDN: the app must work offline and under COEP require-corp).
import { cpSync, mkdirSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const src = join(root, 'node_modules', 'onnxruntime-web', 'dist');
const dest = join(root, 'public', 'ort');
mkdirSync(dest, { recursive: true });
for (const f of readdirSync(src)) {
  if (/^ort-wasm-simd-threaded(\.jsep)?\.(wasm|mjs)$/.test(f)) cpSync(join(src, f), join(dest, f));
}
console.log('copied onnxruntime-web wasm files to public/ort');
