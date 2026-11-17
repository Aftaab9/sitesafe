import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';
import { VitePWA } from 'vite-plugin-pwa';
import { createHash } from 'node:crypto';
import { existsSync, readdirSync, readFileSync } from 'node:fs';

// A short content hash of every model file. It is appended to the model URLs (?v=...) so that a
// retrained model is downloaded again instead of being served from the browser's model cache.
const modelDir = new URL('./public/models/', import.meta.url);
const MODEL_VERSIONS: Record<string, string> = {};
if (existsSync(modelDir)) {
  for (const f of readdirSync(modelDir)) {
    if (f.endsWith('.onnx')) MODEL_VERSIONS[f] = createHash('sha1').update(readFileSync(new URL(f, modelDir))).digest('hex').slice(0, 10);
  }
}

// Cross-origin isolation is needed for multi-threaded WASM (SharedArrayBuffer).
// In production the same headers come from vercel.json / public/_headers.
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  base: './', // relative, so the same build works on Vercel, Cloudflare Pages and GitHub Pages
  resolve: { alias: { '@': '/src' } },
  define: { __MODEL_VERSIONS__: JSON.stringify(MODEL_VERSIONS) },
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['favicon.svg'],
      manifest: {
        name: 'SiteSafe: PPE compliance with privacy',
        short_name: 'SiteSafe',
        description: 'On-device PPE compliance checks. Your images never leave this device.',
        theme_color: '#0b0d10',
        background_color: '#0b0d10',
        display: 'standalone',
        start_url: './',
        scope: './',
        icons: [
          { src: 'icon-192.png', sizes: '192x192', type: 'image/png' },
          { src: 'icon-512.png', sizes: '512x512', type: 'image/png', purpose: 'any maskable' },
        ],
      },
      workbox: {
        globPatterns: ['**/*.{js,css,html,svg,png,woff2,json}'],
        globIgnores: ['**/ort/**', '**/models/**'],
        maximumFileSizeToCacheInBytes: 6 * 1024 * 1024,
        navigateFallback: 'index.html',
        runtimeCaching: [
          {
            // model + runtime binaries: large and immutable, cache on first use
            urlPattern: /\.(?:onnx|wasm|mjs)$/,
            handler: 'CacheFirst',
            options: { cacheName: 'sitesafe-runtime', expiration: { maxEntries: 30 } },
          },
          {
            urlPattern: /\/(?:samples|weather|heatmaps|clips)\//,
            handler: 'CacheFirst',
            options: { cacheName: 'sitesafe-samples', expiration: { maxEntries: 300 } },
          },
        ],
      },
    }),
  ],
  worker: { format: 'es' },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  server: { headers: isolation, port: 5173 },
  preview: { headers: isolation, port: 4173 },
  build: { target: 'es2022', chunkSizeWarningLimit: 1500 },
  test: { environment: 'node', include: ['src/**/*.test.ts'] },
});
