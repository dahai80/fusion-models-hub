import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// issue #56: dev server 11475 (architecture/port-registry.yaml). Proxies the
// hub API surface to the FastAPI server on 11444 so standalone mode mirrors
// the embedded gateway /shell/fusion-models-hub/ layout (same-origin).
// FUSION_DEV_BACKEND / FUSION_MODEL_HUB_WEB_PORT override for parity.
const backend = process.env.FUSION_DEV_BACKEND ?? 'http://127.0.0.1:11444';
const port = Number(process.env.FUSION_MODEL_HUB_WEB_PORT ?? 11475);

const apiPaths = ['/api', '/health', '/metrics'];

export default defineConfig({
    plugins: [react()],
    // Embedded deployment serves dist through the gateway sub-path
    // /shell/fusion-models-hub/ — relative base + hash router works in both
    // embedded and standalone modes.
    base: './',
    server: {
        port,
        strictPort: true,
        proxy: Object.fromEntries(
            apiPaths.map((p) => [p, { target: backend, changeOrigin: true }])
        ),
    },
    preview: {
        port,
        strictPort: true,
        proxy: Object.fromEntries(
            apiPaths.map((p) => [p, { target: backend, changeOrigin: true }])
        ),
    },
    envPrefix: ['VITE_', 'FUSION_'],
});
