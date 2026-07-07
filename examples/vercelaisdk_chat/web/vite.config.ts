import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The Go server owns /api/chat; the dev proxy keeps the client same-origin
// (no CORS handling needed anywhere).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8080',
    },
  },
});
