import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';


export default defineConfig({
  plugins: [react(), tailwindcss()],
  // base: '/',              // прод-сборка на корне
  // server: {
  //   port: 5173,
  //   strictPort: true,
  //   host: true,
  //   proxy: {
  //     // все API-запросы в деве через бэк на 8080
  //     '/api': {
  //       target: 'http://77.37.182.253:8080',
  //       changeOrigin: true,
  //       ws: true,
  //       rewrite: (path) => path.replace(/^\/api/, ''), // СНИМАЕМ /api
  //     },
  //   },
  // },
});
