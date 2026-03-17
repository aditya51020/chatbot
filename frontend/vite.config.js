import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      },
      '/status': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/chat': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/upload': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/scan': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/reset': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/docs': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    }
  }
})
