import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'fs'
import { resolve, dirname } from 'path'
import { homedir } from 'os'
import { fileURLToPath } from 'url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const sslDir = resolve(homedir(), '.localhost-ssl')

function getHttpsConfig() {
  try {
    return {
      key: fs.readFileSync(resolve(sslDir, 'localhost.key')),
      cert: fs.readFileSync(resolve(sslDir, 'localhost.pem')),
    }
  } catch {
    return false
  }
}

// https://vite.dev/config/
export default defineConfig({
  base: './',
  plugins: [react()],
  server: {
    https: getHttpsConfig(),
  },
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        benchmark: resolve(__dirname, 'benchmark.html'),
      },
    },
  },
})
