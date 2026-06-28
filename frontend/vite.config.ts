import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Vite config.
// - VITE_API_BASE_URL (build-time) points at the deployed HF Space API.
// - VITE_BASE sets the public base path; GitHub Pages serves this project site
//   under /boe-rag-assistant/, so the deploy workflow passes that. Defaults to
//   "/" for local dev and other hosts (Vercel/Netlify serve from the root).
export default defineConfig({
  base: process.env.VITE_BASE ?? "/",
  plugins: [react()],
  server: {
    port: 5173,
  },
});
