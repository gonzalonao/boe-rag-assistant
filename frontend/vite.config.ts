import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Vite config. The API base URL is injected at build time via VITE_API_BASE_URL
// (the deployed Hugging Face Space); in dev it defaults to the local uvicorn.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
});
