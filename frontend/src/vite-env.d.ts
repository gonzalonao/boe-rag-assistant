/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of the BOE RAG JSON API (the deployed HF Space). Empty = same origin. */
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
