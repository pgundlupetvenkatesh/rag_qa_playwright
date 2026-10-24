import { defineConfig } from '@playwright/test';

/**
 * Projects, from no dependencies to the most:
 *
 * - `dataset`: pure Node assertions over the golden dataset (Milestone 1).
 * - `client`: the RAG client adapter against a local mock server; offline.
 * - `service`: a running RAG service at `RAG_BASE_URL`, no LLM calls.
 * - `groq`: tests tagged `@groq`; the service must also hold a working Groq key.
 *
 * `service` and `groq` first run `service-setup`, which ingests the golden
 * passages (idempotent). No browser project yet; UI projects land with the UI adapter.
 */
export default defineConfig({
  testDir: './tests',
  fullyParallel: true,
  forbidOnly: Boolean(process.env['CI']),
  retries: 0,
  reporter: process.env['CI'] ? [['list'], ['html', { open: 'never' }]] : [['list']],
  projects: [
    { name: 'dataset', testDir: './tests/dataset' },
    { name: 'client', testDir: './tests/client' },
    { name: 'service-setup', testDir: './tests/integration', testMatch: /.*\.setup\.ts/ },
    {
      name: 'service',
      testDir: './tests/integration',
      grepInvert: /@groq/,
      dependencies: ['service-setup'],
    },
    {
      name: 'groq',
      testDir: './tests/integration',
      grep: /@groq/,
      dependencies: ['service-setup'],
    },
  ],
});
