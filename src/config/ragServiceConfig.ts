/** Base URL used when `RAG_BASE_URL` is unset: the local reference service. */
export const DEFAULT_RAG_BASE_URL = 'http://127.0.0.1:8000';

/**
 * Per-request timeout used when `RAG_TIMEOUT_MS` is unset. Must exceed the
 * service's own provider timeout (30 s by default) so the service reports a
 * slow model as a structured error before the client gives up.
 */
export const DEFAULT_RAG_TIMEOUT_MS = 60_000;

/** Where the RAG service under test lives and how long to wait for it. */
export interface RagServiceConfig {
  readonly baseUrl: string;
  readonly timeoutMs: number;
}

/**
 * Reads the RAG service settings from environment variables.
 *
 * - `RAG_BASE_URL`: service base URL, default {@link DEFAULT_RAG_BASE_URL}.
 * - `RAG_TIMEOUT_MS`: per-request timeout, default {@link DEFAULT_RAG_TIMEOUT_MS}.
 *
 * @throws Error if either variable is set to an invalid value.
 */
export function loadRagServiceConfig(env: NodeJS.ProcessEnv = process.env): RagServiceConfig {
  const baseUrl = env['RAG_BASE_URL']?.trim() || DEFAULT_RAG_BASE_URL;
  if (!URL.canParse(baseUrl)) {
    throw new Error(`RAG_BASE_URL is not a valid URL: ${JSON.stringify(baseUrl)}`);
  }

  const rawTimeout = env['RAG_TIMEOUT_MS']?.trim();
  const timeoutMs = rawTimeout ? Number(rawTimeout) : DEFAULT_RAG_TIMEOUT_MS;
  if (!Number.isInteger(timeoutMs) || timeoutMs <= 0) {
    throw new Error(`RAG_TIMEOUT_MS must be a positive integer, got ${JSON.stringify(rawTimeout)}`);
  }

  return { baseUrl, timeoutMs };
}
