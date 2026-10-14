import type { APIRequestContext } from '@playwright/test';

import {
  type CorpusDocument,
  type Evidence,
  type IngestResult,
  type QueryOptions,
  type RagAnswer,
  type RagClient,
  RagClientError,
  type Readiness,
  type Retrieval,
} from './ragClient';

// Wire shapes of rag_service (camelCase JSON). Kept private: tests see only
// the RagClient types, so another adapter can produce them from another API.

interface WireChunk {
  chunkId: string;
  documentId: string;
  text: string;
  title: string | null;
  source: string | null;
  similarity: number;
}

interface WireReadiness {
  status: 'ready' | 'not_ready';
  checks: Readiness['checks'];
}

interface WireIngest {
  documentsReceived: number;
  documentsCreated: number;
  documentsReplaced: number;
  documentsUnchanged: number;
}

interface WireRetrieve {
  question: string;
  chunks: WireChunk[];
  retrievalMs: number;
}

interface WireQuery {
  question: string;
  answer: string | null;
  abstained: boolean;
  model: string;
  evidence: WireChunk[];
  usage: RagAnswer['usage'];
  timings: { retrievalMs: number; generationMs: number };
}

function toEvidence(chunk: WireChunk): Evidence {
  return {
    chunkId: chunk.chunkId,
    documentId: chunk.documentId,
    text: chunk.text,
    similarity: chunk.similarity,
    ...(chunk.title !== null && { title: chunk.title }),
    ...(chunk.source !== null && { source: chunk.source }),
  };
}

function isStageDetail(
  value: unknown,
): value is { stage: 'retrieval' | 'generation'; code: string; message: string } {
  if (typeof value !== 'object' || value === null) {
    return false;
  }
  const detail = value as Record<string, unknown>;
  return (
    (detail['stage'] === 'retrieval' || detail['stage'] === 'generation') &&
    typeof detail['code'] === 'string' &&
    typeof detail['message'] === 'string'
  );
}

/** Builds the error for a non-success response from FastAPI's `{"detail": ...}` body. */
function responseError(
  call: string,
  status: number,
  body: string,
  retryAfter: string | undefined,
): RagClientError {
  let detail: unknown = body;
  try {
    detail = (JSON.parse(body) as { detail?: unknown }).detail ?? body;
  } catch {
    // Not JSON (e.g. a proxy error page): report the raw body.
  }
  if (isStageDetail(detail)) {
    const seconds = Number(retryAfter);
    return new RagClientError(
      `${call} -> HTTP ${status} ${detail.stage}/${detail.code}: ${detail.message}`,
      {
        status,
        stage: detail.stage,
        code: detail.code,
        ...(retryAfter !== undefined && Number.isFinite(seconds) && { retryAfterSeconds: seconds }),
      },
    );
  }
  const text = typeof detail === 'string' ? detail : JSON.stringify(detail);
  return new RagClientError(`${call} -> HTTP ${status}: ${text}`, { status });
}

/**
 * {@link RagClient} adapter for the reference FastAPI service in `rag_service/`.
 *
 * Sends requests through a Playwright `APIRequestContext`, which supplies the
 * base URL and per-request timeout (see `loadRagServiceConfig`). Responses
 * are mapped to the framework's types; their shape is not validated at run
 * time, the integration tests assert it.
 */
export class RagServiceClient implements RagClient {
  constructor(private readonly request: APIRequestContext) {}

  async ready(): Promise<Readiness> {
    // 503 carries the failed checks: it is an answer, not an error.
    const body = await this.call<WireReadiness>('GET', '/ready', undefined, [200, 503]);
    return { ready: body.status === 'ready', checks: body.checks };
  }

  async ingest(documents: readonly CorpusDocument[]): Promise<IngestResult> {
    const body = await this.call<WireIngest>('POST', '/corpus/documents', { documents });
    return {
      received: body.documentsReceived,
      created: body.documentsCreated,
      replaced: body.documentsReplaced,
      unchanged: body.documentsUnchanged,
    };
  }

  async retrieve(question: string, options: QueryOptions = {}): Promise<Retrieval> {
    const body = await this.call<WireRetrieve>('POST', '/retrieve', {
      question,
      topK: options.topK,
    });
    return {
      question: body.question,
      evidence: body.chunks.map(toEvidence),
      latencyMs: body.retrievalMs,
    };
  }

  async ask(question: string, options: QueryOptions = {}): Promise<RagAnswer> {
    const body = await this.call<WireQuery>('POST', '/query', { question, topK: options.topK });
    return {
      question: body.question,
      answer: body.answer,
      abstained: body.abstained,
      model: body.model,
      evidence: body.evidence.map(toEvidence),
      usage: body.usage,
      timings: { retrievalMs: body.timings.retrievalMs, generationMs: body.timings.generationMs },
    };
  }

  /**
   * Sends one JSON request and decodes the response.
   *
   * @throws {@link RagClientError} when the service is unreachable or exceeds
   * the request timeout (no `status` then), answers
   * with a status outside `okStatuses`, or returns a body that is not JSON.
   */
  private async call<T>(
    method: 'GET' | 'POST',
    path: string,
    data?: object,
    okStatuses: readonly number[] = [200],
  ): Promise<T> {
    const call = `${method} ${path}`;
    let status: number;
    let body: string;
    let retryAfter: string | undefined;
    try {
      // `data` is JSON-encoded; keys whose value is undefined (an omitted topK) are dropped.
      const response = await this.request.fetch(path, { method, data });
      status = response.status();
      body = await response.text();
      retryAfter = response.headers()['retry-after'];
    } catch (cause) {
      const reason = cause instanceof Error ? cause.message : String(cause);
      throw new RagClientError(
        `${call} got no HTTP response (service down, unreachable, or slower than the request timeout): ${reason}`,
      );
    }
    if (!okStatuses.includes(status)) {
      throw responseError(call, status, body, retryAfter);
    }
    try {
      return JSON.parse(body) as T;
    } catch {
      throw new RagClientError(`${call} -> HTTP ${status}: response is not JSON`, { status });
    }
  }
}
