/**
 * System-agnostic contract for talking to a RAG application under test.
 *
 * Tests depend on {@link RagClient}, never on a concrete service, so pointing
 * the framework at another RAG app means writing one adapter, not editing
 * tests. Evaluation labels (expected answers, answerability) are deliberately
 * absent from every type here: they stay on the test side.
 */

/** Pipeline stage a failure is attributed to. */
export type RagStage = 'retrieval' | 'generation';

/** A document to add to the corpus: a stable id and passage text, nothing else. */
export interface CorpusDocument {
  /** Caller-chosen id, stable across runs so re-ingestion is idempotent. */
  readonly id: string;
  /** Passage text. */
  readonly text: string;
  /** Descriptive metadata only; there is deliberately no free-form field. */
  readonly metadata?: {
    readonly title?: string;
    readonly source?: string;
  };
}

/** Outcome of one readiness check. */
export interface ReadinessCheck {
  readonly name: string;
  readonly ok: boolean;
  readonly detail: string;
}

/** Whether the service could do real work right now, and why not if it can't. */
export interface Readiness {
  readonly ready: boolean;
  readonly checks: readonly ReadinessCheck[];
}

/** What one ingestion call did. Each received document is counted once. */
export interface IngestResult {
  readonly received: number;
  readonly created: number;
  readonly replaced: number;
  readonly unchanged: number;
}

/** One retrieved piece of evidence. */
export interface Evidence {
  /** Id of the retrieved chunk. */
  readonly chunkId: string;
  /** Id of the parent document; match expectations on this, not on text. */
  readonly documentId: string;
  readonly text: string;
  /** Cosine similarity to the question; higher is closer. */
  readonly similarity: number;
  readonly title?: string;
  readonly source?: string;
}

/** Result of retrieval alone, with no language model involved. */
export interface Retrieval {
  readonly question: string;
  /** Closest first. */
  readonly evidence: readonly Evidence[];
  readonly latencyMs: number;
}

/** Token counts as reported by the provider; `null` when not reported. */
export interface TokenUsage {
  readonly promptTokens: number | null;
  readonly completionTokens: number | null;
  readonly totalTokens: number | null;
}

/** A generated answer, or an abstention, with the evidence it was given. */
export interface RagAnswer {
  readonly question: string;
  /** `null` exactly when {@link RagAnswer.abstained} is true. */
  readonly answer: string | null;
  readonly abstained: boolean;
  /** Generation model id that produced the answer. */
  readonly model: string;
  /** What retrieval returned and the model was shown. */
  readonly evidence: readonly Evidence[];
  readonly usage: TokenUsage;
  readonly timings: {
    readonly retrievalMs: number;
    readonly generationMs: number;
  };
}

/** Options for {@link RagClient.retrieve} and {@link RagClient.ask}. */
export interface QueryOptions {
  /** Evidence chunks to retrieve; the service default when omitted. */
  readonly topK?: number;
}

/** Operations every RAG adapter provides. */
export interface RagClient {
  /** Reports readiness. A not-ready service resolves; only transport or protocol failures reject. */
  ready(): Promise<Readiness>;
  /** Adds documents, or replaces existing ones with the same id. */
  ingest(documents: readonly CorpusDocument[]): Promise<IngestResult>;
  /** Retrieves evidence for a question without generating an answer. */
  retrieve(question: string, options?: QueryOptions): Promise<Retrieval>;
  /** Retrieves evidence, then answers from it or abstains. */
  ask(question: string, options?: QueryOptions): Promise<RagAnswer>;
}

/** Raised by a {@link RagClient} when a call fails or the service rejects it. */
export class RagClientError extends Error {
  /** HTTP status, absent when the service could not be reached at all. */
  readonly status: number | undefined;
  /** Which stage failed, when the service says so. */
  readonly stage: RagStage | undefined;
  /** Machine-readable error code, when the service supplies one. */
  readonly code: string | undefined;
  /** Seconds to wait before retrying, from a `Retry-After` header. */
  readonly retryAfterSeconds: number | undefined;

  constructor(
    message: string,
    details: {
      status?: number;
      stage?: RagStage;
      code?: string;
      retryAfterSeconds?: number;
    } = {},
  ) {
    super(message);
    this.name = 'RagClientError';
    this.status = details.status;
    this.stage = details.stage;
    this.code = details.code;
    this.retryAfterSeconds = details.retryAfterSeconds;
  }
}
