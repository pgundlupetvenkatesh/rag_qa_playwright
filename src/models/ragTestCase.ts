/**
 * Normalized shape of a golden RAG evaluation case.
 *
 * This is the framework's stable contract: downstream milestones (retrieval
 * scoring, adversarial prompts, observability) read this shape, not SQuAD's.
 * Changing a field here is a breaking change for every consumer.
 */

/** Identifier of the upstream dataset a case was derived from. */
export const DATASET_SOURCE = 'squad_v2';

/** Split the cases were drawn from. Training data is deliberately excluded. */
export const DATASET_SPLIT = 'validation';

/** Provenance back to the original record, for debugging a failing case. */
export interface RagTestCaseMetadata {
  /** The original SQuAD record id. */
  readonly squadId: string;
  /** SQuAD article title, e.g. "Normans". Doubles as a topic label. */
  readonly title: string;
  /** Dataset split the record came from. */
  readonly split: string;
}

/**
 * A single golden evaluation case: a question, the passages it must be
 * answered from, and the accepted answers (empty when the correct behaviour
 * is to abstain).
 *
 * Invariant: `groundTruth` is empty if and only if `answerable` is false.
 * Enforced by the dataset generator and the loader, not by the type.
 */
export interface RagTestCase {
  /** Stable framework id, e.g. "RAG-0001". Unique across the dataset. */
  readonly id: string;
  /** Question text, verbatim from SQuAD. */
  readonly question: string;
  /**
   * Passages the answer must be grounded in, verbatim from SQuAD.
   *
   * Modelled as an array because retrieval evaluation deals in multiple
   * contexts; SQuAD supplies exactly one per case.
   */
  readonly context: readonly string[];
  /**
   * Accepted answer strings. Empty for unanswerable cases - a system under
   * test is expected to abstain rather than answer.
   */
  readonly groundTruth: readonly string[];
  /** False for SQuAD 2.0 unanswerable questions. */
  readonly answerable: boolean;
  /** Upstream dataset identifier, see {@link DATASET_SOURCE}. */
  readonly source: string;
  readonly metadata: RagTestCaseMetadata;
}