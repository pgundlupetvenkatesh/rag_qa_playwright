import { createHash } from 'node:crypto';

import type { CorpusDocument } from '../clients/ragClient';
import type { RagTestCase } from '../models/ragTestCase';

/**
 * Stable document id for a SQuAD passage: `"squad-"` plus the first 16 hex
 * characters of the passage's SHA-1.
 *
 * Must match `passage_document_id` in `rag_service/scripts/ingest_squad.py`,
 * so a golden case's expected document id can be derived from its `context`
 * whichever side ingested the passage.
 */
export function passageDocumentId(passage: string): string {
  return `squad-${createHash('sha1').update(passage, 'utf8').digest('hex').slice(0, 16)}`;
}

/**
 * Builds the corpus documents for golden cases: each distinct passage once,
 * keyed by {@link passageDocumentId}, in first-seen order.
 *
 * Reads only `context`, `source` and `metadata.title`. The question, ground
 * truth and answerability label never leave the test side: a corpus that
 * contained them would turn every retrieval and grounding test into a lookup.
 */
export function toCorpusDocuments(testCases: readonly RagTestCase[]): CorpusDocument[] {
  const byId = new Map<string, CorpusDocument>();
  for (const {
    context,
    source,
    metadata: { title },
  } of testCases) {
    for (const text of context) {
      const id = passageDocumentId(text);
      if (!byId.has(id)) {
        byId.set(id, { id, text, metadata: { title, source } });
      }
    }
  }
  return [...byId.values()];
}
