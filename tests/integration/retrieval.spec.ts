import { RagClientError } from '../../src/clients/ragClient';
import { toCorpusDocuments } from '../../src/utils/corpusDocuments';
import { loadRagTestCases } from '../../src/utils/datasetLoader';
import { expect, test } from './fixtures';

const testCases = loadRagTestCases();

test.describe('RAG service: ingestion and retrieval (no LLM)', () => {
  test('re-ingesting golden passages is idempotent', async ({ ragClient }) => {
    const documents = toCorpusDocuments(testCases.slice(0, 3));

    const result = await ragClient.ingest(documents);

    expect(result).toEqual({ received: 3, created: 0, replaced: 0, unchanged: 3 });
  });

  test('retrieve returns topK well-formed evidence, closest first', async ({ ragClient }) => {
    const testCase = testCases.find((candidate) => candidate.answerable);
    expect(testCase, 'golden set has an answerable case').toBeDefined();

    const retrieval = await ragClient.retrieve(testCase?.question ?? '', { topK: 3 });

    expect(retrieval.question).toBe(testCase?.question);
    expect(retrieval.evidence).toHaveLength(3);
    expect(retrieval.latencyMs).toBeGreaterThanOrEqual(0);
    for (const evidence of retrieval.evidence) {
      expect(evidence.documentId).toMatch(/^squad-[0-9a-f]{16}$/);
      expect(evidence.chunkId.startsWith(`${evidence.documentId}#`)).toBe(true);
      expect(evidence.text.trim()).not.toBe('');
      expect(evidence.similarity).toBeGreaterThanOrEqual(-1);
      expect(evidence.similarity).toBeLessThanOrEqual(1);
    }
    const similarities = retrieval.evidence.map((evidence) => evidence.similarity);
    expect(similarities).toEqual([...similarities].sort((a, b) => b - a));
  });

  test('a blank question is rejected as a client error', async ({ ragClient }) => {
    const error = await ragClient.retrieve('   ').catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(RagClientError);
    expect(error).toMatchObject({ status: 422 });
  });
});
