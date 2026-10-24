import { toCorpusDocuments } from '../../src/utils/corpusDocuments';
import { loadRagTestCases } from '../../src/utils/datasetLoader';
import { expect, test as setup } from './fixtures';

// Ingestion is idempotent by document id, so this is safe against a corpus
// that already holds the full SQuAD set: those passages come back unchanged.
// Nothing is ever deleted.
setup('ingest the golden passages', async ({ ragClient }) => {
  // The first ingest into an empty corpus loads the embedding model.
  setup.setTimeout(180_000);
  const documents = toCorpusDocuments(loadRagTestCases());

  const result = await ragClient.ingest(documents);

  expect(result.received).toBe(documents.length);
  expect(result.created + result.replaced + result.unchanged).toBe(documents.length);
  setup.info().annotations.push({
    type: 'ingest',
    description: `created=${result.created} replaced=${result.replaced} unchanged=${result.unchanged}`,
  });
});
