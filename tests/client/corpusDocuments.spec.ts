import { expect, test } from '@playwright/test';

import type { RagTestCase } from '../../src/models/ragTestCase';
import { passageDocumentId, toCorpusDocuments } from '../../src/utils/corpusDocuments';
import { loadRagTestCases } from '../../src/utils/datasetLoader';

function goldenCase(overrides: Partial<RagTestCase>): RagTestCase {
  return {
    id: 'RAG-9999',
    question: 'Where is Normandy?',
    context: ['Normandy is a region in France.'],
    groundTruth: ['France'],
    answerable: true,
    source: 'squad_v2',
    metadata: { squadId: 'x', title: 'Normans', split: 'validation' },
    ...overrides,
  };
}

test.describe('golden cases to corpus documents', () => {
  test('document ids match rag_service/scripts/ingest_squad.py', () => {
    const [first] = loadRagTestCases();
    // Expected value computed with the Python ingest script's passage_document_id.
    expect(passageDocumentId(first?.context[0] ?? '')).toBe('squad-41d4b11d52acb6f3');
  });

  test('one document per distinct passage, carrying only id, text, title and source', () => {
    const cases = loadRagTestCases();
    const documents = toCorpusDocuments(cases);

    expect(documents).toHaveLength(new Set(cases.flatMap((testCase) => testCase.context)).size);
    for (const document of documents) {
      expect(Object.keys(document).sort()).toEqual(['id', 'metadata', 'text']);
      expect(Object.keys(document.metadata ?? {}).sort()).toEqual(['source', 'title']);
      expect(document.id).toBe(passageDocumentId(document.text));
    }
  });

  test('questions, answers and answerability labels never reach the corpus', () => {
    const documents = toCorpusDocuments([
      goldenCase({ question: 'LEAK-QUESTION?', groundTruth: ['LEAK-ANSWER'] }),
      goldenCase({ id: 'RAG-9998', question: 'LEAK-UNANSWERABLE?', groundTruth: [], answerable: false }),
    ]);

    // Both cases share a passage, so it is sent once.
    expect(documents).toHaveLength(1);
    const payload = JSON.stringify(documents);
    expect(payload).not.toContain('LEAK');
    expect(payload).not.toContain('answerable');
    expect(payload).not.toContain('groundTruth');
  });
});
