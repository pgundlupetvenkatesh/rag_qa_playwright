import { expect, test } from '@playwright/test';

import { DATASET_SOURCE, DATASET_SPLIT, type RagTestCase } from '../../src/models/ragTestCase';
import { loadRagTestCases } from '../../src/utils/datasetLoader';

const EXPECTED_TOTAL = 60;
const EXPECTED_ANSWERABLE = 50;
const EXPECTED_UNANSWERABLE = 10;

// Loading is itself the first assertion: loadRagTestCases throws on any
// structural fault, so a corrupt golden file fails the whole suite at import.
const testCases: RagTestCase[] = loadRagTestCases();
const answerable = testCases.filter((testCase) => testCase.answerable);
const unanswerable = testCases.filter((testCase) => !testCase.answerable);

test.describe('golden RAG dataset', () => {
  test(`contains exactly ${EXPECTED_TOTAL} test cases`, () => {
    expect(testCases).toHaveLength(EXPECTED_TOTAL);
  });

  test(`contains exactly ${EXPECTED_ANSWERABLE} answerable cases`, () => {
    expect(answerable).toHaveLength(EXPECTED_ANSWERABLE);
  });

  test(`contains exactly ${EXPECTED_UNANSWERABLE} unanswerable cases`, () => {
    expect(unanswerable).toHaveLength(EXPECTED_UNANSWERABLE);
  });

  test('every case has a unique id', () => {
    const ids = testCases.map((testCase) => testCase.id);
    expect(ids.every((id) => id.length > 0)).toBe(true);
    expect(new Set(ids).size).toBe(ids.length);
  });

  test('every case has a question and at least one context passage', () => {
    for (const testCase of testCases) {
      expect(testCase.question.trim(), `${testCase.id} question`).not.toBe('');
      expect(testCase.context.length, `${testCase.id} context`).toBeGreaterThan(0);
      expect(
        testCase.context.every((passage) => passage.trim().length > 0),
        `${testCase.id} context passages`,
      ).toBe(true);
    }
  });

  test('answerable cases have at least one ground-truth answer', () => {
    for (const testCase of answerable) {
      expect(testCase.groundTruth.length, `${testCase.id} groundTruth`).toBeGreaterThan(0);
      expect(
        testCase.groundTruth.every((answer) => answer.trim().length > 0),
        `${testCase.id} groundTruth entries`,
      ).toBe(true);
    }
  });

  test('unanswerable cases have an empty groundTruth array', () => {
    for (const testCase of unanswerable) {
      expect(testCase.groundTruth, `${testCase.id} groundTruth`).toEqual([]);
    }
  });

  test('every case is attributed to the SQuAD 2.0 validation split', () => {
    for (const testCase of testCases) {
      expect(testCase.source, `${testCase.id} source`).toBe(DATASET_SOURCE);
      expect(testCase.metadata.split, `${testCase.id} split`).toBe(DATASET_SPLIT);
      expect(testCase.metadata.squadId.trim(), `${testCase.id} squadId`).not.toBe('');
      expect(testCase.metadata.title.trim(), `${testCase.id} title`).not.toBe('');
    }
  });
});