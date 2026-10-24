import { loadRagTestCases } from '../../src/utils/datasetLoader';
import { expect, test } from './fixtures';

const testCases = loadRagTestCases();
// One of each kind keeps live Groq usage to two generations per run.
const sample = [
  testCases.find((testCase) => testCase.answerable),
  testCases.find((testCase) => !testCase.answerable),
];

test.describe('RAG service: live Groq answers', () => {
  // In order, one at a time: the free Groq tier rate-limits bursts.
  test.describe.configure({ mode: 'default' });

  test('the service is ready', { tag: '@groq' }, async ({ ragClient }) => {
    const readiness = await ragClient.ready();

    expect(readiness.ready, JSON.stringify(readiness.checks, null, 2)).toBe(true);
  });

  for (const testCase of sample) {
    test(
      `${testCase?.id} returns an answer or an abstention with its evidence`,
      { tag: '@groq' },
      async ({ ragClient }) => {
        expect(testCase, 'golden set has this kind of case').toBeDefined();

        const answer = await ragClient.ask(testCase?.question ?? '');

        // Contract only; whether the answer is right is a later milestone's metric.
        expect(answer.question).toBe(testCase?.question);
        expect(answer.model).not.toBe('');
        expect(answer.evidence.length).toBeGreaterThan(0);
        if (answer.abstained) {
          expect(answer.answer).toBeNull();
        } else {
          expect(answer.answer?.trim()).toBeTruthy();
        }
        expect(answer.timings.generationMs).toBeGreaterThan(0);
        test.info().annotations.push({
          type: 'answer',
          description: answer.abstained ? '(abstained)' : String(answer.answer),
        });
      },
    );
  }
});
