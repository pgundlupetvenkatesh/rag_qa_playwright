import { expect, test } from '@playwright/test';

import {
  DEFAULT_RAG_BASE_URL,
  DEFAULT_RAG_TIMEOUT_MS,
  loadRagServiceConfig,
} from '../../src/config/ragServiceConfig';

test.describe('RAG service config from environment variables', () => {
  test('defaults apply when the variables are unset or blank', () => {
    const expected = { baseUrl: DEFAULT_RAG_BASE_URL, timeoutMs: DEFAULT_RAG_TIMEOUT_MS };

    expect(loadRagServiceConfig({})).toEqual(expected);
    expect(loadRagServiceConfig({ RAG_BASE_URL: '  ', RAG_TIMEOUT_MS: '' })).toEqual(expected);
  });

  test('set values are read and trimmed', () => {
    expect(
      loadRagServiceConfig({ RAG_BASE_URL: ' http://rag.internal:9000 ', RAG_TIMEOUT_MS: ' 1500 ' }),
    ).toEqual({ baseUrl: 'http://rag.internal:9000', timeoutMs: 1500 });
  });

  test('an invalid base URL is rejected', () => {
    expect(() => loadRagServiceConfig({ RAG_BASE_URL: 'localhost 8000' })).toThrow(
      /RAG_BASE_URL is not a valid URL/,
    );
  });

  for (const timeout of ['0', '-5', '1.5', 'soon']) {
    test(`RAG_TIMEOUT_MS=${timeout} is rejected`, () => {
      expect(() => loadRagServiceConfig({ RAG_TIMEOUT_MS: timeout })).toThrow(
        /RAG_TIMEOUT_MS must be a positive integer/,
      );
    });
  }
});
