import { test as base } from '@playwright/test';

import type { RagClient } from '../../src/clients/ragClient';
import { RagServiceClient } from '../../src/clients/ragServiceClient';
import { loadRagServiceConfig } from '../../src/config/ragServiceConfig';

/**
 * Integration tests: a real RAG service at `RAG_BASE_URL`. Tests tagged
 * `@groq` additionally need the service to hold a working Groq key.
 */
export const test = base.extend<object, { ragClient: RagClient }>({
  ragClient: [
    async ({ playwright }, use) => {
      const config = loadRagServiceConfig();
      const request = await playwright.request.newContext({
        baseURL: config.baseUrl,
        timeout: config.timeoutMs,
      });
      await use(new RagServiceClient(request));
      await request.dispose();
    },
    { scope: 'worker' },
  ],
});

export { expect } from '@playwright/test';
