import { createServer } from 'node:http';
import type { AddressInfo } from 'node:net';

import { test as base } from '@playwright/test';

import type { RagClient } from '../../src/clients/ragClient';
import { RagServiceClient } from '../../src/clients/ragServiceClient';

/** Canned response for one `"METHOD /path"` route. */
export interface Reply {
  status?: number;
  /** Sent as JSON, or verbatim when a string. */
  body: unknown;
  headers?: Record<string, string>;
}

/** A request the mock received. */
export interface Recorded {
  method: string;
  path: string;
  body: unknown;
}

/**
 * Offline tests: a throwaway local HTTP server stands in for the RAG service,
 * so the adapter is exercised over real HTTP without the service or Groq.
 * Unrouted requests get a 404.
 */
export const test = base.extend<{
  serve: (routes: Record<string, Reply>) => Promise<{ client: RagClient; requests: Recorded[] }>;
}>({
  serve: async ({ playwright }, use) => {
    const cleanups: (() => Promise<void>)[] = [];
    await use(async (routes) => {
      const requests: Recorded[] = [];
      const server = createServer(async (req, res) => {
        let raw = '';
        for await (const chunk of req) {
          raw += String(chunk);
        }
        requests.push({
          method: req.method ?? '',
          path: req.url ?? '',
          body: raw ? JSON.parse(raw) : undefined,
        });
        const reply = routes[`${req.method} ${req.url}`];
        if (!reply) {
          res.writeHead(404).end();
          return;
        }
        res
          .writeHead(reply.status ?? 200, { 'content-type': 'application/json', ...reply.headers })
          .end(typeof reply.body === 'string' ? reply.body : JSON.stringify(reply.body));
      });
      await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
      const { port } = server.address() as AddressInfo;

      const request = await playwright.request.newContext({ baseURL: `http://127.0.0.1:${port}` });
      cleanups.push(
        () => request.dispose(),
        () => {
          server.closeAllConnections();
          return new Promise<void>((resolve) => server.close(() => resolve()));
        },
      );
      return { client: new RagServiceClient(request), requests };
    });
    for (const cleanup of cleanups) {
      await cleanup();
    }
  },
});

export { expect } from '@playwright/test';
