import { RagClientError } from '../../src/clients/ragClient';
import { RagServiceClient } from '../../src/clients/ragServiceClient';
import { expect, test } from './mockRagService';

const chunk = {
  chunkId: 'squad-aaaa#0',
  documentId: 'squad-aaaa',
  chunkIndex: 0,
  text: 'Normandy is a region in France.',
  title: 'Normans',
  source: 'squad_v2',
  distance: 0.25,
  similarity: 0.75,
};

const untitledChunk = { ...chunk, chunkId: 'squad-bbbb#1', documentId: 'squad-bbbb', title: null, source: null };

test.describe('RagServiceClient against a mocked service', () => {
  test('ready() resolves for both ready (200) and not ready (503)', async ({ serve }) => {
    const checks = [{ name: 'groqApiKey', ok: false, detail: 'GROQ_API_KEY is not set' }];
    const up = await serve({ 'GET /ready': { body: { status: 'ready', checks: [] } } });
    const down = await serve({
      'GET /ready': { status: 503, body: { status: 'not_ready', checks } },
    });

    expect(await up.client.ready()).toEqual({ ready: true, checks: [] });
    expect(await down.client.ready()).toEqual({ ready: false, checks });
  });

  test('ingest() posts documents and maps the counts', async ({ serve }) => {
    const { client, requests } = await serve({
      'POST /corpus/documents': {
        body: {
          documentsReceived: 2,
          documentsCreated: 1,
          documentsReplaced: 0,
          documentsUnchanged: 1,
          chunksWritten: 1,
          chunksDeleted: 0,
          corpus: { documents: 5, chunks: 6, embeddingModel: 'all-MiniLM-L6-v2' },
        },
      },
    });
    const documents = [
      { id: 'a', text: 'A', metadata: { title: 'T', source: 'squad_v2' } },
      { id: 'b', text: 'B' },
    ];

    const result = await client.ingest(documents);

    expect(result).toEqual({ received: 2, created: 1, replaced: 0, unchanged: 1 });
    expect(requests).toEqual([{ method: 'POST', path: '/corpus/documents', body: { documents } }]);
  });

  test('retrieve() sends question and topK and maps chunks to evidence', async ({ serve }) => {
    const { client, requests } = await serve({
      'POST /retrieve': {
        body: { question: 'Where?', topK: 2, chunks: [chunk, untitledChunk], retrievalMs: 12.5 },
      },
    });

    const retrieval = await client.retrieve('Where?', { topK: 2 });

    expect(requests[0]?.body).toEqual({ question: 'Where?', topK: 2 });
    expect(retrieval).toEqual({
      question: 'Where?',
      latencyMs: 12.5,
      evidence: [
        {
          chunkId: 'squad-aaaa#0',
          documentId: 'squad-aaaa',
          text: chunk.text,
          similarity: 0.75,
          title: 'Normans',
          source: 'squad_v2',
        },
        { chunkId: 'squad-bbbb#1', documentId: 'squad-bbbb', text: chunk.text, similarity: 0.75 },
      ],
    });
  });

  test('ask() omits topK when not given and maps answer, usage and timings', async ({ serve }) => {
    const { client, requests } = await serve({
      'POST /query': {
        body: {
          question: 'Where?',
          answer: 'France.',
          abstained: false,
          model: 'openai/gpt-oss-120b',
          evidence: [chunk],
          usage: { promptTokens: 100, completionTokens: 5, totalTokens: 105 },
          timings: { retrievalMs: 10, generationMs: 300, providerMs: 250 },
        },
      },
    });

    const answer = await client.ask('Where?');

    expect(requests[0]?.body).toEqual({ question: 'Where?' });
    expect(answer).toMatchObject({
      answer: 'France.',
      abstained: false,
      model: 'openai/gpt-oss-120b',
      usage: { promptTokens: 100, completionTokens: 5, totalTokens: 105 },
      timings: { retrievalMs: 10, generationMs: 300 },
    });
    expect(answer.evidence.map((item) => item.documentId)).toEqual(['squad-aaaa']);
  });

  test('a stage error keeps its stage, code, status and Retry-After', async ({ serve }) => {
    const { client } = await serve({
      'POST /query': {
        status: 429,
        headers: { 'retry-after': '7' },
        body: { detail: { stage: 'generation', code: 'rate_limited', message: 'slow down' } },
      },
    });

    const error = await client.ask('Where?').catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(RagClientError);
    expect(error).toMatchObject({
      status: 429,
      stage: 'generation',
      code: 'rate_limited',
      retryAfterSeconds: 7,
    });
  });

  test('a provider timeout is a generation-stage error without Retry-After', async ({ serve }) => {
    const { client } = await serve({
      'POST /query': {
        status: 504,
        body: {
          detail: { stage: 'generation', code: 'provider_timeout', message: 'Groq did not answer in time' },
        },
      },
    });

    await expect(client.ask('Where?')).rejects.toMatchObject({
      status: 504,
      stage: 'generation',
      code: 'provider_timeout',
      retryAfterSeconds: undefined,
    });
  });

  test('a service slower than the client timeout is a client error without a status', async ({
    serve,
  }) => {
    const { client } = await serve(
      { 'POST /query': { delayMs: 2_000, body: {} } },
      { timeoutMs: 200 },
    );

    const error = await client.ask('Where?').catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(RagClientError);
    expect(error).toMatchObject({ status: undefined, stage: undefined });
    expect(String(error)).toContain('no HTTP response');
    expect(String(error)).toMatch(/timeout|timed out/i);
  });

  test('an empty corpus is reported as a retrieval-stage error', async ({ serve }) => {
    const { client } = await serve({
      'POST /retrieve': {
        status: 409,
        body: { detail: { stage: 'retrieval', code: 'corpus_empty', message: 'ingest first' } },
      },
    });

    await expect(client.retrieve('Where?')).rejects.toMatchObject({
      status: 409,
      stage: 'retrieval',
      code: 'corpus_empty',
    });
  });

  test('a validation error without a stage keeps the status and the detail text', async ({
    serve,
  }) => {
    const detail = [{ loc: ['body', 'question'], msg: 'String should have at least 1 character' }];
    const { client } = await serve({ 'POST /retrieve': { status: 422, body: { detail } } });

    const error = await client.retrieve('').catch((caught: unknown) => caught);

    expect(error).toMatchObject({ status: 422, stage: undefined, code: undefined });
    expect(String(error)).toContain('at least 1 character');
  });

  test('a non-JSON success body is a client error, not a crash', async ({ serve }) => {
    const { client } = await serve({ 'GET /ready': { body: '<html>proxy</html>' } });

    await expect(client.ready()).rejects.toThrow(/not JSON/);
  });

  test('an unreachable service is a client error without a status', async ({ playwright }) => {
    // Port 9 (discard) has no listener on a dev machine or CI runner.
    const request = await playwright.request.newContext({ baseURL: 'http://127.0.0.1:9' });
    try {
      const error = await new RagServiceClient(request).ready().catch((caught: unknown) => caught);

      expect(error).toBeInstanceOf(RagClientError);
      expect(error).toMatchObject({ status: undefined });
      expect(String(error)).toContain('no HTTP response');
    } finally {
      await request.dispose();
    }
  });
});
