# rag_service

Reference RAG service (Python, FastAPI) that the Playwright suite in this repo
tests over HTTP. It runs independently of the TypeScript framework, with its own
dependencies, virtualenv and tests.

It has configuration, health and readiness probes, document ingestion into a
persistent ChromaDB corpus, retrieval, and answer generation through Groq that
is grounded in the retrieved evidence.

## Architecture and data flow

```
 Ingest (once; no LLM)
 SQuAD 2.0 validation ──title + context only──► scripts/ingest_squad.py
                                                    │ POST /corpus/documents
                                                    ▼
                         chunk (1000/200 chars) ─► embed (all-MiniLM-L6-v2, ONNX, local)
                                                    ▼
                                         ChromaDB, persistent in .chroma/

 Ask
 client ── POST /retrieve ─► embed question ─► nearest TOP_K chunks ─► evidence   (no LLM)
 client ── POST /query ────► same retrieval ─► Groq (GROQ_MODEL) ────► answer or abstention
                                                                       + the evidence it was shown
```

- **Two models.** The embedding model runs locally and costs nothing. Only
  `/query` (and the model check in `/ready`) calls Groq.
- **No evaluation labels.** The service never reads `data/golden/`. Questions,
  answers and answerability labels stay with the tests, so a test cannot pass
  because the corpus contains the answer.
- **HTTP is the only interface.** The Playwright framework talks to it through
  `RagServiceClient`. Neither side imports the other's code.

## Layout

```
app/__main__.py      python -m app: starts uvicorn on API_HOST:API_PORT
app/main.py          FastAPI routes, dependency wiring, error rendering
app/config.py        Settings: env vars, then rag_service/.env, validated at startup
app/schemas.py       request/response models (camelCase on the wire); closed metadata allowlist
app/readiness.py     /ready checks: key present, Groq model available, Chroma path usable
app/chunking.py      word-aligned overlapping chunks
app/embeddings.py    Embedder protocol; MiniLM through Chroma's ONNX build
app/corpus.py        ChromaDB corpus: idempotent ingest, search, reset
app/generation.py    Generator protocol; Groq prompt, abstention, error mapping
scripts/ingest_squad.py   loads SQuAD passages through the public API
tests/               pytest suite, offline (fake embedder, mocked Groq)
.env.example         placeholder-only template for the gitignored .env
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13 (pinned in `.python-version`).

```bash
cd rag_service
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env        # then set GROQ_API_KEY; .env is gitignored
```

### Groq API key

Create a key at <https://console.groq.com/keys> and put it in `rag_service/.env`:

```bash
GROQ_API_KEY=your_groq_api_key_here    # placeholder; paste your own key locally
```

- **Keep the key in `.env` or a real environment variable, nowhere else.**
  `rag_service/.gitignore` ignores `.env` and `.env.*` (except `.env.example`),
  and the repo's commit hook blocks staging any `.env` file. Never put a real key
  in `.env.example`, a README, a test or a commit message.
- **The service never exposes it.** The setting is a secret type that renders as
  `**********`. `/ready` reports only whether a key is set. Error messages are
  written by the service, not copied from Groq.
- **For CI**, pass the key as a masked secret environment variable
  (`GROQ_API_KEY`), not as a file in the workspace. The repo's workflow does this
  only on a manual run with `groq=true`; see
  [Continuous integration](../README.md#continuous-integration).
- **Without a key** the service still runs: `/health`, `/corpus`, ingestion and
  `/retrieve` work, `/ready` returns 503, and `/query` returns 503
  `missing_credentials`.
- **If a key leaks**, revoke it in the Groq console and create a new one.
  Removing it from Git history alone is not enough.

## Run

```bash
python -m app                        # binds API_HOST:API_PORT (default 127.0.0.1:8000)
uvicorn app.main:app --port 8000     # alternative; --host/--port come from the CLI
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
```

Stop it with Ctrl+C in its terminal. If it was left running in the background:
`lsof -ti tcp:8000 -sTCP:LISTEN | xargs kill`. Stop the service before deleting the Chroma
folder; deleting it under a running service makes later requests fail with 500. Use
`DELETE /corpus` to empty it while the service runs.

### Smoke check

The end-to-end check after setting `GROQ_API_KEY` in `.env`. Run it from
`rag_service/` with the venv active. Each `/query` makes one paid Groq call.

```bash
pytest                                   # offline unit tests
python -m app &                          # start the service (default model: openai/gpt-oss-120b)
curl -s localhost:8000/ready             # expect "status":"ready" with all three checks ok
curl -s localhost:8000/corpus            # expect 1204 documents; if 0, run scripts/ingest_squad.py

# Retrieval only, no LLM:
curl -s -X POST localhost:8000/retrieve -H 'content-type: application/json' \
  -d '{"question": "In what country is Normandy located?", "topK": 3}'

# Retrieval plus a grounded answer:
curl -s -X POST localhost:8000/query -H 'content-type: application/json' \
  -d '{"question": "In what country is Normandy located?"}'

# Should abstain ("abstained": true, "answer": null):
curl -s -X POST localhost:8000/query -H 'content-type: application/json' \
  -d '{"question": "What is the boiling point of mercury on Mars?"}'

kill %1                                  # stop the service
```

To try another model without editing `.env`, override it for one run:
`GROQ_MODEL=qwen/qwen3.8-27b python -m app`. Check `/ready` first; it confirms
the model is available to your key.

Settings are loaded from `rag_service/.env` whatever the working directory, and
real environment variables override it. Invalid values stop the service at
startup with a validation error.

## Endpoints

| Endpoint                 | Meaning                                                                                                       |
|--------------------------|---------------------------------------------------------------------------------------------------------------|
| `GET /health`            | Liveness. Always 200 while the process is up; contacts nothing.                                               |
| `GET /ready`             | Readiness. 200 `ready` or 503 `not_ready`, with one entry per check: `groqApiKey`, `groqModel`, `chromaPath`. |
| `GET /corpus`            | `{documents, chunks, embeddingModel}`.                                                                        |
| `POST /corpus/documents` | Add or replace documents by id. Returns per-request counts plus the corpus size afterwards.                   |
| `DELETE /corpus`         | Delete every document. Guarded, see [Reset](#reset).                                                          |
| `POST /retrieve`         | Top-k chunks for a question. No language model. See [Retrieval](#retrieval).                                  |
| `POST /query`            | Retrieval, then a grounded Groq answer or an abstention. See [Answer generation](#answer-generation).         |

`/ready` makes one live call to Groq (no retries, bounded by
`REQUEST_TIMEOUT_SECONDS`) to confirm that `GROQ_MODEL` is available to your key.
**Model availability varies by Groq account**, so the default model id is not
assumed to work. Check `/ready` after changing `GROQ_MODEL` or the key.

## Ingestion

`POST /corpus/documents`
```json
{"documents": [
  {"id": "doc-1", "text": "Full document text ...", "metadata": {"title": "Optional", "source": "optional"}}
]}
```

- **Validation (422):** an empty batch or more than 1,000 documents; blank or
  whitespace-only text; text over 100,000 characters; an `id` outside
  `[A-Za-z0-9][A-Za-z0-9._:-]{0,127}`; the same id twice in one batch; any
  unknown field. `metadata` accepts only `title` and `source`, so there is no
  field through which questions, answers or answerability labels could reach
  the corpus.
- **Chunking:** each document is split into word-aligned chunks of up to
  `CHUNK_SIZE` characters, overlapping by about `CHUNK_OVERLAP`.
- **What is stored per chunk:** the chunk text, the id `"<documentId>#<chunkIndex>"`,
  and the metadata `documentId`, `chunkIndex`, `chunkCount`, `charStart`/`charEnd`
  (offsets into the document), `fingerprint`, and `title`/`source` when given.
- **Repeating an ingest is safe.** The fingerprint hashes the text, metadata,
  chunk settings and embedding model. If it is identical, the document is
  reported as `documentsUnchanged` and not re-embedded. If anything changed,
  the new chunks are written first, then the old chunk ids the new version no
  longer uses are deleted. A document is never missing partway through a
  replacement, and no stale chunk survives. Changing `CHUNK_SIZE` or
  `CHUNK_OVERLAP` therefore rebuilds every document on its next ingest.
- **Response:** `documentsReceived`, `documentsCreated`, `documentsReplaced`,
  `documentsUnchanged`, `chunksWritten`, `chunksDeleted`, and `corpus`
  (`documents`, `chunks`, `embeddingModel`).

### Loading SQuAD

With the service running:

```bash
python scripts/ingest_squad.py            # --url http://127.0.0.1:8000 by default
python scripts/ingest_squad.py --reset    # empty the corpus first
```

The script reads only the `title` and `context` columns of the SQuAD 2.0
validation split and de-duplicates passages: 11,873 rows become 1,204
documents. Each document id is `squad-` plus the first 16 hex characters of the
passage's SHA-1. A test can therefore compute the expected id from a golden
case's `context` without the service ever reading the golden file. The script
uses only the public API.

### Reset

`DELETE /corpus` drops and recreates the collection. `ALLOW_RESET` controls it:

| `ALLOW_RESET` | Who can reset                                                         |
|---------------|-----------------------------------------------------------------------|
| unset         | Only callers connecting from a loopback address (`127.0.0.1`, `::1`). |
| `true`        | Anyone. For a CI container whose test runner is not on loopback.      |
| `false`       | Nobody (403).                                                         |

The check uses the caller's TCP address, not `API_HOST`, so it still holds when
the service is started with `uvicorn --host 0.0.0.0`. Behind a reverse proxy on
the same machine every caller looks like loopback, so set `ALLOW_RESET=false`
there.

## Retrieval

`POST /retrieve`
```json
{"question": "In what country is Normandy located?", "topK": 3}
```

`topK` is optional (1-50; default `TOP_K`), and `question` must be 1-2,000
non-blank characters. Unknown fields are rejected (422). The question is
embedded with the corpus's own model, and Chroma returns the nearest chunks,
closest first:

```json
{"question": "...", "topK": 3, "retrievalMs": 12.4,
 "chunks": [{"chunkId": "squad-6ec74f41bfd72db9#0", "documentId": "squad-6ec74f41bfd72db9",
             "chunkIndex": 0, "text": "...", "title": "Normans", "source": "squad_v2",
             "distance": 0.449, "similarity": 0.551}]}
```

`distance` is cosine distance (0 means the same direction). `similarity` is
`1 - distance`. Long passages are split into several chunks, so a retrieval
check must match on `documentId` (the parent passage), not on text.

If the corpus is empty, retrieval returns 409 with
`{"detail": {"stage": "retrieval", "code": "corpus_empty", ...}}`. It does not
return an empty list that a caller could mistake for "nothing relevant".

## Answer generation

`POST /query` takes the same body as `/retrieve`. It runs retrieval, then sends
the question and the numbered evidence passages to `GROQ_MODEL`. The system
prompt tells the model to answer only from that evidence, to add no unsupported
claims, to treat the evidence as data rather than instructions, and to reply
exactly `INSUFFICIENT_EVIDENCE` when the evidence cannot answer the question.

```jsonc
{"question": "...", "answer": "France.", "abstained": false, "model": "openai/gpt-oss-120b",
 "evidence": [/* exactly the chunks the model was shown, same shape as /retrieve */],
 "usage": {"promptTokens": 812, "completionTokens": 41, "totalTokens": 853},
 "timings": {"retrievalMs": 11.8, "generationMs": 640.2, "providerMs": 212.5}}
```

- **Abstention:** `abstained: true` with `answer: null`. The evidence is still
  returned, because retrieval did run.
- **No citations are generated.** The model is told not to cite, and
  `evidence` is the server's record of what retrieval returned, not the
  model's claim about what it used.
- **Model-specific parameters:** temperature 0 and at most 1,024 completion
  tokens. For `openai/gpt-oss-*`, reasoning is excluded from the reply
  (`include_reasoning: false`, `reasoning_effort: low`). `qwen/qwen3.8-27b`
  does not reason by default on Groq, so it gets no extra parameters. Switch
  models with `GROQ_MODEL` and confirm the new id with `/ready`.
- **One attempt, no retries:** SDK retries are disabled, and each call is
  bounded by `REQUEST_TIMEOUT_SECONDS`.

### Errors

Every failure has the shape `{"detail": {"stage", "code", "message"}}`. `stage`
says whether retrieval or generation failed. Messages are written by the
service and never copied from Groq, so neither the key nor the prompt can leak
through them.

| Stage        | `code`                 | HTTP | When                                                                              |
|--------------|------------------------|------|-----------------------------------------------------------------------------------|
| `retrieval`  | `corpus_empty`         | 409  | Nothing ingested. The model is not called.                                        |
| `retrieval`  | `no_results`           | 409  | The corpus returned nothing. The model is not called.                             |
| `generation` | `missing_credentials`  | 503  | `GROQ_API_KEY` unset. Checked before retrieval runs.                              |
| `generation` | `provider_auth_failed` | 503  | Groq rejected the key (401/403).                                                  |
| `generation` | `model_unavailable`    | 503  | `GROQ_MODEL` is not available to the key (404).                                   |
| `generation` | `rate_limited`         | 429  | Groq's rate limit. Groq's `Retry-After` is passed through.                        |
| `generation` | `provider_timeout`     | 504  | No answer within `REQUEST_TIMEOUT_SECONDS`.                                       |
| `generation` | `provider_unreachable` | 502  | Network failure reaching Groq.                                                    |
| `generation` | `provider_error`       | 502  | Any other Groq HTTP error.                                                        |
| `generation` | `malformed_response`   | 502  | No choices, empty content, or output cut off (`finish_reason` other than `stop`). |

The Groq integration lives entirely in `app/generation.py`, behind a small
`Generator` protocol. Another provider can replace it without changing
retrieval or the endpoints.

## Performance and dependencies

Measured on this machine (Intel Core i7-8559U, CPU only):

- **First full SQuAD ingest:** 1,204 documents became 1,473 chunks in 113 s,
  about 13 chunks/s. Embedding dominates the time. The resulting store is 15 MB
  on disk.
- **Repeat ingest of the same data:** 0.3 s. Every document is reported
  unchanged and nothing is re-embedded.
- **Memory:** the service uses about 660 MB once the embedding model is loaded.
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`, run through ONNX Runtime
  (bundled with ChromaDB) rather than the `sentence-transformers` package. It is
  the same model with the same 384-dimension vectors, but needs no torch, which
  has no current Intel-Mac builds. It downloads once (~80 MB) to
  `~/.cache/chroma` and loads on the first corpus request, not at startup. It
  embeds at most 256 tokens per chunk.
- **Requirements:** `onnxruntime<1.24` is pinned only on Intel Macs, where newer
  versions have no builds. `datasets` (which brings in pyarrow) is only needed
  by `scripts/ingest_squad.py`.
- **Writes are serialized** with a process-wide lock, so concurrent ingests
  cannot interleave a replacement or race a reset. Run one worker process:
  multiple uvicorn workers would each hold their own lock over the same store.
- **ChromaDB's anonymous telemetry is disabled** in code.

## Configuration

| Variable                  | Default               | Notes                                                                                          |
|---------------------------|-----------------------|------------------------------------------------------------------------------------------------|
| `GROQ_API_KEY`            | unset                 | Required for `/ready` and `/query`. Never logged or returned.                                  |
| `GROQ_MODEL`              | `openai/gpt-oss-120b` | e.g. `qwen/qwen3.8-27b`. Must be verified for your account through `/ready`.                   |
| `EMBEDDING_MODEL`         | `all-MiniLM-L6-v2`    | The only supported value. Chroma's ONNX model runs without torch.                              |
| `CHROMA_PATH`             | `.chroma`             | Relative paths resolve against `rag_service/`. Gitignored.                                     |
| `CHUNK_SIZE`              | `1000`                | Characters. The embedding model reads at most 256 tokens; 1000 chars fits for 99.5% of chunks. |
| `CHUNK_OVERLAP`           | `200`                 | Characters; must be smaller than `CHUNK_SIZE`.                                                 |
| `TOP_K`                   | `4`                   | Passages retrieved per question.                                                               |
| `API_HOST` / `API_PORT`   | `127.0.0.1` / `8000`  | Used by `python -m app`.                                                                       |
| `REQUEST_TIMEOUT_SECONDS` | `30`                  | Timeout for outbound Groq calls.                                                               |
| `ALLOW_RESET`             | unset                 | Who may call `DELETE /corpus`; see [Reset](#reset).                                            |

## Test

| Suite | Command | Needs | Groq calls |
|---|---|---|---|
| Service unit tests | `cd rag_service && pytest` | Nothing (offline) | None |
| Framework, offline | `npm test` (repo root) | Nothing; client tests use a local mock server | None |
| Framework, live service | `npm run test:service` | Service running at `RAG_BASE_URL` | None |
| Framework, live Groq | `npm run test:groq` | Service running **with a working `GROQ_API_KEY`** | About 2 answers per run, plus `/ready`'s model check |

```bash
# Offline: safe to run anywhere, costs nothing.
cd rag_service && source .venv/bin/activate && pytest && cd ..
npm test

# Live: start the service in another terminal first.
cd rag_service && source .venv/bin/activate && python -m app
npm run test:service             # retrieval and ingestion, no LLM
npm run test:groq                # tests tagged @groq; uses your Groq quota
npm run test:integration         # both
```

Live tests are marked twice: by Playwright project (`service`, `groq`) and by
the `@groq` tag on every test that reaches Groq. The live setup re-ingests the
60 golden passages, which is idempotent and never resets the corpus. Point the
tests elsewhere with `RAG_BASE_URL` (and `RAG_TIMEOUT_MS`, default 60000).

The pytest suite reads neither your shell's variables nor `.env`, and they make no network
calls. Groq is replaced by a mocked client (generation) or a mock HTTP transport
(readiness), so no key is needed and no API charges are incurred. The embedding model is replaced by a
deterministic fake (so the ~80 MB model is never needed), and settings and the
corpus are injected through FastAPI dependency overrides. Each test gets its own
temporary Chroma directory.
## Known limitations and costs

**Costs**

- Only Groq calls cost anything: one chat completion per `/query`, plus one
  model listing per `/ready` (no tokens). Ingestion, embedding and `/retrieve`
  run locally and are free.
- A `/query` with the default 4 chunks uses roughly 800-1,000 prompt tokens.
  On Groq's free tier this counts against rate limits; on a paid plan it is
  billed per token. Check current limits and prices in the Groq console.
- `pytest` and `npm test` never call Groq. Only `npm run test:groq` (and
  `test:integration`) does.

**Limitations**

- **No authentication.** Any caller that can reach the port can query or
  ingest. The default bind is `127.0.0.1`; don't expose it publicly as is.
  `DELETE /corpus` alone is guarded (see [Reset](#reset)).
- **One worker process.** Writes are serialised by an in-process lock.
- **Retrieval always returns `topK` chunks**, even when none is relevant; there
  is no similarity threshold. Deciding that the evidence is insufficient is the
  model's job, so abstention quality depends on the model following the
  `INSUFFICIENT_EVIDENCE` instruction.
- **Embedding model limits:** mostly English, at most 256 tokens per chunk, and
  only `all-MiniLM-L6-v2` is supported. Changing it means re-ingesting the
  corpus.
- **No retries.** A Groq rate limit or timeout is returned to the caller as
  429 or 504 rather than retried.
- **Model availability depends on the Groq account.** Check `/ready` after
  changing `GROQ_MODEL` or the key.
- **No quality metrics yet.** The tests check the API contract, not whether
  retrieval found the right passage or the answer is correct.
- **No Docker image yet.** Run it from a local virtualenv.
