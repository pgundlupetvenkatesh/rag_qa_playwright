# rag_service

Reference RAG service (Python, FastAPI) that the Playwright suite in this repo
tests over HTTP. It runs independently of the TypeScript framework, with its own
dependencies, virtualenv and tests.

So far it has configuration, health and readiness probes, and document
ingestion into a persistent ChromaDB corpus. Retrieval and answer generation
are not built yet.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13 (pinned in `.python-version`).

```bash
cd rag_service
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env        # then set GROQ_API_KEY; .env is gitignored
```

## Run

```bash
python -m app                        # binds API_HOST:API_PORT (default 127.0.0.1:8000)
uvicorn app.main:app --port 8000     # alternative; --host/--port come from the CLI
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
```

Settings are loaded from `rag_service/.env` whatever the working directory, and
real environment variables override it. Invalid values stop the service at
startup with a validation error.

## Endpoints

| Endpoint      | Meaning                                                                                  |
|---------------|------------------------------------------------------------------------------------------|
| `GET /health` | Liveness. Always 200 while the process is up; contacts nothing.                          |
| `GET /ready`  | Readiness. 200 `ready` or 503 `not_ready`, with one entry per check: `groqApiKey`, `groqModel`, `chromaPath`. |
| `GET /corpus` | `{documents, chunks, embeddingModel}`.                                                   |
| `POST /corpus/documents` | Add or replace documents by id. Returns per-request counts plus the corpus size afterwards. |
| `DELETE /corpus` | Delete every document. Guarded, see [Reset](#reset).                                  |

`/ready` makes one live call to Groq (no retries, bounded by
`REQUEST_TIMEOUT_SECONDS`) to confirm that `GROQ_MODEL` is available to your key.
**Model availability varies by Groq account**, so the default model id is not
assumed to work. Check `/ready` after changing `GROQ_MODEL` or the key.

## Ingestion

```json
POST /corpus/documents
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

| `ALLOW_RESET` | Who can reset                                                     |
|---------------|-------------------------------------------------------------------|
| unset         | Only callers connecting from a loopback address (`127.0.0.1`, `::1`). |
| `true`        | Anyone. For a CI container whose test runner is not on loopback.  |
| `false`       | Nobody (403).                                                     |

The check uses the caller's TCP address, not `API_HOST`, so it still holds when
the service is started with `uvicorn --host 0.0.0.0`. Behind a reverse proxy on
the same machine every caller looks like loopback, so set `ALLOW_RESET=false`
there.

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
- **Writes are serialised** with a process-wide lock, so concurrent ingests
  cannot interleave a replacement or race a reset. Run one worker process:
  multiple uvicorn workers would each hold their own lock over the same store.
- **ChromaDB's anonymous telemetry is disabled** in code.

## Configuration

| Variable                  | Default              | Notes                                                                 |
|---------------------------|----------------------|-----------------------------------------------------------------------|
| `GROQ_API_KEY`            | unset                | Required for `/ready` to pass. Never logged or returned.              |
| `GROQ_MODEL`              | `openai/gpt-oss-120b` | Must be verified for your account through `/ready`.                   |
| `EMBEDDING_MODEL`         | `all-MiniLM-L6-v2`   | The only supported value. Chroma's ONNX model runs without torch.     |
| `CHROMA_PATH`             | `.chroma`            | Relative paths resolve against `rag_service/`. Gitignored.            |
| `CHUNK_SIZE`              | `1000`               | Characters. The embedding model reads at most 256 tokens; 1000 chars fits for 99.5% of chunks. |
| `CHUNK_OVERLAP`           | `200`                | Characters; must be smaller than `CHUNK_SIZE`.                        |
| `TOP_K`                   | `4`                  | Passages retrieved per question.                                      |
| `API_HOST` / `API_PORT`   | `127.0.0.1` / `8000` | Used by `python -m app`.                                              |
| `REQUEST_TIMEOUT_SECONDS` | `30`                 | Timeout for outbound Groq calls.                                      |
| `ALLOW_RESET`             | unset                | Who may call `DELETE /corpus`; see [Reset](#reset).                   |

## Test

```bash
pytest
```

Tests read neither your shell's variables nor `.env`, and they make no network
calls. Groq is replaced by a mock HTTP transport, the embedding model by a
deterministic fake (so the ~80 MB model is never needed), and settings and the
corpus are injected through FastAPI dependency overrides. Each test gets its own
temporary Chroma directory.