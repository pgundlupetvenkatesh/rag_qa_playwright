# rag_service

Reference RAG service (Python, FastAPI) that the Playwright suite in this repo
tests over HTTP. It runs independently of the TypeScript framework, with its own
dependencies, virtualenv and tests.

So far it has configuration, `GET /health` and `GET /ready`. Ingestion,
retrieval and generation are not built yet.

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

`/ready` makes one live call to Groq (no retries, bounded by
`REQUEST_TIMEOUT_SECONDS`) to confirm that `GROQ_MODEL` is available to your key.
**Model availability varies by Groq account**, so the default model id is not
assumed to work. Check `/ready` after changing `GROQ_MODEL` or the key.

## Configuration

| Variable                  | Default              | Notes                                                                 |
|---------------------------|----------------------|-----------------------------------------------------------------------|
| `GROQ_API_KEY`            | unset                | Required for `/ready` to pass. Never logged or returned.              |
| `GROQ_MODEL`              | `openai/gpt-oss-20b` | Must be verified for your account through `/ready`.                   |
| `EMBEDDING_MODEL`         | `all-MiniLM-L6-v2`   | The only supported value. Chroma's ONNX model runs without torch.     |
| `CHROMA_PATH`             | `.chroma`            | Relative paths resolve against `rag_service/`. Gitignored.            |
| `CHUNK_SIZE`              | `1000`               | Characters. The embedding model reads at most 256 tokens; 1000 chars fits for 99.5% of chunks. |
| `CHUNK_OVERLAP`           | `200`                | Characters; must be smaller than `CHUNK_SIZE`.                        |
| `TOP_K`                   | `4`                  | Passages retrieved per question.                                      |
| `API_HOST` / `API_PORT`   | `127.0.0.1` / `8000` | Used by `python -m app`.                                              |
| `REQUEST_TIMEOUT_SECONDS` | `30`                 | Timeout for outbound Groq calls.                                      |

## Test

```bash
pytest
```

Tests read neither your shell's variables nor `.env`, and they make no network
calls. Groq is replaced by a mock HTTP transport, and settings are injected
through FastAPI dependency overrides.