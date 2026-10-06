# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A reusable AI/RAG quality-engineering framework built on Playwright + TypeScript. It exists to test
RAG applications over their API/UI and stays agnostic of which one. `rag_service/` is a small
**reference** RAG service (Python, FastAPI) living in this repo as a system under test; the framework
must not depend on it beyond HTTP.

Planned stack: Hugging Face datasets (golden data — done), FastAPI + ChromaDB + Groq (reference
service — in progress), Ragas (RAG evaluation), Promptfoo (LLM/adversarial), Allure (reporting),
Langfuse (observability).

## Current state — Milestone 2 in progress

Delivered: Milestone 1 (golden evaluation dataset and its validation). Milestone 2 so far in
`rag_service/`: typed settings, `GET /health`, `GET /ready`, and a persistent ChromaDB corpus with
chunked ingestion (`POST /corpus/documents`), stats (`GET /corpus`), guarded reset
(`DELETE /corpus`), plus `scripts/ingest_squad.py` to load the SQuAD validation passages.

**Not built yet — do not assume these exist:** retrieval/search (`/retrieve`), `/query`, Groq
generation, any TS RAG client/adapter, API tests, Playwright projects/fixtures, browser projects,
Ragas, Promptfoo, Allure, Langfuse, UI tests.

When asked to add functionality, build it rather than assuming helpers are available.

## Commands

```bash
npm install
npm test                    # full Playwright suite
npm run test:dataset        # dataset validation only
npm run typecheck           # tsc --noEmit
npm run docs                # TypeDoc -> docs/api (generated, gitignored)
npx playwright test tests/dataset/ragDataset.spec.ts:29   # single test by line

# Dataset regeneration (Python, only when changing selection logic)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepareDataset.py

# Reference RAG service (own venv, Python 3.13 via rag_service/.python-version)
cd rag_service && uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env         # gitignored; set GROQ_API_KEY
pytest                       # service unit tests, offline
python -m app                # serves on API_HOST:API_PORT (default 127.0.0.1:8000)
python scripts/ingest_squad.py [--reset]   # load 1,204 SQuAD passages via the API (~2 min first time)
```

Playwright browsers are already installed locally. No browser project is declared in
`playwright.config.ts` yet — Milestone 1 tests are pure Node assertions.

## Architecture

**The golden dataset is the contract.** `src/models/ragTestCase.ts` defines `RagTestCase`, the
normalized shape every downstream milestone reads. It is deliberately decoupled from SQuAD's native
schema — changing a field there is a breaking change for all future consumers.

Two design decisions worth preserving:

- `context` is `readonly string[]` even though SQuAD supplies exactly one passage per case. Retrieval
  evaluation deals in multiple contexts; the array shape avoids a breaking change later.
- `groundTruth` empty ⟺ `answerable: false`. This invariant is enforced in both the Python generator
  (before writing) and the TS loader (on read). Later scoring logic depends on it to distinguish a
  correct abstention from a miss — do not weaken it.

**Node ↔ Python boundary is permanent.** Ragas is Python-only, so dataset prep lives in
`scripts/prepareDataset.py` while test execution is TypeScript. When results need to reach Ragas,
export a stable interchange format (JSONL) rather than trying to call Ragas from TS.

**Dataset generation is deterministic and must stay that way.** `scripts/prepareDataset.py` uses a
fixed seed (`SEED = 20240611`) plus round-robin selection across SQuAD article titles, and sorts each
title's records before shuffling so the RNG sees stable input ordering. Re-running reproduces the
same 60 cases byte-for-byte (verified via checksum). Global used-question/used-context sets are
shared across the answerable and unanswerable passes, which is why all 60 cases have unique contexts.
Changing the seed, the sort, or the pass order changes the dataset.

`data/golden/rag_test_cases.json` is **generated but committed** — edit the script, never the JSON.

**`rag_service/` is a separate component, coupled only over HTTP.** It has its own
`requirements.txt`, `.gitignore`, venv and pytest suite, independent of the root dataset-prep env.
TS code never imports from `rag_service/`, and the service never reads `src/` — so the framework
stays usable against any external RAG app. Wire format is camelCase (`ApiModel` in
`rag_service/app/schemas.py` aliases snake_case fields), matching the TS side. Settings are a
pydantic-settings `Settings` in `rag_service/app/config.py`, read from env vars then
`rag_service/.env` (cwd-independent) and validated at startup; `get_settings` is a FastAPI
dependency so tests swap it through `app.dependency_overrides`. `/health` contacts nothing; `/ready`
calls Groq once to confirm `GROQ_MODEL` exists for the key — model availability is per-account, so the
default id is never assumed to work.

**Evaluation labels never reach the RAG corpus.** Questions, `groundTruth` and `answerable` stay on
the test side; the service must not read `data/golden/`. A corpus that contains the answers turns
every retrieval and grounding test into a lookup. Enforced twice: `DocumentMetadata` in
`schemas.py` is a closed allowlist (`title`, `source`; unknown keys → 422), and
`scripts/ingest_squad.py` selects only SQuAD's `title` and `context` columns.

**Corpus ingestion is idempotent by document id** (`rag_service/app/corpus.py`). Chunk ids are
`<documentId>#<chunkIndex>`; each chunk stores a `fingerprint` of text + metadata + chunk params +
embedding model. Identical re-ingest → `unchanged`, nothing re-embedded; changed → upsert new chunks
first, then delete leftover ids, so no stale chunk survives and a document is never missing
mid-replace. Writes run under one process-wide lock — run a single uvicorn worker. The collection
records its embedding model and refuses to open under a different one. SQuAD document ids are
`squad-` + first 16 hex of SHA-1(passage), so the TS side can derive a golden case's expected
document id from `context[0]` without the service reading golden data.

**Destructive endpoints are guarded.** `DELETE /corpus` obeys `ALLOW_RESET`: unset → loopback
callers only (checked on the TCP peer address, not `API_HOST`); `true` → anyone; `false` → 403.

## Conventions

- `tsconfig.json` is `strict` with `noUncheckedIndexedAccess`. `tsc` is typecheck-only; Playwright
  transpiles specs itself. Indexed access yields `T | undefined` by design — handle it, don't assert
  it away with `!`.
- `datasetLoader.ts` accumulates all validation issues and reports them together rather than throwing
  on the first fault. Keep that behaviour when extending it.
- Doc comments are **TSDoc, not JSDoc**. Cross-references are `{@link Foo}`; a bare `{Foo}` after
  `@throws`/`@param` is malformed and renders as literal text. `typedoc.json` sets
  `treatValidationWarningsAsErrors`, so `npm run docs` fails on a broken `{@link}` target.
- Python docstrings in `scripts/` and `rag_service/` are Sphinx-style reStructuredText: `:param:`/`:type:`/`:returns:`/
  `:rtype:`/`:raises:` field lists and `:func:` cross-references. Nothing renders them yet; the
  style is for consistency and a future autodoc build.
- `typedoc.json` uses `entryPointStrategy: "expand"` over `src`, so new modules are documented
  automatically. This is deliberate — it avoids the hand-maintained entry-point list that a
  Sphinx-style `automodule` setup requires. Don't replace it with an explicit `entryPoints` array.

## Gotchas

- The package is CommonJS. Do **not** enable `verbatimModuleSyntax` — it conflicts with ESM syntax in
  a CJS package and breaks every file.
- **TypeScript is held at `^5.9.3` (i.e. `<6.0.0`) deliberately — do not upgrade to 7.x.** TypeDoc reads
  the TypeScript compiler API directly, and the 7.x native port does not expose the surface it needs
  (`typedoc@0.28.20` peer-supports `5.0.x`–`6.0.x`, and forcing past that crashes on import with
  `Cannot read properties of undefined (reading 'SyntaxKind')`). The codebase itself typechecks
  cleanly under both. Revisit only when TypeDoc ships TS 7 support.
- `tsconfig.json` sets `noEmit`, so nothing emits `.d.ts`. Any tool that consumes declaration files
  (e.g. API Extractor) needs its own build config — it cannot reuse this one as-is.
- `CHUNK_SIZE` defaults to 1000 characters because all-MiniLM-L6-v2 embeds at most **256 tokens** and
  silently truncates the rest (Chroma's ONNX tokenizer, `max_length=256`). Measured on the SQuAD
  validation passages with 1000/200 word-aligned chunking (`app/chunking.py`): 1,473 chunks, 235 of
  1,204 passages split, 7 chunks (0.5%) over 256 tokens. Raising it well past ~1100 means long passages are only partly searchable.
  Because passages get split, a retrieved chunk is not a whole golden `context` — retrieval
  checks must match on the chunk's parent-passage id, not on text equality.
- `EMBEDDING_MODEL` only accepts `all-MiniLM-L6-v2`: the Sentence Transformers model, but executed by
  Chroma's bundled ONNX build (`MiniLmOnnxEmbedder` in `app/embeddings.py`), not the
  `sentence-transformers` package — this dev machine is an Intel Mac and torch ships no current
  x86_64 macOS wheels (last is 2.2.2). `requirements.txt` pins `onnxruntime<1.24` **only** on Intel
  macOS (newer versions have no wheels there); Linux/Docker get current releases. Don't drop the
  marker or widen the pin.
- Unit tests never load the real embedding model: they inject `FakeEmbedder` (`tests/conftest.py`)
  through the `Embedder` protocol. Only `scripts/ingest_squad.py` against a running service
  exercises the ONNX model (first full ingest ≈ 113 s on this machine, ~660 MB RSS).
- `rag_service/` tests depend on **`httpx2`, not `httpx`**. Current Starlette deprecates plain `httpx`
  behind FastAPI's `TestClient` and warns on every run; don't "fix" the name back.
- This repo does **not** auto-delete branches on PR merge (`delete_branch_on_merge` is off). Delete
  both the local and the remote branch yourself after a PR merges.

## Project tooling

`.claude/` carries the repo's own agent tooling:

- `skills/commit-rules` — how to scope, stage, and word a commit. Consult before any `git commit`.
- `skills/review-conventions` — the project-specific convention/invariant pass. Complements
  `/code-review`; it does not replace it.
- `skills/update-claude-md` — how to reconcile this file after a significant change.
- `hooks/guard-commit.sh` — blocks `git commit` when a `.env` file or credential-shaped token is
  staged (`PreToolUse`).
- `hooks/guard-foreign-refs.sh` — blocks `git commit` when staged changes name another repository.
  The names are derived from sibling checkouts at run time, so none is hard-coded here; add a
  legitimate one to `.claude/allowed-refs.txt`.
- `hooks/claude-md-reminder.sh` — on `Stop`, flags that `src/`, `scripts/` or `rag_service/` changed while CLAUDE.md
  did not. All hooks are fail-open and loop-safe.

## Working style

- State assumptions; if multiple interpretations exist, ask rather than pick silently.
- Minimum code that solves the problem — no speculative abstractions or config.
- Surgical edits: match existing style, touch only what the request requires, clean up only the
  orphans your change creates.
- Define a verifiable success check before coding. For test-framework work that usually means a
  failing assertion your change makes pass — not "it looks right".
- Verify claims against the code, not memory: open the file, grep the symbol, run the command. Report
  what the run actually printed, including when it failed.
- Every exported symbol carries a TSDoc comment. After touching exported symbols or their comments,
  run `npm run docs` and confirm it exits 0.
- If you notice unrelated dead code, mention it — don't delete it unless asked.
- Commits are atomic: one logical change per commit, each leaving the repo working (full rules in
  `.claude/skills/commit-rules/`).
- For multistep tasks, state a brief plan:
    ```
    1. [Step] → verify: [check]
    2. [Step] → verify: [check]
    3. [Step] → verify: [check]
    ```
