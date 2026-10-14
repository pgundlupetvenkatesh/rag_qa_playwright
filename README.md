# rag_qa_playwright

RAG QA end-2-end with Playwright.

A reusable AI/RAG QA Engineering framework built on Playwright + TypeScript. It is
designed to test RAG applications through their API and UI, without depending on
any particular one. To have something real to test, the repo also ships a small
**reference RAG service** in [`rag_service/`](#reference-rag-service). It is a
separate Python component that the framework reaches only over HTTP, exactly as
it would reach an external application.

## Architecture and data flow

```
 TypeScript framework (Playwright)                    rag_service (Python, FastAPI)
 ─────────────────────────────────                    ─────────────────────────────
 data/golden/rag_test_cases.json                      ChromaDB corpus (.chroma/)
   questions + labels stay here                         passages only, no labels
          │                                                    ▲
          ▼                                                    │ chunk + embed (local MiniLM)
 tests ──► RagClient ──► RagServiceClient ──HTTP──► POST /corpus/documents
                                          ──HTTP──► POST /retrieve ──► nearest chunks
                                          ──HTTP──► POST /query ────► nearest chunks ──► Groq LLM
                                                                                          │
                                          ◄──────── answer or abstention + evidence ◄─────┘
```

- **Golden data** (questions, accepted answers, `answerable`) never leaves the test
  side. Only passage text, title and source are ingested.
- **Tests depend on `RagClient`**, not on `rag_service`. Another RAG app needs only
  another adapter.
- **Only `/query` calls the LLM.** Ingestion and `/retrieve` run locally at no cost.

[`rag_service/README.md`](rag_service/README.md#architecture-and-data-flow) has the
service-side detail, and [`INSIGHT.md`](INSIGHT.md) explains the concepts from scratch.

## Golden Dataset

Loading SQuAD 2.0, normalizing a small slice of it, and validating that slice from a Playwright test.
Retrieval scoring, adversarial testing and UI tests don't exist yet.

### How the pieces fit

```
SQuAD 2.0 (Hugging Face)
        │
        ▼
scripts/prepareDataset.py        Python: select 60 records, normalize to RagTestCase
        │
        ▼
data/golden/rag_test_cases.json  generated, committed
        │
        ▼
src/utils/datasetLoader.ts       TypeScript: read, validate structure, type as RagTestCase[]
        │
        ▼
tests/dataset/ragDataset.spec.ts Playwright: assert composition, invariants, provenance
```

1. **Python normalizes.** `scripts/prepareDataset.py` downloads SQuAD, selects the cases,
   and rewrites each one into the `RagTestCase` shape. This is the only step that
   transforms data, and Node is not involved in it.
2. **The loader validates.** `src/utils/datasetLoader.ts` reads the committed JSON, checks
   every field, and returns a typed array. It throws with every fault listed if anything
   is wrong.
3. **Playwright asserts.** `tests/dataset/ragDataset.spec.ts` loads through that loader and
   asserts the counts, the `groundTruth`/`answerable` invariant, and the provenance.

Playwright is used here purely as a test runner, the role Jest or Vitest would otherwise
fill. `playwright.config.ts` declares no browser project, and the spec never touches a
`page`. It is Playwright rather than a lighter runner because later milestones will drive
real RAG applications through their UI and API, and using it from the start means one
runner, one reporter, and one `npm test` for the whole framework.

### SQuAD 2.0?

[SQuAD 2.0](https://huggingface.co/datasets/rajpurkar/squad_v2) (Stanford Question
Answering Dataset) is a reading-comprehension benchmark. Each record pairs a question
with a paragraph of context drawn from a Wikipedia article. Version 2.0's defining
feature is that it extends the original dataset with **unanswerable** questions —
questions that look plausible against the context but have no supported answer in it.

### Why use it for RAG QA

A RAG system has two failure modes, and SQuAD 2.0 exercises both with human-verified
ground truth:

- **Answerable cases** test whether the system retrieves the right context and grounds
  its answer in it.
- **Unanswerable cases** test whether the system *abstains*. A RAG system that
  confidently answers an unanswerable question is hallucinating, and this is precisely
  the behavior that generic QA benchmarks miss.

Because each case ships with its own context passage, we get retrieval ground truth for
free — later milestones can score whether a system under test retrieved the right
passage, not just whether its final answer looked reasonable.

We use only the SQuAD `validation` split; the training split is deliberately excluded, since
models under test are far more likely to have memorized it.

### Dataset composition

`data/golden/rag_test_cases.json` contains **60 test cases**:

| Kind         | Count | `groundTruth`                       |
|--------------|-------|-------------------------------------|
| Answerable   | 50    | one or more accepted answer strings |
| Unanswerable | 10    | empty array                         |

Selection is **deterministic** (fixed seed, round-robin across SQuAD article titles), so
re-running the preparation script reproduces the same 60 cases byte-for-byte. The
selection spans all 35 article titles in the validation split, with unique question and
context text in every case, rather than 60 consecutive records from one topic.

Shape of each case:

```json
{
  "id": "RAG-0001",
  "question": "When did the Shah kingdom start to collapse?",
  "context": ["The USSR's invasion of Afghanistan was only one sign of insecurity..."],
  "groundTruth": ["January 1979", "1979"],
  "answerable": true,
  "source": "squad_v2",
  "metadata": {
    "squadId": "57265526708984140094c2c0",
    "title": "1973_oil_crisis",
    "split": "validation"
  }
}
```

Questions and contexts are copied from SQuAD verbatim — never summarized or reformatted.

Field by field:

| Field         | Type       | What it is                                                                                                                                                            |
|---------------|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `id`          | `string`   | Stable framework id, `RAG-0001` to `RAG-0060`. Unique across the file; results in later milestones are keyed on it.                                                   |
| `question`    | `string`   | The user query, verbatim from SQuAD. This is what gets sent to the system under test.                                                                                 |
| `context`     | `string[]` | The passage the answer must be grounded in. Always one element today, but an array so retrieval evaluation can hold several passages later without a breaking change. |
| `groundTruth` | `string[]` | Accepted answer strings. One or more for answerable cases, exactly empty for unanswerable ones. Empty means the correct behaviour is to abstain.                      |
| `answerable`  | `boolean`  | Whether the question can be answered from the context. `false` for SQuAD 2.0's deliberately unanswerable questions.                                                   |
| `source`      | `string`   | Upstream dataset identifier, always `squad_v2`. Checked against an exported constant so provenance cannot drift silently.                                             |
| `metadata`    | `object`   | Trail back to the original record, see below.                                                                                                                         |

The `metadata` object:

| Field     | What it is |
|-----------|------------|
| `squadId` | The original SQuAD record id, for looking up the source row when a case fails. |
| `title`   | The Wikipedia article title, e.g. `1973_oil_crisis`. Doubles as a topic label. |
| `split`   | Always `validation`. Training data is excluded so the benchmark cannot leak into a model's training set. |

`groundTruth` and `answerable` must agree: `groundTruth` is empty if and only if
`answerable` is `false`. The rule is enforced three times, in the Python generator before
writing, in the TypeScript loader on read, and in the spec as named tests.

### Why round-robin selection

Sixty cases should represent the whole validation split, not a couple of Wikipedia
articles. The committed dataset shows the effect:

| Pass         | Cases | Distinct titles | Max per title |
|--------------|-------|-----------------|---------------|
| Answerable   | 50    | 35              | 2             |
| Unanswerable | 10    | 10              | 1             |

SQuAD is stored article by article. Each article has a handful of paragraphs, each
paragraph carries several questions written against the same text, and the articles are
very unequal in size. That structure defeats the two obvious ways of picking a subset:

- **A consecutive slice** of the first 60 rows comes almost entirely from one article,
  with many questions sharing the same paragraph.
- **A uniform random sample** inherits the size imbalance. Large articles dominate in
  proportion to their question count, popular paragraphs are picked repeatedly, and with
  only 60 draws several small articles are missed by chance.

Round-robin treats articles, not questions, as the unit of fairness. Each lap gives every
article one turn, so after one lap the dataset spans every topic. Only once every article
has contributed does the second lap hand out second slots. Article size no longer affects
how often it appears.

This matters for a RAG benchmark because retrieval quality is only meaningful if the
questions probe different regions of the corpus. If most questions target the same
passage, a retriever that ranks that passage well looks better than it is. Abstention
behaviour and grounding also vary by subject, and a narrow sample hides that variance.

Round-robin alone is not enough; three other mechanisms in `select_round_robin` make it
work:

- **Within-article randomness.** Each article's queue is shuffled with the seeded
  generator before the laps begin. Otherwise every lap would take the first paragraph of
  every article, usually the lead section.
- **Duplicate rejection.** A candidate is skipped if its normalized question or its exact
  context has already been used. This pushes second-lap picks onto different paragraphs.
  The used sets are shared across both passes, so the unanswerable pass cannot land on a
  passage the answerable pass already claimed.
- **Determinism.** Titles are visited in sorted order and each queue is sorted by record
  id before the shuffle, so the seeded generator sees identical input on every run.
  Changing the seed, the sorts, or the pass order changes the dataset.

The trade-off is that this is not a statistical sample of the split. An article with 300
questions and one with 30 are weighted the same, so the overall score does not mirror
SQuAD's natural topic mix. For a quality-engineering fixture that is the right choice:
the goal is broad, reproducible, hand-inspectable coverage, not an unbiased estimate of
performance on SQuAD as a whole. The generator also fails loudly if the corpus ever
becomes too small or too duplicated to fill the quota, rather than silently returning
fewer cases.

## Reference RAG service

`rag_service/` is a lightweight RAG application written in Python with FastAPI. It
is the system under test for the Playwright API suite. It is being built in small
steps. So far it has probes, a document corpus, retrieval and grounded answer generation:

| Endpoint                 | Meaning                                                                      |
|--------------------------|------------------------------------------------------------------------------|
| `GET /health`            | Liveness: 200 while the process is up.                                       |
| `GET /ready`             | Readiness: 200 or 503, depending on whether a Groq key is set, whether Groq confirms the configured model exists for that key, and whether the Chroma directory is usable. |
| `GET /corpus`            | How many documents and chunks the corpus holds.                              |
| `POST /corpus/documents` | Add or replace documents by stable id. Each is chunked, embedded on the CPU with all-MiniLM-L6-v2, and stored in ChromaDB. Re-sending the same document is a no-op. |
| `DELETE /corpus`         | Empty the corpus. Allowed only from the local machine unless configured otherwise. |
| `POST /retrieve`         | The top-k chunks nearest to a question, with their parent document ids and distances. Never calls the language model. |
| `POST /query`            | Retrieves evidence, then asks a Groq model to answer from that evidence alone or abstain. Returns the answer, the evidence, the model id, token counts and timings. |

`rag_service/scripts/ingest_squad.py` loads the 1,204 unique SQuAD 2.0 validation
passages through that API. It reads only passage text and titles, never questions or
answers.
[`rag_service/README.md`](rag_service/README.md) has the full setup, the list of
configuration variables, the ingestion rules, and measured performance.

Two rules keep it honest as a test target:

- **HTTP is the only coupling.** The TypeScript framework never imports from
  `rag_service/`, and the service never reads `src/`. Pointing the tests at a
  different RAG application should need nothing more than a different base URL.
- **Evaluation labels stay out of the corpus.** The service never reads
  `data/golden/`, so questions, accepted answers, and the `answerable` flag are
  never visible to it. If the corpus contained the answers, every retrieval test
  would pass trivially.

The Groq model id is configurable, and the default (`openai/gpt-oss-120b`) is not
assumed to work. Which models a key can use depends on the Groq account, so
`/ready` asks Groq and reports whether the configured model is available.

### Embeddings

Retrieval finds passages by meaning, not by keyword. The embedding model
`all-MiniLM-L6-v2` turns each chunk, and later each question, into a vector of 384
numbers, with similar meanings giving nearby vectors. ChromaDB stores the chunk
vectors and returns the ones closest to the question by cosine similarity. The API
reports `similarity = 1 - distance`, so higher means more related. Only those chunks
go to the Groq model, which writes the answer.

| | Embedding model | Language model |
|---|---|---|
| Job | Find the relevant chunks | Write the answer from them |
| Model | `all-MiniLM-L6-v2` | `openai/gpt-oss-120b` (default) |
| Runs | Locally, through Chroma's ONNX build, no key needed | On Groq, needs `GROQ_API_KEY` |

- **Limit:** it reads at most 256 tokens, which is why passages are split into
  chunks of about 1,000 characters.
- **One model per corpus:** vectors from different models can't be compared, so the
  collection refuses to open under a different model. Switching models means
  re-ingesting the whole corpus.
- **Why MiniLM is kept for now:** Groq offers no embedding model to this account.
  Stronger local models exist, such as `BAAI/bge-small-en-v1.5`, but switching waits
  for a retrieval hit-rate metric that can show whether one actually retrieves better.

[`INSIGHT.md`](INSIGHT.md) explains vectors and embeddings from the beginning and
compares other models.

## Setup

### Node

```bash
npm install
```

### Python (only for dataset regeneration)

The generated JSON is committed, so you only need this to rebuild it.

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The single dependency is Hugging Face [`datasets`](https://pypi.org/project/datasets/),
which downloads and reads the SQuAD 2.0 parquet files for us.

### Python (reference RAG service)

The service has its own requirements and virtualenv, kept apart from the dataset one,
and runs on Python 3.13 (pinned in `rag_service/.python-version`). With
[uv](https://docs.astral.sh/uv/):

```bash
cd rag_service
uv venv .venv                  # picks up Python 3.13 from .python-version
source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env           # then set GROQ_API_KEY; .env is gitignored
```

Put your real key only in `rag_service/.env` (gitignored, and the commit hook blocks
staging any `.env` file) or in a real environment variable. `.env.example` holds
placeholders only. See
[Groq API key](rag_service/README.md#groq-api-key) for handling, CI and what to do if a
key leaks.

## Usage

### Regenerate the golden dataset

```bash
python scripts/prepareDataset.py
```

Downloads the SQuAD 2.0 `validation` split (cached by Hugging Face after the first run)
and rewrites `data/golden/rag_test_cases.json`. The script fails before writing if the
composition is ever wrong.

### Validate the dataset

```bash
npm run test:dataset        # or: npx playwright test tests/dataset
```

Asserts the counts, id uniqueness, and the answerable/`groundTruth` invariants.

The spec tests the contract, not the content. It does not check the text of any case, the
title spread, or a checksum of the file: exact content would duplicate the fixture in the
test and fail on every legitimate regeneration, while git already surfaces any change to
the committed JSON in the diff. What it guards against is the realistic failure modes of a
generated-but-committed file: a hand edit that drops an answer, a merge that duplicates a
block, a partial regeneration, or a generator change that shifts the answerable split.

Every later milestone will start by loading this dataset, so this spec is the smoke test
that runs first. If it fails, there is no point running retrieval scoring or adversarial
prompts, because the fixture they would score against is broken. That is why it exists as
a separate, fast, browser-free suite rather than as a precondition buried inside a future
evaluation test.

### Run and test the reference RAG service

From `rag_service/` with its virtualenv active:

```bash
pytest                              # service unit tests (offline)
python -m app                       # serve on API_HOST:API_PORT, default 127.0.0.1:8000
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
```

Load the corpus once, with the service running (SQuAD passages and titles only; about
2 minutes the first time, under a second when repeated):

```bash
python scripts/ingest_squad.py
curl http://127.0.0.1:8000/corpus      # {"documents": 1204, "chunks": 1473, ...}
```

Example requests:

```bash
# Retrieval only: no LLM, free.
curl -s -X POST http://127.0.0.1:8000/retrieve -H 'content-type: application/json' \
  -d '{"question": "In what country is Normandy located?", "topK": 3}'

# Retrieval plus a grounded answer: one Groq call.
curl -s -X POST http://127.0.0.1:8000/query -H 'content-type: application/json' \
  -d '{"question": "In what country is Normandy located?"}'
```

The unit tests ignore `.env` and the shell environment, inject settings through FastAPI's
dependency overrides, and replace Groq with a mocked client or HTTP transport. They need neither a
key nor a network connection.

### Test a RAG service through the framework client

Tests talk to a RAG system only through the `RagClient` interface
(`src/clients/ragClient.ts`), which covers readiness, ingestion, retrieval and answers.
`RagServiceClient` is its adapter for the reference service, built on Playwright's
`APIRequestContext`. To target another RAG app, write another adapter; the tests stay as
they are. Two environment variables configure where the service is and how long to wait:

| Variable         | Default                 | Meaning                                                    |
| ---------------- | ----------------------- | ---------------------------------------------------------- |
| `RAG_BASE_URL`   | `http://127.0.0.1:8000` | Base URL of the RAG service under test.                    |
| `RAG_TIMEOUT_MS` | `60000`                 | Per-request timeout. Keep it above the service's Groq timeout. |

The suites are split by what they need:

```bash
npm run test:client       # adapter vs a local mock server: offline, no service, no key
npm run test:service      # a running service; ingestion and retrieval, no LLM calls
npm run test:groq         # a running service holding a working GROQ_API_KEY (live calls)
npm run test:integration  # service + groq
```

| Suite | Needs | Groq calls |
|---|---|---|
| `npm test` (`dataset` + `client`) | Nothing (offline) | None |
| `pytest` in `rag_service/` | Nothing (offline; Groq and the embedder are faked) | None |
| `npm run test:service` | The service running at `RAG_BASE_URL` | None |
| `npm run test:groq` | The service running **with a working `GROQ_API_KEY`** | About 2 answers per run |

Every test that reaches Groq is tagged `@groq`, and only the `groq` project runs them,
so the ordinary commands never use your Groq quota.

`service` and `groq` first run a setup step that ingests the 60 golden passages through
`toCorpusDocuments` (`src/utils/corpusDocuments.ts`). That helper sends each distinct
passage once, under the same `squad-<sha1>` id that `ingest_squad.py` uses, with only its
title and source; questions, answers and `answerable` never leave the test side. Ingestion
is idempotent, so against an already-loaded corpus the setup reports all 60 as unchanged,
and nothing is ever deleted. The Groq tests check the response contract (an answer, or
`null` with `abstained: true`, plus evidence), not whether the answer is right.

### Generate the API documentation

```bash
npm run docs
```

Runs [TypeDoc](https://typedoc.org) over `src/` and writes a browsable HTML reference to
`docs/api/index.html`. Doc comments are TSDoc, so cross-references use `{@link Foo}`; the build fails on a `{@link}` that
does not resolve, rather than emitting a dead link.

> TypeScript is pinned to `^5.9.3` as `TypeDoc` reads the TypeScript compiler API
> directly and does not natively yet support 7.x

Other commands:

```bash
npm test          # the offline suites: dataset + client (no service needed)
npm run typecheck # tsc --noEmit
```

## Layout

```
data/golden/rag_test_cases.json   generated golden dataset (committed)
scripts/prepareDataset.py         SQuAD 2.0 -> normalized JSON
src/models/ragTestCase.ts         normalized test case contract
src/utils/datasetLoader.ts        loads + validates the golden dataset
src/utils/corpusDocuments.ts      golden cases -> label-free corpus documents
src/clients/ragClient.ts          RagClient interface and types (system-agnostic)
src/clients/ragServiceClient.ts   RagClient adapter for rag_service (APIRequestContext)
src/config/ragServiceConfig.ts    RAG_BASE_URL / RAG_TIMEOUT_MS
tests/dataset/ragDataset.spec.ts  dataset validation test
tests/client/                     adapter tests against a local mock server, config tests (offline)
tests/integration/                tests against a running service; @groq ones need a live key
typedoc.json                      API documentation config
docs/api/                         generated API reference (gitignored)
rag_service/app/                  reference RAG service: FastAPI app, settings, readiness, chunking, corpus and retrieval, Groq generation
rag_service/scripts/ingest_squad.py  loads SQuAD validation passages into the service
rag_service/tests/                service unit tests (pytest)
rag_service/requirements.txt      service Python dependencies
rag_service/.env.example          placeholder-only template for the gitignored rag_service/.env
rag_service/README.md             service setup, configuration, ingestion, performance
INSIGHT.md                        beginner explainer: RAG, vectors, embeddings, design choices
```

## Known limitations and costs

- **Costs:** only Groq calls cost anything. That means one completion per `/query`
  (roughly 800-1,000 prompt tokens with the default 4 chunks) and one model listing per
  `/ready`. On the free tier these count against rate limits. Everything else runs
  locally.
- **The reference service has no authentication** and runs as one worker process. It is
  meant for local testing, bound to `127.0.0.1`.
- **The tests check contracts, not quality.** Nothing yet measures whether retrieval found
  the right passage or the answer is correct.
- **Retrieval has no relevance threshold.** Abstention relies on the model.

The full list is in
[`rag_service/README.md`](rag_service/README.md#known-limitations-and-costs).

## Planned

Not yet implemented: retrieval and answer-quality metrics over the golden set, Ragas (RAG evaluation),
Promptfoo (LLM/adversarial testing), Allure (reporting), Langfuse (observability),
and UI tests.