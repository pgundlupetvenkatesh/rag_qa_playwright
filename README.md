# rag_qa_playwright

RAG QA end-2-end with Playwright.

A reusable AI/RAG quality-engineering framework built on Playwright + TypeScript. It is
designed to test **existing** RAG applications — this repository does not contain a RAG
application of its own.

## Golden Dataset

The current milestone covers only the golden evaluation dataset: loading SQuAD 2.0,
normalizing a small slice of it, and validating that slice from a Playwright test.
No RAG client, retrieval scoring, adversarial testing, or UI/API tests exist yet.

### What SQuAD 2.0 is

[SQuAD 2.0](https://huggingface.co/datasets/rajpurkar/squad_v2) (Stanford Question
Answering Dataset) is a reading-comprehension benchmark. Each record pairs a question
with a paragraph of context drawn from a Wikipedia article. Version 2.0's defining
feature is that it extends the original dataset with **unanswerable** questions —
questions that look plausible against the context but have no supported answer in it.

### Why we use it for RAG QA

A RAG system has two failure modes, and SQuAD 2.0 exercises both with human-verified
ground truth:

- **Answerable cases** test whether the system retrieves the right context and grounds
  its answer in it.
- **Unanswerable cases** test whether the system *abstains*. A RAG system that
  confidently answers an unanswerable question is hallucinating, and this is precisely
  the behaviour that generic QA benchmarks miss.

Because each case ships with its own context passage, we get retrieval ground truth for
free — later milestones can score whether a system under test retrieved the right
passage, not just whether its final answer looked reasonable.

We use the `validation` split only; the training split is deliberately excluded, since
models under test are far more likely to have memorized it.

### Dataset composition

`data/golden/rag_test_cases.json` contains **60 test cases**:

| Kind | Count | `groundTruth` |
| --- | --- | --- |
| Answerable | 50 | one or more accepted answer strings |
| Unanswerable | 10 | empty array |

Selection is **deterministic** (fixed seed, round-robin across SQuAD article titles), so
re-running the preparation script reproduces the same 60 cases byte-for-byte. The
selection spans all 35 article titles in the validation split, with unique question and
context text in every case, rather than 60 consecutive records from one topic.

Each case has this shape:

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

## Setup

### Node

```bash
npm install
```

### Python (only needed to regenerate the dataset)

The generated JSON is committed, so you only need this to rebuild it.

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The single dependency is Hugging Face [`datasets`](https://pypi.org/project/datasets/),
which downloads and reads the SQuAD 2.0 parquet files for us.

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

### Generate the API documentation

```bash
npm run docs
```

Runs [TypeDoc](https://typedoc.org) over `src/`, writing a browsable HTML reference to
`docs/api/` (generated, not committed — open `docs/api/index.html`). Doc comments are
TSDoc, so cross-references use `{@link Foo}`; the build fails on a `{@link}` that
does not resolve, rather than emitting a dead link.

> **TypeScript is held at `^5.9.3` for this.** TypeDoc reads the TypeScript compiler API
> directly and does not yet support the 7.x native port.

Other commands:

```bash
npm test          # run the full Playwright suite
npm run typecheck # tsc --noEmit
```

## Layout

```
data/golden/rag_test_cases.json   generated golden dataset (committed)
scripts/prepareDataset.py         SQuAD 2.0 -> normalized JSON
src/models/ragTestCase.ts         normalized test case contract
src/utils/datasetLoader.ts        loads + validates the golden dataset
tests/dataset/ragDataset.spec.ts  dataset validation test
typedoc.json                      API documentation config
docs/api/                         generated API reference (gitignored)
```

## Planned

Not yet implemented: Ragas (RAG evaluation), Promptfoo (LLM/adversarial testing),
Allure (reporting), Langfuse (observability), and the Playwright UI/API adapters for
driving a system under test.