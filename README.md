# rag_qa_playwright

RAG QA end-2-end with Playwright.

A reusable AI/RAG QA Engineering framework built on Playwright + TypeScript. It is
designed to test **existing** RAG applications — this repo does not contain a RAG
application of its own.

## Golden Dataset

Loading SQuAD 2.0, normalizing a small slice of it, and validating that slice from a Playwright test.
No RAG client, retrieval scoring, adversarial testing, or UI/API tests exist yet.

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

Runs [TypeDoc](https://typedoc.org) over `src/` and writes a browsable HTML reference to
`docs/api/index.html`. Doc comments are TSDoc, so cross-references use `{@link Foo}`; the build fails on a `{@link}` that
does not resolve, rather than emitting a dead link.

> TypeScript is pinned to `^5.9.3` as `TypeDoc` reads the TypeScript compiler API
> directly and does not natively yet support 7.x

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