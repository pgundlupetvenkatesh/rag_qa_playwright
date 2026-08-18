# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A reusable AI/RAG quality-engineering framework built on Playwright + TypeScript. It exists to test
**existing, external** RAG applications — we do not build a RAG application here, and no system under
test is wired up yet.

Planned stack, none of it implemented yet: Hugging Face datasets (golden data — done), Ragas (RAG
evaluation), Promptfoo (LLM/adversarial), Allure (reporting), Langfuse (observability).

## Current state — Milestone 1 only

Delivered: the golden evaluation dataset and its validation. That is all.

**Not built yet — do not assume these exist:** any RAG client/adapter, Ragas, Promptfoo, Allure,
Langfuse, UI tests, API tests, Playwright fixtures, browser projects.

When asked to add functionality, build it rather than assuming helpers are available.

## Commands

```bash
npm install
npm test                    # full Playwright suite
npm run test:dataset        # dataset validation only
npm run typecheck           # tsc --noEmit
npx playwright test tests/dataset/ragDataset.spec.ts:29   # single test by line

# Dataset regeneration (Python, only when changing selection logic)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepareDataset.py
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

## Conventions

- `tsconfig.json` is `strict` with `noUncheckedIndexedAccess`. `tsc` is typecheck-only; Playwright
  transpiles specs itself.
- The package is CommonJS. Do **not** enable `verbatimModuleSyntax` — it conflicts with ESM syntax in
  a CJS package and breaks every file.
- TypeScript is 7.x (the native port). It typechecks this codebase cleanly, but it is new; if odd
  compiler behaviour appears, pinning to 5.x is a reasonable fallback.
- `datasetLoader.ts` accumulates all validation issues and reports them together rather than throwing
  on the first fault. Keep that behaviour when extending it.