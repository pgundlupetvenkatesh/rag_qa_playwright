---
name: review-conventions
description: >-
  Review changes to this RAG QA framework against its own conventions and
  invariants — the project-specific checks that generic code review misses.
  Use after writing or editing anything under src/, scripts/, tests/, or the
  golden dataset, and before committing/opening a PR. Covers the
  groundTruth/answerable invariant, dataset determinism, the RagTestCase
  contract, the generated-but-committed dataset, error accumulation in the
  loader, TSDoc/TypeDoc rules, the TypeScript version pin, and references to
  other projects left behind by copied-in files. Complements
  (does not replace) the built-in /code-review for generic bugs.
---

# Review changes against this repo's conventions

Review the current diff for violations of *this repo's* rules. These are the
ones generic review can't know — they come from CLAUDE.md and the code. Run
`/code-review` separately for correctness bugs; this skill is only the
convention/invariant pass.

## Scope

Look at the change under review (`git diff`, `git diff --staged`, or the edits
just made). For each file touched, run the checks below that apply. Verify
every claim against the actual code — open the file, grep for the symbol — do
not assert from memory.

## Checks

### 1. The groundTruth/answerable invariant (highest-value)
`groundTruth` empty ⟺ `answerable: false`. This is enforced in **two** places
and both must stay in agreement:
- `scripts/prepareDataset.py` — before writing (`to_test_case` sets
  `ground_truth` to `[]` when not answerable).
- `src/utils/datasetLoader.ts` — on read (both directions are checked around
  the `answerable && groundTruth.length === 0` /
  `!answerable && groundTruth.length > 0` branches).

Later scoring logic uses this to tell a correct abstention from a miss. A
change that weakens either side, or adds a third path that constructs a
`RagTestCase` without enforcing it, is a bug. Grep both files and confirm the
check still exists in each.

### 2. Never hand-edit the generated dataset
`data/golden/rag_test_cases.json` is generated but committed. A diff that
edits the JSON directly — without a corresponding change to
`scripts/prepareDataset.py` — is wrong, even if the JSON looks correct. The
script is the source of truth; the JSON is its output.

### 3. Dataset determinism
Re-running the generator must reproduce the same 60 cases byte-for-byte. That
rests on four things in `scripts/prepareDataset.py`, all of which are load-bearing:
- the fixed seed (`SEED = 20240611`),
- sorting each title's records (`sorted(grouped[title], key=...id)`) *before*
  `rng.shuffle`, so the RNG sees stable input ordering,
- `titles = sorted(grouped)` for round-robin order,
- one shared `rng` plus shared `used_questions` / `used_contexts` sets across
  the answerable and unanswerable passes — which is why all 60 contexts are
  unique.

Flag any change to seed, sort, pass order, or the sharing of those sets unless
regenerating the dataset is the explicit intent. If it is intentional, the
committed JSON must be regenerated in the same commit.

### 4. RagTestCase is a breaking-change contract
`src/models/ragTestCase.ts` is the shape every downstream milestone reads.
Adding/removing/renaming/retyping a field breaks all future consumers — flag
it for explicit confirmation rather than treating it as an ordinary edit. Two
deliberate decisions to preserve:
- `context` is `readonly string[]` even though SQuAD supplies exactly one
  passage. Do not "simplify" it to a bare `string`.
- The type is decoupled from SQuAD's native schema on purpose. Don't
  reintroduce SQuAD field names into it.

### 5. The loader accumulates, it doesn't fail fast
`datasetLoader.ts` appends every problem to an `issues` array and reports them
together; it throws `DatasetValidationError` only at the boundaries (missing
file, invalid JSON, not an array). A new validation that `throw`s on the first
bad case instead of pushing to `issues` breaks that contract — flag it.

### 6. TSDoc, not JSDoc
Doc comments are TSDoc. Cross-references are `{@link Foo}`; a bare `{Foo}`
after `@throws`/`@param` is malformed and renders as literal text.
`typedoc.json` sets `treatValidationWarningsAsErrors`, so an unresolved
`{@link}` fails `npm run docs`. If the diff touches exported symbols or their
comments, confirm `npm run docs` still exits 0 — don't assume.

### 7. Don't move TypeScript off the 5.x line
`package.json` holds `typescript` at `^5.9.3` (i.e. `<6.0.0`) deliberately:
TypeDoc reads the TS compiler API directly and crashes on the 7.x native
port. A diff bumping TypeScript past 6.0.x breaks `npm run docs`. Flag it and
point at the CLAUDE.md note rather than letting it through as a routine
dependency bump. Likewise, do not enable `verbatimModuleSyntax` — it breaks
every file in this CommonJS package.

### 8. Node ↔ Python boundary
Ragas is Python-only; dataset prep is Python, test execution is TypeScript.
Flag any attempt to call Python from TS at runtime, or to reimplement the
generator in TS. Results cross the boundary as a stable interchange format
(JSONL), not via in-process calls.

### 9. Types stay strict
`tsconfig.json` is `strict` with `noUncheckedIndexedAccess`. Flag new `any`,
non-null assertions (`!`), or `as` casts introduced to silence the compiler
rather than to express a real narrowing. Indexed access returns `T | undefined`
here by design — code must handle it, not assert it away.

### 10. No cross-project references
Files copied in from another checkout drag its name, stack terms, and doc
references along. None of it means anything here, and it goes stale silently
because nothing in this repo exercises it. Flag any mention of another
repository, its tooling, or its documentation — in prose, code comments, or
config — and restate the point in this project's own terms.

`.claude/hooks/guard-foreign-refs.sh` blocks a commit that introduces one,
deriving the names from sibling checkouts at run time so no foreign name is
hard-coded here. Two limits it cannot cover, which is where this check earns
its place:
- It only inspects **staged** changes. A reference already sitting in the tree
  never trips it.
- It matches repository *names*. A borrowed stack term, a URL, or a sentence
  describing another project's setup reads as ordinary prose to it.

When reviewing a file copied in from elsewhere, read its **comments and prose**,
not just its logic — that is where these survive longest.

If a mention is genuinely needed, add the name to `.claude/allowed-refs.txt`
(one per line) rather than weakening the hook.

### 11. CLAUDE.md drift
If the change adds/removes/renames a module, script, command, npm script,
convention, gotcha, or dependency — or lands a milestone — CLAUDE.md likely
needs updating. Note it and hand off to the `update-claude-md` skill; don't
edit CLAUDE.md inline as part of the review.

## Output

Report findings grouped by file, each as: location → which check it violates →
the fix. If a check passes cleanly, say so briefly rather than listing it.
Don't invent issues to fill the list — a clean diff is a valid result.
