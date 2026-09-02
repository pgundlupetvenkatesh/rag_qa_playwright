---
name: update-claude-md
description: >-
  Update CLAUDE.md so it stays accurate after a significant change to this
  RAG QA framework. Use after adding/removing/renaming a module under src/,
  a test spec, a Playwright fixture or project, or a script; changing how
  things are run (npm scripts, Playwright flags, the Python dataset
  pipeline); changing the RagTestCase contract or the dataset composition;
  adding a dependency or tool (Ragas, Promptfoo, Allure, Langfuse); or
  introducing a new convention or gotcha. Trigger when finishing such a
  change, before committing, or when the user asks to "update CLAUDE.md" /
  "keep the docs in sync". Skip for dataset value tweaks, formatting, or
  one-off fixes that don't change structure, commands, conventions, or
  gotchas.
---

# Keep CLAUDE.md in sync

CLAUDE.md is the single onboarding doc for this repo (see its own sections:
What this project is, Current state, Commands, Architecture, Conventions,
Gotchas). It must keep matching the actual code. After a significant change,
reconcile it.

## What counts as "significant"

Update CLAUDE.md when a change touches any of these:

- **Commands** — a new/changed npm script, Playwright invocation or flag, a
  new test directory, a change to the Python dataset pipeline.
- **Current state** — a milestone landing. This section names what exists and
  explicitly lists what does *not*; a new capability must move from the
  "not built yet" list into the delivered list. A stale "not built yet" entry
  is worse than no entry, because it tells the next session to rebuild
  something that already exists.
- **Architecture** — a new or removed module under `src/`, a change to the
  `RagTestCase` contract, the Node↔Python boundary, the dataset generator's
  selection logic, or how a later tool (Ragas, Promptfoo, Allure, Langfuse)
  is wired in.
- **Conventions** — a new pattern code must follow (typing, doc comments,
  error accumulation, fixture style).
- **Gotchas** — a new trap: a version pin with a reason, an invariant enforced
  in two places, a non-obvious failure mode.
- **Dependencies** — a package added/removed that changes how the project is
  set up or run, especially one that constrains a version elsewhere.

Skip it for: dataset value tweaks, formatting/lint-only diffs, comment
changes, and one-off bug fixes that don't alter structure, commands,
conventions, or gotchas.

## Procedure

1. **See what changed.** Review the diff (`git diff`, `git diff --staged`, or
   the change you just made). Identify which of the categories above it hits.
2. **Find the matching section** in CLAUDE.md. Edit in place — don't append a
   duplicate or a changelog. There is exactly one home for each fact.
3. **Verify against the code, not memory.** Before writing a claim, confirm
   it: open the file, grep for the symbol, check the npm script exists in
   `package.json`, the option is really in `playwright.config.ts` or
   `typedoc.json`, the invariant is enforced where you say it is. Never
   document something you haven't confirmed exists. This matters most for
   version pins — state the actual range from `package.json`, not a rounded
   description of it.
4. **Match the existing voice.** CLAUDE.md is terse, declarative, and
   high-signal — file paths in backticks, no marketing, no step-by-step
   tutorials. Add the minimum that makes the doc correct again; prefer
   editing a sentence over adding a paragraph. Remove lines the change made
   false.
5. **Re-read the touched section once** to confirm it now matches reality and
   reads consistently with the rest.

## Notes

- CLAUDE.md and README.md serve different audiences: CLAUDE.md is the concise
  agent/onboarding brief; README.md is the full human walkthrough. A change
  can require updating one, the other, or both — judge per change; don't
  mirror content between them.
- When a constraint exists for a reason that isn't visible in the code,
  record the reason, not just the rule. A bare pin reads as staleness and
  invites the next session to "helpfully" undo it.
- If a change makes an existing gotcha obsolete, delete it rather than
  leaving a stale warning.
- Keep the edit in the same commit/PR as the change it documents so the doc
  and the code never drift.
