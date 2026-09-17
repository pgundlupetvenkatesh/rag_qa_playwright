---
name: commit-rules
description: >-
  Conventions for creating git commits in this repo — how to scope, stage, and
  word a commit. Use whenever about to run `git commit`, when the user asks to
  "commit", "commit this", or "save changes", or when wording a commit message.
  Covers subject/body style, atomic scoping, what not to commit, and required
  trailers. Generic and project-agnostic; no language- or framework-specific
  rules.
---

# Commit rules

Follow these when creating any commit in this repo.

## When to commit

- Commit only when the user asks. Don't auto-commit after making changes.
- If the current branch is the default branch (`main`/`master`), create a
  topic branch first, then commit.
- Commit only what the user asked to include. Review `git status` and
  `git diff --staged` before committing; don't sweep in unrelated changes,
  generated files, or another task's edits.

## Scope — keep commits atomic

- One logical change per commit. If the diff does two unrelated things, split
  it into two commits.
- Each commit should leave the repo in a working state (tests/build not broken
  by that commit alone).
- Don't mix refactors with behavior changes in the same commit.

## Message format

```
<short imperative subject, ~50 chars, no trailing period>

<body: wrap at ~72 chars. Explain WHY the change was made and any
context a reviewer needs — not a restatement of the diff. Omit the body
for trivial, self-evident changes.>
```

- **Subject:** imperative mood ("add", "fix", "remove" — not "added"/"fixes"),
  concise, lowercase start, no period. If the repo already uses a convention
  (e.g. Conventional Commits `type(scope): …`), match it — check recent
  `git log` first.
- **Body:** focus on intent and rationale. Reference issues/PRs if relevant.
- Match the style of recent commits in the repo over these defaults when they
  differ.

## What not to commit

- Secrets, credentials, tokens, `.env` files, or anything gitignored.
- Build artifacts, caches, logs, or editor/OS junk.
- Commented-out code or debug prints left in by accident.
- Unrelated formatting churn mixed into a functional change.

## Required trailer

End every commit message with a Co-Authored-By trailer naming the Claude
model in use (blank line before it), e.g.:

```
Co-Authored-By: Claude <model name> <noreply@anthropic.com>
```

## Before finalizing

1. `git diff --staged` — confirm only intended changes are staged.
2. Subject reads as a command and stands alone.
3. No secrets or stray artifacts included.
4. Trailer present.
