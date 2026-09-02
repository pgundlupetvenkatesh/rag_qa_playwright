#!/usr/bin/env bash
# Stop hook — nudge to reconcile CLAUDE.md when src/ or scripts/ changed this
# session but CLAUDE.md was not touched. Reads the Stop JSON on stdin.
# Exit 2 blocks the stop and feeds the message back to Claude; exit 0 lets it
# stop. Loop-safe (honors stop_hook_active) and fail-open.

input="$(cat)"

# Avoid loops: if this Stop was already triggered by a blocking hook, do nothing.
printf '%s' "$input" | grep -Eq '"stop_hook_active"[[:space:]]*:[[:space:]]*true' && exit 0

cd "${CLAUDE_PROJECT_DIR:-.}" 2>/dev/null || exit 0

# Tracked changes vs HEAD (staged + unstaged) plus untracked files, by path.
changed="$( { git diff --name-only HEAD 2>/dev/null; git ls-files --others --exclude-standard 2>/dev/null; } )"

printf '%s\n' "$changed" | grep -Eq '^(src|scripts)/' || exit 0   # no code change → nothing to do
printf '%s\n' "$changed" | grep -Fxq 'CLAUDE.md' && exit 0        # CLAUDE.md already touched → assume handled

{
  echo "src/ or scripts/ changed this session but CLAUDE.md is untouched."
  echo "Run the update-claude-md skill to check whether the Commands / Architecture / Conventions / Gotchas sections need reconciling (it may legitimately decide no update is needed)."
} >&2
exit 2
