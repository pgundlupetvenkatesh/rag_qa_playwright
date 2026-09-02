#!/usr/bin/env bash
# PreToolUse(Bash) hook — block `git commit` when secrets appear to be staged.
# Reads the tool-call JSON on stdin. Exit 2 blocks the command and feeds the
# message back to Claude; any other path exits 0 (fail-open) so the hook never
# wedges legitimate work.

input="$(cat)"

# Only act on commands that actually run `git commit` (handles `... && git commit`).
printf '%s' "$input" | grep -Eq 'git[[:space:]]+commit' || exit 0

cd "${CLAUDE_PROJECT_DIR:-.}" 2>/dev/null || exit 0

problems=""

# 1) Staged .env files (any path component) — high signal, near-zero false positives.
staged="$(git diff --cached --name-only 2>/dev/null)"
while IFS= read -r f; do
  [ -z "$f" ] && continue
  case "$f" in
    # .env.example, if one is ever added, is the committed secret-free template;
    # its contents are still covered by the credential-signature scan below.
    .env.example|*/.env.example) ;;
    .env|.env.*|*/.env|*/.env.*)
      problems="${problems}  - staged env file: ${f}"$'\n' ;;
  esac
done <<EOF
$staged
EOF

# 2) High-confidence credential signatures in the staged additions. Deliberately
#    narrow (real token shapes, not generic key=value) so placeholders like the
#    README's TMDB_API_KEY=your_api_key_here don't trip it.
if git diff --cached -U0 2>/dev/null | grep -E '^\+' | grep -Eq \
   '(-----BEGIN [A-Z ]*PRIVATE KEY-----|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})'; then
  problems="${problems}  - credential-like token in staged diff"$'\n'
fi

if [ -n "$problems" ]; then
  {
    echo "Commit blocked — potential secrets staged:"
    printf '%s' "$problems"
    echo "Unstage them (git restore --staged <file>) or remove the secret, then retry."
  } >&2
  exit 2
fi

exit 0