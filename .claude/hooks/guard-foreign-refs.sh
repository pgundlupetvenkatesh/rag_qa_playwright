#!/usr/bin/env bash
# PreToolUse(Bash) hook — block `git commit` when staged changes mention another
# repository by name. Copying a file in from a sibling project tends to drag its
# name, stack terms, and doc references along; those are meaningless here and go
# stale silently. Reads the tool-call JSON on stdin. Exit 2 blocks the command
# and feeds the message back to Claude; any other path exits 0 (fail-open) so the
# hook never wedges legitimate work.
#
# The list of foreign names is derived at run time from sibling checkouts beside
# this repo — deliberately, so no other project's name is hard-coded here. If a
# reference is legitimate, add the name to .claude/allowed-refs.txt (one per
# line, # for comments).

input="$(cat)"

# Only act on commands that actually run `git commit` (handles `... && git commit`).
printf '%s' "$input" | grep -Eq 'git[[:space:]]+commit' || exit 0

cd "${CLAUDE_PROJECT_DIR:-.}" 2>/dev/null || exit 0

toplevel="$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0
self="$(basename "$toplevel")"
parent="$(dirname "$toplevel")"
[ -d "$parent" ] || exit 0

allowfile="$toplevel/.claude/allowed-refs.txt"

# A name is only worth matching if it reads like a project identifier rather than
# a dictionary word: at least 6 characters, and either separator-cased or camelCase.
looks_like_project_id() {
  [ "${#1}" -ge 6 ] || return 1
  case "$1" in
    *[_-]*) return 0 ;;
  esac
  printf '%s' "$1" | grep -q '[a-z][A-Z]'
}

names=""
for dir in "$parent"/*/; do
  [ -e "${dir}.git" ] || continue                    # git checkouts only
  name="$(basename "$dir")"
  [ "$name" = "$self" ] && continue                  # this repo is not foreign
  looks_like_project_id "$name" || continue
  if [ -f "$allowfile" ] && grep -v '^[[:space:]]*#' "$allowfile" 2>/dev/null \
     | grep -qixF "$name"; then
    continue                                         # explicitly allowed here
  fi
  names="${names}${name}"$'\n'
done

[ -n "$names" ] || exit 0

problems=""
staged="$(git diff --cached --name-only --diff-filter=ACMR 2>/dev/null)"
while IFS= read -r f; do
  [ -z "$f" ] && continue
  case "$f" in
    # Generated or vendored content: not hand-written, so not our prose to fix.
    data/*|*.lock|package-lock.json|*/package-lock.json) continue ;;
  esac
  added="$(git diff --cached -U0 -- "$f" 2>/dev/null | grep '^+' | grep -v '^+++')"
  [ -z "$added" ] && continue
  while IFS= read -r name; do
    [ -z "$name" ] && continue
    # -w so a name never matches inside a longer word; -F so it is a literal.
    if printf '%s' "$added" | grep -qiwF "$name"; then
      problems="${problems}  - ${f}: mentions ${name}"$'\n'
    fi
  done <<EOF
$names
EOF
done <<EOF
$staged
EOF

if [ -n "$problems" ]; then
  {
    echo "Commit blocked — staged changes reference another project:"
    printf '%s' "$problems"
    echo "Restate the point in this project's own terms and drop the reference."
    echo "If the mention is genuinely needed, add the name to .claude/allowed-refs.txt."
  } >&2
  exit 2
fi

exit 0
