#!/usr/bin/env bash
# Nothing under .cortexkit/ may be tracked except the allowed folders and files.
#
# The fleet layout keeps internal material (design notes, investigations, tool state)
# under .cortexkit/, ignored by git, and tracks only release-notes/, skills/, rules/,
# WORKER.md and top-level project settings files (.cortexkit/<module>.jsonc, which the
# daemon reads as project settings) there. The check lists every tracked path under
# .cortexkit/ and refuses any outside that list. Listing only tracked files that are also
# ignored (`git ls-files -ci`)
# is not enough: a stray `!` rule in .gitignore can un-ignore a private file, and then
# it is tracked without being ignored.
#
# The ignore block is still checked first, so an internal file is never offered for
# commit: an internal path must be ignored and an allowed one must not be. If either
# control fails, or git cannot list the tracked files, the checker reports itself BLIND
# and exits 2 rather than passing.
#
# Exit 0 clean, 1 a disallowed file is tracked, 2 the checker cannot see.

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2

if ! git check-ignore -q --no-index .cortexkit/internal-docs/designs/control.md; then
  echo "BLIND: .cortexkit/internal-docs/ is not ignored; the .cortexkit block is missing from .gitignore" >&2
  exit 2
fi
if git check-ignore -q --no-index .cortexkit/release-notes/control.md; then
  echo "BLIND: .cortexkit/release-notes/ is ignored; the block's allow lines are missing" >&2
  exit 2
fi

if ! tracked="$(git ls-files -- .cortexkit)"; then
  echo "BLIND: git ls-files failed, so the tracked paths under .cortexkit/ are unknown" >&2
  exit 2
fi
disallowed=""
while IFS= read -r path; do
  case "$path" in
    ""|.cortexkit/release-notes/*|.cortexkit/skills/*|.cortexkit/rules/*|.cortexkit/WORKER.md) ;;
    *)
      # A project settings file sits directly under .cortexkit/ and is named after
      # its module, the same pattern the fleet's allowlist uses.
      [[ "$path" =~ ^\.cortexkit/[a-z0-9-]+\.jsonc$ ]] || disallowed="$disallowed$path"$'\n'
      ;;
  esac
done <<<"$tracked"
if [ -n "$disallowed" ]; then
  echo "tracked files under .cortexkit/ that the layout keeps out of git:" >&2
  printf '%s' "$disallowed" | sed 's/^/  /' >&2
  echo "move them to .cortexkit/internal-docs/ (or remove them) and 'git rm --cached' them" >&2
  exit 1
fi
echo "controls hold; no disallowed file is tracked under .cortexkit/"
