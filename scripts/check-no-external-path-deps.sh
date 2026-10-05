#!/usr/bin/env bash
# Refuse any Cargo path dependency that resolves outside the repository.
#
# Fleet rule: crates from other CortexKit repos (subconscious, commons) come
# from crates.io, or from a git dependency pinned to a full commit when they
# aren't published, never from a path such as ../subconscious. A path into a
# sibling checkout makes Cargo.lock record whatever version that checkout
# happens to hold, so every release there breaks `--locked` builds here.
#
# This reads what Cargo actually resolved (`cargo metadata`), not manifest
# text, so it also catches `[patch]` entries and paths reached through another
# path dependency. A package resolved from a path has no `source`; each one
# must have its manifest inside the repository.
#
# Usage:
#   scripts/check-no-external-path-deps.sh                check the workspaces below
#   scripts/check-no-external-path-deps.sh --self-test    prove the check can fail
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd -P)"

# Every Cargo workspace CI builds. bench/parity is a separate workspace.
workspaces=("Cargo.toml" "bench/parity/Cargo.toml")

# Prints each path package outside $2, given the manifest at $1. Exits 3 if
# there are any, 0 if none, and any other code if the check itself broke.
# The "found some" code is deliberately not 1: a crashed Python script also
# exits 1, and a self-test that accepted 1 would pass on a broken check.
check_manifest() {
  local manifest="$1" root="$2" metadata errors
  # Only stdout is JSON. Cargo writes progress and warnings ("Blocking waiting
  # for file lock", "Downloaded ...") to stderr, which must stay out of it.
  errors="$(mktemp)"
  if ! metadata="$(cargo metadata --format-version 1 --locked --manifest-path "$manifest" 2>"$errors")"; then
    printf 'cargo metadata failed for %s:\n' "$manifest" >&2
    cat "$errors" >&2
    rm -f "$errors"
    return 2
  fi
  rm -f "$errors"
  # Plain string concatenation keeps this valid on the Python 3.9 that ships
  # with Xcode, as well as on CI's newer Python.
  python3 -c '
import json, os, sys
root = os.path.realpath(sys.argv[1])
outside = []
checked = 0
for package in json.loads(sys.stdin.read())["packages"]:
    if package["source"] is not None:
        continue
    checked += 1
    manifest = os.path.realpath(package["manifest_path"])
    if os.path.commonpath([root, manifest]) != root:
        outside.append(package["name"] + " " + package["version"] + " -> " + manifest)
for line in outside:
    print(line)
# The workspace members are themselves path packages, so a run that saw none
# read the wrong thing and must not pass.
if checked == 0:
    print("no path packages seen; the check read nothing", file=sys.stderr)
    sys.exit(4)
print("checked " + str(checked) + " path packages", file=sys.stderr)
sys.exit(3 if outside else 0)
' "$root" <<<"$metadata"
}

# Refuses any [patch] or [replace] table in the given manifests. Exits 3 if
# one is found. cargo metadata only lists packages a patch actually replaced,
# so an unused patch pointing at a sibling checkout would pass the check above
# and start applying the day a matching dependency is added. Synapse has no
# need for either table, so the rule is simply: none.
check_no_patch_tables() {
  local found
  found="$(grep -nE '^[[:space:]]*\[(patch(\.|\])|replace\])' "$@" || true)"
  if [ -n "$found" ]; then
    printf '%s\n' "$found"
    return 3
  fi
}

if [ "${1:-}" = "--self-test" ]; then
  scratch="$(mktemp -d)"
  trap 'rm -rf "$scratch"' EXIT
  make_crate() { # dir name [dependency line]
    mkdir -p "$1/src"
    printf '[package]\nname = "%s"\nversion = "0.1.0"\nedition = "2021"\n\n[dependencies]\n%s\n' \
      "$2" "${3:-}" >"$1/Cargo.toml"
    : >"$1/src/lib.rs"
  }
  # A sibling checkout outside the "repository" under test.
  make_crate "$scratch/sibling" sibling
  # Control: a path dependency that stays inside the repository passes.
  make_crate "$scratch/repo/inner" inner
  make_crate "$scratch/repo/app" app 'inner = { path = "../inner" }'
  (cd "$scratch/repo/app" && cargo generate-lockfile -q)
  check_manifest "$scratch/repo/app/Cargo.toml" "$scratch/repo" >/dev/null ||
    { echo "self-test FAILED: an in-repo path dependency was refused" >&2; exit 1; }
  # Planted break 1: a direct path dependency into the sibling.
  make_crate "$scratch/repo/app" app 'sibling = { path = "../../sibling" }'
  (cd "$scratch/repo/app" && cargo generate-lockfile -q)
  rc=0; check_manifest "$scratch/repo/app/Cargo.toml" "$scratch/repo" >/dev/null || rc=$?
  [ "$rc" = 3 ] || { echo "self-test FAILED: a path dependency outside the repo was not refused as one (rc=$rc)" >&2; exit 1; }
  # Planted break 2: a [patch] redirecting a registry crate to the sibling.
  make_crate "$scratch/repo/app" app 'sibling = "0.1"'
  printf '\n[patch.crates-io]\nsibling = { path = "../../sibling" }\n' >>"$scratch/repo/app/Cargo.toml"
  (cd "$scratch/repo/app" && cargo generate-lockfile -q --offline)
  rc=0; check_manifest "$scratch/repo/app/Cargo.toml" "$scratch/repo" >/dev/null || rc=$?
  [ "$rc" = 3 ] || { echo "self-test FAILED: a [patch] path outside the repo was not refused as one (rc=$rc)" >&2; exit 1; }
  # Planted break 3: a [patch] no dependency uses, invisible to cargo metadata.
  make_crate "$scratch/repo/app" app
  printf '\n[patch.crates-io]\nsibling = { path = "../../sibling" }\n' >>"$scratch/repo/app/Cargo.toml"
  check_no_patch_tables "$scratch/repo/inner/Cargo.toml" >/dev/null ||
    { echo "self-test FAILED: a manifest without [patch] was refused" >&2; exit 1; }
  rc=0; check_no_patch_tables "$scratch/repo/app/Cargo.toml" >/dev/null || rc=$?
  [ "$rc" = 3 ] || { echo "self-test FAILED: an unused [patch] table was not refused (rc=$rc)" >&2; exit 1; }
  echo "path-dependency self-test: in-repo controls passed, 3 planted breaks refused"
  exit 0
fi

status=0
for workspace in "${workspaces[@]}"; do
  rc=0
  found="$(check_manifest "$repo_root/$workspace" "$repo_root")" || rc=$?
  if [ "$rc" = 3 ]; then
    printf 'refused: %s resolves path dependencies outside the repository:\n%s\n' "$workspace" "$found" >&2
    status=1
  elif [ "$rc" != 0 ]; then
    printf 'refused: the path-dependency check itself failed for %s (exit %s)\n' "$workspace" "$rc" >&2
    status=1
  fi
done
manifests=()
while IFS= read -r manifest; do
  manifests+=("$repo_root/$manifest")
done < <(git -C "$repo_root" ls-files -- 'Cargo.toml' '**/Cargo.toml')
if [ "${#manifests[@]}" = 0 ]; then
  echo "refused: found no tracked Cargo.toml to scan for [patch] tables" >&2
  status=1
else
  rc=0
  found="$(check_no_patch_tables "${manifests[@]}")" || rc=$?
  if [ "$rc" = 3 ]; then
    printf 'refused: [patch] or [replace] tables are not allowed:\n%s\n' "$found" >&2
    status=1
  elif [ "$rc" != 0 ]; then
    printf 'refused: the [patch] scan itself failed (exit %s)\n' "$rc" >&2
    status=1
  fi
fi
[ "$status" = 0 ] && echo "no path dependency resolves outside the repository (${#workspaces[@]} workspaces, ${#manifests[@]} manifests scanned for [patch])"
exit "$status"
