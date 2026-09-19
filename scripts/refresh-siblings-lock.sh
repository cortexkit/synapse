#!/usr/bin/env bash
# Rewrites siblings.lock from the sibling checkouts' current HEADs and
# refreshes Cargo.lock against them, so both pins move in one commit.
set -euo pipefail
cd "$(dirname "$0")/.."
subc=$(git -C ../subconscious rev-parse HEAD)
commons=$(git -C ../commons rev-parse HEAD)
sed -i.bak -e "s/^subconscious=.*/subconscious=$subc/" -e "s/^commons=.*/commons=$commons/" siblings.lock
rm -f siblings.lock.bak

cargo update -w

# `cargo update -w` re-resolves workspace path deps, but a sibling's OWN
# git-pinned transitive dep does not necessarily follow its manifest in the same
# pass. A plain metadata run is a minimal re-resolve that lets those edges
# settle before anything asserts on the result.
cargo metadata --format-version 1 >/dev/null

# The assertion that makes this script a gate rather than a hope: CI builds with
# --locked, so a refreshed lock that cannot satisfy --locked is a red train
# discovered minutes later instead of here. Checked with metadata rather than a
# build because it resolves the same graph in seconds.
if ! cargo metadata --format-version 1 --locked >/dev/null 2>/tmp/refresh-siblings-locked.err; then
  echo "refresh-siblings-lock: the refreshed Cargo.lock does NOT satisfy --locked." >&2
  echo "refresh-siblings-lock: CI builds locked, so landing this would go red. Cargo said:" >&2
  sed 's/^/    /' /tmp/refresh-siblings-locked.err >&2
  exit 1
fi
rm -f /tmp/refresh-siblings-locked.err

echo "siblings.lock: subconscious=$subc commons=$commons"
echo "refresh-siblings-lock: Cargo.lock satisfies --locked"
