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
echo "siblings.lock: subconscious=$subc commons=$commons"
