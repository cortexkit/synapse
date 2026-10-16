#!/usr/bin/env bash
# Keep landing policy in one testable planner; callers may run this from any directory.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
exec python3 scripts/land-gates.py "$@"
