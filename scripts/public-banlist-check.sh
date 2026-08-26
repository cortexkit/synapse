#!/usr/bin/env bash
# Check only tracked working-tree content for release-blocking private markers.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# Build patterns from fragments so the verifier does not match its own source.
dot='[.]'
users='/Users'
test_path='/test'
mac_prefix='tests'
mac_suffix='-MacBook-Pro'
private_root='cortex'
private_root+=$'kit/'

patterns=(
    "192${dot}168${dot}"
    "${mac_prefix}${mac_suffix}"
    "asu""sallyko"
    "m1""bench"
    "ufu""kaltinok"
    "${users}${test_path}"
    "ufu""k2"
    "[[:alnum:]_.-]*${dot}vast${dot}ai"
    "tensor""dock"
    "ses_"'[0-9a-f]{12}'
    "ckapp"'-'
    "rm_""toolu"
    "ct_"'00000000'
    "bg_"'[0-9a-f]{8}'
    "${private_root}sub""conscious"
    "${private_root}com""mons"
)

failures=0
waived_file='.github/workflows/ci.yml'
waived_hits=0

check_pattern() {
    local matcher="$1"
    local pattern="$2"
    local waived_output

    if git grep -nI "$matcher" -- "$pattern" -- ':!scripts/public-banlist-check.sh' ":!${waived_file}"; then
        failures=1
    fi

    waived_output="$(git grep -nI "$matcher" -- "$pattern" -- "$waived_file" || true)"
    if [[ -n "$waived_output" ]]; then
        waived_hits=$((waived_hits + $(printf '%s\n' "$waived_output" | wc -l | tr -d ' ')))
    fi
}

for pattern in "${patterns[@]}"; do
    check_pattern -E "$pattern"
done

if [[ -f .banlist.local ]]; then
    while IFS= read -r literal || [[ -n "$literal" ]]; do
        [[ -z "$literal" || "$literal" == \#* ]] && continue
        check_pattern -F "$literal"
    done < .banlist.local
fi

echo "WAIVED: ${waived_file}: ${waived_hits} matching marker(s)"
echo "reason: workflow replacement is a tracked pre-flip release blocker, out of scope for tip cleanup"

if (( failures )); then
    echo "public banlist check failed: remove the reported tracked-file markers" >&2
    exit 1
fi

if (( waived_hits )); then
    echo "public banlist check passed: all remaining markers are in the documented waiver"
else
    echo "public banlist check passed: no tracked-file markers found"
fi
