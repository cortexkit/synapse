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
# The sibling repositories (subconscious, commons) are going public alongside
# this one, so references to them are not private markers.
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
)

failures=0

check_pattern() {
    local matcher="$1"
    local pattern="$2"

    if git grep -nI "$matcher" -- "$pattern" -- ':!scripts/public-banlist-check.sh'; then
        failures=1
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

if (( failures )); then
    echo "public banlist check failed: remove the reported tracked-file markers" >&2
    exit 1
fi

echo "public banlist check passed: no tracked-file markers found"
