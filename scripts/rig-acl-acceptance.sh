#!/usr/bin/env bash
# Two-uid acceptance for the campaign rig's controller read grant.
#
# The driver lets the controller read a workspace it hands to the candidate by
# writing an inherited, read-only ACL entry on the candidate directories BEFORE
# the chown. A single-uid test cannot show that the entry is what grants the
# read, because the test user owns the files. This script uses the rig's real
# candidate account so it can.
#
# The grant is revocable by the party it constrains: after the chown the
# candidate owns the directories and can strip the entry. The design is only
# sound while "unreadable" means "refuse", so the negative arms assert that the
# harness refuses AND names EACCES with the path. The positive arm runs the same
# verification without a strip, so a verifier that refuses everything cannot
# pass the negative arms by accident.
#
# Protected files are made mode 600 before the strip. A world-readable file is
# readable through its other bits once the directory can be searched, so
# stripping its entry would prove nothing about the entry.
#
# Usage: scripts/rig-acl-acceptance.sh   (needs the rig's NOPASSWD sudoers)
set -euo pipefail

CANDIDATE="${CANDIDATE:-ck-candidate}"
CONTROLLER="$(id -un)"
ROOT="$HOME/ck-campaign/workspaces/acl-acceptance-$$"
HARNESS="$(cd "$(dirname "$0")/.." && pwd)/bench/campaign/ane-direct-embed-harness.sh"
GRANT="$CONTROLLER allow read,readattr,readextattr,readsecurity,list,search,file_inherit,directory_inherit"

as_candidate() { sudo -n -u "$CANDIDATE" "$@"; }

cleanup() {
  # The candidate owns the three trees, so it clears their contents; the
  # controller owns the parent and removes the now-empty entries.
  as_candidate chmod -R u+rwx "$ROOT/workspace" "$ROOT/candidate-home" "$ROOT/candidate-tmp" 2>/dev/null || true
  as_candidate find "$ROOT/workspace" "$ROOT/candidate-home" "$ROOT/candidate-tmp" -mindepth 1 -delete 2>/dev/null || true
  rm -rf "$ROOT" 2>/dev/null || true
  if [ -e "$ROOT" ]; then echo "LEFTOVER: $ROOT" >&2; fi
}
trap cleanup EXIT

# The harness body is Python in a bash heredoc; load its definitions only.
PY_BODY="$(mktemp)"
awk "/<<'PY'\$/{f=1;next} /^PY\$/{f=0} f" "$HARNESS" > "$PY_BODY"

mkdir -p "$ROOT/workspace/probe" "$ROOT/candidate-home" "$ROOT/candidate-tmp"
chmod 755 "$ROOT"
printf 'pre-grant\n' > "$ROOT/workspace/probe/rows.jsonl"

# 1. Grant while the controller still owns the trees, exactly as the driver will.
chmod -R +a "$GRANT" "$ROOT/workspace" "$ROOT/candidate-home" "$ROOT/candidate-tmp"
# 2. Hand over with the rig's permitted three-path chown.
sudo -n /usr/sbin/chown -R -- "$CANDIDATE" "$ROOT/workspace" "$ROOT/candidate-home" "$ROOT/candidate-tmp"
as_candidate chmod 700 "$ROOT/workspace" "$ROOT/workspace/probe"
# 3. A file the candidate creates AFTER the grant, which must inherit the entry.
as_candidate sh -c "umask 077; printf 'post-grant\n' > '$ROOT/workspace/probe/created.txt'"
as_candidate chmod 600 "$ROOT/workspace/probe/rows.jsonl"

verify() {
  /usr/bin/python3 - "$PY_BODY" "$1" <<'CHECK'
import sys
src = open(sys.argv[1]).read()
cut = src.find("\nif __name__")
ns = {"__name__": "acl_acceptance"}
exec(compile(src if cut < 0 else src[:cut], "harness", "exec"), ns)
path = __import__("pathlib").Path(sys.argv[2])
try:
    ns["verify_regular_file"](path, "protected file")
    ns["sha256_file"](path)
except Exception as error:
    print("REFUSED: %s" % error)
    sys.exit(1)
print("READ")
CHECK
}

fail=0
expect() { # expect <READ|REFUSED> <label> <path>
  local want="$1" label="$2" path="$3" out
  out="$(verify "$path" 2>&1)" || true
  local got="READ"; case "$out" in REFUSED:*) got="REFUSED";; esac
  if [ "$got" != "$want" ]; then
    echo "FAIL  $label: expected $want, got: $out"; fail=1; return
  fi
  if [ "$want" = "REFUSED" ]; then
    case "$out" in
      *"Errno 13"*"$path"*|*"$path"*"Errno 13"*) ;;
      *) echo "FAIL  $label: refused, but not naming EACCES and the path: $out"; fail=1; return;;
    esac
  fi
  echo "ok    $label -> $got"
}

echo "== entry on a post-grant file (inheritance), as the candidate sees it"
as_candidate ls -le "$ROOT/workspace/probe/created.txt" | sed 's/^/   /'

echo "== positive arms: no strip, private files, controller must read"
expect READ "pre-grant file, mode 600"   "$ROOT/workspace/probe/rows.jsonl"
expect READ "post-grant file, mode 600"  "$ROOT/workspace/probe/created.txt"

echo "== negative arm: candidate strips the entry off one private file"
as_candidate chmod -N "$ROOT/workspace/probe/rows.jsonl"
expect REFUSED "stripped file, mode 600" "$ROOT/workspace/probe/rows.jsonl"
expect READ    "untouched sibling still reads" "$ROOT/workspace/probe/created.txt"

echo "== negative arm: candidate strips the entry off the containing directory"
as_candidate chmod -N "$ROOT/workspace/probe"
expect REFUSED "file under a stripped dir" "$ROOT/workspace/probe/created.txt"

rm -f "$PY_BODY"
if [ "$fail" -ne 0 ]; then echo "ACL ACCEPTANCE: FAILED"; exit 1; fi
echo "ACL ACCEPTANCE: PASSED"
