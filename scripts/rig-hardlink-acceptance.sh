#!/usr/bin/env bash
# Two-uid acceptance: handing a campaign workspace to the candidate must never
# change the owner of anything outside that workspace.
#
# A local `git clone` hard-links the source repository's object files into the
# clone. A hard link is the same inode, and `chown -R` changes the inode, so
# chowning a hard-linked clone to the candidate also hands the candidate the
# SOURCE repository's objects. As owner, the candidate can then make those
# files writable and rewrite them, and git does not re-hash loose objects on an
# ordinary read. This happened on the M5 rig: 978 of 997 object files in the
# operator's synapse checkout ended up owned by the candidate account.
#
# The script runs the handover the driver performs, using the rig's own
# permitted chown, against a scratch source repository, in two ways:
#   hazard arm   plain `git clone`             -> the source MUST end up leaked
#   safe arm     `git clone --no-hardlinks`    -> the source MUST stay clean
# The hazard arm is what gives the safe arm meaning: if the hazard stopped
# reproducing, a clean safe arm would prove nothing about the clone flag.
#
# It also sweeps the operator's project checkouts for any file the candidate
# owns outside the rig, which is the invariant itself and does not depend on
# how a driver clones.
#
# Usage: scripts/rig-hardlink-acceptance.sh   (needs the rig's NOPASSWD sudoers)
set -euo pipefail

CANDIDATE="${CANDIDATE:-ck-candidate}"
RIG="$HOME/ck-campaign/workspaces"
PROJECTS="${PROJECTS:-$HOME/Work/Projects/CortexKit}"
SRC="$(mktemp -d "$HOME/ck-campaign/scratch/hardlink-src.XXXXXX")"
HAZARD="$RIG/hardlink-hazard-$$"
SAFE="$RIG/hardlink-safe-$$"

as_candidate() { sudo -n -u "$CANDIDATE" "$@"; }

cleanup() {
  for root in "$HAZARD" "$SAFE"; do
    [ -d "$root" ] || continue
    as_candidate chmod -R u+rwx "$root/workspace" "$root/candidate-home" "$root/candidate-tmp" 2>/dev/null || true
    as_candidate find "$root/workspace" "$root/candidate-home" "$root/candidate-tmp" -mindepth 1 -delete 2>/dev/null || true
    rm -rf "$root" 2>/dev/null || true
  done
  # The hazard arm leaves the scratch source's objects owned by the candidate.
  # Replace them with controller-owned copies so the controller can delete it.
  if [ -d "$SRC" ]; then
    find "$SRC" -type f -user "$CANDIDATE" 2>/dev/null | while IFS= read -r f; do
      cp -p "$f" "$f.reown" && mv -f "$f.reown" "$f"
    done
    rm -rf "$SRC"
  fi
  for p in "$HAZARD" "$SAFE" "$SRC"; do [ -e "$p" ] && echo "LEFTOVER: $p" >&2; done
  return 0
}
trap cleanup EXIT

# A small source repository owned by the controller, with loose objects.
git -C "$SRC" init -q
printf 'protected\n' > "$SRC/file.txt"
git -C "$SRC" add file.txt
git -C "$SRC" -c user.name=acceptance -c user.email=acceptance@invalid commit -qm seed

handover() { # handover <root> [clone flags...]
  local root="$1"; shift
  mkdir -p "$root/candidate-home" "$root/candidate-tmp"
  chmod 755 "$root"
  git clone -q "$@" "$SRC" "$root/workspace"
  sudo -n /usr/sbin/chown -R -- "$CANDIDATE" "$root/workspace" "$root/candidate-home" "$root/candidate-tmp"
}

leaked() { find "$SRC/.git/objects" -type f -user "$CANDIDATE" | wc -l | tr -d ' '; }

fail=0
echo "== hazard arm: plain git clone, then the candidate handover"
handover "$HAZARD"
n="$(leaked)"
if [ "$n" -gt 0 ]; then
  echo "ok    source objects now candidate-owned: $n (the hazard reproduces)"
else
  echo "FAIL  hazard did not reproduce, so the safe arm below proves nothing"; fail=1
fi

# Restore the source before the safe arm so its result is not inherited.
find "$SRC" -type f -user "$CANDIDATE" | while IFS= read -r f; do cp -p "$f" "$f.reown" && mv -f "$f.reown" "$f"; done
echo "   source restored: $(leaked) candidate-owned"

echo "== safe arm: git clone --no-hardlinks, then the same handover"
handover "$SAFE" --no-hardlinks
n="$(leaked)"
if [ "$n" -eq 0 ]; then
  echo "ok    source objects candidate-owned: 0"
else
  echo "FAIL  --no-hardlinks still leaked $n source objects"; fail=1
fi

echo "== invariant: files the candidate owns under $PROJECTS (rig excluded)"
n="$(find "$PROJECTS" -user "$CANDIDATE" -not -path "$HOME/ck-campaign/*" 2>/dev/null | wc -l | tr -d ' ')"
if [ "$n" -eq 0 ]; then
  echo "ok    0"
else
  echo "FAIL  $n files owned by $CANDIDATE, e.g.:"
  find "$PROJECTS" -user "$CANDIDATE" -not -path "$HOME/ck-campaign/*" 2>/dev/null | head -3 | sed 's/^/        /'
  fail=1
fi

if [ "$fail" -ne 0 ]; then echo "HARDLINK ACCEPTANCE: FAILED"; exit 1; fi
echo "HARDLINK ACCEPTANCE: PASSED"
