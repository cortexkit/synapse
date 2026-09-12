# Repo-local preflight sourced by scripts/train-push.sh before a train is
# pushed: the same workflow-shape assertions CI runs as its first step, run
# here first because a drifted workflow would fail in seconds on CI anyway and
# failing locally is free.
bash scripts/check-train-preconditions.sh || refuse "train preconditions failed — see scripts/check-train-preconditions.sh"

# CI compiles 7 of the 22 workspace members. The Metal engine and the ANE and
# MLX workers need a Mac, and the macos runner left with the M1 box, so this
# preflight is the ONLY automated gate they have — synapse-engine-owned is the
# primary production engine on this platform and would otherwise reach master
# having been compiled nowhere but an author's terminal. Warm cost is about ten
# seconds; a cold build is slower and still cheaper than shipping it unbuilt.
# Delete this block the day a Mac runner rejoins CI, not before.
if [ "$(uname -s)" = "Darwin" ]; then
  mac_crates="-p synapse-engine-owned -p synapse-worker-ane -p synapse-worker-mlx"
  # The Metal toolchain lives in full Xcode; Command Line Tools alone cannot
  # compile the shaders, and the failure is a confusing linker error rather
  # than a missing-tool message.
  if [ -d /Applications/Xcode.app ]; then
    export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
  fi
  # shellcheck disable=SC2086
  cargo clippy $mac_crates --all-targets -- -D warnings \
    || refuse "macOS-only crates failed clippy (CI cannot run these; see scripts/train-push.local.sh)"
  cargo test -p synapse-engine-owned --lib \
    || refuse "synapse-engine-owned lib tests failed (CI cannot run these)"
fi
