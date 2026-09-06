# Repo-local preflight sourced by scripts/train-push.sh before a train is
# pushed: the same workflow-shape assertions CI runs as its first step, run
# here first because a drifted workflow would fail in seconds on CI anyway and
# failing locally is free.
bash scripts/check-train-preconditions.sh || refuse "train preconditions failed — see scripts/check-train-preconditions.sh"
