#!/bin/bash
# Smart App Control on this machine randomly refuses freshly linked build-script
# and proc-macro binaries ("An Application Control policy has blocked this file",
# os error 4551). A relink gets a new PE timestamp/PDB GUID and is usually
# allowed, so clean only the refused package and rerun the same cargo command.
# Usage: build-retry.sh <log-prefix> <cargo args...>
log=$1; shift
for attempt in $(seq 1 25); do
  cargo "$@" > "$log.$attempt.log" 2>&1
  status=$?
  if [ $status -eq 0 ]; then echo "attempt=$attempt exit=0"; exit 0; fi
  if ! grep -q "Application Control policy has blocked" "$log.$attempt.log"; then
    echo "attempt=$attempt exit=$status (not an Application Control refusal)"; tail -n 30 "$log.$attempt.log"; exit $status
  fi
  blocked=$(grep -o "could not execute process \`[^\`]*\`\|failed to run custom build command for \`[^\`]*\`\|error\[E0463\][^\n]*\|could not compile \`[^\`]*\`" "$log.$attempt.log" | head -3)
  echo "attempt=$attempt blocked: $blocked"
  for pkg in $(grep -o "failed to run custom build command for \`[^ ]*" "$log.$attempt.log" | sed 's/.*`//') $(grep -o "could not compile \`[^\`]*\`" "$log.$attempt.log" | sed 's/could not compile `//; s/`//'); do
    cargo clean --release -p "$pkg" >/dev/null 2>&1 || true
  done
  # A refused proc-macro DLL surfaces as a load failure; drop the named DLLs too.
  for dll in $(grep -o "target\\\\release\\\\deps\\\\[A-Za-z0-9_-]*\.dll" "$log.$attempt.log" | sort -u); do
    rm -f "$(echo "$dll" | tr '\\' '/')"
  done
done
echo "gave up after 25 attempts"; exit 1
