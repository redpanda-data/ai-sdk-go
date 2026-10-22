#!/usr/bin/env bash
# Fails when a host off the agent job's allowlist is reachable. Harden-Runner
# turns its block off for the whole job, leaving only a line in its step's log,
# when an allowed host doesn't resolve at startup, so the job stops here rather
# than run the agent with the network open.
# Args: the URL to probe (default https://example.com, which isn't allowed).
set -euo pipefail
probe="${1:-https://example.com}"

if curl --silent --output /dev/null --max-time 10 "$probe"; then
  echo "::error::Reached $probe, which isn't on the allowlist, so the egress block is off. Harden-Runner's step log says why (\"Error resolving allowed domain\"); fix the list or re-run."
  exit 1
fi
echo "Egress block on: $probe is unreachable."
