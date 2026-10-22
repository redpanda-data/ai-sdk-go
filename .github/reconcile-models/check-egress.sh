#!/usr/bin/env bash
# Fails when a host off the agent job's allowlist is reachable. Harden-Runner
# turns its block off for the whole job, leaving only a "Reverting agent"
# annotation, when an allowed host doesn't resolve at startup, so the job
# stops here rather than run the agent with the network open.
# Args: the URL to probe (default https://example.com, which isn't allowed).
set -euo pipefail
probe="${1:-https://example.com}"

if curl --silent --output /dev/null --max-time 10 "$probe"; then
  echo "::error::Reached $probe, which isn't on the allowlist, so the egress block is off. Look for Harden-Runner's \"Reverting agent\" annotation, then re-run."
  exit 1
fi
echo "Egress block on: $probe is unreachable."
