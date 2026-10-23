#!/usr/bin/env bash
# Fails unless Harden-Runner's egress block is on before Claude starts.
# Harden-Runner's agent writes "Initialized" to its status file once its DNS
# proxy and firewall are up, which can be 11 seconds after its step ends: the
# step waits about 9 seconds, then passes anyway. When an allowed host doesn't
# resolve, the agent turns the block off instead and never writes it. So wait
# for the status, then probe a host off the allowlist.
# Args: the status file (default /home/agent/agent.status) and the URL to probe
# (default https://example.com, which isn't allowed).
set -euo pipefail
status="${1:-/home/agent/agent.status}"
probe="${2:-https://example.com}"

for _ in {1..30}; do
  grep -qs Initialized "$status" && break
  sleep 2
done
if ! grep -qs Initialized "$status"; then
  echo "::error::Harden-Runner didn't report its egress block on within 60 seconds. Its step log says why (\"Error resolving allowed domain\" means a listed host didn't resolve); fix the list or re-run."
  exit 1
fi
if curl --silent --output /dev/null --max-time 10 "$probe"; then
  echo "::error::Reached $probe, which isn't on the allowlist, so the egress block is off."
  exit 1
fi
echo "Egress block on: Harden-Runner reports it initialized, and $probe is unreachable."
