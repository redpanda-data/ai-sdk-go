#!/usr/bin/env bash
# Builds the Go test server, starts it, runs JS conformance tests, cleans up.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Build Go test server
echo "==> Building Go test server..."
SERVER_BIN="$(mktemp)"
ADDR_FILE="$(mktemp)"
trap 'rm -f "$SERVER_BIN" "$ADDR_FILE"; [ -n "${SERVER_PID:-}" ] && kill "$SERVER_PID" 2>/dev/null || true' EXIT

go build -o "$SERVER_BIN" ./testserver/

# Start the server, capturing stdout for the LISTEN_ADDR line
echo "==> Starting test server..."
"$SERVER_BIN" -port 0 > "$ADDR_FILE" 2>&1 &
SERVER_PID=$!

# Wait for LISTEN_ADDR to appear (up to 5s)
for _ in $(seq 1 50); do
    if grep -q "LISTEN_ADDR=" "$ADDR_FILE" 2>/dev/null; then break; fi
    sleep 0.1
done

LISTEN_ADDR="$(grep "LISTEN_ADDR=" "$ADDR_FILE" | head -1 | sed 's/LISTEN_ADDR=//')"
if [ -z "$LISTEN_ADDR" ]; then
    echo "ERROR: failed to get server address"
    cat "$ADDR_FILE"
    exit 1
fi

PORT="$(echo "$LISTEN_ADDR" | cut -d: -f2)"
echo "==> Server listening on $LISTEN_ADDR"

# Wait for health check
for _ in $(seq 1 50); do
    if curl -sf "http://$LISTEN_ADDR/health" > /dev/null 2>&1; then break; fi
    sleep 0.1
done

# Run JS tests
echo "==> Running conformance tests..."
TEST_SERVER_PORT="$PORT" node --test conformance.test.mjs
STATUS=$?

echo "==> Done (exit $STATUS)"
exit $STATUS
