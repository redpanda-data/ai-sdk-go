#!/usr/bin/env bash
# Starts the auth proxy and the Unix socket the sandbox reaches it through.
# Only this step sees $VERTEX_ACCESS_TOKEN. Needs VERTEX_DIR, PROXY_PORT and
# VERTEX_TEST_PROJECT.
set -euo pipefail
: "${VERTEX_ACCESS_TOKEN:?no access token}"
here="$(cd "$(dirname "$0")" && pwd -P)"

sudo useradd --system --no-create-home --shell /usr/sbin/nologin credproxy
# Root owns $VERTEX_DIR, so test code can't replace the proxy or the binary.
sudo install -d -m 0755 "$VERTEX_DIR" "$VERTEX_DIR/sock"
sudo install -m 0644 "$here/proxy.py" "$VERTEX_DIR/proxy.py"
sudo install -m 0755 "$RUNNER_TEMP/dl/vertex.test" "$VERTEX_DIR/vertex.test"
sudo install -m 0644 -o credproxy /dev/null /var/log/credproxy.log

# The suite calls only models under locations/global; any other path gets 403.
prefix="/v1/projects/$VERTEX_TEST_PROJECT/locations/global/publishers/"
printf '%s\n' "$VERTEX_ACCESS_TOKEN" | sudo -u credproxy setsid bash -c \
  "exec python3 '$VERTEX_DIR/proxy.py' 127.0.0.1 '$PROXY_PORT' aiplatform.googleapis.com '$prefix' >>/var/log/credproxy.log 2>&1" \
  >/dev/null 2>&1 &
for _ in {1..40}; do ss -ltn | grep -q "127.0.0.1:$PROXY_PORT" && break; sleep 0.5; done
ss -ltn | grep -q "127.0.0.1:$PROXY_PORT" || { echo "::error::the auth proxy did not start"; cat /var/log/credproxy.log; exit 1; }

sudo setsid socat "UNIX-LISTEN:$VERTEX_DIR/sock/p.sock,fork,mode=0666" "TCP:127.0.0.1:$PROXY_PORT" >/dev/null 2>&1 &
for _ in {1..20}; do [[ -S "$VERTEX_DIR/sock/p.sock" ]] && break; sleep 0.5; done
[[ -S "$VERTEX_DIR/sock/p.sock" ]] || { echo "::error::the socket bridge did not start"; exit 1; }

# The auth action keeps the token in its step output file, and in a credentials
# file when asked to.
rm -f "$GITHUB_WORKSPACE"/gha-creds-*.json "$RUNNER_TEMP"/gha-creds-*.json
rm -f "$RUNNER_TEMP"/_runner_file_commands/set_output_* 2>/dev/null || true
