# shellcheck shell=bash
# The sandbox the conformance test binary runs in: its one way out is the Unix
# socket to the auth proxy. Needs VERTEX_DIR (root-owned: the binary and the
# socket), PROXY_PORT and VERTEX_TEST_PROJECT.

# The bash -c scripts are single-quoted on purpose: they expand inside.
# shellcheck disable=SC2016
sandbox() {
  bwrap --die-with-parent --new-session \
    --unshare-net --unshare-pid --unshare-ipc --unshare-uts --unshare-user --uid 1000 --gid 1000 \
    --ro-bind / / --tmpfs /run --dev /dev --proc /proc --tmpfs /tmp --tmpfs "$RUNNER_TEMP" \
    --ro-bind "$VERTEX_DIR" "$VERTEX_DIR" \
    --clearenv --setenv PATH /usr/bin:/bin --setenv HOME /tmp --setenv TMPDIR /tmp \
    --setenv VERTEX_BASE_URL "http://127.0.0.1:$PROXY_PORT" \
    --setenv VERTEX_TEST_PROJECT "$VERTEX_TEST_PROJECT" \
    --setenv VERTEX_DIR "$VERTEX_DIR" --setenv GITHUB_WORKSPACE "$GITHUB_WORKSPACE" --setenv RUNNER_TEMP "$RUNNER_TEMP" \
    --chdir "$GITHUB_WORKSPACE/providers/vertex/conformance" \
    bash -c '
      socat "TCP-LISTEN:$0,bind=127.0.0.1,fork,reuseaddr" "UNIX-CONNECT:$1" &
      for _ in {1..50}; do (exec 3<>"/dev/tcp/127.0.0.1/$0") 2>/dev/null && break; sleep 0.1; done
      exec "${@:2}"' "$PROXY_PORT" "$VERTEX_DIR/sock/p.sock" "$@"
}

# sandbox_check fails unless the isolation holds, and returns the failed probe's
# exit code below, or the sandbox's own when it cannot start. The proxy answers
# a path outside its prefix with 403, which proves the bridge works without
# spending the token.
# shellcheck disable=SC2016
sandbox_check() {
  local rc=0
  echo planted >"$RUNNER_TEMP/probe-secret"
  PROBE_SECRET=planted sandbox bash -c '
    [[ -z "${PROBE_SECRET:-}" && -z "${ACTIONS_ID_TOKEN_REQUEST_URL:-}" ]] || exit 11
    [[ -z "$(ls -A "$RUNNER_TEMP" 2>/dev/null)" ]] || exit 12
    if curl -sS --max-time 5 -o /dev/null https://example.com 2>/dev/null; then exit 13; fi
    # 7 is a refused connection; anything else means something answers.
    curl -sS --max-time 5 -o /dev/null --unix-socket /var/run/docker.sock http://localhost/_ping 2>/dev/null
    [[ $? -eq 7 ]] || exit 15
    if sudo -n true 2>/dev/null; then exit 16; fi
    [[ "$(curl -sS --max-time 10 -o /dev/null -w "%{http_code}" "$VERTEX_BASE_URL/probe")" == 403 ]] || exit 17
    for dir in "$GITHUB_WORKSPACE" "$VERTEX_DIR"; do
      if touch "$dir/.probe" 2>/dev/null; then exit 18; fi
    done' _ || rc=$?
  rm -f "$RUNNER_TEMP/probe-secret" "$GITHUB_WORKSPACE/.probe" "$VERTEX_DIR/.probe"
  return "$rc"
}
