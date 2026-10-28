# shellcheck shell=bash
# The sandbox the reconcile scripts run patched code in, sourced by pr-job.sh
# and carry.sh. sandbox runs a command on the checkout in the current
# directory with no network, an empty environment and a read-only checkout:
# only catalog/snapshot.json, which the checks regenerate, and Go's caches
# are writable, so the code can't reach secrets or change what runs after
# it. Docker's socket is hidden too. SANDBOX_RO lists more directories the
# command must not write (outputs, private copies). sandbox_check fails
# unless all of that holds; call it before the first sandbox run. Source it
# after Go and the modules are installed: nothing inside can download.

repo="${repo:-$(pwd -P)}"
git_dir="$(cd "$(git rev-parse --git-dir)" && pwd -P)"
git_common="$(cd "$(git rev-parse --git-common-dir)" && pwd -P)"
gocache="$(go env GOCACHE)"
gomodcache="$(go env GOMODCACHE)"
mkdir -p "$gocache" "$gomodcache"

sandbox() {
  local tmp=/tmp
  if [[ "$(uname -s)" == Darwin ]]; then
    tmp="$(cd "${TMPDIR:-/tmp}" && pwd -P)"
    mkdir -p "$tmp/sandbox-home"
  fi
  local env=(env -i PATH="$PATH" HOME="$tmp/sandbox-home" TMPDIR="$tmp"
    GOROOT="$(go env GOROOT)" GOCACHE="$gocache" GOMODCACHE="$gomodcache"
    GOFLAGS=-mod=readonly GOPROXY=off GOSUMDB=off GOTOOLCHAIN=local GOTELEMETRY=off)
  local ro=() deny="" dir
  for dir in "${SANDBOX_RO[@]+"${SANDBOX_RO[@]}"}"; do
    ro+=(--ro-bind "$dir" "$dir")
    deny+=" (subpath \"$dir\")"
  done
  # The Docker socket would give patched code root, and neither a read-only
  # mount nor --unshare-net stops a Unix socket connection, so hide /run, where
  # it lives. The macOS profile, for local runs only, denies it by name.
  case "$(uname -s)" in
    Linux)
      "${env[@]}" bwrap --die-with-parent --new-session --unshare-net --unshare-pid \
        --ro-bind / / --tmpfs /run --dev /dev --proc /proc --tmpfs /tmp --dir /tmp/sandbox-home \
        --ro-bind "$repo" "$repo" --bind "$gocache" "$gocache" --bind "$gomodcache" "$gomodcache" \
        --bind "$repo/catalog/snapshot.json" "$repo/catalog/snapshot.json" \
        --ro-bind "$git_dir" "$git_dir" --ro-bind "$git_common" "$git_common" "${ro[@]+"${ro[@]}"}" \
        "$@"
      ;;
    Darwin)
      "${env[@]}" sandbox-exec -p "(version 1)
        (allow default)
        (deny network-outbound)
        (allow network-outbound (remote ip \"localhost:*\") (remote unix-socket))
        (deny network-outbound (remote unix-socket (path-regex #\"docker\\.sock$\")))
        (deny file-write*)
        (allow file-write* (subpath \"$gocache\") (subpath \"$gomodcache\")
          (subpath \"$tmp\") (subpath \"/private/tmp\") (regex #\"^/dev/\"))
        (deny file-write* (subpath \"$repo\"))
        (allow file-write* (literal \"$repo/catalog/snapshot.json\") (subpath \"$gocache\") (subpath \"$gomodcache\"))
        (deny file-write* (subpath \"$git_dir\") (subpath \"$git_common\")$deny)" \
        "$@"
      ;;
    *)
      echo "sandbox: no sandbox for $(uname -s)" >&2
      return 1
      ;;
  esac
}

# Fails unless the sandbox hides the environment, blocks the network and
# Docker's socket, and keeps the checkout, .git and SANDBOX_RO read-only.
sandbox_check() {
  local dirs=("$repo" "$repo/.build" "$git_common" "${SANDBOX_RO[@]+"${SANDBOX_RO[@]}"}") dir
  if RECONCILE_PROBE_SECRET=probe sandbox bash -c '
    [[ -z "${RECONCILE_PROBE_SECRET:-}" ]] || exit 11
    if curl -sS --max-time 5 -o /dev/null https://proxy.golang.org 2>/dev/null; then exit 12; fi
    # 7 is a refused connection; anything else means something answers.
    curl -sS --max-time 5 -o /dev/null --unix-socket /var/run/docker.sock http://localhost/_ping 2>/dev/null
    [[ $? -eq 7 ]] || exit 14
    for dir in "$@"; do
      if touch "$dir/.reconcile-probe" 2>/dev/null; then exit 13; fi
    done' _ "${dirs[@]}"; then
    return 0
  fi
  for dir in "${dirs[@]}"; do rm -f "$dir/.reconcile-probe"; done
  return 1
}
