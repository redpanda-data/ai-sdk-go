#!/usr/bin/env python3
"""Build the engine request envelope (docs/verdict-contract.md, schema 1).

Runs in the AGENT job with a read-only GITHUB_TOKEN. Everything here is data
the trusted job also derives independently; the envelope exists so the agent
does not have to guess at PR metadata or CI state, and so ADP receives the
same input later. Nothing from the PR is executed to build it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import wait_for_checks  # noqa: E402


def gh(*args: str):
    out = subprocess.run(
        ["gh", "api", *args], check=True, capture_output=True, text=True
    ).stdout
    return json.loads(out) if out.strip() else None


def gh_items(endpoint: str, jq_item: str) -> list:
    """Paginated list endpoints: `--paginate --jq '.[] | {...}'` emits one JSON
    object per line across pages (NDJSON), so parse line by line. A single
    json.loads on the concatenated output breaks past the first page."""
    out = subprocess.run(
        ["gh", "api", "--paginate", endpoint, "--jq", jq_item],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [json.loads(ln) for ln in out.splitlines() if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pr", required=True, type=int)
    ap.add_argument("--head-sha", required=True)
    ap.add_argument("--base-sha", required=True)
    ap.add_argument(
        "--config", required=True, help="repo ai-merge config (from BASE ref)"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--wait-seconds",
        type=int,
        default=900,
        help="wait up to this long for OTHER check runs on the head commit to complete",
    )
    ap.add_argument(
        "--ignore-check",
        action="append",
        default=[],
        help="check-run names to ignore while waiting (this mechanism's own jobs)",
    )
    a = ap.parse_args()

    import yaml  # PyYAML pinned in requirements.txt

    with open(a.config) as fh:
        cfg = yaml.safe_load(fh) or {}

    pr = gh(f"repos/{a.repo}/pulls/{a.pr}")
    files = gh_items(
        f"repos/{a.repo}/pulls/{a.pr}/files",
        ".[] | {path: .filename, status, additions, deletions, previous_filename}",
    )
    # The agent starts at the same moment as CI. Wait (bounded) for the OTHER
    # check runs on this commit — including the repo's required checks, which
    # may not even exist yet on the first poll. Our own jobs are ignored.
    raw_required = cfg.get("ci_checks")
    if raw_required is not None and not (
        isinstance(raw_required, list) and all(isinstance(x, str) for x in raw_required)
    ):
        print(
            "::warning::config `ci_checks` must be a list of strings; ignoring for the wait "
            "(the trusted job refuses on it)"
        )
        raw_required = None
    required = list(raw_required or []) or None

    def _fetch():
        return gh_items(
            f"repos/{a.repo}/commits/{a.head_sha}/check-runs",
            ".check_runs[] | {name, status, conclusion}",
        )

    checks, ci_status = wait_for_checks(
        _fetch, set(a.ignore_check), required, a.wait_seconds
    )
    print(f"ci_status={ci_status} ({len(checks)} check runs)")
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a") as fh:
            fh.write(f"ci_status={ci_status}\n")
    envelope = {
        "schema": "1",
        "repo": a.repo,
        "pr": a.pr,
        "title": pr.get("title", ""),
        "body": pr.get("body") or "",
        "head_sha": a.head_sha,
        "base_sha": a.base_sha,
        "author": (pr.get("user") or {}).get("login", ""),
        "changed_files": files or [],
        "ci_checks": checks,
        "ci_status": ci_status,
        "config": {
            "excluded_paths": cfg.get("excluded_paths", []),
            "generated_paths": cfg.get("generated_paths", []),
            "dependency_paths": cfg.get("dependency_paths", []),
            "prompt_profile": cfg.get("prompt_profile", "code"),
        },
    }
    with open(a.out, "w") as fh:
        json.dump(envelope, fh, indent=2)
    print(f"envelope: {len(envelope['changed_files'])} files, {len(checks)} check runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
