#!/usr/bin/env python3
"""Cheap eligibility precheck for the AGENT job.

Runs the same deterministic guardrails as the trusted job, minus org
membership (which needs the App token and lives in the approve job). Its only
purpose is to avoid waiting for CI and spending a model call on a PR the gates
will refuse anyway (excluded path, opt-out label, invalid config). It decides
nothing: the approve job re-runs the full gates regardless.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guardrails import evaluate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pr", required=True, type=int)
    ap.add_argument("--config", required=True)
    ap.add_argument("--skipped", default="false")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import yaml

    with open(a.config) as fh:
        cfg = yaml.safe_load(fh) or {}
    out = subprocess.run(
        [
            "gh",
            "api",
            "--paginate",
            f"repos/{a.repo}/pulls/{a.pr}/files",
            "--jq",
            ".[] | {filename, status, additions, deletions, previous_filename, "
            "has_patch: (.patch != null)}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    files = [json.loads(ln) for ln in out.splitlines() if ln.strip()]
    # Membership is asserted here ONLY so it does not mask the path/config
    # outcome; the approve job verifies it for real.
    pr = {
        "author_is_member": True,
        "membership_check_status": "204",
        "author": "precheck",
    }
    r = evaluate(cfg, files, pr, True, skipped=a.skipped.lower() == "true")
    with open(a.out, "w") as fh:
        json.dump({"eligible": r["eligible"], "reasons": r["reasons"]}, fh, indent=2)
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a") as fh:
            fh.write(f"eligible={'true' if r['eligible'] else 'false'}\n")
    verdict = (
        "eligible" if r["eligible"] else "NOT eligible — " + "; ".join(r["reasons"])
    )
    print("precheck:", verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
