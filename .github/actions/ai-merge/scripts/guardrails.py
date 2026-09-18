#!/usr/bin/env python3
"""Deterministic, fail-closed guardrail evaluation for AI-approved merges.

Reads the versioned config (from the trusted base ref), the PR's changed-files
list and PR metadata, and decides whether the PR is even *eligible* for AI
approval. Eligibility is necessary but NOT sufficient: the AI verdict still has
to say "approve" (see decide.py). Pure logic, no network.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import yaml

from common import match_any

# Baseline exclusions applied in EVERY repo regardless of local config. These
# are the never-auto-mergeable categories: CI/CD, infrastructure-as-code,
# release/container tooling, and IAM/auth directories plus credential, secret
# and key files. Repos add their own layout-specific auth code paths on top.
BASELINE_EXCLUDED = [
    ".github/**",
    ".buildkite/**",
    "**/*.tf",
    "**/*.tfvars",
    "terraform/**",
    "release/**",
    "**/Dockerfile*",
    "**/entrypoint*.sh",
    "**/.goreleaser*",
    "**/iam/**",
    "**/auth/**",
    "**/*credential*",
    "**/*secret*",
    "**/*.pem",
    "**/*.key",
]

# `**/` so both root-level and nested (monorepo) lockfiles are detected.
DEFAULT_DEP_PATHS = [
    "**/go.mod",
    "**/go.sum",
    "**/package.json",
    "**/package-lock.json",
    "**/pnpm-lock.yaml",
    "**/yarn.lock",
    "**/requirements*.txt",
    "**/poetry.lock",
]


def _as_number(value, name: str, default, reasons: list[str]):
    """Numeric config with a reason on garbage (not a crash). bool is not a
    number here either."""
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        reasons.append(f"config `{name}` must be a number")
        return default
    return value


def _as_list(value, name: str, reasons: list[str]) -> list[str]:
    """A YAML scalar where a list was meant becomes a one-element list; any
    other non-list type is a config error and disqualifies the PR."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return value
    reasons.append(f"config `{name}` must be a list of strings")
    return []


def _unreviewable(f: dict[str, Any]) -> bool:
    """A changed file with no patch is binary or too large for GitHub to
    render: the model would have nothing to review, yet its additions/deletions
    can be 0 and slip past the size gate. Pure deletions and zero-change renames
    also have no patch but hide nothing, so they stay reviewable."""
    if f.get("has_patch", True):
        return False
    zero = int(f.get("additions", 0)) == 0 and int(f.get("deletions", 0)) == 0
    if f.get("status") == "removed":
        return False
    if f.get("status") == "renamed" and zero:
        return False
    return True


def evaluate(
    config: dict[str, Any],
    files: list[dict[str, Any]],
    pr: dict[str, Any],
    config_present: bool,
    skipped: bool = False,
) -> dict[str, Any]:
    reasons: list[str] = []

    if skipped:
        reasons.append("opt-out label present: author excluded this PR from AI approval")

    if any(not f.get("filename") for f in files):
        reasons.append("changed-files list contains an entry without a filename")
    files = [f for f in files if f.get("filename")]
    unreviewable = sorted(f["filename"] for f in files if _unreviewable(f))
    if unreviewable:
        reasons.append(
            "file(s) with no reviewable patch (binary or too large): "
            + ", ".join(unreviewable)
        )

    if not config_present:
        reasons.append("no versioned config at the configured path on the base ref")
    else:
        if "version" not in config:
            reasons.append("config is missing `version`")
        if config.get("enabled") is not True:
            reasons.append("config `enabled` is not boolean true")
    version = str(config.get("version", "unknown")) if config_present else "unknown"

    # Membership is verified against the org via the App token by the action
    # (pr["author_is_member"]). author_association is NOT used: with private
    # membership (GitHub's default) it reports members as CONTRIBUTOR/NONE.
    if pr.get("author_is_member") is not True:
        status = str(pr.get("membership_check_status", "unknown"))
        if status == "404":
            reasons.append(
                f"author {pr.get('author')!r} is not a member of the organization "
                "(verified via API)"
            )
        else:
            # The check itself failed (403 = App lacks Members:read, 401, 429,
            # network). Fail closed, but say so: this is an operator problem,
            # not an outsider PR.
            reasons.append(
                f"org membership check for {pr.get('author')!r} failed (HTTP "
                f"{status}); treated as non-member. Verify the ai-merge App has "
                "Organization -> Members: read and see the run log"
            )

    excluded = BASELINE_EXCLUDED + _as_list(
        config.get("excluded_paths"), "excluded_paths", reasons
    )
    touched_excluded = sorted(
        {
            f["filename"]
            for f in files
            if match_any(f["filename"], excluded)
            or (
                f.get("previous_filename")
                and match_any(f["previous_filename"], excluded)
            )
        }
    )
    if touched_excluded:
        reasons.append("touches excluded path(s): " + ", ".join(touched_excluded))

    max_files = int(_as_number(config.get("max_changed_files"), "max_changed_files", 50, reasons))
    max_lines = int(_as_number(config.get("max_total_lines"), "max_total_lines", 800, reasons))
    changed_files = len(files)
    total_lines = sum(
        int(f.get("additions", 0)) + int(f.get("deletions", 0)) for f in files
    )
    if changed_files > max_files:
        reasons.append(f"{changed_files} files changed > max {max_files}")
    if total_lines > max_lines:
        reasons.append(f"{total_lines} lines changed > max {max_lines}")

    dep_paths = _as_list(config.get("dependency_paths"), "dependency_paths", reasons)
    if not dep_paths and "dependency_paths" not in config:
        dep_paths = DEFAULT_DEP_PATHS
    # Check the old name too: renaming a manifest away must not dodge the
    # supply-chain checks (mirrors the exclusion logic above).
    is_dependency = any(
        match_any(f["filename"], dep_paths)
        or (f.get("previous_filename") and match_any(f["previous_filename"], dep_paths))
        for f in files
    )

    threshold = float(_as_number(config.get("min_confidence"), "min_confidence", 0.8, reasons))
    return {
        "eligible": len(reasons) == 0,
        "reasons": reasons,
        "config_version": version,
        "is_dependency": is_dependency,
        "changed_files": changed_files,
        "total_lines": total_lines,
        "confidence_threshold": threshold,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--files", required=True)
    ap.add_argument("--pr", required=True)
    ap.add_argument("--config-present", default="true")
    ap.add_argument("--skipped", default="false")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    present = args.config_present.lower() == "true"
    skipped = args.skipped.lower() == "true"
    # A malformed config must make the PR ineligible, never crash the step.
    try:
        config = yaml.safe_load(open(args.config).read() or "") or {}
        if not isinstance(config, dict):
            raise ValueError("config root must be a mapping")
    except (OSError, ValueError, yaml.YAMLError):
        config, present = {}, False

    files = json.load(open(args.files)) or []
    pr = json.load(open(args.pr)) or {}

    result = evaluate(config, files, pr, present, skipped=skipped)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
