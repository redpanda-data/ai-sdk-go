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
import math
import sys
from typing import Any

import yaml

from common import match_any

# Baseline exclusions applied in EVERY repo regardless of local config. These
# are the never-auto-approvable categories: CI/CD, infrastructure-as-code,
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
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        reasons.append(f"config `{name}` must be a finite number")
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


ENGINES = {"single-call", "agent-action"}

TEST_PATTERNS = [
    "**/*_test.go",
    "**/*_test.py",
    "**/test_*.py",
    "**/tests/**",
    "**/*.test.ts",
    "**/*.spec.ts",
    "**/testdata/**",
]


def _unreviewable(f: dict[str, Any]) -> bool:
    """A changed file with no patch is binary or too large for GitHub to
    render: the model would have nothing to review, yet its additions/deletions
    can be 0 and slip past the size gate. Pure deletions and zero-change renames
    also have no patch but hide nothing, so they stay reviewable."""
    # has_patch is REQUIRED (fail closed if the projection ever drops it).
    if f.get("has_patch") is True:
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
        reasons.append(
            "opt-out label present: author excluded this PR from AI approval"
        )

    if any(not f.get("filename") for f in files):
        reasons.append("changed-files list contains an entry without a filename")
    files = [f for f in files if f.get("filename")]
    unreviewable = sorted(f["filename"] for f in files if _unreviewable(f))
    if unreviewable:
        reasons.append(
            "file(s) with no reviewable patch (binary or too large): "
            + ", ".join(unreviewable)
        )

    # Generated files: machine-produced AND verified by a REQUIRED CI check that
    # regenerates and compares them (so a hand edit fails CI and cannot merge).
    # They do not count against the size caps and are not sent to the model;
    # the model reviews the hand-written delta and is told what was generated.
    # An excluded path always wins over a generated path.
    generated_paths = _as_list(
        config.get("generated_paths"), "generated_paths", reasons
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

    def _lines(f):
        return int(f.get("additions", 0)) + int(f.get("deletions", 0))

    def _is_generated(f):
        # Both names must match (mirrors the exclusion matcher): renaming a
        # hand-written file INTO a generated path must not hide it from review.
        if not match_any(f["filename"], generated_paths) or match_any(
            f["filename"], excluded
        ):
            return False
        prev = f.get("previous_filename")
        return not prev or match_any(prev, generated_paths)

    generated = [f for f in files if _is_generated(f)]
    generated_names = {f["filename"] for f in generated}
    reviewable = [f for f in files if f["filename"] not in generated_names]
    tests = [f for f in reviewable if match_any(f["filename"], TEST_PATTERNS)]
    source = [
        f for f in reviewable if f["filename"] not in {t["filename"] for t in tests}
    ]

    # Size is NOT a risk score and does not gate here. Counts are recorded as
    # signals for the audit record and a future risk-scoring layer. The only
    # size rule is "the whole reviewable diff must fit in one review", which is
    # enforced downstream by review.py against `max_diff_chars` (config, chars).
    changed_files = len(files)
    total_lines = sum(_lines(f) for f in files)
    reviewable_files = len(reviewable)
    reviewable_lines = sum(_lines(f) for f in reviewable)
    generated_lines = sum(_lines(f) for f in generated)
    max_diff_chars = int(
        _as_number(config.get("max_diff_chars"), "max_diff_chars", 120_000, reasons)
    )

    # Judgment engine selection (docs/verdict-contract.md). Validated here so a
    # typo fails closed with a reason instead of a crash in the action.
    engine = config.get("engine", "single-call")
    if engine not in ENGINES:
        reasons.append(f"config `engine` must be one of {sorted(ENGINES)}")
        engine = "single-call"
    shadow_engines = [
        e
        for e in _as_list(config.get("shadow_engines"), "shadow_engines", reasons)
        if e != engine
    ]
    bad_shadow = [e for e in shadow_engines if e not in ENGINES]
    if bad_shadow:
        reasons.append(f"config `shadow_engines` has unknown engine(s): {bad_shadow}")
    if max_diff_chars <= 0:
        reasons.append("config `max_diff_chars` must be a positive number")

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

    threshold = float(
        _as_number(config.get("min_confidence"), "min_confidence", 0.8, reasons)
    )
    if not 0.0 <= threshold <= 1.0:
        # An out-of-range threshold (e.g. 95 meaning "95%") must not be silently
        # replaced by a looser default downstream: refuse the PR instead.
        reasons.append(
            f"config `min_confidence` must be between 0 and 1 (got {threshold})"
        )
    return {
        "eligible": len(reasons) == 0,
        "reasons": reasons,
        "config_version": version,
        "is_dependency": is_dependency,
        "changed_files": changed_files,
        "total_lines": total_lines,
        "reviewable_files": reviewable_files,
        "reviewable_lines": reviewable_lines,
        "generated_files": sorted(generated_names),
        "generated_lines": generated_lines,
        # Composition signals for the audit record (and a future risk-scoring
        # layer). Informational: they do not gate in v1.
        "test_lines": sum(_lines(f) for f in tests),
        "source_lines": sum(_lines(f) for f in source),
        "tests_changed_with_source": bool(tests) and bool(source),
        "confidence_threshold": threshold,
        "max_diff_chars": max_diff_chars,
        "engine": engine,
        "shadow_engines": shadow_engines,
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
