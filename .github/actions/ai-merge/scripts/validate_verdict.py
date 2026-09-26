#!/usr/bin/env python3
"""Validate an externally produced verdict against the verdict contract.

The judgment engine (agent job, later ADP) is untrusted-by-construction: it
reads PR content. This validator is the seam that makes its output safe to
act on. Anything that fails validation becomes a NON-approving verdict with an
explicit reason, so the pipeline fails closed and the audit says why.

Checks (all must pass):
  - JSON object with the required fields and types (see docs/verdict-contract.md)
  - verdict in {approve, request_changes, comment}; scope in the scope enum
  - confidence is a real number in [0, 1] (bool/str/NaN rejected)
  - reviewed_fully is a real bool
  - meta.head_sha == the PR head the trusted job is about to approve
  - meta.engine present; meta.run_id equals this workflow run (same-run provenance)
  - every evidence.path exists in the PR's tree at head OR is a changed file
  - an APPROVE verdict must cite at least one evidence entry
  - no secret-shaped string anywhere (else the verdict is rejected, not echoed)
  - all free text is link-stripped, HTML-escaped and length-capped for the comment
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

from sanitize import sanitize_verdict, scan_for_secrets

VERDICTS = {"approve", "request_changes", "comment"}
SCOPES = {"mechanical", "localized", "broad"}


def _abstain(reason: str, engine: str = "unknown") -> dict[str, Any]:
    return {
        "verdict": "comment",
        "confidence": 0.0,
        "reviewed_fully": False,
        "scope": "unknown",
        "summary": f"Agent verdict rejected: {reason}",
        "concerns": [reason],
        "supply_chain": {"checked": False},
        "evidence": [],
        "engine": engine,
        "error": reason,
    }


def _num(v) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    if not math.isfinite(f) or not 0.0 <= f <= 1.0:
        return None
    return f


def validate(
    verdict: Any,
    meta: Any,
    expected_head_sha: str,
    expected_run_id: str,
    tree_paths: set[str],
    changed_paths: set[str],
) -> dict[str, Any]:
    engine = (
        str((meta or {}).get("engine", "unknown"))
        if isinstance(meta, dict)
        else "unknown"
    )

    if not isinstance(meta, dict):
        return _abstain("verdict metadata missing or not an object", engine)
    if str(meta.get("head_sha", "")) != expected_head_sha:
        return _abstain(
            f"verdict head_sha {meta.get('head_sha')!r} does not match "
            f"PR head {expected_head_sha!r}",
            engine,
        )
    if str(meta.get("run_id", "")) != str(expected_run_id):
        return _abstain("verdict was not produced by this workflow run", engine)
    if not meta.get("engine"):
        return _abstain("verdict metadata has no engine stamp", engine)

    if not isinstance(verdict, dict):
        return _abstain("verdict is not a JSON object", engine)
    # Output-side defense: a secret-shaped string anywhere in the verdict means
    # the agent was steered into exfiltration. Reject the whole verdict and
    # never echo the content (the hint is a 12-char prefix only).
    hit = scan_for_secrets(verdict)
    if hit:
        return _abstain(
            f"verdict contains a secret-shaped string (category: {hit}); rejected and not recorded",
            engine,
        )
    v = verdict.get("verdict")
    if v not in VERDICTS:
        return _abstain(f"verdict value {v!r} not in {sorted(VERDICTS)}", engine)
    conf = _num(verdict.get("confidence"))
    if conf is None:
        return _abstain(
            f"confidence {verdict.get('confidence')!r} is not a finite number in [0,1]",
            engine,
        )
    rf = verdict.get("reviewed_fully")
    if not isinstance(rf, bool):
        return _abstain("reviewed_fully must be a boolean", engine)
    if verdict.get("scope") not in SCOPES:
        return _abstain(
            f"scope {verdict.get('scope')!r} not in {sorted(SCOPES)}", engine
        )
    if not isinstance(verdict.get("summary"), str) or not verdict["summary"].strip():
        return _abstain("summary must be a non-empty string", engine)
    concerns = verdict.get("concerns", [])
    if not isinstance(concerns, list) or not all(isinstance(c, str) for c in concerns):
        return _abstain("concerns must be a list of strings", engine)

    evidence = verdict.get("evidence", [])
    if not isinstance(evidence, list):
        return _abstain("evidence must be a list", engine)
    allowed = tree_paths | changed_paths

    def _norm(path: str) -> str:
        # The agent reads the PR at ./pr/<path>; accept that spelling and plain
        # "./" prefixes, then require the repo-relative path to exist.
        p = path.strip()
        while p.startswith("./"):
            p = p[2:]
        if p.startswith("pr/") and p not in allowed and p[3:] in allowed:
            p = p[3:]
        return p

    for e in evidence:
        if not isinstance(e, dict) or not isinstance(e.get("path"), str):
            return _abstain(
                "evidence entries must be objects with a string path", engine
            )
        e["path"] = _norm(e["path"])
        if e["path"] not in allowed:
            return _abstain(
                f"evidence cites a path not in the PR tree: {e['path']!r}", engine
            )
        line = e.get("line")
        if "line" in e and (
            isinstance(line, bool) or not isinstance(line, int) or line < 1
        ):
            return _abstain(
                f"evidence line for {e['path']!r} is not a positive integer", engine
            )
    if v == "approve" and not evidence:
        return _abstain(
            "an approve verdict must cite at least one evidence entry", engine
        )

    sc = verdict.get("supply_chain", {"checked": False})
    if not isinstance(sc, dict):
        return _abstain("supply_chain must be an object", engine)

    out = {
        "verdict": v,
        "confidence": conf,
        "reviewed_fully": rf,
        "scope": verdict["scope"],
        "summary": verdict["summary"].strip(),
        "concerns": concerns,
        "supply_chain": sc,
        "evidence": evidence,
        "engine": engine,
        "model": str(meta.get("model", "")) or None,
        "prompt_version": str(meta.get("prompt_version", "")) or "unknown",
    }
    if isinstance(verdict.get("risk"), dict):
        out["risk"] = verdict["risk"]
    return sanitize_verdict(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdict", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--head-sha", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument(
        "--tree", required=True, help="newline-separated repo paths at head"
    )
    ap.add_argument("--files", required=True, help="files.json from the trusted job")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    def _load(p):
        try:
            with open(p) as fh:
                return json.load(fh)
        except Exception as e:  # noqa: BLE001
            return {"__error__": f"{type(e).__name__}: {e}"}

    verdict = _load(a.verdict)
    meta = _load(a.meta)
    if isinstance(verdict, dict) and "__error__" in verdict:
        result = _abstain(f"could not read agent verdict ({verdict['__error__']})")
    elif isinstance(meta, dict) and "__error__" in meta:
        result = _abstain(
            f"could not read agent verdict metadata ({meta['__error__']})"
        )
    else:
        try:
            with open(a.tree) as fh:
                tree = {ln.strip() for ln in fh if ln.strip()}
        except OSError:
            tree = set()
        try:
            with open(a.files) as fh:
                changed = {f["filename"] for f in json.load(fh) if f.get("filename")}
        except Exception:  # noqa: BLE001
            changed = set()
        result = validate(verdict, meta, a.head_sha, a.run_id, tree, changed)

    with open(a.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(
        f"engine={result.get('engine')} verdict={result['verdict']} "
        f"confidence={result['confidence']} reviewed_fully={result['reviewed_fully']}"
        + (f" REJECTED: {result['error']}" if result.get("error") else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
