#!/usr/bin/env python3
"""Combine guardrails + AI verdict into the final approve/abstain decision.

Approve ONLY if every condition holds:
  - guardrails eligible
  - AI verdict == "approve"
  - AI confidence is a finite number in [0, 1] and >= threshold
  - if a dependency update: supply-chain checks are AFFIRMATIVELY clean
    (checked is true and every flag is explicitly false)
Anything else abstains. Any error also abstains: this script always writes a
decision and never exits non-zero on bad input.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

from common import load_json

SUPPLY_CHAIN_FLAGS = ("new_maintainers", "unusual_version_jump", "added_install_scripts")


def _unit_float(value):
    """A finite float in [0, 1], else None. Accepts only real numerics: bool
    is rejected explicitly (float(True) == 1.0 would pass any threshold), and
    strings are rejected because the schema says number, not "0.9"/"NaN"."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    if math.isnan(f) or math.isinf(f) or not 0.0 <= f <= 1.0:
        return None
    return f


def decide(guardrails: dict, verdict) -> dict:
    reasons: list[str] = []

    if not guardrails.get("eligible"):
        reasons.append("guardrails not satisfied")

    if not isinstance(verdict, dict):
        reasons.append("no AI verdict produced")
        return {"approve": False, "reasons": reasons}

    if verdict.get("verdict") != "approve":
        reasons.append(f"AI verdict is {verdict.get('verdict')!r}")

    threshold = _unit_float(guardrails.get("confidence_threshold"))
    if threshold is None:
        threshold = 0.8
    confidence = _unit_float(verdict.get("confidence"))
    if confidence is None or confidence < threshold:
        reasons.append(
            f"AI confidence {verdict.get('confidence')!r} is not a valid value "
            f">= threshold {threshold}"
        )

    if guardrails.get("is_dependency"):
        sc = verdict.get("supply_chain")
        sc = sc if isinstance(sc, dict) else {}
        # Fail closed: the gate requires an affirmative clean result, not the
        # mere absence of a `true` flag.
        if sc.get("checked") is not True or any(
            sc.get(k) is not False for k in SUPPLY_CHAIN_FLAGS
        ):
            reasons.append(f"supply-chain checks not affirmatively clean: {sc!r}")

    return {"approve": not reasons, "reasons": reasons}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--guardrails", required=True)
    ap.add_argument("--verdict", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    try:
        guardrails = load_json(args.guardrails)
        if not isinstance(guardrails, dict):
            guardrails = {"eligible": False}
        result = decide(guardrails, load_json(args.verdict))
    except Exception as e:  # noqa: BLE001 - deliberate catch-all, fail closed
        result = {"approve": False, "reasons": [f"decision error: {e}"]}

    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
