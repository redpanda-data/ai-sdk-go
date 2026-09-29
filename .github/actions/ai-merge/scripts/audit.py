#!/usr/bin/env python3
"""Render the PR audit record (sticky comment, step summary, and the approving
review's body). Every evaluated PR gets one: verdict, config/prompt version,
model, guardrail and supply-chain results, and a link to the run. Best-effort
loads mean it renders even when an earlier step failed."""

from __future__ import annotations

import argparse
import os

from common import load_json
from sanitize import sanitize_verdict, scan_for_secrets

MARKER = "<!-- ai-approved-merge -->"

MISSING_GUARDRAILS = {
    "eligible": False,
    "reasons": [
        "internal error: guardrails were not evaluated (an earlier pipeline "
        "step failed) — see the run link"
    ],
    "config_version": "unknown",
    "is_dependency": False,
}


def render(
    guardrails: dict,
    verdict: dict | None,
    decision: dict | None,
    run_url: str,
    dry_run: bool = False,
    approve_outcome: str = "",
    shadows: dict | None = None,
) -> str:
    approve = bool((decision or {}).get("approve"))
    # Second layer: never render model text unsanitised, whichever engine produced it.
    if isinstance(verdict, dict):
        verdict = (
            sanitize_verdict(verdict)
            if not scan_for_secrets(verdict)
            else {
                "verdict": "comment",
                "confidence": 0.0,
                "reviewed_fully": False,
                "scope": "unknown",
                "summary": "verdict withheld: contained a secret-shaped string",
                "concerns": [],
                "evidence": [],
            }
        )
    if shadows:
        shadows = {
            k: (
                sanitize_verdict(v)
                if isinstance(v, dict) and not scan_for_secrets(v)
                else {"verdict": "withheld"}
            )
            for k, v in shadows.items()
        }
    # The headline IS the outcome. The mechanism's name never appears in a way
    # that could read as a claim of approval on a PR that was not approved.
    if dry_run:
        status = (
            "🟡 AI review (dry run): would have approved — no approval posted"
            if approve
            else "⏸️ AI review (dry run): human review required"
        )
    elif approve:
        # Reflect what actually happened, not just what was decided.
        if approve_outcome == "head-moved":
            status = (
                "⏸️ AI review: approval decided but NOT posted — new commits arrived "
                "during review; the new push is re-evaluated"
            )
        elif approve_outcome in ("", "success"):
            status = "✅ AI review: approved — merging is the author's call"
        else:
            status = (
                f"⚠️ AI review: approval decided but the approve step did not succeed "
                f"({approve_outcome}) — see run"
            )
    else:
        status = "⏸️ AI review: human review required"

    v = verdict or {}
    lines = [
        MARKER,
        f"### {status}",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Config version | `{guardrails.get('config_version', 'unknown')}` |",
        f"| Prompt version | `{v.get('prompt_version', 'n/a')}` |",
        f"| Model | `{v.get('model', 'n/a')}` |",
        f"| Engine | `{guardrails.get('engine', 'single-call')}`"
        + (f" ({v['engine']})" if v.get("engine") else "")
        + " |",
        (
            "| AI verdict | rejected — see reasons |"
            if v.get("error")
            else f"| AI verdict | `{v.get('verdict', 'n/a')}` "
            f"(confidence {v.get('confidence', 'n/a')}, "
            f"reviewed fully: {v.get('reviewed_fully', 'n/a')}, "
            f"scope: {v.get('scope', 'n/a')}) |"
        ),
        f"| Files / lines (all) | {guardrails.get('changed_files', '?')} / "
        f"{guardrails.get('total_lines', '?')} |",
        f"| CI at this commit | {guardrails.get('ci_status', 'unknown')} |",
        f"| Reviewable diff | {guardrails.get('reviewable_diff_chars', '?')} chars "
        f"(limit {guardrails.get('max_diff_chars', '?')}) |",
        f"| Reviewable files / lines | {guardrails.get('reviewable_files', '?')} / "
        f"{guardrails.get('reviewable_lines', '?')} |",
        f"| Generated (CI-verified) files / lines | "
        f"{len(guardrails.get('generated_files') or [])} / "
        f"{guardrails.get('generated_lines', 0)} |",
        f"| Test / source lines | {guardrails.get('test_lines', '?')} / "
        f"{guardrails.get('source_lines', '?')} "
        f"(tests changed with source: {guardrails.get('tests_changed_with_source', '?')}) |",
        f"| Dependency update | {guardrails.get('is_dependency', False)} |",
        f"| Run | {run_url} |",
    ]

    if v.get("summary"):
        lines += ["", "**Review summary**", "", str(v["summary"])]
    concerns = v.get("concerns") if isinstance(v.get("concerns"), list) else []
    if v.get("error"):
        # a rejected placeholder's only concern is its error, shown once in the reasons
        concerns = []
    if concerns:
        lines += ["", "**Concerns**"] + [f"- {c}" for c in concerns]

    if v.get("evidence"):
        lines += ["", "**Evidence cited by the engine**"] + [
            f"- `{e.get('path')}`"
            + (f":{e['line']}" if e.get("line") else "")
            + f" — {e.get('note', '')}"
            for e in v["evidence"][:12]
        ]

    if shadows:
        lines += [
            "",
            "**Shadow engines (recorded for comparison; did not decide)**",
            "",
            "| Engine | Verdict | Confidence | Reviewed fully | Scope |",
            "|---|---|---|---|---|",
        ]
        for name, sv in shadows.items():
            lines.append(
                f"| `{name}` | `{sv.get('verdict', 'n/a')}` | "
                f"{sv.get('confidence', 'n/a')} | "
                f"{sv.get('reviewed_fully', 'n/a')} | {sv.get('scope', 'n/a')} |"
            )

    if guardrails.get("generated_files"):
        lines += ["", "**Generated files (not sent to the model)**"] + [
            f"- {g}" for g in guardrails["generated_files"]
        ]

    if guardrails.get("is_dependency"):
        sc = v.get("supply_chain")
        sc = sc if isinstance(sc, dict) else {}
        lines += [
            "",
            "**Supply-chain checks**",
            f"- checked: {sc.get('checked')}",
            f"- new maintainers: {sc.get('new_maintainers')}",
            f"- unusual version jump: {sc.get('unusual_version_jump')}",
            f"- added install scripts: {sc.get('added_install_scripts')}",
        ]
        if sc.get("notes"):
            lines += [f"- notes: {sc['notes']}"]

    if not guardrails.get("eligible"):
        # The gates decided; the decision block would only restate that.
        lines += ["", "**Not eligible — reasons**"] + [
            f"- {r}" for r in guardrails.get("reasons", [])
        ]
    elif decision and decision.get("reasons") and not approve:
        lines += ["", "**Not approved — reasons**"] + [
            f"- {r}" for r in decision["reasons"]
        ]

    lines += [
        "",
        "_Generated by the AI review gate. The bot only "
        "posts an approval; the author merges. This is an audit record; the branch ruleset is "
        "what enforces it._",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--guardrails", required=True)
    ap.add_argument("--verdict", default="")
    ap.add_argument("--decision", default="")
    ap.add_argument("--run-url", required=True)
    ap.add_argument("--dry-run", default="false")
    ap.add_argument("--approve-outcome", default="")
    ap.add_argument(
        "--shadows", default="", help="comma-separated shadow-<engine>.json paths"
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    guardrails = load_json(args.guardrails)
    if not isinstance(guardrails, dict):
        guardrails = MISSING_GUARDRAILS
    verdict = load_json(args.verdict)
    decision = load_json(args.decision)

    shadows = {}
    for sp in [x for x in args.shadows.split(",") if x.strip()]:
        name = os.path.basename(sp)[len("shadow-") : -len(".json")]
        sv = load_json(sp)
        shadows[name] = sv if isinstance(sv, dict) else {"verdict": "unavailable"}
    body = render(
        guardrails,
        verdict if isinstance(verdict, dict) else None,
        decision if isinstance(decision, dict) else None,
        args.run_url,
        dry_run=str(args.dry_run).lower() == "true",
        approve_outcome=args.approve_outcome,
        shadows=shadows,
    )
    with open(args.out, "w") as fh:
        fh.write(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
