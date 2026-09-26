#!/usr/bin/env python3
"""Render prompts/agent.md with the envelope. Author-controlled text (title,
body) is fenced as untrusted data exactly as review.py does."""

from __future__ import annotations

import argparse
import json
import re
import sys

FENCE_TAGS = ("pr_title", "pr_body")
_CLOSER = re.compile(r"</\s*(" + "|".join(FENCE_TAGS) + r")\s*>", re.IGNORECASE)


def fence(tag: str, text: str) -> str:
    safe = _CLOSER.sub(lambda m: f"&lt;/{m.group(1).lower()}&gt;", text or "")
    return f"<{tag}>\n{safe}\n</{tag}>"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--envelope", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.template) as fh:
        tpl = fh.read()
    with open(a.envelope) as fh:
        env = json.load(fh)

    files = (
        "\n".join(
            f"- {f['path']} ({f['status']}, +{f['additions']}/-{f['deletions']}"
            + (f", was {f['previous_filename']}" if f.get("previous_filename") else "")
            + ")"
            for f in env["changed_files"]
        )
        or "- (none)"
    )
    checks = (
        "\n".join(
            f"- {c['name']}: {c['status']}"
            + (f" / {c['conclusion']}" if c.get("conclusion") else "")
            for c in env["ci_checks"]
        )
        or "- (no check runs reported yet)"
    )
    cfg = env["config"]

    values = {
        "REPO": env["repo"],
        "PR": str(env["pr"]),
        "HEAD_SHA": env["head_sha"],
        "PR_TITLE_FENCED": fence("pr_title", env["title"]),
        "PR_BODY_FENCED": fence("pr_body", env["body"]),
        "CHANGED_FILES": files,
        "CI_CHECKS": checks,
        "CI_STATUS": {
            "passed": "ALL CI check runs for this commit COMPLETED SUCCESSFULLY.",
            "failed": "At least one CI check run for this commit FAILED. Do NOT approve.",
            "pending": "CI check runs were still running when the wait timed out. Do NOT approve; "
            "say so in concerns.",
            "none": "NO CI check runs ran for this commit (CI is path-filtered here). Nothing has "
            "verified this change. Approve ONLY if the change cannot affect behaviour "
            "(comments, docs, non-executed text) and state that explicitly in the summary.",
        }.get(env.get("ci_status", "none")),
        "EXCLUDED_PATHS": ", ".join(cfg["excluded_paths"]) or "(none)",
        "GENERATED_PATHS": ", ".join(cfg["generated_paths"]) or "(none)",
        "DEPENDENCY_PATHS": ", ".join(cfg["dependency_paths"]) or "(none)",
    }
    # One pass over the template: a placeholder-looking string inside a PR
    # title/body is inserted verbatim and never expanded.
    out = re.sub(
        r"\{\{([A-Z_]+)\}\}", lambda m: values.get(m.group(1), m.group(0)), tpl
    )
    with open(a.out, "w") as fh:
        fh.write(out)
    print(f"prompt rendered: {len(out)} chars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
