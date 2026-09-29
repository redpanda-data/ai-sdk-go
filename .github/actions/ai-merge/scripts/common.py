"""Shared helpers for the AI-approved-merge scripts."""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache


def load_json(path: str):
    """Best-effort JSON load: missing/empty/corrupt file => None (fail closed)."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (ValueError, OSError):
        return None


@lru_cache(maxsize=512)
def glob_to_regex(pattern: str) -> re.Pattern:
    """Compile a glob with real globstar semantics.

    `**/` matches zero or more directories, a trailing `**` matches the rest of
    the path, and `*` / `?` never cross a `/`. (fnmatch has none of this: its
    `*` crosses `/` and a `**/` prefix leaves a literal `/` that a root-level
    file can never satisfy.)
    """
    out: list[str] = []
    i, n = 0, len(pattern)
    while i < n:
        c = pattern[i]
        if c == "*":
            if pattern.startswith("**/", i):
                out.append("(?:.*/)?")
                i += 3
            elif pattern.startswith("**", i):
                out.append(".*")
                i += 2
            else:
                out.append("[^/]*")
                i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def match_any(path: str, patterns: list[str]) -> bool:
    return any(glob_to_regex(p).match(path) for p in patterns)


def classify_checks(
    checks: list[dict],
    ignore_names: set[str] | None = None,
    required_names: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Classify a commit's check runs into passed | failed | pending | none.

    Used by BOTH the agent job (to decide when to start) and the trusted job
    (to gate the decision), so the two can never disagree on what CI said.

    - `ignore_names`: this mechanism's own jobs.
    - `required_names`: the repo's real CI checks. `passed` requires EVERY one
      of them to be present with a substantive success. Any other check (e.g.
      an @-mention review bot that runs on every PR) can still make the result
      `failed` or `pending`, but can never on its own make it `passed` —
      otherwise a docs-only PR would "pass CI" on the strength of a bot.
    - `action_required` = "a workflow is waiting for a human to allow it" — not
      a test result, so it counts as neither.
    """
    ignore = ignore_names or set()
    runs = [c for c in checks if c.get("name") not in ignore]
    if any(c.get("status") != "completed" for c in runs):
        return "pending"
    substantive = [
        c
        for c in runs
        if c.get("conclusion") not in ("action_required", "skipped", None)
    ]
    # A real failure is the most important signal: report it even if other
    # required checks are missing.
    if any(c.get("conclusion") not in ("success", "neutral") for c in substantive):
        return "failed"
    if required_names:
        by_name = {c.get("name"): c for c in substantive}
        if any(r not in by_name for r in required_names):
            return "none"
    elif not substantive:
        return "none"
    return "passed"


def wait_for_checks(
    fetch,
    ignore_names: set[str],
    required_names: list[str] | None,
    wait_seconds: int,
    grace_seconds: int = 90,
    poll_seconds: int = 20,
    sleep=None,
    clock=None,
) -> tuple[list[dict], str]:
    """Poll `fetch()` until CI is settled or the budget is spent.

    Keeps waiting while: any run is pending; a required check has not appeared
    yet; or NO run exists yet and we are inside the grace period (GitHub can
    take a moment to create check runs after a push — an empty first poll
    must not be mistaken for "no CI ran"). Returns (checks, status).
    """
    import time as _time

    sleep = sleep or _time.sleep
    clock = clock or _time.monotonic
    start = clock()
    while True:
        checks = fetch()
        status = classify_checks(checks, ignore_names, required_names)
        elapsed = clock() - start
        if status == "pending":
            settling = True
        elif status == "none":
            # Nothing (or not everything required) has run yet. Inside the grace
            # period GitHub may simply not have created the runs. After it, if
            # every other run is complete and a required check still hasn't
            # appeared, the path filter almost certainly excluded it: stop
            # waiting — 90 s instead of the full budget on a docs-only PR.
            settling = elapsed < grace_seconds
        else:
            settling = False
        if not settling or elapsed >= wait_seconds:
            return checks, status
        sleep(poll_seconds)
