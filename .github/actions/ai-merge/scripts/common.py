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
