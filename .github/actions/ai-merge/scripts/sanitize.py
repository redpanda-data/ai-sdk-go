"""Output-side defense for model-produced text that will be posted publicly.

The agent reads untrusted content and its text ends up in a PR comment on a
(possibly public) repo. Tool-level scoping keeps secrets out of the model's
reach *best-effort*; this module makes sure that even if something leaks into
the verdict, it never reaches the comment: secret-shaped strings REJECT the
whole verdict, and all text is length-capped, link-stripped and HTML-escaped.
"""

from __future__ import annotations

import html
import math
import re

# Deliberately broad. A false positive costs one abstain; a false negative
# posts a credential on a public PR.
SECRET_PATTERNS = [
    ("anthropic-api-key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}")),
    ("github-token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b")),
    ("github-pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("aws-access-key-id", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("aws-secret-reference", re.compile(r"\baws_secret_access_key\b", re.I)),
    ("private-key-block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("slack-token", re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b")),
    (
        "jwt",
        re.compile(
            r"\beyJ[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{10,}\b"
        ),
    ),
    (
        "env-assignment",
        re.compile(
            r"\b(?:ANTHROPIC_API_KEY|GITHUB_TOKEN|AWS_SESSION_TOKEN)\s*=\s*\S{8,}"
        ),
    ),
    ("proc-environ-path", re.compile(r"/proc/(?:self|\d+)/environ")),
]
_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
_MENTION = re.compile(r"(?<![\w/])@([A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?)\b")

LIMITS = {"summary": 1500, "concern": 500, "note": 300, "path": 300}
MAX_ITEMS = {"concerns": 15, "evidence": 15, "factors": 15}


def find_secret(text: str) -> str | None:
    """Return the CATEGORY of the first secret-shaped match (never its text), or None."""
    for name, pat in SECRET_PATTERNS:
        if pat.search(text or ""):
            return name
    return None


def scan_for_secrets(obj) -> str | None:
    """Walk any JSON-ish object; return a redacted hint if any string contains a secret."""
    if isinstance(obj, str):
        return find_secret(obj)
    if isinstance(obj, dict):
        for v in obj.values():
            hit = scan_for_secrets(v)
            if hit:
                return hit
    if isinstance(obj, list):
        for v in obj:
            hit = scan_for_secrets(v)
            if hit:
                return hit
    return None


def sanitize_text(text, max_len: int) -> str:
    """Public-comment-safe: no links, no @mentions, no HTML, bounded length."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = _URL.sub("[link removed]", text)
    text = _MENTION.sub(
        lambda m: "@\u200b" + m.group(1), text
    )  # zero-width space defuses pings
    text = html.escape(text, quote=False)
    text = re.sub(r"[\r\n]+", " ", text).strip()
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text


def sanitize_verdict(v: dict) -> dict:
    """Return a copy with every free-text field made comment-safe and lists bounded."""
    out = dict(v)
    out["summary"] = sanitize_text(v.get("summary", ""), LIMITS["summary"])
    out["concerns"] = [
        sanitize_text(c, LIMITS["concern"]) for c in (v.get("concerns") or [])
    ][: MAX_ITEMS["concerns"]]
    ev = []
    for e in (v.get("evidence") or [])[: MAX_ITEMS["evidence"]]:
        if isinstance(e, dict):
            ev.append(
                {
                    **e,
                    "path": sanitize_text(e.get("path", ""), LIMITS["path"]),
                    "note": sanitize_text(e.get("note", ""), LIMITS["note"]),
                }
            )
    out["evidence"] = ev
    risk = v.get("risk")
    if isinstance(risk, dict):
        facs = []
        for f in (risk.get("factors") or [])[: MAX_ITEMS["factors"]]:
            if isinstance(f, dict):
                facs.append(
                    {
                        **f,
                        "name": sanitize_text(f.get("name", ""), 120),
                        "evidence": sanitize_text(
                            f.get("evidence", ""), LIMITS["note"]
                        ),
                    }
                )
        score = risk.get("score")
        out["risk"] = {
            "score": (
                float(score)
                if isinstance(score, (int, float))
                and not isinstance(score, bool)
                and math.isfinite(float(score))
                and 0.0 <= float(score) <= 1.0
                else None
            ),
            "factors": facs,
        }
    return out
