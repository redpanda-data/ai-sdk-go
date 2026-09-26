"""Guards THIS repo's real `.github/ai-merge.yml` against drift: the file most
likely to rot as the layout changes, and the only one nothing else exercised.
Skips (passes) where no repo config exists, e.g. the canonical mechanism repo."""

import os

import yaml

from guardrails import evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.normpath(os.path.join(HERE, "..", "..", "..", "ai-merge.yml"))
PR_OK = {"author_is_member": True, "membership_check_status": "204", "author": "a"}


def _cfg():
    if not os.path.exists(CONFIG):
        # Canonical mechanism repo has no repo config: skip VISIBLY under pytest
        # (rather than a silent green), and fall back gracefully without it.
        try:
            import pytest

            pytest.skip("no repo config at " + CONFIG)
        except ImportError:
            return None
    return yaml.safe_load(open(CONFIG))


def _one(path, add=3, dele=0):
    return [
        {
            "filename": path,
            "status": "modified",
            "additions": add,
            "deletions": dele,
            "has_patch": True,
        }
    ]


def test_real_config_excludes_credential_and_ci_defining_files():
    cfg = _cfg()
    if cfg is None:
        return
    for path in (
        "providers/openai/provider.go",
        "providers/bedrock/provider.go",
        "providers/bedrock/mantle.go",
        "tool/mcp/transport.go",
        "tool/builtin/webfetch/dial.go",
        ".claude/settings.json",
        ".claude-pr/.claude/settings.json",
        "CLAUDE.md",
        "providers/foo/internal/provider.go",
        "Taskfile.yaml",
        "taskfiles/install.yaml",
        ".golangci.yaml",
        "LICENSE",
        "header.txt",
    ):
        r = evaluate(cfg, _one(path), PR_OK, True)
        assert not r["eligible"], f"{path} must be excluded: {r['reasons']}"


def test_real_config_keeps_catalog_work_in_scope():
    cfg = _cfg()
    if cfg is None:
        return
    for path in (
        "catalog/snapshot.json",
        "providers/bedrock/models.go",
        "providers/vertex/models.go",
        "providers/bedrock/cross_region.go",
    ):
        r = evaluate(cfg, _one(path), PR_OK, True)
        assert r["eligible"], f"{path} should be eligible: {r['reasons']}"


def test_real_config_flags_dependency_changes():
    cfg = _cfg()
    if cfg is None:
        return
    assert evaluate(cfg, _one("go.mod", 1, 1), PR_OK, True)["is_dependency"]
    assert evaluate(cfg, _one("examples/x/go.sum", 2, 2), PR_OK, True)["is_dependency"]
