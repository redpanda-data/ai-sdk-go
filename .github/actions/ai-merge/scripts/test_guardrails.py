"""Unit tests for the guardrail gate.
Run from this directory: `cd .github/actions/ai-merge/scripts && pytest`."""

from guardrails import evaluate

CFG = {
    "version": 1,
    "enabled": True,
    "excluded_paths": ["providers/**/auth*.go"],
    "max_changed_files": 50,
    "max_total_lines": 800,
    "min_confidence": 0.8,
    "dependency_paths": ["**/go.mod", "**/go.sum"],
}
PR_OK = {"author_association": "MEMBER", "number": 1}


def _files(*specs):
    return [
        {"filename": n, "status": "modified", "additions": a, "deletions": d}
        for n, a, d in specs
    ]


def test_happy_path_eligible():
    r = evaluate(
        CFG,
        _files(
            ("providers/bedrock/models.go", 27, 0), ("catalog/snapshot.json", 146, 1)
        ),
        PR_OK,
        True,
    )
    assert r["eligible"], r["reasons"]
    assert r["is_dependency"] is False


def test_missing_config_not_eligible():
    assert not evaluate({}, _files(("a.go", 1, 0)), PR_OK, False)["eligible"]


def test_config_must_have_version_and_boolean_enabled():
    no_version = {k: v for k, v in CFG.items() if k != "version"}
    assert not evaluate(no_version, _files(("a.go", 1, 0)), PR_OK, True)["eligible"]
    for bad in (False, "true", 1, None):
        assert not evaluate(
            {**CFG, "enabled": bad}, _files(("a.go", 1, 0)), PR_OK, True
        )["eligible"], f"enabled={bad!r} must not qualify"


def test_baseline_excludes_ci_and_iac():
    for name in (
        ".github/workflows/test.yaml",
        "terraform/main.tf",
        ".buildkite/pipeline.yml",
    ):
        r = evaluate(CFG, _files((name, 3, 0)), PR_OK, True)
        assert not r["eligible"], name


def test_baseline_excludes_root_level_infra():
    # fnmatch had no globstar; `**/` must match a bare root-level file.
    for name in ("Dockerfile", "main.tf", ".goreleaser.yml", "entrypoint.sh"):
        r = evaluate(CFG, _files((name, 3, 0)), PR_OK, True)
        assert not r["eligible"], f"{name} should be excluded"


def test_baseline_excludes_iam_auth_and_secrets():
    for name in (
        "iam/policy.json",
        "svc/auth/handler.go",
        "pkg/credentials.go",
        "config/app_secrets.yaml",
        "certs/server.pem",
        "deploy/tls.key",
    ):
        r = evaluate({**CFG, "excluded_paths": []}, _files((name, 1, 0)), PR_OK, True)
        assert not r["eligible"], f"{name} should be excluded by baseline"


def test_repo_pattern_with_mid_globstar_matches_direct_child():
    # `providers/**/auth*.go` must catch providers/auth.go, not only nested.
    for name in ("providers/auth_signer.go", "providers/bedrock/auth_signer.go"):
        r = evaluate(CFG, _files((name, 2, 0)), PR_OK, True)
        assert not r["eligible"], name


def test_collaborator_and_non_member_rejected():
    for assoc in ("COLLABORATOR", "CONTRIBUTOR", "NONE", ""):
        r = evaluate(CFG, _files(("a.go", 1, 0)), {"author_association": assoc}, True)
        assert not r["eligible"], assoc


def test_size_bounds():
    many = _files(*[(f"pkg/f{i}.go", 1, 0) for i in range(51)])
    assert not evaluate(CFG, many, PR_OK, True)["eligible"]
    assert not evaluate(CFG, _files(("big.go", 900, 0)), PR_OK, True)["eligible"]


def test_dependency_detection_root_and_nested():
    assert evaluate(CFG, _files(("go.mod", 2, 1)), PR_OK, True)["is_dependency"]
    assert evaluate(CFG, _files(("services/api/go.sum", 2, 1)), PR_OK, True)[
        "is_dependency"
    ]


def test_scalar_dependency_paths_is_coerced_not_exploded():
    # A YAML scalar must not become single-character patterns.
    cfg = {**CFG, "dependency_paths": "**/Cargo.lock"}
    assert evaluate(cfg, _files(("Cargo.lock", 1, 1)), PR_OK, True)["is_dependency"]
    assert not evaluate(cfg, _files(("a.go", 1, 1)), PR_OK, True)["is_dependency"]


def test_non_list_paths_config_fails_closed():
    r = evaluate({**CFG, "excluded_paths": 42}, _files(("a.go", 1, 0)), PR_OK, True)
    assert not r["eligible"]


def test_renamed_into_excluded_path():
    files = [
        {
            "filename": "terraform/x.tf",
            "previous_filename": "docs/x.md",
            "status": "renamed",
            "additions": 0,
            "deletions": 0,
        }
    ]
    assert not evaluate(CFG, files, PR_OK, True)["eligible"]


def test_renamed_manifest_still_counts_as_dependency_change():
    files = [
        {
            "filename": "deps.txt",
            "previous_filename": "go.mod",
            "status": "renamed",
            "additions": 1,
            "deletions": 1,
        }
    ]
    assert evaluate(CFG, files, PR_OK, True)["is_dependency"] is True
