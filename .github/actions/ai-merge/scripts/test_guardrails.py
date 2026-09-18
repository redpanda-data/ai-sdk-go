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
PR_OK = {"author_is_member": True, "membership_check_status": "204", "author": "alice", "number": 1}


def _files(*specs):
    return [
        {"filename": n, "status": "modified", "additions": a, "deletions": d}
        for n, a, d in specs
    ]


def test_happy_path_eligible():
    r = evaluate(
        CFG,
        _files(("providers/bedrock/models.go", 27, 0), ("catalog/snapshot.json", 146, 1)),
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
        assert not evaluate({**CFG, "enabled": bad}, _files(("a.go", 1, 0)), PR_OK, True)[
            "eligible"
        ], f"enabled={bad!r} must not qualify"


def test_baseline_excludes_ci_and_iac():
    for name in (".github/workflows/test.yaml", "terraform/main.tf", ".buildkite/pipeline.yml"):
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


def test_non_member_rejected_regardless_of_association():
    # Membership comes from the API check, never from author_association:
    # even a payload saying MEMBER must not qualify without author_is_member.
    for pr in (
        {"author_is_member": False, "author_association": "MEMBER"},
        {"author_is_member": None},
        {"author_is_member": "true"},
        {"author_association": "OWNER"},
        {},
    ):
        r = evaluate(CFG, _files(("a.go", 1, 0)), pr, True)
        assert not r["eligible"], pr


def test_size_bounds():
    many = _files(*[(f"pkg/f{i}.go", 1, 0) for i in range(51)])
    assert not evaluate(CFG, many, PR_OK, True)["eligible"]
    assert not evaluate(CFG, _files(("big.go", 900, 0)), PR_OK, True)["eligible"]


def test_dependency_detection_root_and_nested():
    assert evaluate(CFG, _files(("go.mod", 2, 1)), PR_OK, True)["is_dependency"]
    assert evaluate(CFG, _files(("services/api/go.sum", 2, 1)), PR_OK, True)["is_dependency"]


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


def test_membership_check_failure_is_distinguished_from_non_member():
    real_404 = {"author_is_member": False, "membership_check_status": "404", "author": "bob"}
    r = evaluate(CFG, _files(("a.go", 1, 0)), real_404, True)
    assert not r["eligible"] and any("is not a member" in x for x in r["reasons"])
    for status in ("403", "401", "429", "unknown"):
        broken = {"author_is_member": False, "membership_check_status": status, "author": "bob"}
        r = evaluate(CFG, _files(("a.go", 1, 0)), broken, True)
        assert not r["eligible"], status
        assert any("check" in x and "failed" in x for x in r["reasons"]), r["reasons"]


def test_opt_out_label_makes_pr_ineligible():
    r = evaluate(CFG, _files(("a.go", 1, 0)), PR_OK, True, skipped=True)
    assert not r["eligible"]
    assert any("opt-out" in x for x in r["reasons"])


def test_binary_or_patchless_file_is_ineligible():
    binary = [{"filename": "assets/logo.png", "status": "added", "additions": 0,
               "deletions": 0, "has_patch": False}]
    r = evaluate(CFG, binary, PR_OK, True)
    assert not r["eligible"]
    assert any("no reviewable patch" in x for x in r["reasons"])
    # A text file whose patch GitHub omitted for size is equally unreviewable.
    huge = [{"filename": "gen/big.go", "status": "modified", "additions": 5000,
             "deletions": 0, "has_patch": False}]
    assert not evaluate({**CFG, "max_total_lines": 10000}, huge, PR_OK, True)["eligible"]


def test_pure_deletion_and_zero_change_rename_stay_reviewable():
    files = [
        {"filename": "old.bin", "status": "removed", "additions": 0, "deletions": 0,
         "has_patch": False},
        {"filename": "b.go", "previous_filename": "a.go", "status": "renamed",
         "additions": 0, "deletions": 0, "has_patch": False},
    ]
    assert evaluate(CFG, files, PR_OK, True)["eligible"]


def test_non_numeric_config_values_fail_closed_with_reason():
    for key in ("max_changed_files", "max_total_lines", "min_confidence"):
        r = evaluate({**CFG, key: "sixty"}, _files(("a.go", 1, 0)), PR_OK, True)
        assert not r["eligible"], key
        assert any(key in x and "number" in x for x in r["reasons"]), r["reasons"]
    r = evaluate({**CFG, "max_total_lines": True}, _files(("a.go", 1, 0)), PR_OK, True)
    assert not r["eligible"]


def test_file_entry_without_filename_fails_closed():
    r = evaluate(CFG, [{"status": "added", "additions": 1, "deletions": 0}], PR_OK, True)
    assert not r["eligible"]
