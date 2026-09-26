"""Unit tests for the guardrail gate.
Run from this directory: `cd .github/actions/ai-merge/scripts && pytest`."""

from guardrails import evaluate

CFG = {
    "version": 1,
    "enabled": True,
    "excluded_paths": ["providers/**/auth*.go"],
    "min_confidence": 0.8,
    "require_ci_pass": False,  # CI gate has its own tests below
    "dependency_paths": ["**/go.mod", "**/go.sum"],
}
PR_OK = {
    "author_is_member": True,
    "membership_check_status": "204",
    "author": "alice",
    "number": 1,
}


def _files(*specs):
    return [
        {
            "filename": n,
            "status": "modified",
            "additions": a,
            "deletions": d,
            "has_patch": True,
        }
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
            "has_patch": True,
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
            "has_patch": True,
        }
    ]
    assert evaluate(CFG, files, PR_OK, True)["is_dependency"] is True


def test_membership_check_failure_is_distinguished_from_non_member():
    real_404 = {
        "author_is_member": False,
        "membership_check_status": "404",
        "author": "bob",
    }
    r = evaluate(CFG, _files(("a.go", 1, 0)), real_404, True)
    assert not r["eligible"] and any("is not a member" in x for x in r["reasons"])
    for status in ("403", "401", "429", "unknown"):
        broken = {
            "author_is_member": False,
            "membership_check_status": status,
            "author": "bob",
        }
        r = evaluate(CFG, _files(("a.go", 1, 0)), broken, True)
        assert not r["eligible"], status
        assert any("check" in x and "failed" in x for x in r["reasons"]), r["reasons"]


def test_opt_out_label_makes_pr_ineligible():
    r = evaluate(CFG, _files(("a.go", 1, 0)), PR_OK, True, skipped=True)
    assert not r["eligible"]
    assert any("opt-out" in x for x in r["reasons"])


def test_binary_or_patchless_file_is_ineligible():
    binary = [
        {
            "filename": "assets/logo.png",
            "status": "added",
            "additions": 0,
            "deletions": 0,
            "has_patch": False,
        }
    ]
    r = evaluate(CFG, binary, PR_OK, True)
    assert not r["eligible"]
    assert any("no reviewable patch" in x for x in r["reasons"])
    # A text file whose patch GitHub omitted for size is equally unreviewable.
    huge = [
        {
            "filename": "gen/big.go",
            "status": "modified",
            "additions": 5000,
            "deletions": 0,
            "has_patch": False,
        }
    ]
    assert not evaluate(CFG, huge, PR_OK, True)["eligible"]


def test_pure_deletion_and_zero_change_rename_stay_reviewable():
    files = [
        {
            "filename": "old.bin",
            "status": "removed",
            "additions": 0,
            "deletions": 0,
            "has_patch": False,
        },
        {
            "filename": "b.go",
            "previous_filename": "a.go",
            "status": "renamed",
            "additions": 0,
            "deletions": 0,
            "has_patch": False,
        },
    ]
    assert evaluate(CFG, files, PR_OK, True)["eligible"]


def test_non_numeric_config_values_fail_closed_with_reason():
    for key in ("min_confidence", "max_diff_chars"):
        r = evaluate({**CFG, key: "sixty"}, _files(("a.go", 1, 0)), PR_OK, True)
        assert not r["eligible"], key
        assert any(key in x and "number" in x for x in r["reasons"]), r["reasons"]
    r = evaluate({**CFG, "max_diff_chars": True}, _files(("a.go", 1, 0)), PR_OK, True)
    assert not r["eligible"]
    r = evaluate({**CFG, "max_diff_chars": 0}, _files(("a.go", 1, 0)), PR_OK, True)
    assert not r["eligible"]


def test_file_entry_without_filename_fails_closed():
    r = evaluate(
        CFG, [{"status": "added", "additions": 1, "deletions": 0}], PR_OK, True
    )
    assert not r["eligible"]


def test_out_of_range_min_confidence_fails_closed():
    for v in (95, -1, 1.5):
        r = evaluate({**CFG, "min_confidence": v}, _files(("a.go", 1, 0)), PR_OK, True)
        assert not r["eligible"], v
        assert any("between 0 and 1" in x for x in r["reasons"]), r["reasons"]


def test_missing_has_patch_is_treated_as_unreviewable():
    # The action always sets has_patch; if the projection ever drops it, fail closed.
    r = evaluate(
        CFG,
        [{"filename": "a.go", "status": "modified", "additions": 1, "deletions": 0}],
        PR_OK,
        True,
    )
    assert not r["eligible"]


GEN_CFG = {**CFG, "generated_paths": ["catalog/snapshot.json", "docs/generated/**"]}


def test_generated_files_are_split_out_as_signals():
    files = _files(
        ("providers/bedrock/models.go", 27, 0), ("catalog/snapshot.json", 6000, 5000)
    )
    r = evaluate(GEN_CFG, files, PR_OK, True)
    assert r["eligible"], r["reasons"]
    assert r["generated_files"] == ["catalog/snapshot.json"]
    assert r["reviewable_lines"] == 27 and r["generated_lines"] == 11000
    # Without the declaration the snapshot is reviewable; size alone never gates.
    r2 = evaluate(CFG, files, PR_OK, True)
    assert r2["eligible"] and r2["reviewable_lines"] == 11027


def test_excluded_path_wins_over_generated():
    cfg = {**GEN_CFG, "generated_paths": [".github/**"]}
    r = evaluate(cfg, _files((".github/workflows/x.yml", 3, 0)), PR_OK, True)
    assert not r["eligible"]
    assert r["generated_files"] == []


def test_generated_file_must_still_be_text():
    binary = [
        {
            "filename": "catalog/snapshot.json",
            "status": "modified",
            "additions": 0,
            "deletions": 0,
            "has_patch": False,
        }
    ]
    assert not evaluate(GEN_CFG, binary, PR_OK, True)["eligible"]


def test_composition_signals():
    files = _files(
        ("pkg/a.go", 10, 2), ("pkg/a_test.go", 30, 0), ("catalog/snapshot.json", 100, 0)
    )
    r = evaluate(GEN_CFG, files, PR_OK, True)
    assert r["test_lines"] == 30 and r["source_lines"] == 12
    assert r["tests_changed_with_source"] is True
    r2 = evaluate(GEN_CFG, _files(("pkg/a.go", 10, 2)), PR_OK, True)
    assert r2["tests_changed_with_source"] is False


def test_rename_into_generated_path_is_not_treated_as_generated():
    # A hand-written file moved INTO a generated path must stay reviewable.
    files = [
        {
            "filename": "docs/generated/x.md",
            "previous_filename": "docs/handwritten.md",
            "status": "renamed",
            "additions": 900,
            "deletions": 0,
            "has_patch": True,
        }
    ]
    cfg = {**CFG, "generated_paths": ["docs/generated/**"]}
    r = evaluate(cfg, files, PR_OK, True)
    assert r["generated_files"] == [] and r["reviewable_lines"] == 900
    assert r["eligible"]  # size never gates; the file is simply reviewable
    # A rename WITHIN generated paths stays generated.
    files[0]["previous_filename"] = "docs/generated/old.md"
    assert evaluate(cfg, files, PR_OK, True)["generated_files"] == [
        "docs/generated/x.md"
    ]


def test_non_finite_numeric_config_fails_closed_with_reason():
    for v in (float("inf"), float("-inf"), float("nan")):
        r = evaluate({**CFG, "max_diff_chars": v}, _files(("a.go", 1, 0)), PR_OK, True)
        assert not r["eligible"], v
        assert any("finite number" in x for x in r["reasons"]), r["reasons"]


def test_large_hand_written_change_is_not_refused_on_size():
    r = evaluate(CFG, _files(("pkg/big.go", 4000, 0)), PR_OK, True)
    assert r["eligible"] and r["reviewable_lines"] == 4000
    assert r["max_diff_chars"] == 120_000
    assert (
        evaluate({**CFG, "max_diff_chars": 5000}, _files(("a.go", 1, 0)), PR_OK, True)[
            "max_diff_chars"
        ]
        == 5000
    )


def test_engine_config_validated():
    r = evaluate(
        {
            **CFG,
            "engine": "agent-action",
            "shadow_engines": ["single-call", "agent-action"],
        },
        _files(("a.go", 1, 0)),
        PR_OK,
        True,
    )
    assert r["eligible"] and r["engine"] == "agent-action"
    assert r["shadow_engines"] == ["single-call"]  # primary is dropped from shadows
    assert evaluate(CFG, _files(("a.go", 1, 0)), PR_OK, True)["engine"] == "single-call"
    r = evaluate({**CFG, "engine": "adp"}, _files(("a.go", 1, 0)), PR_OK, True)
    assert not r["eligible"] and any("engine" in x for x in r["reasons"])
    r = evaluate(
        {**CFG, "shadow_engines": ["nope"]}, _files(("a.go", 1, 0)), PR_OK, True
    )
    assert not r["eligible"]


PASSED = [{"name": "Test", "status": "completed", "conclusion": "success"}]


def test_ci_gate_refuses_failed_pending_none_and_unknown():
    cfg = {**CFG, "require_ci_pass": True}
    ok = evaluate(cfg, _files(("a.go", 1, 0)), PR_OK, True, checks=PASSED)
    assert ok["eligible"] and ok["ci_status"] == "passed"
    cases = {
        "failed": [{"name": "Test", "status": "completed", "conclusion": "failure"}],
        "pending": [{"name": "Test", "status": "in_progress", "conclusion": None}],
        "none": [],
    }
    for expected, checks in cases.items():
        r = evaluate(cfg, _files(("a.go", 1, 0)), PR_OK, True, checks=checks)
        assert not r["eligible"] and r["ci_status"] == expected, (
            expected,
            r["reasons"],
        )
    r = evaluate(cfg, _files(("a.go", 1, 0)), PR_OK, True)  # checks not supplied
    assert not r["eligible"] and r["ci_status"] == "unknown"


def test_ci_gate_ignores_own_jobs_and_action_required():
    cfg = {**CFG, "require_ci_pass": True}
    checks = PASSED + [
        {
            "name": "gates + approval (trusted)",
            "status": "in_progress",
            "conclusion": None,
        },
        {"name": "Claude Code", "status": "completed", "conclusion": "action_required"},
    ]
    r = evaluate(
        cfg,
        _files(("a.go", 1, 0)),
        PR_OK,
        True,
        checks=checks,
        own_check_names={"gates + approval (trusted)"},
    )
    assert r["eligible"] and r["ci_status"] == "passed"


def test_ci_gate_can_be_disabled_only_by_explicit_bool():
    r = evaluate({**CFG, "require_ci_pass": "no"}, _files(("a.go", 1, 0)), PR_OK, True)
    assert not r["eligible"] and any("require_ci_pass" in x for x in r["reasons"])


def test_diff_size_gate_applies_to_every_engine():
    diff = (
        "diff --git a/pkg/a.go b/pkg/a.go\n--- a/pkg/a.go\n+++ b/pkg/a.go\n"
        + "+x\n" * 3000
    )
    cfg = {**CFG, "engine": "agent-action", "max_diff_chars": 1000}
    r = evaluate(cfg, _files(("pkg/a.go", 3000, 0)), PR_OK, True, diff=diff)
    assert not r["eligible"] and any(
        "too large to review in one pass" in x for x in r["reasons"]
    )
    assert r["reviewable_diff_chars"] == len(diff)
    ok = evaluate(
        {**cfg, "max_diff_chars": 100_000},
        _files(("pkg/a.go", 3000, 0)),
        PR_OK,
        True,
        diff=diff,
    )
    assert ok["eligible"]


def test_diff_size_gate_ignores_generated_hunks():
    gen = (
        "diff --git a/catalog/snapshot.json b/catalog/snapshot.json\n--- a/catalog/snapshot.json\n"
        "+++ b/catalog/snapshot.json\n" + "+g\n" * 5000
    )
    small = "diff --git a/pkg/a.go b/pkg/a.go\n--- a/pkg/a.go\n+++ b/pkg/a.go\n+x\n"
    cfg = {**CFG, "generated_paths": ["catalog/snapshot.json"], "max_diff_chars": 500}
    files = _files(("pkg/a.go", 1, 0), ("catalog/snapshot.json", 5000, 0))
    r = evaluate(cfg, files, PR_OK, True, diff=gen + small)
    assert r["eligible"], r["reasons"]
    assert r["reviewable_diff_chars"] == len(small)


def test_ci_gate_with_required_checks_rejects_bot_only_pass():
    cfg = {**CFG, "require_ci_pass": True, "ci_checks": ["Test", "Golangci Lint"]}
    bot_only = [
        {"name": "claude-review", "status": "completed", "conclusion": "success"}
    ]
    r = evaluate(cfg, _files(("a.go", 1, 0)), PR_OK, True, checks=bot_only)
    assert not r["eligible"] and r["ci_status"] == "none"
    full = bot_only + [
        {"name": "Test", "status": "completed", "conclusion": "success"},
        {"name": "Golangci Lint", "status": "completed", "conclusion": "success"},
    ]
    assert evaluate(cfg, _files(("a.go", 1, 0)), PR_OK, True, checks=full)["eligible"]


def test_diff_unavailable_is_a_size_refusal_not_a_crash():
    r = evaluate(CFG, _files(("a.go", 1, 0)), PR_OK, True, diff_unavailable=True)
    assert not r["eligible"] and any("too large to review" in x for x in r["reasons"])
