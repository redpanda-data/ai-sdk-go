"""validate_verdict.py: the seam that makes an untrusted engine's output safe."""

import json
import os
import subprocess
import sys
import tempfile

from validate_verdict import validate

HEAD = "a" * 40
RUN = "123456"
TREE = {"pkg/a.go", "pkg/a_test.go", "go.mod", "README.md"}
CHANGED = {"pkg/a.go", "pkg/a_test.go"}
META = {
    "engine": "agent-action@deadbeef",
    "model": "claude-opus-5",
    "head_sha": HEAD,
    "run_id": RUN,
}
GOOD = {
    "verdict": "approve",
    "confidence": 0.92,
    "reviewed_fully": True,
    "scope": "localized",
    "summary": "Adds a nil check and a test.",
    "concerns": [],
    "supply_chain": {"checked": False},
    "evidence": [{"path": "pkg/a.go", "line": 12, "note": "nil guard"}],
}


def _v(verdict=GOOD, meta=META, head=HEAD, run=RUN, tree=TREE, changed=CHANGED):
    return validate(verdict, meta, head, run, tree, changed)


def test_good_verdict_passes_and_is_normalised():
    r = _v()
    assert (
        r["verdict"] == "approve" and r["reviewed_fully"] is True and "error" not in r
    )
    assert r["engine"] == "agent-action@deadbeef"


def test_head_sha_mismatch_rejected():
    r = _v(meta={**META, "head_sha": "b" * 40})
    assert r["verdict"] != "approve" and "head_sha" in r["error"]


def test_wrong_run_rejected():
    r = _v(meta={**META, "run_id": "999"})
    assert r["verdict"] != "approve" and "workflow run" in r["error"]


def test_missing_engine_stamp_rejected():
    r = _v(meta={k: v for k, v in META.items() if k != "engine"})
    assert r["verdict"] != "approve" and "engine" in r["error"]


def test_forged_shapes_rejected():
    for bad in (
        {**GOOD, "confidence": True},
        {**GOOD, "confidence": "0.95"},
        {**GOOD, "confidence": float("nan")},
        {**GOOD, "confidence": 1.5},
        {**GOOD, "reviewed_fully": "true"},
        {**GOOD, "reviewed_fully": 1},
        {**GOOD, "verdict": "APPROVE"},
        {**GOOD, "scope": "tiny"},
        {**GOOD, "summary": ""},
        {**GOOD, "concerns": "none"},
        {**GOOD, "supply_chain": []},
        "approve",
        None,
        [],
    ):
        r = _v(verdict=bad)
        assert r["verdict"] != "approve" and r["confidence"] == 0.0, bad


def test_evidence_must_be_in_tree_and_required_for_approve():
    r = _v(verdict={**GOOD, "evidence": [{"path": "secrets/prod.key", "note": "x"}]})
    assert r["verdict"] != "approve" and "not in the PR tree" in r["error"]
    r = _v(verdict={**GOOD, "evidence": []})
    assert r["verdict"] != "approve" and "at least one evidence" in r["error"]
    # non-approve verdicts do not need evidence
    r = _v(verdict={**GOOD, "verdict": "comment", "evidence": []})
    assert r["verdict"] == "comment" and "error" not in r
    # a deleted file is a changed file: citing it is fine
    r = _v(
        verdict={**GOOD, "evidence": [{"path": "pkg/gone.go", "note": "removed"}]},
        changed=CHANGED | {"pkg/gone.go"},
    )
    assert "error" not in r


def test_bad_evidence_line_rejected():
    for line in (0, -1, "12", True):
        r = _v(
            verdict={
                **GOOD,
                "evidence": [{"path": "pkg/a.go", "line": line, "note": "x"}],
            }
        )
        assert r["verdict"] != "approve", line


def test_cli_missing_files_fail_closed():
    with tempfile.TemporaryDirectory() as td:
        files = os.path.join(td, "files.json")
        json.dump([{"filename": "pkg/a.go"}], open(files, "w"))
        tree = os.path.join(td, "tree.txt")
        open(tree, "w").write("pkg/a.go\n")
        out = os.path.join(td, "out.json")
        subprocess.run(
            [
                sys.executable,
                os.path.join(os.path.dirname(__file__), "validate_verdict.py"),
                "--verdict",
                "/nonexistent/v.json",
                "--meta",
                "/nonexistent/m.json",
                "--head-sha",
                HEAD,
                "--run-id",
                RUN,
                "--tree",
                tree,
                "--files",
                files,
                "--out",
                out,
            ],
            check=True,
            capture_output=True,
        )
        r = json.load(open(out))
        assert (
            r["verdict"] == "comment"
            and r["confidence"] == 0.0
            and "could not read" in r["error"]
        )


def test_cli_good_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        v = os.path.join(td, "v.json")
        json.dump(GOOD, open(v, "w"))
        m = os.path.join(td, "m.json")
        json.dump(META, open(m, "w"))
        files = os.path.join(td, "files.json")
        json.dump([{"filename": p} for p in CHANGED], open(files, "w"))
        tree = os.path.join(td, "tree.txt")
        open(tree, "w").write("\n".join(TREE) + "\n")
        out = os.path.join(td, "out.json")
        subprocess.run(
            [
                sys.executable,
                os.path.join(os.path.dirname(__file__), "validate_verdict.py"),
                "--verdict",
                v,
                "--meta",
                m,
                "--head-sha",
                HEAD,
                "--run-id",
                RUN,
                "--tree",
                tree,
                "--files",
                files,
                "--out",
                out,
            ],
            check=True,
            capture_output=True,
        )
        assert json.load(open(out))["verdict"] == "approve"


def test_secret_in_any_field_rejects_and_is_not_echoed():
    key = "sk-ant-api03-" + "x" * 40
    for field, value in (
        ("summary", f"looks fine, btw {key}"),
        ("concerns", [f"env had {key}"]),
        ("evidence", [{"path": "pkg/a.go", "note": key}]),
        ("risk", {"score": 0.1, "factors": [{"name": "n", "evidence": key}]}),
    ):
        r = _v(verdict={**GOOD, field: value})
        assert r["verdict"] != "approve" and r["confidence"] == 0.0
        assert "secret-shaped" in r["error"] and "anthropic-api-key" in r["error"]
        assert key not in json.dumps(r) and "sk-ant" not in json.dumps(r), (
            "secret must never be echoed, not even a prefix"
        )
    for tok in (
        "ghp_" + "a" * 36,
        "AKIAABCDEFGHIJKLMNOP",
        "-----BEGIN RSA PRIVATE KEY-----",
        "ANTHROPIC_API_KEY=abcdefgh",
        "/proc/self/environ",
    ):
        assert _v(verdict={**GOOD, "summary": f"x {tok} y"})["verdict"] != "approve", (
            tok
        )


def test_free_text_is_comment_safe():
    r = _v(
        verdict={
            **GOOD,
            "summary": "see https://evil.example/p?a=1 and ping @weeco <img src=x onerror=alert(1)>",
            "concerns": ["a" * 5000],
            "evidence": [
                {"path": "pkg/a.go", "line": 1, "note": "<b>x</b> www.example.com"}
            ],
        }
    )
    assert "https://" not in r["summary"] and "[link removed]" in r["summary"]
    assert (
        "@weeco" not in r["summary"] and "weeco" in r["summary"]
    )  # zero-width space inserted
    assert "<img" not in r["summary"] and "&lt;img" in r["summary"]
    assert len(r["concerns"][0]) <= 500
    assert (
        "<b>" not in r["evidence"][0]["note"] and "www." not in r["evidence"][0]["note"]
    )


def test_list_lengths_are_bounded():
    r = _v(
        verdict={
            **GOOD,
            "concerns": [f"c{i}" for i in range(100)],
            "evidence": [{"path": "pkg/a.go", "note": f"e{i}"} for i in range(100)],
        }
    )
    assert len(r["concerns"]) == 15 and len(r["evidence"]) == 15


def test_evidence_paths_with_pr_prefix_are_normalised():
    for spelled in ("pr/pkg/a.go", "./pr/pkg/a.go", "./pkg/a.go"):
        r = _v(
            verdict={**GOOD, "evidence": [{"path": spelled, "line": 3, "note": "x"}]}
        )
        assert "error" not in r, (spelled, r.get("error"))
        assert r["evidence"][0]["path"] == "pkg/a.go"
    # a real file literally named pr/... in the repo still resolves to itself
    r = _v(
        verdict={**GOOD, "evidence": [{"path": "pr/real.go", "note": "x"}]},
        tree=TREE | {"pr/real.go"},
    )
    assert r["evidence"][0]["path"] == "pr/real.go"
