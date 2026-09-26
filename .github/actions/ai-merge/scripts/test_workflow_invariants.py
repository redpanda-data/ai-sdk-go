"""Pins the isolation properties of the agent job in the enrolled repo's
workflow, so a future edit cannot quietly re-enable code execution. Skips
where there is no agent job (e.g. the canonical mechanism repo)."""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
WF = os.path.normpath(
    os.path.join(HERE, "..", "..", "..", "workflows", "ai-approved-merge.yml")
)


def _wf():
    if not os.path.exists(WF):
        return None
    s = open(WF).read()
    return s if "claude-code-action" in s else None


def test_agent_job_isolation_flags():
    s = _wf()
    if s is None:
        return
    for must in (
        "pull_request_target:",  # definitions come from the base branch
        "persist-credentials: false",  # PR checkout carries no token
        "--disallowedTools",
        "Bash",  # never executes anything
        "WebFetch",
        "WebSearch",  # no network egress via tools
        "mcp__github__",  # no GitHub-side actions via the Claude App
        "--setting-sources user",  # PR's .claude/settings.json hooks never load
        "--strict-mcp-config",  # PR's .mcp.json never spawns servers
        "CLAUDE_CODE_SAFE_MODE",  # CLAUDE.md/plugins/skills/hooks/MCP off
        "claude-code-action/base-action@",  # the bare harness: no GitHub bot behaviour, no MCP
        "prompt_file:",  # prompt from a file written by trusted base code
        "claude_env:",
        "--max-turns",
        "Read(./pr/**)",  # Read (and best-effort Grep/Glob/LS) scoped to the PR
        "Read(//proc/**)",  # ... and explicitly denied from process env
        "Read(~/.*/**)",  # dotfiles in HOME denied...
        "Read(~/work/_temp/**)",  # ...and the runner's own temp/actions dirs
        "Read(//etc/**)",
        "PR head moved",  # superseded runs skip via the LIVE head, not job results
        "checks: read",  # the CI gate queries check runs in the trusted job
        "ci_token:",
        "own_check_names:",
        "timeout-minutes:",
        "actions/upload-artifact@",
        "actions/download-artifact@",
    ):
        assert must in s, f"agent job must contain: {must}"


def test_agent_job_has_no_approval_credentials():
    s = _wf()
    if s is None:
        return
    agent = s.split("  agent:", 1)[1].split("  approve:", 1)[0]
    assert "AI_MERGE_APP_PRIVATE_KEY" not in agent and "AI_MERGE_APP_ID" not in agent
    assert "create-github-app-token" not in agent
    # the full claude-code-action (GitHub bot behaviour) must never be used here
    assert "uses: anthropics/claude-code-action@" not in agent


def test_approve_job_requires_agent_and_always_runs():
    s = _wf()
    if s is None:
        return
    approve = s.split("  approve:", 1)[1]
    assert "needs: agent" in approve and "always()" in approve
    assert "verdict_path:" in approve and "run_id:" in approve


def test_ci_wait_ignores_exactly_our_own_jobs():
    s = _wf()
    if s is None:
        return
    import re
    import yaml

    wf = yaml.safe_load(s)
    job_names = {j.get("name") for j in wf["jobs"].values()}
    ignored = set(re.findall(r'--ignore-check "([^"]+)"', s))
    assert ignored == job_names, (
        f"--ignore-check must list exactly the job names: {ignored} vs {job_names}"
    )


def test_model_call_is_gated_on_precheck_and_ci_wait_is_bounded():
    s = _wf()
    if s is None:
        return
    agent = s.split("  agent:", 1)[1].split("  approve:", 1)[0]
    assert "precheck.py" in agent
    run_agent = agent.split("- name: Run the review agent", 1)[1]
    assert "steps.precheck.outputs.eligible == 'true'" in run_agent.split("uses:", 1)[0]
    assert (
        "steps.precheck.outputs.engine == 'agent-action'"
        in run_agent.split("uses:", 1)[0]
    )
    assert "--wait-seconds" in agent and "timeout-minutes: 35" in agent
    # the CI wait must run for EVERY engine (single-call approvals depend on it)
    wait = agent.split("- name: Wait for CI", 1)[1].split("- name:", 1)[0]
    assert "steps.precheck.outputs.engine" not in wait
    # stale runs are detected from the live head, not from needs.agent.result
    assert "needs.agent.result" not in s


def test_deny_rules_do_not_cover_the_workspace():
    # The GitHub-hosted workspace is /home/runner/work/<repo>/<repo>. Deny rules
    # beat allow rules, so ANY deny that matches HOME or /home wholesale would
    # block Read(./pr/**) and the agent could read nothing (fails closed, but the
    # engine would abstain on every PR). Pin the interaction, not just presence.
    s = _wf()
    if s is None:
        return
    import re

    denies = re.search(r'--disallowedTools "([^"]+)"', s).group(1).split(",")
    reads = [d for d in denies if d.startswith("Read(")]
    blanket = [
        d
        for d in reads
        if d
        in (
            "Read(~/**)",
            "Read(//home/**)",
            "Read(//home/runner/**)",
            "Read(~/work/**)",
            "Read(./**)",
            "Read(**)",
        )
    ]
    assert not blanket, f"deny rules would block the workspace: {blanket}"
    for d in reads:
        inner = d[len("Read(") : -1]
        assert (
            inner.startswith("//")
            and not inner.startswith("//home")
            or inner.startswith("~/.")
            or inner.startswith("~/work/_")
            or inner.startswith("~/runners/")
            or inner.startswith("./.git")
            or inner.startswith("./pr/.git")
        ), f"unexpected deny shape: {d}"


def test_own_check_names_match_job_names():
    s = _wf()
    if s is None:
        return
    import re
    import yaml

    wf = yaml.safe_load(s)
    job_names = {j.get("name") for j in wf["jobs"].values()}
    m = re.search(r'own_check_names: "([^"]+)"', s)
    # '|'-separated: job names contain commas
    assert m and set(m.group(1).split("|")) == job_names, (m and m.group(1), job_names)
