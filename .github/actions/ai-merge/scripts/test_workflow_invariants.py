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
        "Read(~/**)",
        "Read(//etc/**)",
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
    assert "--wait-seconds" in agent and "timeout-minutes: 35" in agent
