# AI-approved merge mechanism — VENDORED COPY

> **Canonical source:** maintained internally by Redpanda DevProd. This directory is a
> verbatim copy plus the hardening from the review rounds on this repo's enrollment PR.
>
> **What the bot does:** posts a binding approval on eligible low-risk PRs. It never
> merges. The author merges, or enables GitHub's own auto-merge per PR.
>
> **Judgment engine:** `agent-action` — Claude Code (the Agent SDK harness) runs in an
> unprivileged job via the bare `claude-code-action/base-action` (not the GitHub-bot
> action), after a cheap precheck and after waiting for the repo's CI to finish. It reads
> the PR with Read/Glob/Grep/LS scoped to the PR checkout (no shell, no network tools, no
> GitHub tools or tokens, no hooks/MCP/CLAUDE.md from the PR, process env and system paths
> denied), sees the CI outcome, and writes one verdict file. The trusted job validates it
> (same run, same head SHA, schema, evidence paths, no secret-shaped strings, comment-safe
> text) before the deterministic gates decide. Contract and the swap path to other engines
> (ADP later): `../docs/verdict-contract.md`. `shadow_engines` records the single-call
> verdict alongside for comparison; it never decides.
>
> This repo is public and GitHub does not allow public repos to use reusable
> workflows or actions from a private repo, so the mechanism is vendored here and
> executed from the PR's **base ref** (default branch only). Do not edit this copy
> directly: change it in the canonical copy, then re-sync (`rsync -a --delete` from a
> checkout of the canonical copy into `.github/actions/ai-merge/`), re-apply this note
> and the `action.yml` header.
>
> **Enrolling another repo:** Redpanda private repos should call the canonical reusable
> workflow directly (ask DevProd) rather than vendor. Public repos vendor as done here:
> copy `.github/actions/ai-merge/`, `.github/workflows/ai-approved-merge.yml` and
> `test-ai-merge.yml`, add `.github/ai-merge.yml`, the three secrets, the App
> installation, the `ai-merge-skip` label, and a ruleset (1 approval + dismiss stale
> approvals on push).
>
> Outsider PRs on public repos are excluded twice: fork PRs never run, and the author
> must be a verified org member (and could not merge anyway). GitHub App bots
> (dependabot/renovate) are skipped at the gate, so dependency bumps are in scope when
> authored by humans or by bot USER accounts that are org members.


Decision logic for the reusable workflow `.github/workflows/ai-approved-merge.yml`.
Every enrolled repo shares this one implementation.

**Model:** the *repo* opts in (caller workflow + `.github/ai-merge.yml`); every PR
is then evaluated automatically. A PR opts *out* with the `ai-merge-skip` label.

## Pipeline

`guardrails.py` (eligibility: config valid, org-member author, no excluded paths,
reviewable files) → `review.py` (Anthropic API, versioned prompt, strict JSON verdict;
supply-chain checklist on dependency PRs) → `decide.py` (approve only if eligible ∧
verdict=approve ∧ reviewed_fully ∧ confidence valid and ≥ threshold ∧ supply chain
affirmatively clean) → approval pinned to the reviewed SHA (the bot never merges; the author does) →
`audit.py` (sticky comment; also the review body).

## Fail-closed guarantees

- Enrolled repos trigger on **`pull_request_target`**, never `pull_request`: the
  workflow that holds the App key must come from the base branch, not the PR.
  Safe because the PR head is never checked out or executed.
- The gate only ever **adds** an approval; it never blocks a PR.
- Binary or patchless files (no reviewable diff) make a PR ineligible.
- **Size is not a risk score and does not gate.** The only size rule is that the whole
  reviewable diff must fit in one review (`max_diff_chars`, default 120K chars,
  per-repo), and the model must set `reviewed_fully: true` — an approval is never
  given without it. A repo may declare `generated_paths`: machine-produced files that a
  **required** CI check regenerates and compares (so a hand edit cannot merge). Those
  are omitted from the diff the model sees; the model is told they changed. An excluded
  path always wins over a generated path; generated files must still be text. Counts
  (files, reviewable / generated / test / source lines, tests-changed-with-source) and
  the model's `scope` are recorded in every audit as inputs for a future risk-scoring
  layer.
- The `ai-merge-skip` opt-out is handled inside the run: adding it after an
  approval withdraws the bot's approval.
- Config is read from the **base ref**; a PR cannot relax its own guardrails.
- Any error (API, malformed model JSON, malformed config, missing file) yields a
  non-approving result and still posts the audit comment.
- Baseline exclusions apply in every repo: CI/CD, IaC, release/container tooling,
  `iam/` and `auth/` dirs, credential/secret/key files. Globs have real globstar
  semantics (`**/` = zero or more dirs) — see `common.glob_to_regex`.
- Org membership is verified via the App token (`GET /orgs/{org}/members/{login}`),
  never from `author_association`, which misreports members with private
  membership. The App therefore needs **Organization → Members: Read-only**.
- Approvals are minted by the **ai-merge GitHub App** (the Actions token cannot
  approve PRs) and are pinned to the reviewed commit. **The bot never merges**:
  the approval satisfies the ruleset's review requirement and the author merges
  (or enables GitHub's own auto-merge). App permissions: pull_requests write,
  contents read, members read. The enrolling ruleset **must** enable "dismiss
  stale approvals on new commits".
- `dry_run` defaults **true**.

## Tests

Run from this directory (bare imports resolve here); CI runs them via
`.github/workflows/test-ai-merge.yml`.

```bash
cd .github/actions/ai-merge/scripts && pip install -r requirements.txt pytest && pytest -q
```

## Versioning

Bump the enrolled repo's config `version` on any guardrail change and
`review.py:PROMPT_VERSION` on any prompt/schema change; both appear in every audit
record. Callers pin `@ai-merge/v1`; the reusable workflow references the action at
the same tag, so tagging one commit versions both together. A dedicated
osv-scanner pass is deferred to phase 2 (needs a read-only checkout).
