# AI-approved merge mechanism (DEVPROD-4812) — VENDORED COPY

> **Canonical source:** `redpanda-data/devprod-infra` → `.github/actions/ai-merge`
> **Vendored from commit:** `da8d5b7` (tag `ai-merge/v1.1.0`)
>
> This repo is public and GitHub does not allow public repos to use reusable
> workflows or actions from a private repo, so the mechanism is vendored here and
> executed from the PR's **base ref**. Do not edit this copy directly: change it in
> devprod-infra, then re-sync with
> `rsync -a --delete ../devprod-infra/.github/actions/ai-merge/ .github/actions/ai-merge/`,
> re-apply this note and the `action.yml` header, and update the commit above.
>
> **Enrolling another repo:**
> - **Private repo (most repos): do NOT vendor.** Add a thin caller workflow that uses
>   `redpanda-data/devprod-infra/.github/workflows/ai-approved-merge.yml@ai-merge/v1`
>   plus `.github/ai-merge.yml`, the three secrets, App installation, the `ai-merge-skip`
>   label, and a ruleset (1 approval + dismiss stale approvals on push).
> - **Public repo:** GitHub blocks public→private reusable workflows, so vendor as done
>   here: copy `.github/actions/ai-merge/`, `.github/workflows/ai-approved-merge.yml`
>   and `test-ai-merge.yml`, then the same config/secrets/App/label/ruleset steps.
>
> Outsider PRs on public repos are excluded twice: fork PRs never run, and the author
> must be a verified org member.

Decision logic for the reusable workflow `.github/workflows/ai-approved-merge.yml`.
Every enrolled repo shares this one implementation.

**Model:** the *repo* opts in (caller workflow + `.github/ai-merge.yml`); every PR
is then evaluated automatically. A PR opts *out* with the `ai-merge-skip` label.

## Pipeline

`guardrails.py` (eligibility: config valid, org-member author, no excluded paths,
size bounds) → `review.py` (Anthropic API, versioned prompt, strict JSON verdict;
supply-chain checklist on dependency PRs) → `decide.py` (approve only if eligible ∧
verdict=approve ∧ confidence valid and ≥ threshold ∧ supply chain affirmatively
clean) → approve pinned to the reviewed SHA + `--match-head-commit` auto-merge →
`audit.py` (sticky comment; also the review body).

## Fail-closed guarantees

- The gate only ever **adds** an approval; it never blocks a PR.
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
  approve PRs) and are pinned to the reviewed commit. The enrolling ruleset
  **must** enable "dismiss stale approvals on new commits".
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
