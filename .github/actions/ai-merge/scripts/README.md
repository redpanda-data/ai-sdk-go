# AI-approved merge mechanism (DEVPROD-4812) — VENDORED COPY

> **Canonical source:** `redpanda-data/devprod-infra` → `.github/actions/ai-merge`
> **Vendored from commit:** `6f3fd35`
>
> This repo is public and GitHub does not allow public repos to use reusable
> workflows or actions from a private repo, so the mechanism is vendored here and
> executed from the PR's **base ref**. Do not edit this copy directly: change it in
> devprod-infra, then re-sync with
> `rsync -a --delete ../devprod-infra/.github/actions/ai-merge/ .github/actions/ai-merge/`
> and update the commit above.
>
> **Enrolling another repo the same way:** copy `.github/actions/ai-merge/`,
> `.github/workflows/ai-approved-merge.yml` and `.github/workflows/test-ai-merge.yml`;
> add `.github/ai-merge.yml`; set secrets `AI_MERGE_APP_ID`, `AI_MERGE_APP_PRIVATE_KEY`,
> `ANTHROPIC_API_KEY`; install the App on the repo; create the `ai-merge-skip` label;
> ruleset: require 1 approval + dismiss stale approvals on push.


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
