# Verdict contract (schema 1)

The AI-approved-merge mechanism separates **judgment** from **enforcement**.
Judgment engines are pluggable; enforcement never changes. This document is the
seam: any engine that produces a verdict matching this contract can be selected
with `engine:` in a repo's config. Current engines: `single-call` (one model call
in the trusted job), `agent-action` (Claude Code via `claude-code-action/base-action` in an
unprivileged job). A future `adp` engine would receive the same request envelope
over an authenticated channel and return the same verdict.

## Trust model an engine must live inside

- The engine **never** receives approval credentials and cannot approve.
- The engine may **read** PR content; it must never **execute** it.
- Enforcement (org membership, excluded paths, reviewable files, config,
  confidence threshold, supply-chain rule) runs in the trusted job and does not
  depend on the engine's honesty. A compromised engine can at most produce a
  wrong verdict; it cannot bypass a gate.
- The trusted job validates every verdict with `scripts/validate_verdict.py`
  before using it. Anything invalid becomes a non-approving verdict with the
  reason recorded in the audit.

## Request envelope (trusted → engine)

```json
{"schema":"1","repo":"owner/name","pr":123,"title":"…","body":"…",
 "head_sha":"…","base_sha":"…","author":"login",
 "changed_files":[{"path":"…","status":"added|modified|removed|renamed",
                   "additions":0,"deletions":0,"previous_filename":null}],
 "ci_checks":[{"name":"…","status":"completed","conclusion":"success"}],
 "config":{"excluded_paths":[…],"generated_paths":[…],"dependency_paths":[…],
           "prompt_profile":"code"}}
```

## Verdict (engine → trusted)

```json
{"verdict":"approve|request_changes|comment",
 "confidence":0.0,               // finite number in [0,1]; bool/str rejected
 "reviewed_fully":true,          // real bool; approval requires true
 "scope":"mechanical|localized|broad",
 "summary":"…","concerns":["…"],
 "supply_chain":{"checked":bool,"new_maintainers":bool|null,
                 "unusual_version_jump":bool|null,"added_install_scripts":bool|null},
 "evidence":[{"path":"repo/relative","line":12,"note":"…"}],   // approve ⇒ ≥1
 "risk":{"score":0.0,"factors":[{"name":"…","weight":0.0,"evidence":"…"}]}  // optional
}
```

## Metadata (produced by the trusted side of the engine job, never by the model)

```json
{"engine":"agent-action@<action-sha>|adp-pr-review@<ver>|single-call@<prompt-ver>",
 "model":"…","head_sha":"…","run_id":"<workflow run id>","turns":12}
```

## Validation rules (fail closed)

1. `meta.head_sha` equals the PR head at approval time (else: head moved).
2. `meta.run_id` equals the current workflow run (same-run provenance).
3. Field types and enums exactly as above.
4. Every `evidence.path` exists in the PR tree at head or is a changed file.
5. `approve` requires at least one `evidence` entry.
6. Decision rule (unchanged): approve only if eligible ∧ verdict=approve ∧
   reviewed_fully ∧ confidence ≥ threshold ∧ supply chain affirmatively clean.

## Shadow engines

`shadow_engines: [single-call]` runs additional engines for the audit record
only. Their verdicts never decide. Use it to compare engines on real PRs before
switching `engine:`.
