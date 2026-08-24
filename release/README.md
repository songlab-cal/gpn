# Release review material

This directory contains executable-review artifacts, not credentials or a release
trigger.

- `main-ruleset.json` is the exact proposed GitHub repository-ruleset payload. It
  is not applied automatically.
- `governance-audit-2026-08-19.md` records the read-only settings and vulnerability
  audit performed during modernization.
- `external-mutations.json` is the machine-readable ledger of completed and
  approval-gated external actions.
- `external-mutations.schema.json` defines that ledger's contract.

Pending entries have one of two dispositions. `approval_ready` entries have exact
targets and payloads suitable for a maintainer decision. `deferred` entries still
have named blockers and require a separate future approval; approval of the
cumulative modernization review does not authorize them. Merge and final-release
actions become approval-ready only after the final review packet names the
cumulative PR that binds the exact tree and ordered component heads. `depends_on`
records rollout order, and every exact file-writing proposal is paired with an
immutable target head when the external system provides one.

The cumulative `0.9.0` review packet lives under `release/0.9.0/`. Its component
manifest is immutable, while the cumulative pull-request body binds review to the
final head commit and tree (a Git object cannot embed its own identity). Nothing
in this directory grants authorization to merge, publish, change repository
settings, or write to Hugging Face.
