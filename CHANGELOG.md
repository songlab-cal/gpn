# Changelog

GPN follows Semantic Versioning. Package releases are distinct from immutable
Hugging Face model revisions.

## Unreleased

### Added

- Published-model fixtures and offline scientific regression tests for every
  supported model family.
- Canonical prepared-data training recipes for GPN and GPN-Star.
- Explicit AutoClass registration, a stable `gpn` CLI, uv-managed development,
  Ruff, incremental mypy and jaxtyping, offline CI, and built-wheel checks.
- Sphinx/MyST documentation prepared for Read the Docs.
- A maintainer release runbook, reviewable GitHub ruleset proposal, external-
  mutation manifest, and pull-request checklist.

### Changed

- The maintained package now uses the `src/gpn` layout and explicit dependency
  extras.
- GPN-MSA is deprecated and supported for inference only.
- The minimum Transformers version is 5.5, the first release that resolves all
  known 2026 Transformers code-execution advisories affecting older supported
  versions.

### Removed

- Historical paper analyses, dataset-building workflows, GPN-MSA training, and
  retired notebooks from `main`. They remain available at the prepared
  `analysis-archive-2026-08-18` tag, pending final publication approval.

## 0.9.0a1 — 2026-08-18

- Claimed the `gpn` PyPI distribution with an installable alpha package.
- Added PEP 621 metadata, a canonical Python 3.13 and Transformers 5.15.0
  environment, explicit Transformers AutoClass registration, and Trusted
  Publishing through GitHub Actions.
