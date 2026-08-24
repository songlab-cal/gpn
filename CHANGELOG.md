# Changelog

GPN follows Semantic Versioning. Package releases are distinct from immutable
Hugging Face model revisions.

## 0.9.0 — Unreleased

### Added

- Published-model fixtures and offline scientific regression tests for every
  supported model family and the Sorghum GPN fine-tune.
- Canonical prepared-data training recipes for GPN and GPN-Star.
- Explicit AutoClass registration, a stable `gpn` CLI, uv-managed development,
  Ruff, package-wide mypy and jaxtyping, offline CI, and built-wheel checks.
- A consistent family-first CLI for GPN, GPN-MSA, and GPN-Star inference.
- Sphinx/MyST documentation prepared for Read the Docs.
- A maintainer release runbook, reviewable GitHub ruleset proposal, external-
  mutation manifest, and pull-request checklist.

### Changed

- The maintained package now uses the `src/gpn` layout and explicit dependency
  extras.
- GPN-MSA is deprecated and supported for inference only.
- Published legacy GPN checkpoint classes now carry concrete config, tensor-shape,
  and Transformers output types instead of blanket `Any` annotations.
- The supported runtime is Python 3.13 with Transformers 5.15.0, matching the
  environment used for the committed scientific fixtures.

### Fixed

- Center-window embedding pooling now uses exactly the requested number of tokens,
  including odd and single-token windows, and rejects oversized windows.
- GPN-Star embedding output now removes the singleton target-species axis before
  DataFrame serialization and reports an error for multiple targets.
- Single-sequence GPN logits now retrieve the correct interval for odd genomic
  window sizes and report malformed genome windows explicitly.

### Removed

- Historical paper analyses, dataset-building workflows, GPN-MSA training, and
  retired notebooks from `main`. They remain available at the prepared
  `analysis-archive-2026-08-18` tag, pending final publication approval.
- The unsupported auxiliary-feature disabling flag, which represented an ablation
  rather than a maintained inference mode.

## 0.9.0a1 — 2026-08-18

- Claimed the `gpn` PyPI distribution with an installable alpha package.
- Added PEP 621 metadata, a canonical Python 3.13 and Transformers 5.15.0
  environment, explicit Transformers AutoClass registration, and Trusted
  Publishing through GitHub Actions.
