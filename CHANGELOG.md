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

### Changed

- The maintained package now uses the `src/gpn` layout and explicit dependency
  extras.
- GPN-MSA is deprecated and supported for inference only.

### Removed

- Historical paper analyses, dataset-building workflows, GPN-MSA training, and
  retired notebooks from `main`. They remain available at the prepared
  `analysis-archive-2026-08-18` tag, pending final publication approval.

## 0.9.0a1 — 2026-08-18

- Claimed the `gpn` PyPI distribution with an installable alpha package.
- Added PEP 621 metadata, Python 3.11–3.13 support, explicit Transformers AutoClass
  registration, and Trusted Publishing through GitHub Actions.
