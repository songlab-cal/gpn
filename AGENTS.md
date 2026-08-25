# Repository guidance

## Maintained scope

Maintain `src/gpn/`, `tests/`, `docs/`, and the GPN and GPN-Star recipes. GPN
supports training and inference. GPN-MSA is deprecated and supports inference
only. PhyloGPN supports inference only. GPN-Star supports training and inference.
The Sorghum gene-expression model supports inference only. Dataset-building
workflows are not maintained.

Historical analysis, dataset builders, GPN-MSA training, and retired notebooks
are preserved at `analysis-archive-2026-08-18`. Do not restore or modernize them
on `main`.

## Development workflow

Use Python 3.13 and uv. The supported development environment is:

```bash
uv sync --extra train --group dev --group docs
uv run pre-commit install
```

Before proposing a change, run:

```bash
uv run pre-commit run --all-files
uv run pytest
python docs/prepare_notebooks.py
uv run sphinx-build -n -W --keep-going -b html docs docs/_build/html
```

Normal tests are network-free. Published-model tests download pinned Hugging
Face assets and run only with `uv run pytest --run-published-models`. Never add
scheduled Hub checks.

## Scientific changes

Treat fixture updates as scientific changes. Record coordinates, assembly,
species/nucleotide/label order, model revision, GPN and Transformers versions,
device and dtype, generation command, artifact checksum, numerical tolerances,
and the reason for approving new expectations or tolerance changes. Do not
casually regenerate stored notebook output or fixtures.

Genomic intervals are zero-based and half-open; VCF positions are one-based.
Variant log-likelihood ratios are alternate minus reference. Give jaxtyping axes
stable semantic names; put runtime shape checks at boundaries and in tests, not
in hot loops.

## Package architecture

Keep model-family implementations under `src/gpn/ss/`, `src/gpn/msa/`,
`src/gpn/star/`, and `src/gpn/phylo/`. A module directly under `src/gpn/` must
contain functionality genuinely shared by at least two families; family-specific
models, data access, inference adapters, losses, and utilities belong inside that
family's directory.

Share behavior through small common functions and composition. Do not introduce
inheritance between GPN model families. Do not restore retired compatibility
modules or import-side-effect registration shims.

## APIs and dependencies

Register Hugging Face Auto classes through the explicit
`gpn.register_auto_classes()` API. Do not rely on imports for side effects and do
not add a custom model-loading abstraction.

Use modern syntax such as `list[str]` and `X | None`; import from `typing` only
for constructs that still require it. Add `from __future__ import
annotations` only when forward references, import cycles, or deliberate runtime
annotation behavior make it necessary.

Place dependencies in base runtime, a feature extra, `dev`, or `docs` according
to their actual consumers. Research-only dependencies never belong in the root
project.

Keep the three canonical model demos and the lightweight GPN-Star precomputed-score
workflow under `colabs/`. Documentation renders them without execution. Do not add
notebook-only scientific logic, local paths, secrets, downloaded models, or
whole-genome MSAs. Refresh the three model-demo outputs together with
`python -m docs.refresh_notebooks` only in a dedicated compute allocation, then
review output diffs and the scientific audit.

## Research branches

Paper-specific and exploratory analysis must not be committed to `main`.
Projects may live indefinitely on any off-main branch and evolve independently.
See `docs/development/research.md` and `CONTRIBUTING.md` for non-binding project
conventions.
Contributing reusable code back to the maintained package is encouraged but
optional; duplicated non-core analysis code is acceptable.

## Protected operations

Releases, tags, PyPI publication, model-card writes, and merges require explicit
maintainer authorization. More specific `AGENTS.md` files may refine guidance
for a subtree without expanding maintained scope.
