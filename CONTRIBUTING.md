# Contributing to GPN

Thank you for helping maintain GPN as reliable scientific software. Bug fixes,
tests, documentation, and focused improvements to supported model workflows are
welcome.

GPN supports training and inference. GPN-MSA is deprecated and supports inference
only. PhyloGPN supports inference only. GPN-Star supports training and inference.
The Sorghum gene-expression model supports inference only. Dataset-building
workflows are not maintained.

## Local setup

Install Python 3.13 and [uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync --extra train --group dev --group docs
uv run pre-commit install
```

Use the locked environment and include `uv.lock` changes whenever dependency
metadata changes.

## Checks

Run the same offline gates used for pull requests:

```bash
uv run pre-commit run --all-files
uv run pytest
python docs/prepare_notebooks.py
uv run sphinx-build -n -W --keep-going -b html docs docs/_build/html
```

The published-model suite is an explicit networked audit, not a routine CI job:

```bash
uv run pytest --run-published-models
```

When a scientific fixture changes, document its coordinates, reference
assembly, species/nucleotide/label order, pinned model revision, GPN and
Transformers versions, device and dtype, generation command, checksum,
numerical tolerances, and why the new expectation or tolerance is correct.
Genomic intervals are zero-based and half-open, VCF positions are one-based,
and variant log-likelihood ratios are alternate minus reference.

Committed quick-start outputs are refreshed only on a dedicated compute node with
`python -m docs.refresh_notebooks`; see `docs/development/index.md`. Do not execute the
refresh on a login node or download a whole-genome MSA for it.

## Dependency placement

- Base dependencies are required to import GPN and register all supported model
  families.
- The `inference` and `train` extras contain dependencies for those maintained
  workflows.
- Optional experiment tracking lives in the `tracking` extra.
- Tests and quality tools belong to the `dev` group; documentation tools belong
  to the `docs` group.
- Artifact build and inspection tools used by the publishing workflow belong to
  the `release` group.
- Paper-specific and exploratory dependencies do not belong in the root
  project.

## Research and exploratory work

Research analysis must stay off `main`, but it does not need a separate
repository. Any branch name and ancestry are acceptable, and branches may remain
indefinitely. A useful, non-binding convention is an `analysis/<project>/`
directory with its own `pyproject.toml`, `uv.lock`, `.python-version`, and README,
pinning the GPN release or commit it uses. Tooling and structure are otherwise up
to each project.

Research branches may evolve independently and may duplicate non-core analysis
code. Contributing reusable pieces back to `main` is encouraged but optional.

## Maintainer releases

Follow the [maintainer release runbook](docs/release.md). Releases are built from a
reviewed commit on `main` and published by a GitHub Release tagged `v<version>`.
The workflow verifies the version and ancestry, builds with locked tools, and uses
PyPI Trusted Publishing with attestations. Do not upload distributions manually.
