# Development

The maintained surface is `src/gpn`, tests, documentation, and the two
prepared-data training recipes. Historical analysis and dataset construction are
preserved separately from `main`.

## Setup and checks

```bash
uv sync --extra train --group dev --group docs
uv run pre-commit run --all-files
uv run pytest
python docs/prepare_notebooks.py
uv run sphinx-build -n -W --keep-going -b html docs docs/_build/html
```

Normal checks are offline. See the root
[`CONTRIBUTING.md`](https://github.com/songlab-cal/gpn/blob/main/CONTRIBUTING.md)
and [`AGENTS.md`](https://github.com/songlab-cal/gpn/blob/main/AGENTS.md) for
dependency placement, scientific-fixture approval, jaxtyping conventions, and the
off-main research lifecycle.

The `release` dependency group is intentionally separate from ordinary
development and documentation. It contains only the locked build and distribution
inspection tools used by the release workflow; see the [release runbook](release.md).

## Documentation policy

- Markdown under `docs/` is the source for all published documentation.
- Three existing model demos and one lightweight precomputed-score workflow live
  under `colabs/`. All four retain committed outputs; Sphinx and Read the Docs
  never execute any notebook.
- Notebook output refreshes are deliberate scientific changes. Record package,
  Transformers, model revision, dtype, and device metadata and compare numerical
  results to fixtures.
- Do not commit local paths, secrets, downloaded models, whole-genome alignments,
  or transient build output.

Refresh all three outputs together on a dedicated compute node. The notebooks use
only pinned model revisions and the tiny checked-in alignment fixture; the command
must never be pointed at or made to download a whole-genome MSA.

```bash
# Submit this command through Slurm with at most 8 CPUs and 1 GPU.
uv run --no-sync --with seaborn --with scikit-learn \
  python -m docs.refresh_notebooks
```

The script refuses to run outside a Slurm allocation unless `--allow-local` is
passed explicitly for another dedicated compute environment. Review every output
diff and rerun the published-model audit before accepting a refresh.

```{toctree}
:hidden:
:maxdepth: 1

validation
research
release
```
