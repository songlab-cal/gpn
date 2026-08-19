# Hugging Face asset maintenance

This directory separates three things that are easy to conflate:

1. [`manifest.json`](manifest.json) is the reviewed compatibility contract for the
   five checkpoints that the installed `gpn` package supports.
2. [`audits/2026-08-19.json`](audits/2026-08-19.json) is a dated observation of the
   broader public GPN asset surface. Inventory membership does not imply support.
3. `card-proposals/` contains reviewable documentation replacements. These files
   are not synchronized to Hugging Face automatically.

The JSON Schema in [`manifest.schema.json`](manifest.schema.json) documents the
manifest format. Offline tests keep it synchronized with the numerical scientific
baseline.

## Deliberate audit

Run the metadata audit only when changing a supported checkpoint, a Hub asset, or
the tested Transformers/PyTorch compatibility range:

```bash
uv run python hub/audit_hub.py --output hub/audits/YYYY-MM-DD.json
```

The command ignores locally configured credentials and reads only public repository
metadata plus `README.md` and `config.json` files smaller than 1 MiB. It never calls
`snapshot_download` and never downloads weights, dataset rows, or alignment
archives. Review the generated diff; do not replace approved revisions just because
a default branch moved.

Numerical validation is separate and opt-in:

```bash
HF_HOME=/path/to/a/controlled/cache \
  uv run --extra inference pytest tests/test_published_models.py \
  --run-published-models
```

Run that command through a compute allocation, reuse an existing cache, and keep
the repository's CPU/GPU and disk limits. It downloads pinned checkpoints but uses
the checked-in 3.5 KiB MSA fixture. Never download the 42.27 GB
`multiz100way-pigz/99.zarr.tar.gz` archive for this audit.

## Applying card proposals

Model-card writes are protected operations. A maintainer should review a proposal,
resolve every `TODO(maintainer)` fact, and explicitly authorize the external write.
After any card or config update, rerun the metadata audit and update the observed
Hub revision; only change an approved model revision after the numerical regression
also passes.
