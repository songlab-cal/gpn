# Canonical GPN-Star training recipe

This is the maintained example for training GPN-Star from scratch on an already
prepared interval dataset and a local multiple-sequence alignment (MSA). It does
not build or download either input.

## Prepared input contract

`dataset_name` must identify a local or Hugging Face dataset with `train` and
`validation` splits. Each row must contain:

- `chrom`, `start`, `end`, and `strand`, defining a fixed-length interval;
- `lowercase`, a boolean array aligned to the interval;
- `phyloP`, a numeric array aligned to the interval; and
- `phastCons`, a numeric array aligned to the interval.

`msa_path` must point to a local directory whose children follow GPN-Star's
`<species-count>/all.zarr` layout. A species-count directory itself is also
accepted. `phylo_dist_path` must contain `pairwise.npy` and `in_clade.npy` in
exactly the same species order as the concatenated MSA arrays. Training samples
20 target species per interval. Species index 0 must be the intended reference
species (human for hg38); the trainer always includes it as a target. These
inputs are large and deliberately remain outside this repository.

Install the training dependencies:

```bash
uv sync --extra train --group dev
```

The repository lock deliberately selects CPU-only PyTorch for portable tests.
Before a GPU run, replace that wheel with the build recommended by the
[PyTorch installation selector](https://pytorch.org/get-started/locally/) for
the machine's CUDA runtime, after the final `uv sync`:

```bash
uv pip install --python .venv/bin/python --reinstall torch \
  --index <PyTorch-CUDA-wheel-index>
```

A later `uv sync` will restore the locked CPU build, so the GPU launch below
uses `--no-sync`.

Edit all three prepared-input paths in `cpu-smoke.json`, then verify plumbing
with one small CPU step:

```bash
uv run python -m gpn.star.train recipes/gpn_star_training/cpu-smoke.json
```

For a realistic four-GPU starting point, edit `gpu.json` and run:

```bash
uv run --no-sync torchrun --standalone --nproc_per_node=4 \
  -m gpn.star.train recipes/gpn_star_training/gpu.json
```

The GPU profile records the architecture of the published 200M-parameter model.
Its effective global batch size is
`16 examples/GPU × 4 accumulation steps × 4 GPUs = 256`. Training duration and
evaluation cadence are starting points to revisit for each prepared dataset.

The paper-specific workflow remains available at the proposed, currently
unpublished `analysis-archive-2026-08-18` tag.
