# Canonical GPN training recipe

This is the maintained example for training GPN from scratch on the public,
already-prepared Brassicales sequence dataset. It is intentionally not an exact
executable profile of every published run and does not include raw-data or
dataset construction.

## Prepared input contract

Both profiles use
[`songlab/genomes-brassicales-balanced-v1`](https://huggingface.co/datasets/songlab/genomes-brassicales-balanced-v1)
at immutable revision `d11c6084dd2bb5575f9ce224cbcc435a687e67bf`, the
dataset used for the published Brassicales model. It provides `train`,
`validation`, and `test` splits. Every row contains `seq`, a 512-nucleotide DNA
string; the accompanying assembly, chromosome, zero-based half-open coordinates,
and strand columns are provenance and are not model inputs.

Any replacement `dataset_name` must identify a local or Hugging Face dataset with
`train` and `validation` splits and a fixed-length `seq` string in every row. Set
`dataset_revision` for a Hub dataset and omit it for an immutable local snapshot.
Uppercase bases receive normal loss weight; lowercase bases receive the configured
soft-mask weight. The tokenizer must use the seven-token GPN vocabulary. The
published `songlab/gpn-brassicales` tokenizer is pinned in both example configs.

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

The GPN trainer streams this Hub dataset, so the one-step smoke run does not
materialize the complete dataset locally:

```bash
uv run gpn ss train recipes/gpn_training/cpu-smoke.json
```

For a realistic four-GPU starting point, edit `gpu.json` and run:

```bash
uv run --no-sync torchrun --standalone --nproc-per-node=4 --module gpn.cli \
  ss train recipes/gpn_training/gpu.json
```

The GPU config has effective global batch size
`64 examples/GPU × 8 accumulation steps × 4 GPUs = 2048`. Adjust gradient
accumulation when changing the world size. Choose training duration and evaluation
cadence for the size and diversity of the prepared dataset; the example is a
maintained starting point, not a claim that 120,000 steps is optimal everywhere.

The maintained contract starts at an already-prepared dataset. The historical
Brassicales and animal-promoter construction and training commands remain
available, without support guarantees, at the `analysis-archive-2026-08-18` tag.
