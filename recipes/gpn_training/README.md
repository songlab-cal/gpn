# Canonical GPN training recipe

This is the maintained example for training GPN from scratch on an already
prepared sequence dataset. It is intentionally not an exact executable profile of
every published run and does not include dataset construction.

## Prepared input contract

`dataset_name` must identify a local or Hugging Face dataset with `train` and
`validation` splits. Every row must contain `seq`, a fixed-length DNA string.
Uppercase bases receive normal loss weight; lowercase bases receive the configured
soft-mask weight. The tokenizer must use the seven-token GPN vocabulary. The
published `songlab/gpn-brassicales` tokenizer is pinned in both example configs.

Prepare those inputs independently, then install the training dependencies:

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

Edit the paths in `cpu-smoke.json`, then verify plumbing with one small CPU step:

```bash
uv run python -m gpn.ss.run_mlm recipes/gpn_training/cpu-smoke.json
```

For a realistic four-GPU starting point, edit `gpu.json` and run:

```bash
uv run --no-sync torchrun --standalone --nproc_per_node=4 \
  -m gpn.ss.run_mlm recipes/gpn_training/gpu.json
```

The GPU config has effective global batch size
`64 examples/GPU × 8 accumulation steps × 4 GPUs = 2048`. Adjust gradient
accumulation when changing the world size. Choose training duration and evaluation
cadence for the size and diversity of the prepared dataset; the example is a
maintained starting point, not a claim that 120,000 steps is optimal everywhere.

The historical Brassicales and animal-promoter commands remain available at the
`analysis-archive-2026-08-18` tag.
