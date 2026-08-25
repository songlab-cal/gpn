# Canonical GPN-Star training recipe

This is the maintained example for training human GPN-Star V100 from scratch on
a public, already-prepared interval dataset and a compatible local
multiple-sequence alignment (MSA). It does not build or download an MSA.
The trainer accepts one YAML profile per run so the complete configuration is
reviewable and version-controlled.

## Prepared input contract

Both profiles use
[`songlab/gpn-msa-sapiens-dataset`](https://huggingface.co/datasets/songlab/gpn-msa-sapiens-dataset)
at immutable revision `57c0e187c674761955518f3579eb0d7b5a0b7078`.
Although published for GPN-MSA, this is also the human GPN-Star V100 interval
dataset: alignment species selection is separate from interval selection. It
provides `train`, `validation`, and `test` splits. Each row contains:

- `chrom`, `start`, and `end`, with zero-based half-open coordinates whose
  chromosome naming must match the MSA;
- `strand`, either `+` or `-`;
- `lowercase`, a length-128 boolean array aligned to the interval;
- `phyloP`, a length-128 floating-point array aligned to the interval; and
- `phastCons`, a length-128 floating-point array aligned to the interval.

Any replacement `dataset_name` must provide the same columns in `train` and
`validation` splits, with every array length equal to `end - start`. Set
`dataset_revision` for a Hub dataset and omit it for an immutable local snapshot.
Unlike the GPN trainer, GPN-Star currently materializes its Arrow dataset in the
Hugging Face cache; the pinned public dataset is approximately 1.65 GB.

`msa_path` must point to a local directory whose children follow GPN-Star's
`<species-count>/all.zarr` layout. A species-count directory itself is also
accepted. Each chromosome in a Zarr store is indexed by reference-genome
position and has species on its second axis. The human V100 recipe requires the
full V100 alignment; the reduced alignment used by the published GPN-MSA model
is not interchangeable.

`phylo_dist_path` must contain `pairwise.npy` with shape `(N, N)` and
`in_clade.npy` with shape `(N,)`, where `N` is the total number of concatenated
MSA species. Both arrays must use exactly the MSA species order. Training samples
20 target species per interval. Species index 0 must be the intended reference
species (human for hg38); the trainer always includes it as a target. These large
alignment inputs deliberately remain outside this repository.

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

Edit the two local alignment paths in `cpu-smoke.yaml`, then verify the complete
Hub-dataset/MSA plumbing with one small CPU step:

```bash
uv run gpn star train recipes/gpn_star_training/cpu-smoke.yaml
```

For a realistic four-GPU starting point, edit `gpu.yaml` and run:

```bash
uv run --no-sync torchrun --standalone --nproc-per-node=4 --module gpn.cli \
  star train recipes/gpn_star_training/gpu.yaml
```

The GPU profile records the architecture of the published 200M-parameter model.
Its effective global batch size is
`16 examples/GPU × 4 accumulation steps × 4 GPUs = 256`. Training duration and
evaluation cadence are starting points to revisit for each prepared dataset.

The maintained contract starts at the prepared Hub dataset, MSA, and
phylogenetic-distance arrays. The paper-specific raw-data construction workflow
remains available, without support guarantees, at the proposed, currently
unpublished `analysis-archive-2026-08-18` tag.
