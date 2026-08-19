# Command-line interface

The installed `gpn` command exposes a deliberately small maintained surface. It
does not expose dataset construction or every historical scorer module.

```text
gpn ss {train,vep,logits,embedding} ...
gpn msa {vep,logits,embedding} ...
gpn star {train,vep,logits,embedding} ...
```

`ss` denotes the single-species GPN family. Use
`gpn <family> <command> --help` for the complete arguments of a leaf command.
Top-level and group help are lightweight; model, dataset, and PyTorch modules are
imported only after a leaf command is selected.

## Installation

Inference commands require the inference extra:

```bash
pip install "gpn[inference]"
```

GPN and GPN-Star training require the training extra:

```bash
pip install "gpn[train]"
```

## GPN

Train GPN from a prepared dataset using one of the maintained recipe profiles:

```bash
gpn ss train recipes/gpn_training/cpu-smoke.json
```

Score variants whose `pos` column uses one-based VCF coordinates:

```bash
gpn ss vep variants.parquet genome.fa.gz 512 songlab/gpn-brassicales scores.parquet \
  --is-file --per-device-batch-size 64
```

The same family group exposes masked-nucleotide logits and sequence embeddings:

```bash
gpn ss logits POSITIONS_PATH GENOME_PATH WINDOW_SIZE MODEL_PATH OUTPUT_PATH
gpn ss embedding WINDOWS_PATH GENOME_PATH CENTER_WINDOW_SIZE MODEL_PATH OUTPUT_PATH
```

VEP inputs must be biallelic SNVs: `ref` and `alt` are distinct, uppercase, single
characters from `A`, `C`, `G`, and `T`. The reference allele must match the local
genome or MSA at `chrom:pos`; inference stops on a mismatch.

The output `score` is the raw, uncalibrated alternate-minus-reference
log-likelihood ratio averaged over forward and reverse-complement sequence
orientations. In particular, negative values mean the alternate is less likely
than the reference under the model.

## GPN-MSA

GPN-MSA is deprecated and supports inference only. Its public commands cover
variant scoring, masked-nucleotide logits, and embeddings:

```bash
gpn msa vep INPUT_PATH LOCAL_MSA_PATH WINDOW_SIZE MODEL_PATH OUTPUT_PATH
gpn msa logits INPUT_PATH LOCAL_MSA_PATH WINDOW_SIZE MODEL_PATH OUTPUT_PATH
gpn msa embedding INPUT_PATH LOCAL_MSA_PATH WINDOW_SIZE MODEL_PATH OUTPUT_PATH
```

`LOCAL_MSA_PATH` must point directly to an existing GPN-MSA Zarr store. Its target
genome, species count, species order, and preprocessing must match the selected
checkpoint. The CLI never downloads a whole-genome alignment. Use `--is-file` for
Parquet, CSV/TSV, or VCF input; otherwise `INPUT_PATH` is passed to Hugging Face
Datasets.

VEP and logits use one-based `pos` coordinates and require an even `WINDOW_SIZE`.
Embedding inputs instead use zero-based, half-open `start` and `end` coordinates.
GPN-MSA VEP emits the same raw, uncalibrated forward/reverse-averaged LLR described
above.

## GPN-Star

Train from prepared intervals and local MSAs with the maintained recipe:

```bash
gpn star train recipes/gpn_star_training/cpu-smoke.json
```

The three core inference operations share this shape:

```bash
gpn star vep INPUT_PATH LOCAL_MSA_PATH WINDOW_SIZE MODEL_PATH OUTPUT_PATH
gpn star logits INPUT_PATH LOCAL_MSA_PATH WINDOW_SIZE MODEL_PATH OUTPUT_PATH
gpn star embedding INPUT_PATH LOCAL_MSA_PATH WINDOW_SIZE MODEL_PATH OUTPUT_PATH
```

GPN-Star inference supports durable batch checkpoints; see
`gpn star vep --help` for `--checkpoint-batch-size` and related options.

`LOCAL_MSA_PATH` is not itself an `all.zarr` store. It is either a numeric
species-count directory containing `all.zarr`, or a parent containing one or more
such numeric directories. Every selected alignment must match the target genome,
species order, and evolutionary scale expected by the checkpoint. VEP and logits
use one-based `pos` coordinates and an even `WINDOW_SIZE`; embeddings use
zero-based, half-open `start` and `end`. GPN-Star VEP outputs a raw, uncalibrated
LLR. Any checkpoint-specific calibration is a separate downstream operation.

## AutoClass-only inference

PhyloGPN and the sorghum gene-expression fine-tune remain supported through the
explicit `register_auto_classes(...)` plus Transformers AutoClass APIs documented
in the README and their model cards. They intentionally do not have dedicated CLI
commands.

## Devices and distributed execution

Inference enables FP16 and `torch.compile` automatically when CUDA is available
and disables both on CPU. Override either choice with `--fp16`/`--no-fp16` and
`--torch-compile`/`--no-torch-compile`. The deprecated GPN-MSA path preserves its
historical eager fallback when a compiled custom operation is unsupported.

For distributed training, launch the public module entry point:

```bash
torchrun --standalone --nproc-per-node=4 --module gpn.cli \
  star train recipes/gpn_star_training/gpu.json
```

The supported command-line contract is the `gpn` family tree above. Python
modules behind it are implementation details unless documented as public APIs.
