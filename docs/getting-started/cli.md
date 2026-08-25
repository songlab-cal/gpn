# Command-line interface

The installed `gpn` command exposes a deliberately small maintained surface. It
does not expose dataset construction or every historical scorer module.

```text
gpn ss {train,vep,logits,embedding} ...
gpn msa {vep,logits,embedding} ...
gpn star {train,vep,logits,embedding} ...
```

`ss` denotes the single-species GPN family. Cyclopts derives each command from
its typed Python signature. Use `gpn <family> <command> --help` for the complete
arguments, including the pinned Transformers `TrainingArguments` surface.

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
gpn ss train recipes/gpn_training/cpu-smoke.yaml
```

Score variants whose `pos` column uses one-based VCF coordinates:

```bash
gpn ss vep \
  --input-path variants.parquet \
  --genome-path genome.fa.gz \
  --window-size 512 \
  --model-path songlab/gpn-brassicales \
  --output-path scores.parquet \
  --per-device-eval-batch-size 64
```

The same family group exposes masked-nucleotide logits and sequence embeddings:

```bash
gpn ss logits \
  --input-path POSITIONS_PATH \
  --genome-path GENOME_PATH \
  --window-size WINDOW_SIZE \
  --model-path MODEL_PATH \
  --output-path OUTPUT_PATH
gpn ss embedding \
  --input-path WINDOWS_PATH \
  --genome-path GENOME_PATH \
  --center-window-size CENTER_WINDOW_SIZE \
  --model-path MODEL_PATH \
  --output-path OUTPUT_PATH
```

Embedding averages exactly `CENTER_WINDOW_SIZE` positions, for both odd and even
values.

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
gpn msa vep \
  --input-path INPUT_PATH \
  --msa-path LOCAL_MSA_PATH \
  --window-size WINDOW_SIZE \
  --model-path MODEL_PATH \
  --output-path OUTPUT_PATH
gpn msa logits \
  --input-path INPUT_PATH \
  --msa-path LOCAL_MSA_PATH \
  --window-size WINDOW_SIZE \
  --model-path MODEL_PATH \
  --output-path OUTPUT_PATH
gpn msa embedding \
  --input-path INPUT_PATH \
  --msa-path LOCAL_MSA_PATH \
  --window-size WINDOW_SIZE \
  --model-path MODEL_PATH \
  --output-path OUTPUT_PATH
```

`LOCAL_MSA_PATH` must point directly to an existing GPN-MSA Zarr store. Its target
genome, species count, species order, and preprocessing must match the selected
checkpoint. The CLI never downloads a whole-genome alignment. Parquet, CSV/TSV,
VCF, GTF, and GFF inputs are detected from existing local paths; otherwise
`INPUT_PATH` is passed to Hugging Face Datasets as a dataset identifier.

VEP and logits use one-based `pos` coordinates and require an even `WINDOW_SIZE`.
Embedding inputs instead use zero-based, half-open `start` and `end` coordinates.
The embedding result averages exactly `CENTER_WINDOW_SIZE` positions, for both
odd and even values.
GPN-MSA VEP emits the same raw, uncalibrated forward/reverse-averaged LLR described
above.

## GPN-Star

Train from prepared intervals and local MSAs with the maintained recipe:

```bash
gpn star train recipes/gpn_star_training/cpu-smoke.yaml
```

The three core inference operations share this shape:

```bash
gpn star vep \
  --input-path INPUT_PATH \
  --msa-path LOCAL_MSA_PATH \
  --window-size WINDOW_SIZE \
  --model-path MODEL_PATH \
  --output-path OUTPUT_PATH
gpn star logits \
  --input-path INPUT_PATH \
  --msa-path LOCAL_MSA_PATH \
  --window-size WINDOW_SIZE \
  --model-path MODEL_PATH \
  --output-path OUTPUT_PATH
gpn star embedding \
  --input-path INPUT_PATH \
  --msa-path LOCAL_MSA_PATH \
  --window-size WINDOW_SIZE \
  --model-path MODEL_PATH \
  --output-path OUTPUT_PATH
```

All three inference families support the same durable, process-count-independent
batch checkpoints. Set `--checkpoint-batch-size`; the directory defaults to
`OUTPUT_PATH_checkpoints`. See any inference command's help for revision and
cleanup options.

`LOCAL_MSA_PATH` is not itself an `all.zarr` store. It is either a numeric
species-count directory containing `all.zarr`, or a parent containing one or more
such numeric directories. Every selected alignment must match the target genome,
species order, and evolutionary scale expected by the checkpoint. VEP and logits
use one-based `pos` coordinates and an even `WINDOW_SIZE`; embeddings use
zero-based, half-open `start` and `end`. Embeddings average exactly
`CENTER_WINDOW_SIZE` positions, for both odd and even values. GPN-Star VEP outputs
a raw, uncalibrated LLR. Any checkpoint-specific calibration is a separate
downstream operation.

## AutoClass-only inference

PhyloGPN and the sorghum gene-expression fine-tune remain supported through the
explicit `register_auto_classes(...)` plus Transformers AutoClass APIs documented
in the README and their model cards. They intentionally do not have dedicated CLI
commands.

## Devices and distributed execution

Inference defaults to FP32 without compilation on every device. This makes the
default explicit and predictable. A typical GPU invocation adds
`--bf16-full-eval --torch-compile`; unsupported compilation fails visibly for
every model family. Other current and future Trainer flags are exposed directly
because this release pins Transformers exactly.

GPN forces the small set of prediction invariants it owns: training and
evaluation phases are disabled, predictions are retained, input columns are not
pruned, incomplete batches are retained, and dataloader results stay in input
order. `--push-to-hub` is rejected. Direct inference also verifies that it
produced exactly one output row per input row. `OUTPUT_PATH` is always the
scientific Parquet result; Transformers' `--output-dir` is only Trainer working
state and defaults to a temporary directory when omitted.

For multi-GPU inference, launch that same CLI through `torchrun`. Every process
participates in prediction, while rank zero alone commits checkpoint batches and
the final Parquet output:

```bash
torchrun --standalone --nproc-per-node=4 --module gpn.cli \
  ss vep \
  --input-path variants.parquet \
  --genome-path genome.fa.gz \
  --window-size 512 \
  --model-path songlab/gpn-brassicales \
  --output-path scores.parquet \
  --per-device-eval-batch-size 64 \
  --dataloader-num-workers 4 \
  --bf16-full-eval
```

`--per-device-eval-batch-size` is per process, so multiply it by the number of
processes for the nominal global batch size. Durable checkpoints use the same
process-count-independent row ranges in single- and multi-GPU runs.

`--dataloader-num-workers` is also per process and defaults to zero. In the
example above, four local ranks each create four DataLoader workers: 16 worker
processes in total, in addition to the four rank processes. Size the CPU
allocation for all local ranks and workers; the flag is not a job-wide worker
budget.

`OMP_NUM_THREADS` controls CPU threading separately from the DataLoader worker
count and is not required for correctness. When more than one local process is
launched and the variable is unset, `torchrun` sets it to `1` to avoid CPU
oversubscription. Keep that default unless profiling identifies a CPU compute
bottleneck; if you tune it, account for the threads used by every local process
within the job's CPU allocation.

Distributed training uses the public module entry point as well:

```bash
torchrun --standalone --nproc-per-node=4 --module gpn.cli \
  star train recipes/gpn_star_training/gpu.yaml
```

The supported command-line contract is the `gpn` family tree above. Python
modules behind it are implementation details unless documented as public APIs.
