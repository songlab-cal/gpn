# GPN-MSA

GPN-MSA is a genomic language model trained on a multispecies whole-genome
alignment. It is deprecated in favor of GPN-Star: the maintained package supports
inference from the published model, but not GPN-MSA training.

## Quick start

```python
from gpn import register_auto_classes
from transformers import AutoModelForMaskedLM

register_auto_classes("msa")
model = AutoModelForMaskedLM.from_pretrained("songlab/gpn-msa-sapiens")
```

File-backed inference uses `gpn msa {vep,logits,embedding}` and a compatible
local Zarr alignment. The checkpoint and store must agree on target assembly,
species count, species order, and preprocessing. See the
{doc}`command-line guide <../getting-started/cli>` for the input contract.

## Published assets

The [GPN-MSA collection](https://huggingface.co/collections/songlab/gpn-msa-65319280c93c85e11c803887)
accompanies the [Nature Biotechnology paper](https://doi.org/10.1038/s41587-024-02511-w).

### Alignment and training inputs

- [`songlab/multiz100way`](https://huggingface.co/datasets/songlab/multiz100way): processed 100-way vertebrate alignment in ZIP-backed Zarr stores.
- [`songlab/multiz100way-pigz`](https://huggingface.co/datasets/songlab/multiz100way-pigz): compressed 100-way vertebrate alignment archive.
- [`lpigou/89.zarr`](https://huggingface.co/datasets/lpigou/89.zarr): the alignment representation used by the original workflow.
- [`songlab/gpn-msa-sapiens-dataset`](https://huggingface.co/datasets/songlab/gpn-msa-sapiens-dataset): prepared human training regions. The asset remains public, but dataset construction and GPN-MSA training are not maintained.

### Model and evaluations

- [`songlab/gpn-msa-sapiens`](https://huggingface.co/songlab/gpn-msa-sapiens): the published human GPN-MSA checkpoint.
- [`songlab/clinvar`](https://huggingface.co/datasets/songlab/clinvar): missense variants with clinical pathogenic or benign labels.
- [`songlab/cosmic`](https://huggingface.co/datasets/songlab/cosmic): somatic missense mutations in cancer.
- [`songlab/omim`](https://huggingface.co/datasets/songlab/omim): regulatory variants implicated in Mendelian disorders.
- [`songlab/gnomad`](https://huggingface.co/datasets/songlab/gnomad): genome-wide variants with allele-frequency information.

### Genome-wide predictions

- [`songlab/gpn-msa-hg38-scores`](https://huggingface.co/datasets/songlab/gpn-msa-hg38-scores): hg38 genome-wide variant scores.
- [`songlab/gpn-msa-hg38-gene-essentiality-scores`](https://huggingface.co/datasets/songlab/gpn-msa-hg38-gene-essentiality-scores): gene-essentiality predictions.

## Historical analysis

The [GPN-MSA paper analysis](https://github.com/songlab-cal/gpn/tree/analysis-archive-2026-08-18/analysis/gpn-msa_human)
is preserved in the historical archive.
