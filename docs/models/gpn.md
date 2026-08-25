# GPN

GPN, also known as GPN-SS, is a single-sequence genomic language model trained on
unaligned genomes. The maintained package supports both inference and training on
prepared datasets.

## Quick start

```python
from gpn import register_auto_classes
from transformers import AutoModelForMaskedLM

register_auto_classes("ss")

model = AutoModelForMaskedLM.from_pretrained("songlab/gpn-brassicales")
```

The {doc}`GPN tutorial <../_notebooks/gpn_demo>` demonstrates tokenization,
embeddings, masked nucleotide probabilities, and plots. File-backed inference is
available through `gpn ss {vep,logits,embedding}`; see the
{doc}`command-line guide <../getting-started/cli>`. The canonical prepared-data
training recipe lives in
[`recipes/gpn_training`](https://github.com/songlab-cal/gpn/tree/main/recipes/gpn_training).

## Published assets

### Brassicales model

The [GPN collection](https://huggingface.co/collections/songlab/gpn-653191edcb0270ed05ad2c3e)
accompanies the [PNAS paper](https://doi.org/10.1073/pnas.2311219120).

- [Brassicales pretraining dataset](https://huggingface.co/datasets/songlab/genomes-brassicales-balanced-v1): balanced sequences from *Arabidopsis thaliana* and seven other Brassicales.
- [`songlab/gpn-brassicales`](https://huggingface.co/songlab/gpn-brassicales): the published masked language model.
- [`songlab/tokenizer-dna-mlm`](https://huggingface.co/songlab/tokenizer-dna-mlm): the seven-token DNA tokenizer used by the published model.
- [Processed Arabidopsis data](https://huggingface.co/datasets/gonzalobenegas/processed-data-arabidopsis): supporting resources used by the published study.

### Animal promoter model and TraitGym

The [TraitGym collection](https://huggingface.co/collections/songlab/traitgym-6796d4fbb825d5b94e65d30f)
accompanies the [causal regulatory variant benchmark](https://doi.org/10.1101/2025.02.11.637758).

- [Animal promoter pretraining dataset](https://huggingface.co/datasets/songlab/gpn-animal-promoter-dataset): promoter sequences used for GPN pretraining.
- [`songlab/gpn-animal-promoter`](https://huggingface.co/songlab/gpn-animal-promoter): the published promoter model.
- [`songlab/TraitGym`](https://huggingface.co/datasets/songlab/TraitGym): the benchmark datasets and model predictions.
- [Training checkpoints](https://huggingface.co/datasets/songlab/gpn-animal-promoter-checkpoints): intermediate checkpoints from the published run.
- [TraitGym leaderboard](https://huggingface.co/spaces/songlab/TraitGym-leaderboard): an interactive comparison of submitted predictions.

(sorghum-expression)=
### Sorghum gene-expression fine-tune

The sorghum model is a fine-tuned application of GPN, not a separate model
family. It supports inference through the same `register_auto_classes("ss")`
path. The [sorghum collection](https://huggingface.co/collections/songlab/sorghum-gene-expression-prediction-68963dd31658bfb98c07ae1b)
accompanies the [Nature Biotechnology paper](https://doi.org/10.1038/s41587-026-03046-y).

- [`songlab/gxa-sorghum-v1`](https://huggingface.co/datasets/songlab/gxa-sorghum-v1): sorghum gene-expression data from Gene Expression Atlas.
- [`songlab/gpn-brassicales-gxa-sorghum-v1`](https://huggingface.co/songlab/gpn-brassicales-gxa-sorghum-v1): a gene-expression model fine-tuned from `gpn-brassicales`.

## Historical analyses

- [Brassicales GPN analysis](https://github.com/songlab-cal/gpn/tree/analysis-archive-2026-08-18/analysis/gpn_arabidopsis)
- [Animal-promoter and TraitGym analysis](https://github.com/songlab-cal/gpn/tree/analysis-archive-2026-08-18/analysis/gpn_animal_promoter)
- [Sorghum gene-expression analysis](https://github.com/songlab-cal/gpn/tree/analysis-archive-2026-08-18/analysis/gpn_sorghum_expression)
