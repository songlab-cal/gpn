---
license: mit
library_name: transformers
inference: false
tags:
- biology
- dna
- genomics
- language-model
- variant-effect-prediction
---

# GPN-Star hg38 V100 200M

This is the 200M-parameter GPN-Star checkpoint for hg38 with the vertebrate
100-way alignment. GPN-Star learns from alignments across evolutionary scales and
supports masked likelihoods, embeddings, and variant-effect prediction. See the
[preprint](https://doi.org/10.1101/2025.09.21.677619),
[package documentation](https://github.com/songlab-cal/gpn#readme), and
[fixture-sized demo](https://github.com/songlab-cal/gpn/blob/main/colabs/gpn_star_demo.ipynb).

## Install and load

```bash
uv add 'gpn[inference]'
```

```python
from gpn import register_auto_classes
from transformers import AutoModelForMaskedLM

MODEL_ID = "songlab/gpn-star-hg38-v100-200m"
REVISION = "0c949f132d35619a3eb188b402848c998a3313ae"

register_auto_classes("star")
model = AutoModelForMaskedLM.from_pretrained(
    MODEL_ID,
    revision=REVISION,
).eval()
```

The checkpoint bundles `phylo_dist/pairwise.npy` and
`phylo_dist/in_clade.npy`. The installed package resolves those portable assets
even though the historical config retains a training-system path. No custom loader
or `trust_remote_code` is required.

## Inputs and scores

GPN-Star does not use a text tokenizer. A forward pass requires integer alignment
tokens in the checkpoint's exact species order:

- `input_ids`: masked target-species tokens, shaped
  `(batch, length, target_species)`;
- `source_ids`: aligned source-species tokens, shaped
  `(batch, length, source_species)`; and
- `target_species`: the target column indices for the batch.

The model produces `-ACGT?` logits. A raw variant score is
`logit(alternate) - logit(reference)`. Calibrated LLRs additionally subtract the
matching neutral mean from `calibration_table/llr.parquet`; do not mix raw and
calibrated values. The maintained CLI documents strand averaging and genomic
coordinates.

## Intended use and limitations

- Research inference on hg38 positions represented by the compatible vertebrate
  100-way alignment.
- GPN-Star training is maintained only for already-prepared inputs. Alignment and
  dataset construction are outside the package contract.
- The checkpoint is not a clinical model, and its LLR is not a probability of
  pathogenicity.
- Assembly, target column, species order, preprocessing, and calibration release
  must all match. The model card intentionally does not instruct users to download
  a whole-genome MSA.

The tested immutable revision is
`0c949f132d35619a3eb188b402848c998a3313ae`. Please cite the
[GPN-Star preprint](https://doi.org/10.1101/2025.09.21.677619).
