---
license: mit
library_name: transformers
inference: false
datasets:
- songlab/gpn-msa-sapiens-dataset
- songlab/multiz100way
tags:
- biology
- deprecated
- dna
- genomics
- language-model
- variant-effect-prediction
---

# GPN-MSA Sapiens

> **Deprecated — inference only.** GPN-MSA training and dataset construction are no
> longer maintained. New alignment-based training should use GPN-Star.

This checkpoint was trained for human inference with a 100-way vertebrate alignment
using human as the target and 89 ordered auxiliary species. See the
[paper](https://doi.org/10.1038/s41587-024-02511-w) and
[inference documentation](https://github.com/songlab-cal/gpn/blob/main/docs/cli.md#gpn-msa).

## Install and load

```bash
uv add 'gpn[inference]'
```

```python
from gpn import register_auto_classes
from transformers import AutoModelForMaskedLM

MODEL_ID = "songlab/gpn-msa-sapiens"
REVISION = "4a7d4f75449cb2abd560b2af024d76f99233c6db"

register_auto_classes("msa")
model = AutoModelForMaskedLM.from_pretrained(
    MODEL_ID,
    revision=REVISION,
).eval()
```

Do not use the generic Hugging Face fill-mask widget or NLP pipeline. Valid forward
passes require two aligned tensors:

- `input_ids`: target-genome tokens with shape `(batch, length)`; and
- `aux_features`: the same positions for the exact 89 auxiliary species, in the
  published order, with shape `(batch, length, 89)`.

The checkpoint config internally expands the categorical auxiliary tokens to 445
features. The exact species order and a 128 bp regression fixture are recorded in
the [package scientific baseline](https://github.com/songlab-cal/gpn/blob/main/tests/fixtures/published_model_baseline.json).
Real analyses should point the documented CLI at an already-prepared, compatible
local Zarr store; this repository does not maintain or download a whole-genome MSA.

## Scores and limitations

Masked logits use the `-ACGT?` token order. Variant scores are
`logit(alternate) - logit(reference)` and are averaged with the reverse-complement
score by the maintained VEP command. They are raw model scores, not calibrated
pathogenicity probabilities.

Compatibility requires the same hg38 coordinate system, sequence preprocessing,
species set, and species order used by the checkpoint. A different MSA cannot be
substituted solely because it has the same number of species. GPN-MSA is retained
for reproducing published inference; it should not be selected for new training.

The tested immutable revision is
`4a7d4f75449cb2abd560b2af024d76f99233c6db`. Please cite the
[GPN-MSA publication](https://doi.org/10.1038/s41587-024-02511-w).
