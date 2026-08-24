---
# TODO(maintainer): add the precise model license after confirming it.
library_name: transformers
inference: false
datasets:
- songlab/gxa-sorghum-v1
base_model:
- songlab/gpn-brassicales
tags:
- biology
- dna
- gene-expression
- genomics
- regression
---

# GPN Sorghum gene-expression model

This checkpoint fine-tunes GPN sequence representations to predict a 26-element
nonnegative Sorghum gene-expression profile on the `log(1 + TPM)` scale from a
512 bp sequence. It accompanies the
[Nature Biotechnology publication](https://doi.org/10.1038/s41587-026-03046-y)
and is maintained for inference only; its dataset-building and fine-tuning workflow
is historical.

## Install and load

```bash
uv add gpn
```

```python
import torch
from pathlib import Path

from gpn import register_auto_classes
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "songlab/gpn-brassicales-gxa-sorghum-v1"
REVISION = "53209151b497d4840d50526d44c0460b6e6768b7"
DATASET_REVISION = "0545539b3229946b90c1073c99a97bfb9f95cd83"

register_auto_classes("ss")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    revision=REVISION,
).eval()

sequence = "A" * 512  # replace with a real 512 bp Sorghum sequence
inputs = tokenizer(
    sequence,
    return_tensors="pt",
    return_attention_mask=False,
    return_token_type_ids=False,
)
with torch.inference_mode():
    predictions = model(**inputs).logits[0]  # (26,), nonnegative log(1 + TPM)

labels_path = hf_hub_download(
    "songlab/gxa-sorghum-v1",
    "labels.txt",
    repo_type="dataset",
    revision=DATASET_REVISION,
)
labels = Path(labels_path).read_text().splitlines()
prediction_by_tissue = dict(zip(labels, predictions.tolist()))
```

The checkpoint config currently exposes generic `LABEL_0` through `LABEL_25`;
`labels.txt` at the pinned dataset revision is the authoritative output order.
Do not reorder the outputs alphabetically.

## Intended use and limitations

- Research inference of the published Sorghum expression targets from 512 bp
  sequences represented like the training examples.
- Outputs are nonnegative predictions on the `log(1 + TPM)` training-target scale.
  TODO(maintainer): document the exact reference assembly, TSS convention, RNA-seq
  accessions, and chromosome split before publishing this card.
- Predictions are not validated for other assemblies, species, window definitions,
  or experimental protocols.
- The generic Hugging Face text-classification widget is not an appropriate genomic
  interface for this 26-output regression model.

The tested immutable model revision is
`53209151b497d4840d50526d44c0460b6e6768b7`. The model and dataset licenses must
be confirmed before publishing this replacement card.
