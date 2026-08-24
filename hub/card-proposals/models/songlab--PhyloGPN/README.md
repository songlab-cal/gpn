---
# TODO(maintainer): replace the current ambiguous `cc` value with the precise license.
library_name: transformers
inference: false
tags:
- biology
- dna
- genomics
- language-model
- variant-effect-prediction
---

# PhyloGPN

PhyloGPN is a convolutional model trained to predict F81 substitution-rate
parameters from DNA sequence using a mammalian phylogeny. Its stationary
distribution can be used to compare the relative viability of alleles at a locus,
and its hidden states can be used as sequence embeddings. See the
[paper](https://doi.org/10.1007/978-3-031-90252-9_7) and
[demo](https://github.com/songlab-cal/gpn/blob/main/colabs/phylogpn_demo.ipynb).

## Preferred installed-package path

```bash
uv add gpn
```

```python
import torch
from gpn import register_auto_classes
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "songlab/PhyloGPN"
REVISION = "3556db4c469e67d25f0f7a0a6653b48be3eebf51"

register_auto_classes("phylo")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
model = AutoModel.from_pretrained(MODEL_ID, revision=REVISION).eval()

sequences = ["TATAAA", "GGCCAATCT"]
pad = tokenizer.pad_token * 240
inputs = tokenizer(
    [pad + sequence + pad for sequence in sequences],
    return_tensors="pt",
    padding=True,
)["input_ids"]
with torch.inference_mode():
    log_rate_parameters = model(input_ids=inputs)
    padded_embeddings = model.get_embeddings(inputs)

log_rates = [
    {
        nucleotide: log_rate_parameters[nucleotide][index, : len(sequence)]
        for nucleotide in "ACGT"
    }
    for index, sequence in enumerate(sequences)
]
embeddings = [
    padded_embeddings[index, : len(sequence)]
    for index, sequence in enumerate(sequences)
]
```

The Hub repository retains copied Python files and `auto_map` as a legacy
remote-code compatibility path. The installed `gpn` implementation above is
canonical and does not require `trust_remote_code`.

## Inputs, outputs, and limitations

PhyloGPN has a 481-base receptive field. Symmetric padding of 240 bases preserves
one prediction per original input position. The forward result maps each nucleotide
in `ACGT` to its log rate parameter; `model.get_embeddings(input_ids)` returns the
corresponding hidden representations.

Allele comparisons follow the repository-wide convention
`log_rate(alternate) - log_rate(reference)` at the same position. An illustrative
comparison between two nucleotides does not itself verify that the stated reference
allele matches a reference genome. These are model-based evolutionary quantities,
not direct pathogenicity probabilities. Users should preserve sequence orientation,
padding, and allele order, and should expect distribution shift away from the
training phylogeny and species.

The tested immutable revision is
`3556db4c469e67d25f0f7a0a6653b48be3eebf51`. The exact license must be resolved
before publishing this replacement card.
