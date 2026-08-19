---
license: mit
library_name: transformers
inference: false
datasets:
- songlab/gpn-animal-promoter-dataset
tags:
- biology
- dna
- genomics
- language-model
- variant-effect-prediction
---

# GPN Animal Promoter

This GPN-family checkpoint was trained on animal promoter sequence for the
[TraitGym study](https://doi.org/10.1101/2025.02.11.637758). It is a published
research asset, but it is not one of the five checkpoints in the maintained
numerical support matrix. Collection membership should not be read as a package
compatibility commitment.

## Load with the canonical package implementation

```bash
uv add gpn
```

```python
from gpn import register_auto_classes
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL_ID = "songlab/gpn-animal-promoter"
OBSERVED_REVISION = "7cf3276a03b5e243efd421b8939ed3d1e7dcf9cc"

register_auto_classes("gpn")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    revision=OBSERVED_REVISION,
)
model = AutoModelForMaskedLM.from_pretrained(
    MODEL_ID,
    revision=OBSERVED_REVISION,
).eval()
```

This replaces the obsolete implicit `import gpn.model` pattern. The installed
package is canonical; no model code should be copied into this card.

## Scope and limitations

Use this asset to reproduce the related research on animal promoter sequences and
consult `songlab/gpn-animal-promoter-dataset` for its training data. The checkpoint
has not received the fixture-backed likelihood regression used for the maintained
GPN Brassicales checkpoint. Users should independently validate model/package
compatibility and biological performance for their species, assembly, sequence
window, and downstream task.

TODO(maintainer): add the exact species composition, promoter/window definition,
training objective and hyperparameters, evaluation summary, and dataset provenance
before publishing this replacement card.

Please cite the [TraitGym preprint](https://doi.org/10.1101/2025.02.11.637758) when
using this research checkpoint.
