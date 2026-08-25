# Models

GPN comprises four related model families with different biological inputs and
maintenance boundaries. Support means that an approved published checkpoint
loads through the documented installed-package API and has a fixture-backed
numerical regression; it does not imply support for every historical workflow.

| Family | Published checkpoint | Support | Required input | Registration |
| --- | --- | --- | --- | --- |
| {doc}`GPN <gpn>` | `songlab/gpn-brassicales` | training + inference | DNA sequence or local reference genome | `register_auto_classes("ss")` |
| {doc}`GPN-MSA <gpn-msa>` | `songlab/gpn-msa-sapiens` | deprecated; inference only | compatible local multispecies Zarr | `register_auto_classes("msa")` |
| {doc}`PhyloGPN <phylogpn>` | `songlab/PhyloGPN` | inference only | DNA sequence | `register_auto_classes("phylo")` |
| {doc}`GPN-Star <gpn-star>` | `songlab/gpn-star-hg38-v100-200m` | training + inference | compatible local MSA hierarchy | `register_auto_classes("star")` |

## Standard loading pattern

```python
from gpn import register_auto_classes
from transformers import AutoModelForMaskedLM

register_auto_classes("ss")
model = AutoModelForMaskedLM.from_pretrained(
    "songlab/gpn-brassicales",
    revision="eb9c35d0d18571abe84390d22e74f2b21d319ce3",
)
```

Use the Transformers AutoClass appropriate to the task. GPN intentionally does
not wrap `from_pretrained`: callers retain direct control of immutable revisions,
device placement, dtype, and local cache behavior.

## Compatibility boundaries

- GPN and GPN-Star consume already-prepared training data; raw-data and alignment
  construction are outside the package contract.
- GPN-MSA training is retired. New alignment-based training should use GPN-Star.
- A checkpoint and MSA must agree on assembly, target genome, species count,
  species order, preprocessing, and tokenizer.
- PhyloGPN has no maintained training or fine-tuning path.

The {ref}`sorghum-expression` checkpoint is an inference-only fine-tune of GPN,
not a separate model family.

Immutable revisions and expected outputs are recorded in
[`tests/fixtures/published_model_baseline.json`](https://github.com/songlab-cal/gpn/blob/main/tests/fixtures/published_model_baseline.json).

```{toctree}
:hidden:
:maxdepth: 1

gpn
gpn-msa
phylogpn
gpn-star
```
