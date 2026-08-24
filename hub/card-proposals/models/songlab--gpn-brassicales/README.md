---
license: mit
library_name: transformers
pipeline_tag: fill-mask
inference: false
datasets:
- songlab/genomes-brassicales-balanced-v1
tags:
- biology
- dna
- genomics
- language-model
- variant-effect-prediction
---

# GPN Brassicales

This is the published GPN masked DNA language model trained on *Arabidopsis
thaliana* and seven other Brassicales genomes. It can produce sequence embeddings,
masked-nucleotide logits, and variant scores. See the
[GPN paper](https://doi.org/10.1073/pnas.2311219120),
[package documentation](https://github.com/songlab-cal/gpn#readme), and
[quick start](https://github.com/songlab-cal/gpn/blob/main/colabs/gpn_quick_start.ipynb).

## Install and load

The installed `gpn` package is the canonical model implementation. Registration is
explicit and has no network side effects.

```bash
uv add gpn
```

```python
from gpn import register_auto_classes
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL_ID = "songlab/gpn-brassicales"
REVISION = "eb9c35d0d18571abe84390d22e74f2b21d319ce3"

register_auto_classes("ss")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
model = AutoModelForMaskedLM.from_pretrained(
    MODEL_ID,
    revision=REVISION,
).eval()
```

Use `AutoModel` instead when only hidden representations are needed. No
`trust_remote_code` flag or import for registration side effects is required.
Hosted inference is disabled because the custom architecture must first be
registered from the installed package; use the local registered AutoClass path
above instead of a generated registration-free widget snippet.

## Inputs and outputs

The tokenizer accepts DNA sequence text and provides the checkpoint's seven-token
vocabulary. `AutoModelForMaskedLM` returns one logit per token at every input
position. To score a substitution at a masked position, use
`logit(alternate) - logit(reference)`: more negative values indicate that the model
prefers the reference allele more strongly. See the package's scientific validation
guide for strand averaging and coordinate conventions.

## Intended use and limitations

- Intended for research on plant genomic sequence, embeddings, and variant-effect
  prediction, especially in Brassicales.
- The checkpoint is not a clinical model and its scores are not probabilities of
  pathogenicity or direct measurements of organismal fitness.
- Performance can shift across species, assemblies, sequence contexts, and variant
  classes that differ from the training and evaluation data.
- Dataset construction is historical and is not a maintained package workflow. GPN
  training on already-prepared data remains supported.

## Provenance and citation

The tested model revision is
`eb9c35d0d18571abe84390d22e74f2b21d319ce3`. The current installed-package
compatibility record and expected likelihoods live in the
[`gpn` repository](https://github.com/songlab-cal/gpn).

Please cite [Benegas, Batra, and Song,
PNAS 2023](https://doi.org/10.1073/pnas.2311219120).
