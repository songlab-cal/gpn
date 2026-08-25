# GPN — Genomic Pretrained Network

[![CI](https://github.com/songlab-cal/gpn/actions/workflows/ci.yml/badge.svg)](https://github.com/songlab-cal/gpn/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/gpn)](https://pypi.org/project/gpn/)
[![Python](https://img.shields.io/pypi/pyversions/gpn)](https://pypi.org/project/gpn/)
[![License](https://img.shields.io/github/license/songlab-cal/gpn)](https://github.com/songlab-cal/gpn/blob/main/LICENSE)

[**Quick start**](#quick-start) · [**Models**](#model-family) ·
[**Demos**](#demos) · [**Documentation**](https://github.com/songlab-cal/gpn/blob/main/docs/index.md) ·
[**Research & papers**](https://github.com/songlab-cal/gpn/blob/main/docs/development/research.md)

![GPN-Star architecture, evolutionary scales, and genomic prediction tasks](docs/_static/gpn_star_overview.png)

The GPN family of genomic language models.

GPN learns evolutionary constraint from DNA sequences and multispecies
alignments. The `gpn` package provides the canonical implementations for
published Song Lab checkpoints through explicit, standard
[Transformers AutoClasses](https://huggingface.co/docs/transformers/en/model_doc/auto).

## Quick start

Install the model APIs from PyPI:

```bash
pip install gpn
```

Load the published Brassicales model without mutable remote code or imports for
side effects:

```python
from gpn import register_auto_classes
from transformers import AutoModelForMaskedLM, AutoTokenizer

register_auto_classes("ss")

model_id = "songlab/gpn-brassicales"
model_revision = "eb9c35d0d18571abe84390d22e74f2b21d319ce3"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_revision)
model = AutoModelForMaskedLM.from_pretrained(
    model_id, revision=model_revision
).eval()
```

Use `register_auto_classes("msa")` for GPN-MSA,
`register_auto_classes("phylo")` for PhyloGPN, and
`register_auto_classes("star")` for GPN-Star. The package deliberately leaves
revision, dtype, device placement, and cache choices in the normal Transformers
API.

## Model family

| Model | What it learns from | Maintained support | Published checkpoint |
| --- | --- | --- | --- |
| [GPN](https://doi.org/10.1073/pnas.2311219120) | unaligned genomes | training + inference | [`songlab/gpn-brassicales`](https://huggingface.co/songlab/gpn-brassicales) |
| [GPN-MSA](https://doi.org/10.1038/s41587-024-02511-w) | a multispecies alignment | deprecated; inference only | [`songlab/gpn-msa-sapiens`](https://huggingface.co/songlab/gpn-msa-sapiens) |
| [PhyloGPN](https://doi.org/10.1007/978-3-031-90252-9_7) | phylogenetic substitution rates | inference only | [`songlab/PhyloGPN`](https://huggingface.co/songlab/PhyloGPN) |
| [GPN-Star](https://doi.org/10.1101/2025.09.21.677619) | alignments across evolutionary scales | training + inference | [`songlab/gpn-star-hg38-v100-200m`](https://huggingface.co/songlab/gpn-star-hg38-v100-200m) |

The [sorghum gene-expression checkpoint](https://huggingface.co/songlab/gpn-brassicales-gxa-sorghum-v1)
is an inference-only fine-tune of GPN, not a separate model family.

The [support guide](https://github.com/songlab-cal/gpn/blob/main/docs/models/index.md)
defines required inputs, compatibility boundaries, immutable validated revisions,
and output semantics. Classification in a Hugging Face collection does not
automatically make every historical asset part of the package support contract.

## Command line

Install file-backed inference dependencies with `pip install "gpn[inference]"` or
training dependencies with `pip install "gpn[train]"`.

```text
gpn ss {train,vep,logits,embedding} ...
gpn msa {vep,logits,embedding} ...
gpn star {train,vep,logits,embedding} ...
```

GPN-MSA has no training command. Dataset and whole-genome-alignment construction
are outside the maintained package. See the
[CLI guide](https://github.com/songlab-cal/gpn/blob/main/docs/getting-started/cli.md)
for local MSA layouts, coordinate systems, precision controls, distributed
execution, durable GPN-Star checkpoints, and raw LLR semantics.
PhyloGPN and the sorghum gene-expression fine-tune are maintained through explicit
Transformers AutoClass registration and intentionally have no dedicated CLI.

## Demos

Only three existing demos are maintained, with portable setup, explicit
registration, immutable model revisions, and small committed outputs. They render
statically in the documentation without requiring a notebook kernel.

| GPN | PhyloGPN | GPN-Star |
| --- | --- | --- |
| [Notebook](https://github.com/songlab-cal/gpn/blob/main/colabs/gpn_demo.ipynb) | [Notebook](https://github.com/songlab-cal/gpn/blob/main/colabs/phylogpn_demo.ipynb) | [Notebook](https://github.com/songlab-cal/gpn/blob/main/colabs/gpn_star_demo.ipynb) |

The GPN-Star demo uses a 3.5 KiB fixture from the published locus; it never
downloads a whole-genome MSA. See the [alignment guide](https://github.com/songlab-cal/gpn/blob/main/docs/models/gpn-star.md#alignment-data)
when running the CLI against the full public alignment.

For a lightweight executable workflow, use the
[precomputed GPN-Star score notebook](https://github.com/songlab-cal/gpn/blob/main/colabs/gpn_star_precomputed_scores.ipynb)
to score the OMIM TraitGym benchmark and compute global AUPRC without downloading
a model or MSA.

## Reproducible science

Every supported family has an immutable published revision and an approved
fixture-backed numerical regression. Normal pull-request tests are offline; the
published checkpoints are audited deliberately rather than through a recurring
Hub monitor.

- [Scientific validation and score conventions](https://github.com/songlab-cal/gpn/blob/main/docs/development/validation.md)
- [GPN training on prepared data](https://github.com/songlab-cal/gpn/tree/main/recipes/gpn_training)
- [GPN-Star training on prepared data](https://github.com/songlab-cal/gpn/tree/main/recipes/gpn_star_training)
- [Research branch lifecycle](https://github.com/songlab-cal/gpn/blob/main/docs/development/research.md)

Variant scores use `alternate_logit - reference_logit`; negative values mean the
alternate is less likely under the model. Genomic intervals are zero-based and
half-open, while VEP positions are one-based.

## Development and help

GPN supports one reproducible runtime: Python 3.13 with the exact Transformers
version declared by the package and the complete environment pinned in `uv.lock`:

```bash
git clone https://github.com/songlab-cal/gpn.git
cd gpn
uv sync --extra train --group dev --group docs
uv run pre-commit run --all-files
uv run pytest
```

- Ask usage questions in [Discussions](https://github.com/songlab-cal/gpn/discussions).
- Report bugs and scientific regressions in [Issues](https://github.com/songlab-cal/gpn/issues).
- Read [CONTRIBUTING.md](https://github.com/songlab-cal/gpn/blob/main/CONTRIBUTING.md)
  before proposing maintained or off-main research work.

## Citation

Please cite the paper for each model or fine-tuned application you use. Copyable
BibTeX entries are collected in the [citation guide](https://github.com/songlab-cal/gpn/blob/main/docs/reference/citations.md).

GPN is developed in the [Song Lab at UC Berkeley](https://people.eecs.berkeley.edu/~yss/group.html)
and distributed under the [MIT License](https://github.com/songlab-cal/gpn/blob/main/LICENSE).
