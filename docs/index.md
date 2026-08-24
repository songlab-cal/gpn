---
myst:
  html_meta:
    description: Genomic pretrained networks for sequence and alignment-based inference.
---

<div class="gpn-hero">

# Genomic Pre-trained Network

GPN is a family of genomic language models for learning constraint from DNA
sequences and multispecies alignments. The maintained Python package gives
scientists explicit, versioned access to published checkpoints through standard
Transformers AutoClasses.

</div>

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item-card} Start with a model
:link: models
:link-type: doc

Choose the supported model family and understand its inputs and outputs.
:::

:::{grid-item-card} Explore the demos
:link: tutorials/index
:link-type: doc

Read annotated code and committed scientific plots without starting a kernel.
:::

:::{grid-item-card} Reproduce the science
:link: scientific-validation
:link-type: doc

Inspect immutable model revisions, fixture provenance, and score conventions.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

installation
models
cli
alignments
tutorials/index
scientific-validation
api
hub-assets
research
archive
development
citations
```

## What is maintained?

- **GPN and GPN-Star:** training on prepared inputs and inference.
- **GPN-MSA:** deprecated, inference only.
- **PhyloGPN:** inference only.
- **Sorghum gene expression:** inference-only fine-tune of GPN.
- **Research workflows and dataset construction:** preserved historically, not
  maintained on `main`.

The [model support matrix](models.md) is the authoritative starting point.
