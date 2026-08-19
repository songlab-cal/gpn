# Quick starts

These are the only notebooks maintained on `main`. They are existing scientific
quick starts with portability and correctness fixes; they do not add new training,
VEP, GPN-MSA, or Sorghum tutorials.

```{toctree}
:hidden:
:maxdepth: 1

../_notebooks/gpn_quick_start
../_notebooks/gpn_star_quick_start
../_notebooks/phylogpn_quick_start
```

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item-card} GPN quick start
:link: ../_notebooks/gpn_quick_start
:link-type: doc

Tokenization, embeddings, masked nucleotide probabilities, and plots for the
published Brassicales checkpoint.
:::

:::{grid-item-card} GPN-Star quick start
:link: ../_notebooks/gpn_star_quick_start
:link-type: doc

Alignment-aware logits, raw LLRs, and mutation-rate calibration using the tiny
published fixture.
:::

:::{grid-item-card} PhyloGPN quick start
:link: ../_notebooks/phylogpn_quick_start
:link-type: doc

Rate parameters, nucleotide probabilities, and a zero-shot substitution score.
:::

::::

Read the committed code and plots here without a kernel, or use the explicit
“Open in Colab” link at the top of a notebook when you want to execute it.
