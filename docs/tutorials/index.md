# Demos

The three model demos are existing scientific notebooks with portability and
correctness fixes. The fourth notebook is a lightweight executable workflow for
joining variants to public GPN-Star scores; it does not download a model or MSA.

```{toctree}
:hidden:
:maxdepth: 1

../_notebooks/gpn_demo
../_notebooks/phylogpn_demo
../_notebooks/gpn_star_demo
../_notebooks/gpn_star_precomputed_scores
```

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} GPN demo
:link: ../_notebooks/gpn_demo
:link-type: doc

Tokenization, embeddings, masked nucleotide probabilities, and plots for the
published Brassicales checkpoint.
:::

:::{grid-item-card} PhyloGPN demo
:link: ../_notebooks/phylogpn_demo
:link-type: doc

Rate parameters, nucleotide probabilities, and a zero-shot substitution score.
:::

:::{grid-item-card} GPN-Star demo
:link: ../_notebooks/gpn_star_demo
:link-type: doc

Alignment-aware logits, raw LLRs, and mutation-rate calibration using the tiny
published fixture.
:::

:::{grid-item-card} Precomputed GPN-Star scores
:link: ../_notebooks/gpn_star_precomputed_scores
:link-type: doc

Join a variant table to immutable, chromosome-sharded genome-wide scores without
downloading a model or whole-genome alignment.
:::

::::

Read the committed code and plots here without a kernel. Deliberate output
refreshes use the pinned Python 3.13 environment and guarded process documented in
{doc}`../development/index`.
