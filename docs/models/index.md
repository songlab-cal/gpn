# Models

GPN comprises four genomic language model families with different biological
inputs and modeling objectives.

| Model | Paper | Notes |
| --- | --- | --- |
| {doc}`GPN <gpn>` | [Benegas et al. 2023](https://doi.org/10.1073/pnas.2311219120) | Requires unaligned genomes |
| {doc}`GPN-MSA <gpn-msa>` | [Benegas et al. 2025](https://www.nature.com/articles/s41587-024-02511-w) | Requires aligned genomes for training and inference; deprecated in favor of GPN-Star |
| {doc}`PhyloGPN <phylogpn>` | [Albors et al. 2025](https://link.springer.com/chapter/10.1007/978-3-031-90252-9_7) | Uses an alignment during training, but does not require it for inference or fine-tuning |
| {doc}`GPN-Star <gpn-star>` | [Ye et al. 2025](https://doi.org/10.1101/2025.09.21.677619) | Requires aligned genomes for training and inference |

## Maintained support

GPN and GPN-Star support training on prepared data and inference. GPN-MSA is
deprecated and supports inference only. PhyloGPN and the sorghum gene-expression
model support inference only. Dataset and whole-genome-alignment construction
are outside the maintained package.

```{toctree}
:hidden:
:maxdepth: 1

gpn
gpn-msa
phylogpn
gpn-star
```
