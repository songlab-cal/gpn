# PhyloGPN

PhyloGPN learns position-specific parameters of the Felsenstein F81 substitution
model from a phylogenetic alignment. Inference needs only DNA sequence, which
makes the published model useful for transfer learning and zero-shot variant
scoring away from a reference genome. The maintained package supports inference,
not training or fine-tuning.

## Quick start

```python
from gpn import register_auto_classes
from transformers import AutoModel

register_auto_classes("phylo")

model = AutoModel.from_pretrained("songlab/PhyloGPN")
```

The {doc}`PhyloGPN tutorial <../_notebooks/phylogpn_demo>` demonstrates rate
parameters, nucleotide probabilities, and a zero-shot substitution score. The
published model is [`songlab/PhyloGPN`](https://huggingface.co/songlab/PhyloGPN),
described in [A Phylogenetic Approach to Genomic Language Modeling](https://doi.org/10.1007/978-3-031-90252-9_7).
