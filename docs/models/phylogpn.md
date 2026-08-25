# PhyloGPN

PhyloGPN learns position-specific parameters of the Felsenstein F81 substitution
model from a phylogenetic alignment. Inference needs only DNA sequence, which
makes the published model useful for transfer learning and zero-shot variant
scoring away from a reference genome. The maintained package supports inference,
not training or fine-tuning.

## Quick start

```python
from gpn import register_auto_classes
from transformers import AutoModel, AutoTokenizer

register_auto_classes("phylo")

model_id = "songlab/PhyloGPN"
revision = "3556db4c469e67d25f0f7a0a6653b48be3eebf51"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
model = AutoModel.from_pretrained(model_id, revision=revision).eval()
```

The {doc}`PhyloGPN tutorial <../_notebooks/phylogpn_demo>` demonstrates rate
parameters, nucleotide probabilities, and a zero-shot substitution score. The
published model is [`songlab/PhyloGPN`](https://huggingface.co/songlab/PhyloGPN),
described in [A Phylogenetic Approach to Genomic Language Modeling](https://doi.org/10.1007/978-3-031-90252-9_7).
