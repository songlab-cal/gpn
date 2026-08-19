# Hugging Face assets

The installed `gpn` package is the canonical implementation of supported custom
model code. Model cards should use explicit registration plus standard Transformers
AutoClasses; they should not require imports for side effects or mutable copied
Python files.

## Supported checkpoint families

- [GPN collection](https://huggingface.co/collections/songlab/gpn-653191edcb0270ed05ad2c3e)
- [GPN-MSA collection](https://huggingface.co/collections/songlab/gpn-msa-65319280c93c85e11c803887)
- [GPN-Star collection](https://huggingface.co/collections/songlab/gpn-star-68c0c055acc2ee51d5c4f129)
- [PhyloGPN](https://huggingface.co/songlab/PhyloGPN)
- [Sorghum gene-expression collection](https://huggingface.co/collections/songlab/sorghum-gene-expression-prediction-68963dd31658bfb98c07ae1b)

The [model support page](models.md) distinguishes package support from historical
availability. A machine-readable full asset inventory and dated compatibility
report are prepared in the separate Hugging Face audit workstream before card or
collection updates are applied.
