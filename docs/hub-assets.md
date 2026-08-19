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
availability. The repository now keeps:

- a [machine-readable supported-model manifest](https://github.com/songlab-cal/gpn/blob/main/hub/manifest.json);
- a [dated compatibility report](https://github.com/songlab-cal/gpn/blob/main/hub/audits/2026-08-19.md)
  covering the broader public asset inventory; and
- [review-only card proposals](https://github.com/songlab-cal/gpn/tree/main/hub/card-proposals).

All five supported checkpoint heads matched their approved revisions in the dated
audit. Card, config, and collection findings are documented separately from runtime
compatibility: an empty card does not make a numerically validated checkpoint fail,
and collection membership does not create a package-support promise.

The audit is deliberate rather than scheduled. Normal CI validates the committed
manifest and report without contacting the Hub. Model-card and collection writes
require maintainer review and explicit authorization; local proposals are never
published automatically.
