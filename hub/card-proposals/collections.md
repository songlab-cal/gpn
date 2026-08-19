# Proposed collection updates

These are review notes, not API payloads.

## GPN

Description: “Published GPN model, tokenizer, and research artifacts for the PNAS
2023 Brassicales study. The `gpn` package is the canonical maintained
implementation; collection membership does not imply ongoing dataset-workflow
support.”

Add item notes distinguishing checkpoint, standalone tokenizer, historical training
data, and personal-namespace processed analysis data. Identify
`gonzalobenegas/processed-data-arabidopsis` as externally owned or remove it from the
curated collection after maintainer review.

## GPN-MSA

Description: “GPN-MSA publication assets. The model is deprecated and maintained
for inference only; training and dataset construction are historical. New
alignment-based training should use GPN-Star.”

Label the checkpoint, genome-wide scores, and evaluation datasets separately. Do
not imply that every dataset carries Song Lab or MIT licensing over its upstream
source.

## GPN-Star

Keep the current model/score role notes, then add the package documentation, support
policy, and paper link. Decide whether to add all 16 non-collected public model-size
variants or instead say explicitly that the collection curates only the three 200M
human models.

## Sorghum gene expression

Description: “Inference checkpoint and associated 26-output Sorghum expression
dataset for the Nature Biotechnology 2026 study. Model inference is maintained in
`gpn`; dataset construction and fine-tuning are historical.”

Add the publication link and item notes for the model and dataset. Resolve their
licenses and upstream terms before applying the update.

## TraitGym

Keep the paper-level collection, but describe `gpn-animal-promoter` as a published
research checkpoint rather than part of the five-checkpoint maintained support
matrix. Replace its implicit-registration example through a separate reviewed card
update.
