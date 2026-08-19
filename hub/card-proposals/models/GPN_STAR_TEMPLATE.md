# Shared GPN-Star card template — do not publish verbatim

Apply the canonical V100 card structure to every public GPN-Star checkpoint, but
replace each bracketed field with verified repository facts:

- model ID and immutable revision;
- target assembly and target column;
- alignment family, exact species count/order, and public alignment artifact;
- model size;
- compatible calibration table and score release; and
- individually validated versus inventory-only status.

Every card should retain the installed-package `register_auto_classes("star")` plus
`AutoModelForMaskedLM.from_pretrained(..., revision=...)` pattern, describe
`input_ids`, `source_ids`, and `target_species`, distinguish raw from calibrated
LLRs, and state that users must provide a compatible local MSA. Do not add a
whole-alignment download to a quick start.

Use the public inventory in `hub/audits/2026-08-19.json` for observed revisions,
not as proof of numerical validation. In particular:

- V100 maps to the vertebrate 100-way family and has an approved numerical fixture;
- P243 likely maps to the public `243.tar.gz`, but exact order/checksum must be
  verified before publication; and
- M447's deployable public archive mapping remains a `TODO(maintainer)`—do not infer
  it from a cluster directory or checkpoint name.

The same template applies to the mm39 V35, galGal6 V77, dm6 I124, ce11 N135, and
tair10 B18 model-size variants after their respective public alignment archives,
species files, trees, and checksums are bound explicitly.
