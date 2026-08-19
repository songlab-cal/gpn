# Scientific validation

GPN protects published behavior in three layers.

## Offline architecture tests

Tiny deterministic configurations test AutoClass registration, forward passes,
serialization, coordinate and allele validation, and score direction without
network access.

## Fixture-backed regressions

Small checked-in fixtures reproduce the published quick-start inputs for GPN,
GPN-MSA, GPN-Star, PhyloGPN, and Sorghum expression. The MSA fixture is a compressed
128 bp slice, not a whole-genome alignment. Its source coordinates, public archive
revision, byte hashes, model revisions, dtype/device, and tolerances live in the
[fixture provenance](https://github.com/songlab-cal/gpn/blob/main/tests/fixtures/README.md).

## Published-checkpoint audit

The opt-in suite downloads immutable model revisions and verifies real outputs:

```bash
uv run pytest --run-published-models
```

This suite is run deliberately when a model asset, compatibility range, or golden
output changes. It is not part of normal CI and is not a scheduled Hub monitor.

## Score and coordinate vocabulary

- Nucleotide logits are unnormalized model outputs in documented `A, C, G, T`
  order.
- Probabilities are a softmax over those four logits.
- A raw variant LLR is `alternate_logit - reference_logit`; negative values mean
  the alternate is less likely under the model.
- GPN, GPN-MSA, and GPN-Star VEP average forward and reverse-complement LLRs.
- GPN-Star calibrated LLRs additionally subtract the neutral mean for the matching
  sequence context. The CLI emits raw scores; calibration is downstream.
- Genomic intervals are zero-based and half-open. VEP and logits positions are
  one-based, matching VCF conventions.
- Public VEP accepts distinct uppercase biallelic SNVs from `A`, `C`, `G`, `T` and
  fails if the reference does not match the local genome or MSA.
