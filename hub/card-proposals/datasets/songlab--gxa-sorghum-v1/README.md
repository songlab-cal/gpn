---
# TODO(maintainer): add the precise dataset license and upstream source terms.
tags:
- biology
- gene-expression
- genomics
- sorghum
- tabular
---

# Sorghum gene-expression sequence dataset

This dataset accompanies the
[Sorghum gene-expression prediction study](https://doi.org/10.1038/s41587-026-03046-y)
and the `songlab/gpn-brassicales-gxa-sorghum-v1` inference checkpoint. It contains
40,702 examples split across `train.parquet`, `validation.parquet`, and
`test.parquet`, plus the ordered output names in `labels.txt`.

## Schema

- `chrom`: source chromosome;
- `start`, `end`: zero-based, half-open coordinates;
- `strand`: gene orientation;
- `seq`: 512 bp model input sequence; and
- `labels`: length-26 expression target vector on the `log(1 + TPM)` scale, in the
  exact order given by `labels.txt`.

Use immutable dataset revision
`0545539b3229946b90c1073c99a97bfb9f95cd83` when reproducing the published model
regression. The maintained package downloads only `labels.txt` and one small test
table for its opt-in numerical check; it does not maintain this dataset's
construction workflow.

## Required provenance before publication

TODO(maintainer): add the Sorghum reference assembly, TSS/window definition, RNA-seq
sources and accessions, chromosome split rules, filtering, and the exact
license/upstream terms. These facts must not be inferred from the serialized tables
alone.

The model's 26 outputs must be interpreted in `labels.txt` order. Its current
generic `LABEL_0` through `LABEL_25` config names are not biological labels.
