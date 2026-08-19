# Historical analysis archive

This directory is the final historical snapshot of the GPN research analyses as
of 2026-08-18. It is preserved by the proposed annotated tag
`analysis-archive-2026-08-18` and is not part of the maintained Python package.

The snapshot contains the analysis associated with:

- GPN on Brassicales: `gpn_arabidopsis/`;
- GPN on animal promoters and TraitGym: `gpn_animal_promoter/`;
- GPN-MSA: `gpn-msa_human/`;
- GPN-Star: `gpn-star/`; and
- Sorghum gene-expression fine-tuning: `gpn_sorghum_expression/`.

The code is retained for scientific provenance, not as a supported workflow. It
is not tested against current dependencies and will not receive compatibility or
security maintenance. Some steps depend on institutional absolute paths, omitted
large inputs, partial or unlocked environments, and mutable external resources.
Historical commands may therefore require reconstruction.

Maintained successors live outside this directory:

- canonical GPN and GPN-Star recipes under `recipes/`, which begin with already
  prepared inputs and do not build datasets;
- published-model likelihood fixtures under `tests/fixtures/`;
- maintained GPN-MSA inference (training is deprecated and unsupported); and
- maintained Sorghum and PhyloGPN inference.

After the archive tag is published, inspect this snapshot without changing the
current checkout:

```bash
git show analysis-archive-2026-08-18:analysis/ARCHIVE.md
git worktree add ../gpn-analysis-archive analysis-archive-2026-08-18
```

Subdirectories may contain their own licensing or attribution files; those remain
authoritative for their contents. The repository-level license does not replace
third-party licenses associated with external data, tools, or copied resources.
