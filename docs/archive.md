# Historical analysis archive

The historical research tree is preserved at commit
`30dee6cf45849dfdcfc043ca8baf44fd6ba51d74` by the annotated tag
`analysis-archive-2026-08-18`. The tag exists locally in the modernization stack
but must not be pushed or published as a GitHub release until final maintainer
approval.

## Snapshot manifest

| Archived content | Files | Git object |
| --- | ---: | --- |
| `analysis/` | 208 | tree `b2231967438488218791eeb55a3e0f29a8a52ea5` |
| `workflow/` | 20 | tree `063604ebb0a40d24df1576a4421d2cf465b3fbb1` |
| `examples/msa/` | 4 | tree `6083ac11310c3c9f28b6a8fa03efd9d0947925ce` |
| `gpn/msa/train.py` | 1 | blob `4727c2bf22db0d4459b270865bcf0567a6674d63` |

The `analysis/` count consists of 60 GPN-MSA files, 83 GPN-Star files,
15 animal-promoter files, 30 Brassicales files, 19 Sorghum-expression files,
and the archive README.

After the tag is published, inspect or check out the snapshot with:

```bash
git show analysis-archive-2026-08-18:analysis/ARCHIVE.md
git worktree add ../gpn-analysis-archive analysis-archive-2026-08-18
```

The archived code is scientific provenance, not supported software. It may rely
on institutional absolute paths, unavailable large inputs, mutable external
assets, and historical environments. Maintained replacements are the GPN and
GPN-Star prepared-data recipes, published-model fixtures, and the supported
model inference code. GPN-MSA training is intentionally not maintained.

## Proposed GitHub release

Title: `Historical GPN analysis archive (2026-08-18)`

Body:

> Immutable snapshot of the paper-specific and exploratory analyses removed
> from the maintained GPN package. This release preserves the former dataset
> workflows, GPN-MSA training implementation, and retired notebooks for
> provenance. The snapshot is unmaintained and may require reconstruction of
> historical environments and external inputs. Use the main branch for current
> GPN software, training recipes, inference, tests, and documentation.
