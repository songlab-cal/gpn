# Historical analysis archive

This maintainer-facing manifest records the exact contents of the archive. It is
kept outside the public documentation navigation because it describes a release
operation, not the supported package.

The historical research tree is preserved at commit
`30dee6cf45849dfdcfc043ca8baf44fd6ba51d74` by the annotated tag
`analysis-archive-2026-08-18`. The exact tag object
`312a6c70de6700e729bcea4c9a67ab42a72f05f7` and its lightweight
[GitHub Release](https://github.com/songlab-cal/gpn/releases/tag/analysis-archive-2026-08-18)
were published and verified on 2026-08-25.

## Snapshot manifest

| Archived content | Files | Git object |
| --- | ---: | --- |
| `analysis/` | 208 | tree `b2231967438488218791eeb55a3e0f29a8a52ea5` |
| `workflow/` | 20 | tree `063604ebb0a40d24df1576a4421d2cf465b3fbb1` |
| `examples/msa/` | 4 | tree `6083ac11310c3c9f28b6a8fa03efd9d0947925ce` |
| `gpn/msa/train.py` | 1 | blob `4727c2bf22db0d4459b270865bcf0567a6674d63` |

The same snapshot preserves package utilities retired during dependency
pruning. These paths are recorded separately because some useful historical
functions were removed from otherwise maintained data modules:

| Retired package content | Git object |
| --- | --- |
| `gpn/data.py` | blob `154bf78748edb2395af1f98ffea49e150a782f61` |
| `gpn/star/data.py` | blob `dfc0faef8ed11482a5c50f6b13d456d23c216499` |
| `gpn/ss/data.py` | blob `800e44fb06b96d81f8f51d8d14194adf418386e2` |
| `gpn/ss/filter_assemblies.py` | blob `50b95c39de373597ea04c1e50a372b5fe22d6c14` |
| `gpn/ss/finetune.py` | blob `ad19df07615d8991fc0eac83685be044bff5621c` |
| `gpn/ss/train_tokenizer_ss.py` | blob `eb86cba870ed766deab0336ddcc2f9aa5bf1046a` |

The `analysis/` count consists of 30 Brassicales files, 15 animal-promoter files,
60 GPN-MSA files, 83 GPN-Star files, 19 Sorghum-expression files, and the archive
README.

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

## Published GitHub release

Title: `Historical GPN analysis archive (2026-08-18)`

Body:

> Immutable snapshot of the paper-specific and exploratory analyses removed
> from the maintained GPN package. This release preserves the former dataset
> workflows, GPN-MSA training implementation, and retired notebooks for
> provenance. The snapshot is unmaintained and may require reconstruction of
> historical environments and external inputs. Use the main branch for current
> GPN software, training recipes, inference, tests, and documentation.
