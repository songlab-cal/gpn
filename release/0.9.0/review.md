# GPN 0.9.0 review packet

Status: **final release candidate; version 0.9.0 remains unreleased. Publishing
the release still requires explicit approval of the exact final PR head and
tree.**

The single pull request, [#100](https://github.com/songlab-cal/gpn/pull/100), was
the binding review surface for the complete modernization. It was squash-merged
as `ebb3df3548fc8f364c339f0e5ddad86330c1f949`; the resulting tree
`aebc93d290f5d202511827be724cec70acefe07a` matched the approved PR tree. The
subsequent documentation-hosting follow-up was merged as
`305c29a1db9bf327c7d2bc049b8800d8dc131fdb`. The final release PR is a small,
separate review surface for the dated changelog, applied-action evidence, and
release binding.

## Candidate boundary

- Release version: `0.9.0`
- Release-PR base `main`: `305c29a1db9bf327c7d2bc049b8800d8dc131fdb`
- Review PR: [#100](https://github.com/songlab-cal/gpn/pull/100)
- Final release PR: [#110](https://github.com/songlab-cal/gpn/pull/110)
- Modernization merge: `ebb3df3548fc8f364c339f0e5ddad86330c1f949`
- Reviewed modernization tree: `aebc93d290f5d202511827be724cec70acefe07a`
- Maintained package: `src/gpn`
- Historical analysis: removed from `main` and published at the immutable
  `analysis-archive-2026-08-18` tag
- Dataset-building workflows and GPN-MSA training: unsupported and absent
- GPN-MSA: deprecated inference only
- Supported inference families: GPN, GPN-MSA, PhyloGPN, and GPN-Star; supported
  GPN checkpoints include the Sorghum gene-expression fine-tune

## Evidence reviewed

- Offline published-model fixtures cover all supported families with immutable
  model revisions and provenance.
- Three existing model demos are retained as static, non-executed documentation;
  their recorded outputs were not recomputed for the package-version bump. A
  fourth executable Colab was run on a Slurm CPU allocation and retains its table
  previews and global AUPRC output. It joins the pinned OMIM TraitGym
  regulatory-variant benchmark to immutable precomputed GPN-Star scores without
  downloading a model or MSA.
- The TraitGym and score datasets used by the executable workflow are
  commit-pinned; institutional filesystem paths were not committed as supported
  configuration. The repository documentation inventories the assets in each
  model-family collection. Auditing and editing the Hub-hosted cards and
  collection descriptions are deferred to issue #81 and absent from this
  milestone.
- The historical-versus-current PhyloGPN output discrepancy is documented in
  issue #108. The current immutable 2026 checkpoint is reproducible through both
  the pinned Hub implementation and maintained local implementation; determining
  the intent of the earlier checkpoint replacement is explicitly deferred until
  after 0.9.0 by the maintainer.
- A one-off remote lazy-Parquet validation matched all 3,380 OMIM TraitGym rows
  with no missing variants. The joined scores produced global AUPRC
  `0.764397828826`, versus `0.764514758694` from the released
  `GPN-Star-M447.parquet` prediction file.
- The root dependency graph excludes historical analysis and dataset-building
  environments. Research branches are documented as independent uv projects that
  pin a GPN version or commit.
- AutoClass registration is explicit through `gpn.register_auto_classes()`; there
  is no import-for-side-effects contract or public `load_model` abstraction.
- The family-first CLI uses named scientific inputs. Multi-process inference is
  launched through `torchrun`; every rank predicts while rank zero alone commits
  checkpoint shards and the final output.

## Validation required at the final candidate

The final candidate tip must pass the complete locked Python 3.13 suite, strict
Sphinx build with notebook execution disabled, every pre-commit hook, artifact
reproducibility checks, a clean wheel install, an independent complete-diff
review, and all required GitHub CI contexts. Exact counts and artifact hashes
belong in the pull-request body because the source distribution
contains this review packet and cannot contain its own stable hash.

## External mutation boundary

[`../external-mutations.json`](../external-mutations.json) is authoritative. Only
entries marked `approval_ready` may be included in the final maintainer decision.
Entries marked `deferred`, if any, remain unauthorized and require separate
future review. Before asking for approval for the remaining actions, the review
surface must state:

1. the exact final release-PR head and tree;
2. the exact `publish-gpn-0-9-0` action requested; and
3. that every deferred action is excluded.

Squash merging creates a new commit ID. After merging the single PR, verify that
the resulting `main` tree equals the approved tree, record the new `main` commit,
and stop if it differs. No tag or release is created merely by merging.
