# GPN 0.9.0 review packet

Status: **review candidate; no merge, release, tag, setting change, deployment,
or Hugging Face write is authorized by this file.**

The single pull request, [#100](https://github.com/songlab-cal/gpn/pull/100),
against `main` is the binding review surface for the complete modernization. Its
body records the exact head commit and Git tree after this packet is committed.
Because a Git object cannot contain its own hash, those two identifiers cannot be
embedded in this file; final maintainer approval must repeat both identifiers
from the pull request body.

## Candidate boundary

- Release version: `0.9.0`
- Base `main`: `690557d949309cf4f4234554888bb5421c49aede`
- Review PR: [#100](https://github.com/songlab-cal/gpn/pull/100)
- Maintained package: `src/gpn`
- Historical analysis: removed from `main`; local annotated archive tag prepared,
  but publication remains approval-gated
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
Entries marked `deferred` remain unauthorized and require separate future review.
Before asking for approval, the PR body must state:

1. its exact head commit and tree;
2. the exact `approval_ready` action IDs requested; and
3. that every `deferred` action is excluded.

Squash merging creates a new commit ID. After merging the single PR, verify that
the resulting `main` tree equals the approved tree, record the new `main` commit,
and stop if it differs. No tag or release is created merely by merging.
