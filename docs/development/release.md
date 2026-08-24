# Maintainer release runbook

GPN releases are immutable, reviewed scientific artifacts. PyPI publishing is an
effect of publishing a GitHub Release; maintainers never upload a wheel or source
distribution from a workstation.

## Release boundary

- Package releases use semantic versions and tags of the form `v<version>`.
- Hugging Face model revisions are separate immutable compatibility inputs. A
  package release does not move or rewrite a model revision.
- `main` is the only release branch. The release workflow rejects a tag whose
  commit is not an ancestor of `main` or whose name differs from the package
  version.
- The `pypi` environment has no required human reviewer while there is only one
  active release maintainer. The release event, exact tag, ancestry check, locked
  build tools, isolated publish job, and PyPI Trusted Publisher are the safeguards.

## Before final approval

1. Freeze scope and assemble the cumulative review PR against `main`.
2. Keep Hugging Face asset auditing and card changes outside this release; that
   work remains tracked separately in issue #81.
3. Set the final package version and update `CHANGELOG.md`.
4. Confirm that the published-model fixture baseline uses the intended immutable
   revisions. Run the opt-in published-model tests deliberately if a compatibility
   input changed.
5. Complete the review packet and external-mutation manifest under `release/`.
   Record the exact cumulative head and tree plus the ordered component PR heads.
   A squash merge creates new commit IDs, so approval binds to the reviewed tree;
   record the resulting `main` commit and verify its tree immediately after merge.
6. Obtain explicit approval for the cumulative code diff and only the pending
   mutations marked `approval_ready`. Entries marked `deferred` are explicitly
   outside that approval and require a later, separate approval after their
   blockers are resolved. Keep unreviewed component PRs unmerged; merge reviewed
   components bottom-up only after explicit maintainer authorization.

## Reproduce the release candidate

Use a fresh clone or worktree checked out at the exact candidate commit so an old
`dist/` directory cannot contaminate the review. The commands below refuse a dirty
checkout and write artifacts to new temporary directories. Record
`candidate_commit`, `candidate_tree`, and the final hashes in the review packet.
Normal tests are network-free.

```bash
set -euo pipefail
test -z "$(git status --porcelain)"
candidate_commit="$(git rev-parse HEAD)"
candidate_tree="$(git rev-parse HEAD^{tree})"
artifact_dir="$(mktemp -d)"
rebuilt_wheel_dir="$(mktemp -d)"
uv sync --locked --python 3.13 --extra train --group dev --group docs --group release
uv run pre-commit run --all-files --show-diff-on-failure
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run pytest
python docs/prepare_notebooks.py
uv run sphinx-build -n -W --keep-going -b html docs docs/_build/release
test -z "$(git status --porcelain)"
uv build --no-build-isolation --out-dir "${artifact_dir}"
uv run --no-sync twine check "${artifact_dir}"/*
uv run --no-sync check-wheel-contents "${artifact_dir}"/*.whl
uv build --no-build-isolation "${artifact_dir}"/*.tar.gz \
  --wheel --out-dir "${rebuilt_wheel_dir}"
uv run --no-sync twine check "${rebuilt_wheel_dir}"/*.whl
uv run --no-sync check-wheel-contents "${rebuilt_wheel_dir}"/*.whl
sha256sum "${artifact_dir}"/* "${rebuilt_wheel_dir}"/*
```

Install the rebuilt wheel into a clean CPU environment outside the checkout; run
`gpn --version`, `gpn --help`, import `gpn`, and call
`gpn.register_auto_classes()`. Record SHA-256 hashes for both distributions in the
review packet. The rebuilt wheel hash should match the directly built wheel; if it
does not, investigate rather than selecting one artifact for release.

If a change affects training, device placement, precision, or notebook output,
also run its documented manual validation in a dedicated Slurm allocation with no
more than eight CPUs and one GPU. Never download a whole-genome MSA for release
validation.

## Merge and publish

After approval, merge the component PRs bottom-up and keep descendants restacked.
Do not publish while the final `main` tree differs from the approved cumulative
tip. Record the new `main` commit created by the squash merges, verify its tree is
the approved tree, and rerun the release-candidate checks on that commit.

1. Create the annotated historical archive tag/Release only if it is among the
   approved external mutations.
2. Create tag `v<version>` at the exact approved `main` commit and publish its
   GitHub Release. Do not run the publishing workflow manually and do not upload
   distributions yourself.
3. The release workflow builds once, passes the immutable artifacts between jobs,
   and publishes with the `pypi` environment, GitHub OIDC, and PyPI attestations.
4. Verify the GitHub Actions run, artifact hashes, PyPI metadata and attestations,
   and a clean `pip install gpn==<version>` on Python 3.13.
5. Apply only the separately approved Hub and documentation mutations, then verify
   their public rendering without moving pinned model revisions.

PyPI files cannot be replaced. If a release is defective, stop downstream
mutations, document the incident, yank the affected version when appropriate, and
publish a new patch release. Never reuse a tag or version.

## Repository rules

`release/main-ruleset.json` is the reviewable proposal for `main`. It requires a
PR, the four offline CI contexts, an up-to-date branch, and resolved review threads;
each context is bound to the GitHub Actions app rather than accepting a same-named
status from another source. It blocks force pushes and deletion. It intentionally
requires zero approvals so a solo maintainer is not deadlocked. The absence of
bypass actors means the rules also apply to administrators. In an emergency, an
administrator may temporarily disable the ruleset in repository settings, document
why, restore it immediately, and route the resulting change through a PR.

Research branches remain outside this ruleset and may live indefinitely. Automatic
deletion applies only when GitHub recognizes a branch as the merged PR head.
