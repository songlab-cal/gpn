# Repository governance audit — 2026-08-19

This is a read-only snapshot of `songlab-cal/gpn` before the final modernization
review. It contains no credentials, alert secrets, or private asset names.

## Observed settings

| Control | Observed state |
| --- | --- |
| Default branch | `main` |
| Branch protection / repository rulesets | none |
| Merge methods | squash only |
| Delete merged branches | enabled |
| Private vulnerability reporting | disabled |
| Dependabot security updates | disabled |
| Secret scanning and push protection | disabled |
| Code scanning | no analysis configured |
| `pypi` environment | present; no deployment protection rules |

The workflows already set top-level `contents: read`, disable persisted checkout
credentials, and grant `id-token: write` only to the PyPI publish job. The release
workflow separates build and publish jobs.

## Dependency alerts

Three open Dependabot alerts point to the historical
`analysis/gpn-star/interpretation/requirements.txt`, which is removed in the
modernization stack:

| Advisory | Severity | Affected Transformers | First patched |
| --- | --- | --- | --- |
| `GHSA-69w3-r845-3855` / `CVE-2026-1839` | medium | `<5.0.0rc3` | `5.0.0rc3` |
| `GHSA-29pf-2h5f-8g72` / `CVE-2026-4372` | high | `<5.3.0` | `5.3.0` |
| `GHSA-fgcw-684q-jj6r` / `CVE-2026-5241` | high | `<5.5.0` | `5.5.0` |

The modernization package and lock both pin Transformers 5.15.0, the version used
for the committed scientific fixtures and canonical Python 3.13 environment. The
alerts should close naturally when the historical manifest is removed from
`main`; they must not be dismissed merely to make the dashboard green.

## Approval-gated rollout

After final review and merge, create the proposed ruleset, enable Dependabot
security updates, secret scanning, and push protection, then verify their
public/API state. Every required status context in
the ruleset is bound to the GitHub Actions app. Do not require a PR approval or
`pypi` environment reviewer while there is only one active maintainer; either
would create a recovery deadlock. Keep research branches outside the `main`
ruleset.

The exact proposed mutations and verification steps are recorded in
`external-mutations.json`. Only entries marked `approval_ready` are eligible for
the final modernization approval; `deferred` entries require later approval after
their blockers are resolved. This audit did not change a repository setting or
dismiss an alert.
