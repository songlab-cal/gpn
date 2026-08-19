## Summary

<!-- Explain the user-visible or maintainer-visible outcome. -->

## Verification

<!-- List exact commands and any manual or scientific validation. -->

## Review checklist

- [ ] The change stays within the maintained scope in `AGENTS.md`.
- [ ] Tests, documentation, and dependency metadata are updated together.
- [ ] Normal tests remain offline; no model, dataset, or MSA is downloaded in CI.
- [ ] Scientific output changes include fixture provenance, tolerances, and an
      independent numerical review.
- [ ] Public API removals or deprecations are recorded in `CHANGELOG.md`.
- [ ] No secret, private asset, or institutional filesystem path is committed.
- [ ] External mutations (releases, tags, settings, PyPI, or Hub writes) are
      listed explicitly and have maintainer authorization.
