# Research branches

The `main` branch is the maintained GPN package: model implementations, supported
training and inference, tests, documentation, and small scientific fixtures.
Paper-specific and exploratory analysis remains valuable, but it evolves under a
different lifecycle and must not be merged into `main`.

## Earlier research

The original research monorepo is preserved by the annotated tag
[`analysis-archive-2026-08-18`](https://github.com/songlab-cal/gpn/tree/analysis-archive-2026-08-18).
It contains the full historical `analysis/` tree, former dataset-building
workflows, the retired GPN-MSA training implementation, and retired notebooks.
The {doc}`model pages <../models/index>` collect the associated papers,
checkpoints, datasets, benchmarks, and score resources.

## Paper and exploratory branches

Future GPN-related analysis should stay in this GitHub repository so citations,
stars, issues, and project identity remain concentrated. Such work may start from
any ancestry, use any branch name, and remain available indefinitely without
formal branch protection. It must stay off `main`.

The following are non-binding recommendations, not repository requirements:

- keep a project under `analysis/<project-name>/` on its research branch;
- give that subdirectory its own `pyproject.toml`, `uv.lock`, and
  `.python-version`, so research dependencies never enter the root package;
- include a short README describing purpose, status, important inputs, primary
  commands, and the pinned GPN release or Git commit;
- pin GPN and otherwise let the project evolve independently instead of repeatedly
  merging or rebasing changes from `main`;
- keep project changes inside its subdirectory and propose reusable package work
  separately to `main`; contributing reusable code is encouraged but optional,
  and duplicated non-core analysis code is acceptable; and
- create an annotated tag for a publication while retaining its branch.

Each project decides its own tools, tests, CI, storage, and documentation. This
repository does not impose a project template or project-level agent instructions.
