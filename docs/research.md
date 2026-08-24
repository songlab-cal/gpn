# Research code lifecycle

The `main` branch is the maintained GPN package: model implementations, supported
training and inference, tests, documentation, and small scientific fixtures.
Paper-specific and exploratory analysis remains valuable, but it evolves under a
different lifecycle and must not be merged into `main`.

## Historical archive

The original research monorepo is preserved by the immutable tag
`analysis-archive-2026-08-18`. The tag is prepared locally during modernization
but will only be published after maintainer approval. It targets commit
`30dee6cf45849dfdcfc043ca8baf44fd6ba51d74` and contains the full `analysis/`
tree, the former dataset-building workflows, the GPN-MSA training implementation,
and the removed example notebooks. See the [exact archive manifest](archive.md).

| Project | Status and publication | Archived path | Related assets |
| --- | --- | --- | --- |
| GPN on Brassicales | Published, [PNAS 2023](https://doi.org/10.1073/pnas.2311219120) | [`analysis/gpn_arabidopsis/`](https://github.com/songlab-cal/gpn/tree/30dee6cf45849dfdcfc043ca8baf44fd6ba51d74/analysis/gpn_arabidopsis) | [GPN collection](https://huggingface.co/collections/songlab/gpn-653191edcb0270ed05ad2c3e) |
| GPN animal promoter / TraitGym | Preprint, [bioRxiv 2025](https://doi.org/10.1101/2025.02.11.637758) | [`analysis/gpn_animal_promoter/`](https://github.com/songlab-cal/gpn/tree/30dee6cf45849dfdcfc043ca8baf44fd6ba51d74/analysis/gpn_animal_promoter) | [TraitGym collection](https://huggingface.co/collections/songlab/traitgym-6796d4fbb825d5b94e65d30f) |
| GPN-MSA | Published, [Nature Biotechnology 2025](https://doi.org/10.1038/s41587-024-02511-w) | [`analysis/gpn-msa_human/`](https://github.com/songlab-cal/gpn/tree/30dee6cf45849dfdcfc043ca8baf44fd6ba51d74/analysis/gpn-msa_human) | [GPN-MSA collection](https://huggingface.co/collections/songlab/gpn-msa-65319280c93c85e11c803887) |
| GPN-Star | Preprint, [bioRxiv 2025](https://doi.org/10.1101/2025.09.21.677619) | [`analysis/gpn-star/`](https://github.com/songlab-cal/gpn/tree/30dee6cf45849dfdcfc043ca8baf44fd6ba51d74/analysis/gpn-star) | [GPN-Star collection](https://huggingface.co/collections/songlab/gpn-star-68c0c055acc2ee51d5c4f129) |
| Sorghum expression | Published, [Nature Biotechnology 2026](https://doi.org/10.1038/s41587-026-03046-y) | [`analysis/gpn_sorghum_expression/`](https://github.com/songlab-cal/gpn/tree/30dee6cf45849dfdcfc043ca8baf44fd6ba51d74/analysis/gpn_sorghum_expression) | [Sorghum collection](https://huggingface.co/collections/songlab/sorghum-gene-expression-prediction-68963dd31658bfb98c07ae1b) |

This index is curated: future entries should cover notable work, not every
exploration.

## Future paper and exploratory branches

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
