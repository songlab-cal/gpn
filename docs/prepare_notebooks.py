"""Copy the canonical Colab notebooks into the Sphinx source tree."""

from __future__ import annotations

import shutil
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY_ROOT / "colabs"
DESTINATION = Path(__file__).resolve().parent / "_notebooks"
MODEL_DEMOS = (
    "gpn_demo.ipynb",
    "gpn_star_demo.ipynb",
    "phylogpn_demo.ipynb",
)
WORKFLOWS = ("gpn_star_precomputed_scores.ipynb",)
NOTEBOOKS = MODEL_DEMOS + WORKFLOWS


def main() -> None:
    """Create a deterministic generated notebook directory for Sphinx."""

    actual = tuple(sorted(path.name for path in SOURCE.glob("*.ipynb")))
    if actual != tuple(sorted(NOTEBOOKS)):
        raise RuntimeError(
            "Expected the three canonical demos and precomputed-score workflow; "
            f"found: {', '.join(actual) or 'none'}"
        )

    DESTINATION.mkdir(parents=True, exist_ok=True)
    for stale_path in DESTINATION.glob("*.ipynb"):
        if stale_path.name not in NOTEBOOKS:
            stale_path.unlink()
    for notebook in NOTEBOOKS:
        shutil.copyfile(SOURCE / notebook, DESTINATION / notebook)


if __name__ == "__main__":
    main()
