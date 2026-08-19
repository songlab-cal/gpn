"""Copy the canonical Colab notebooks into the Sphinx source tree."""

from __future__ import annotations

import shutil
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY_ROOT / "colabs"
DESTINATION = Path(__file__).resolve().parent / "_notebooks"
NOTEBOOKS = (
    "gpn_quick_start.ipynb",
    "gpn_star_quick_start.ipynb",
    "phylogpn_quick_start.ipynb",
)


def main() -> None:
    """Create a deterministic generated notebook directory for Sphinx."""

    actual = tuple(sorted(path.name for path in SOURCE.glob("*.ipynb")))
    if actual != tuple(sorted(NOTEBOOKS)):
        raise RuntimeError(
            "Expected exactly the three canonical quick starts; "
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
