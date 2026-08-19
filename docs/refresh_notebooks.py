"""Deliberately refresh the committed outputs of the three quick starts."""

from __future__ import annotations

import argparse
import ast
import os
import re
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from docs.prepare_notebooks import NOTEBOOKS, REPOSITORY_ROOT, SOURCE

_EXPECTED_WARNING_FILTERS = (
    "ignore:IProgress not found",
    "ignore:`torch.backends.cuda.sdp_kernel()` is deprecated:FutureWarning",
)
_EXPECTED_STDERR = re.compile(
    r"\[transformers\].*ConvNetModel LOAD REPORT.*songlab/gpn-brassicales"
    r".*cls\.decoder\.\{0, 2, 3\}\.(?:weight|bias).*UNEXPECTED"
    r".*can be ignored when loading from different task/architecture",
    flags=re.DOTALL,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-local",
        action="store_true",
        help="Allow execution outside a Slurm allocation (never use on a login node).",
    )
    return parser.parse_args()


def _validate_environment(*, allow_local: bool) -> None:
    if not allow_local and "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError(
            "Notebook refreshes must run in a Slurm allocation; "
            "pass --allow-local only on another dedicated compute environment."
        )

    actual = tuple(sorted(path.name for path in SOURCE.glob("*.ipynb")))
    if actual != tuple(sorted(NOTEBOOKS)):
        raise RuntimeError(
            "Expected exactly the three canonical quick starts; "
            f"found: {', '.join(actual) or 'none'}"
        )


def _configure_kernel_environment() -> None:
    """Make committed output quiet without discarding unexpected stderr."""

    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    warning_filters = list(_EXPECTED_WARNING_FILTERS)
    if existing_filters := os.environ.get("PYTHONWARNINGS"):
        warning_filters.append(existing_filters)
    os.environ["PYTHONWARNINGS"] = ",".join(warning_filters)


def _execute(path: Path) -> nbformat.NotebookNode:
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code" and "skip-execution" not in cell.metadata.get(
            "tags", []
        ):
            ast.parse(cell.source, filename=f"{path}:{cell.get('id', 'cell')}")

    client = NotebookClient(
        notebook,
        timeout=30 * 60,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPOSITORY_ROOT / "tests" / "fixtures")}},
    )
    executed = client.execute()
    for cell in executed.cells:
        cell.metadata.pop("execution", None)
        stderr = [
            "".join(output.get("text", ""))
            for output in cell.get("outputs", [])
            if output.get("output_type") == "stream" and output.get("name") == "stderr"
        ]
        unexpected_stderr = [
            text for text in stderr if not _EXPECTED_STDERR.search(text)
        ]
        if unexpected_stderr:
            details = "\n".join(unexpected_stderr)
            raise RuntimeError(f"{path.name} emitted stderr:\n{details}")
        if stderr:
            cell.outputs = [
                output
                for output in cell.get("outputs", [])
                if not (
                    output.get("output_type") == "stream"
                    and output.get("name") == "stderr"
                )
            ]
    return executed


def main() -> None:
    """Execute all quick starts, writing none unless all three succeed."""

    args = _parse_args()
    _validate_environment(allow_local=args.allow_local)
    _configure_kernel_environment()

    executed = [(SOURCE / name, _execute(SOURCE / name)) for name in NOTEBOOKS]
    for path, notebook in executed:
        nbformat.write(notebook, path)


if __name__ == "__main__":
    main()
