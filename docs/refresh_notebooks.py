"""Deliberately refresh the committed outputs of the three quick starts."""

from __future__ import annotations

import argparse
import ast
import json
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
_PROVENANCE_CELL_ID = "gpn-refresh-provenance"
_PROVENANCE_PREFIX = "__GPN_REFRESH_PROVENANCE__="


def _provenance_cell() -> nbformat.NotebookNode:
    """Create a temporary cell that observes the executing kernel."""

    source = f'''\
import json as _gpn_json
import platform as _gpn_platform
from datetime import date as _gpn_date
from importlib.metadata import version as _gpn_version

_gpn_model = globals().get("model_for_mlm", globals().get("model"))
if _gpn_model is None:
    raise RuntimeError("Quick start did not define model_for_mlm or model")
_gpn_model_id = globals().get("MODEL_ID")
_gpn_model_revision = globals().get("MODEL_REVISION")
if not isinstance(_gpn_model_id, str) or not isinstance(_gpn_model_revision, str):
    raise RuntimeError("Quick start did not define string MODEL_ID and MODEL_REVISION")
_gpn_resolved_model_id = getattr(_gpn_model.config, "_name_or_path", None)
_gpn_resolved_revision = getattr(_gpn_model.config, "_commit_hash", None)
if not isinstance(_gpn_resolved_model_id, str) or not _gpn_resolved_model_id:
    raise RuntimeError("Loaded model config does not expose a model name or path")
if _gpn_resolved_model_id != _gpn_model_id:
    raise RuntimeError(
        "Loaded model identity does not match MODEL_ID: "
        f"{{_gpn_resolved_model_id}} != {{_gpn_model_id}}"
    )
if not isinstance(_gpn_resolved_revision, str) or not _gpn_resolved_revision:
    raise RuntimeError("Loaded model config does not expose a resolved commit hash")
if _gpn_resolved_revision != _gpn_model_revision:
    raise RuntimeError(
        "Loaded model revision does not match MODEL_REVISION: "
        f"{{_gpn_resolved_revision}} != {{_gpn_model_revision}}"
    )
_gpn_parameter = next(_gpn_model.parameters())
print(
    "{_PROVENANCE_PREFIX}"
    + _gpn_json.dumps(
        {{
            "last_scientific_validation": _gpn_date.today().isoformat(),
            "model_id": _gpn_resolved_model_id,
            "model_revision": _gpn_resolved_revision,
            "output_environment": {{
                "device": str(_gpn_parameter.device),
                "dtype": str(_gpn_parameter.dtype),
                "gpn": _gpn_version("gpn"),
                "python": _gpn_platform.python_version(),
                "torch": _gpn_version("torch"),
                "transformers": _gpn_version("transformers"),
            }},
        }},
        sort_keys=True,
    )
)
'''
    return nbformat.v4.new_code_cell(source=source, id=_PROVENANCE_CELL_ID)


def _consume_provenance(notebook: nbformat.NotebookNode) -> dict[str, object]:
    """Remove the temporary cell and parse its kernel-generated payload."""

    cell = notebook.cells.pop()
    if cell.get("id") != _PROVENANCE_CELL_ID:
        raise RuntimeError("Notebook refresh provenance cell was not executed last")

    for output in cell.get("outputs", []):
        if output.get("output_type") != "stream" or output.get("name") != "stdout":
            continue
        text = "".join(output.get("text", ""))
        for line in text.splitlines():
            if line.startswith(_PROVENANCE_PREFIX):
                return dict(json.loads(line.removeprefix(_PROVENANCE_PREFIX)))
    raise RuntimeError("Executing kernel did not emit notebook provenance")


def _update_provenance(
    notebook: nbformat.NotebookNode, provenance: dict[str, object]
) -> None:
    gpn_metadata = notebook.metadata.setdefault("gpn", {})
    gpn_metadata["last_scientific_validation"] = provenance[
        "last_scientific_validation"
    ]
    gpn_metadata["model_id"] = provenance["model_id"]
    gpn_metadata["model_revision"] = provenance["model_revision"]
    gpn_metadata["output_environment"] = provenance["output_environment"]


def _remove_expected_stderr(cell: nbformat.NotebookNode, *, notebook_name: str) -> None:
    """Remove only complete, recognized diagnostic streams from a cell."""

    stderr = [
        "".join(output.get("text", ""))
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream" and output.get("name") == "stderr"
    ]
    unexpected_stderr = [
        text for text in stderr if _EXPECTED_STDERR.fullmatch(text.strip()) is None
    ]
    if unexpected_stderr:
        details = "\n".join(unexpected_stderr)
        raise RuntimeError(f"{notebook_name} emitted stderr:\n{details}")
    if stderr:
        cell.outputs = [
            output
            for output in cell.get("outputs", [])
            if not (
                output.get("output_type") == "stream" and output.get("name") == "stderr"
            )
        ]


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

    notebook.cells.append(_provenance_cell())
    client = NotebookClient(
        notebook,
        timeout=30 * 60,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPOSITORY_ROOT / "tests" / "fixtures")}},
    )
    executed = client.execute()
    _update_provenance(executed, _consume_provenance(executed))
    for cell in executed.cells:
        cell.metadata.pop("execution", None)
        _remove_expected_stderr(cell, notebook_name=path.name)
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
