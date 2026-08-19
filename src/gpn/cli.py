"""Public command-line interface for maintained GPN workflows."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from types import ModuleType

_OPTIONAL_IMPORTS = {
    "Bio",
    "accelerate",
    "datasets",
    "joblib",
    "pandas",
    "pyarrow",
    "tqdm",
    "zarr",
}


@dataclass(frozen=True)
class _Command:
    module: str
    extra: str
    prog: str
    fixed_command: str | None = None


def _version() -> str:
    return importlib.metadata.version("gpn")


def _add_leaf(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    *,
    command: _Command,
    help: str,
) -> None:
    # Leaf help belongs to the existing workflow parser. The top-level parser uses
    # parse_known_args so every remaining positional and option is forwarded exactly.
    parser = subparsers.add_parser(name, add_help=False, help=help)
    parser.set_defaults(command=command)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpn",
        description="Train and run inference with maintained GPN model families.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_version()}",
    )
    commands = parser.add_subparsers(dest="group", required=True)

    ss = commands.add_parser(
        "ss",
        help="train or run inference with GPN",
        description="Train or run core inference operations with single-species GPN.",
    )
    ss_commands = ss.add_subparsers(dest="ss_command", required=True)
    _add_leaf(
        ss_commands,
        "train",
        command=_Command("gpn.ss.train", "train", "gpn ss train"),
        help="train GPN on an already prepared sequence dataset",
    )
    for name, module, help_text in (
        ("vep", "gpn.ss.run_vep", "score variants with GPN"),
        ("logits", "gpn.ss.get_logits", "compute masked-nucleotide logits with GPN"),
        ("embedding", "gpn.ss.get_embeddings", "extract GPN embeddings"),
    ):
        _add_leaf(
            ss_commands,
            name,
            command=_Command(module, "inference", f"gpn ss {name}"),
            help=help_text,
        )

    msa = commands.add_parser(
        "msa",
        help="run deprecated GPN-MSA inference",
        description=(
            "Run inference with the deprecated GPN-MSA family. Training is not "
            "supported; use GPN-Star for new alignment-based training."
        ),
    )
    msa_commands = msa.add_subparsers(dest="msa_command", required=True)
    for name, help_text in (
        ("vep", "score variants with GPN-MSA"),
        ("logits", "compute masked-nucleotide logits with GPN-MSA"),
        ("embedding", "extract GPN-MSA embeddings"),
    ):
        _add_leaf(
            msa_commands,
            name,
            command=_Command(
                "gpn.msa.inference",
                "inference",
                f"gpn msa {name}",
                fixed_command=name,
            ),
            help=help_text,
        )

    star = commands.add_parser(
        "star",
        help="train or run inference with GPN-Star",
        description="Train or run core inference operations with GPN-Star.",
    )
    star_commands = star.add_subparsers(dest="star_command", required=True)
    _add_leaf(
        star_commands,
        "train",
        command=_Command("gpn.star.train", "train", "gpn star train"),
        help="train GPN-Star on prepared intervals and local MSAs",
    )
    for name, help_text in (
        ("vep", "score variants with GPN-Star"),
        ("logits", "compute masked-nucleotide logits with GPN-Star"),
        ("embedding", "extract GPN-Star embeddings"),
    ):
        _add_leaf(
            star_commands,
            name,
            command=_Command(
                "gpn.star.inference",
                "inference",
                f"gpn star {name}",
                fixed_command=name,
            ),
            help=help_text,
        )
    return parser


def _load_module(command: _Command, parser: argparse.ArgumentParser) -> ModuleType:
    try:
        return importlib.import_module(command.module)
    except ModuleNotFoundError as error:
        missing = error.name or "an optional dependency"
        if missing.split(".", maxsplit=1)[0] not in _OPTIONAL_IMPORTS:
            raise
        parser.exit(
            2,
            f"gpn: {missing!r} is required for this command; "
            f"install gpn[{command.extra}]\n",
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the public CLI and return its process exit status."""

    parser = _build_parser()
    namespace, remaining = parser.parse_known_args(argv)
    command: _Command = namespace.command
    module = _load_module(command, parser)
    target = getattr(module, "main")
    original_prog = sys.argv[0]
    try:
        sys.argv[0] = command.prog
        if command.fixed_command is None:
            result = target(remaining)
        else:
            result = target(remaining, command=command.fixed_command)
    finally:
        sys.argv[0] = original_prog
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
