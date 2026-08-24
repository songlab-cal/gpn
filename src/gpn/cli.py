"""Typed public command-line interface for maintained GPN workflows."""

from __future__ import annotations

import importlib.metadata
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter
from transformers import TrainingArguments

from gpn.arguments import CheckpointArguments

type PredictionArguments = Annotated[
    TrainingArguments,
    Parameter(name="*", group="Transformers prediction options"),
]
type CheckpointOptions = Annotated[
    CheckpointArguments,
    Parameter(name="*", group="Checkpoint options"),
]

_DEFAULT_PREDICTION_ARGUMENTS = TrainingArguments()
_DEFAULT_CHECKPOINT_OPTIONS = CheckpointArguments()


def _version() -> str:
    return importlib.metadata.version("gpn")


app = App(
    name="gpn",
    help="Train and run inference with maintained GPN model families.",
    version=_version,
)
ss_app = App(
    name="ss",
    help="Train or run inference with single-species GPN.",
)
msa_app = App(
    name="msa",
    help="Run inference with deprecated GPN-MSA checkpoints.",
)
star_app = App(
    name="star",
    help="Train or run inference with GPN-Star.",
)
app.command(ss_app)
app.command(msa_app)
app.command(star_app)


@ss_app.command(name="train")
def ss_train(profile: Path) -> None:
    """Train GPN from a human-readable YAML profile."""

    from gpn.ss.train import main

    main(profile)


@ss_app.command(name="vep")
def ss_vep(
    input_path: str,
    genome_path: str,
    window_size: int,
    model_path: str,
    output_path: Path,
    *,
    tokenizer_path: str | None = None,
    n_prefix: int = 0,
    split: str = "test",
    is_file: bool = False,
    checkpoint: CheckpointOptions = _DEFAULT_CHECKPOINT_OPTIONS,
    trainer: PredictionArguments = _DEFAULT_PREDICTION_ARGUMENTS,
) -> None:
    """Score variants with single-species GPN.

    Parameters
    ----------
    input_path
        Dataset or local file with chrom, one-based pos, ref, and alt columns.
    genome_path
        Reference-genome FASTA path.
    window_size
        Number of genomic bases supplied to the model.
    model_path
        Local or Hugging Face model identifier.
    output_path
        Destination Parquet file.
    """

    from gpn.ss.inference import vep

    vep(
        input_path,
        genome_path,
        window_size,
        model_path,
        output_path,
        tokenizer_path=tokenizer_path,
        n_prefix=n_prefix,
        split=split,
        is_file=is_file,
        training_arguments=trainer,
        checkpoint_arguments=checkpoint,
    )


@ss_app.command(name="logits")
def ss_logits(
    input_path: str,
    genome_path: str,
    window_size: int,
    model_path: str,
    output_path: Path,
    *,
    tokenizer_path: str | None = None,
    n_prefix: int = 0,
    split: str = "test",
    is_file: bool = False,
    checkpoint: CheckpointOptions = _DEFAULT_CHECKPOINT_OPTIONS,
    trainer: PredictionArguments = _DEFAULT_PREDICTION_ARGUMENTS,
) -> None:
    """Compute strand-averaged masked-nucleotide logits with GPN."""

    from gpn.ss.inference import logits

    logits(
        input_path,
        genome_path,
        window_size,
        model_path,
        output_path,
        tokenizer_path=tokenizer_path,
        n_prefix=n_prefix,
        split=split,
        is_file=is_file,
        training_arguments=trainer,
        checkpoint_arguments=checkpoint,
    )


@ss_app.command(name="embedding")
def ss_embedding(
    input_path: str,
    genome_path: str,
    center_window_size: int,
    model_path: str,
    output_path: Path,
    *,
    tokenizer_path: str | None = None,
    split: str = "test",
    is_file: bool = False,
    checkpoint: CheckpointOptions = _DEFAULT_CHECKPOINT_OPTIONS,
    trainer: PredictionArguments = _DEFAULT_PREDICTION_ARGUMENTS,
) -> None:
    """Extract strand-averaged GPN embeddings over interval centers."""

    from gpn.ss.inference import embedding

    embedding(
        input_path,
        genome_path,
        center_window_size,
        model_path,
        output_path,
        tokenizer_path=tokenizer_path,
        split=split,
        is_file=is_file,
        training_arguments=trainer,
        checkpoint_arguments=checkpoint,
    )


@msa_app.command(name="vep")
def msa_vep(
    input_path: str,
    msa_path: str,
    window_size: int,
    model_path: str,
    output_path: Path,
    *,
    split: str = "test",
    is_file: bool = False,
    checkpoint: CheckpointOptions = _DEFAULT_CHECKPOINT_OPTIONS,
    trainer: PredictionArguments = _DEFAULT_PREDICTION_ARGUMENTS,
) -> None:
    """Score variants with a deprecated GPN-MSA checkpoint."""

    from gpn.msa.inference import vep

    vep(
        input_path,
        msa_path,
        window_size,
        model_path,
        output_path,
        split=split,
        is_file=is_file,
        training_arguments=trainer,
        checkpoint_arguments=checkpoint,
    )


@msa_app.command(name="logits")
def msa_logits(
    input_path: str,
    msa_path: str,
    window_size: int,
    model_path: str,
    output_path: Path,
    *,
    split: str = "test",
    is_file: bool = False,
    checkpoint: CheckpointOptions = _DEFAULT_CHECKPOINT_OPTIONS,
    trainer: PredictionArguments = _DEFAULT_PREDICTION_ARGUMENTS,
) -> None:
    """Compute masked-nucleotide logits with a deprecated GPN-MSA checkpoint."""

    from gpn.msa.inference import logits

    logits(
        input_path,
        msa_path,
        window_size,
        model_path,
        output_path,
        split=split,
        is_file=is_file,
        training_arguments=trainer,
        checkpoint_arguments=checkpoint,
    )


@msa_app.command(name="embedding")
def msa_embedding(
    input_path: str,
    msa_path: str,
    window_size: int,
    model_path: str,
    output_path: Path,
    *,
    center_window_size: int = 100,
    split: str = "test",
    is_file: bool = False,
    checkpoint: CheckpointOptions = _DEFAULT_CHECKPOINT_OPTIONS,
    trainer: PredictionArguments = _DEFAULT_PREDICTION_ARGUMENTS,
) -> None:
    """Extract embeddings with a deprecated GPN-MSA checkpoint."""

    from gpn.msa.inference import embedding

    embedding(
        input_path,
        msa_path,
        window_size,
        model_path,
        output_path,
        center_window_size=center_window_size,
        split=split,
        is_file=is_file,
        training_arguments=trainer,
        checkpoint_arguments=checkpoint,
    )


@star_app.command(name="train")
def star_train(profile: Path) -> None:
    """Train GPN-Star from a human-readable YAML profile."""

    from gpn.star.train import main

    main(profile)


@star_app.command(name="vep")
def star_vep(
    input_path: str,
    msa_path: str,
    window_size: int,
    model_path: str,
    output_path: Path,
    *,
    split: str = "test",
    is_file: bool = False,
    checkpoint: CheckpointOptions = _DEFAULT_CHECKPOINT_OPTIONS,
    trainer: PredictionArguments = _DEFAULT_PREDICTION_ARGUMENTS,
) -> None:
    """Score variants with GPN-Star."""

    from gpn.star.inference import vep

    vep(
        input_path,
        msa_path,
        window_size,
        model_path,
        output_path,
        split=split,
        is_file=is_file,
        training_arguments=trainer,
        checkpoint_arguments=checkpoint,
    )


@star_app.command(name="logits")
def star_logits(
    input_path: str,
    msa_path: str,
    window_size: int,
    model_path: str,
    output_path: Path,
    *,
    phylo_dist_path: str | None = None,
    split: str = "test",
    is_file: bool = False,
    checkpoint: CheckpointOptions = _DEFAULT_CHECKPOINT_OPTIONS,
    trainer: PredictionArguments = _DEFAULT_PREDICTION_ARGUMENTS,
) -> None:
    """Compute masked-nucleotide logits with GPN-Star."""

    from gpn.star.inference import logits

    logits(
        input_path,
        msa_path,
        window_size,
        model_path,
        output_path,
        phylo_dist_path=phylo_dist_path,
        split=split,
        is_file=is_file,
        training_arguments=trainer,
        checkpoint_arguments=checkpoint,
    )


@star_app.command(name="embedding")
def star_embedding(
    input_path: str,
    msa_path: str,
    window_size: int,
    model_path: str,
    output_path: Path,
    *,
    center_window_size: int = 100,
    split: str = "test",
    is_file: bool = False,
    checkpoint: CheckpointOptions = _DEFAULT_CHECKPOINT_OPTIONS,
    trainer: PredictionArguments = _DEFAULT_PREDICTION_ARGUMENTS,
) -> None:
    """Extract GPN-Star embeddings over interval centers."""

    from gpn.star.inference import embedding

    embedding(
        input_path,
        msa_path,
        window_size,
        model_path,
        output_path,
        center_window_size=center_window_size,
        split=split,
        is_file=is_file,
        training_arguments=trainer,
        checkpoint_arguments=checkpoint,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the public CLI and return its process exit status."""

    try:
        result = app(argv, result_action="return_value")
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
