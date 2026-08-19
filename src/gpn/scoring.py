"""Small, model-agnostic helpers for nucleotide likelihood scores."""

import operator
from collections.abc import Sequence
from typing import SupportsIndex, cast

import torch
from jaxtyping import Float
from torch import Tensor

_CANONICAL_NUCLEOTIDES = frozenset("ACGT")


def validate_snv_batch(
    chromosomes: Sequence[object],
    positions: Sequence[object],
    references: Sequence[object],
    alternates: Sequence[object],
) -> tuple[list[str], list[int], list[str], list[str]]:
    """Validate and normalize a batch of one-based canonical SNVs.

    The public VEP interfaces intentionally accept only biallelic SNVs. Rejecting
    malformed rows before sequence lookup prevents multi-base or ambiguous alleles
    from being silently interpreted as tokenizer input.
    """

    lengths = {
        len(chromosomes),
        len(positions),
        len(references),
        len(alternates),
    }
    if len(lengths) != 1:
        raise ValueError(
            "Variant columns chrom, pos, ref, and alt must have equal length"
        )

    normalized_chromosomes: list[str] = []
    normalized_positions: list[int] = []
    normalized_references: list[str] = []
    normalized_alternates: list[str] = []
    for row, (chromosome, position, reference, alternate) in enumerate(
        zip(chromosomes, positions, references, alternates, strict=True)
    ):
        if not isinstance(chromosome, str) or not chromosome:
            raise ValueError(f"Variant row {row} has an empty or non-string chromosome")
        if isinstance(position, bool):
            raise ValueError(f"Variant row {row} position must be a positive integer")
        try:
            normalized_position = operator.index(cast(SupportsIndex, position))
        except TypeError as error:
            raise ValueError(
                f"Variant row {row} position must be a positive integer"
            ) from error
        if normalized_position <= 0:
            raise ValueError(
                f"Variant row {row} position must be a positive one-based coordinate"
            )
        if not isinstance(reference, str) or reference not in _CANONICAL_NUCLEOTIDES:
            raise ValueError(
                f"Variant row {row} reference allele must be one of A, C, G, T"
            )
        if not isinstance(alternate, str) or alternate not in _CANONICAL_NUCLEOTIDES:
            raise ValueError(
                f"Variant row {row} alternate allele must be one of A, C, G, T"
            )
        if reference == alternate:
            raise ValueError(
                f"Variant row {row} at {chromosome}:{normalized_position} has "
                "identical reference and alternate alleles"
            )

        normalized_chromosomes.append(chromosome)
        normalized_positions.append(normalized_position)
        normalized_references.append(reference)
        normalized_alternates.append(alternate)

    return (
        normalized_chromosomes,
        normalized_positions,
        normalized_references,
        normalized_alternates,
    )


def require_reference_matches(
    observed: Sequence[object],
    expected: Sequence[object],
    chromosomes: Sequence[object],
    positions: Sequence[object],
    *,
    orientation: str,
) -> None:
    """Raise a useful error when reference sequence and variant rows disagree."""

    lengths = {len(observed), len(expected), len(chromosomes), len(positions)}
    if len(lengths) != 1:
        raise ValueError("Reference-validation arrays must have equal length")
    for row, (actual, wanted, chromosome, position) in enumerate(
        zip(observed, expected, chromosomes, positions, strict=True)
    ):
        if actual != wanted:
            raise ValueError(
                "Reference allele mismatch for variant row "
                f"{row} at {chromosome}:{position} ({orientation} orientation): "
                f"genome has {actual!r}, input has {wanted!r}"
            )


def validate_centered_window_size(window_size: int) -> None:
    """Require a positive even window for alignment-centered inference."""

    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if window_size % 2:
        raise ValueError("window_size must be even for centered MSA inference")


def nucleotide_probabilities(
    logits: Float[Tensor, "... nucleotide"],
) -> Float[Tensor, "... nucleotide"]:
    """Normalize nucleotide logits along their final axis.

    Args:
        logits: Floating-point model outputs whose final axis follows a documented
            nucleotide order, normally ``A, C, G, T``.

    Returns:
        Probabilities with the same shape, normalized across nucleotides.

    Raises:
        TypeError: If ``logits`` is not floating point.
        ValueError: If ``logits`` has no non-empty nucleotide axis.
    """

    if not logits.is_floating_point():
        raise TypeError("Nucleotide logits must be floating point")
    if logits.ndim == 0 or logits.shape[-1] == 0:
        raise ValueError("Nucleotide logits must have a non-empty final axis")
    return torch.softmax(logits, dim=-1)


def log_likelihood_ratio(
    logits: Float[Tensor, "... nucleotide"],
    reference_index: int,
    alternate_index: int,
) -> Float[Tensor, "..."]:
    """Return the alternate-minus-reference log-likelihood ratio.

    The score sign is deliberate: negative values mean the alternate nucleotide is
    less likely than the reference under the model.

    Args:
        logits: Model outputs with nucleotides on the final axis.
        reference_index: Reference nucleotide index in that axis.
        alternate_index: Alternate nucleotide index in that axis.

    Returns:
        ``alternate_logit - reference_logit`` for every leading position.

    Raises:
        IndexError: If either nucleotide index is outside the final axis.
        TypeError: If an index is not an integer or ``logits`` is not floating
            point.
        ValueError: If ``logits`` has no non-empty nucleotide axis.
    """

    if not logits.is_floating_point():
        raise TypeError("Nucleotide logits must be floating point")
    if logits.ndim == 0 or logits.shape[-1] == 0:
        raise ValueError("Nucleotide logits must have a non-empty final axis")

    try:
        reference_index = operator.index(reference_index)
        alternate_index = operator.index(alternate_index)
    except TypeError as error:
        raise TypeError("Nucleotide indices must be integers") from error

    nucleotide_count = logits.shape[-1]
    for role, index in (
        ("reference", reference_index),
        ("alternate", alternate_index),
    ):
        if not 0 <= index < nucleotide_count:
            raise IndexError(
                f"{role.capitalize()} nucleotide index {index} is outside "
                f"[0, {nucleotide_count})"
            )
    return logits[..., alternate_index] - logits[..., reference_index]
