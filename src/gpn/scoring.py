"""Small, model-agnostic helpers for nucleotide likelihood scores."""

import operator

import torch
from jaxtyping import Float
from torch import Tensor


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
