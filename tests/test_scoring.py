import pytest
import torch

from gpn.scoring import (
    log_likelihood_ratio,
    nucleotide_probabilities,
    require_reference_matches,
    validate_centered_window_size,
    validate_snv_batch,
)


def test_nucleotide_probabilities_normalize_final_axis():
    logits = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 1.0, 0.0, -1.0],
        ]
    )

    probabilities = nucleotide_probabilities(logits)

    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(2))
    torch.testing.assert_close(
        probabilities[0],
        torch.full((4,), 0.25),
    )


def test_log_likelihood_ratio_is_alternate_minus_reference():
    logits = torch.tensor(
        [
            [7.0, -2.0, -1.0, -3.0],
            [0.5, 0.25, 1.5, -0.5],
        ]
    )

    actual = log_likelihood_ratio(logits, reference_index=0, alternate_index=3)

    torch.testing.assert_close(actual, torch.tensor([-10.0, -1.0]))


@pytest.mark.parametrize("function", [nucleotide_probabilities, log_likelihood_ratio])
def test_scoring_rejects_non_floating_logits(function):
    logits = torch.tensor([1, 2, 3, 4])
    args = () if function is nucleotide_probabilities else (0, 1)

    with pytest.raises(TypeError, match="floating point"):
        function(logits, *args)


@pytest.mark.parametrize("shape", [(), (0,)])
def test_scoring_rejects_missing_nucleotide_axis(shape):
    logits = torch.empty(shape)

    with pytest.raises(ValueError, match="non-empty final axis"):
        nucleotide_probabilities(logits)


@pytest.mark.parametrize("index", [-1, 4])
def test_log_likelihood_ratio_rejects_out_of_range_indices(index):
    logits = torch.zeros(4)

    with pytest.raises(IndexError, match="Reference nucleotide index"):
        log_likelihood_ratio(logits, reference_index=index, alternate_index=1)


def test_log_likelihood_ratio_rejects_non_integer_indices():
    logits = torch.zeros(4)

    with pytest.raises(TypeError, match="indices must be integers"):
        log_likelihood_ratio(logits, reference_index=0.0, alternate_index=1)


def test_validate_snv_batch_accepts_canonical_one_based_variants():
    actual = validate_snv_batch(["chr1"], [42], ["A"], ["C"])

    assert actual == (["chr1"], [42], ["A"], ["C"])


@pytest.mark.parametrize(
    ("positions", "references", "alternates", "message"),
    (
        ([0], ["A"], ["C"], "positive one-based"),
        ([1], ["a"], ["C"], "reference allele"),
        ([1], ["N"], ["C"], "reference allele"),
        ([1], ["AA"], ["C"], "reference allele"),
        ([1], ["A"], ["A"], "identical reference"),
    ),
)
def test_validate_snv_batch_rejects_ambiguous_or_non_snv_rows(
    positions,
    references,
    alternates,
    message,
):
    with pytest.raises(ValueError, match=message):
        validate_snv_batch(["chr1"], positions, references, alternates)


def test_reference_mismatch_reports_source_coordinate():
    with pytest.raises(ValueError, match=r"chr7:123.*genome has 'C'.*input has 'A'"):
        require_reference_matches(
            ["C"],
            ["A"],
            ["chr7"],
            [123],
            orientation="forward",
        )


@pytest.mark.parametrize("window_size", [0, -2, 5])
def test_centered_msa_windows_must_be_positive_and_even(window_size):
    with pytest.raises(ValueError, match="window_size"):
        validate_centered_window_size(window_size)
