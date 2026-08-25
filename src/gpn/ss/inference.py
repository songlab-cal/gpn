"""Maintained GPN single-species inference operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from Bio.Seq import Seq
from datasets import Dataset, disable_caching
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    TrainingArguments,
)

from gpn import register_auto_classes
from gpn.arguments import CheckpointArguments
from gpn.data import load_dataset_from_file_or_dir
from gpn.inference import (
    build_run_signature,
    execute_inference,
    resource_identity,
    tokenizer_identity,
)
from gpn.scoring import (
    require_reference_matches,
    validate_positions_batch,
    validate_snv_batch,
)
from gpn.ss.data import Genome, token_input_id


def _centered_bounds(positions: np.ndarray, window_size: int) -> tuple[Any, Any]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    start: np.ndarray = positions - window_size // 2
    return start, start + window_size


def _tokenize_sequences(tokenizer: Any, sequences: list[str]) -> Any:
    return tokenizer(
        sequences,
        padding=False,
        truncation=False,
        return_token_type_ids=False,
        return_attention_mask=False,
        return_special_tokens_mask=False,
    )["input_ids"]


class MLMforVEPModel(torch.nn.Module):
    """Average alternate-minus-reference likelihood across both strands."""

    def __init__(self, model_path: str):
        super().__init__()
        register_auto_classes("ss")
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        self.model.eval()

    def get_llr(
        self,
        input_ids: torch.Tensor,
        pos: torch.Tensor,
        ref: torch.Tensor,
        alt: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.model(input_ids=input_ids).logits
        logits = logits[torch.arange(len(pos), device=pos.device), pos]
        row = torch.arange(len(ref), device=ref.device)
        return logits[row, alt] - logits[row, ref]

    def forward(
        self,
        input_ids_fwd: torch.Tensor | None = None,
        pos_fwd: torch.Tensor | None = None,
        ref_fwd: torch.Tensor | None = None,
        alt_fwd: torch.Tensor | None = None,
        input_ids_rev: torch.Tensor | None = None,
        pos_rev: torch.Tensor | None = None,
        ref_rev: torch.Tensor | None = None,
        alt_rev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        llr_fwd = self.get_llr(input_ids_fwd, pos_fwd, ref_fwd, alt_fwd)
        llr_rev = self.get_llr(input_ids_rev, pos_rev, ref_rev, alt_rev)
        return (llr_fwd + llr_rev) / 2


class MLMforLogitsModel(torch.nn.Module):
    """Return strand-averaged A/C/G/T masked-token logits."""

    def __init__(self, model_path: str, nucleotide_ids: list[int]):
        super().__init__()
        register_auto_classes("ss")
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        self.model.eval()
        self.nucleotide_ids = nucleotide_ids

    def get_logits(
        self,
        input_ids: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.model(input_ids=input_ids).logits
        return logits[torch.arange(len(pos), device=pos.device), pos]

    def forward(
        self,
        input_ids_fwd: torch.Tensor | None = None,
        pos_fwd: torch.Tensor | None = None,
        input_ids_rev: torch.Tensor | None = None,
        pos_rev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        a, c, g, t = self.nucleotide_ids
        logits_fwd = self.get_logits(input_ids_fwd, pos_fwd)[:, [a, c, g, t]]
        logits_rev = self.get_logits(input_ids_rev, pos_rev)[:, [t, g, c, a]]
        return (logits_fwd + logits_rev) / 2


class ModelCenterEmbedding(torch.nn.Module):
    """Return a strand-averaged embedding over a central window."""

    def __init__(self, model_path: str, center_window_size: int):
        super().__init__()
        if center_window_size <= 0:
            raise ValueError("center_window_size must be positive")
        register_auto_classes("ss")
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()
        self.center_window_size = center_window_size

    def get_center_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedding = self.model(input_ids=input_ids).last_hidden_state
        center = embedding.shape[1] // 2
        left = center - self.center_window_size // 2
        right = left + self.center_window_size
        if left < 0 or right > embedding.shape[1]:
            raise ValueError("center_window_size exceeds the input window")
        return embedding[:, left:right].mean(axis=1)

    def forward(
        self,
        input_ids_fwd: torch.Tensor | None = None,
        input_ids_rev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedding_fwd = self.get_center_embedding(input_ids_fwd)
        embedding_rev = self.get_center_embedding(input_ids_rev)
        return (embedding_fwd + embedding_rev) / 2


class VEPInference:
    def __init__(
        self,
        model_path: str,
        genome: Genome,
        window_size: int,
        tokenizer: Any,
        n_prefix: int = 0,
    ):
        if n_prefix < 0:
            raise ValueError("n_prefix must be non-negative")
        _centered_bounds(np.array([0]), window_size)
        self.model = MLMforVEPModel(model_path)
        self.genome = genome
        self.window_size = window_size
        self.tokenizer = tokenizer
        self.n_prefix = n_prefix

    def tokenize_function(self, variants: dict[str, list[Any]]) -> dict[str, Any]:
        return _tokenize_variant_batch(
            variants,
            self.genome,
            self.window_size,
            self.tokenizer,
            n_prefix=self.n_prefix,
        )

    def postprocess(self, predictions: Any) -> pd.DataFrame:
        return pd.DataFrame(predictions, columns=["score"])


class LogitsInference:
    def __init__(
        self,
        model_path: str,
        genome: Genome,
        window_size: int,
        tokenizer: Any,
        n_prefix: int = 0,
    ):
        if n_prefix < 0:
            raise ValueError("n_prefix must be non-negative")
        _centered_bounds(np.array([0]), window_size)
        nucleotide_ids = [
            token_input_id(nucleotide, tokenizer, n_prefix) for nucleotide in "ACGT"
        ]
        self.model = MLMforLogitsModel(model_path, nucleotide_ids)
        self.genome = genome
        self.window_size = window_size
        self.tokenizer = tokenizer
        self.n_prefix = n_prefix

    def tokenize_function(self, positions: dict[str, list[Any]]) -> dict[str, Any]:
        chromosome_values, position_values = validate_positions_batch(
            positions["chrom"], positions["pos"]
        )
        chromosomes = np.array(chromosome_values)
        zero_based = np.array(position_values) - 1
        start, end = _centered_bounds(zero_based, self.window_size)
        forward, reverse = zip(
            *(
                self.genome.get_seq_fwd_rev(chromosomes[i], start[i], end[i])
                for i in range(len(chromosomes))
            )
        )
        center_fwd = self.window_size // 2
        center_rev = center_fwd - 1 if self.window_size % 2 == 0 else center_fwd

        def prepare(sequences: tuple[str, ...], center: int) -> tuple[Any, list[int]]:
            array = np.array(
                [list(sequence.upper()) for sequence in sequences], dtype="object"
            )
            if array.ndim != 2 or array.shape[1] != self.window_size:
                raise ValueError("Genome windows do not match window_size")
            array[:, center] = self.tokenizer.mask_token
            return (
                _tokenize_sequences(
                    self.tokenizer, ["".join(sequence) for sequence in array]
                ),
                [center + self.n_prefix for _ in array],
            )

        input_ids_fwd, pos_fwd = prepare(forward, center_fwd)
        input_ids_rev, pos_rev = prepare(reverse, center_rev)
        return {
            "input_ids_fwd": input_ids_fwd,
            "pos_fwd": pos_fwd,
            "input_ids_rev": input_ids_rev,
            "pos_rev": pos_rev,
        }

    def postprocess(self, predictions: Any) -> pd.DataFrame:
        return pd.DataFrame(predictions, columns=list("ACGT"))


class EmbeddingInference:
    def __init__(
        self,
        model_path: str,
        genome: Genome,
        tokenizer: Any,
        center_window_size: int,
    ):
        self.model = ModelCenterEmbedding(model_path, center_window_size)
        self.genome = genome
        self.tokenizer = tokenizer

    def tokenize_function(self, windows: dict[str, list[Any]]) -> dict[str, Any]:
        chrom, start, end = windows["chrom"], windows["start"], windows["end"]
        forward, reverse = zip(
            *(
                self.genome.get_seq_fwd_rev(chrom[i], start[i], end[i])
                for i in range(len(chrom))
            )
        )
        return {
            "input_ids_fwd": _tokenize_sequences(self.tokenizer, list(forward)),
            "input_ids_rev": _tokenize_sequences(self.tokenizer, list(reverse)),
        }

    def postprocess(self, predictions: Any) -> pd.DataFrame:
        columns = [f"embedding_{index}" for index in range(predictions.shape[1])]
        return pd.DataFrame(predictions, columns=columns)


def _tokenize_variant_batch(
    variants: dict[str, list[Any]],
    genome: Genome,
    window_size: int,
    tokenizer: Any,
    n_prefix: int = 0,
) -> dict[str, Any]:
    chromosomes, positions, references, alternates = validate_snv_batch(
        variants["chrom"], variants["pos"], variants["ref"], variants["alt"]
    )
    chrom = np.array(chromosomes)
    zero_based = np.array(positions) - 1
    start, end = _centered_bounds(zero_based, window_size)
    forward, reverse = zip(
        *(genome.get_seq_fwd_rev(chrom[i], start[i], end[i]) for i in range(len(chrom)))
    )
    seq_fwd = np.array([list(seq.upper()) for seq in forward], dtype="object")
    seq_rev = np.array([list(seq.upper()) for seq in reverse], dtype="object")
    if seq_fwd.ndim != 2 or seq_fwd.shape[1] != window_size:
        raise ValueError("Forward genome windows do not match window_size")
    if seq_rev.ndim != 2 or seq_rev.shape[1] != window_size:
        raise ValueError("Reverse genome windows do not match window_size")
    ref_fwd = np.array(references)
    alt_fwd = np.array(alternates)
    ref_rev = np.array([str(Seq(value).reverse_complement()) for value in ref_fwd])
    alt_rev = np.array([str(Seq(value).reverse_complement()) for value in alt_fwd])
    pos_fwd = window_size // 2
    pos_rev = pos_fwd - 1 if window_size % 2 == 0 else pos_fwd

    def prepare(
        sequence: np.ndarray,
        center: int,
        reference: np.ndarray,
        alternate: np.ndarray,
        orientation: str,
    ) -> tuple[Any, list[int], list[int], list[int]]:
        require_reference_matches(
            sequence[:, center].tolist(),
            reference.tolist(),
            chromosomes,
            positions,
            orientation=orientation,
        )
        sequence[:, center] = tokenizer.mask_token
        return (
            _tokenize_sequences(tokenizer, ["".join(value) for value in sequence]),
            [center + n_prefix for _ in sequence],
            [token_input_id(value, tokenizer, n_prefix) for value in reference],
            [token_input_id(value, tokenizer, n_prefix) for value in alternate],
        )

    input_ids_fwd, positions_fwd, references_fwd, alternates_fwd = prepare(
        seq_fwd, pos_fwd, ref_fwd, alt_fwd, "forward"
    )
    input_ids_rev, positions_rev, references_rev, alternates_rev = prepare(
        seq_rev, pos_rev, ref_rev, alt_rev, "reverse-complement"
    )
    return {
        "input_ids_fwd": input_ids_fwd,
        "pos_fwd": positions_fwd,
        "ref_fwd": references_fwd,
        "alt_fwd": alternates_fwd,
        "input_ids_rev": input_ids_rev,
        "pos_rev": positions_rev,
        "ref_rev": references_rev,
        "alt_rev": alternates_rev,
    }


def _load_inputs(
    input_path: str,
    genome_path: str,
    model_path: str,
    tokenizer_path: str | None,
    split: str,
    is_file: bool,
) -> tuple[Dataset, Genome, Any]:
    disable_caching()
    dataset = load_dataset_from_file_or_dir(
        input_path,
        split=split,
        is_file=is_file,
    )
    genome = Genome(genome_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
    return dataset, genome, tokenizer


def _execute(
    *,
    operation: str,
    dataset: Dataset,
    input_path: str,
    genome_path: str,
    model_path: str,
    tokenizer_path: str | None,
    output_path: Path,
    split: str,
    is_file: bool,
    inference: Any,
    training_arguments: TrainingArguments,
    checkpoint_arguments: CheckpointArguments,
    operation_arguments: dict[str, Any],
) -> Path | None:
    def build_signature() -> dict[str, Any]:
        return build_run_signature(
            family="ss",
            operation=operation,
            dataset=dataset,
            input_path=input_path,
            split=split,
            is_file=is_file,
            model_path=model_path,
            inference=inference,
            training_arguments=training_arguments,
            checkpoint_revision=checkpoint_arguments.checkpoint_revision,
            resources={
                "genome": resource_identity(genome_path),
                "tokenizer": tokenizer_identity(
                    inference.tokenizer,
                    tokenizer_path or model_path,
                ),
            },
            operation_arguments=operation_arguments,
        )

    return execute_inference(
        dataset,
        inference,
        training_arguments,
        checkpoint_arguments,
        output_path=output_path,
        run_signature_factory=build_signature,
        output_prefix=f"gpn-ss-{operation}-",
    )


def vep(
    input_path: str,
    genome_path: str,
    window_size: int,
    model_path: str,
    output_path: Path,
    *,
    tokenizer_path: str | None,
    n_prefix: int,
    split: str,
    is_file: bool,
    training_arguments: TrainingArguments,
    checkpoint_arguments: CheckpointArguments,
) -> Path | None:
    dataset, genome, tokenizer = _load_inputs(
        input_path, genome_path, model_path, tokenizer_path, split, is_file
    )
    inference = VEPInference(model_path, genome, window_size, tokenizer, n_prefix)
    return _execute(
        operation="vep",
        dataset=dataset,
        input_path=input_path,
        genome_path=genome_path,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        output_path=output_path,
        split=split,
        is_file=is_file,
        inference=inference,
        training_arguments=training_arguments,
        checkpoint_arguments=checkpoint_arguments,
        operation_arguments={"n_prefix": n_prefix, "window_size": window_size},
    )


def logits(
    input_path: str,
    genome_path: str,
    window_size: int,
    model_path: str,
    output_path: Path,
    *,
    tokenizer_path: str | None,
    n_prefix: int,
    split: str,
    is_file: bool,
    training_arguments: TrainingArguments,
    checkpoint_arguments: CheckpointArguments,
) -> Path | None:
    dataset, genome, tokenizer = _load_inputs(
        input_path, genome_path, model_path, tokenizer_path, split, is_file
    )
    inference = LogitsInference(model_path, genome, window_size, tokenizer, n_prefix)
    return _execute(
        operation="logits",
        dataset=dataset,
        input_path=input_path,
        genome_path=genome_path,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        output_path=output_path,
        split=split,
        is_file=is_file,
        inference=inference,
        training_arguments=training_arguments,
        checkpoint_arguments=checkpoint_arguments,
        operation_arguments={"n_prefix": n_prefix, "window_size": window_size},
    )


def embedding(
    input_path: str,
    genome_path: str,
    center_window_size: int,
    model_path: str,
    output_path: Path,
    *,
    tokenizer_path: str | None,
    split: str,
    is_file: bool,
    training_arguments: TrainingArguments,
    checkpoint_arguments: CheckpointArguments,
) -> Path | None:
    dataset, genome, tokenizer = _load_inputs(
        input_path, genome_path, model_path, tokenizer_path, split, is_file
    )
    inference = EmbeddingInference(model_path, genome, tokenizer, center_window_size)
    return _execute(
        operation="embedding",
        dataset=dataset,
        input_path=input_path,
        genome_path=genome_path,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        output_path=output_path,
        split=split,
        is_file=is_file,
        inference=inference,
        training_arguments=training_arguments,
        checkpoint_arguments=checkpoint_arguments,
        operation_arguments={"center_window_size": center_window_size},
    )
