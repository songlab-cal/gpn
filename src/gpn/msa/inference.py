"""Inference-only support for the deprecated GPN-MSA family."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, disable_caching
from jaxtyping import Float, Int
from torch import Tensor
from transformers import AutoModel, AutoModelForMaskedLM, TrainingArguments

from gpn import register_auto_classes
from gpn.arguments import CheckpointArguments
from gpn.data import ReverseComplementer, Tokenizer, load_dataset_from_file_or_dir
from gpn.inference import (
    build_run_signature,
    execute_inference,
    resource_identity,
)
from gpn.msa.data import GenomeMSA
from gpn.scoring import (
    require_reference_matches,
    validate_centered_window_size,
    validate_positions_batch,
    validate_snv_batch,
)


class MLMforVEPModel(torch.nn.Module):
    def __init__(self, model_path: str, model_revision: str | None = None):
        super().__init__()
        register_auto_classes("msa")
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_path,
            revision=model_revision,
        )
        self.model.eval()

    def get_llr(
        self,
        input_ids: Int[Tensor, "batch position"],
        aux_features: Int[Tensor, "batch position auxiliary"],
        pos: Int[Tensor, "... batch"],
        ref: Int[Tensor, "... batch"],
        alt: Int[Tensor, "... batch"],
    ) -> Float[Tensor, "... batch"]:
        logits = self.model(input_ids=input_ids, aux_features=aux_features).logits
        logits = logits[torch.arange(len(pos), device=pos.device), pos]
        row = torch.arange(len(ref), device=ref.device)
        return logits[row, alt] - logits[row, ref]

    def forward(
        self,
        input_ids_fwd: Int[Tensor, "batch position"] | None = None,
        aux_features_fwd: Int[Tensor, "batch position auxiliary"] | None = None,
        pos_fwd: Int[Tensor, "... batch"] | None = None,
        ref_fwd: Int[Tensor, "... batch"] | None = None,
        alt_fwd: Int[Tensor, "... batch"] | None = None,
        input_ids_rev: Int[Tensor, "batch position"] | None = None,
        aux_features_rev: Int[Tensor, "batch position auxiliary"] | None = None,
        pos_rev: Int[Tensor, "... batch"] | None = None,
        ref_rev: Int[Tensor, "... batch"] | None = None,
        alt_rev: Int[Tensor, "... batch"] | None = None,
    ) -> Float[Tensor, "... batch"]:
        llr_fwd = self.get_llr(
            input_ids_fwd, aux_features_fwd, pos_fwd, ref_fwd, alt_fwd
        )
        llr_rev = self.get_llr(
            input_ids_rev, aux_features_rev, pos_rev, ref_rev, alt_rev
        )
        return (llr_fwd + llr_rev) / 2


class MLMforLogitsModel(torch.nn.Module):
    def __init__(self, model_path: str, model_revision: str | None = None):
        super().__init__()
        register_auto_classes("msa")
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_path,
            revision=model_revision,
        )
        self.model.eval()
        tokenizer = Tokenizer()
        self.nucleotide_ids = [tokenizer.vocab.index(base) for base in "ACGT"]

    def get_logits(
        self,
        input_ids: Int[Tensor, "batch position"],
        aux_features: Int[Tensor, "batch position auxiliary"],
        pos: Int[Tensor, "... batch"],
    ) -> Float[Tensor, "batch nucleotide"]:
        logits = self.model(input_ids=input_ids, aux_features=aux_features).logits
        return logits[torch.arange(len(pos), device=pos.device), pos]

    def forward(
        self,
        input_ids_fwd: Int[Tensor, "batch position"] | None = None,
        aux_features_fwd: Int[Tensor, "batch position auxiliary"] | None = None,
        pos_fwd: Int[Tensor, "... batch"] | None = None,
        input_ids_rev: Int[Tensor, "batch position"] | None = None,
        aux_features_rev: Int[Tensor, "batch position auxiliary"] | None = None,
        pos_rev: Int[Tensor, "... batch"] | None = None,
    ) -> Float[Tensor, "batch nucleotide"]:
        a, c, g, t = self.nucleotide_ids
        logits_fwd = self.get_logits(input_ids_fwd, aux_features_fwd, pos_fwd)[
            :, [a, c, g, t]
        ]
        logits_rev = self.get_logits(input_ids_rev, aux_features_rev, pos_rev)[
            :, [t, g, c, a]
        ]
        return (logits_fwd + logits_rev) / 2


class ModelCenterEmbedding(torch.nn.Module):
    def __init__(
        self,
        model_path: str,
        center_window_size: int,
        model_revision: str | None = None,
    ):
        super().__init__()
        if center_window_size <= 0:
            raise ValueError("center_window_size must be positive")
        register_auto_classes("msa")
        self.model = AutoModel.from_pretrained(model_path, revision=model_revision)
        self.model.eval()
        self.center_window_size = center_window_size

    def get_center_embedding(
        self,
        input_ids: Int[Tensor, "batch position"],
        aux_features: Int[Tensor, "batch position auxiliary"],
    ) -> Float[Tensor, "batch hidden"]:
        embedding = self.model(
            input_ids=input_ids, aux_features=aux_features
        ).last_hidden_state
        center = embedding.shape[1] // 2
        left = center - self.center_window_size // 2
        right = left + self.center_window_size
        if left < 0 or right > embedding.shape[1]:
            raise ValueError("center_window_size exceeds the input window")
        return embedding[:, left:right].mean(axis=1)

    def forward(
        self,
        input_ids_fwd: Int[Tensor, "batch position"] | None = None,
        input_ids_rev: Int[Tensor, "batch position"] | None = None,
        aux_features_fwd: Int[Tensor, "batch position auxiliary"] | None = None,
        aux_features_rev: Int[Tensor, "batch position auxiliary"] | None = None,
    ) -> Float[Tensor, "batch hidden"]:
        embedding_fwd = self.get_center_embedding(input_ids_fwd, aux_features_fwd)
        embedding_rev = self.get_center_embedding(input_ids_rev, aux_features_rev)
        return (embedding_fwd + embedding_rev) / 2


def _alignment(
    genome_msa: GenomeMSA,
    chrom: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return genome_msa.get_msa_batch_fwd_rev(chrom, start, end, tokenize=True)


class VEPInference:
    def __init__(
        self,
        model_path: str,
        genome_msa: GenomeMSA,
        window_size: int,
        model_revision: str | None = None,
    ):
        validate_centered_window_size(window_size)
        self.genome_msa = genome_msa
        self.window_size = window_size
        self.tokenizer = Tokenizer()
        self.model = MLMforVEPModel(model_path, model_revision=model_revision)
        self.reverse_complementer = ReverseComplementer()

    def tokenize_function(self, variants: dict[str, list[Any]]) -> dict[str, Any]:
        chromosomes, positions, references, alternates = validate_snv_batch(
            variants["chrom"], variants["pos"], variants["ref"], variants["alt"]
        )
        chrom = np.array(chromosomes)
        pos = np.array(positions) - 1
        start = pos - self.window_size // 2
        end = pos + self.window_size // 2
        msa_fwd, msa_rev = _alignment(self.genome_msa, chrom, start, end)
        pos_fwd = self.window_size // 2
        pos_rev = pos_fwd - 1
        ref_fwd = np.array(
            [np.frombuffer(value.encode("ascii"), dtype="S1") for value in references]
        )
        alt_fwd = np.array(
            [np.frombuffer(value.encode("ascii"), dtype="S1") for value in alternates]
        )
        ref_rev = self.reverse_complementer(ref_fwd)
        alt_rev = self.reverse_complementer(alt_fwd)

        def prepare(
            msa: np.ndarray,
            center: int,
            reference: np.ndarray,
            alternate: np.ndarray,
            orientation: str,
        ) -> tuple[Any, Any, Any, Any, Any]:
            reference_ids = self.tokenizer(reference.flatten())
            alternate_ids = self.tokenizer(alternate.flatten())
            input_ids, aux_features = msa[:, :, 0], msa[:, :, 1:]
            require_reference_matches(
                [self.tokenizer.vocab[int(value)] for value in input_ids[:, center]],
                [self.tokenizer.vocab[int(value)] for value in reference_ids],
                chromosomes,
                positions,
                orientation=orientation,
            )
            input_ids[:, center] = self.tokenizer.mask_token_id()
            return (
                input_ids.astype(np.int64),
                aux_features,
                np.full(len(input_ids), center),
                reference_ids.astype(np.int64),
                alternate_ids.astype(np.int64),
            )

        fwd = prepare(msa_fwd, pos_fwd, ref_fwd, alt_fwd, "forward")
        rev = prepare(msa_rev, pos_rev, ref_rev, alt_rev, "reverse-complement")
        return {
            "input_ids_fwd": fwd[0],
            "aux_features_fwd": fwd[1],
            "pos_fwd": fwd[2],
            "ref_fwd": fwd[3],
            "alt_fwd": fwd[4],
            "input_ids_rev": rev[0],
            "aux_features_rev": rev[1],
            "pos_rev": rev[2],
            "ref_rev": rev[3],
            "alt_rev": rev[4],
        }

    def postprocess(self, predictions: Any) -> pd.DataFrame:
        return pd.DataFrame(predictions, columns=["score"])


class LogitsInference:
    def __init__(
        self,
        model_path: str,
        genome_msa: GenomeMSA,
        window_size: int,
        model_revision: str | None = None,
    ):
        validate_centered_window_size(window_size)
        self.genome_msa = genome_msa
        self.window_size = window_size
        self.tokenizer = Tokenizer()
        self.model = MLMforLogitsModel(model_path, model_revision=model_revision)

    def tokenize_function(self, positions: dict[str, list[Any]]) -> dict[str, Any]:
        chromosome_values, position_values = validate_positions_batch(
            positions["chrom"], positions["pos"]
        )
        chrom = np.array(chromosome_values)
        pos = np.array(position_values) - 1
        start = pos - self.window_size // 2
        end = pos + self.window_size // 2
        msa_fwd, msa_rev = _alignment(self.genome_msa, chrom, start, end)
        pos_fwd = self.window_size // 2
        pos_rev = pos_fwd - 1

        def prepare(msa: np.ndarray, center: int) -> tuple[Any, Any, Any]:
            input_ids, aux_features = msa[:, :, 0], msa[:, :, 1:]
            input_ids[:, center] = self.tokenizer.mask_token_id()
            return (
                input_ids.astype(np.int64),
                aux_features,
                np.full(len(input_ids), center),
            )

        fwd = prepare(msa_fwd, pos_fwd)
        rev = prepare(msa_rev, pos_rev)
        return {
            "input_ids_fwd": fwd[0],
            "aux_features_fwd": fwd[1],
            "pos_fwd": fwd[2],
            "input_ids_rev": rev[0],
            "aux_features_rev": rev[1],
            "pos_rev": rev[2],
        }

    def postprocess(self, predictions: Any) -> pd.DataFrame:
        return pd.DataFrame(predictions, columns=list("ACGT"))


class EmbeddingInference:
    def __init__(
        self,
        model_path: str,
        genome_msa: GenomeMSA,
        window_size: int,
        center_window_size: int,
        model_revision: str | None = None,
    ):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.genome_msa = genome_msa
        self.window_size = window_size
        self.tokenizer = Tokenizer()
        self.model = ModelCenterEmbedding(
            model_path,
            center_window_size,
            model_revision=model_revision,
        )

    def tokenize_function(self, windows: dict[str, list[Any]]) -> dict[str, Any]:
        chrom = np.array(windows["chrom"])
        start = np.array(windows["start"])
        end = np.array(windows["end"])
        msa_fwd, msa_rev = _alignment(self.genome_msa, chrom, start, end)

        def prepare(msa: np.ndarray) -> tuple[Any, Any]:
            return msa[:, :, 0].astype(np.int64), msa[:, :, 1:]

        fwd = prepare(msa_fwd)
        rev = prepare(msa_rev)
        return {
            "input_ids_fwd": fwd[0],
            "aux_features_fwd": fwd[1],
            "input_ids_rev": rev[0],
            "aux_features_rev": rev[1],
        }

    def postprocess(self, predictions: Any) -> pd.DataFrame:
        columns = [f"embedding_{index}" for index in range(predictions.shape[1])]
        return pd.DataFrame(predictions, columns=columns)


def _load_inputs(
    input_path: str,
    msa_path: str,
    split: str,
    is_file: bool,
) -> tuple[Dataset, GenomeMSA]:
    disable_caching()
    dataset = load_dataset_from_file_or_dir(
        input_path,
        split=split,
        is_file=is_file,
    )
    genome_msa = GenomeMSA(
        msa_path,
        subset_chroms=dataset.unique("chrom"),
        in_memory=False,
    )
    return dataset, genome_msa


def _execute(
    *,
    operation: str,
    dataset: Dataset,
    input_path: str,
    msa_path: str,
    model_path: str,
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
            family="msa",
            operation=operation,
            dataset=dataset,
            input_path=input_path,
            split=split,
            is_file=is_file,
            model_path=model_path,
            inference=inference,
            training_arguments=training_arguments,
            checkpoint_revision=checkpoint_arguments.checkpoint_revision,
            resources={"msa": resource_identity(msa_path)},
            operation_arguments=operation_arguments,
        )

    return execute_inference(
        dataset,
        inference,
        training_arguments,
        checkpoint_arguments,
        output_path=output_path,
        run_signature_factory=build_signature,
        output_prefix=f"gpn-msa-{operation}-",
    )


def _run(
    *,
    operation: str,
    input_path: str,
    msa_path: str,
    window_size: int,
    model_path: str,
    model_revision: str | None,
    output_path: Path,
    split: str,
    is_file: bool,
    center_window_size: int | None,
    training_arguments: TrainingArguments,
    checkpoint_arguments: CheckpointArguments,
) -> Path | None:
    dataset, genome_msa = _load_inputs(input_path, msa_path, split, is_file)
    inference: Any
    if operation == "vep":
        inference = VEPInference(
            model_path,
            genome_msa,
            window_size,
            model_revision=model_revision,
        )
    elif operation == "logits":
        inference = LogitsInference(
            model_path,
            genome_msa,
            window_size,
            model_revision=model_revision,
        )
    elif operation == "embedding":
        if center_window_size is None:
            raise ValueError("center_window_size is required for embedding")
        inference = EmbeddingInference(
            model_path,
            genome_msa,
            window_size,
            center_window_size,
            model_revision=model_revision,
        )
    else:
        raise ValueError(f"Unknown GPN-MSA inference operation: {operation}")
    operation_arguments = {
        "model_revision": model_revision,
        "window_size": window_size,
    }
    if center_window_size is not None:
        operation_arguments["center_window_size"] = center_window_size
    return _execute(
        operation=operation,
        dataset=dataset,
        input_path=input_path,
        msa_path=msa_path,
        model_path=model_path,
        output_path=output_path,
        split=split,
        is_file=is_file,
        inference=inference,
        training_arguments=training_arguments,
        checkpoint_arguments=checkpoint_arguments,
        operation_arguments=operation_arguments,
    )


def vep(
    input_path: str,
    msa_path: str,
    window_size: int,
    model_path: str,
    output_path: Path,
    *,
    model_revision: str | None,
    split: str,
    is_file: bool,
    training_arguments: TrainingArguments,
    checkpoint_arguments: CheckpointArguments,
) -> Path | None:
    return _run(
        operation="vep",
        input_path=input_path,
        msa_path=msa_path,
        window_size=window_size,
        model_path=model_path,
        model_revision=model_revision,
        output_path=output_path,
        split=split,
        is_file=is_file,
        center_window_size=None,
        training_arguments=training_arguments,
        checkpoint_arguments=checkpoint_arguments,
    )


def logits(
    input_path: str,
    msa_path: str,
    window_size: int,
    model_path: str,
    output_path: Path,
    *,
    model_revision: str | None,
    split: str,
    is_file: bool,
    training_arguments: TrainingArguments,
    checkpoint_arguments: CheckpointArguments,
) -> Path | None:
    return _run(
        operation="logits",
        input_path=input_path,
        msa_path=msa_path,
        window_size=window_size,
        model_path=model_path,
        model_revision=model_revision,
        output_path=output_path,
        split=split,
        is_file=is_file,
        center_window_size=None,
        training_arguments=training_arguments,
        checkpoint_arguments=checkpoint_arguments,
    )


def embedding(
    input_path: str,
    msa_path: str,
    window_size: int,
    model_path: str,
    output_path: Path,
    *,
    center_window_size: int,
    model_revision: str | None,
    split: str,
    is_file: bool,
    training_arguments: TrainingArguments,
    checkpoint_arguments: CheckpointArguments,
) -> Path | None:
    return _run(
        operation="embedding",
        input_path=input_path,
        msa_path=msa_path,
        window_size=window_size,
        model_path=model_path,
        model_revision=model_revision,
        output_path=output_path,
        split=split,
        is_file=is_file,
        center_window_size=center_window_size,
        training_arguments=training_arguments,
        checkpoint_arguments=checkpoint_arguments,
    )
