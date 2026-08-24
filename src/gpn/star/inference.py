"""Maintained GPN-Star inference operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, disable_caching
from transformers import AutoModel, AutoModelForMaskedLM, TrainingArguments

from gpn import register_auto_classes
from gpn.arguments import CheckpointArguments
from gpn.data import (
    ReverseComplementer,
    Tokenizer,
    load_dataset_from_file_or_dir,
)
from gpn.inference import (
    build_run_signature,
    execute_inference,
    resource_identity,
)
from gpn.scoring import (
    require_reference_matches,
    validate_centered_window_size,
    validate_positions_batch,
    validate_snv_batch,
)
from gpn.star.data import GenomeMSA
from gpn.star.utils import find_directory_sum_paths


class MLMforVEPModel(torch.nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        register_auto_classes("star")
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        self.model.eval()

    def get_llr(
        self,
        input_ids: torch.Tensor,
        source_ids: torch.Tensor,
        target_species: torch.Tensor,
        pos: torch.Tensor,
        ref: torch.Tensor,
        alt: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.model(
            input_ids=input_ids,
            source_ids=source_ids,
            target_species=target_species,
        ).logits
        logits = logits[torch.arange(len(pos), device=pos.device), pos, 0]
        row = torch.arange(len(ref), device=ref.device)
        return logits[row, alt] - logits[row, ref]

    def forward(
        self,
        input_ids_fwd: torch.Tensor | None = None,
        source_ids_fwd: torch.Tensor | None = None,
        pos_fwd: torch.Tensor | None = None,
        ref_fwd: torch.Tensor | None = None,
        alt_fwd: torch.Tensor | None = None,
        input_ids_rev: torch.Tensor | None = None,
        source_ids_rev: torch.Tensor | None = None,
        pos_rev: torch.Tensor | None = None,
        ref_rev: torch.Tensor | None = None,
        alt_rev: torch.Tensor | None = None,
        target_species: torch.Tensor | None = None,
    ) -> torch.Tensor:
        llr_fwd = self.get_llr(
            input_ids_fwd,
            source_ids_fwd,
            target_species,
            pos_fwd,
            ref_fwd,
            alt_fwd,
        )
        llr_rev = self.get_llr(
            input_ids_rev,
            source_ids_rev,
            target_species,
            pos_rev,
            ref_rev,
            alt_rev,
        )
        return (llr_fwd + llr_rev) / 2


class MLMforLogitsModel(torch.nn.Module):
    def __init__(self, model_path: str, phylo_dist_path: str | None = None):
        super().__init__()
        register_auto_classes("star")
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_path,
            phylo_dist_path=phylo_dist_path,
        )
        self.model.eval()
        tokenizer = Tokenizer()
        self.nucleotide_ids = [tokenizer.vocab.index(base) for base in "ACGT"]

    def get_logits(
        self,
        input_ids: torch.Tensor,
        source_ids: torch.Tensor,
        target_species: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.model(
            input_ids=input_ids,
            source_ids=source_ids,
            target_species=target_species,
        ).logits
        return logits[torch.arange(len(pos), device=pos.device), pos].squeeze(-2)

    def forward(
        self,
        input_ids_fwd: torch.Tensor | None = None,
        source_ids_fwd: torch.Tensor | None = None,
        pos_fwd: torch.Tensor | None = None,
        input_ids_rev: torch.Tensor | None = None,
        source_ids_rev: torch.Tensor | None = None,
        pos_rev: torch.Tensor | None = None,
        target_species: torch.Tensor | None = None,
    ) -> torch.Tensor:
        a, c, g, t = self.nucleotide_ids
        logits_fwd = self.get_logits(
            input_ids_fwd, source_ids_fwd, target_species, pos_fwd
        )[:, [a, c, g, t]]
        logits_rev = self.get_logits(
            input_ids_rev, source_ids_rev, target_species, pos_rev
        )[:, [t, g, c, a]]
        return (logits_fwd + logits_rev) / 2


class ModelCenterEmbedding(torch.nn.Module):
    def __init__(self, model_path: str, center_window_size: int):
        super().__init__()
        if center_window_size <= 0:
            raise ValueError("center_window_size must be positive")
        register_auto_classes("star")
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()
        self.center_window_size = center_window_size

    def get_center_embedding(
        self,
        input_ids: torch.Tensor,
        source_ids: torch.Tensor,
        target_species: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.model(
            input_ids=input_ids,
            source_ids=source_ids,
            target_species=target_species,
        ).last_hidden_state
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
        source_ids_fwd: torch.Tensor | None = None,
        source_ids_rev: torch.Tensor | None = None,
        target_species: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedding_fwd = self.get_center_embedding(
            input_ids_fwd, source_ids_fwd, target_species
        )
        embedding_rev = self.get_center_embedding(
            input_ids_rev, source_ids_rev, target_species
        )
        return (embedding_fwd + embedding_rev) / 2


def _alignment(
    genome_msa_list: list[GenomeMSA],
    chrom: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    forward, reverse = zip(
        *(
            genome_msa.get_msa_batch_fwd_rev(chrom, start, end, tokenize=True)
            for genome_msa in genome_msa_list
        )
    )
    return np.concatenate(forward, axis=-1), np.concatenate(reverse, axis=-1)


def _target_species(batch_size: int) -> np.ndarray:
    return np.zeros((batch_size, 1), dtype=int)


class VEPInference:
    def __init__(
        self,
        model_path: str,
        genome_msa_list: list[GenomeMSA],
        window_size: int,
    ):
        validate_centered_window_size(window_size)
        self.genome_msa_list = genome_msa_list
        self.window_size = window_size
        self.tokenizer = Tokenizer()
        self.model = MLMforVEPModel(model_path)
        self.reverse_complementer = ReverseComplementer()

    def tokenize_function(self, variants: dict[str, list[Any]]) -> dict[str, Any]:
        chromosomes, positions, references, alternates = validate_snv_batch(
            variants["chrom"], variants["pos"], variants["ref"], variants["alt"]
        )
        chrom = np.array(chromosomes)
        pos = np.array(positions) - 1
        start = pos - self.window_size // 2
        end = pos + self.window_size // 2
        msa_fwd, msa_rev = _alignment(self.genome_msa_list, chrom, start, end)
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
            input_ids = msa[:, :, :1]
            require_reference_matches(
                [self.tokenizer.vocab[int(value)] for value in input_ids[:, center, 0]],
                [self.tokenizer.vocab[int(value)] for value in reference_ids],
                chromosomes,
                positions,
                orientation=orientation,
            )
            input_ids[:, center, 0] = self.tokenizer.mask_token_id()
            msa[:, center, 0] = self.tokenizer.mask_token_id()
            return (
                input_ids,
                msa,
                np.full(input_ids.shape[0], center),
                reference_ids.astype(np.int64),
                alternate_ids.astype(np.int64),
            )

        fwd = prepare(msa_fwd, pos_fwd, ref_fwd, alt_fwd, "forward")
        rev = prepare(msa_rev, pos_rev, ref_rev, alt_rev, "reverse-complement")
        return {
            "input_ids_fwd": fwd[0],
            "source_ids_fwd": fwd[1],
            "pos_fwd": fwd[2],
            "ref_fwd": fwd[3],
            "alt_fwd": fwd[4],
            "input_ids_rev": rev[0],
            "source_ids_rev": rev[1],
            "pos_rev": rev[2],
            "ref_rev": rev[3],
            "alt_rev": rev[4],
            "target_species": _target_species(len(chrom)),
        }

    def postprocess(self, predictions: Any) -> pd.DataFrame:
        return pd.DataFrame(predictions, columns=["score"])


class LogitsInference:
    def __init__(
        self,
        model_path: str,
        genome_msa_list: list[GenomeMSA],
        window_size: int,
        phylo_dist_path: str | None = None,
    ):
        validate_centered_window_size(window_size)
        self.genome_msa_list = genome_msa_list
        self.window_size = window_size
        self.tokenizer = Tokenizer()
        self.model = MLMforLogitsModel(model_path, phylo_dist_path)

    def tokenize_function(self, positions: dict[str, list[Any]]) -> dict[str, Any]:
        chromosome_values, position_values = validate_positions_batch(
            positions["chrom"], positions["pos"]
        )
        chrom = np.array(chromosome_values)
        pos = np.array(position_values) - 1
        start = pos - self.window_size // 2
        end = pos + self.window_size // 2
        msa_fwd, msa_rev = _alignment(self.genome_msa_list, chrom, start, end)
        pos_fwd = self.window_size // 2
        pos_rev = pos_fwd - 1

        def prepare(msa: np.ndarray, center: int) -> tuple[Any, Any, Any]:
            input_ids = msa[:, :, :1]
            input_ids[:, center] = self.tokenizer.mask_token_id()
            return input_ids.astype(np.int64), msa, np.full(len(input_ids), center)

        fwd = prepare(msa_fwd, pos_fwd)
        rev = prepare(msa_rev, pos_rev)
        return {
            "input_ids_fwd": fwd[0],
            "source_ids_fwd": fwd[1],
            "pos_fwd": fwd[2],
            "input_ids_rev": rev[0],
            "source_ids_rev": rev[1],
            "pos_rev": rev[2],
            "target_species": _target_species(len(chrom)),
        }

    def postprocess(self, predictions: Any) -> pd.DataFrame:
        return pd.DataFrame(predictions, columns=list("ACGT"))


class EmbeddingInference:
    def __init__(
        self,
        model_path: str,
        genome_msa_list: list[GenomeMSA],
        window_size: int,
        center_window_size: int,
    ):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.genome_msa_list = genome_msa_list
        self.window_size = window_size
        self.tokenizer = Tokenizer()
        self.model = ModelCenterEmbedding(model_path, center_window_size)

    def tokenize_function(self, windows: dict[str, list[Any]]) -> dict[str, Any]:
        chrom = np.array(windows["chrom"])
        start = np.array(windows["start"])
        end = np.array(windows["end"])
        msa_fwd, msa_rev = _alignment(self.genome_msa_list, chrom, start, end)

        def prepare(msa: np.ndarray) -> tuple[Any, Any]:
            return msa[:, :, :1].astype(np.int64), msa

        fwd = prepare(msa_fwd)
        rev = prepare(msa_rev)
        return {
            "input_ids_fwd": fwd[0],
            "source_ids_fwd": fwd[1],
            "input_ids_rev": rev[0],
            "source_ids_rev": rev[1],
            "target_species": _target_species(len(chrom)),
        }

    def postprocess(self, predictions: Any) -> pd.DataFrame:
        columns = [f"embedding_{index}" for index in range(predictions.shape[1])]
        return pd.DataFrame(predictions, columns=columns)


def _load_inputs(
    input_path: str,
    msa_path: str,
    split: str,
    is_file: bool,
) -> tuple[Dataset, dict[int, str], list[GenomeMSA]]:
    disable_caching()
    dataset = load_dataset_from_file_or_dir(
        input_path,
        split=split,
        is_file=is_file,
    )
    msa_paths = find_directory_sum_paths(msa_path)
    genome_msa_list = [
        GenomeMSA(
            path,
            n_species=n_species,
            subset_chroms=dataset.unique("chrom"),
            in_memory=False,
        )
        for n_species, path in msa_paths.items()
    ]
    return dataset, msa_paths, genome_msa_list


def _execute(
    *,
    operation: str,
    dataset: Dataset,
    input_path: str,
    msa_paths: dict[int, str],
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
        msa_resources = [
            {
                "order": order,
                "n_species": n_species,
                "resource": resource_identity(path),
            }
            for order, (n_species, path) in enumerate(msa_paths.items())
        ]
        config = getattr(getattr(inference.model, "model", None), "config", None)
        resolved_phylo_dist_path = getattr(config, "phylo_dist_path", None)
        return build_run_signature(
            family="star",
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
                "msa": msa_resources,
                "phylo_dist": (
                    resource_identity(resolved_phylo_dist_path)
                    if resolved_phylo_dist_path is not None
                    else None
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
        output_prefix=f"gpn-star-{operation}-",
    )


def _run(
    *,
    operation: str,
    input_path: str,
    msa_path: str,
    window_size: int,
    model_path: str,
    output_path: Path,
    split: str,
    is_file: bool,
    center_window_size: int | None,
    phylo_dist_path: str | None,
    training_arguments: TrainingArguments,
    checkpoint_arguments: CheckpointArguments,
) -> Path | None:
    dataset, msa_paths, genome_msa_list = _load_inputs(
        input_path, msa_path, split, is_file
    )
    inference: Any
    if operation == "vep":
        inference = VEPInference(model_path, genome_msa_list, window_size)
    elif operation == "logits":
        inference = LogitsInference(
            model_path,
            genome_msa_list,
            window_size,
            phylo_dist_path=phylo_dist_path,
        )
    elif operation == "embedding":
        if center_window_size is None:
            raise ValueError("center_window_size is required for embedding")
        inference = EmbeddingInference(
            model_path, genome_msa_list, window_size, center_window_size
        )
    else:
        raise ValueError(f"Unknown GPN-Star inference operation: {operation}")
    operation_arguments: dict[str, Any] = {"window_size": window_size}
    if center_window_size is not None:
        operation_arguments["center_window_size"] = center_window_size
    if phylo_dist_path is not None:
        operation_arguments["phylo_dist_path"] = phylo_dist_path
    return _execute(
        operation=operation,
        dataset=dataset,
        input_path=input_path,
        msa_paths=msa_paths,
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
        output_path=output_path,
        split=split,
        is_file=is_file,
        center_window_size=None,
        phylo_dist_path=None,
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
    phylo_dist_path: str | None,
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
        output_path=output_path,
        split=split,
        is_file=is_file,
        center_window_size=None,
        phylo_dist_path=phylo_dist_path,
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
        output_path=output_path,
        split=split,
        is_file=is_file,
        center_window_size=center_window_size,
        phylo_dist_path=None,
        training_arguments=training_arguments,
        checkpoint_arguments=checkpoint_arguments,
    )
