import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import datasets
import transformers
import yaml
from datasets import DatasetDict, IterableDatasetDict, load_dataset
from transformers import HfArgumentParser, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint


def load_training_dataset(
    dataset_name: str,
    dataset_config_name: str | None,
    *,
    dataset_revision: str | None,
    cache_dir: str | None,
    streaming: bool = False,
) -> DatasetDict | IterableDatasetDict:
    """Load a training dataset with an explicit, independently pinned revision."""

    return load_dataset(
        dataset_name,
        dataset_config_name,
        revision=dataset_revision,
        cache_dir=cache_dir,
        streaming=streaming,
    )


def parse_training_arguments(
    model_arguments: type[Any],
    data_arguments: type[Any],
    profile: Path,
) -> tuple[Any, Any, "GPNTrainingArguments"]:
    """Parse one reviewed YAML training profile."""

    if profile.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Training profiles must use the .yaml or .yml extension")
    values = yaml.safe_load(profile.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Training profile must contain a YAML mapping")
    parser = HfArgumentParser((model_arguments, data_arguments, GPNTrainingArguments))
    parsed = parser.parse_dict(values)
    reject_unsupported_hub_push(parsed[2])
    return parsed


def reject_unsupported_hub_push(training_args: "GPNTrainingArguments") -> None:
    """Prevent Transformers' inherited Hub flag from silently doing nothing."""

    if training_args.push_to_hub:
        raise ValueError(
            "push_to_hub is not supported by the GPN training entry points; "
            "review and publish the saved checkpoint separately."
        )


def configure_training_logging(
    training_args: "GPNTrainingArguments", logger: logging.Logger
) -> None:
    """Use one logging setup for both maintained trainers."""

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.info("Training arguments: %s", training_args)


def find_training_checkpoint(
    training_args: "GPNTrainingArguments",
) -> str | bool | None:
    """Resolve explicit or automatic resume state without overwriting output."""

    if training_args.resume_from_checkpoint is not None:
        return training_args.resume_from_checkpoint
    output_dir = Path(training_args.output_dir)
    if not (
        output_dir.is_dir()
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        return None
    checkpoint = get_last_checkpoint(str(output_dir))
    if checkpoint is not None:
        return checkpoint
    if any(output_dir.iterdir()) and not training_args.overwrite_output_dir:
        raise ValueError(
            f"Output directory '{output_dir}' is not empty and has no checkpoint; "
            "choose another directory or set overwrite_output_dir."
        )
    return None


def train_and_save(
    trainer: Trainer,
    training_args: "GPNTrainingArguments",
    checkpoint: str | bool | None,
) -> None:
    """Run the configured training phase and persist its reproducibility state."""

    if not training_args.do_train:
        return
    result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()


def evaluate_and_save(trainer: Trainer, training_args: "GPNTrainingArguments") -> None:
    """Run evaluation and report loss-derived perplexity."""

    if not training_args.do_eval:
        return
    metrics = trainer.evaluate()
    try:
        metrics["perplexity"] = math.exp(metrics["eval_loss"])
    except OverflowError:
        metrics["perplexity"] = float("inf")
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


@dataclass
class GPNTrainingArguments(TrainingArguments):
    """Training arguments shared by the maintained GPN trainers.

    Transformers 5 removed ``overwrite_output_dir`` from its public training
    arguments, but GPN still uses the flag to make intentionally disposable
    smoke-test output explicit. Owning the compatibility field here keeps the
    command-line contract stable across the supported Transformers range.
    """

    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": "Allow training to replace an existing non-empty output directory."
        },
    )
