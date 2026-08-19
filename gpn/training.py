from dataclasses import dataclass, field

from transformers import TrainingArguments


def hf_token_kwargs(use_auth_token: bool) -> dict[str, bool]:
    """Translate GPN's stable CLI flag to the current Hugging Face API."""

    return {"token": True} if use_auth_token else {}


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
