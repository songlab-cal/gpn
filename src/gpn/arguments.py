"""Typed options shared by the public CLI and inference runtime."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckpointArguments:
    """Durable batching options shared by every inference command."""

    checkpoint_batch_size: int | None = None
    checkpoint_dir: Path | None = None
    checkpoint_revision: str | None = None
    cleanup_checkpoints: bool = False

    def __post_init__(self) -> None:
        if self.checkpoint_batch_size is not None and self.checkpoint_batch_size <= 0:
            raise ValueError("checkpoint_batch_size must be positive")
        if self.checkpoint_batch_size is None and (
            self.checkpoint_dir is not None
            or self.checkpoint_revision is not None
            or self.cleanup_checkpoints
        ):
            raise ValueError(
                "checkpoint_dir, checkpoint_revision, and cleanup_checkpoints "
                "require checkpoint_batch_size"
            )
