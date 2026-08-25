from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from gpn.data import Tokenizer as StarTokenizer
from gpn.ss.model import GPNConfig, GPNForMaskedLM
from gpn.ss.train import (
    DataTrainingArguments as GPNDataTrainingArguments,
)
from gpn.ss.train import (
    ModelArguments as GPNModelArguments,
)
from gpn.star.model import GPNStarConfig, GPNStarForMaskedLM
from gpn.star.train import (
    DataCollatorForLanguageModelingSimplified as StarDataCollator,
)
from gpn.star.train import (
    DataTrainingArguments as StarDataTrainingArguments,
)
from gpn.star.train import (
    ModelArguments as StarModelArguments,
)
from gpn.training import (
    GPNTrainingArguments,
    find_training_checkpoint,
    load_training_dataset,
    parse_training_arguments,
    reject_unsupported_hub_push,
)

ROOT = Path(__file__).parents[1]
RECIPE_DIRECTORIES = (
    ROOT / "recipes" / "gpn_training",
    ROOT / "recipes" / "gpn_star_training",
)
DATASET_REVISIONS = {
    "gpn_training": (
        "songlab/genomes-brassicales-balanced-v1",
        "d11c6084dd2bb5575f9ce224cbcc435a687e67bf",
    ),
    "gpn_star_training": (
        "songlab/gpn-msa-sapiens-dataset",
        "57c0e187c674761955518f3579eb0d7b5a0b7078",
    ),
}


@pytest.mark.parametrize("recipe_directory", RECIPE_DIRECTORIES)
def test_training_profiles_are_paired_and_use_prepared_inputs(recipe_directory):
    smoke = yaml.safe_load((recipe_directory / "cpu-smoke.yaml").read_text())
    gpu = yaml.safe_load((recipe_directory / "gpu.yaml").read_text())

    dataset_name, dataset_revision = DATASET_REVISIONS[recipe_directory.name]
    assert smoke["dataset_name"] == dataset_name
    assert smoke["dataset_revision"] == dataset_revision
    assert gpu["dataset_name"] == smoke["dataset_name"]
    assert gpu["dataset_revision"] == smoke["dataset_revision"]
    assert smoke["use_cpu"] is True
    assert smoke["max_steps"] == 1
    assert smoke["do_eval"] is False
    assert smoke["eval_strategy"] == "no"
    assert gpu["do_train"] is True
    assert gpu["do_eval"] is True
    assert gpu["report_to"] == []

    if recipe_directory.name == "gpn_star_training":
        for key in ("msa_path", "phylo_dist_path"):
            assert smoke[key].startswith("/path/to/prepared-")
            assert gpu[key] == smoke[key]


@pytest.mark.parametrize(
    ("recipe_directory", "argument_types"),
    (
        (
            ROOT / "recipes" / "gpn_training",
            (GPNModelArguments, GPNDataTrainingArguments, GPNTrainingArguments),
        ),
        (
            ROOT / "recipes" / "gpn_star_training",
            (StarModelArguments, StarDataTrainingArguments, GPNTrainingArguments),
        ),
    ),
)
@pytest.mark.parametrize("profile", ("cpu-smoke.yaml", "gpu.yaml"))
def test_training_profile_is_accepted_by_entrypoint_parser(
    recipe_directory, argument_types, profile, monkeypatch
):
    monkeypatch.setattr(
        "transformers.training_args.is_torch_bf16_gpu_available", lambda: True
    )
    values = yaml.safe_load((recipe_directory / profile).read_text())
    parsed = parse_training_arguments(
        argument_types[0],
        argument_types[1],
        recipe_directory / profile,
    )

    assert parsed[1].dataset_name == values["dataset_name"]
    assert parsed[1].dataset_revision == values["dataset_revision"]
    assert parsed[2].output_dir == values["output_dir"]
    if "overwrite_output_dir" in values:
        assert parsed[2].overwrite_output_dir is True


def test_training_parser_rejects_non_yaml_profiles(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text("{}")

    with pytest.raises(ValueError, match="must use the .yaml"):
        parse_training_arguments(
            GPNModelArguments,
            GPNDataTrainingArguments,
            profile,
        )


def test_training_dataset_revision_is_forwarded_independently(monkeypatch):
    calls = {}

    def fake_load_dataset(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return "dataset"

    monkeypatch.setattr("gpn.training.load_dataset", fake_load_dataset)

    result = load_training_dataset(
        "songlab/example",
        "config",
        dataset_revision="dataset-commit",
        cache_dir="cache",
        streaming=True,
    )

    assert result == "dataset"
    assert calls == {
        "args": ("songlab/example", "config"),
        "kwargs": {
            "revision": "dataset-commit",
            "cache_dir": "cache",
            "streaming": True,
        },
    }


def test_overwrite_output_does_not_resume_existing_checkpoint(tmp_path):
    (tmp_path / "checkpoint-1").mkdir()
    arguments = SimpleNamespace(
        resume_from_checkpoint=None,
        output_dir=str(tmp_path),
        do_train=True,
        overwrite_output_dir=True,
    )

    assert find_training_checkpoint(arguments) is None


def test_hub_push_is_rejected_instead_of_silently_ignored():
    with pytest.raises(ValueError, match="push_to_hub is not supported"):
        reject_unsupported_hub_push(SimpleNamespace(push_to_hub=True))


def test_star_collator_uses_stable_keyword_arguments():
    collator = StarDataCollator(
        tokenizer=StarTokenizer(),
        clades=torch.arange(20),
        mlm_probability=0.15,
    )

    assert collator.mlm_probability == pytest.approx(0.15)
    assert not hasattr(collator, "tokenizer")

    source_ids = np.random.default_rng(42).integers(0, 6, (8, 20))
    target_species = np.array([0, 1])
    batch = collator(
        [
            {
                "input_ids": source_ids[:, :2],
                "loss_weight": np.ones((8, 2)),
                "target_species": target_species,
                "source_ids": source_ids,
            }
        ]
    )
    assert batch["labels"].shape == (1, 8, 2)


def test_star_collator_normalizes_production_numpy_dtypes():
    collator = StarDataCollator(
        tokenizer=StarTokenizer(),
        clades=torch.arange(20),
        mlm_probability=0,
    )
    examples = [
        {
            "input_ids": np.full((4, 2), 2, dtype=np.uint8),
            "source_ids": np.full((4, 20), 2, dtype=np.uint8),
            "target_species": np.array([0, 1], dtype=np.int32),
            "loss_weight": np.ones((4, 2), dtype=float),
        }
    ]

    batch = collator.torch_call(examples)

    assert batch["input_ids"].dtype == torch.long
    assert batch["source_ids"].dtype == torch.long
    assert batch["target_species"].dtype == torch.long
    assert batch["labels"].dtype == torch.long
    assert torch.equal(
        batch["labels"],
        torch.full((1, 4, 2), -100, dtype=torch.long),
    )
    assert batch["loss_weight"].dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_star_collator_masks_cuda_tensors():
    device = torch.device("cuda")
    collator = StarDataCollator(
        tokenizer=StarTokenizer(),
        clades=torch.arange(20),
        mlm_probability=1.0,
    )
    inputs = torch.full((1, 4, 2), 2, dtype=torch.long, device=device)
    source_ids = torch.full((1, 4, 20), 2, dtype=torch.long, device=device)
    target_species = torch.tensor([[0, 1]], dtype=torch.long, device=device)

    masked_inputs, labels, masked_source_ids = collator.torch_mask_tokens(
        inputs,
        source_ids,
        target_species,
    )

    assert masked_inputs.device == device
    assert labels.device == device
    assert masked_source_ids.device == device


def test_tiny_gpn_training_step():
    model = GPNForMaskedLM(
        GPNConfig(
            num_hidden_layers=2,
            hidden_size=16,
            intermediate_size=32,
            first_kernel_size=3,
            rest_kernel_size=3,
            dilation_cycle=2,
        )
    )
    input_ids = torch.randint(0, model.config.vocab_size, (2, 16))
    labels = input_ids.clone()
    labels[:, ::2] = -100

    loss = model(input_ids=input_ids, labels=labels).loss
    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_tiny_gpn_star_training_step(tmp_path):
    species = 20
    asset_dir = tmp_path / "source-phylo-dist"
    asset_dir.mkdir()
    pairwise = np.ones((species, species), dtype=np.float32)
    np.fill_diagonal(pairwise, 0.0)
    np.save(asset_dir / "pairwise.npy", pairwise)
    np.save(asset_dir / "in_clade.npy", np.zeros(species, dtype=np.float32))

    model = GPNStarForMaskedLM(
        GPNStarConfig(
            phylo_dist_path=str(asset_dir),
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=16,
            intermediate_size=32,
            max_position_embeddings=32,
        )
    )
    batch_size, length = 1, 8
    source_ids = torch.randint(
        0, model.config.vocab_size, (batch_size, length, species)
    )
    target_species = torch.tensor([[0, 1]])
    input_ids = torch.gather(
        source_ids,
        dim=2,
        index=target_species[:, None, :].expand(-1, length, -1),
    )
    labels = input_ids.clone()
    labels[:, ::2] = -100

    loss = model(
        input_ids=input_ids,
        source_ids=source_ids,
        target_species=target_species,
        labels=labels,
    ).loss
    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())

    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(checkpoint)
    asset_dir.rename(tmp_path / "moved-source-phylo-dist")
    reloaded = GPNStarForMaskedLM.from_pretrained(checkpoint)

    assert reloaded.config.phylo_dist_path == str(checkpoint / "phylo_dist")
    assert (checkpoint / "phylo_dist" / "pairwise.npy").is_file()
    assert (checkpoint / "phylo_dist" / "in_clade.npy").is_file()
