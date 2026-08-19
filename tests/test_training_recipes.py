import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import HfArgumentParser

from gpn.model import GPNConfig, GPNForMaskedLM
from gpn.ss.run_mlm import (
    DataTrainingArguments as GPNDataTrainingArguments,
)
from gpn.ss.run_mlm import (
    ModelArguments as GPNModelArguments,
)
from gpn.star.data import Tokenizer as StarTokenizer
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
from gpn.training import GPNTrainingArguments, hf_token_kwargs

ROOT = Path(__file__).parents[1]
RECIPE_DIRECTORIES = (
    ROOT / "recipes" / "gpn_training",
    ROOT / "recipes" / "gpn_star_training",
)


@pytest.mark.parametrize("recipe_directory", RECIPE_DIRECTORIES)
def test_training_profiles_are_paired_and_use_prepared_inputs(recipe_directory):
    smoke = json.loads((recipe_directory / "cpu-smoke.json").read_text())
    gpu = json.loads((recipe_directory / "gpu.json").read_text())

    assert smoke["dataset_name"].startswith("/path/to/prepared-")
    assert gpu["dataset_name"] == smoke["dataset_name"]
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
@pytest.mark.parametrize("profile", ("cpu-smoke.json", "gpu.json"))
def test_training_profile_is_accepted_by_entrypoint_parser(
    recipe_directory, argument_types, profile, monkeypatch
):
    monkeypatch.setattr(
        "transformers.training_args.is_torch_bf16_gpu_available", lambda: True
    )
    values = json.loads((recipe_directory / profile).read_text())
    parsed = HfArgumentParser(argument_types).parse_dict(values)

    assert parsed[0].model_type in {"GPN", "GPNStar"}
    assert parsed[1].dataset_name == values["dataset_name"]
    assert parsed[2].output_dir == values["output_dir"]
    if "overwrite_output_dir" in values:
        assert parsed[2].overwrite_output_dir is True


@pytest.mark.parametrize("module", ("gpn.ss.run_mlm", "gpn.star.train"))
def test_training_entrypoint_help_is_offline(module):
    environment = os.environ.copy()
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "--dataset_name" in result.stdout


def test_hugging_face_auth_uses_current_token_keyword():
    assert hf_token_kwargs(False) == {}
    assert hf_token_kwargs(True) == {"token": True}


def test_star_collator_uses_stable_keyword_arguments():
    collator = StarDataCollator(
        tokenizer=StarTokenizer(),
        clades=torch.arange(20),
        mlm_probability=0.15,
    )

    assert collator.mlm_probability == pytest.approx(0.15)


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
    pairwise = np.ones((species, species), dtype=np.float32)
    np.fill_diagonal(pairwise, 0.0)
    np.save(tmp_path / "pairwise.npy", pairwise)
    np.save(tmp_path / "in_clade.npy", np.zeros(species, dtype=np.float32))

    model = GPNStarForMaskedLM(
        GPNStarConfig(
            phylo_dist_path=str(tmp_path),
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
