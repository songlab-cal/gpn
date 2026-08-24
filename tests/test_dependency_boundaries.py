from importlib import metadata, util

import pytest


@pytest.mark.parametrize(
    "module",
    (
        "gpn.ss.data",
        "gpn.ss.filter_assemblies",
        "gpn.ss.finetune",
        "gpn.ss.train_tokenizer_ss",
    ),
)
def test_archived_dataset_and_finetuning_modules_are_not_packaged(module):
    assert util.find_spec(module) is None


def test_dataset_building_dependencies_are_not_declared():
    requirements = "\n".join(metadata.requires("gpn") or ()).lower()

    for package in ("bioframe", "pybigwig"):
        assert package not in requirements


def test_zstandard_is_training_only():
    requirements = [
        requirement.lower()
        for requirement in metadata.requires("gpn") or ()
        if requirement.lower().startswith("zstandard")
    ]

    assert requirements == ["zstandard>=0.22; extra == 'train'"]


def test_experiment_tracking_is_opt_in():
    requirements = [
        requirement.lower()
        for requirement in metadata.requires("gpn") or ()
        if requirement.lower().startswith("wandb")
    ]

    assert requirements == ["wandb>=0.18; extra == 'tracking'"]
