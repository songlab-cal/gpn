"""Explicit registration of GPN model families with Transformers AutoClasses."""

from threading import Lock
from typing import Literal

ModelFamily = Literal["ss", "msa", "star", "phylo"]

_FAMILIES: tuple[ModelFamily, ...] = ("ss", "msa", "star", "phylo")
_registered_families: set[ModelFamily] = set()
_registration_lock = Lock()


def register_auto_classes(*families: ModelFamily) -> None:
    """Register GPN configurations and models with Transformers AutoClasses.

    Calling this function repeatedly is safe. With no arguments, all installed GPN
    model families are registered. Pass a family name to register only that family
    and avoid importing the other implementations.

    Raises:
        ValueError: If an unknown family is requested or a Transformers mapping is
            already owned by a different implementation.
    """

    requested = families or _FAMILIES
    unknown = set(requested).difference(_FAMILIES)
    if unknown:
        choices = ", ".join(_FAMILIES)
        invalid = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown GPN model family: {invalid}. Choose from: {choices}")

    with _registration_lock:
        for family in requested:
            if family in _registered_families:
                continue
            if family == "ss":
                _register_ss()
            elif family == "msa":
                _register_msa()
            elif family == "star":
                _register_star()
            else:
                _register_phylo()
            _registered_families.add(family)


def _register_ss() -> None:
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForMaskedLM,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
    )

    from .ss.model import (
        ConvNetConfig,
        ConvNetForMaskedLM,
        ConvNetForSequenceClassification,
        ConvNetModel,
        GPNConfig,
        GPNForMaskedLM,
        GPNForSequenceClassification,
        GPNForTokenClassification,
        GPNModel,
    )

    AutoConfig.register("GPN", GPNConfig)
    AutoModel.register(GPNConfig, GPNModel)
    AutoModelForMaskedLM.register(GPNConfig, GPNForMaskedLM)
    AutoModelForSequenceClassification.register(GPNConfig, GPNForSequenceClassification)
    AutoModelForTokenClassification.register(GPNConfig, GPNForTokenClassification)

    AutoConfig.register("ConvNet", ConvNetConfig)
    AutoModel.register(ConvNetConfig, ConvNetModel)
    AutoModelForMaskedLM.register(ConvNetConfig, ConvNetForMaskedLM)
    AutoModelForSequenceClassification.register(
        ConvNetConfig, ConvNetForSequenceClassification
    )



def _register_msa() -> None:
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

    from .msa.model import GPNMSAConfig, GPNMSAForMaskedLM, GPNMSAModel

    # Keep the published checkpoint identifier even though the Python API uses
    # the clearer GPN-MSA family name.
    AutoConfig.register("GPNRoFormer", GPNMSAConfig)
    AutoModel.register(GPNMSAConfig, GPNMSAModel)
    AutoModelForMaskedLM.register(GPNMSAConfig, GPNMSAForMaskedLM)


def _register_star() -> None:
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

    from .star.model import GPNStarConfig, GPNStarForMaskedLM, GPNStarModel

    AutoConfig.register("GPNStar", GPNStarConfig)
    AutoModel.register(GPNStarConfig, GPNStarModel)
    AutoModelForMaskedLM.register(GPNStarConfig, GPNStarForMaskedLM)


def _register_phylo() -> None:
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    from .phylo.model import PhyloGPNConfig, PhyloGPNModel, PhyloGPNTokenizer

    AutoConfig.register("phylogpn", PhyloGPNConfig)
    AutoModel.register(PhyloGPNConfig, PhyloGPNModel)
    AutoTokenizer.register(
        PhyloGPNConfig,
        slow_tokenizer_class=PhyloGPNTokenizer,
    )
