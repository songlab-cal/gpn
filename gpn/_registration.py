"""Explicit registration of GPN model families with Transformers AutoClasses."""

from __future__ import annotations

from threading import Lock
from typing import Literal

ModelFamily = Literal["gpn", "star"]

_FAMILIES: tuple[ModelFamily, ...] = ("gpn", "star")
_registered_families: set[ModelFamily] = set()
_registration_lock = Lock()


def register_auto_classes(*families: ModelFamily) -> None:
    """Register GPN configurations and models with Transformers AutoClasses.

    Calling this function repeatedly is safe. With no arguments, all installed GPN
    model families are registered. Pass ``"gpn"`` or ``"star"`` to register only
    one family and avoid importing the other implementation.

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
            if family == "gpn":
                _register_gpn()
            else:
                _register_star()
            _registered_families.add(family)


def _register_gpn() -> None:
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForMaskedLM,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
    )

    from .legacy import (
        ConvNetConfig,
        ConvNetForMaskedLM,
        ConvNetForSequenceClassification,
        ConvNetModel,
        GPNRoFormerConfig,
        GPNRoFormerForMaskedLM,
        GPNRoFormerModel,
    )
    from .model import (
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

    AutoConfig.register("GPNRoFormer", GPNRoFormerConfig)
    AutoModel.register(GPNRoFormerConfig, GPNRoFormerModel)
    AutoModelForMaskedLM.register(GPNRoFormerConfig, GPNRoFormerForMaskedLM)


def _register_star() -> None:
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

    from .star.model import GPNStarConfig, GPNStarForMaskedLM, GPNStarModel

    AutoConfig.register("GPNStar", GPNStarConfig)
    AutoModel.register(GPNStarConfig, GPNStarModel)
    AutoModelForMaskedLM.register(GPNStarConfig, GPNStarForMaskedLM)
