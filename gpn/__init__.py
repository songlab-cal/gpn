from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

from .model import ConvNetConfig, ConvNetModel, ConvNetForMaskedLM


AutoConfig.register("ConvNet", ConvNetConfig)
AutoModel.register(ConvNetConfig, ConvNetModel)
AutoModelForMaskedLM.register(ConvNetConfig, ConvNetForMaskedLM)