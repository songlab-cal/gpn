from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

from .convnet import ConvNetConfig, ConvNetModel, ConvNetForMaskedLM

AutoConfig.register("ConvNet", ConvNetConfig)
AutoModel.register(ConvNetConfig, ConvNetModel)
AutoModelForMaskedLM.register(ConvNetConfig, ConvNetForMaskedLM)
