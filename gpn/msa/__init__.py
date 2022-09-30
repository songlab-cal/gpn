from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

from .model import MSAConvNetConfig, MSAConvNetModel, MSAConvNetForMaskedLM

AutoConfig.register("MSAConvNet", MSAConvNetConfig)
AutoModel.register(MSAConvNetConfig, MSAConvNetModel)
AutoModelForMaskedLM.register(MSAConvNetConfig, MSAConvNetForMaskedLM)
