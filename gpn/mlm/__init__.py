from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

from .convnet import ConvNetConfig, ConvNetModel, ConvNetForMaskedLM

AutoConfig.register("ConvNet", ConvNetConfig)
AutoModel.register(ConvNetConfig, ConvNetModel)
AutoModelForMaskedLM.register(ConvNetConfig, ConvNetForMaskedLM)

from .unet import UNetConfig, UNetModel, UNetForMaskedLM

AutoConfig.register("UNet", UNetConfig)
AutoModel.register(UNetConfig, UNetModel)
AutoModelForMaskedLM.register(UNetConfig, UNetForMaskedLM)

from .convtransformer import ConvTransformerConfig, ConvTransformerModel, ConvTransformerForMaskedLM

AutoConfig.register("ConvTransformer", ConvTransformerConfig)
AutoModel.register(ConvTransformerConfig, ConvTransformerModel)
AutoModelForMaskedLM.register(ConvTransformerConfig, ConvTransformerForMaskedLM)
