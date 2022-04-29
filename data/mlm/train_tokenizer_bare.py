import numpy as np
import pandas as pd
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from transformers import BertTokenizerFast, LongformerTokenizerFast, RobertaTokenizerFast, PreTrainedTokenizerFast, AutoTokenizer


dataset = ["acgt"]

batch_size = 1000

def batch_iterator():
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i + batch_size]

special_tokens = ["[MASK]"]

tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = normalizers.Lowercase()
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()  # should not be used at all
trainer = trainers.BpeTrainer(
    vocab_size=5,
    special_tokens=special_tokens,
    initial_alphabet=["a", "c", "g", "t"],
)


tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(dataset))
cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")
#tokenizer.post_processor = processors.TemplateProcessing(
#    single="[CLS]:0 $A:0 [SEP]:0",
#    pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
#    special_tokens=[
#        ("[CLS]", cls_token_id),
#        ("[SEP]", sep_token_id),
#    ],
#)
#tokenizer = BertTokenizerFast(tokenizer_object=tokenizer)
#tokenizer = LongformerTokenizerFast(tokenizer_object=tokenizer)
#tokenizer = RobertaTokenizerFast(tokenizer_object=tokenizer)
tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, mask_token="[MASK]")
print(len(tokenizer))
tokenizer.save_pretrained(f"./tokenizer_bare/")

tokenizer2 = AutoTokenizer.from_pretrained("./tokenizer_bare/")
print(tokenizer2.get_vocab())
