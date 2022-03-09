import numpy as np
import pandas as pd
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from transformers import BertTokenizerFast
import sys

# based on examples here:
# https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/tokenizer_training.ipynb

with open(sys.argv[1]) as file:
    dataset = file.readlines()
#dataset = dataset[np.random.choice()]
print(len(dataset))
print(dataset[0])


model = sys.argv[2]
vocab_size = int(sys.argv[3])

batch_size = 1000

def batch_iterator():
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i + batch_size]


if model == "bpe":
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.Lowercase()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[CLS]", "[SEP]", "[UNK]", "[PAD]", "[MASK]"])
elif model == "unigram":
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.normalizer = normalizers.Lowercase()
    trainer = trainers.UnigramTrainer(vocab_size=vocab_size, special_tokens=["[CLS]", "[SEP]", "[UNK]", "[PAD]", "[MASK]"], unk_token="[UNK]")


tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(dataset))
cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")
tokenizer.post_processor = processors.TemplateProcessing(
    single="[CLS]:0 $A:0 [SEP]:0",
    pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", cls_token_id),
        ("[SEP]", sep_token_id),
    ],
)
tokenizer = BertTokenizerFast(tokenizer_object=tokenizer)
tokenizer.save_pretrained(f"./tokenizer_{model}_{vocab_size}/")
