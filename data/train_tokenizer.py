import numpy as np
import pandas as pd
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from transformers import BertTokenizerFast, LongformerTokenizerFast, RobertaTokenizerFast
import sys

# based on examples here:
# https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/tokenizer_training.ipynb

with open(sys.argv[1]) as file:
    #dataset = file.readlines()
    dataset = file.read().splitlines()
print(len(dataset))

n_seqs = int(sys.argv[4])
dataset = dataset[:n_seqs]
print(len(dataset))

#dataset = np.random.choice(dataset, size=len(dataset)//100, replace=False)
#print(len(dataset))

#chunk_size = 1000
#new_dataset = []
#for d in dataset:
#    new_dataset += [d[i:i + chunk_size] for i in range(0, len(d), chunk_size)]
#dataset = new_dataset
#print(len(dataset))

model = sys.argv[2]
vocab_size = int(sys.argv[3])

batch_size = 1000

def batch_iterator():
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i + batch_size]

special_tokens = ["[CLS]", "[SEP]", "[UNK]", "[PAD]", "[MASK]"]

if model == "bpe":
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()  # should not be used at all
    trainer = trainers.BpeTrainer(
        #vocab_size=vocab_size - len(special_tokens),
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=["a", "c", "g", "t"],
    )
elif model == "unigram":
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.normalizer = normalizers.Lowercase()
    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size - len(special_tokens),
        special_tokens=special_tokens,
        unk_token="[UNK]",
        initial_alphabet=["a", "c", "g", "t"],
        #max_piece_length=8,
        #shrinking_factor=0.75,
        #n_sub_iterations=2,
    )


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
#tokenizer = BertTokenizerFast(tokenizer_object=tokenizer)
tokenizer = LongformerTokenizerFast(tokenizer_object=tokenizer)
#tokenizer = RobertaTokenizerFast(tokenizer_object=tokenizer)
print(len(tokenizer))
tokenizer.save_pretrained(f"./tokenizer_{model}_{vocab_size}_{n_seqs}_v7/")
