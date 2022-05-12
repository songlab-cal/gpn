import numpy as np
import pandas as pd
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from transformers import PreTrainedTokenizerFast, AlbertTokenizer
import sentencepiece as spm
import sys

# based on examples here:
# https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/tokenizer_training.ipynb

with open(sys.argv[1]) as file:
    dataset = file.read().splitlines()
print(len(dataset))

n_seqs = int(sys.argv[4])
dataset = dataset[:n_seqs]
print(len(dataset))


model = sys.argv[2]
vocab_size = int(sys.argv[3])
output_path = f"./tokenizer_{model}_{vocab_size}_v10"

batch_size = 1000

def batch_iterator():
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i + batch_size]

special_tokens = ["[MASK]", "[PAD]", "[UNK]"]
initial_alphabet = ["a", "c", "g", "t"]

if model == "bpe":
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()  # should not be used at all
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=initial_alphabet,
    )
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(dataset))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, mask_token="[MASK]", pad_token="[PAD]", unk_token="[UNK]")
elif model == "unigram": 
    spm.SentencePieceTrainer.train(
        input=sys.argv[1],
        model_prefix=output_path,
        vocab_size=vocab_size,
        num_threads=32,
        seed_sentencepiece_size=50000,
        add_dummy_prefix=False,
        bos_piece="[CLS]",
        bos_id=0,
        eos_piece="[SEP]",
        eos_id=1,
        unk_piece="[UNK]",
        unk_id=2,
        pad_piece="[PAD]",
        pad_id=3,
        user_defined_symbols="[MASK]",
    )

    #sp_model_kwargs = dict(enable_sampling=True, nbest_size=-1, alpha=1.5)
    sp_model_kwargs = dict(enable_sampling=False)
    tokenizer = AlbertTokenizer(
        vocab_file=output_path + ".model",
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="[UNK]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        extra_ids=0,
        sp_model_kwargs=sp_model_kwargs,
        do_lower_case=True,
    )

print(len(tokenizer))
tokenizer.save_pretrained(output_path)