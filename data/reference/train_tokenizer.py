from Bio import SeqIO
import pandas as pd
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from transformers import BertTokenizerFast
import sys

# based on examples here:
# https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/tokenizer_training.ipynb

#chunk_size = 4192
chunk_size = 1000

record_lengths = []

dataset = []
for record in SeqIO.parse("tair10.contigs.fa", "fasta"):
    record_lengths.append(len(record))
    if len(record) < 100: continue
    for seq in (str(record.seq), str(record.seq.reverse_complement())):
        chunks = [seq[i:i+chunk_size] for i in range(0, len(seq), chunk_size)]
        dataset += chunks
print(len(dataset))
#dataset = dataset[:1000]
#print(len(dataset))
#
print(record_lengths)

batch_size = 1000

#vocab_size = 20000
#vocab_size = 32000
model = sys.argv[1]
vocab_size = int(sys.argv[2])

def batch_iterator():
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i + batch_size]


if model == "bpe":
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.Lowercase()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[CLS]", "[SEP]", "<unk>", "<pad>", "[MASK]"])
elif model == "unigram":
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.normalizer = normalizers.Lowercase()
    trainer = trainers.UnigramTrainer(vocab_size=vocab_size, special_tokens=["[CLS]", "[SEP]", "<unk>", "<pad>", "[MASK]"], unk_token="<unk>")


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
