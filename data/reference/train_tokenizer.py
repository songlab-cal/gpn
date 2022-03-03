from Bio import SeqIO
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from transformers import BertTokenizerFast

chunk_size = 4192
#chunk_size = 1000

dataset = []
for record in SeqIO.parse("tair10.contigs.fa", "fasta"):
    if len(record) < 100: continue
    for seq in (str(record.seq), str(record.seq.reverse_complement())):
        chunks = [seq[i:i+chunk_size] for i in range(0, len(seq), chunk_size)]
        dataset += chunks
print(len(dataset))
#dataset = dataset[:1000]
#print(len(dataset))

batch_size = 1000

#vocab_size = 20000
vocab_size = 32000

def batch_iterator():
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i + batch_size]


#tokenizer = Tokenizer(models.Unigram())
tokenizer = Tokenizer(models.BPE())

tokenizer.normalizer = normalizers.Lowercase()

#trainer = trainers.UnigramTrainer(vocab_size=vocab_size, special_tokens=["[CLS]", "[SEP]", "<unk>", "<pad>", "[MASK]"], unk_token="<unk>")
trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[CLS]", "[SEP]", "<unk>", "<pad>", "[MASK]"])


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
tokenizer.save_pretrained(f"./tokenizer_{vocab_size}/")

# can be loaded like this
# tokenizer = AutoTokenizer.from_pretrained("./tokenizer/")
