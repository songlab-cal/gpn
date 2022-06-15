from Bio import SeqIO
from Bio.Seq import Seq
import numpy as np
import pandas as pd
import sys
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForMaskedLM

from convnet import ConvNetForMaskedLM



# TODO: should load both genome and tokenizer later, to avoid memory leak with num_workers>0
class LogitsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        examples_path=None,
        genome_path=None,
        tokenizer_path=None,
        max_length=None,
        window_size=None,
    ):
        self.examples_path = examples_path
        self.genome_path = genome_path
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length
        self.window_size = window_size

        self.examples = pd.read_csv(self.examples_path)
        #self.examples = self.examples.head(100)

        df_pos = self.examples.copy()
        df_pos["start"] = df_pos.pos - self.window_size // 2
        df_pos["end"] = df_pos.start + self.window_size
        df_pos["strand"] = "+"
        df_neg = df_pos.copy()
        df_neg.strand = "-"

        self.df = pd.concat([df_pos, df_neg], ignore_index=True)
        # TODO: might consider interleaving this so the first 4 rows correspond to first variant, etc.
        # can sort_values to accomplish that, I guess
        self.genome = SeqIO.to_dict(SeqIO.parse(self.genome_path, "fasta"))

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = self.genome[row.chromosome][row.start : row.end].seq
        window_pos = self.window_size // 2
        assert len(seq) == self.window_size

        if row.strand == "-":
            seq = seq.reverse_complement()
            window_pos = self.window_size - window_pos - 1  # TODO: check this
        seq = str(seq)

        seq_list = list(seq)
        seq_list[window_pos] = "[MASK]"
        seq = "".join(seq_list)

        x = self.tokenizer(
            seq,
            padding="max_length",
            max_length=self.max_length,
            return_token_type_ids=False,
            return_tensors="pt",
            truncation=True,
        )
        x["input_ids"] = x["input_ids"].flatten()
        x["attention_mask"] = x["attention_mask"].flatten()
        mask_token_id = self.tokenizer.convert_tokens_to_ids("[MASK]")
        x["pos"] = torch.where(x["input_ids"] == mask_token_id)[0][0]
        return x


def seq2kmer(seq, k):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space
    """
    kmer = [seq[x : x + k] for x in range(len(seq) + 1 - k)]
    kmers = " ".join(kmer)
    return kmers


class MLMforLogitsModel(torch.nn.Module):
    def __init__(self, model_class, model_path):
        super().__init__()
        self.model = model_class.from_pretrained(model_path)

    def forward(self, pos=None, **kwargs):
        logits = self.model(**kwargs).logits
        logits = logits[torch.arange(len(pos)), pos]
        return logits


model_name = sys.argv[1]

examples_path = "./perplexity_examples.tsv.gz"
genome_path = "../../data/vep/tair10.fa"
output_path = f"Logits_examples_{model_name}.parquet"
output_dir = f"results_Logits_examples_{model_name}"  # not really used but necessary for trainer


if model_name == "window-128_tokenization-no_model-bert":
    model_path = "./results_128_bert/checkpoint-200000/"
    max_length = 128
    window_size = 128
    model_class = AutoModelForMaskedLM
    data_class = LogitsDataset
    batch_size = 512
elif model_name == "window-1000_tokenization-bpe8192_model-bert":
    model_path = "./old_bpe/results/checkpoint-200000/"
    max_length = 200
    window_size = 1000
    model_class = AutoModelForMaskedLM
    data_class = LogitsDataset
    batch_size = 512
elif model_name == "window-128_tokenization-no_model-convnet":
    model_path = "./results_128_cycle/checkpoint-200000/"
    max_length = 128
    window_size = 128
    model_class = ConvNetForMaskedLM
    data_class = LogitsDataset
    batch_size = 512
elif model_name == "window-512_tokenization-no_model-convnet":
    model_path = "./results_512_convnet/checkpoint-400000/"
    max_length = 512
    window_size = 512
    model_class = ConvNetForMaskedLM
    data_class = LogitsDataset
    batch_size = 128
elif model_name == "window-512_tokenization-no_model-convnet800k":
    model_path = "./results_512_convnet/checkpoint-800000/"
    max_length = 512
    window_size = 512
    model_class = ConvNetForMaskedLM
    data_class = LogitsDataset
    batch_size = 128
elif model_name == "window-512_tokenization-no_model-convnet200k":
    model_path = "./results_512_convnet/checkpoint-200000/"
    max_length = 512
    window_size = 512
    model_class = ConvNetForMaskedLM
    data_class = LogitsDataset
    batch_size = 128
elif model_name == "window-512_tokenization-no_model-convnet800kfinetune150k":
    model_path = "./results_512_convnet_finetuning_v2/checkpoint-950000/"
    max_length = 512
    window_size = 512
    model_class = ConvNetForMaskedLM
    data_class = LogitsDataset
    batch_size = 128
elif model_name == "window-512_tokenization-no_model-convnet800kfinetune200k":
    model_path = "./results_512_convnet_finetuning_v2/checkpoint-1000000/"
    max_length = 512
    window_size = 512
    model_class = ConvNetForMaskedLM
    data_class = LogitsDataset
    batch_size = 128
elif model_name == "DNABERT":
    model_path = "armheb/DNA_bert_6"
    max_length = 512
    window_size = 512
    model_class = AutoModelForMaskedLM
    data_class = LogitsDatasetDNABERT
    batch_size = 128


d = data_class(
    examples_path=examples_path,
    genome_path=genome_path,
    tokenizer_path=model_path,
    max_length=max_length,
    window_size=window_size,
)
print(d.tokenizer.get_vocab())
#print(d[0])
#print(d[10000])
#raise Exception("debug")

model = MLMforLogitsModel(model_class=model_class, model_path=model_path)

training_args = TrainingArguments(
    output_dir=output_dir, per_device_eval_batch_size=batch_size, dataloader_num_workers=0,
)

trainer = Trainer(model=model, args=training_args,)

pred = trainer.predict(test_dataset=d).predictions
print(pred.shape)

examples = d.examples
n_examples = len(examples)
pred_pos = pred[:n_examples, 2:]
pred_neg = pred[n_examples:, 2:]
pred_neg = pred_neg[:, [3, 2, 1, 0]]  # complement
avg_pred = np.stack((pred_pos, pred_neg)).mean(axis=0)

examples.loc[:, ["a", "c", "g", "t"]] = avg_pred
print(examples)
examples.to_parquet(output_path, index=False)
