from Bio import SeqIO
from Bio.Seq import Seq
import numpy as np
import pandas as pd
import sys
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForMaskedLM

from convnet import ConvNetForMaskedLM



# TODO: should load both genome and tokenizer later, to avoid memory leak with num_workers>0
class VEPDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        variants_path=None,
        genome_path=None,
        tokenizer_path=None,
        max_length=None,
        window_size=None,
    ):
        self.variants_path = variants_path
        self.genome_path = genome_path
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length
        self.window_size = window_size

        self.variants = pd.read_parquet(self.variants_path)
        #self.variants = self.variants.head(100000)

        df_pos = self.variants.copy()
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
        ref_str = row.ref
        alt_str = row.alt

        if row.strand == "-":
            seq = seq.reverse_complement()
            window_pos = self.window_size - window_pos - 1  # TODO: check this
            ref_str = str(Seq(ref_str).reverse_complement())
            alt_str = str(Seq(alt_str).reverse_complement())
        seq = str(seq)

        assert seq[window_pos] == ref_str

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
        x["ref"] = torch.tensor(self.tokenizer.encode(ref_str, add_special_tokens=False)[0], dtype=torch.int64)
        x["alt"] = torch.tensor(self.tokenizer.encode(alt_str, add_special_tokens=False)[0], dtype=torch.int64)
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


class VEPDatasetDNABERT(torch.utils.data.Dataset):
    def __init__(
        self,
        variants_path=None,
        genome_path=None,
        tokenizer_path=None,
        max_length=None,
        window_size=None,
    ):
        self.variants_path = variants_path
        self.genome_path = genome_path
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length
        self.window_size = window_size

        self.variants = pd.read_parquet(self.variants_path)
        #self.variants = self.variants.head(100000)

        df_pos = self.variants.copy()
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
        K = 6

        assert K==6
        assert self.window_size % 2 == 0
        
        row = self.df.iloc[idx]
        seq = self.genome[row.chromosome][row.start : row.end].seq
        #window_pos = self.window_size // 2
        assert len(seq) == self.window_size
        ref_str = row.ref
        alt_str = row.alt
        kmer_pos = K // 2

        if row.strand == "-":
            seq = seq.reverse_complement()
            #window_pos = self.window_size - window_pos - 1  # TODO: check this
            ref_str = str(Seq(ref_str).reverse_complement())
            alt_str = str(Seq(alt_str).reverse_complement())
            kmer_pos = K - kmer_pos - 1

        seq = str(seq)
        seq_list = seq2kmer(seq, K).split(" ")
        window_pos = len(seq_list) // 2
        ref_kmer_str = seq_list[window_pos]
        assert ref_kmer_str[kmer_pos] == ref_str
        alt_kmer_str = list(ref_kmer_str) # copy
        alt_kmer_str[kmer_pos] = alt_str
        alt_kmer_str = "".join(alt_kmer_str)

        range_low = 2 if row.strand == "+" else 3
        range_high = 4 if row.strand == "+" else 3

        for i in range(window_pos-range_low, window_pos+range_high):  # need to do -3 and +3 I believe for negative strand
            seq_list[i] = "[MASK]"
        seq = " ".join(seq_list)

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
        x["pos"] = torch.tensor(1 + window_pos, dtype=torch.int64) # to account for CLS token
        x["ref"] = torch.tensor(self.tokenizer.encode(ref_kmer_str, add_special_tokens=False)[0], dtype=torch.int64)
        x["alt"] = torch.tensor(self.tokenizer.encode(alt_kmer_str, add_special_tokens=False)[0], dtype=torch.int64)
        return x


class MLMforVEPModel(torch.nn.Module):
    def __init__(self, model_class, model_path):
        super().__init__()
        self.model = model_class.from_pretrained(model_path)

    def forward(self, pos=None, ref=None, alt=None, **kwargs):
        logits = self.model(**kwargs).logits
        logits = logits[torch.arange(len(pos)), pos]
        logits_ref = logits[torch.arange(len(ref)), ref]
        logits_alt = logits[torch.arange(len(alt)), alt]
        llr = logits_alt - logits_ref
        return llr


model_name = sys.argv[1]

variants_path = "../../data/vep/variants/filt.parquet"
genome_path = "../../data/vep/tair10.fa"
output_path = f"vep_full_{model_name}.parquet"
output_dir = f"results_vep_full_{model_name}"  # not really used but necessary for trainer


if model_name == "window-128_tokenization-no_model-bert":
    model_path = "./results_128_bert/checkpoint-200000/"
    max_length = 128
    window_size = 128
    model_class = AutoModelForMaskedLM
    data_class = VEPDataset
    batch_size = 512
elif model_name == "window-1000_tokenization-bpe8192_model-bert":
    model_path = "./old_bpe/results/checkpoint-200000/"
    max_length = 200
    window_size = 1000
    model_class = AutoModelForMaskedLM
    data_class = VEPDataset
    batch_size = 512
elif model_name == "window-128_tokenization-no_model-convnet":
    model_path = "./results_128_cycle/checkpoint-200000/"
    max_length = 128
    window_size = 128
    model_class = ConvNetForMaskedLM
    data_class = VEPDataset
    batch_size = 512
elif model_name == "DNABERT":
    model_path = "armheb/DNA_bert_6"
    max_length = 512
    window_size = 512
    model_class = AutoModelForMaskedLM
    data_class = VEPDatasetDNABERT
    batch_size = 128


d = data_class(
    variants_path=variants_path,
    genome_path=genome_path,
    tokenizer_path=model_path,
    max_length=max_length,
    window_size=window_size,
)
#print(d[0])
#print(d[10000])
#raise Exception("debug")

model = MLMforVEPModel(model_class=model_class, model_path=model_path)

training_args = TrainingArguments(
    output_dir=output_dir, per_device_eval_batch_size=batch_size, dataloader_num_workers=0,
)

trainer = Trainer(model=model, args=training_args,)

pred = trainer.predict(test_dataset=d).predictions
print(pred.shape)

variants = d.variants
n_variants = len(variants)
pred_pos = pred[:n_variants]
pred_neg = pred[n_variants:]
avg_pred = np.stack((pred_pos, pred_neg)).mean(axis=0)

variants.loc[:, "model_llr"] = avg_pred
print(variants)
variants.to_parquet(output_path, index=False)
