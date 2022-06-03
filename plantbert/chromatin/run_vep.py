from Bio import SeqIO
from Bio.Seq import Seq
import numpy as np
import pandas as pd
import sys
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments

from plantbert.chromatin.data import encode_dna_seq, seq2kmer
from plantbert.chromatin.model import PlantBertModel, DeepSEAModel, DNABERTModel


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

        df_ref_pos = self.variants.copy()
        df_ref_pos["start"] = df_ref_pos.pos - self.window_size // 2
        df_ref_pos["end"] = df_ref_pos.start + self.window_size
        df_ref_pos["strand"] = "+"
        df_ref_pos["status"] = "ref"
        df_ref_neg = df_ref_pos.copy()
        df_ref_neg.strand = "-"
        df_alt_pos = df_ref_pos.copy()
        df_alt_pos.status = "alt"
        df_alt_neg = df_alt_pos.copy()
        df_alt_neg.strand = "-"
        self.df = pd.concat(
            [df_ref_pos, df_ref_neg, df_alt_pos, df_alt_neg], ignore_index=True
        )
        # TODO: might consider interleaving this so the first 4 rows correspond to first variant, etc.
        # can sort_values to accomplish that, I guess
        self.genome = SeqIO.to_dict(SeqIO.parse(self.genome_path, "fasta"))

        if self.tokenizer_path is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = self.genome[row.chromosome][row.start : row.end].seq
        assert len(seq) == self.window_size
        assert seq[self.window_size // 2] == row.ref

        if row.status == "alt":
            seq_list = list(str(seq))
            seq_list[self.window_size // 2] = row.alt
            seq = Seq("".join(seq_list))

        if row.strand == "-":
            seq = seq.reverse_complement()
        seq = str(seq)

        x = self.tokenize_seq(seq)
        return x


class PlantBertVEPDataset(VEPDataset):
    def tokenize_seq(self, seq):
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
        return x


class DeepSEAVEPDataset(VEPDataset):
    def tokenize_seq(self, seq):
        input_ids = torch.tensor(encode_dna_seq(seq).astype(int), dtype=torch.int64)
        return {"input_ids": input_ids}


class DNABERTVEPDataset(VEPDataset):
    def tokenize_seq(self, seq):
        x = seq2kmer(seq, 6)
        x = self.tokenizer(
            x,
            padding="max_length",
            max_length=1000,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        x["input_ids"] = x["input_ids"].flatten()
        x["attention_mask"] = x["attention_mask"].flatten()
        return x


model_type = sys.argv[1]


variants_path = "../../data/vep/variants/filt.parquet"
genome_path = "../../data/vep/tair10.fa"
max_length = 200
window_size = 1000
output_path = f"vep_full_{model_type}.parquet"
output_dir = f"results_vep_full_{model_type}"  # not really used but necessary for trainer


if model_type == "DeepSEA":
    data_class = DeepSEAVEPDataset
    model_class = DeepSEAModel
    model_ckpt = "DeepSEA/checkpoints/epoch=25-step=86631.ckpt"
    #model_ckpt = "lightning_logs/version_18/checkpoints/epoch=9-step=33319.ckpt"
    tokenizer_path = None
    per_device_eval_batch_size = 2048
elif model_type =="DNABERT":
    data_class = DNABERTVEPDataset
    model_class = DNABERTModel
    model_ckpt = "DNABERT/checkpoints/epoch=7-step=27079.ckpt"
    tokenizer_path = "armheb/DNA_bert_6"
    per_device_eval_batch_size = 64
elif model_type =="PlantBert":
    data_class = PlantBertVEPDataset
    model_class = PlantBertModel
    #model_ckpt = "version_6/checkpoints/epoch=9-step=33449.ckpt"
    #model_ckpt = "lightning_logs/version_33/checkpoints/epoch=7-step=26655.ckpt"
    model_ckpt = "lightning_logs/version_2znci1qx/epoch_6-step_46648.ckpt"  # bpe dropout version
    tokenizer_path = "../mlm/old_bpe/results/checkpoint-200000/"
    per_device_eval_batch_size = 512
elif model_type =="ConvNet":
    data_class = PlantBertVEPDataset
    model_class = PlantBertModel
    #model_ckpt = "lightning_logs/version_3kimm4yz/epoch_6-step_23324.ckpt"
    #tokenizer_path = "../mlm/results_128_cycle/checkpoint-200000/"
    model_ckpt = "lightning_logs/version_148q8xs0/epoch_5-step_19992.ckpt"
    tokenizer_path = "../mlm/results_512_convnet/checkpoint-400000/"
    max_length = 1000
    per_device_eval_batch_size = 512


d = data_class(
    variants_path=variants_path,
    genome_path=genome_path,
    tokenizer_path=tokenizer_path,
    max_length=max_length,
    window_size=window_size,
)

if model_type == "PlantBert":
    model = model_class.load_from_checkpoint(
        model_ckpt, language_model_path=tokenizer_path
    )
else:
    model = model_class.load_from_checkpoint(model_ckpt,)

training_args = TrainingArguments(
    output_dir=output_dir, per_device_eval_batch_size=per_device_eval_batch_size, dataloader_num_workers=0,
)

trainer = Trainer(model=model, args=training_args,)

pred = trainer.predict(test_dataset=d).predictions
print(pred.shape)

n_variants = len(d.variants)
pred_ref_pos = pred[0 * n_variants : 1 * n_variants]
pred_ref_neg = pred[1 * n_variants : 2 * n_variants]
pred_alt_pos = pred[2 * n_variants : 3 * n_variants]
pred_alt_neg = pred[3 * n_variants : 4 * n_variants]

pred_ref = np.stack((pred_ref_pos, pred_ref_neg)).mean(axis=0)
pred_alt = np.stack((pred_alt_pos, pred_alt_neg)).mean(axis=0)
#delta_pred = pred_alt - pred_ref

variants = d.variants
#variants.loc[:, model.feature_names] = delta_pred
variants.loc[:, [f"model_pred_ref_{f}" for f in model.feature_names]] = pred_ref
variants.loc[:, [f"model_pred_alt_{f}" for f in model.feature_names]] = pred_alt
print(variants)
variants.to_parquet(output_path, index=False)
