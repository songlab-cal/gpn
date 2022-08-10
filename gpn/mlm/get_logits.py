from Bio import SeqIO
from Bio.Seq import Seq
import numpy as np
import pandas as pd
import sys
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForMaskedLM

import gpn.mlm


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

        self.examples = pd.read_parquet(self.examples_path)
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


class MLMforLogitsModel(torch.nn.Module):
    def __init__(self, model_class, model_path):
        super().__init__()
        self.model = model_class.from_pretrained(model_path)

    def forward(self, pos=None, **kwargs):
        logits = self.model(**kwargs).logits
        logits = logits[torch.arange(len(pos)), pos]
        return logits


model_ckpt = sys.argv[1]  # e.g. "./results_512_convnet_finetuning_v2/checkpoint-1000000/"
input_path = sys.argv[2]  
output_path = sys.argv[3]  # parquet file

genome_path = "../../data/vep/tair10.fa"
output_dir = f"results_{output_path}_dir"  # not really used but necessary for trainer

max_length = 512
window_size = 512
model_class = AutoModelForMaskedLM
data_class = LogitsDataset
batch_size = 128

d = data_class(
    examples_path=input_path,
    genome_path=genome_path,
    tokenizer_path=model_ckpt,
    max_length=max_length,
    window_size=window_size,
)
print(d.tokenizer.get_vocab())
vocab = d.tokenizer.get_vocab()
id_a = vocab["a"]
id_c = vocab["c"]
id_g = vocab["g"]
id_t = vocab["t"]
print(id_a, id_c, id_g, id_t)

model = MLMforLogitsModel(model_class=model_class, model_path=model_ckpt)

training_args = TrainingArguments(
    output_dir=output_dir, per_device_eval_batch_size=batch_size, dataloader_num_workers=0,
)

trainer = Trainer(model=model, args=training_args,)

pred = trainer.predict(test_dataset=d).predictions
print(pred.shape)

examples = d.examples
n_examples = len(examples)
pred_pos = pred[:n_examples, [id_a, id_c, id_g, id_t]]
pred_neg = pred[n_examples:, [id_t, id_g, id_c, id_a]]
avg_pred = np.stack((pred_pos, pred_neg)).mean(axis=0)

examples.loc[:, ["A", "C", "G", "T"]] = avg_pred
print(examples)
examples.to_parquet(output_path, index=False)
