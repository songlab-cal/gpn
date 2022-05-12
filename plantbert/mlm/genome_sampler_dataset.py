from Bio import SeqIO
import gzip
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import AutoTokenizer


#def insert_spaces(seq, prob):
#    new_seq = []
#    for x in seq:
#        if np.random.random() < prob:
#            new_seq += [" ", x, " "]
#        else:
#            new_seq.append(x)
#    new_seq = "".join(new_seq)
#    return new_seq


class GenomeSamplerDataset(IterableDataset):
    def __init__(
        self,
        fasta_path=None,
        tokenizer_path=None,
        window_size=None,
        max_length=None,
        random_seed=None,
        min_contig_size=None,
        #use_fast_tokenizer=None,
    ):
        super().__init__()
        self.fasta_path = fasta_path
        self.tokenizer_path = tokenizer_path
        self.window_size = window_size
        self.max_length = max_length
        self.random_seed = random_seed
        self.min_contig_size = min_contig_size
        #self.use_fast_tokenizer = use_fast_tokenizer
        #print("self.use_fast_tokenizer: ", self.use_fast_tokenizer)
        # TODO: figure out if fasta and tokenizer should be loaded and instantiated in __init__
        # on in __iter__ (for good memory/compute performance with multiple workers)
        # also some data structures are better than others (e.g. np array better than python list)

    def __iter__(self):
        print("Loading fasta.")
        with gzip.open(self.fasta_path, "rt") as handle:
            contigs = [
                contig
                for contig in SeqIO.parse(handle, "fasta")
                if len(contig) >= self.min_contig_size
            ]
        print("Done.")
        contig_sizes = np.array(
            [max(len(contig) - self.window_size, 1) for contig in contigs]
        )
        contig_probs = contig_sizes / contig_sizes.sum()
        n_contigs = len(contigs)
        print("n_contigs: ", n_contigs)
        print("contig_sizes: ", pd.Series(contig_sizes).describe())
        print("contig_probs: ", pd.Series(contig_probs).describe())

        print("Loading tokenizer.")
        #tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=self.use_fast_tokenizer)
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        print("Done.")

        seed = self.random_seed
        worker_info = get_worker_info()
        if worker_info is not None:
            seed = seed * (worker_info.id + 1)
        rs = np.random.RandomState(seed=seed)
        print("worker_info: ", worker_info, " seed: ", seed)

        while True:
            contig_index = rs.choice(n_contigs, p=contig_probs)
            contig = contigs[contig_index]
            if len(contig) > self.window_size:
                start = rs.randint(len(contig) - self.window_size)
            else:
                start = 0
            end = start + self.window_size
            seq = contig[start:end].seq
            strand = rs.choice(["+", "-"])
            if strand == "-":
                seq = seq.reverse_complement()
            seq = str(seq)
            #x = tokenizer(
            #    seq,
            #    padding="max_length",
            #    max_length=self.max_length,
            #    return_token_type_ids=False,
            #    return_tensors="pt",
            #    truncation=True,
            #)
            #x["input_ids"] = x["input_ids"].flatten()
            #x["attention_mask"] = x["attention_mask"].flatten()

            #seq = insert_spaces(seq, 0.02)

            x = tokenizer(
                seq,
                return_token_type_ids=False,
                return_attention_mask=False,
                return_tensors="pt",
            )
            x["input_ids"] = x["input_ids"].flatten()

            # x["global_attention_mask"] = torch.zeros_like(x["input_ids"])
            # x["global_attention_mask"][0] = 1
            yield x


# d = GenomeSamplerDataset(fasta_path="./all.contigs.fa.gz", tokenizer_path="./tokenizer_unigram_1019_v2/", window_size=1000, max_length=280, random_seed=42, min_contig_size=500)
# dl = DataLoader(d, batch_size=4, num_workers=0)
# i = 0
# for x in dl:
#    print(x["attention_mask"].sum(dim=1))
#    raise Exception("debug")
#    i += 1
#    if i > 3: break
