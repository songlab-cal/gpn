from Bio import SeqIO
from Bio.Seq import Seq
#import gzip
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

        print("Loading parquet.")
        self.contigs = pd.read_parquet(self.fasta_path)
        print(self.contigs.shape)
        self.contigs = self.contigs[self.contigs.contig_len >= self.min_contig_size]
        print(self.contigs.shape)
        if not "contig_weight" in self.contigs.columns:
            print("Setting contig weights according to lengths.")
            self.contigs["contig_weight"] = (1 + self.contigs.contig_len - self.window_size).clip(lower=1)
        else:
            print("Using predefined contig weights.")
        self.contigs["contig_prob"] = self.contigs.contig_weight / self.contigs.contig_weight.sum()
        print(self.contigs[["contig_len", "contig_weight", "contig_prob"]])
        print("Done.")


    def __iter__(self):


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
            contig_index = rs.choice(len(self.contigs), p=self.contigs.contig_prob.values)
            contig = self.contigs.iloc[contig_index]
            if contig.contig_len > self.window_size:
                start = rs.randint(contig.contig_len - self.window_size)
            else:
                start = 0
            end = start + self.window_size
            seq = contig.seq[start:end]
            strand = rs.choice(["+", "-"])
            if strand == "-":
                seq = str(Seq(seq).reverse_complement())
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

            x = tokenizer(
                seq,
                return_token_type_ids=False,
                return_attention_mask=False,
                return_tensors="pt",
            )
            x["input_ids"] = x["input_ids"].flatten()
            x["special_tokens_mask"] = torch.tensor(np.char.islower(np.array(list(seq))))
            #print(seq, x["special_tokens_mask"])
            #raise Exception("debug")
            #if "species_id" in contig:
            #    x["species_id"] = torch.tensor(contig.species_id, dtype=torch.int64)

            # x["global_attention_mask"] = torch.zeros_like(x["input_ids"])
            # x["global_attention_mask"][0] = 1
            yield x


#d = GenomeSamplerDataset(fasta_path="../../data/mlm/genomes/all.contigs.parquet", tokenizer_path="../../data/mlm/tokenizer_bare/", window_size=512, random_seed=42, min_contig_size=512)
#dl = DataLoader(d, batch_size=4, num_workers=0)
#i = 0
#for x in dl:
#    print(x)
#    i += 1
#    if i > 3: break
