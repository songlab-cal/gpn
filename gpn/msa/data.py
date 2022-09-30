from Bio.AlignIO.MafIO import MafIndex
from collections.abc import Mapping
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from tqdm import tqdm
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union


tqdm.pandas()


def find_pos_incl_gaps(seq, pos):
    seen = -1
    for i, x in enumerate(seq):
        if x != "-":
            seen += 1
        if seen == pos:
            return i
    raise Exception("find_pos_incl_gaps off-limits")


class GenomeMafIndex:
    def __init__(self, path=None, chroms=None, species_path=None):
        self.species = pd.read_csv(species_path, header=None).values.astype(str).ravel()
        print(self.species)
        self.target_species = self.species[0]
        self.maf_indices = {
            chrom: MafIndex(
                Path(path) / f"{chrom}.mafindex",
                Path(path) / f"{chrom}.maf",
                f"{self.target_species}.chr{chrom}",  # be careful about the chr prefix
            )
            for chrom in chroms
        }

    def access(self, chrom, start, end, strand, max_length=None):
        maf_index = self.maf_indices[chrom]
        if strand == "+":
            strand_id = 1
        elif strand == "-":
            strand_id = -1
        else:
            raise Exception(f"Strand {strand} not supported.")

        alignment = maf_index.get_spliced([start], [end], strand=strand_id)
        all_species = [record.id.split(".")[0] for record in alignment]

        if max_length is not None:
            seq = str(alignment[all_species.index(self.target_species)].seq)
            # print("seq: ", seq)
            if strand == "+":
                target_pos = max_length // 2
            else:
                target_pos = max_length // 2 - 1
            center = find_pos_incl_gaps(seq, target_pos)
            if strand == "+":
                left = center - max_length // 2
                right = center + max_length // 2
            else:
                left = center - max_length // 2 + 1
                right = center + max_length // 2 + 1
            alignment = alignment[:, left:right]
            # print(target_pos, center, left, right)

        return [
            str(alignment[all_species.index(s)].seq)
            if s in all_species
            else "-" * max_length
            for s in self.species
        ]


class GenomeMSASamplerDataset(IterableDataset):
    def __init__(
        self,
        intervals_path=None,
        data_path=None,
        tokenizer_path=None,
        window_size=None,
        random_seed=None,
        species_path=None,
    ):
        super().__init__()
        self.intervals_path = intervals_path
        self.data_path = data_path
        self.tokenizer_path = tokenizer_path
        self.window_size = window_size
        self.random_seed = random_seed
        self.species_path = species_path

        print("Loading intervals...")
        self.contigs = pd.read_csv(self.intervals_path, sep="\t")
        self.contigs["contig_len"] = self.contigs.end - self.contigs.start
        print(self.contigs.shape)
        self.contigs = self.contigs[self.contigs.contig_len >= self.window_size]
        print(self.contigs.shape)
        if not "contig_weight" in self.contigs.columns:
            print("Setting contig weights according to lengths.")
            self.contigs["contig_weight"] = (
                1 + self.contigs.contig_len - self.window_size
            ).clip(lower=1)
        else:
            print("Using predefined contig weights.")
        self.contigs["contig_prob"] = (
            self.contigs.contig_weight / self.contigs.contig_weight.sum()
        )
        print(self.contigs[["contig_len", "contig_weight", "contig_prob"]])
        print("Done.")

    def __iter__(self):
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        print("Done.")

        print("Loading MAF...")
        self.maf_index = GenomeMafIndex(
            path=self.data_path,
            chroms=self.contigs.chrom.unique(),
            species_path=self.species_path,
        )
        print("Done.")

        seed = self.random_seed
        worker_info = get_worker_info()
        if worker_info is not None:
            seed = seed * (worker_info.id + 1)
        rs = np.random.RandomState(seed=seed)

        while True:
            contig_index = rs.choice(
                len(self.contigs), p=self.contigs.contig_prob.values
            )
            contig = self.contigs.iloc[contig_index]
            if contig.contig_len > self.window_size:
                start = rs.randint(contig.contig_len - self.window_size)
            else:
                start = 0
            end = start + self.window_size
            strand = rs.choice(["+", "-"])
            seqs = self.maf_index.access(
                contig.chrom,
                contig.start + start,
                contig.start + end,
                strand,
                max_length=self.window_size,
            )
            # print(worker_info, seqs)
            x = tokenizer(
                seqs,
                return_token_type_ids=False,
                return_attention_mask=False,
                return_tensors="pt",
            )
            # x["input_ids"] = x["input_ids"].flatten()
            # x["special_tokens_mask"] = torch.tensor(np.char.islower(np.array(list(seq))))
            yield x


class GenomeMSAFixedDataset(Dataset):
    def __init__(
        self,
        intervals_path=None,
        data_path=None,
        tokenizer_path=None,
        window_size=None,
        step_size=None,
        species_path=None,
    ):
        super().__init__()
        self.intervals_path = intervals_path
        self.data_path = data_path
        self.tokenizer_path = tokenizer_path
        self.window_size = window_size
        self.step_size = step_size
        self.species_path = species_path

        print("Loading intervals...")
        contigs = pd.read_csv(self.intervals_path, sep="\t")
        print("Done.")

        def get_contig_windows(contig):
            windows = pd.DataFrame(
                dict(start=np.arange(contig.start, contig.end - window_size, step_size))
            )
            windows["end"] = windows.start + window_size
            windows["chrom"] = contig.chrom
            windows["strand"] = "+"
            windows_neg = windows.copy()
            windows_neg.strand = "-"
            windows = pd.concat([windows, windows_neg], ignore_index=True)
            return windows

        print("Creating windows...")
        self.windows = pd.concat(
            contigs.progress_apply(get_contig_windows, axis=1).values, ignore_index=True
        ).sample(frac=1.0, random_state=42)
        print(self.windows)
        print("Done.")

    def load_tokenizer(self):
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        print("Done.")

    def load_maf(self):
        print("Loading MAF...")
        self.maf_index = GenomeMafIndex(
            path=self.data_path,
            chroms=self.windows.chrom.unique(),
            species_path=self.species_path,
        )
        print("Done.")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        if not hasattr(self, "tokenizer"):
            self.load_tokenizer()
            self.load_maf()

        window = self.windows.iloc[idx]
        seqs = self.maf_index.access(
            window.chrom,
            window.start,
            window.end,
            window.strand,
            max_length=self.window_size,
        )
        # print(window, seqs)
        x = self.tokenizer(
            seqs,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_tensors="pt",
        )
        # x["input_ids"] = x["input_ids"].flatten()
        # x["special_tokens_mask"] = torch.tensor(np.char.islower(np.array(list(seq))))
        return x


class DataCollatorForLanguageModelingMSA(DataCollatorForLanguageModeling):
    def torch_mask_tokens(
        self, inputs: Any, special_tokens_mask: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        # print(inputs.shape)  # b r c
        # raise Exception("debug")
        labels = inputs.clone()

        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        # if special_tokens_mask is None:
        #    special_tokens_mask = [
        #        self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        #    ]
        #    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        # else:
        #    special_tokens_mask = special_tokens_mask.bool()

        # probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # print(labels)
        # raise Exception("debug")

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


# d = GenomeMSASamplerDataset(
#    intervals_path="intervals/Homo_sapiens.train.tsv.gz",
#    data_path=".",
#    tokenizer_path="tokenizer",
#    window_size=32,
#    random_seed=42,
# )
# i = 0
# for x in d:
#    print(x)
#    i += 1
#    if i > 100: break

# d = GenomeMSAFixedDataset(
#    intervals_path="intervals/genome.test.tsv.gz",
#    data_path=".",
#    tokenizer_path="tokenizer",
#    window_size=64,
#    step_size=32,
# )
# print(d[0])

# i = 0
# dl = DataLoader(d, batch_size=4, num_workers=2)
# for batch in dl:
#    if i % 100 == 0: print(i)
#    #print(batch)
#    i += 1
#    if i > 10000: break
