import gzip
import multiprocessing as mp

import numpy as np
import pandas as pd
import zarr
from Bio import SeqIO
from Bio.Seq import Seq
from datasets import Dataset, load_dataset
from joblib import Parallel, delayed
from tqdm import tqdm


def load_fasta(path, subset_chroms=None):
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as handle:
        genome = pd.Series(
            {
                rec.id: str(rec.seq)
                for rec in SeqIO.parse(handle, "fasta")
                if subset_chroms is None or rec.id in subset_chroms
            }
        )
    return genome


# Some standard formats
def load_table(path):
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif "csv" in path:
        df = pd.read_csv(path)
    elif "tsv" in path:
        df = pd.read_csv(path, sep="\t")
    elif "vcf" in path:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            comment="#",
            usecols=[0, 1, 3, 4],
            dtype={0: str},
        ).rename(columns={0: "chrom", 1: "pos", 3: "ref", 4: "alt"})
    elif "gtf" in path or "gff" in path:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            comment="#",
            dtype={"chrom": str},
            names=[
                "chrom",
                "source",
                "feature",
                "start",
                "end",
                "score",
                "strand",
                "frame",
                "attribute",
            ],
        )
        df.start -= 1
    df.chrom = df.chrom.astype(str)
    return df


class Genome:
    def __init__(self, path, subset_chroms=None):
        self._genome = load_fasta(path, subset_chroms=subset_chroms)
        self.chrom_sizes = {chrom: len(seq) for chrom, seq in self._genome.items()}

    def get_seq(self, chrom, start, end, strand="+"):
        chrom_size = self.chrom_sizes[chrom]
        seq = self._genome[chrom][max(start, 0) : min(end, chrom_size)]

        if start < 0:
            seq = "N" * (-start) + seq  # left padding
        if end > chrom_size:
            seq = seq + "N" * (end - chrom_size)  # right padding

        if strand == "-":
            seq = str(Seq(seq).reverse_complement())
        return seq

    def get_seq_fwd_rev(self, chrom, start, end):
        seq_fwd = self.get_seq(chrom, start, end)
        seq_rev = str(Seq(seq_fwd).reverse_complement())
        return seq_fwd, seq_rev


def load_dataset_from_file_or_dir(
    path,
    split="test",
    is_file=False,
    **kwargs,
):
    if is_file:
        return Dataset.from_pandas(load_table(path))
    else:
        return load_dataset(path, split=split, **kwargs)


def token_input_id(token, tokenizer, n_prefix=0):
    return tokenizer(token)["input_ids"][n_prefix]


def _get_msa(i, chroms, starts, ends, strands, obj, kwargs):
    return obj.get_msa(chroms[i], starts[i], ends[i], strand=strands[i], **kwargs)


def _get_msa_fwd_rev(i, chroms, starts, ends, obj, kwargs):
    return obj.get_msa_fwd_rev(chroms[i], starts[i], ends[i], **kwargs)


def _run_vep(i, chroms, poss, refs, alts, obj, kwargs):
    return obj.run_vep(chroms[i], poss[i], refs[i], alts[i], **kwargs)


class GenomeMSA(object):
    def __init__(self, path, subset_chroms=None, in_memory=False):
        self.reverse_complementer = ReverseComplementer()
        self.tokenizer = Tokenizer()

        print("Loading MSA...")
        self.f = zarr.open(path, mode="r")
        chroms = self.f.keys()
        if subset_chroms is not None:
            chroms = [chrom for chrom in chroms if chrom in subset_chroms]
        if in_memory:
            self.data = pd.Series({chrom: self.f[chrom][:] for chrom in tqdm(chroms)})
            # self.f.close()
        else:
            # pd.Series does not work with h5py/zarr object
            # (attempts to load all data into memory)
            # beware: dict has issues with parallelism in Pytorch
            self.data = {chrom: self.f[chrom] for chrom in chroms}
        print("Loading MSA... Done")

    def get_msa(self, chrom, start, end, strand="+", tokenize=False):
        msa = self.data[chrom][start:end].view("S1")
        if strand == "-":
            msa = self.reverse_complementer(msa, position_axis=0)
        if tokenize:
            msa = self.tokenizer(msa)
        return msa

    def get_msa_fwd_rev(self, chrom, start, end, tokenize=False):
        msa_fwd = self.get_msa(chrom, start, end)
        msa_rev = self.reverse_complementer(msa_fwd, position_axis=0)
        if tokenize:
            msa_fwd = self.tokenizer(msa_fwd)
            msa_rev = self.tokenizer(msa_rev)
        return msa_fwd, msa_rev

    def get_msa_batch(
        self, chroms, starts, ends, strands, backend=None, n_jobs=None, **kwargs
    ):
        if backend == "multiprocessing":
            with mp.Pool(processes=n_jobs) as pool:
                msa_batch = pool.starmap(
                    _get_msa,
                    [
                        (i, chroms, starts, ends, strands, self, kwargs)
                        for i in range(len(chroms))
                    ],
                )
        elif backend == "joblib":
            msa_batch = Parallel(n_jobs=n_jobs)(
                delayed(_get_msa)(i, chroms, starts, ends, strands, self, kwargs)
                for i in range(len(chroms))
            )
        elif backend is None:
            msa_batch = [
                _get_msa(i, chroms, starts, ends, strands, self, kwargs)
                for i in range(len(chroms))
            ]
        msa_batch = np.array(msa_batch)
        return msa_batch

    def get_msa_batch_fwd_rev(
        self, chroms, starts, ends, backend=None, n_jobs=None, **kwargs
    ):
        if backend == "multiprocessing":
            with mp.Pool(processes=n_jobs) as pool:
                msa_batch_fwd, msa_batch_rev = zip(
                    *pool.starmap(
                        _get_msa_fwd_rev,
                        [
                            (i, chroms, starts, ends, self, kwargs)
                            for i in range(len(chroms))
                        ],
                    )
                )
        elif backend is None:
            msa_batch_fwd, msa_batch_rev = zip(
                *[
                    _get_msa_fwd_rev(i, chroms, starts, ends, self, kwargs)
                    for i in range(len(chroms))
                ]
            )
        msa_batch_fwd = np.array(msa_batch_fwd)
        msa_batch_rev = np.array(msa_batch_rev)
        return msa_batch_fwd, msa_batch_rev

    def run_vep(self, chrom, pos, ref, alt, pseudocounts=1):
        msa = np.char.upper(self.data[chrom][pos - 1].view("S1"))
        assert msa[0] == ref.encode("ascii"), f"{ref=} does not match {msa[0]=}"
        msa = msa[1:]  # exclude target species
        ref_count = (msa == ref.encode("ascii")).sum() + pseudocounts
        alt_count = (msa == alt.encode("ascii")).sum() + pseudocounts
        ref_prob = ref_count / (ref_count + alt_count)
        alt_prob = alt_count / (ref_count + alt_count)
        return np.log(alt_prob) - np.log(ref_prob)

    def run_vep_batch(
        self, chroms, poss, refs, alts, backend=None, n_jobs=None, **kwargs
    ):
        if backend == "multiprocessing":
            with mp.Pool(processes=n_jobs) as pool:
                vep_batch = pool.starmap(
                    _run_vep,
                    [
                        (i, chroms, poss, refs, alts, self, kwargs)
                        for i in range(len(chroms))
                    ],
                )
        elif backend == "joblib":
            vep_batch = Parallel(n_jobs=n_jobs)(
                delayed(_run_vep)(i, chroms, poss, refs, alts, self, kwargs)
                for i in tqdm(range(len(chroms)))
            )
        elif backend is None:
            vep_batch = [
                _run_vep(i, chroms, poss, refs, alts, self, kwargs)
                for i in tqdm(range(len(chroms)))
            ]
        return np.array(vep_batch)


# Utilities for processing DNA sequences represented as np byte arrays, e.g.
# np.array([b'A', b'C', b'G', b'T', b'N'], dtype='|S1')
# with any number of axes (e.g. batch, species, position)


class Tokenizer(object):
    def __init__(self, vocab="-ACGT?"):
        # -: gap/unknown/pad (simple for now, could split in the future)
        # ?: mask
        unk = vocab.index("-")
        self.table = np.full((256,), unk, dtype=np.uint8)
        for i, c in enumerate(vocab):
            self.table[ord(c)] = i
        self.vocab = vocab
        self.mask_token = "?"
        self.pad_token = "-"

    def __call__(self, x):
        return self.table[np.char.upper(x).view(np.uint8)]

    def __len__(self):
        return len(self.vocab)

    def mask_token_id(self):
        return self.vocab.index("?")

    def unk_token_id(self):
        return self.vocab.index("-")

    def pad_token_id(self):
        return self.vocab.index("-")

    def nucleotide_token_id_start(self):
        return self.vocab.index("A")

    def nucleotide_token_id_end(self):
        return self.vocab.index("T") + 1


class ReverseComplementer(object):
    def __init__(self):
        # Define the complement mapping.
        complement_mapping = {
            b"A": b"T",
            b"T": b"A",
            b"C": b"G",
            b"G": b"C",
            b"a": b"t",
            b"t": b"a",
            b"c": b"g",
            b"g": b"c",
        }

        # Create a translation table that maps each byte to its complement.
        # If a byte does not represent a recognized character, it maps to itself.
        self.table = np.array(
            [
                complement_mapping.get(chr(i).encode(), chr(i).encode())
                for i in range(256)
            ],
            dtype="|S1",
        )

    def __call__(self, x, position_axis=-1):
        # Reverse the sequence and apply the complement rule.
        return self.table[np.flip(x, axis=position_axis).view(np.uint8)]
