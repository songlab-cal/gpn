from Bio import SeqIO, bgzf
from Bio.Seq import Seq
import bioframe as bf
from datasets import load_dataset, Dataset
import gzip
from joblib import Parallel, delayed
import multiprocessing as mp
import numpy as np
import pandas as pd
import pyBigWig
from tqdm import tqdm

tqdm.pandas()
import zarr


DEFINED_SYMBOLS = np.frombuffer("ACGTacgt".encode("ascii"), dtype="S1")
UNMASKED_SYMBOLS = np.frombuffer("ACGT".encode("ascii"), dtype="S1")


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


def save_fasta(path, genome):
    with bgzf.BgzfWriter(path, "wb") if path.endswith(".gz") else open(
        path, "w"
    ) as handle:
        SeqIO.write(genome.values(), handle, "fasta")


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


def load_repeatmasker(path):
    df = pd.read_csv(path, sep="\t").rename(
        columns=dict(genoName="chrom", genoStart="start", genoEnd="end")
    )
    df.chrom = df.chrom.astype(str)
    return df


class Genome:
    def __init__(self, path, subset_chroms=None):
        self._genome = load_fasta(path, subset_chroms=subset_chroms)

    def get_seq(self, chrom, start, end, strand="+"):
        seq = self._genome[chrom][start:end]
        if strand == "-":
            seq = str(Seq(seq).reverse_complement())
        return seq

    def get_nuc(self, chrom, pos, strand="+"):
        # pos is assumed to be 1-based as in VCF
        seq = self._genome[chrom][pos - 1]
        if strand == "-":
            seq = str(Seq(seq).reverse_complement())
        return seq

    def filter_chroms(self, chroms):
        self._genome = self._genome[chroms]

    def get_seq_fwd_rev(self, chrom, start, end):
        seq_fwd = self.get_seq(chrom, start, end)
        seq_rev = str(Seq(seq_fwd).reverse_complement())
        return seq_fwd, seq_rev

    def get_all_intervals(self):
        return pd.DataFrame(
            [
                {"chrom": chrom, "start": 0, "end": len(seq)}
                for chrom, seq in self._genome.items()
            ]
        )

    def get_intervals_matching_symbols(self, symbols):
        def get_intervals_matching_symbols_chrom(chrom):
            complete_interval = pd.DataFrame(
                {"chrom": [chrom.name], "start": [0], "end": [len(chrom.seq)]}
            )
            intervals = pd.DataFrame(
                dict(
                    start=np.where(
                        ~np.isin(
                            np.frombuffer(chrom.seq.encode("ascii"), dtype="S1"),
                            symbols,
                        )
                    )[0]
                )
            )
            if len(intervals) > 0:
                intervals["chrom"] = chrom.name
                intervals["end"] = intervals.start + 1
                intervals = bf.merge(intervals).drop(columns="n_intervals")
                return bf.subtract(complete_interval, intervals)
            return complete_interval

        return pd.concat(
            self._genome.rename("seq")
            .to_frame()
            .progress_apply(
                get_intervals_matching_symbols_chrom,
                axis=1,
            )
            .values,
            ignore_index=True,
        )

    def get_defined_intervals(self):
        return self.get_intervals_matching_symbols(DEFINED_SYMBOLS)

    def get_unmasked_intervals(self):
        return self.get_intervals_matching_symbols(UNMASKED_SYMBOLS)


def add_space_every_k(seq, k):
    return " ".join([seq[x : x + k] for x in range(0, len(seq), k)])


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


# TODO: maybe call it I or Is, or ivals instead of intervals


def get_annotation_features(annotation, feature):
    annotation_features = annotation[annotation.feature == feature]
    return bf.merge(
        bf.sanitize_bedframe(annotation_features[["chrom", "start", "end"]])
    )


def intersect_intervals(a, b):
    return bf.overlap(a, b, how="inner", return_overlap=True)[
        ["chrom", "overlap_start", "overlap_end"]
    ].rename(columns=dict(overlap_start="start", overlap_end="end"))


def union_intervals(a, b):
    return bf.merge(pd.concat([a, b], ignore_index=True)).drop(columns="n_intervals")


def intervals_size(intervals):
    return (intervals.end - intervals.start).sum()


def add_flank(intervals, flank):
    return bf.merge(bf.expand(intervals, pad=flank)).drop(columns="n_intervals")


def add_jitter(intervals, magnitude, seed=42):
    # After using this function, we recommend intersecting with
    # Genome.get_all_intervals(), to avoid getting out of chromosome bounds
    # or smaller subsets such as Genome.get_defined_intervals()
    rng = np.random.default_rng(seed)
    jitter = rng.integers(-magnitude, magnitude, size=len(intervals), endpoint=True)
    new_intervals = intervals.copy()
    new_intervals.start += jitter
    new_intervals.end += jitter
    return bf.merge(new_intervals)


def filter_length(intervals, min_interval_len):
    return intervals[intervals.end - intervals.start >= min_interval_len]


def filter_defined(intervals, genome, include_flank=None):
    defined = genome.get_defined_intervals()
    if include_flank is not None:
        defined = add_flank(defined, include_flank)
    return intersect_intervals(intervals, defined)


def filter_unmasked(intervals, genome, include_flank=None):
    unmasked = genome.get_unmasked_intervals()
    if include_flank is not None:
        unmasked = add_flank(unmasked, include_flank)
    return intersect_intervals(intervals, unmasked)


def filter_annotation_features(
    intervals,
    annotation,
    feature,
    include_flank=None,
    jitter=None,
):
    annotation_features = get_annotation_features(annotation, feature)
    if include_flank is not None:
        annotation_features = add_flank(annotation_features, include_flank)
    if jitter is not None:
        annotation_features = add_jitter(annotation_features, jitter)
    return intersect_intervals(intervals, annotation_features)


def get_promoters(annotation, upstream_size, downstream_size=0):
    # not exactly getting promoters, just gettting regions upstream of TSS

    def get_promoter(transcript):
        if transcript.strand == "+":
            start, end = (
                transcript.start - upstream_size,
                transcript.start + downstream_size,
            )
        else:
            start, end = (
                transcript.end - downstream_size,
                transcript.end + upstream_size,
            )
        return pd.Series(dict(chrom=transcript.chrom, start=start, end=end))

    transcripts = annotation[annotation.feature.isin(["mRNA", "transcript"])]
    promoters = transcripts.apply(get_promoter, axis=1)
    return bf.merge(promoters).drop(columns="n_intervals")


def get_random_intervals(intervals, size, n, seed=42):
    rng = np.random.default_rng(seed)
    interval_size = (intervals.end - intervals.start).values
    # the number of random intervals that can be generated per interval
    # e.g. if target size is 512, an interval of size 512 can produce 1 interval,
    # and interval of size 513 can produce 2 intervals
    interval_w = 1 + interval_size - size
    interval_p = interval_w / interval_w.sum()
    rand_interval_index = rng.choice(len(intervals), p=interval_p, size=n)

    rand_intervals = []
    for i in range(n):
        interval = intervals.iloc[rand_interval_index[i]]
        start = rng.integers(interval.start, interval.end - size, endpoint=True)
        end = start + size
        rand_intervals.append([interval.chrom, start, end])
    rand_intervals = pd.DataFrame(rand_intervals, columns=["chrom", "start", "end"])
    return bf.merge(rand_intervals).drop(columns="n_intervals")


def get_balanced_intervals(
    defined_intervals, annotation, window_size, promoter_upstream=1000
):
    # there's the issue of pseudogenes though... should be aware
    exons = add_flank(get_annotation_features(annotation, "exon"), window_size // 2)
    print("exons: ", intervals_size(exons) / intervals_size(defined_intervals))
    promoters = add_flank(
        get_promoters(annotation, promoter_upstream), window_size // 2
    )
    print("promoters: ", intervals_size(promoters) / intervals_size(defined_intervals))
    intervals = union_intervals(exons, promoters)
    intervals = intersect_intervals(add_jitter(intervals, 100), defined_intervals)
    # in case they collide with undefined intervals
    intervals = filter_length(intervals, window_size)
    print("intervals: ", intervals_size(intervals) / intervals_size(defined_intervals))
    # maybe add a 0.5 factor
    n_random_intervals = intervals_size(intervals) // window_size
    random_intervals = get_random_intervals(
        defined_intervals, window_size, n_random_intervals
    )
    print(
        "random_intervals: ",
        intervals_size(random_intervals) / intervals_size(defined_intervals),
    )
    intervals = union_intervals(intervals, random_intervals)
    print("intervals: ", intervals_size(intervals) / intervals_size(defined_intervals))
    print((intervals.end - intervals.start).min())
    assert (intervals.end - intervals.start).min() >= window_size
    return intervals


def make_windows(intervals, window_size, step_size, add_rc=False):
    return pd.concat(
        intervals.progress_apply(
            lambda interval: get_interval_windows(
                interval, window_size, step_size, add_rc
            ),
            axis=1,
        ).values,
        ignore_index=True,
    )


def get_interval_windows(interval, window_size, step_size, add_rc):
    windows = pd.DataFrame(
        dict(start=np.arange(interval.start, interval.end - window_size + 1, step_size))
    )
    windows["end"] = windows.start + window_size
    windows["chrom"] = interval.chrom
    windows = windows[["chrom", "start", "end"]]  # just re-ordering
    windows["strand"] = "+"
    if add_rc:
        windows_neg = windows.copy()  # TODO: this should be optional
        windows_neg.strand = "-"
        return pd.concat([windows, windows_neg], ignore_index=True)
    return windows


def get_seq(intervals, genome):
    intervals["seq"] = intervals.progress_apply(
        lambda i: genome.get_seq(i.chrom, i.start, i.end, i.strand),
        axis=1,
    )
    return intervals


class BigWig(object):
    def __init__(self, path):
        self.bw = pyBigWig.open(path)

    def get_features(self, chrom, start, end, strand="+"):
        x = self.bw.values(chrom, start, end, numpy=True)
        if strand == "-":
            x = x[::-1]
        return x


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
            #self.f.close()
        else:
            # pd.Series does not work with h5py/zarr object
            # (attempts to load all data into memory)
            # beware: dict has issues with parallelism in Pytorch
            self.data = {chrom: self.f[chrom] for chrom in chroms}
        print("Loading MSA... Done")

    def get_msa(self, chrom, start, end, strand="+", tokenize=False):
        msa = self.data[chrom][start:end]
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
        msa = np.char.upper(self.data[chrom][pos - 1])
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
