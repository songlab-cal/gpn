from Bio import SeqIO
import numpy as np
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import AutoTokenizer


class GenomeSamplerDataset(IterableDataset):
    def __init__(self, fasta_path=None, tokenizer_path=None, window_size=None, max_length=None, random_seed=None):
        super().__init__()
        self.fasta_path = fasta_path
        self.tokenizer_path = tokenizer_path
        self.window_size = window_size
        self.max_length = max_length
        self.random_seed = random_seed
        # TODO: figure out if fasta and tokenizer should be loaded and instantiated in __init__
        # on in __iter__ (for good memory/compute performance with multiple workers)
        # also some data structures are better than others (e.g. np array better than python list)

    def __iter__(self):
        print("Loading fasta.")
        contigs = list(SeqIO.parse(self.fasta_path, "fasta"))
        print("Done.")
        contig_sizes = np.array([len(contig) for contig in contigs])
        contig_probs = contig_sizes / contig_sizes.sum()
        n_contigs = len(contigs)

        print("Loading tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        print("Done.")

        seed = self.random_seed
        worker_info = get_worker_info()
        if worker_info is not None:
            seed = seed * worker_info.id
        rs = np.random.RandomState(seed=seed)
        print("worker_info: ", worker_info, " seed: ", seed)

        while True:
            contig_index = rs.choice(n_contigs, p=contig_probs)
            contig = contigs[contig_index]
            start = rs.randint(len(contig)-self.window_size)
            end = start + self.window_size
            seq = contig[start:end].seq
            strand = rs.choice(["+", "-"])
            if strand == "-":
                seq = seq.reverse_complement()
            seq = str(seq)
            x = tokenizer(seq, padding="max_length", max_length=self.max_length, return_token_type_ids=False, return_tensors="pt", truncation=True)
            x["input_ids"] = x["input_ids"].flatten()
            x["attention_mask"] = x["attention_mask"].flatten()
            yield x


#d = GenomeSamplerDataset(fasta_path="tair10.fa", tokenizer_path="./tokenizer_unigram_251_v2/", window_size=1000, max_length=280, random_seed=42)
#dl = DataLoader(d, batch_size=4, num_workers=3)
#i = 0
#for x in dl:
#    print(x)
#    i += 1
#    if i > 3: break
