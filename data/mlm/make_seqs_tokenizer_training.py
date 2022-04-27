from Bio import SeqIO
import gzip
import numpy as np
from tqdm import tqdm


fasta_path = "./genomes/all.contigs.fa.gz"
min_contig_size = 1001
window_size = 1000
n_seqs = 5000
output_path = "./seqs_tokenizer_training_1k_5k.txt"


print("Loading fasta.")

with gzip.open(fasta_path, "rt") as handle:
    contigs = [contig for contig in SeqIO.parse(handle, "fasta") if len(contig) > min_contig_size]
print("Done.")
contig_sizes = np.array([len(contig) for contig in contigs])
contig_probs = contig_sizes / contig_sizes.sum()
print(contig_sizes)
n_contigs = len(contigs)


rs = np.random.RandomState(seed=42)

with open(output_path, "a") as f:
    for _ in tqdm(range(n_seqs)):
        contig_index = rs.choice(n_contigs, p=contig_probs)
        contig = contigs[contig_index]
        start = rs.randint(len(contig)-window_size)
        end = start + window_size
        seq = contig[start:end].seq
        seq = seq.upper()
        strand = rs.choice(["+", "-"])
        if strand == "-":
            seq = seq.reverse_complement()
        seq = str(seq)
        f.write(seq + "\n")
