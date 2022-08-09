from Bio import SeqIO
import pandas as pd


chromosome = 5
start = 3500000
end = 3600000

chromosome_seq = SeqIO.to_dict(SeqIO.parse("../../data/mlm/tair10.fa", "fasta"))["Chr5"]
print(chromosome_seq)

rows = []
nucleotides = ["A", "C", "G", "T"]

for i in range(start, end):
    ref = chromosome_seq[i]
    for alt in nucleotides:
        if alt == ref: continue
        rows.append([chromosome, i+1, '.', ref, alt, '.', '.', '.'])

df = pd.DataFrame(data=rows)
print(df)
df.to_csv("example.vcf.gz", sep="\t", index=False, header=False)