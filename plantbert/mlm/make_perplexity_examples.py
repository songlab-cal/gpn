from collections import Counter
import numpy as np
import pandas as pd


chr_len = 26975502


gtf = pd.read_csv(
    "../../data/vep/tair10.gff", sep='\t', header=None, comment="#",
    names=['chromosome', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute'],
)
gtf = gtf[gtf.chromosome=="Chr5"]
gtf = gtf[gtf.feature != "chromosome"]  # redundant
gtf = gtf[gtf.feature != "protein"]  # redundant
gtf = gtf[gtf.feature != "gene"]  # redundant
print(gtf.shape)

overlaps = pd.Series(["Intergenic"] * chr_len)
i = -1
for row_index, row in gtf.iterrows():
    i += 1
    if i % 10000 == 0: print(i)
    overlaps[row.start:row.end] += "," + row.feature
overlaps = overlaps.str.split(",").apply(Counter)

segmentation = pd.Series([""] * chr_len)
segmentation[overlaps==Counter(Intergenic=1)] = "intergenic"
segmentation[overlaps==Counter(Intergenic=1, mRNA=1)] = "intron"
segmentation[overlaps==Counter(Intergenic=1, mRNA=1, exon=1, CDS=1)] = "cds"
segmentation[overlaps==Counter(Intergenic=1, mRNA=1, exon=1, five_prime_UTR=1)] = "five_prime_utr"
segmentation[overlaps==Counter(Intergenic=1, mRNA=1, exon=1, three_prime_UTR=1)] = "three_prime_utr"

segmentation = segmentation[segmentation!=""]
border = 10000
segmentation = segmentation[border:-border]
print(segmentation.value_counts())

segmentation_subset = segmentation.groupby(segmentation).sample(n=10000, random_state=42).to_frame().rename(columns={0: "Region"})
segmentation_subset = segmentation_subset.sample(frac=1, random_state=42)  # just shuffle
print(segmentation_subset)


segmentation_subset["chromosome"] = "Chr5"
segmentation_subset["pos"] = segmentation_subset.index.values
print(segmentation_subset)
segmentation_subset.to_csv("perplexity_examples.tsv.gz", index=False)