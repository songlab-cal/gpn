from Bio import SeqIO
import bioframe as bf
import numpy as np
import pandas as pd
import sys


input_path = sys.argv[1]
output_path = sys.argv[2]
repeats_cut_away_from_border = int(sys.argv[3])  # try 128
min_contig_len = int(sys.argv[4])


defined_symbols = list("ACGTacgt")
repeats_symbols = list("acgt")


def find_good_intervals(record):
    x = pd.DataFrame(dict(chrom=[record.id], start=[0], end=[len(record)]))
    seq = np.array(list(str(record.seq)))
    
    undefined = pd.DataFrame(dict(start=np.where(~np.isin(seq, defined_symbols))[0]))
    undefined["chrom"] = record.id
    undefined["end"] = undefined.start + 1
    undefined = bf.merge(undefined)
    x = bf.subtract(x, undefined)

    repeats = pd.DataFrame(dict(start=np.where(np.isin(seq, repeats_symbols))[0]))
    repeats["chrom"] = record.id
    repeats["end"] = repeats.start + 1
    repeats = bf.merge(repeats)
    repeats = bf.merge(pd.concat([repeats, undefined], ignore_index=True))
    repeats.start += repeats_cut_away_from_border
    repeats.end -= repeats_cut_away_from_border
    repeats = repeats.query('end-start > 0')
    x = bf.subtract(x, repeats)

    x = x[x.end-x.start>=min_contig_len]

    return x


intervals = []
for record in SeqIO.parse(input_path, "fasta"):
    print(record.id)
    intervals.append(find_good_intervals(record))

intervals = pd.concat(intervals, ignore_index=True)
print(intervals)
intervals.to_csv(output_path, sep="\t", header=False, index=False)