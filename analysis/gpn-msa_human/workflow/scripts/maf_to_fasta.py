import argparse
from Bio import AlignIO, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from gpn.data import load_fasta
import numpy as np
import pandas as pd
from tqdm import tqdm


def main(args):
    all_species = pd.read_csv(args.species_path, header=None)
    all_species["position"] = np.arange(len(all_species))
    all_species = all_species.set_index(0).position

    ref = load_fasta(args.ref_path).values[0]
    ref = np.frombuffer(ref.encode("ascii"), dtype="S1")

    X = np.full((len(ref), len(all_species)), b"-", dtype="S1")
    X[:, 0] = ref

    print("Reading MAF...")
    pbar = tqdm(total=len(ref))
    i = 0
    for msa in AlignIO.parse(args.maf_path, "maf"):
        if len(msa) < 2:
            continue
        target = msa[0]
        start = target.annotations["start"]
        end = start + target.annotations["size"]

        msa_np = []
        species = []
        for rec in msa[1:]:  # ignore target
            species.append(rec.id.split(".")[0])
            msa_np.append(np.frombuffer(str(rec.seq).encode("ascii"), dtype="S1"))
        msa_np = np.column_stack(msa_np)
        species_idx = all_species[species].values
        X[start:end, species_idx] = msa_np
        # TODO: deal with gaps in ref
        if i % 100_000 == 0:
            pbar.update(start)
        i += 1
    pbar.close()

    print("Writing output fasta...")

    def seq_iterator():
        for i, species in enumerate(tqdm(all_species.index)):
            yield SeqRecord(
                Seq(X[:, i].tobytes().decode("ascii")), id=species, description=""
            )

    with open(args.output_path, "w") as handle:
        SeqIO.write(seq_iterator(), handle, "fasta-2line")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MAF to Fasta, assuming no ref gaps in MAF."
    )
    parser.add_argument("species_path", help="Species path", type=str)
    parser.add_argument("ref_path", help="Ref path", type=str)
    parser.add_argument("maf_path", help="MAF path", type=str)
    parser.add_argument("output_path", help="Output path", type=str)
    args = parser.parse_args()
    main(args)
