import argparse
import pandas as pd


parser = argparse.ArgumentParser(
    description="""
Filter assembly metadata downloaded from NCBI Genome
(https://www.ncbi.nlm.nih.gov/data-hub/genome).

You can choose a set of taxa and apply filters such as annotation level,
assembly level.

Currently tested only on RefSeq assemblies (GCF*) but should be easy
to generalize for GenBank assemblies (GCA*). See difference here:
https://www.ncbi.nlm.nih.gov/books/NBK50679/#RefSeqFAQ.what_is_the_difference_between_1
"""
)
parser.add_argument("input_path", help="Input path (tsv file)", type=str)
parser.add_argument("output_path", help="Output path (tsv file)", type=str)
parser.add_argument("--priority_assemblies", help="Always included", nargs="+")
parser.add_argument(
    "--keep_one_per_genus", help="Keep one per genus", action="store_true"
)
parser.add_argument(
    "--subsample_n",
    type=int,
    help="Number of accessions to keep",
)
parser.add_argument(
    "--max_size",
    type=int,
    help="Filter out assemblies larger than this size (in bp)",
)
args = parser.parse_args()

assemblies = pd.read_csv(args.input_path, sep="\t", index_col=0)
assemblies = assemblies[assemblies.index.str.startswith("GCF")]
if args.max_size is not None:
    assemblies = assemblies[
        assemblies["Assembly Stats Total Sequence Length"] <= args.max_size
    ]
assemblies["genus"] = assemblies["Organism Name"].str.split(" ").str[0]
assemblies["Assembly Level"] = pd.Categorical(
    assemblies["Assembly Level"],
    ["Complete", "Chromosome", "Scaffold", "Contig"],  # preference order
)
assemblies.loc[:, "Priority"] = "1_Low"
if args.priority_assemblies is not None:
    assemblies.loc[args.priority_assemblies, "Priority"] = "0_High"

if args.keep_one_per_genus:
    assemblies = assemblies.sort_values(
        ["Priority", "Assembly Level", "Organism Name"]
    ).drop_duplicates("genus")
if args.subsample_n is not None:
    assemblies = (
        assemblies.sample(frac=1, random_state=42)
        .sort_values(
            "Priority",
            kind="stable",
        )
        .head(args.subsample_n)
    )

assemblies.to_csv(args.output_path, sep="\t")
