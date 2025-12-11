import pandas as pd
import numpy as np
from os import system, makedirs
from os.path import isfile, exists
from pyensembl import EnsemblRelease
import pyBigWig
from sklearn.metrics import roc_auc_score, average_precision_score

from tqdm import tqdm
import multiprocessing as mp

# Output path
SAVE_PATH = "/scratch/users/czye/GPN/genome_wide_eda/"

# Models
TABIX_BIN = "/scratch/users/czye/conda/envs/gpn_env/bin/tabix"
GPN_WG_PATH = "/scratch/users/czye/GPN/genome_wide_eda/scores.tsv.bgz"
CADD_WG_PATH = "/scratch/users/czye/GPN/genome_wide_eda/whole_genome_SNVs.tsv.gz"
PHYLOP100_WG_PATH = "/scratch/users/czye/GPN/genome_wide_eda/hg38.phyloP100way.bw"
PHYLOP241_WG_PATH = "/scratch/users/czye/GPN/genome_wide_eda/cactus241way.phyloP.bw"

# Data
GNOMAD_PLI_V2_PATH = (
    "/scratch/users/czye/GPN/genome_wide_eda/gnomad.v2.1.1.lof_metrics.by_gene.txt"
)
GNOMAD_PLI_V4_PATH = (
    "/scratch/users/czye/GPN/genome_wide_eda/gnomad.v4.0.constraint_metrics.tsv"
)
DEPMAP_GENE_DEP_PATH = (
    "/scratch/users/czye/GPN/genome_wide_eda/DepMap_GeneDependency.csv.gz"
)
HS_COEFF_PATH = "/scratch/users/czye/GPN/genome_wide_eda/elife-83172-supp2-v2.txt"

STOP_GAIN_LIST = "/scratch/users/czye/GPN/genome_wide_eda/stop_gain_list.parquet"
SPLICE_LIST = "/scratch/users/czye/GPN/genome_wide_eda/splice_list.parquet"
LOST_LIST = "/scratch/users/czye/GPN/genome_wide_eda/start_stop_lost_list.parquet"

variant_types = [
    "all",
    "exon_splice",
    "all_variants",
    "exon_splice_variants",
    "intron_variants",
    "plof",
    "high_impact",
    "stop_gain",
    "splice",
    "lost",
]


def in_exon_and_splice(value, exon_regions):
    return any((start - 2) <= value <= (end + 2) for start, end in exon_regions)


def extract_gpn_score_per_gene(gene):
    ensembl = EnsemblRelease(110)
    gene_to_transcript = pd.read_csv(f"{SAVE_PATH}/gene_to_transcript.csv").set_index(
        "gene"
    )
    transcript_id = gene_to_transcript.loc[gene, "transcript"]

    try:
        transcript = ensembl.transcript_by_id(transcript_id)
    except:
        # print(gene)
        return
    start = transcript.start
    end = transcript.end
    chrom = transcript.contig

    #
    exon_regions = [(exon.start, exon.end) for exon in transcript.exons]

    df_filter = {}
    filter_consequences = {
        "stop_gain": STOP_GAIN_LIST,
        "splice": SPLICE_LIST,
        "lost": LOST_LIST,
    }
    for c, path in filter_consequences.items():
        df_filter[c] = pd.read_parquet(path)
        df_filter[c].columns = [0, 1, 2, 3]
        df_filter[c][0] = df_filter[c][0].astype(int)

    try:
        p = {}
        for t in variant_types:
            p[t] = {}

        if not exists(f"{SAVE_PATH}/extracted_scores_genes"):
            makedirs(f"{SAVE_PATH}/extracted_scores_genes")

        # Extract GPN_MSA scores
        if not isfile(f"{SAVE_PATH}/extracted_scores_genes/{gene}_GPN_MSA.tsv.gz"):
            system(
                f"{TABIX_BIN} {GPN_WG_PATH} {chrom}:{'{:,}'.format(start)}-{'{:,}'.format(end)} | gzip -c > {SAVE_PATH}/extracted_scores_genes/{gene}_GPN_MSA.tsv.gz"
            )
        df = pd.read_csv(
            f"{SAVE_PATH}/extracted_scores_genes/{gene}_GPN_MSA.tsv.gz",
            sep="\t",
            compression="gzip",
            header=None,
        )

        # write CDS variant list
        # out_df = df.loc[df[1].apply(in_exon_and_splice, exon_regions=exon_regions), [0, 1, 2, 3]].copy()
        # out_df[4] = transcript_id
        # out_df.to_csv(f'{SAVE_PATH}/variants_by_gene_exon+splice.csv', header = False, mode = 'a')

        is_type = {}
        for c in filter_consequences:
            df_c = df.merge(df_filter[c], on=[0, 1, 2, 3], how="inner")
            is_type[c] = (
                df[0].astype(str) + "_" + df[1].astype(str) + df[2] + df[3]
            ).isin(
                (df_c[0].astype(str) + "_" + df_c[1].astype(str) + df_c[2] + df_c[3])
            )
            p[c]["GPN-MSA"] = -df_c[4]
        p["plof"]["GPN-MSA"] = pd.concat(
            [p["stop_gain"]["GPN-MSA"], p["splice"]["GPN-MSA"]], axis=0
        )
        p["high_impact"]["GPN-MSA"] = pd.concat(
            [p["stop_gain"]["GPN-MSA"], p["splice"]["GPN-MSA"], p["lost"]["GPN-MSA"]],
            axis=0,
        )

        is_exon_and_splice_variants = df[1].apply(
            in_exon_and_splice, exon_regions=exon_regions
        )
        p["all_variants"]["GPN-MSA"] = -df[4]
        p["exon_splice_variants"]["GPN-MSA"] = -df.loc[is_exon_and_splice_variants, 4]
        p["intron_variants"]["GPN-MSA"] = -df.loc[~is_exon_and_splice_variants, 4]

        # extract top 0.01pct intron variants
        out_df = df.loc[~is_exon_and_splice_variants, [0, 1, 2, 3, 4]].copy()
        out_df = out_df.loc[
            p["intron_variants"]["GPN-MSA"]
            > p["intron_variants"]["GPN-MSA"].quantile(1 - 0.0001)
        ]
        out_df[5] = transcript_id
        out_df.to_csv(
            f"{SAVE_PATH}/variants_intron_0.01pct.csv", header=False, mode="a"
        )

        df = df.groupby(1)[4].mean().reset_index()
        p["all"]["GPN-MSA"] = -df[4]  # negate GPN-MSA score

        is_exon_and_splice = df[1].apply(in_exon_and_splice, exon_regions=exon_regions)
        p["exon_splice"]["GPN-MSA"] = -df.loc[is_exon_and_splice, 4]

        # Extract CADD scores
        if not isfile(f"{SAVE_PATH}/extracted_scores_genes/{gene}_CADD.tsv.gz"):
            system(
                f"{TABIX_BIN} {CADD_WG_PATH} {chrom}:{'{:,}'.format(start)}-{'{:,}'.format(end)} | gzip -c > {SAVE_PATH}/extracted_scores_genes/{gene}_CADD.tsv.gz"
            )
        df = pd.read_csv(
            f"{SAVE_PATH}/extracted_scores_genes/{gene}_CADD.tsv.gz",
            sep="\t",
            compression="gzip",
            header=None,
        )

        for c in filter_consequences:
            df_c = df.merge(df_filter[c], on=[0, 1, 2, 3], how="inner")
            p[c]["CADD-Raw"] = df_c[4]
        p["plof"]["CADD-Raw"] = pd.concat(
            [p["stop_gain"]["CADD-Raw"], p["splice"]["CADD-Raw"]], axis=0
        )
        p["high_impact"]["CADD-Raw"] = pd.concat(
            [
                p["stop_gain"]["CADD-Raw"],
                p["splice"]["CADD-Raw"],
                p["lost"]["CADD-Raw"],
            ],
            axis=0,
        )

        p["all_variants"]["CADD-Raw"] = df[4]
        p["exon_splice_variants"]["CADD-Raw"] = df.loc[is_exon_and_splice_variants, 4]
        p["intron_variants"]["CADD-Raw"] = -df.loc[~is_exon_and_splice_variants, 4]

        df = df.groupby(1)[4].mean().reset_index()
        p["all"]["CADD-Raw"] = df[4]
        p["exon_splice"]["CADD-Raw"] = df.loc[is_exon_and_splice, 4]

        # Extract PhloP100Way
        # Note: per-site score, thus size / 3
        bw = pyBigWig.open(PHYLOP100_WG_PATH)
        p["all"]["PhyloP100Way"] = pd.Series(bw.values(f"chr{chrom}", start - 1, end))
        p["exon_splice"]["PhyloP100Way"] = p["all"]["PhyloP100Way"].loc[
            is_exon_and_splice
        ]
        p["intron_variants"]["PhyloP100Way"] = p["all"]["PhyloP100Way"].loc[
            ~is_exon_and_splice
        ]
        p["all_variants"]["PhyloP100Way"] = p["all"]["PhyloP100Way"]
        p["exon_splice_variants"]["PhyloP100Way"] = p["exon_splice"]["PhyloP100Way"]
        for c in filter_consequences:
            p[c]["PhyloP100Way"] = (
                p["all"]["PhyloP100Way"].repeat(3).loc[is_type[c].values]
            )
        p["plof"]["PhyloP100Way"] = pd.concat(
            [p["stop_gain"]["PhyloP100Way"], p["splice"]["PhyloP100Way"]], axis=0
        )
        p["high_impact"]["PhyloP100Way"] = pd.concat(
            [
                p["stop_gain"]["PhyloP100Way"],
                p["splice"]["PhyloP100Way"],
                p["lost"]["PhyloP100Way"],
            ],
            axis=0,
        )

        bw = pyBigWig.open(PHYLOP241_WG_PATH)
        p["all"]["PhyloP241Way"] = pd.Series(bw.values(f"chr{chrom}", start - 1, end))
        p["exon_splice"]["PhyloP241Way"] = p["all"]["PhyloP241Way"].loc[
            is_exon_and_splice
        ]
        p["intron_variants"]["PhyloP241Way"] = p["all"]["PhyloP241Way"].loc[
            ~is_exon_and_splice
        ]
        p["all_variants"]["PhyloP241Way"] = p["all"]["PhyloP241Way"]
        p["exon_splice_variants"]["PhyloP241Way"] = p["exon_splice"]["PhyloP241Way"]
        for c in filter_consequences:
            p[c]["PhyloP241Way"] = (
                p["all"]["PhyloP241Way"].repeat(3).loc[is_type[c].values]
            )
        p["plof"]["PhyloP241Way"] = pd.concat(
            [p["stop_gain"]["PhyloP241Way"], p["splice"]["PhyloP241Way"]], axis=0
        )
        p["high_impact"]["PhyloP241Way"] = pd.concat(
            [
                p["stop_gain"]["PhyloP241Way"],
                p["splice"]["PhyloP241Way"],
                p["lost"]["PhyloP241Way"],
            ],
            axis=0,
        )

        # Output
        percentiles = [
            0.5,
            0.25,
            0.1,
            0.05,
            0.01,
            0.005,
            0.001,
            0.0005,
            0.0001,
            0.00005,
            0.00001,
        ]
        top_n = [10, 50, 100, 500, 1000, 5000, 10000]
        output = [
            p[c][model].quantile(1 - q)
            for c in variant_types
            for model in ["GPN-MSA", "CADD-Raw", "PhyloP100Way", "PhyloP241Way"]
            for q in percentiles
        ]

        output = output + [
            p[c][model].nlargest(n).mean()
            for c in [
                "all",
                "exon_splice",
                "all_variants",
                "exon_splice_variants",
                "plof",
                "high_impact",
            ]
            for model in ["GPN-MSA", "CADD-Raw", "PhyloP100Way", "PhyloP241Way"]
            for n in top_n
        ]

        output = output + [
            p[c][model].nlargest(n).median()
            for c in [
                "all",
                "exon_splice",
                "all_variants",
                "exon_splice_variants",
                "plof",
                "high_impact",
            ]
            for model in ["GPN-MSA", "CADD-Raw", "PhyloP100Way", "PhyloP241Way"]
            for n in top_n
        ]

        output = output + [
            p[c][model].nlargest(n).iloc[-1]
            for c in [
                "all",
                "exon_splice",
                "all_variants",
                "exon_splice_variants",
                "plof",
                "high_impact",
            ]
            for model in ["GPN-MSA", "CADD-Raw", "PhyloP100Way", "PhyloP241Way"]
            for n in top_n
        ]

        output = [p[c]["GPN-MSA"].shape[0] for c in variant_types] + output
        return [gene] + output

    except:
        return


if __name__ == "__main__":
    # Get pLI tables
    df_constr_v2 = pd.read_csv(GNOMAD_PLI_V2_PATH, sep="\t")
    df_constr_v2 = df_constr_v2[["gene", "transcript", "pLI"]]
    df_constr_v4 = pd.read_csv(GNOMAD_PLI_V4_PATH, sep="\t")
    df_constr_v4 = df_constr_v4[["transcript", "lof_hc_lc.pLI"]]
    df_constr_v4.columns = ["transcript", "pLI_v4"]
    df_constr = df_constr_v2.merge(
        df_constr_v4, left_on="transcript", right_on="transcript", how="inner"
    )
    df_constr = df_constr.set_index("gene")

    df_constr["transcript"].to_csv(f"{SAVE_PATH}/gene_to_transcript.csv")
    # print(df_constr)

    # Extract score stats in parallel

    n_workers = mp.cpu_count()

    print(f"Extracting scores in parallel, {n_workers} workers...")
    with mp.Pool(n_workers - 1) as p:
        r = list(
            tqdm(
                p.imap_unordered(extract_gpn_score_per_gene, df_constr.index),
                total=df_constr.index.shape[0],
            )
        )

    scores_stats = pd.DataFrame([x for x in r if x is not None])
    scores_stats.columns = (
        ["gene", "gene_length", "exon_splice_length"]
        + [
            f"num_variants_{t}"
            for t in [
                "all",
                "exon_splice",
                "intron",
                "plof",
                "high_impact",
                "stop_gain",
                "splice",
                "lost",
            ]
        ]
        + [
            f"{model}_{t}_{stat}"
            for t in variant_types
            for model in ["GPN-MSA", "CADD-Raw", "PhyloP100Way", "PhyloP241Way"]
            for stat in [
                "median",
                "25pct",
                "10pct",
                "5pct",
                "1pct",
                "0.5pct",
                "0.1pct",
                "0.05pct",
                "0.01pct",
                "0.005pct",
                "0.001pct",
            ]
        ]
        + [
            f"{model}_{t}_top{stat}mean"
            for t in [
                "all",
                "exon_splice",
                "all_variants",
                "exon_splice_variants",
                "plof",
                "high_impact",
            ]
            for model in ["GPN-MSA", "CADD-Raw", "PhyloP100Way", "PhyloP241Way"]
            for stat in [10, 50, 100, 500, 1000, 5000, 10000]
        ]
        + [
            f"{model}_{t}_top{stat}median"
            for t in [
                "all",
                "exon_splice",
                "all_variants",
                "exon_splice_variants",
                "plof",
                "high_impact",
            ]
            for model in ["GPN-MSA", "CADD-Raw", "PhyloP100Way", "PhyloP241Way"]
            for stat in [10, 50, 100, 500, 1000, 5000, 10000]
        ]
        + [
            f"{model}_{t}_{stat}th"
            for t in [
                "all",
                "exon_splice",
                "all_variants",
                "exon_splice_variants",
                "plof",
                "high_impact",
            ]
            for model in ["GPN-MSA", "CADD-Raw", "PhyloP100Way", "PhyloP241Way"]
            for stat in [10, 50, 100, 500, 1000, 5000, 10000]
        ]
    )

    scores_stats = scores_stats.set_index("gene")

    # Merge pLI tables
    df_constr = df_constr.merge(
        scores_stats, left_index=True, right_index=True, how="inner"
    )

    # Merge DepMap gene dependency table
    df_gene_dep = pd.read_csv(DEPMAP_GENE_DEP_PATH, index_col=0)
    df_gene_dep.columns = df_gene_dep.columns.str.split(
        " ", expand=True
    ).get_level_values(0)
    df_gene_dep = df_gene_dep.T
    # Dependent cell line defined as probability > 0.5
    df_gene_dep["DepMap_num_dependent_cell_lines"] = df_gene_dep.apply(
        lambda x: (x > 0.5).sum(), axis=1
    )
    df_gene_dep = df_gene_dep[["DepMap_num_dependent_cell_lines"]]
    df_constr = df_constr.merge(
        df_gene_dep, left_index=True, right_index=True, how="left"
    )

    # Merge hs (heterozygous selection) coefficients
    df_hs = pd.read_csv(HS_COEFF_PATH, sep="\t")
    df_hs = df_hs[["Ensembl_transcript_id", "log10_map"]]
    df_hs.columns = ["transcript", "hs_coeff"]
    df_constr = df_constr.merge(df_hs, on="transcript", how="left")

    # Define highly dependent genes and non-essential genes
    df_constr["is_highly_dependent"] = np.nan
    df_constr.loc[
        df_constr["DepMap_num_dependent_cell_lines"] > 1000, "is_highly_dependent"
    ] = 1
    df_constr.loc[
        df_constr["DepMap_num_dependent_cell_lines"] < 1, "is_highly_dependent"
    ] = 0

    # Classifiers
    cls_scores = (
        [
            f"{model}_{t}_{stat}"
            for t in variant_types
            for model in ["GPN-MSA", "CADD-Raw", "PhyloP100Way", "PhyloP241Way"]
            for stat in [
                "median",
                "25pct",
                "10pct",
                "5pct",
                "1pct",
                "0.5pct",
                "0.1pct",
                "0.05pct",
                "0.01pct",
                "0.005pct",
                "0.001pct",
            ]
        ]
        + [
            f"{model}_{t}_top{stat}mean"
            for t in [
                "all",
                "exon_splice",
                "all_variants",
                "exon_splice_variants",
                "plof",
                "high_impact",
            ]
            for model in ["GPN-MSA", "CADD-Raw", "PhyloP100Way", "PhyloP241Way"]
            for stat in [10, 50, 100, 500, 1000, 5000, 10000]
        ]
        + [
            f"{model}_{t}_top{stat}median"
            for t in [
                "all",
                "exon_splice",
                "all_variants",
                "exon_splice_variants",
                "plof",
                "high_impact",
            ]
            for model in ["GPN-MSA", "CADD-Raw", "PhyloP100Way", "PhyloP241Way"]
            for stat in [10, 50, 100, 500, 1000, 5000, 10000]
        ]
        + [
            f"{model}_{t}_{stat}th"
            for t in [
                "all",
                "exon_splice",
                "all_variants",
                "exon_splice_variants",
                "plof",
                "high_impact",
            ]
            for model in ["GPN-MSA", "CADD-Raw", "PhyloP100Way", "PhyloP241Way"]
            for stat in [10, 50, 100, 500, 1000, 5000, 10000]
        ]
        + ["pLI", "pLI_v4", "hs_coeff"]
    )
    not_na = ~df_constr[["is_highly_dependent"] + cls_scores].isna().any(axis=1)
    print("Is highly dependent gene")
    print(df_constr.loc[not_na, "is_highly_dependent"].astype(bool).value_counts())
    # Write score stats
    df_constr.to_parquet(f"{SAVE_PATH}/scores_stats_by_gene.parquet")

    # results
    DepMap_cls_performance = pd.DataFrame(index=cls_scores)
    for stat in DepMap_cls_performance.index:
        DepMap_cls_performance.loc[stat, "AUROC"] = roc_auc_score(
            df_constr.loc[not_na, "is_highly_dependent"], df_constr.loc[not_na, stat]
        )
        DepMap_cls_performance.loc[stat, "AUPRC"] = average_precision_score(
            df_constr.loc[not_na, "is_highly_dependent"], df_constr.loc[not_na, stat]
        )
    print(DepMap_cls_performance)
    # Write performance metrics
    DepMap_cls_performance.to_parquet(f"{SAVE_PATH}/depmap_cls_performance.parquet")
