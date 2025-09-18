from Bio.Seq import Seq
import bioframe as bf
from scipy.spatial.distance import cdist
from gpn.data import Genome
from liftover import get_lifter
import numpy as np
import os
import pandas as pd
import polars as pl
import re
from tqdm import tqdm


CHROMS = [str(i) for i in range(1, 23)]
NUCLEOTIDES = list("ACGT")
COORDINATES = ["chrom", "pos", "ref", "alt"]
ALPHA_COLS = [f"alpha{i+1}" for i in range(10)]

NON_EXONIC_CONSEQUENCES = [
    "intergenic_variant",
    "intron_variant",
    "upstream_gene_variant",
    "downstream_gene_variant",
]

TRAITGYM_PATH = "/accounts/projects/yss/gbenegas/projects/functionality-prediction/"


def filter_snp(V):
    V = V[V.ref.isin(NUCLEOTIDES)]
    V = V[V.alt.isin(NUCLEOTIDES)]
    return V


def reverse_complement(seq):
    return str(Seq(seq).reverse_complement())


def sort_variants(V):
    chrom_order = [str(i) for i in range(1, 23)] + ["X", "Y"]
    V.chrom = pd.Categorical(V.chrom, categories=chrom_order, ordered=True)
    V = V.sort_values(["chrom", "pos", "ref", "alt"])
    V.chrom = V.chrom.astype(str)
    return V


def check_ref_alt(V, genome):
    V["ref_nuc"] = V.progress_apply(
        lambda v: genome.get_nuc(v.chrom, v.pos).upper(), axis=1
    )
    mask = V["ref"] != V["ref_nuc"]
    V.loc[mask, ["ref", "alt"]] = V.loc[mask, ["alt", "ref"]].values
    V = V[V["ref"] == V["ref_nuc"]]
    V.drop(columns=["ref_nuc"], inplace=True)
    return V


def lift_hg19_to_hg38(V):
    converter = get_lifter("hg19", "hg38", one_based=True)

    def get_new_coords(v):
        try:
            res = converter[v.chrom][v.pos]
            assert len(res) == 1
            chrom, pos, strand = res[0]
            chrom = chrom.replace("chr", "")
            ref = v.ref
            alt = v.alt
            if strand == "-":
                ref = reverse_complement(ref)
                alt = reverse_complement(alt)
            return chrom, pos, ref, alt
        except:
            return v.chrom, -1, v.ref, v.alt

    V[["chrom", "pos", "ref", "alt"]] = V.apply(
        get_new_coords, axis=1, result_type="expand"
    )
    return V


def match_features(pos, neg, continuous_features, categorical_features, k):
    pos = pos.set_index(categorical_features)
    neg = neg.set_index(categorical_features)
    res_pos = []
    res_neg = []
    for x in tqdm(pos.index.drop_duplicates()):
        pos_x = pos.loc[[x]].reset_index()
        try:
            neg_x = neg.loc[[x]].reset_index()
        except KeyError:
            print(f"WARNING: no match for {x}")
            continue
        if len(pos_x) * k > len(neg_x):
            print("WARNING: subsampling positive set")
            pos_x = pos_x.sample(len(neg_x) // k, random_state=42)
        if len(continuous_features) == 0:
            neg_x = neg_x.sample(len(pos_x) * k, random_state=42)
        else:
            neg_x = find_closest(pos_x, neg_x, continuous_features, k)
        res_pos.append(pos_x)
        res_neg.append(neg_x)
    res_pos = pd.concat(res_pos, ignore_index=True)
    res_pos["match_group"] = np.arange(len(res_pos))
    res_neg = pd.concat(res_neg, ignore_index=True)
    res_neg["match_group"] = np.repeat(res_pos.match_group.values, k)
    res = pd.concat([res_pos, res_neg], ignore_index=True)
    return res


def find_closest(pos, neg, cols, k):
    D = cdist(pos[cols], neg[cols])
    closest = []
    for i in range(len(pos)):
        js = np.argsort(D[i])[:k].tolist()
        closest += js
        D[:, js] = np.inf  # ensure they cannot be picked up again
    return neg.iloc[closest]


def parse_log(log_path):
    """return M (regression‑SNP count) and h2 (observed‑scale heritability)"""
    M = h2 = None
    rx_M = re.compile(r"After merging with regression SNP LD,\s*([\d,]+)\s*SNPs")
    rx_h2 = re.compile(r"Total Observed scale h2:\s*([0-9.eE+-]+)")
    with open(log_path) as fh:
        for line in fh:
            if M is None:
                m = rx_M.search(line)
                if m:
                    M = int(m.group(1).replace(",", ""))
            if h2 is None:
                m = rx_h2.search(line)
                if m:
                    h2 = float(m.group(1))
            if M is not None and h2 is not None:
                break
    if M is None or h2 is None:
        sys.exit("✖  Could not find M or h2 in the log file.")
    return M, h2


def add_tau_star(df, M, h2, p):
    """add τ* and its SE to a .results dataframe (binary annotations)"""
    sd = np.sqrt(p * (1 - p))  # std‑dev of 0/1 variable
    scale = sd * M / h2
    df["tau_star"] = df["Coefficient"] * scale
    df["tau_star_se"] = df["Coefficient_std_error"] * scale
    return df


def maf_match(V):
    n_bins = 100
    bins = np.linspace(0, 0.5, n_bins + 1)
    V["maf_bin"] = pd.cut(V.maf, bins=bins, labels=np.arange(n_bins))
    V["maf_bin"] = V["maf_bin"].astype(
        pd.CategoricalDtype(categories=np.arange(n_bins))
    )
    V_pos = V.query("label")
    V_neg = V.query("not label")
    # pos_hist = V_pos.maf_bin.value_counts().sort_index().values
    # neg_hist = V_neg.maf_bin.value_counts().sort_index().values

    pos_hist = (
        V_pos.groupby("maf_bin", observed=False)
        .size()
        .reindex(np.arange(n_bins), fill_value=0)
        .values
    )
    neg_hist = (
        V_neg.groupby("maf_bin", observed=False)
        .size()
        .reindex(np.arange(n_bins), fill_value=0)
        .values
    )

    pos_dist = pos_hist / len(V_pos)
    pos_dist_ratio_to_max = pos_dist / pos_dist.max()
    neg_hist_max = neg_hist[pos_hist.argmax()]
    upper_bound = np.floor(neg_hist_max * pos_dist_ratio_to_max)
    downsample = (neg_hist / upper_bound).min()
    target_n_samples = np.floor(upper_bound * downsample).astype(int)
    V_neg_matched = pd.concat(
        [
            V_neg[V_neg.maf_bin == i].sample(
                target_n_samples[i], replace=False, random_state=42
            )
            for i in range(n_bins)
        ]
    )
    V = pd.concat([V_pos, V_neg_matched])
    return V


rule download_genome:
    output:
        "results/genome.fa.gz",
    shell:
        "wget -O {output} {config[genome_url]}"


rule download_annotation:
    output:
        "results/annotation.gtf.gz",
    shell:
        "wget -O {output} {config[annotation_url]}"


rule tabix_input:
    input:
        "{variants}.parquet",
    output:
        "{variants}.tabix_input.tsv.gz",
    run:
        V = pd.read_parquet(input[0])
        V = V[V.pos != -1]
        V["start"] = V.pos
        V["end"] = V.start
        V = V[["chrom", "start", "end"]].drop_duplicates()
        V.to_csv(output[0], sep="\t", index=False, header=False)


rule process_tabix_output_gpnmsa:
    input:
        "results/variants/merged.parquet",
        "results/tmp/GPN-MSA.tabix_output.tsv",
    output:
        "results/variant_scores/GPN-MSA_LLR.parquet",
    run:
        V = pl.read_parquet(input[0])
        print(V)
        score = pl.read_csv(
            input[1],
            separator="\t",
            has_header=False,
            new_columns=COORDINATES + ["score"],
            schema_overrides={"chrom": str},
        )
        print(score)
        V = V.join(score, on=COORDINATES, how="left")
        print(V)
        V.select("score").write_parquet(output[0])


rule abs_llr:
    input:
        "results/features/{model}_LLR.parquet",
    output:
        "results/features/{model}_absLLR.parquet",
    run:
        V = pl.read_parquet(input[0])
        print(V)
        V = V.with_columns(pl.col("score").abs())
        print(V)
        V.write_parquet(output[0])


rule minus_llr:
    input:
        "results/features/{model}_LLR.parquet",
    output:
        "results/features/{model}_minusLLR.parquet",
    run:
        V = pl.read_parquet(input[0])
        print(V)
        V = V.with_columns(-pl.col("score"))
        print(V)
        V.write_parquet(output[0])


rule inner_product:
    input:
        "results/features/{model}_InnerProducts.parquet",
    output:
        "results/features/{model}_InnerProduct.parquet",
    run:
        (
            pl.read_parquet(input[0])
            .sum_horizontal()
            .to_frame("score")
            .write_parquet(output[0])
        )


rule process_tabix_output_cadd:
    input:
        "results/variants/merged.parquet",
        "results/tmp/CADD.tabix_output.tsv",
    output:
        "results/variant_scores/CADD.parquet",
    run:
        V = pl.read_parquet(input[0])
        print(V)
        score = (
            pl.read_csv(
                input[1],
                separator="\t",
                has_header=False,
                new_columns=COORDINATES + ["RawScore", "PHRED"],
                schema_overrides={"chrom": str},
            )
            .rename({"RawScore": "score"})
            .select(COORDINATES + ["score"])
        )
        print(score)
        V = V.join(score, on=COORDINATES, how="left")
        print(V)
        V.select("score").write_parquet(output[0])


rule extract_feature:
    input:
        "results/annotation.gtf.gz",
    output:
        "results/intervals/{feature,CDS|exon}/{flank,\d+}.parquet",
    run:
        feature = wildcards.feature
        flank = int(wildcards.flank)

        df = (
            pl.read_csv(
                input[0],
                has_header=False,
                separator="\t",
                comment_prefix="#",
                new_columns=[
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
                schema_overrides={"chrom": str},
            )
            .with_columns(pl.col("start") - 1)  # gtf to bed conversion
            .filter(feature=feature)
            .select(["chrom", "start", "end"])
            .unique()
            .sort(["chrom", "start", "end"])
            .to_pandas()
        )
        df = bf.expand(df, pad=flank)
        df = bf.merge(df).drop(columns="n_intervals")
        df.start = np.maximum(0, df.start)
        print(df)
        df.to_parquet(output[0], index=False)


rule process_features:
    input:
        "results/variants/merged.parquet",
        "results/features/{model}.parquet",
    output:
        "results/variant_scores/{model}.parquet",
    run:
        V = pd.read_parquet(input[0])
        print(V)
        score = pd.read_parquet(input[1], columns=["score"]).score.values
        print(score)
        V["score"] = np.nan
        V.loc[V.pos != -1, "score"] = score
        print(V.score.isna().sum())
        print(V)
        V[["score"]].to_parquet(output[0], index=False)


rule extract_score:
    input:
        "results/features/{features}.parquet",
    output:
        "results/features/{features}_extract_{score}.parquet",
    run:
        (
            pl.read_parquet(input[0], columns=[wildcards.score])
            .rename({wildcards.score: "score"})
            .write_parquet(output[0])
        )


rule make_ensembl_vep_input:
    input:
        "{anything}.parquet",
    output:
        temp("{anything}.ensembl_vep.input.tsv.gz"),
    threads: workflow.cores
    run:
        df = pd.read_parquet(input[0], columns=COORDINATES)
        df = df[df.pos != -1]
        df = df.drop_duplicates()
        df = df.sort_values(["chrom", "pos"])
        df["start"] = df.pos
        df["end"] = df.start
        df["allele"] = df.ref + "/" + df.alt
        df["strand"] = "+"
        df.to_csv(
            output[0],
            sep="\t",
            header=False,
            index=False,
            columns=["chrom", "start", "end", "allele", "strand"],
        )


# additional snakemake args (SCF):
# --use-singularity --singularity-args "--bind /scratch/users/gbenegas"
# or in savio:
# --use-singularity --singularity-args "--bind /global/scratch/projects/fc_songlab/gbenegas"
rule install_ensembl_vep_cache:
    output:
        directory("results/ensembl_vep_cache"),
    singularity:
        "docker://ensemblorg/ensembl-vep:release_109.1"
    threads: workflow.cores
    shell:
        "INSTALL.pl -c {output} -a cf -s homo_sapiens -y GRCh38"


rule run_ensembl_vep:
    input:
        "{anything}.ensembl_vep.input.tsv.gz",
        "results/ensembl_vep_cache",
    output:
        temp("{anything}.ensembl_vep.output.tsv.gz"),
    singularity:
        "docker://ensemblorg/ensembl-vep:release_109.1"
    threads: workflow.cores
    shell:
        """
        vep -i {input[0]} -o {output} --fork {threads} --cache \
        --dir_cache {input[1]} --format ensembl \
        --most_severe --compress_output gzip --tab --distance 1000 --offline
        """


rule process_ensembl_vep:
    input:
        "{anything}.parquet",
        "{anything}.ensembl_vep.output.tsv.gz",
    output:
        "{anything}.annot.parquet",
    run:
        V = pd.read_parquet(input[0])
        V2 = pd.read_csv(
            input[1], sep="\t", header=None, comment="#", usecols=[0, 6]
        ).rename(columns={0: "variant", 6: "consequence"})
        V2["chrom"] = V2.variant.str.split("_").str[0]
        V2["pos"] = V2.variant.str.split("_").str[1].astype(int)
        V2["ref"] = V2.variant.str.split("_").str[2].str.split("/").str[0]
        V2["alt"] = V2.variant.str.split("_").str[2].str.split("/").str[1]
        V2.drop(columns=["variant"], inplace=True)
        V = V.merge(V2, on=COORDINATES, how="left")
        print(V)
        V.to_parquet(output[0], index=False)


rule download_cre:
    output:
        temp("results/intervals/cre.tsv"),
    shell:
        "wget -O {output} https://downloads.wenglab.org/Registry-V4/GRCh38-cCREs.bed"


rule process_cre:
    input:
        "results/intervals/cre.tsv",
    output:
        "results/intervals/cre.parquet",
    run:
        df = pl.read_csv(
            input[0],
            separator="\t",
            has_header=False,
            columns=[0, 1, 2, 5],
            new_columns=["chrom", "start", "end", "cre_class"],
        ).with_columns(pl.col("chrom").str.replace("chr", ""))
        print(df)
        df.write_parquet(output[0])


rule cre_annotation_v1:
    input:
        "{anything}.annot.parquet",
        "results/intervals/cre.parquet",
    output:
        "{anything}.annot_with_cre_v1.parquet",
    run:
        V = pd.read_parquet(input[0])
        V["start"] = V.pos - 1
        V["end"] = V.pos
        cre = pd.read_parquet(input[1])
        cre_classes = ["PLS", "pELS", "dELS"]
        for c in cre_classes:
            I = cre[cre.cre_class == c]
            V = bf.coverage(V, I)
            V.loc[
                (V.consequence.isin(NON_EXONIC_CONSEQUENCES)) & (V.coverage > 0),
                "consequence",
            ] = c
            V = V.drop(columns=["coverage"])
        V = V.drop(columns=["start", "end"])
        V.to_parquet(output[0], index=False)


rule cre_annotation_v2:
    input:
        "{anything}.annot.parquet",
        "results/intervals/cre.parquet",
    output:
        "{anything}.annot_with_cre_v2.parquet",
    run:
        V = pd.read_parquet(input[0])
        V["start"] = V.pos - 1
        V["end"] = V.pos
        cre = pd.read_parquet(input[1])
        cre_classes = list(
            reversed(config["cre_classes"])
        )  # desceding order of priority
        cre_flank_classes = [cre_class + "-flank" for cre_class in cre_classes]
        for c in tqdm(cre_classes):
            I = cre[cre.cre_class == c]
            I = bf.expand(I, pad=500)
            I = bf.merge(I).drop(columns="n_intervals")
            V = bf.coverage(V, I)
            V.loc[
                (V.consequence.isin(NON_EXONIC_CONSEQUENCES)) & (V.coverage > 0),
                "consequence",
            ] = (
                c + "-flank"
            )
            V = V.drop(columns=["coverage"])
        for c in tqdm(cre_classes):
            I = cre[cre.cre_class == c]
            V = bf.coverage(V, I)
            V.loc[
                (V.consequence.isin(cre_flank_classes)) & (V.coverage > 0),
                "consequence",
            ] = c
            V = V.drop(columns=["coverage"])
        V = V.drop(columns=["start", "end"])
        print(V.consequence.value_counts())
        V.to_parquet(output[0], index=False)


rule functional_class_annotation:
    input:
        "results/variants/merged.parquet",
        "results/intervals/exon/17.parquet",
        "results/intervals/cre.parquet",
    output:
        "results/variants/functional_class.parquet",
    run:
        # remember to do the sorting
        V = pd.read_parquet(input[0])
        V["start"] = V.pos - 1
        V["end"] = V.pos
        n = len(V)
        V["original_order"] = np.arange(n)
        V["functional_class"] = "other"
        exon = pd.read_parquet(input[1])
        cre = pd.read_parquet(input[2])
        promoter = cre[cre.cre_class == "PLS"]
        enhancer = cre[cre.cre_class.isin(["pELS", "dELS"])]
        # exon last to overwrite other
        regions = [
            ("promoter", promoter),
            ("enhancer", enhancer),
            ("exon", exon),
        ]
        for region_name, region in regions:
            V = bf.coverage(V, region)
            V.loc[V.coverage > 0, "functional_class"] = region_name
            V = V.drop(columns="coverage")
        print(V.functional_class.value_counts())
        V = V.sort_values("original_order")
        assert len(V) == n
        V[["functional_class"]].to_parquet(output[0], index=False)


rule replace_zero_scores:
    input:
        "results/variant_scores/{model}.parquet",
    output:
        "results/variant_scores/{model}/replace_zero/{x}.parquet",
    run:
        (
            pl.read_parquet(input[0])
            .with_columns(
                pl.when(pl.col("score") == 0)
                .then(pl.lit(float(wildcards.x)))
                .otherwise(pl.col("score"))
                .alias("score")
            )
            .write_parquet(output[0])
        )


ruleorder: replace_zero_scores > quantile_score


rule score_consequence:
    input:
        "results/variants/merged.{annot_mode}.parquet",
        "results/maf/merged.parquet",
        "results/variant_scores/{model}.parquet",
    output:
        "results/score_consequence/{annot_mode,annot|annot_with_cre_v1|annot_with_cre_v2}/{model}.parquet",
    run:
        V = pd.read_parquet(input[0])
        MAF = pd.read_parquet(input[1])
        score = pd.read_parquet(input[2])
        assert sorted(score.score.unique()) == [0, 1]
        V = pd.concat([V, MAF, score], axis=1)
        V = V[V.score == 1]
        categories = ["all", "common", "low-frequency"]
        all_res = []
        for category in categories:
            if category == "all":
                V2 = V
            elif category == "common":
                V2 = V[V.MAF > 5 / 100]
            else:
                V2 = V[V.MAF <= 5 / 100]
            res = V2.consequence.value_counts().reset_index()
            total_row = pd.DataFrame(
                {
                    "consequence": ["total"],
                    "count": [len(V2)],
                }
            )
            res = pd.concat([res, total_row], ignore_index=True)
            res = res.sort_values("count", ascending=False)
            res["category"] = category
            all_res.append(res)
        res = pd.concat(all_res, ignore_index=True)
        print(res)
        res.to_parquet(output[0])


rule global_score_consequence:
    input:
        "results/variants/merged.{annot_mode}.parquet",
        "results/maf/merged.parquet",
    output:
        "results/global_score_consequence/{annot_mode,annot|annot_with_cre_v1|annot_with_cre_v2}.parquet",
    run:
        V = pd.read_parquet(input[0])
        MAF = pd.read_parquet(input[1])
        assert len(V) == len(MAF)
        V = pd.concat([V, MAF], axis=1)
        categories = ["all", "common", "low-frequency"]
        all_res = []
        for category in categories:
            if category == "all":
                V2 = V
            elif category == "common":
                V2 = V[V.MAF > 5 / 100]
            else:
                V2 = V[V.MAF <= 5 / 100]
            res = V2.consequence.value_counts().reset_index()
            total_row = pd.DataFrame(
                {
                    "consequence": ["total"],
                    "count": [len(V2)],
                }
            )
            res = pd.concat([res, total_row], ignore_index=True)
            res = res.sort_values("count", ascending=False)
            res["category"] = category
            all_res.append(res)
        res = pd.concat(all_res, ignore_index=True)
        print(res)
        res.to_parquet(output[0])


rule subset_consequence:
    input:
        "results/variants/merged.annot.parquet",
        "results/variant_scores/quantile/{model}.parquet",
    output:
        "results/variant_scores/quantile/subset_consequence/{consequence,missense|non-missense}/{model}.parquet",
    run:
        consequence = wildcards.consequence
        V = pd.read_parquet(input[0])
        score = pd.read_parquet(input[1])
        assert len(V) == len(score)
        V = pd.concat([V, score], axis=1)
        if consequence == "missense":
            mask = V.consequence == "missense_variant"
        elif consequence == "non-missense":
            mask = V.consequence != "missense_variant"
        print(V.score.value_counts())
        V.loc[~mask, "score"] = 0
        print(V.score.value_counts())
        raise Exception("debug")
        V[["score"]].to_parquet(output[0])


rule rsid_chrom:
    input:
        "results/1000G_EUR_Phase3_plink/1000G.EUR.QC.{chrom}.bim",
    output:
        "results/variants/rsid/{chrom}.parquet",
    wildcard_constraints:
        chrom="|".join(CHROMS),
    run:
        (
            pl.read_csv(
                input[0],
                has_header=False,
                separator="\t",
                columns=[1],
                new_columns=["rsid"],
            ).write_parquet(output[0])
        )


rule rsid_merge:
    input:
        expand("results/variants/rsid/{chrom}.parquet", chrom=CHROMS),
    output:
        "results/variants/rsid/merged.parquet",
    run:
        pl.concat([pl.read_parquet(f) for f in input]).write_parquet(output[0])


# TODO: make sure it has MAF as well
rule create_example_variants:
    input:
        "results/variants/merged.annot_with_cre_v2.parquet",
        "results/variants/rsid/merged.parquet",
        "results/maf/merged.parquet",
        f"results/variant_scores/{config['gpn_star_p']}.parquet",
        f"results/variant_scores/{config['gpn_star_m']}.parquet",
        f"results/variant_scores/{config['gpn_star_v']}.parquet",
    output:
        "results/test.parquet",
    run:
        dfs = [pl.read_parquet(f) for f in input]
        dfs[3] = dfs[3].rename({"score": "GPN-STAR-P"})
        dfs[4] = dfs[4].rename({"score": "GPN-STAR-M"})
        dfs[5] = dfs[5].rename({"score": "GPN-STAR-V"})
        assert len(set(len(df) for df in dfs)) == 1
        df = pl.concat(dfs, how="horizontal")
        df.write_parquet(output[0])
