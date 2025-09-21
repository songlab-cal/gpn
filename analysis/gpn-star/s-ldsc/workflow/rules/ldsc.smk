rule download_baseline:
    output:
        directory("results/baselineLD_v2.2"),
    shell:
        """
        mkdir -p {output} && cd {output} &&
        wget -O https://zenodo.org/records/10515792/files/1000G_Phase3_baselineLD_v2.2_ldscores.tgz?download=1 &&
        tar xvzf 1000G_Phase3_baselineLD_v2.2_ldscores.tgz?download=1 &&
        rm 1000G_Phase3_baselineLD_v2.2_ldscores.tgz?download=1
        """


# rule download_sumstats:
# wget https://zenodo.org/records/10515792/files/sumstats_indep107.tgz?download=1
# tar xvzf sumstats_indep107.tgz?download=1
# will be in sumstats_107 directory
# check out sumstats_107/readme.txt,traits_indep107.txt,traits_indep107.xlsx

# frqfile
# https://zenodo.org/records/10515792/files/1000G_Phase3_frq.tgz?download=1
# tar xzvf 1000G_Phase3_frq.tgz?download=1
# creates dir 1000G_Phase3_frq

# weights
# https://zenodo.org/records/10515792/files/1000G_Phase3_weights_hm3_no_MHC.tgz?download=1
# creates dir  1000G_Phase3_weights_hm3_no_MHC

# plink files (needed for custom annotations)
# wget https://zenodo.org/records/10515792/files/1000G_Phase3_plinkfiles.tgz?download=1
# tar xzvf 1000G_Phase3_plinkfiles.tgz?download=1
# will create dir 1000G_EUR_Phase3_plink


# hm3 list
# wget https://zenodo.org/records/10515792/files/hm3_no_MHC.list.txt?download=1


# --use-conda --conda-frontend mamba
rule test_ldsc:
    output:
        touch("results/test_ldsc2.done"),
    conda:
        "../envs/ldsc.yaml"
    threads: workflow.cores
    shell:
        "ldsc.py -h"


rule run_ldsc_base:
    output:
        "results/output/base/{trait}.results",
        "results/output/base/{trait}.log",
    params:
        "results/output/base/{trait}",
    conda:
        "../envs/ldsc.yaml"
    threads: workflow.cores
    shell:
        """
        ldsc.py \
        --h2 results/sumstats_107/{wildcards.trait}.sumstats.gz \
        --ref-ld-chr results/baselineLD_v2.2/baselineLD. \
        --frqfile-chr results/1000G_Phase3_frq/1000G.EUR.QC. \
        --w-ld-chr results/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC. \
        --overlap-annot --print-coefficients \
        --out {params}
        """


# convert bim to parquet, also hg19 to hg38
# need to keep exact order
# also find ref and alt
# maybe put pos = -1 for failures (in case other columns will have no guarantees)
rule process_variants:
    input:
        "results/1000G_EUR_Phase3_plink/1000G.EUR.QC.{chrom}.bim",
        "results/genome.fa.gz",
    output:
        "results/variants/{chrom}.parquet",
    wildcard_constraints:
        chrom="|".join(CHROMS),
    run:
        chrom = wildcards.chrom
        V = pd.read_csv(
            input[0],
            sep="\t",
            header=None,
            names=["chrom", "rsid", "cm", "pos", "ref", "alt"],
            usecols=COORDINATES,
            dtype={"chrom": str},
        )
        print(len(V))
        assert V.ref.isin(NUCLEOTIDES).all()
        assert V.alt.isin(NUCLEOTIDES).all()
        V = lift_hg19_to_hg38(V)
        print((V.pos == -1).sum())
        V.loc[V.chrom != chrom, "pos"] = -1
        V.chrom = chrom
        print((V.pos == -1).sum())
        genome = Genome(input[1], subset_chroms=[chrom])
        V["ref_nuc"] = V.progress_apply(
            lambda v: genome.get_nuc(v.chrom, v.pos).upper() if v.pos != -1 else "",
            axis=1,
        )
        mask = V["ref"] != V["ref_nuc"]
        V.loc[mask, ["ref", "alt"]] = V.loc[mask, ["alt", "ref"]].values
        V.loc[V["ref"] != V["ref_nuc"], "pos"] = -1
        print((V.pos == -1).sum())
        V.drop(columns=["ref_nuc"], inplace=True)
        print(V)
        V.to_parquet(output[0], index=False)


rule merge_variants:
    input:
        expand("results/variants/{chrom}.parquet", chrom=CHROMS),
    output:
        "results/variants/merged.parquet",
    run:
        V = pd.concat([pd.read_parquet(f) for f in input])
        V.to_parquet(output[0], index=False)


rule filter_merged_variants:
    input:
        "results/variants/merged.parquet",
    output:
        "results/variants/merged_filt.parquet",
    run:
        V = pd.read_parquet(input[0])
        V = V[V.pos != -1]
        V.to_parquet(output[0], index=False)


rule process_maf:
    input:
        "results/1000G_Phase3_frq/1000G.EUR.QC.{chrom}.frq",
    wildcard_constraints:
        chrom="|".join(CHROMS),
    output:
        "results/maf/{chrom}.parquet",
    run:
        df = pd.read_csv(input[0], delim_whitespace=True, usecols=["MAF"])
        df.to_parquet(output[0], index=False)


rule merge_maf:
    input:
        expand("results/maf/{chrom}.parquet", chrom=CHROMS),
    output:
        "results/maf/merged.parquet",
    run:
        V = pd.concat([pd.read_parquet(f) for f in input])
        print(V)
        V.to_parquet(output[0], index=False)


# some special care to handle ties...
# also need to consider the quantile only within MAF > 5%
# to have same Prop._SNPs
rule quantile_score:
    input:
        "results/variant_scores/{model}.parquet",
        "results/maf/merged.parquet",
    output:
        "results/variant_scores/quantile/{model}/{q}.parquet",
    run:
        q = float(wildcards.q)
        V = pd.read_parquet(input[0])
        MAF = pd.read_parquet(input[1])
        V = pd.concat([V, MAF], axis=1)
        V["common"] = V.MAF > 0.05
        print(V)
        n = int(V.common.sum() * q)
        V2 = V.copy()
        V2["idx_V1"] = np.arange(len(V2))
        V2 = V2.sample(frac=1, random_state=42)
        V2 = V2.sort_values("score", ascending=False, kind="stable")
        V2["idx_V2"] = np.arange(len(V2))
        # find index of n'th common variant
        # but index in V2, not V
        nth_common_variant = V2.loc[V2.common, "idx_V2"].iloc[n]
        V2 = V2.head(nth_common_variant)
        V.score = 0
        V.loc[V2.idx_V1, "score"] = 1
        print(V)
        print(V.score.sum())
        print(V[V.common].score.sum())
        V[["score"]].to_parquet(output[0], index=False)


rule quantile_score_noncds:
    input:
        "results/variants/merged.parquet",
        "results/maf/merged.parquet",
        "results/variant_scores/{model}.parquet",
        "results/intervals/CDS/0.parquet",
    output:
        "results/variant_scores/quantile_nonCDS/{model}/{q}.parquet",
    run:
        q = float(wildcards.q)
        V = pd.read_parquet(input[0])
        MAF = pd.read_parquet(input[1])
        score = pd.read_parquet(input[2])
        V = pd.concat([V, MAF, score], axis=1)
        V["common"] = V.MAF > 0.05
        CDS = pd.read_parquet(input[3])
        V["start"] = V.pos - 1
        V["end"] = V.pos
        V = bf.coverage(V, CDS)
        # ensure CDS variants are not picked
        V.loc[V.coverage > 0, "score"] = V.score.min() - 1
        print(V.coverage.value_counts())
        V = V.drop(columns=["start", "end", "coverage"])
        print(V)
        n = int(V.common.sum() * q)
        V2 = V.copy()
        V2["idx_V1"] = np.arange(len(V2))
        V2 = V2.sample(frac=1, random_state=42)
        V2 = V2.sort_values("score", ascending=False, kind="stable")
        V2["idx_V2"] = np.arange(len(V2))
        # find index of n'th common variant
        # but index in V2, not V
        nth_common_variant = V2.loc[V2.common, "idx_V2"].iloc[n]
        V2 = V2.head(nth_common_variant)
        V.score = 0
        V.loc[V2.idx_V1, "score"] = 1
        print(V)
        print(V.score.sum())
        print(V[V.common].score.sum())
        V[["score"]].to_parquet(output[0], index=False)


rule quantile_score_cds:
    input:
        "results/variants/merged.parquet",
        "results/maf/merged.parquet",
        "results/variant_scores/{model}.parquet",
        "results/intervals/CDS/0.parquet",
    output:
        "results/variant_scores/quantile_CDS/{model}/{q}.parquet",
    run:
        q = float(wildcards.q)
        V = pd.read_parquet(input[0])
        MAF = pd.read_parquet(input[1])
        score = pd.read_parquet(input[2])
        V = pd.concat([V, MAF, score], axis=1)
        V["common"] = V.MAF > 0.05
        CDS = pd.read_parquet(input[3])
        V["start"] = V.pos - 1
        V["end"] = V.pos
        V = bf.coverage(V, CDS)
        # ensure non-CDS variants are not picked
        V.loc[V.coverage == 0, "score"] = V.score.min() - 1
        print(V.coverage.value_counts())
        V = V.drop(columns=["start", "end", "coverage"])
        print(V)
        n = int(V.common.sum() * q)
        V2 = V.copy()
        V2["idx_V1"] = np.arange(len(V2))
        V2 = V2.sample(frac=1, random_state=42)
        V2 = V2.sort_values("score", ascending=False, kind="stable")
        V2["idx_V2"] = np.arange(len(V2))
        # find index of n'th common variant
        # but index in V2, not V
        nth_common_variant = V2.loc[V2.common, "idx_V2"].iloc[n]
        V2 = V2.head(nth_common_variant)
        V.score = 0
        V.loc[V2.idx_V1, "score"] = 1
        print(V)
        print(V.score.sum())
        print(V[V.common].score.sum())
        V[["score"]].to_parquet(output[0], index=False)


rule variant_scores_fill_null:
    input:
        "results/variant_scores/{model}.parquet",
    output:
        "results/variant_scores/{model}_fillnull.parquet",
    run:
        (pl.read_parquet(input[0]).fill_null(strategy="mean").write_parquet(output[0]))


rule variant_scores_minmax:
    input:
        "results/variant_scores/{model}.parquet",
    output:
        "results/variant_scores/{model}_minmax.parquet",
    run:
        score = pl.read_parquet(input[0])["score"].to_numpy()
        df = pl.DataFrame(
            {
                "score": MinMaxScaler()
                .fit_transform(np.expand_dims(score, axis=1))
                .squeeze()
            }
        )
        df.write_parquet(output[0])


rule variant_scores_split:
    input:
        "results/variants/merged.parquet",
        "results/variant_scores/{anything}.parquet",
    output:
        expand(
            "results/variant_scores_by_chrom/{{anything}}/{chrom}.annot.gz",
            chrom=CHROMS,
        ),
    run:
        V = pd.read_parquet(input[0])
        scores = pd.read_parquet(input[1])
        for chrom, path in zip(CHROMS, output):
            chrom_mask = (V.chrom == chrom).values
            scores[chrom_mask].to_csv(path, sep="\t", index=False, compression="gzip")


rule ld_score:
    input:
        "results/variant_scores_by_chrom/{anything}/{chrom}.annot.gz",
    output:
        # there's actually other files
        "results/variant_scores_by_chrom/{anything}/{chrom}.l2.ldscore.gz",
    params:
        "results/variant_scores_by_chrom/{anything}/{chrom}",
    conda:
        "../envs/ldsc.yaml"
    shell:
        """
        ldsc.py \
        --l2 \
        --bfile results/1000G_EUR_Phase3_plink/1000G.EUR.QC.{wildcards.chrom} \
        --ld-wind-cm 1 \
        --annot {input[0]} \
        --thin-annot \
        --out {params} \
        --print-snps results/hm3_no_MHC.list.txt
        """


rule run_ldsc_annot:
    input:
        expand(
            "results/variant_scores_by_chrom/{{annot}}/{chrom}.l2.ldscore.gz",
            chrom=CHROMS,
        ),
    output:
        "results/output/{annot}/{trait}.results",
        "results/output/{annot}/{trait}.log",
    params:
        annot="results/variant_scores_by_chrom/{annot}/",
        out="results/output/{annot}/{trait}",
    conda:
        "../envs/ldsc.yaml"
    shell:
        """
        ldsc.py \
        --h2 results/sumstats_107/{wildcards.trait}.sumstats.gz \
        --ref-ld-chr results/baselineLD_v2.2/baselineLD.,{params.annot} \
        --frqfile-chr results/1000G_Phase3_frq/1000G.EUR.QC. \
        --w-ld-chr results/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC. \
        --overlap-annot --print-coefficients \
        --out {params.out}
        """


rule find_regression_snps:
    output:
        "results/regression_snps.parquet",
    run:
        dfs = []
        for chrom in tqdm(CHROMS):
            V_ref = pd.read_table(
                f"results/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC.{chrom}.l2.ldscore.gz"
            )
            V_maf = pd.read_table(
                f"results/1000G_Phase3_frq/1000G.EUR.QC.{chrom}.frq", sep="\s+"
            )
            V_maf = V_maf.loc[V_maf.MAF > 0.05]
            V_used = V_ref[["SNP"]].merge(V_maf[["SNP"]], on="SNP", how="inner")
            V_used["used"] = True
            V_annot = pd.read_table(
                f"results/baselineLD_v2.2/baselineLD.{chrom}.annot.gz", usecols=["SNP"]
            )
            V_annot = V_annot.merge(V_used, on="SNP", how="left")
            V_annot.used = V_annot.used.fillna(False)
            dfs.append(V_annot)
        df = pd.concat(dfs)
        print(df)
        df.to_parquet(output[0], index=False)


rule n_regression_and_annot_variants:
    input:
        "results/regression_snps.parquet",
        "results/variant_scores/{annot}.parquet",
    output:
        "results/n_regression_and_annot_variants/{annot}.parquet",
    run:
        regression_snps = pd.read_parquet(input[0])["used"]
        annot_snps = pd.read_parquet(input[1])["score"] == 1
        n_variants = regression_snps.sum()
        n_variants_annot = (regression_snps & annot_snps).sum()
        res = pd.DataFrame(
            dict(n_variants=[n_variants], n_variants_annot=[n_variants_annot])
        )
        res.to_parquet(output[0], index=False)


rule ldsc_process_results:
    input:
        "results/n_regression_and_annot_variants/{annot}.parquet",
        "results/output/{annot}/{trait}.results",
        "results/output/{annot}/{trait}.log",
    output:
        "results/output/{annot}/{trait}.parquet",
    run:
        n_var = pd.read_parquet(input[0])
        n_variants = n_var.n_variants.iloc[0]
        n_variants_annot = n_var.n_variants_annot.iloc[0]
        df = pd.read_csv(input[1], sep="\t")
        M, h2 = parse_log(input[2])
        df = add_tau_star(df, M, h2, n_variants_annot / n_variants)
        df = df[df.Category == "L2_1"]
        df.to_parquet(output[0], index=False)
