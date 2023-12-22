quant_methods = [
    "ge",  # eQTL
    "leafcutter",  # sQTL
]

# https://github.com/eQTL-Catalogue/eQTL-Catalogue-resources/blob/master/tabix/tabix_ftp_paths.tsv
metadata_eqtl = pd.read_csv("config/tabix_ftp_paths.tsv", sep="\t")
metadata_eqtl = metadata_eqtl[
    (metadata_eqtl.study_label == "GTEx") &
    (metadata_eqtl.quant_method.isin(quant_methods))
]

# https://github.com/eQTL-Catalogue/eQTL-Catalogue-resources/issues/35
# As far as I understand, .cc files only contain the molecular traits with at least
# one significant association
metadata_eqtl['ftp_path'] = metadata_eqtl['ftp_path'].str.replace(".all.tsv.gz", ".cc.tsv.gz")

sample_groups = metadata_eqtl.sample_group.unique()
metadata_eqtl.set_index(["sample_group", "quant_method"], inplace=True)


rule etql_download_sumstats:
    output:
        temp("results/eqtl/{sample_group}/{quant_method}/sumstats.tsv.gz"),
    params:
        lambda wildcards: metadata_eqtl.loc[(wildcards.sample_group, wildcards.quant_method), "ftp_path"],
    shell:
        "wget -O {output} {params}"


rule etql_download_credible_sets:
    output:
        temp("results/eqtl/{sample_group}/{quant_method}/credible_sets.tsv.gz"),
    params:
        lambda wildcards: metadata_eqtl.loc[(wildcards.sample_group, wildcards.quant_method), "ftp_cs_path"],
    shell:
        "wget -O {output} {params}"


# for now we only want MAF
rule eqtl_process_sumstats:
    input:
        "results/eqtl/{sample_group}/{quant_method}/sumstats.tsv.gz",
    output:
        temp("results/eqtl/{sample_group}/{quant_method}/sumstats.parquet"),
    run:
        V = pd.read_csv(
            input[0], sep="\t", dtype={"chromosome": str},
            usecols=["chromosome", "position", "ref", "alt", "maf"]
        ).rename(
            columns={"chromosome": "chrom", "position": "pos"}
        )
        V = filter_snp(V)
        V = V.groupby(COORDINATES).maf.mean().reset_index()
        V.to_parquet(output[0], index=False)


# for now we only want pip
rule eqtl_process_credible_sets:
    input:
        "results/eqtl/{sample_group}/{quant_method}/credible_sets.tsv.gz",
    output:
        temp("results/eqtl/{sample_group}/{quant_method}/credible_sets.parquet"),
    run:
        V = pd.read_csv(input[0], sep="\t", usecols=["variant", "pip"])
        V = V.groupby("variant").pip.max().reset_index()
        V[COORDINATES] = V.variant.str.split("_", expand=True)
        V.drop(columns="variant", inplace=True)
        V.chrom = V.chrom.str.replace("chr", "")
        V.pos = V.pos.astype(int)
        V = filter_snp(V)
        V[COORDINATES + ["pip"]].to_parquet(output[0], index=False)


rule eqtl_merge_sumstats_and_credible_sets:
    input:
        "results/eqtl/{sample_group}/{quant_method}/credible_sets.parquet",
        "results/eqtl/{sample_group}/{quant_method}/sumstats.parquet",
    output:
        "results/eqtl/{sample_group}/{quant_method}/merged.parquet",
    wildcard_constraints:
        sample_group="|".join(sample_groups),
        quant_method="|".join(quant_methods),
    run:
        V1 = pd.read_parquet(input[0])
        V2 = pd.read_parquet(input[1])
        V = V1.merge(V2, on=COORDINATES, how="inner")
        V.to_parquet(output[0], index=False)


rule eqtl_merge_sample_groups:
    input:
        expand("results/eqtl/{sample_group}/{{quant_method}}/merged.parquet", sample_group=sample_groups),
    output:
        "results/eqtl/merged/{quant_method}/merged.parquet",
    run:
        V = pd.concat([pd.read_parquet(f) for f in input], ignore_index=True)
        V = V.groupby(COORDINATES).agg({"pip": "max", "maf": "mean"}).reset_index()
        V = V[(V.pip > 0.9) | (V.pip < 0.01)]
        V = sort_chrom_pos(V)
        V.to_parquet(output[0], index=False)


rule eqtl_match:
    input:
        "results/eqtl/merged/{quant_method}/merged.parquet",
    output:
        "results/eqtl/matched/{quant_method}/test.parquet",
    run:
        V = pd.read_parquet(input[0])
        V["label"] = V.pip > 0.9
        print(V.label.value_counts())
        V = match_columns(V, "label", ["maf"])
        print(V.label.value_counts())
        V.to_parquet(output[0], index=False)
