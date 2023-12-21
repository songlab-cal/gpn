quant_methods = [
    "ge",  # eQTL
    #"leafcutter",  # sQTL
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
metadata_eqtl.ftp_path.replace(".all.tsv.gz", ".cc.tsv.gz", inplace=True)

sample_groups = metadata_eqtl.sample_group.unique()
sample_groups = sample_groups[:1]  # just to iterate faster

metadata_eqtl.set_index(["sample_group", "quant_method"], inplace=True)


rule etql_download_sumstats:
    output:
        "results/eqtl/{sample_group}/{quant_method}/sumstats.tsv.gz",  # TODO: make temp
    params:
        lambda wildcards: metadata_eqtl.loc[(wildcards.sample_group, wildcards.quant_method), "ftp_path"],
    shell:
        "wget -O {output} {params}"


rule etql_download_credible_sets:
    output:
        "results/eqtl/{sample_group}/{quant_method}/credible_sets.tsv.gz",  # TODO: make temp
    params:
        lambda wildcards: metadata_eqtl.loc[(wildcards.sample_group, wildcards.quant_method), "ftp_cs_path"],
    shell:
        "wget -O {output} {params}"


# for now we only want MAF
rule eqtl_process_sumstats:
    input:
        "results/eqtl/{sample_group}/{quant_method}/sumstats.tsv.gz",
    output:
        "results/eqtl/{sample_group}/{quant_method}/sumstats.parquet",
    run:
        V = pd.read_csv(
            input[0], sep="\t", dtype={"chromosome": str},
            columns=["chromosome", "position", "ref", "alt", "maf"]
        ).rename(
            columns={"chromosome": "chrom", "position": "pos"}
        ).drop_duplicates(subset=COORDINATES)
        V = filter_snp(V)
        print(V)
        V.to_parquet(output[0], index=False)
