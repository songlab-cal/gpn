rule rice_download_variants:
    output:
        "results/variants/msu7/genotypes.vcf.gz",
    shell:
        "wget -O {output} https://ricevarmap.ncpgr.cn/media/Genotypes/Imputated_genotypes/rice4k_geno_no_del.vcf.gz"


# for some mysterious reason the command outputs a lot of stuff to the terminal
# when there are newlines between the different subcommands in the pipeline
rule rice_process_variants_1:
    input:
        "results/variants/msu7/genotypes.vcf.gz",
    output:
        "results/variants/msu7/all.tsv.gz",
    shell:
        r"""
        echo '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">' | bcftools annotate -h - {input} -Ou | bcftools view -f PASS,. -v snps -m2 -M2 -Ou | bcftools +fill-tags - -Ou -- -t AC,AN,AF | bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t%INFO/AC\t%INFO/AN\t%INFO/AF\n' - | gzip > {output}
        """


rule rice_process_variants_2:
    input:
        "results/variants/msu7/all.tsv.gz",
    output:
        "results/variants/msu7/all.parquet",
    run:
        import polars as pl

        V = (
            pl.read_csv(
                input[0],
                has_header=False,
                separator="\t",
                new_columns=COORDINATES + ["AC", "AN", "AF"],
            )
            .filter(pl.col("AF") != 0, pl.col("AF") != 1)
            .with_columns(
                pl.col("chrom").str.replace("chr0", "").str.replace("chr", "")
            )
        )
        print(V)
        V.write_parquet(output[0])


# 4,726 accessions
# max possible AN: 2x4726 = 9452
# In [5]: V["AN"].describe()
# Out[5]:
# shape: (9, 2)
# ┌────────────┬─────────────┐
# │ statistic  ┆ value       │
# │ ---        ┆ ---         │
# │ str        ┆ f64         │
# ╞════════════╪═════════════╡
# │ count      ┆ 1.302961e7  │
# │ null_count ┆ 0.0         │
# │ mean       ┆ 8219.509975 │
# │ std        ┆ 1980.559461 │
# │ min        ┆ 6.0         │
# │ 25%        ┆ 7772.0      │
# │ 50%        ┆ 9328.0      │
# │ 75%        ┆ 9448.0      │
# │ max        ┆ 9452.0      │
# └────────────┴─────────────┘


# rice also seems to be selfing
# In [4]: V["AC"].describe()
# Out[4]:
# shape: (9, 2)
# ┌────────────┬─────────────┐
# │ statistic  ┆ value       │
# │ ---        ┆ ---         │
# │ str        ┆ f64         │
# ╞════════════╪═════════════╡
# │ count      ┆ 1.302961e7  │
# │ null_count ┆ 0.0         │
# │ mean       ┆ 1252.538808 │
# │ std        ┆ 2075.565008 │
# │ min        ┆ 2.0         │
# │ 25%        ┆ 22.0        │
# │ 50%        ┆ 152.0       │
# │ 75%        ┆ 1340.0      │
# │ max        ┆ 9442.0      │
# └────────────┴─────────────┘


rule rice_filter_variants:
    input:
        "results/variants/msu7/all.parquet",
    output:
        "results/variants/msu7/filt.parquet",
    run:
        import polars as pl

        max_possible_AN = 9452
        min_AN_threshold = 0.75 * max_possible_AN

        V = (
            pl.read_parquet(input[0])
            .filter(pl.col("AN") > min_AN_threshold)
            .with_columns(
                pl.when(pl.col("AC") == 2)
                .then(True)
                .when(pl.col("AF") > 20 / 100)
                .then(False)
                .otherwise(None)
                .alias("label")
            )
            .drop_nulls(subset="label")
        )
        print(V["label"].value_counts())
        V.write_parquet(output[0])


rule rice_conservation:
    input:
        expand(
            "results/variant_scores/msu7/filt/{model}.parquet",
            model=[
                "phastCons9way",
                "phyloP9way",
            ],
        ),
