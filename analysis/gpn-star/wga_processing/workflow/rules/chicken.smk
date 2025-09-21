rule chicken_download_snp:
    output:
        temp("results/variants/galGal6/snp.tsv.gz"),
    shell:
        "wget -O {output} http://animal.omics.pro/code/source/download/Chicken/variation/GRCg6a_SNPs.anno.tab.gz"


rule chicken_process_snp:
    input:
        "results/variants/galGal6/snp.tsv.gz",
    output:
        "results/variants/galGal6/snp.parquet",
    run:
        import polars as pl

        V = (
            pl.read_csv(
                input[0],
                separator="\t",
                columns=["Chrom", "Pos", "Alleles", "minAllele", "MAF"],
                schema_overrides={"Chrom": str},
            )
            .rename({"Chrom": "chrom", "Pos": "pos"})
            .with_columns(
                pl.col("Alleles")
                .str.split_exact("/", 1)
                .struct.rename_fields(["ref", "alt"])
                .alias("fields")
            )
            .unnest("fields")
            .with_columns(
                pl.when(pl.col("alt") == pl.col("minAllele"))
                .then(pl.col("MAF"))
                .otherwise(1 - pl.col("MAF"))
                .alias("AF")
            )
            .select(["chrom", "pos", "ref", "alt", "AF"])
        )
        print(V)
        V.write_parquet(output[0])


rule chicken_conservation:
    input:
        expand(
            "results/variant_scores/galGal6/snp.annot/{model}.parquet",
            model=[
                "phastCons77way",
                "phyloP77way",
            ],
        ),
