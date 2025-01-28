ruleorder: spliceai_extract_chrom > parquet_to_tsv


rule spliceai_extract_chrom:
    input:
        "results/spliceai/spliceai_scores.masked.snv.hg38.vcf.gz",  # downloaded from their website
    output:
        temp("results/spliceai/chrom/{chrom}.tsv"),
    wildcard_constraints:
        chrom="|".join(CHROMS),
    shell:
        "tabix {input} {wildcards.chrom} > {output}"


rule spliceai_process_chrom:
    input:
        "results/spliceai/chrom/{chrom}.tsv",
    output:
        "results/spliceai/chrom/{chrom}.parquet",
    wildcard_constraints:
        chrom="|".join(CHROMS),
    threads: workflow.cores // 4
    run:
        # there seems to be consequences in different genes for same SNP, e.g.:
        # 1       11193627        .       G       A       .       .       SpliceAI=A|ANGPTL7|0.00|0.00|0.00|0.00|-22|43|3|18
        # 1       11193627        .       G       A       .       .       SpliceAI=A|MTOR|0.00|0.00|0.00|0.00|-16|35|31|35
        (
            pl.read_csv(
                input[0], has_header=False, separator="\t", columns=[0, 1, 3, 4, 7],
                new_columns=COORDINATES + ["INFO"],
                dtypes={"chrom": str},
            )
            .with_columns(
                -(
                    pl.col("INFO").str.split("|").list.slice(2, 4)
                    .cast(pl.Array(pl.Float32, 4)).arr.max()
                ).alias("score")
            )
            .drop("INFO")
            .group_by(COORDINATES).agg(pl.min("score"))
            .sort(COORDINATES)
            .write_parquet(output[0])
        )


rule run_vep_spliceai_in_memory:
    input:
        "results/{dataset}/test.parquet",
        expand("results/spliceai/chrom/{chrom}.parquet", chrom=CHROMS),
    output:
        "results/preds/results/{dataset}/SpliceAI.parquet",
    threads: workflow.cores
    run:
        V = pl.read_parquet(input[0], columns=COORDINATES)
        preds = pl.concat([
            pl.read_parquet(path).join(V, on=COORDINATES, how="inner")
            for path in tqdm(input[1:])
        ])
        V = V.join(preds, on=COORDINATES, how="left")
        print(V)
        V.select("score").write_parquet(output[0])
