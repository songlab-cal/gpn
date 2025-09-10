rule download_gpnmsa_all:
    output:
        "results/gpnmsa/all.tsv.gz",
        "results/gpnmsa/all.tsv.gz.tbi",
    shell:
        """
        wget https://huggingface.co/datasets/songlab/gpn-msa-hg38-scores/resolve/main/scores.tsv.bgz -O {output[0]} &&
        wget https://huggingface.co/datasets/songlab/gpn-msa-hg38-scores/resolve/main/scores.tsv.bgz.tbi -O {output[1]}
        """


ruleorder: gpnmsa_extract_chrom > parquet_to_tsv


rule gpnmsa_extract_chrom:
    input:
        "results/gpnmsa/all.tsv.gz",
    output:
        temp("results/gpnmsa/chrom/{chrom}.tsv.gz"),
    wildcard_constraints:
        chrom="|".join(CHROMS),
    shell:
        "tabix {input} {wildcards.chrom} | gzip -c > {output}"
        

rule gpnmsa_process_chrom:
    input:
        "results/gpnmsa/chrom/{chrom}.tsv.gz",
    output:
        "results/gpnmsa/chrom/{chrom}.parquet",
    wildcard_constraints:
        chrom="|".join(CHROMS),
    run:
        (
            pl.read_csv(
                input[0], has_header=False, separator="\t", columns=[0, 1, 2, 3, 4],
                new_columns=COORDINATES + ["score"],
                dtypes={"chrom": str, "score": pl.Float32},
            )
            .with_columns(pl.col("score"))
            .write_parquet(output[0])
        )

rule run_vep_gpnmsa_in_memory:
    input:
        "{dataset}/test.parquet",
        expand("results/gpnmsa/chrom/{chrom}.parquet", chrom=CHROMS),
    output:
        "results/preds/{dataset}/hg38/GPN-MSA.parquet",
    threads: workflow.cores
    run:
        V = pl.read_parquet(input[0], columns=COORDINATES)
        preds = pl.concat([
            pl.read_parquet(path).join(V, on=COORDINATES, how="inner")
            for path in tqdm(input[1:])
        ]).unique(subset=COORDINATES)
        V = V.join(preds, on=COORDINATES, how="left")
        print(V)
        V.select("score").write_parquet(output[0])