rule download_cadd_all:
    output:
        "results/cadd/all.tsv.gz",
        "results/cadd/all.tsv.gz.tbi",
    shell:
        """
        wget https://krishna.gs.washington.edu/download/CADD/v1.7/GRCh38/whole_genome_SNVs.tsv.gz -O {output[0]} &&
        wget https://krishna.gs.washington.edu/download/CADD/v1.7/GRCh38/whole_genome_SNVs.tsv.gz.tbi -O {output[1]}
        """


ruleorder: cadd_extract_chrom > parquet_to_tsv


rule cadd_extract_chrom:
    input:
        "results/cadd/all.tsv.gz",
    output:
        temp("results/cadd/chrom/{chrom}.tsv.gz"),
    wildcard_constraints:
        chrom="|".join(CHROMS),
    shell:
        "tabix {input} {wildcards.chrom} | gzip -c > {output}"


rule cadd_process_chrom:
    input:
        "results/cadd/chrom/{chrom}.tsv.gz",
    output:
        "results/cadd/chrom/{chrom}.parquet",
    wildcard_constraints:
        chrom="|".join(CHROMS),
    run:
        (
            pl.read_csv(
                input[0],
                has_header=False,
                separator="\t",
                columns=[0, 1, 2, 3, 4],
                new_columns=COORDINATES + ["score"],
                dtypes={"chrom": str, "score": pl.Float32},
            )
            .with_columns(-pl.col("score"))  # negate score
            .write_parquet(output[0])
        )


# rule prepare_tabix_input:
#    output:
#        temp("results/preds/{dataset}/tabix.input.tsv.gz"),
#    run:
#        df = load_dataset(wildcards["dataset"], split="test").to_pandas()
#        df["start"] = df.pos
#        df["end"] = df.start
#        df = df[["chrom", "start", "end"]].drop_duplicates()
#        df.to_csv(output[0], sep="\t", index=False, header=False)
#
#
# rule run_tabix_CADD:
#    input:
#        "results/cadd/all.tsv.gz",
#        "{anything}/tabix.input.tsv.gz",
#    output:
#        temp("{anything}/tabix.output.tsv"),
#    shell:
#        "tabix {input[0]} -R {input[1]} > {output}"
#
#
# rule process_tabix_output_CADD:
#    input:
#        "results/preds/{dataset}/tabix.output.tsv",
#    output:
#        "results/preds/{dataset}/CADD.{score,RawScore|PHRED}.parquet",
#    run:
#        cols = ["chrom", "pos", "ref", "alt"]
#        df1 = load_dataset(wildcards["dataset"], split="test").to_pandas()[cols]
#        print(df1)
#        df2 = pd.read_csv(
#            input[0], sep="\t", header=None,
#            names=["chrom", "pos", "ref", "alt", "RawScore", "PHRED"],
#            usecols=["chrom", "pos", "ref", "alt", wildcards.score],
#            dtype={"chrom": str},
#        )
#        df2["score"] = -df2[wildcards.score]
#        df2.drop(columns=[wildcards.score], inplace=True)
#        print(df2)
#        df = df1.merge(df2, how="left", on=cols)
#        print(df)
#        df.to_parquet(output[0], index=False)
#
#
# ruleorder: run_vep_cadd_in_memory > process_tabix_output_CADD


rule run_vep_cadd_in_memory:
    input:
        "{dataset}/test.parquet",
        expand("results/cadd/chrom/{chrom}.parquet", chrom=CHROMS),
    output:
        "results/preds/{dataset}/hg38/CADD.RawScore.parquet",
    threads: workflow.cores
    run:
        V = pl.read_parquet(input[0], columns=COORDINATES)
        preds = pl.concat(
            [
                pl.read_parquet(path).join(V, on=COORDINATES, how="inner")
                for path in tqdm(input[1:])
            ]
        ).unique(subset=COORDINATES)
        V = V.join(preds, on=COORDINATES, how="left")
        print(V)
        V.select("score").write_parquet(output[0])
