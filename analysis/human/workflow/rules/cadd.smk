rule download_cadd_all:
    output:
        "results/cadd/all.tsv.gz",
        "results/cadd/all.tsv.gz.tbi",
    shell:
        """
        wget https://krishna.gs.washington.edu/download/CADD/v1.6/GRCh38/whole_genome_SNVs.tsv.gz -O {output[0]} &&
        wget https://krishna.gs.washington.edu/download/CADD/v1.6/GRCh38/whole_genome_SNVs.tsv.gz.tbi -O {output[1]}
        """


rule prepare_tabix_input:
    output:
        temp("results/preds/{dataset}/tabix.input.tsv.gz"),
    run:
        df = load_dataset(wildcards["dataset"], split="test").to_pandas()
        df["start"] = df.pos
        df["end"] = df.start
        df = df[["chrom", "start", "end"]].drop_duplicates()
        df.to_csv(output[0], sep="\t", index=False, header=False)


rule run_tabix_CADD:
    input:
        "results/cadd/all.tsv.gz",
        "{anything}/tabix.input.tsv.gz",
    output:
        temp("{anything}/tabix.output.tsv"),
    shell:
        "tabix {input[0]} -R {input[1]} > {output}"


rule process_tabix_output_CADD:
    input:
        "results/preds/{dataset}/tabix.output.tsv",
    output:
        "results/preds/{dataset}/CADD.{score,RawScore|PHRED}.parquet",
    run:
        cols = ["chrom", "pos", "ref", "alt"]
        df1 = load_dataset(wildcards["dataset"], split="test").to_pandas()[cols]
        print(df1)
        df2 = pd.read_csv(
            input[0], sep="\t", header=None,
            names=["chrom", "pos", "ref", "alt", "RawScore", "PHRED"],
            usecols=["chrom", "pos", "ref", "alt", wildcards.score],
            dtype={"chrom": str},
        )
        df2["score"] = -df2[wildcards.score]
        df2.drop(columns=[wildcards.score], inplace=True)
        print(df2)
        df = df1.merge(df2, how="left", on=cols)
        print(df)
        df.to_parquet(output[0], index=False)
