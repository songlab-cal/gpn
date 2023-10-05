rule run_tabix_spliceai:
    input:
        "results/spliceai/spliceai_scores.masked.snv.hg38.vcf.gz",  # downloaded from their website
        "{anything}/tabix.input.tsv.gz",
    output:
        temp("{anything}/tabix.output.spliceai.tsv"),
    shell:
        "tabix {input[0]} -R {input[1]} > {output}"


rule process_tabix_output_spliceai:
    input:
        "results/preds/{dataset}/tabix.output.spliceai.tsv",
    output:
        "results/preds/{dataset}/SpliceAI.parquet",
    run:
        cols = ["chrom", "pos", "ref", "alt"]
        df1 = load_dataset(wildcards["dataset"], split="test").to_pandas()[cols]
        df2 = pd.read_csv(
            input[0], sep="\t", header=None,
            names=["chrom", "pos", "id", "ref", "alt", "qual", "filter", "INFO"],
            usecols=["chrom", "pos", "ref", "alt", "INFO"],
            dtype={"chrom": str},
        )

        def get_spliceai_score(INFO):
            deltas = [float(x) for x in INFO.split("|")[2:6]]
            score = -np.max(deltas)
            return score

        df2["score"] = df2.INFO.progress_apply(get_spliceai_score)
        df2.drop(columns=["INFO"], inplace=True)
        # there seems to be consequences in different genes for same SNP, e.g.:
        # 1       11193627        .       G       A       .       .       SpliceAI=A|ANGPTL7|0.00|0.00|0.00|0.00|-22|43|3|18
        # 1       11193627        .       G       A       .       .       SpliceAI=A|MTOR|0.00|0.00|0.00|0.00|-16|35|31|35
        df2 = df2.groupby(cols).score.min().to_frame()
        df = df1.merge(df2, how="left", on=cols)
        print(df)
        df.to_parquet(output[0], index=False)
