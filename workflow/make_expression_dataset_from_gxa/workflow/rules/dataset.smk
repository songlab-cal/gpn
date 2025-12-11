rule dataset:
    input:
        "results/gxa/merged_tss_expression_group.parquet",
        "results/ensembl/genome.fa.gz",
    output:
        expand("results/dataset/{split}.parquet", split=SPLITS),
    run:
        n_upstream = config["n_upstream"]
        n_downstream = config["n_downstream"]
        df = pd.read_parquet(input[0])
        genome = Genome(input[1], subset_chroms=df.chrom.unique())

        df["start"] = (df.pos - n_upstream).where(
            df.strand == "+", df.pos - n_downstream
        )
        df["end"] = (df.pos + n_downstream).where(df.strand == "+", df.pos + n_upstream)
        df.drop(columns="pos", inplace=True)

        df["seq"] = df.progress_apply(
            lambda x: genome(x.chrom, x.start, x.end, x.strand),
            axis=1,
        )

        exp_cols = [
            col
            for col in df.columns
            if col not in ["chrom", "start", "end", "strand", "seq"]
        ]
        df[exp_cols] = np.log1p(df[exp_cols]).round(3)
        df["labels"] = df[exp_cols].apply(lambda row: row.tolist(), axis=1)
        df.drop(columns=exp_cols, inplace=True)

        for path, split in zip(output, SPLITS):
            df[df.chrom.isin(SPLIT_CHROMS[split])].to_parquet(path, index=False)


rule dataset_label_metadata:
    input:
        "results/gxa/merged_tss_expression_group.parquet",
    output:
        "results/dataset/labels.txt",
    run:
        cols = [
            col
            for col in pd.read_parquet(input[0]).columns
            if col not in ["chrom", "pos", "strand"]
        ]
        df = pd.DataFrame({"label": cols})
        df.to_csv(output[0], header=False, index=False)
