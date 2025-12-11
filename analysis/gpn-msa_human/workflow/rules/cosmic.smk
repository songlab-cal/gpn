rule cosmic_process_1:
    input:
        "results/cosmic/Cosmic_MutantCensus_v98_GRCh38.tsv.gz",
    output:
        "results/cosmic/data.parquet",
    run:
        df = pd.read_csv(
            input[0],
            sep="\t",
            converters={"CHROMOSOME": str, "GENOME_START": str},
            usecols=[
                "CHROMOSOME",
                "GENOME_START",
                "GENOMIC_WT_ALLELE",
                "GENOMIC_MUT_ALLELE",
                "COSMIC_SAMPLE_ID",
                "GENOMIC_MUTATION_ID",
                "MUTATION_DESCRIPTION",
            ],
        ).rename(
            columns={
                "CHROMOSOME": "chrom",
                "GENOME_START": "pos",
                "GENOMIC_WT_ALLELE": "ref",
                "GENOMIC_MUT_ALLELE": "alt",
                "MUTATION_DESCRIPTION": "consequence",
            }
        )
        df = df[df.chrom.isin(CHROMS)]
        df.pos = df.pos.astype(int)
        df = df[(df.ref.str.len() == 1) & (df.alt.str.len() == 1)]
        df = df[df.consequence.str.contains("missense")]
        df = df.drop_duplicates(["GENOMIC_MUTATION_ID", "COSMIC_SAMPLE_ID"])
        print(df.shape)
        df.to_parquet(output[0], index=False)


rule cosmic_count_samples:
    input:
        "results/cosmic/data.parquet",
        "results/cosmic/Cosmic_Sample_v98_GRCh38.tsv.gz",
    output:
        "results/cosmic/variants.parquet",
    run:
        df = pd.read_parquet(input[0])
        samples = pd.read_csv(input[1], sep="\t")
        print(samples.shape)
        samples = samples[
            (samples["WHOLE_GENOME_SCREEN"] == "y")
            | (samples["WHOLE_EXOME_SCREEN"] == "y")
        ]
        # WGS: 6233
        # WES: 37294
        print(samples.shape)
        df = df[df.COSMIC_SAMPLE_ID.isin(samples.COSMIC_SAMPLE_ID)]
        df = (
            df.groupby(
                ["chrom", "pos", "ref", "alt", "GENOMIC_MUTATION_ID", "consequence"]
            )
            .size()
            .rename("n_samples")
            .reset_index()
        )
        df["total_samples"] = len(samples)
        df["freq"] = df.n_samples / df.total_samples
        print(df)
        df.to_parquet(output[0], index=False)


rule filter_cosmic:
    input:
        "results/cosmic/variants.parquet",
    output:
        "results/cosmic/filt/test.parquet",
    run:
        df = pd.read_parquet(input[0])
        df = df[df.freq > config["cosmic_min_freq"]]
        chrom_order = [str(i) for i in range(1, 23)] + ["X", "Y"]
        df["chrom"] = pd.Categorical(df["chrom"], categories=chrom_order, ordered=True)
        df = df.sort_values(["chrom", "pos"])
        df.chrom = df.chrom.astype(str)
        print(df)
        df.to_parquet(output[0], index=False)
