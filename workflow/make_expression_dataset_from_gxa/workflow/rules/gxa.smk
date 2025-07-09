rule gxa_download_expression:
    output:
        temp("results/gxa/transcript_expression/{study}.tsv"),
    shell:
        "wget -O {output} http://ftp.ebi.ac.uk/pub/databases/microarray/data/atlas/experiments/{wildcards.study}/{wildcards.study}-transcripts-tpms.tsv"


rule gxa_process_expression:
    input:
        "results/gxa/transcript_expression/{study}.tsv",
    output:
        "results/gxa/transcript_expression/{study}.parquet",
    run:
        (
            pd.read_csv(input[0], sep="\t")
            .rename(columns={"GeneID": "transcript_id"})
            .drop(columns=["Gene ID", "Gene Name"])
            .to_parquet(output[0], index=False)
        )


rule gxa_download_metadata:
    output:
        "results/gxa/metadata/{study}.xml",
    shell:
        "wget -O {output} http://ftp.ebi.ac.uk/pub/databases/microarray/data/atlas/experiments/{wildcards.study}/{wildcards.study}-configuration.xml"


rule gxa_process_metadata:
    input:
        "results/gxa/metadata/{study}.xml",
        "results/gxa/transcript_expression/{study}.parquet",
    output:
        "results/gxa/metadata/{study}.parquet",
    run:
        with open(input[0], 'r') as file:
            tree = ET.parse(file)
            root = tree.getroot()
        metadata = []
        for group in root.findall('.//assay_group'):
            group_label = group.get('label')
            for assay in group.findall('assay'):
                metadata.append([
                    group_label,
                    get_assay_name(assay),
                ])
        metadata = pd.DataFrame(metadata, columns=['assay_group', 'assay'])

        expression = pd.read_parquet(input[1])
        expression_assays = set(expression.columns) - set(["transcript_id"])
        metadata_assays = set(metadata.assay)
        assert expression_assays == metadata_assays, (expression_assays-metadata_assays, metadata_assays-expression_assays)
        metadata.to_parquet(output[0], index=False)


# aggregate expression by TSS
rule gxa_tss_expression:
    input:
        "results/gxa/transcript_expression/{study}.parquet",
        "results/ensembl/tss.parquet",
    output:
        "results/gxa/tss_expression/{study}.parquet",
    run:
        expression = pd.read_parquet(input[0])
        var = expression.var(numeric_only=True)
        exclude_samples = var[var < config["min_variance"]].index.values
        expression.drop(columns=exclude_samples, inplace=True)
        tss = pd.read_parquet(input[1])
        df = expression.merge(tss, how="inner", on="transcript_id").drop(columns="transcript_id")
        df = df.groupby(["chrom", "pos", "strand"]).sum().reset_index()
        df.to_parquet(output[0], index=False)


# average expression across replicates
rule gxa_tss_expression_group:
    input:
        "results/gxa/tss_expression/{study}.parquet",
        "results/gxa/metadata/{study}.parquet",
    output:
        "results/gxa/tss_expression_group/{study}.parquet",
    run:
        df = pd.read_parquet(input[0])
        df["tss"] = df["chrom"] + ":" + df["pos"].astype(str) + ":" + df["strand"]
        df = df.drop(columns=['chrom', 'pos', 'strand']).set_index('tss')
        metadata = pd.read_parquet(input[1])

        # First, let's calculate the average correlation between replicates for each
        # assay group
        # Note this is on all chromosomes not just test chrom, so not perfectly
        # comparable to model results
        chosen_assay_groups = []
        for assay_group in tqdm(metadata.assay_group.unique()):
            cols = list(
                set(df.columns)
                .intersection(set(metadata[metadata.assay_group == assay_group].assay.values))
            )
            if len(cols) < 2:
                print(f"Skipping {assay_group} due to insufficient replicates")
            df2 = df[cols]
            if avg_correlation(np.log1p(df2), "pearson") > config["min_correlation"]:
                chosen_assay_groups.append(assay_group)
        cols = list(
            set(df.columns)
            .intersection(set(metadata[metadata.assay_group.isin(chosen_assay_groups)].assay.values))
        )
        df = df[cols]

        # Transpose the expression dataframe to have samples as rows and genes as columns
        df_transposed = df.transpose()

        # Merge the transposed dataframe with the metadata dataframe on assay name
        merged_df = (
            df_transposed.merge(metadata, left_index=True, right_on='assay')
            .drop(columns=['assay'])
        )

        # Group by assay_group and calculate the mean for each group
        grouped_df = merged_df.groupby('assay_group').mean()

        # Transpose back to the original format
        final_df = grouped_df.transpose().reset_index()

        # If you want to split the index back into separate columns for chrom, pos, and strand
        final_df["chrom"] = final_df["index"].str.split(":").str[0]
        final_df["pos"] = final_df["index"].str.split(":").str[1].astype(int)
        final_df["strand"] = final_df["index"].str.split(":").str[2]
        final_df = final_df.drop(columns=['index'])

        final_df.to_parquet(output[0], index=False)


rule gxa_merge_studies:
    input:
        expand(
            "results/gxa/tss_expression_group/{study}.parquet",
            study=config["gxa"]["studies"]
        ),
    output:
        "results/gxa/merged_tss_expression_group.parquet",
    run:
        dfs = []
        for path, study in zip(input, config["gxa"]["studies"]):
            df = pd.read_parquet(path)
            columns = [study + "_" + col if col not in ["chrom", "pos", "strand"] else col for col in df.columns]
            df.columns = columns
            dfs.append(df)
        df = reduce(lambda x, y: pd.merge(x, y, on=["chrom", "pos", "strand"], how='inner'), dfs)
        df.to_parquet(output[0], index=False)