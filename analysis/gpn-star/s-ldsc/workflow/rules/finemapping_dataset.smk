rule finemapping_intermediate_dataset:
    input:
        TRAITGYM_PATH + "results/gwas/processed.parquet",
        TRAITGYM_PATH + "results/ldscore/UKBB.EUR.ldscore.annot_with_cre.parquet",
        "results/variants/merged_filt.parquet",
        "results/features/TSS_dist.parquet",
        "results/features/closest_gene.parquet",
    output:
        "results/finemapping/intermediate_dataset.parquet",
    run:
        V = (
            pl.read_parquet(input[0])
            .with_columns(
                pl.when(pl.col("pip") > 0.9)
                .then(True)
                .when(pl.col("pip") < 0.01)
                .then(False)
                .otherwise(None)
                .alias("label")
            )
            .drop_nulls()
        )
        annot = pl.read_parquet(input[1])
        V = V.join(annot, on=COORDINATES, how="inner")
        V_ldsc = pl.concat(
            [
                pl.read_parquet(input[2]),
                pl.read_parquet(input[3]).rename({"score": "tss_dist"}),
                pl.read_parquet(input[4]),
            ],
            how="horizontal",
        )
        V = V.join(V_ldsc, on=COORDINATES, how="inner")
        print(V)
        V.write_parquet(output[0])


rule finemapping_dataset:
    input:
        "results/finemapping/intermediate_dataset.parquet",
    output:
        "results/finemapping/dataset/{matching_features}/{k,\d+}/test.parquet",
    run:
        k = int(wildcards.k)
        matching_features = config["matching_features"][wildcards.matching_features]
        continuous = matching_features["continuous"]
        categorical = matching_features["categorical"]
        V = pd.read_parquet(input[0])
        for f in continuous:
            V[f"{f}_scaled"] = RobustScaler().fit_transform(V[f].values.reshape(-1, 1))
        continuous_scaled = [f"{f}_scaled" for f in continuous]
        V = match_features(V[V.label], V[~V.label], continuous_scaled, categorical, k)
        V = V.drop(columns=continuous_scaled)
        V = sort_variants(V)
        print(V)
        print(f"{V.label.value_counts()=}")
        print(f"{V.groupby('label')[continuous].median()=}")
        V.to_parquet(output[0], index=False)


rule finemapping_missense:
    input:
        "results/finemapping/intermediate_dataset.parquet",
    output:
        "results/finemapping/missense_maf_matched/test.parquet",
    run:
        V = pd.read_parquet(input[0])
        V = V[V.consequence == "missense_variant"]
        print(f"{V.label.value_counts()=}")
        print(f"{V.groupby('label').maf.median()=}")
        V = maf_match(V)
        V = sort_variants(V)
        print(f"{V.label.value_counts()=}")
        print(f"{V.groupby('label').maf.median()=}")
        V.to_parquet(output[0], index=False)
