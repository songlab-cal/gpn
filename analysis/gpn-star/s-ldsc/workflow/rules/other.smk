rule gwas_effect_sizes_agg:
    input:
        expand("results/sumstats_107/{trait}.sumstats.gz", trait=traits),
    output:
        "results/gwas_effect_sizes/aggregated.parquet",
    run:
        (
            pl.concat([
                pl.from_pandas(pd.read_csv(f, sep=r"\s+", usecols=["SNP", "Z"]))
                for f in tqdm(input)
            ])
            .with_columns((pl.col("Z") ** 2).alias("Z2"))
            .group_by("SNP")
            .agg(
                pl.col("Z2").max().alias("max_Z2"),
                pl.col("Z2").mean().alias("mean_Z2"),
            )
            .write_parquet(output[0])
        )


rule gwas_effect_sizes_quantile:
    input:
        "results/gwas_effect_sizes/aggregated.parquet",
        "results/variant_scores/quantile/{model}/{q}.parquet",
        "results/variants/rsid/merged.parquet",
    output:
        "results/gwas_effect_sizes/quantile/{model}/{q}.parquet",
    run:
        scores = pl.read_parquet(input[1])
        rsids = pl.read_parquet(input[2])
        assert len(scores) == len(rsids)
        (
            pl.concat([rsids, scores], how="horizontal")
            .filter(pl.col("score") == 1)
            .join(
                pl.read_parquet(input[0]),
                left_on="rsid", right_on="SNP", how="inner",
            )
            .write_parquet(output[0])
        )


rule gwas_effect_sizes_histogram:
    input:
        p=f"results/gwas_effect_sizes/quantile/{config['gpn_star_p']}/0.001.parquet",
        m=f"results/gwas_effect_sizes/quantile/{config['gpn_star_m']}/0.001.parquet",
        v=f"results/gwas_effect_sizes/quantile/{config['gpn_star_v']}/0.001.parquet",
        bg="results/gwas_effect_sizes/aggregated.parquet",
    output:
        ecdf="results/gwas_effect_sizes/ecdf.svg",
        kde="results/gwas_effect_sizes/kde.svg",
    run:
        cols = ["mean_Z2", "max_Z2", "model"]
        df = pl.concat([
            pl.read_parquet(input.p).with_columns(model=pl.lit("GPN-Star-P")).select(cols),
            pl.read_parquet(input.m).with_columns(model=pl.lit("GPN-Star-M")).select(cols),
            pl.read_parquet(input.v).with_columns(model=pl.lit("GPN-Star-V")).select(cols),
            pl.read_parquet(input.bg).with_columns(model=pl.lit("All")).select(cols).sample(30000, seed=42),
        ])
        for plot_fn, out in [(sns.ecdfplot, output.ecdf), (sns.kdeplot, output.kde)]:
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            for ax, col in zip(axes, ["mean_Z2", "max_Z2"]):
                kwargs = {"common_norm": False} if plot_fn is sns.kdeplot else {}
                plot_fn(data=df, x=col, hue="model", log_scale=(True, False), ax=ax, **kwargs)
                ax.set_xlabel(col.replace("_", " ").replace("Z2", "$Z^2$"))
            sns.despine()
            fig.tight_layout()
            fig.savefig(out)


rule gwas_effect_sizes_test:
    input:
        p=f"results/gwas_effect_sizes/quantile/{config['gpn_star_p']}/0.001.parquet",
        m=f"results/gwas_effect_sizes/quantile/{config['gpn_star_m']}/0.001.parquet",
        bg="results/gwas_effect_sizes/aggregated.parquet",
    output:
        "results/gwas_effect_sizes/mann_whitney_test.parquet",
    run:
        df_p = pl.read_parquet(input.p)
        df_m = pl.read_parquet(input.m)
        df_all = pl.read_parquet(input.bg)
        bg_p = df_all.filter(~pl.col("SNP").is_in(df_p["rsid"]))
        bg_m = df_all.filter(~pl.col("SNP").is_in(df_m["rsid"]))
        comparisons = [
            ("P_vs_M", df_p, df_m),
            ("P_vs_bg", df_p, bg_p),
            ("M_vs_bg", df_m, bg_m),
        ]
        rows = []
        for comparison, df_a, df_b in comparisons:
            for col in ["mean_Z2", "max_Z2"]:
                stat, pval = mannwhitneyu(
                    df_a[col].to_numpy(), df_b[col].to_numpy(),
                    alternative="two-sided",
                )
                rows.append({
                    "comparison": comparison,
                    "metric": col,
                    "median_a": df_a[col].median(),
                    "median_b": df_b[col].median(),
                    "n_a": len(df_a),
                    "n_b": len(df_b),
                    "U": stat,
                    "p_value": pval,
                })
        pl.DataFrame(rows).write_parquet(output[0])
