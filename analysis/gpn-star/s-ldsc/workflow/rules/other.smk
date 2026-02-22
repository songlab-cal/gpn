rule download_kanai_et_al:
    output:
        "results/kanai_et_al/supp_tables.xlsx",
    shell:
        "wget --no-check-certificate -O {output} 'https://www.medrxiv.org/content/medrxiv/early/2021/09/05/2021.09.03.21262975/DC2/embed/media-2.xlsx?download=true'"


rule process_kanai_et_al:
    input:
        "results/kanai_et_al/supp_tables.xlsx",
        "results/gwas_effect_sizes/aggregated.parquet",
    output:
        "results/kanai_et_al/high_pip_effect_sizes.parquet",
    run:
        (
            pl.from_pandas(
                pd.read_excel(
                    input[0], sheet_name="ST3_high_PIP_pairs", header=2,
                    usecols=["variant", "rsid", "cohort"],
                )
            )
            .filter(pl.col("cohort") == "UKBB")
            .with_columns(
                pl.col("variant").str.split(":").list.get(2).alias("ref"),
                pl.col("variant").str.split(":").list.get(3).alias("alt"),
            )
            .filter(
                pl.col("ref").is_in(NUCLEOTIDES) & pl.col("alt").is_in(NUCLEOTIDES)
            )
            .select("rsid")
            .unique()
            .join(
                pl.read_parquet(input[1]),
                left_on="rsid", right_on="SNP", how="inner",
            )
            .write_parquet(output[0])
        )


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
        kanai="results/kanai_et_al/high_pip_effect_sizes.parquet",
        bg="results/gwas_effect_sizes/aggregated.parquet",
    output:
        ecdf="results/gwas_effect_sizes/ecdf.svg",
        kde="results/gwas_effect_sizes/kde.svg",
    run:
        hue_col = "variant set"
        cols = ["mean_Z2", "max_Z2", hue_col]
        bg_full = pl.read_parquet(input.bg)
        parts = [
            (pl.read_parquet(input.p), "GPN-Star-P top 0.1%"),
            (pl.read_parquet(input.m), "GPN-Star-M top 0.1%"),
            (pl.read_parquet(input.v), "GPN-Star-V top 0.1%"),
            (pl.read_parquet(input.kanai), "Kanai et al. finemapped high PIP"),
        ]
        df = pl.concat([
            d.with_columns(pl.lit(f"{name}\n(n={len(d):,})").alias(hue_col)).select(cols)
            for d, name in parts
        ] + [
            bg_full.sample(30000, seed=42)
            .with_columns(pl.lit(f"All\n(n={len(bg_full):,})").alias(hue_col))
            .select(cols)
        ])
        for plot_fn, out in [(sns.ecdfplot, output.ecdf), (sns.kdeplot, output.kde)]:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            for i, (ax, col) in enumerate(zip(axes, ["mean_Z2", "max_Z2"])):
                kwargs = {"common_norm": False} if plot_fn is sns.kdeplot else {}
                plot_fn(data=df, x=col, hue=hue_col, log_scale=(True, False), ax=ax, **kwargs)
                ax.set_xlabel(col.replace("_", " ").replace("Z2", "$Z^2$"))
                legend = ax.get_legend()
                if i == 0:
                    handles = legend.legend_handles
                    labels = [t.get_text() for t in legend.get_texts()]
                legend.remove()
            fig.legend(
                handles, labels, loc="upper center", ncol=3,
                bbox_to_anchor=(0.5, 1.0), frameon=False,
            )
            sns.despine()
            fig.subplots_adjust(top=0.78)
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
