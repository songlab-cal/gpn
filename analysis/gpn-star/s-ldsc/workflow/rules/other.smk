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


# Trait matching between Kanai et al. (2021) fine-mapped variants and our 107 sumstats.
#
# Kanai et al. ran their own UKBB GWAS using BOLT-LMM (continuous traits) and
# SAIGE (binary traits) on ~360K UKBB Europeans.
#
# Our sumstats come from three sources:
#   - UKB_460K.*: Price lab (Loh et al. 2018, Nat Genet), BOLT-LMM on ~459K UKBB
#     Europeans. Same method and cohort as Kanai, likely very similar analyses
#     (possibly different covariates/QC).
#   - PANUKBB.*: Pan-UKBB (Broad Institute), multi-ancestry mixed model on UKBB.
#     Same cohort but different method.
#   - PASS.*/GBMI.*: External meta-analyses from various consortia. Different GWAS
#     entirely, though some include UKBB in their meta-analysis.
#
# This mapping only includes traits where: (1) phenotype clearly matches between
# Kanai and our sumstats, and (2) both are UKBB-only GWAS. No exact same-publication
# matches exist since Kanai ran their own analysis pipeline.
#
# kanai_trait -> sumstats file name
KANAI_SUMSTATS_MATCHING = {
    "Age_at_Menarche": "UKB_460K.repro_MENARCHE_AGE",
    "ALP": "UKB_460K.biochemistry_AlkalinePhosphatase",
    "AST": "UKB_460K.biochemistry_AspartateAminotransferase",
    "Balding_Type4": "UKB_460K.body_BALDING1",
    "BrC": "PANUKBB.Breast_Cancer_female",
    "DBP": "UKB_460K.bp_DIASTOLICadjMEDz",
    "Diverticulosis": "PANUKBB.Diverticulosis_And_Diverticulitis",
    "eBMD": "UKB_460K.bmd_HEEL_TSCOREz",
    "Hypothyroidism": "UKB_460K.disease_HYPOTHYROIDISM_SELF_REP",
    "IGF1": "UKB_460K.biochemistry_IGF1",
    "Inguinal_Hernia": "PANUKBB.Inguinal_Hernia",
    "Morning_Person": "UKB_460K.other_MORNINGPERSON",
    "P": "UKB_460K.biochemistry_Phosphate",
    "PrC": "UKB_460K.cancer_PROSTATE",
    "RBC": "UKB_460K.blood_RED_COUNT",
    "sCr": "UKB_460K.biochemistry_Creatinine",
    "SkC": "PANUKBB.C44_Other_Malignant_Neoplasms_Of_Skin",
    "TBil": "UKB_460K.biochemistry_TotalBilirubin",
    "TC": "UKB_460K.biochemistry_Cholesterol",
    "Testosterone_M": "UKB_460K.biochemistry_Testosterone_Male",
    "TP": "UKB_460K.biochemistry_TotalProtein",
    "VitD": "UKB_460K.biochemistry_VitaminD",
    "WBC": "UKB_460K.blood_WHITE_COUNT",
    "WHRadjBMI": "UKB_460K.body_WHRadjBMIz",
}


rule kanai_sumstats_matching:
    input:
        "results/kanai_et_al/supp_tables.xlsx",
        "config/traits_indep107.tsv",
    output:
        "results/kanai_et_al/sumstats_matching.tsv",
    run:
        kanai_traits = (
            pl.from_pandas(
                pd.read_excel(
                    input[0], sheet_name="ST2_overview_traits", header=2,
                    usecols=[
                        "cohort", "trait", "description",
                        "definition", "model_marginal", "n_samples",
                    ],
                )
            )
            .filter(pl.col("cohort") == "UKBB")
        )
        our_traits = pl.read_csv(input[1], separator="\t")
        rows = []
        for row in kanai_traits.iter_rows(named=True):
            kt = row["trait"]
            our_file = KANAI_SUMSTATS_MATCHING.get(kt)
            if our_file is None:
                continue
            our_row = our_traits.filter(
                pl.col("File name") == our_file
            ).row(0, named=True)
            rows.append({
                "kanai_trait": kt,
                "kanai_description": row["description"],
                "kanai_definition": row["definition"],
                "kanai_model": row["model_marginal"],
                "kanai_n": row["n_samples"],
                "sumstats_file": our_file,
                "sumstats_trait": our_row["Trait"],
                "sumstats_n": our_row["Mean N"],
            })
        (
            pl.DataFrame(rows)
            .sort("kanai_trait")
            .write_csv(output[0], separator="\t")
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
            fig.savefig(out, bbox_inches="tight")


HEIGHT_SUMSTATS = "results/sumstats_107/PASS.Height.Yengo2022.sumstats.gz"
HEIGHT_KANAI_TRAIT = "Height"


rule gwas_effect_sizes_height:
    input:
        HEIGHT_SUMSTATS,
    output:
        "results/gwas_effect_sizes/height.parquet",
    run:
        (
            pl.from_pandas(pd.read_csv(input[0], sep=r"\s+", usecols=["SNP", "Z"]))
            .with_columns((pl.col("Z") ** 2).alias("Z2"))
            .write_parquet(output[0])
        )


rule kanai_height:
    input:
        "results/kanai_et_al/supp_tables.xlsx",
        "results/gwas_effect_sizes/height.parquet",
    output:
        "results/kanai_et_al/high_pip_height.parquet",
    run:
        (
            pl.from_pandas(
                pd.read_excel(
                    input[0], sheet_name="ST3_high_PIP_pairs", header=2,
                    usecols=["variant", "rsid", "cohort", "trait"],
                )
            )
            .filter(
                (pl.col("cohort") == "UKBB") & (pl.col("trait") == HEIGHT_KANAI_TRAIT)
            )
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


rule gwas_effect_sizes_height_histogram:
    input:
        p=f"results/gwas_effect_sizes/quantile/{config['gpn_star_p']}/0.001.parquet",
        m=f"results/gwas_effect_sizes/quantile/{config['gpn_star_m']}/0.001.parquet",
        v=f"results/gwas_effect_sizes/quantile/{config['gpn_star_v']}/0.001.parquet",
        kanai="results/kanai_et_al/high_pip_height.parquet",
        height="results/gwas_effect_sizes/height.parquet",
        rsids="results/variants/rsid/merged.parquet",
        scores_p=f"results/variant_scores/quantile/{config['gpn_star_p']}/0.001.parquet",
        scores_m=f"results/variant_scores/quantile/{config['gpn_star_m']}/0.001.parquet",
        scores_v=f"results/variant_scores/quantile/{config['gpn_star_v']}/0.001.parquet",
    output:
        ecdf="results/gwas_effect_sizes/height_ecdf.svg",
        kde="results/gwas_effect_sizes/height_kde.svg",
    run:
        height = pl.read_parquet(input.height)
        rsids = pl.read_parquet(input.rsids)
        hue_col = "variant set"
        col = "Z2"

        def select_model(scores_path, name):
            scores = pl.read_parquet(scores_path)
            return (
                pl.concat([rsids, scores], how="horizontal")
                .filter(pl.col("score") == 1)
                .join(height, left_on="rsid", right_on="SNP", how="inner")
            ), name

        parts = [
            select_model(input.scores_p, "GPN-Star-P top 0.1%"),
            select_model(input.scores_m, "GPN-Star-M top 0.1%"),
            select_model(input.scores_v, "GPN-Star-V top 0.1%"),
            (pl.read_parquet(input.kanai), "Kanai et al. finemapped high PIP"),
        ]
        df = pl.concat([
            d.with_columns(pl.lit(f"{name}\n(n={len(d):,})").alias(hue_col)).select([col, hue_col])
            for d, name in parts
        ] + [
            height.sample(30000, seed=42)
            .with_columns(pl.lit(f"All\n(n={len(height):,})").alias(hue_col))
            .select([col, hue_col])
        ]).filter(pl.col(col) > 0)
        for plot_fn, out in [(sns.ecdfplot, output.ecdf), (sns.kdeplot, output.kde)]:
            fig, ax = plt.subplots(figsize=(5, 4))
            kwargs = {"common_norm": False} if plot_fn is sns.kdeplot else {}
            plot_fn(data=df, x=col, hue=hue_col, log_scale=(True, False), ax=ax, **kwargs)
            ax.set_xlabel("$Z^2$ (Height)")
            legend = ax.get_legend()
            handles = legend.legend_handles
            labels = [t.get_text() for t in legend.get_texts()]
            legend.remove()
            fig.legend(
                handles, labels, loc="upper center", ncol=3,
                bbox_to_anchor=(0.5, 1.0), frameon=False,
            )
            sns.despine()
            fig.subplots_adjust(top=0.75)
            fig.savefig(out, bbox_inches="tight")


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
