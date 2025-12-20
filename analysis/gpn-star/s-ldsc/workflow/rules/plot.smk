rule plot_score_overlap:
    input:
        variants="results/variants/merged.annot_with_cre_v2.parquet",
        maf="results/maf/merged.parquet",
        scores=expand(
            "results/variant_scores/quantile/{model}/0.001.parquet",
            model=[config["gpn_star_p"], config["gpn_star_m"], config["gpn_star_v"]],
        ),
    output:
        all_svg="results/plots/score_overlap/all.svg",
        all_pdf="results/plots/score_overlap/all.pdf",
        common_svg="results/plots/score_overlap/common.svg",
        common_pdf="results/plots/score_overlap/common.pdf",
    run:
        V = pl.read_parquet(input.variants, columns=["consequence"])
        MAF = pl.read_parquet(input.maf)
        V = (
            pl.concat([V, MAF], how="horizontal")
            .with_columns(common=pl.col("MAF") > 0.05)
            .drop("MAF")
        )

        models = {
            "GPN-Star (P)": config["gpn_star_p"],
            "GPN-Star (M)": config["gpn_star_m"],
            "GPN-Star (V)": config["gpn_star_v"],
        }

        for i, (name, model) in enumerate(models.items()):
            V = V.with_columns(
                pl.read_parquet(input.scores[i])["score"].eq(1).alias(name)
            )

        # All variants
        subsets_all = {name: set(V[name].arg_true()) for name in models}
        fig = plot_venn(subsets_all, config["palette"])
        fig.savefig(output.all_svg, bbox_inches="tight")
        fig.savefig(output.all_pdf, bbox_inches="tight")
        plt.close(fig)

        # Common variants only
        subsets_common = {
            name: set((V[name] & V["common"]).arg_true()) for name in models
        }
        fig = plot_venn(subsets_common, config["palette"])
        fig.savefig(output.common_svg, bbox_inches="tight")
        fig.savefig(output.common_pdf, bbox_inches="tight")
        plt.close(fig)


def get_score_consequence_inputs(wildcards):
    models = config["score_consequence_models"]
    return [
        f"results/score_consequence/annot_with_cre_v2/quantile/{model}/0.001.parquet"
        for model in models.values()
    ]


def get_model_comparison_outputs():
    outputs = []
    for model1, model2 in config["model_comparisons"]:
        m1 = model1.replace(" ", "_").replace("(", "").replace(")", "")
        m2 = model2.replace(" ", "_").replace("(", "").replace(")", "")
        for category in ["all", "common"]:
            outputs.append(f"results/latex/score_consequence_comparison_{m1}_vs_{m2}_{category}.tex")
    return outputs


rule score_consequence_analysis:
    input:
        global_count="results/global_score_consequence/annot_with_cre_v2.parquet",
        score_consequence=get_score_consequence_inputs,
    output:
        plot_all_svg="results/plots/score_consequence/all.svg",
        plot_all_pdf="results/plots/score_consequence/all.pdf",
        plot_common_svg="results/plots/score_consequence/common.svg",
        plot_common_pdf="results/plots/score_consequence/common.pdf",
        plot_common_1pct_svg="results/plots/score_consequence/common_1pct.svg",
        plot_common_1pct_pdf="results/plots/score_consequence/common_1pct.pdf",
        latex_all="results/latex/score_consequence_enrichment_all.tex",
        latex_common="results/latex/score_consequence_enrichment_common.tex",
        model_comparisons=get_model_comparison_outputs(),
    run:
        os.makedirs("results/plots/score_consequence", exist_ok=True)
        os.makedirs("results/latex", exist_ok=True)

        models = config["score_consequence_models"]
        renaming = config["consequence_renaming"]

        global_count = pd.read_parquet(input.global_count).rename(
            columns={"count": "global_count"}
        )

        df = pd.concat(
            pd.read_parquet(path).assign(model=model_name)
            for model_name, path in zip(models.keys(), input.score_consequence)
        )
        df = df.merge(global_count, on=["consequence", "category"], how="left")
        df["other_count"] = df["global_count"] - df["count"]

        # Main analysis for GPN-Star (P)
        df2 = df.query('model == "GPN-Star (P)" and consequence != "total"')

        # All variants
        enrich_all = calculate_enrichment(
            df2.query("category == 'all'").copy(), renaming
        )
        fig = plot_enrichment(enrich_all)
        fig.savefig(output.plot_all_svg, bbox_inches="tight")
        fig.savefig(output.plot_all_pdf, bbox_inches="tight")
        plt.close(fig)

        with open(output.latex_all, "w") as f:
            f.write(enrich_to_latex(enrich_all))

        # Common variants
        enrich_common = calculate_enrichment(
            df2.query("category == 'common'").copy(), renaming
        )
        fig = plot_enrichment(enrich_common)
        fig.savefig(output.plot_common_svg, bbox_inches="tight")
        fig.savefig(output.plot_common_pdf, bbox_inches="tight")
        plt.close(fig)

        with open(output.latex_common, "w") as f:
            f.write(enrich_to_latex(enrich_common))

        # Common variants > 1% proportion
        fig = plot_enrichment(enrich_common[enrich_common.proportion > 1e-2], figsize=3.5)
        fig.savefig(output.plot_common_1pct_svg, bbox_inches="tight")
        fig.savefig(output.plot_common_1pct_pdf, bbox_inches="tight")
        plt.close(fig)

        # Model comparisons
        df3 = df.query('consequence != "total"')

        for model1, model2 in config["model_comparisons"]:
            m1 = model1.replace(" ", "_").replace("(", "").replace(")", "")
            m2 = model2.replace(" ", "_").replace("(", "").replace(")", "")

            for category in ["all", "common"]:
                result = calculate_enrichment_two_models(
                    df3.query(f"category == '{category}'"), model1, model2, renaming
                ).query("q_value < 0.05").sort_values("odds_ratio", ascending=False)

                latex_path = f"results/latex/score_consequence_comparison_{m1}_vs_{m2}_{category}.tex"
                with open(latex_path, "w") as f:
                    f.write(enrich_two_models_to_latex(result))


# LDSC heritability enrichment rules

rule ldsc_part1_main:
    input:
        traits="config/traits_indep107.tsv",
        polygenicity="config/polygenicity.tsv",
    output:
        h2_full="results/plots/ldsc/fig3_h2_enrich_full.svg",
        tau_full="results/plots/ldsc/fig3_tau_est_full.svg",
        tau_p_full="results/plots/ldsc/fig3_tau_p_full.svg",
        h2_cds="results/plots/ldsc/fig3_h2_enrich_cds.svg",
        h2_noncds="results/plots/ldsc/fig3_h2_enrich_noncds.svg",
        poly_main="results/plots/ldsc/polygenicity_main.svg",
        poly_supp="results/plots/ldsc/polygenicity_supp.pdf",
    run:
        os.makedirs("results/plots/ldsc", exist_ok=True)
        plt.rcParams["font.size"] = 12

        traits = pd.read_csv(input.traits, sep="\t")
        traits = traits[traits["File name"] != "PASS.Multiple_Sclerosis.IMSGC2019"]
        traits.loc[traits["File name"] == "UKB_460K.cancer_PROSTATE", "Trait"] = "Prostate Cancer_1"

        conservation_models = list(config["conservation"].keys())
        gpn_star_models = [config["gpn_star_p"], config["gpn_star_m"], config["gpn_star_v"]]
        other_models = ["GPN-MSA_absLLR", "CADD"]
        enformer_tissue_agnostic = "cV2F_tissue_agnostic_extract_Enformer.Enformer_all_all"
        models_part1 = gpn_star_models + conservation_models + [enformer_tissue_agnostic] + other_models

        res = load_ldsc_results(
            traits, models_part1, [0.001],
            ["quantile", "quantile_CDS", "quantile_nonCDS"],
            config["model_renaming"],
        )
        agg_res = run_ldsc_meta_analysis(res)
        palette = config["palette"]

        # Panel A - Full analysis
        df = agg_res[(agg_res.approach == "quantile") & (agg_res.q == 0.001)]
        df = df.sort_values("Enrichment", ascending=False)

        fig = plot_bar_ldsc(df, "Enrichment", "Heritability enrichment", palette, 1, major_locator=10)
        fig.savefig(output.h2_full, bbox_inches="tight")
        plt.close(fig)

        fig = plot_bar_ldsc(df, "tau_star", r"Conditional effect $(\tau^*)$", palette, 0)
        fig.savefig(output.tau_full, bbox_inches="tight")
        plt.close(fig)

        fig = plot_bar_ldsc(df, "Coefficient_p_minuslog10", r"Conditional effect $(-log_{10}p)$", palette, 0)
        fig.savefig(output.tau_p_full, bbox_inches="tight")
        plt.close(fig)

        # Panel B - CDS
        df = agg_res[(agg_res.approach == "quantile_CDS") & (agg_res.q == 0.001)]
        df = df.sort_values("Enrichment", ascending=False)
        fig = plot_bar_ldsc(df, "Enrichment", "Heritability enrichment", palette, 1, major_locator=10)
        fig.savefig(output.h2_cds, bbox_inches="tight")
        plt.close(fig)

        # Panel C - non-CDS
        df = agg_res[(agg_res.approach == "quantile_nonCDS") & (agg_res.q == 0.001)]
        df = df.sort_values("Enrichment", ascending=False)
        fig = plot_bar_ldsc(df, "Enrichment", "Heritability enrichment", palette, 1, major_locator=10)
        fig.savefig(output.h2_noncds, bbox_inches="tight")
        plt.close(fig)

        # Polygenicity analysis
        polygenicity = pd.read_csv(input.polygenicity, sep="\t")
        polygenicity.trait = polygenicity.trait.replace({
            "General risk tolerance": "General Risk Tolerance",
            "Bipolar disorder": "Bipolar disorder (all cases)",
            "Years of education": "Education Years",
            "Chronotype (morning person)": "Morning Person",
            "Body mass index": "BMI",
            "Reported drinks per week": "Drinks Per Week",
            "Sleep duration": "Sleep Duration",
            "Reaction time": "Reaction Time",
            "Age at menarche": "Menarche Age",
            "Creatinine level": "Creatinine",
            "Blood pressure - diastolic": "Diastolic (Blood Pressure)",
            "Atrial fibrillation": "Atrial Fibrillation",
            "Waist-hip ratio (corrected for BMI)": "WHR BMI ratio",
            "IGF1 level in blood": "IGF1",
            "Alzheimer's disease": "Alzheimer's disease",
            "Phosphate levels": "Phosphate",
            "Total protein level in blood": "Total Protein",
            "Aspartate aminotransferase level": "Aspartate Aminotransferase",
            "Hypothyroidism": "Hypothyroidism (Self reported)",
            "Coronary artery disease": "Coronary Artery Disease (Aragam)",
            "Breast cancer": "Breast Cancer (female)",
            "Inflammatory bowel disease": "IBD",
            "Red blood cell count": "Rbc Count",
        })

        def primate_specificity(rows):
            p_row = rows[rows.model == "GPN-Star (P)"].iloc[0]
            m_row = rows[rows.model == "GPN-Star (M)"].iloc[0]
            pooled_std = np.sqrt(p_row["Enrichment_std_error"]**2 + m_row["Enrichment_std_error"]**2)
            mean_diff = p_row["Enrichment"] - m_row["Enrichment"]
            return pd.Series({"mean_diff": mean_diff})

        res_quantile = res[(res.approach == "quantile") & (res.q == 0.001)]
        df_poly = res_quantile.groupby("trait").apply(primate_specificity).reset_index()
        df_poly = df_poly.merge(polygenicity, how="inner", on="trait")
        df_poly["Enr(Primates) - Enr(Mammals)"] = df_poly.mean_diff
        df_poly["log_10 (effective polygenicity)"] = df_poly.polygenicity_effective

        # Plot polygenicity
        plt.rcParams["font.size"] = 10
        x_col = "log_10 (effective polygenicity)"
        y_col = "Enr(Primates) - Enr(Mammals)"

        stat, pvalue = pearsonr(df_poly[x_col], df_poly[y_col])
        X = sm.add_constant(df_poly[x_col])
        ols_model = sm.OLS(df_poly[y_col], X).fit()

        fig, ax = plt.subplots(figsize=(2.3, 2.3))
        ax.plot(10 ** df_poly[x_col], df_poly[y_col], "o")
        ax.set_xscale("log")
        x_line_log = np.linspace(df_poly[x_col].min() * 0.98, df_poly[x_col].max() * 1.02, 100)
        y_line = ols_model.predict(sm.add_constant(x_line_log))
        ax.plot(10**x_line_log, y_line, "--", color="black", linewidth=1)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.set_title(rf"$r=${stat:.2f}, $p=${pvalue:.2g}", fontsize=10)
        ax.set_xlabel("Effective polygenicity")
        ax.set_ylabel(r"$\Delta$ Enrichment" + "\n" + r"(Primates $-$ Mammals)")
        sns.despine()
        plt.tight_layout()
        fig.savefig(output.poly_main, bbox_inches="tight")
        plt.close(fig)

        # Polygenicity with labels
        df_poly["trait2"] = df_poly.trait.replace({
            "Drinks Per Week": "Drinks per week",
            "Bipolar disorder (all cases)": "Bipolar disorder",
            "Breast Cancer (female)": "Breast Cancer",
            "General Risk Tolerance": "Risk tolerance",
            "Education Years": "Edu years",
            "Diastolic (Blood Pressure)": "Blood pressure",
            "Total Protein": "Protein",
            "Atrial Fibrillation": "AFib",
            "Hypothyroidism (Self reported)": "Hypothyroidism",
            "Coronary Artery Disease (Aragam)": "Coronary Artery Disease",
            "Alzheimer's disease": "Alzheimer's",
        })

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(10 ** df_poly[x_col], df_poly[y_col], "o")
        ax.set_xscale("log")
        ax.plot(10**x_line_log, y_line, "--", color="black", linewidth=1)
        texts = [ax.text(10 ** row[x_col], row[y_col], row["trait2"], fontsize=8) for _, row in df_poly.iterrows()]
        adjust_text(texts)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.set_title(rf"$r=${stat:.2f}, $p=${pvalue:.2g}", fontsize=10)
        ax.set_xlabel("Effective polygenicity")
        ax.set_ylabel(r"$\Delta$ Enrichment" + "\n" + r"(Primates $-$ Mammals)")
        sns.despine()
        plt.tight_layout()
        fig.savefig(output.poly_supp, bbox_inches="tight")
        plt.close(fig)


rule ldsc_part2_quantiles:
    input:
        traits="config/traits_indep107.tsv",
    output:
        h2_V="results/plots/ldsc/fig3_h2_enrich_top0.05_V.svg",
        h2_M="results/plots/ldsc/fig3_h2_enrich_top0.05_M.svg",
        h2_P="results/plots/ldsc/fig3_h2_enrich_top0.05_P.svg",
    run:
        os.makedirs("results/plots/ldsc", exist_ok=True)
        plt.rcParams["font.size"] = 12

        traits = pd.read_csv(input.traits, sep="\t")
        traits = traits[traits["File name"] != "PASS.Multiple_Sclerosis.IMSGC2019"]

        conservation_models = list(config["conservation"].keys())
        gpn_star_models = [config["gpn_star_p"], config["gpn_star_m"], config["gpn_star_v"]]
        models_part2 = gpn_star_models + conservation_models

        res = load_ldsc_results(
            traits, models_part2, config["quantiles"], ["quantile"], config["model_renaming"]
        )
        agg_res = run_ldsc_meta_analysis(res).sort_values("Enrichment", ascending=False)
        agg_res["Model"] = agg_res.model
        agg_res["Heritability enrichment"] = agg_res["Enrichment"]
        agg_res["Heritability enrichment_sd"] = agg_res["Enrichment_sd"]

        palette = config["palette"]
        outputs = {"V": output.h2_V, "M": output.h2_M, "P": output.h2_P}

        for timescale, out_path in outputs.items():
            fig = plot_agg_relplot(
                agg_res[agg_res.model.str.endswith(f"({timescale})")],
                palette=palette,
                x_label="Fraction of top constrained SNPs",
                y_label="Heritability enrichment",
                values_to_plot=["Heritability enrichment"],
            )
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)


rule ldsc_part3_tissue:
    input:
        traits="config/traits_indep107.tsv",
        trait_tissues="config/trait_tissues.tsv",
    output:
        scatter="results/plots/ldsc/by_tissue_scatter.pdf",
        tissue_bar="results/plots/ldsc/by_tissue_select_bar.svg",
        trait_bar="results/plots/ldsc/by_trait_select_bar.svg",
    run:
        os.makedirs("results/plots/ldsc", exist_ok=True)
        plt.rcParams["font.size"] = 12

        traits = pd.read_csv(input.traits, sep="\t")
        traits = traits[traits["File name"] != "PASS.Multiple_Sclerosis.IMSGC2019"]
        traits.loc[traits["File name"] == "UKB_460K.cancer_PROSTATE", "Trait"] = "Prostate Cancer_1"

        trait_tissues = pd.read_csv(input.trait_tissues, sep="\t", index_col=0)
        trait_tissues = trait_tissues.astype(bool)
        trait_tissues = trait_tissues[trait_tissues.index != "Multiple Sclerosis"]
        trait_tissues.rename(columns={"blood/immune": "blood"}, inplace=True)
        counts = trait_tissues.groupby(trait_tissues.index).cumcount()
        trait_tissues.index = np.where(
            counts > 0, trait_tissues.index + "_" + counts.astype(str), trait_tissues.index
        )

        tissue_order = trait_tissues.sum(axis=0).sort_values(ascending=False).index.tolist()
        tissue_traits = {t: set(trait_tissues[trait_tissues[t]].index) for t in trait_tissues.columns}

        gpn_star_top_model = config["gpn_star_top_model"]
        gpn_star_top_model_name = config["model_renaming"][gpn_star_top_model]
        filt_seg_tissues = config["filt_seg_tissues"]
        enformer_models = config["enformer_models"]
        seg_models = [f"filt_SEG/{gpn_star_top_model}/{tissue}" for tissue in filt_seg_tissues]
        models_part3 = [gpn_star_top_model] + seg_models + enformer_models

        model_renaming_tissue_specific = {gpn_star_top_model: f"{gpn_star_top_model_name}-all"}
        for tissue in filt_seg_tissues:
            tissue_short = tissue.replace("group_", "").split("_")[0].lower()
            model_renaming_tissue_specific[f"filt_SEG/{gpn_star_top_model}/{tissue}"] = f"{gpn_star_top_model_name}-{tissue_short}"
        for model in enformer_models:
            model_renaming_tissue_specific[model] = f"Enformer-{model.split('_')[-2]}"

        res = load_ldsc_results(traits, models_part3, [0.001], ["quantile"], model_renaming_tissue_specific)
        res["model_tissue"] = res.model.str.split("-").str[-1]

        model_order = [gpn_star_top_model_name, "Enformer"]
        palette = config["palette"]

        # Scatter plot by tissue
        df = pd.concat([
            run_ldsc_meta_analysis(
                res[res.trait.isin(tissue_traits[tissue]) & res.model_tissue.isin(["all", tissue])]
            ).assign(tissue=tissue)
            for tissue in tissue_traits.keys()
        ])
        df["model_model"] = df.model.str.split("-").str[:-1].str.join("-")
        df["model_tissue"] = df.model.str.split("-").str[-1]
        df["tissue_specific"] = (df.model_tissue == df.tissue).map({True: "Yes", False: "No"})
        df["tissue"] = pd.Categorical(df["tissue"], categories=tissue_order, ordered=True)
        df["model_model"] = pd.Categorical(df["model_model"], categories=model_order, ordered=True)
        df = df.sort_values(["tissue", "model_model"])

        tissue_specific_markers = {"No": "o", "Yes": "^"}
        unique_tissues = df["tissue"].cat.categories.drop_duplicates()

        fig, axes = plt.subplots(3, 3, figsize=(5.5, 5), sharex=False, sharey=False)
        for i, tissue_name in enumerate(unique_tissues):
            if i >= 9:
                break
            ax = axes.flatten()[i]
            tissue_df = df[df["tissue"] == tissue_name]
            ax.set_title(f"{tissue_name} traits")
            if tissue_df.empty:
                continue
            for _, row in tissue_df.iterrows():
                ax.errorbar(
                    x=row["Enrichment"], y=row["tau_star"],
                    xerr=row["Enrichment_sd"], yerr=row["tau_star_sd"],
                    marker=tissue_specific_markers[row["tissue_specific"]],
                    fmt="", color=palette[row["model_model"]], ecolor=palette[row["model_model"]],
                )
            ax.axvline(x=1, linestyle="--", color="black", linewidth=1)
            ax.axhline(y=0, linestyle="--", color="black", linewidth=1)
        for i in range(len(unique_tissues), 9):
            axes.flatten()[i].set_visible(False)

        color_legend_elements = [
            Line2D([0], [0], marker="o", color="w", label=m, markerfacecolor=palette[m], markersize=8)
            for m in model_order
        ]
        fig.legend(handles=color_legend_elements, title="Model", loc="center left", bbox_to_anchor=(1, 2/3), frameon=False)
        marker_legend_elements = [
            Line2D([0], [0], marker=m, color="grey", label=k, linestyle="None")
            for k, m in tissue_specific_markers.items()
        ]
        fig.legend(handles=marker_legend_elements, title="Tissue-specific", loc="center left", bbox_to_anchor=(1, 1/3), frameon=False)
        fig.supxlabel("Heritability enrichment")
        fig.supylabel(r"Standardized coefficient ($\tau^{\star}$)")
        fig.tight_layout()
        sns.despine()
        fig.savefig(output.scatter, bbox_inches="tight")
        plt.close(fig)

        # Bar plot by tissue (subset)
        subset_tissues = ["brain", "blood", "liver"]
        df_subset = df[df.tissue.isin(subset_tissues)].copy()
        df_subset["tissue"] = pd.Categorical(df_subset["tissue"], categories=subset_tissues, ordered=True)

        def barplot_with_errors(data, x, y, hue, y_err, order, hue_order, palette_dict, **kwargs):
            ax = plt.gca()
            n_hue = len(hue_order)
            x_indices = np.arange(len(order))
            bar_width = 0.9 / n_hue
            for j, model in enumerate(hue_order):
                model_df = data[data[hue] == model].set_index(x).reindex(order)
                bar_pos = x_indices - 0.45 + j * bar_width + bar_width / 2
                ax.bar(bar_pos, model_df[y], bar_width, label=model, color=palette_dict.get(model, "gray"), yerr=model_df[y_err])
            ax.set_xticks(x_indices)
            ax.set_xticklabels(order)

        g = sns.FacetGrid(df_subset, col="tissue", height=2, aspect=0.8, sharey=False, col_wrap=3)
        g.map_dataframe(barplot_with_errors, x="tissue_specific", y="Enrichment", y_err="Enrichment_sd", hue="model_model", palette_dict=palette, hue_order=model_order, order=["No", "Yes"])
        g.set(ylim=1)
        for ax, title in zip(g.axes.flat, g.col_names):
            ax.set_title(f"{title.capitalize()} traits", fontsize=10)
        g.set_axis_labels("Tissue-specific", "Heritability\nenrichment")
        g.add_legend(title="Model")
        g.tight_layout()
        sns.despine()
        g.savefig(output.tissue_bar, bbox_inches="tight")
        plt.close()

        # Bar plot by trait (select traits)
        select_tissue_trait = {"brain": "Schizophrenia", "blood": "Lupus (SLE)", "liver": "IGF1"}
        select_trait_tissue = {v: k for k, v in select_tissue_trait.items()}
        select_traits = list(select_tissue_trait.values())

        df_trait = pd.concat([
            res[(res.trait == trait) & res.model_tissue.isin(["all", select_trait_tissue[trait]])]
            for trait in select_traits
        ])
        df_trait["tissue"] = df_trait.trait.map(select_trait_tissue)
        df_trait["model_model"] = df_trait.model.str.split("-").str[:-1].str.join("-")
        df_trait["tissue_specific"] = (df_trait.model_tissue == df_trait.tissue).map({True: "Yes", False: "No"})
        df_trait.rename(columns={"Enrichment_std_error": "Enrichment_sd"}, inplace=True)
        df_trait["tissue"] = pd.Categorical(df_trait["tissue"], categories=subset_tissues, ordered=True)

        g = sns.FacetGrid(df_trait, col="tissue", height=2, aspect=0.8, sharey=False)
        g.map_dataframe(barplot_with_errors, x="tissue_specific", y="Enrichment", y_err="Enrichment_sd", hue="model_model", palette_dict=palette, hue_order=model_order, order=["No", "Yes"])
        g.set(ylim=1)
        for ax, title in zip(g.axes.flat, g.col_names):
            trait = df_trait[df_trait["tissue"] == title]["trait"].iloc[0]
            if "Lupus" in trait:
                trait = "Lupus"
            ax.set_title(trait, fontsize=10)
        g.set_axis_labels("Tissue-specific", "Heritability\nenrichment")
        g.add_legend(title="Model")
        g.tight_layout()
        sns.despine()
        g.savefig(output.trait_bar, bbox_inches="tight")
        plt.close()


rule ldsc_part4_ablation:
    input:
        traits="config/traits_indep107.tsv",
    output:
        h2="results/plots/ldsc/fig3_h2_enrich_top0.05_primate36.svg",
    run:
        os.makedirs("results/plots/ldsc", exist_ok=True)
        plt.rcParams["font.size"] = 12

        traits = pd.read_csv(input.traits, sep="\t")
        traits = traits[traits["File name"] != "PASS.Multiple_Sclerosis.IMSGC2019"]

        ablation_models = config["ablation_models"]
        models_part4 = [config["gpn_star_top_model"]] + ablation_models + ["phyloP_239p", "phastCons_43p"]

        res = load_ldsc_results(
            traits, models_part4, config["quantiles"], ["quantile"], config["model_renaming_part4"]
        )
        agg_res = run_ldsc_meta_analysis(res).sort_values("Enrichment", ascending=False)
        agg_res["Model"] = agg_res.model
        agg_res["Heritability enrichment"] = agg_res["Enrichment"]
        agg_res["Heritability enrichment_sd"] = agg_res["Enrichment_sd"]

        palette = config["palette_part4"]
        fig = plot_agg_relplot(
            agg_res,
            palette=palette,
            x_label="Fraction of top constrained SNPs",
            y_label="Heritability enrichment",
            values_to_plot=["Heritability enrichment"],
        )
        fig.savefig(output.h2, bbox_inches="tight")
        plt.close(fig)


rule ldsc_supp_tables:
    input:
        trait_tissues="config/trait_tissues.tsv",
    output:
        latex="results/latex/trait_tissues.tex",
    run:
        os.makedirs("results/latex", exist_ok=True)

        trait_tissues = pd.read_csv(input.trait_tissues, sep="\t", index_col=0)
        trait_tissues = trait_tissues.astype(bool)
        trait_tissues = trait_tissues[trait_tissues.index != "Multiple Sclerosis"]
        trait_tissues.rename(columns={"blood/immune": "blood"}, inplace=True)

        tissue_order = trait_tissues.sum(axis=0).sort_values(ascending=False).index.tolist()
        df = trait_tissues[tissue_order].sort_index().replace({True: 1, False: 0})

        with open(output.latex, "w") as f:
            f.write(df.to_latex(escape=False))


rule ldsc_consequence_combined:
    input:
        traits="config/traits_indep107.tsv",
    output:
        svg="results/plots/ldsc/consequence_combined.svg",
        pdf="results/plots/ldsc/consequence_combined.pdf",
    run:
        os.makedirs("results/plots/ldsc", exist_ok=True)
        plt.rcParams["font.size"] = 12

        traits = pd.read_csv(input.traits, sep="\t")
        traits = traits[traits["File name"] != "PASS.Multiple_Sclerosis.IMSGC2019"]

        consequences = config["consequences_to_explore"]
        gpn_star_top_model = config["gpn_star_top_model"]

        # Load baseline consequence results
        res_baseline = load_consequence_ldsc_results(traits, consequences)
        agg_baseline = run_consequence_meta_analysis(res_baseline)
        agg_baseline["variant_set"] = "All"

        # Load quantile within consequence results
        res_quantile = load_quantile_consequence_ldsc_results(
            traits, consequences, [gpn_star_top_model], 0.001, config["model_renaming"]
        )
        agg_quantile = run_consequence_meta_analysis(res_quantile)
        agg_quantile["variant_set"] = "Top 0.1%"

        # Combine
        agg_res = pd.concat([agg_baseline, agg_quantile], ignore_index=True)

        # Sort consequences by top 0.1% enrichment
        consequence_order = (
            agg_res[agg_res["variant_set"] == "Top 0.1%"]
            .sort_values("Enrichment", ascending=False)["consequence"]
            .tolist()
        )

        fig = plt.figure(figsize=(10, 6))
        hue_order = ["All", "Top 0.1%"]
        sns.barplot(
            data=agg_res, y="consequence", x="Enrichment",
            hue="variant_set", hue_order=hue_order, order=consequence_order
        )

        # Add error bars
        for i, consequence in enumerate(consequence_order):
            for j, variant_set in enumerate(hue_order):
                row = agg_res[(agg_res["consequence"] == consequence) & (agg_res["variant_set"] == variant_set)]
                if not row.empty:
                    y_pos = i + (j - 0.5) * 0.4
                    plt.errorbar(
                        x=row["Enrichment"].values[0], y=y_pos,
                        xerr=row["Enrichment_sd"].values[0], fmt="none", ecolor="black"
                    )

        plt.xlim(left=1)
        plt.xlabel("Heritability enrichment")
        plt.ylabel("")
        plt.legend(title="Variant set")
        sns.despine()
        plt.tight_layout()
        fig.savefig(output.svg, bbox_inches="tight")
        fig.savefig(output.pdf, bbox_inches="tight")
        plt.close(fig)


rule ldsc_consequence_model_comparison:
    input:
        traits="config/traits_indep107.tsv",
    output:
        svg="results/plots/ldsc/consequence_model_comparison.svg",
        pdf="results/plots/ldsc/consequence_model_comparison.pdf",
    run:
        os.makedirs("results/plots/ldsc", exist_ok=True)
        plt.rcParams["font.size"] = 12

        traits = pd.read_csv(input.traits, sep="\t")
        traits = traits[traits["File name"] != "PASS.Multiple_Sclerosis.IMSGC2019"]

        consequences = config["consequence_comparison_consequences"]
        models = config["consequence_comparison_models"]
        res = load_quantile_consequence_ldsc_results(
            traits, consequences, models, 0.001, config["model_renaming"]
        )
        agg_res = run_consequence_model_meta_analysis(res)

        palette = config["palette"]

        fig, axes = plt.subplots(len(consequences), 1, figsize=(3, 2.5 * len(consequences)), sharex=False)
        if len(consequences) == 1:
            axes = [axes]

        for ax, consequence in zip(axes, consequences):
            df = agg_res[agg_res["consequence"] == consequence].copy()
            df = df.sort_values("Enrichment", ascending=False)

            sns.barplot(data=df, x="Enrichment", y="model", palette=palette, ax=ax)
            ax.errorbar(
                x=df["Enrichment"], y=range(len(df)),
                xerr=df["Enrichment_sd"], fmt="none", ecolor="black"
            )
            ax.set_xlim(left=1)
            ax.set_title(consequence)
            ax.set_xlabel("")
            ax.set_ylabel("")

        axes[-1].set_xlabel("Heritability enrichment")
        sns.despine()
        plt.tight_layout()
        fig.savefig(output.svg, bbox_inches="tight")
        fig.savefig(output.pdf, bbox_inches="tight")
        plt.close(fig)


rule all_plots:
    input:
        rules.plot_score_overlap.output,
        rules.score_consequence_analysis.output,
        rules.ldsc_part1_main.output,
        rules.ldsc_part2_quantiles.output,
        rules.ldsc_part3_tissue.output,
        rules.ldsc_part4_ablation.output,
        rules.ldsc_supp_tables.output,
        rules.ldsc_consequence_combined.output,
        rules.ldsc_consequence_model_comparison.output,
