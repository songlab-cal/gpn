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
        import os
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
