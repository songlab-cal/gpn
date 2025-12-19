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
