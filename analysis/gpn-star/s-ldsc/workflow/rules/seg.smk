seg_cfg = pd.read_csv(
    "config/SEG/Multi_tissue_gene_expr.ldcts",
    sep="\t",
    header=None,
    names=["tissue", "paths"],
)
# LDSC fails with parens in file name
seg_cfg.tissue = seg_cfg.tissue.str.replace("(", "_", regex=False).str.replace(
    ")", "_", regex=False
)
seg_cfg.set_index("tissue", inplace=True)
seg_cfg = seg_cfg[seg_cfg.paths.str.contains("GTEx")]
seg_cfg["tissue_id"] = seg_cfg.paths.str.split(".").str[1]
seg_cfg.drop(columns=["paths"], inplace=True)
gtex_tissues = seg_cfg.index.values


ruleorder: seg_merge > process_features


def get_input_seg_merge(wc):
    tissue_id = seg_cfg.loc[wc.tissue, "tissue_id"]
    return [
        f"results/SEG/Multi_tissue_gene_expr_1000Gv3_ldscores/GTEx.{tissue_id}.{chrom}.annot.gz"
        for chrom in CHROMS
    ]


rule seg_merge:
    input:
        get_input_seg_merge,
    output:
        "results/variant_scores/SEG/GTEx/{tissue}.parquet",
    wildcard_constraints:
        tissues="|".join(gtex_tissues),
    run:
        V = pl.concat([pl.read_csv(f) for f in input])
        V = V.with_columns(pl.col("ANNOT").cast(pl.Boolean))
        V.write_parquet(output[0])


# taking union over annotations
rule seg_tissue_group:
    input:
        lambda wc: expand(
            "results/variant_scores/SEG/GTEx/{tissue}.parquet",
            tissue=config["SEG_GTEx_tissue_groups"][wc.group],
        ),
    output:
        "results/variant_scores/SEG/GTEx/group_{group}.parquet",
    run:
        V = pl.concat(
            (
                pl.read_parquet(f).rename({"ANNOT": f"ANNOT_{i}"})
                for i, f in enumerate(input)
            ),
            how="horizontal",
        )
        V = V.with_columns(pl.any_horizontal(pl.all()).alias("ANNOT"))
        print(V.sum())
        V.select("ANNOT").write_parquet(output[0])


rule filter_seg:
    input:
        "results/variant_scores/{model}.parquet",
        "results/variant_scores/SEG/GTEx/{tissue}.parquet",
    output:
        "results/variant_scores/filt_SEG/{model}/{tissue}.parquet",
    run:
        df = pd.concat([pd.read_parquet(input[0]), pd.read_parquet(input[1])], axis=1)
        df.loc[~df.ANNOT, "score"] = df.score.min() - 1
        df[["score"]].to_parquet(output[0], index=False)
