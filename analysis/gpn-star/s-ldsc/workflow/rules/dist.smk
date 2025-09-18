rule tss_dist_feature:
    input:
        "results/variants/merged_filt.parquet",
        TRAITGYM_PATH + "results/tss.parquet",
    output:
        "results/features/TSS_dist.parquet",
        "results/features/closest_gene.parquet",
    run:
        V = pd.read_parquet(input[0])
        V["idx"] = V.index
        V["start"] = V["pos"] - 1
        V["end"] = V["pos"]
        tss = pd.read_parquet(input[1])
        V = bf.closest(V, tss).rename(columns={"distance": "score", "gene_id_": "gene"})
        V = V.sort_values("idx")
        V[["score"]].to_parquet(output[0], index=False)
        V[["gene"]].to_parquet(output[1], index=False)
