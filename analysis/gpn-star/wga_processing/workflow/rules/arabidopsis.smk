rule arabidopsis_download_variants:
    output:
        "results/variants/tair10/all.parquet",
    run:
        import polars as pl

        V = pl.read_parquet(
            "https://huggingface.co/datasets/gonzalobenegas/processed-data-arabidopsis/resolve/main/variants/all/variants.parquet",
            columns=COORDINATES + ["AC", "AF"],
        )
        print(V)
        V.write_parquet(output[0])
