rule cV2F_download:
    output:
        "results/cV2F/scores.tsv",
    shell:
        "wget -O {output} https://mskcc.box.com/shared/static/hsrogtr3fddtmd53hyy5ph7dlp20eq72.txt"


rule cV2F_process:
    input:
        "results/variants/merged_filt.parquet",
        "results/cV2F/scores.tsv",
    output:
        "results/features/cV2F.parquet",
    run:
        V = pl.read_parquet(input[0])
        n = len(V)
        print(V)
        score = pl.read_csv(
            input[1],
            separator="\t",
            columns=["CHR", "BP", "cV2F"],
            schema_overrides={"CHR": str},
        ).rename({"CHR": "chrom", "BP": "pos", "cV2F": "score"})
        print(score)
        assert len(score) == score.n_unique(["chrom", "pos"])
        V = V.join(score, on=["chrom", "pos"], how="left")
        print(V["score"].is_null().sum())
        print(V)
        assert len(V) == n
        V[["score"]].write_parquet(output[0])


rule cV2F_process_tissue_agnostic:
    input:
        "results/cV2F_box/Tissue_agnostic/cV2F_features.{chrom}.txt.gz",
    output:
        temp("results/cV2F/tissue_agnostic/{chrom}.parquet"),
    wildcard_constraints:
        chrom="|".join(CHROMS),
    run:
        (
            pl.read_csv(input[0], separator="\t", schema_overrides={"CHR": str})
            .rename({"CHR": "chrom", "BP": "pos"})
            .drop(["SNP", "CM"])
            .write_parquet(output[0])
        )


rule cV2F_merge_tissue_agnostic:
    input:
        expand("results/cV2F/tissue_agnostic/{chrom}.parquet", chrom=CHROMS),
    output:
        "results/cV2F/tissue_agnostic/merged.parquet",
    run:
        V = pl.concat([pl.read_parquet(f) for f in input])
        print(V)
        V.write_parquet(output[0])


rule cV2F_tissue_agnostic_features:
    input:
        "results/variants/merged_filt.parquet",
        "results/cV2F/tissue_agnostic/merged.parquet",
    output:
        "results/features/cV2F_tissue_agnostic.parquet",
    run:
        V = pl.read_parquet(input[0])
        n = len(V)
        tissue_agnostic = pl.read_parquet(input[1])
        print(len(tissue_agnostic))
        # just droping ~150, I think the issue is hg19 -> hg38 distinct variants
        # mapping into the same variant
        tissue_agnostic = tissue_agnostic.unique(["chrom", "pos"])
        print(len(tissue_agnostic))
        assert len(tissue_agnostic) == tissue_agnostic.n_unique(["chrom", "pos"])
        V = V.join(tissue_agnostic, on=["chrom", "pos"], how="left")
        print(V)
        assert len(V) == n
        print(V["Coding_UCSC"].is_null().sum())
        V.drop(COORDINATES).write_parquet(output[0])


rule cV2F_tissue_agnostic_subset:
    input:
        "results/features/cV2F_tissue_agnostic.parquet",
    output:
        "results/features/cV2F_tissue_agnostic_subset_{subset}.parquet",
    run:
        (
            pl.read_parquet(
                input[0], columns=config["tissue_agnostic_subsets"][wildcards.subset]
            ).write_parquet(output[0])
        )
