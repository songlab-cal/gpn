rule download_conservation:
    output:
        "results/conservation/{conservation}.bw",
    params:
        lambda wildcards: config["conservation"][wildcards.conservation],
    wildcard_constraints:
        conservation="|".join(conservation_models),
    shell:
        "wget {params} -O {output}"


rule conservation_score:
    input:
        "results/variants/merged.parquet",
        "results/conservation/{model}.bw",
    output:
        "results/variant_scores/{model}.parquet",
    wildcard_constraints:
        model="|".join(conservation_models),
    run:
        import pyBigWig

        V = pd.read_parquet(input[0])
        bw = pyBigWig.open(input[1])
        V["score"] = V.progress_apply(
            lambda v: (
                bw.values(f"chr{v.chrom}", v.pos - 1, v.pos)[0]
                if v.pos != -1
                else float("nan")
            ),
            axis=1,
        )
        V = V[["score"]]
        V.to_parquet(output[0], index=False)


# just subsetting, the inverse of process_features
rule conservation_features:
    input:
        "results/variant_scores/{model}.parquet",
        "results/variants/merged.parquet",
    output:
        "results/features/{model}.parquet",
    wildcard_constraints:
        model="|".join(conservation_models),
    run:
        V = pl.concat([pl.read_parquet(path) for path in input], how="horizontal")
        print(V)
        V = V.filter(pl.col("pos") != -1)
        print(V)
        V.select("score").write_parquet(output[0])


ruleorder: conservation_score > process_features
