rule primateai3d_process:
    input:
        "results/primateai3d/PrimateAI-3D_scores.csv.gz",
    output:
        "results/primateai3d/scores.parquet",
    run:
        V = pd.read_csv(
            input[0],
            usecols=["chr", "pos", "non_flipped_ref", "non_flipped_alt", "score_PAI3D"],
        ).rename(
            columns={
                "chr": "chrom",
                "non_flipped_ref": "ref",
                "non_flipped_alt": "alt",
                "score_PAI3D": "score",
            }
        )
        V.chrom = V.chrom.str.replace("chr", "")
        V.score = -V.score
        print(V)
        V.to_parquet(output[0], index=False)


rule primateai3d_run_vep:
    input:
        "{dataset}/test.parquet",
        "results/primateai3d/scores.parquet",
    output:
        "results/preds/{dataset}/PrimateAI-3D.parquet",
    run:
        V = pl.read_parquet(input[0])
        score = pl.read_parquet(input[1])
        V = V.join(score, on=COORDINATES, how="left")
        V.select("score").write_parquet(output[0])
