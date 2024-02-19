rule download_phyloP:
    output:
        "results/conservation/phyloP.bw",
    shell:
        "wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw -O {output}"


rule download_phastCons:
    output:
        "results/conservation/phastCons.bw",
    shell:
        "wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons100way/hg38.phastCons100way.bw -O {output}"


rule donwload_phyloP_zoonomia:
    output:
        "results/conservation/phyloP-Zoonomia.bw"
    shell:
        "wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/cactus241way/cactus241way.phyloP.bw -O {output}"


rule run_vep_conservation:
    input:
        "results/conservation/{model}.bw",
    output:
        "results/preds/{dataset}/{model,phyloP|phastCons|phyloP-Zoonomia|phyloP.470way|phastCons.470way}.parquet",
    threads: workflow.cores // 3
    run:
        import pyBigWig

        df = load_dataset(wildcards["dataset"], split="test").to_pandas()
        bw = pyBigWig.open(input[0])
        df["score"] = df.progress_apply(
            lambda v: -bw.values(f"chr{v.chrom}", v.pos-1, v.pos)[0],
            axis=1
        )
        print(df)
        df = df[["score"]]
        df.to_parquet(output[0], index=False)
