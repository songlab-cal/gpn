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
        "results/conservation/phyloP-Zoonomia.bw",
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
            lambda v: -bw.values(f"chr{v.chrom}", v.pos - 1, v.pos)[0], axis=1
        )
        print(df)
        df = df[["score"]]
        df.to_parquet(output[0], index=False)


rule run_vep_conservation_combination:
    input:
        "results/conservation/phyloP.bw",
        "results/conservation/phastCons.bw",
    output:
        "results/preds/{dataset}/conservation_combination.parquet",
    threads: workflow.cores // 4
    run:
        import pyBigWig

        df = load_dataset(wildcards["dataset"], split="test").to_pandas()[
            ["chrom", "pos"]
        ]
        df.chrom = "chr" + df.chrom
        phyloP = pyBigWig.open(input[0])
        phastCons = pyBigWig.open(input[1])

        get_phyloP = lambda v: phyloP.values(v.chrom, v.pos - 1, v.pos)[0]
        df["phyloP"] = df.progress_apply(get_phyloP, axis=1)


        def get_phastCons(v):
            pos = v.pos - 1
            kernel = 7
            start = pos - kernel // 2
            end = pos + kernel // 2 + 1
            return np.nanmax(phastCons.values(v.chrom, start, end))


        df["phastCons"] = df.progress_apply(get_phastCons, axis=1)

        df["score"] = -(np.fmax(df.phyloP, 1.0) * (0.1 + df.phastCons))
        print(df)
        df = df[["score"]]
        df.to_parquet(output[0], index=False)
