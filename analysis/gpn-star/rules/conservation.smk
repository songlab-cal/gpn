rule download_phyloP100way:
    output:
        "results/conservation/hg38/phyloP100way.bw",
    shell:
        "wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw -O {output}"

rule download_phastCons100way:
    output:
        "results/conservation/hg38/phastCons100way.bw",
    shell:
        "wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons100way/hg38.phastCons100way.bw -O {output}"

rule donwload_phyloP447way:
    output:
        "results/conservation/hg38/phyloP447way.bw"
    shell:
        "wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP447way/hg38.phyloP447way.bw -O {output}"

rule download_phastCons470way:
    output:
        "results/conservation/hg38/phastCons470way.bw"
    shell:
        "wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons470way/hg38.phastCons470way.bw -O {output}"

rule donwload_phyloP243way:
    output:
        "results/conservation/hg38/phyloP243way.bw"
    shell:
        "wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP447way/hg38.phyloP447wayPrimates.bw -O {output}"

rule download_phastCons43way:
    output:
        "results/conservation/hg38/phastCons43way.bw"
    shell:
        "wget https://cgl.gi.ucsc.edu/data/cactus/zoonomia-2021-track-hub/hg38/phyloPPrimates.bigWig -O {output}"


# mm39
rule download_phyloP35way:
    output:
        "results/conservation/mm39/phyloP35way.bw",
    shell:
        "wget https://hgdownload.soe.ucsc.edu/goldenPath/mm39/phyloP35way/mm39.phyloP35way.bw -O {output}"

rule download_phastCons35way:
    output:
        "results/conservation/mm39/phastCons35way.bw",
    shell:
        "wget https://hgdownload.soe.ucsc.edu/goldenPath/mm39/phastCons35way/mm39.phastCons35way.bw -O {output}"

# get vep scores
rule run_vep_conservation:
    input:
        "results/conservation/{genome}/{model}.bw",
    output:
        "results/preds/{dataset}/{genome}/{model}.parquet",
    threads: workflow.cores // 3
    wildcard_constraints:
        model="(phyloP|phastCons).*"
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