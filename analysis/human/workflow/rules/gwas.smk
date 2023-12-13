rule gwas_download:
    output:
        temp("results/gwas/UKBB_94traits_release1.1.tar.gz"),
        "results/gwas/raw/release1.1/UKBB_94traits_release1.bed.gz",
        "results/gwas/raw/release1.1/UKBB_94traits_release1.cols",
    params:
        directory("results/gwas/raw"),
    shell:
        """
        wget -O {output[0]} https://www.dropbox.com/s/cdsdgwxkxkcq8cn/UKBB_94traits_release1.1.tar.gz?dl=1 &&
        mkdir -p {params} &&
        tar -xzvf {output[0]} -C {params}
        """


rule gwas_process:
    input:
        "results/gwas/raw/release1.1/UKBB_94traits_release1.bed.gz",
        "results/gwas/raw/release1.1/UKBB_94traits_release1.cols",
        "results/genome.fa.gz",
    output:
        "results/gwas/processed.parquet",
    run:
        V = pd.read_csv(
            input[0], sep="\t", header=None,
            names=pd.read_csv(input[1], header=None, sep="\t")[0].values
        ).rename(columns={"chromosome": "chrom", "end": "pos", "allele1": "ref", "allele2": "alt"})[
            ["chrom", "pos", "ref", "alt", "trait", "method", "pip", "region", "maf", "LD_HWE", "LD_SV"]
        ]
        V.chrom = V.chrom.str.replace("chr", "")
        print(V.shape)
        V = lift_hg19_to_hg38(V)
        V = V[V.pos != -1]
        print(V.shape)
        V = V[(V.ref.str.len()==1) & (V.alt.str.len()==1)]
        print(V.shape)
        genome = Genome(input[2])
        V = check_ref_alt(V, genome)
        print(V.shape)
        V.to_parquet(output[0], index=False) 


rule gwas_match:
    input:
        "results/gwas/processed.parquet",
    output:
        "results/gwas/matched/test.parquet",
    run:
        V = pd.read_parquet(input[0])
        V = V[V.method=="SUSIE"]
        V = V[(~V.LD_HWE) & (~V.LD_SV)]
        cols = ["chrom", "pos", "ref", "alt"]
        V = V.groupby(cols).agg({"pip": "max", "maf": "mean"}).reset_index()
        V.loc[V.pip > 0.9, "label"] = True
        V.loc[V.pip < 0.01, "label"] = False
        V = V.dropna(subset="label")
        V["label"] = V["label"].astype(bool)
        print(V.label.value_counts())
        V = match_columns(V, "label", ["maf"])
        print(V.label.value_counts())
        V.to_parquet(output[0], index=False)
