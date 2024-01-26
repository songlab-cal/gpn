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


rule gwas_filt:
    input:
        "results/gwas/processed.parquet",
    output:
        "results/gwas/filt.parquet",
    run:
        V = pd.read_parquet(input[0])
        V = V[V.method=="SUSIE"]
        V = V[(~V.LD_HWE) & (~V.LD_SV)]
        # reinterpreting trait as "causal in trait" rather than "tested" in trait
        V.loc[V.pip <= 0.9, "trait"] = ""
        V = V.groupby(COORDINATES).agg({
            "pip": "max", "maf": "mean", "trait": "unique",
        }).reset_index()
        V.loc[V.pip > 0.9, "label"] = True
        V.loc[V.pip < 0.01, "label"] = False
        V = V.dropna(subset="label")
        V.trait = V.trait.progress_apply(
            lambda traits: ",".join(sorted([trait for trait in traits if trait != ""]))
        )
        print(V)
        V.to_parquet(output[0], index=False)


rule gwas_make_ensembl_vep_input:
    input:
        "results/gwas/filt.parquet",
    output:
        "results/gwas/filt.ensembl_vep.input.tsv.gz",
    run:
        df = pd.read_parquet(input[0])
        df["start"] = df.pos
        df["end"] = df.start
        df["allele"] = df.ref + "/" + df.alt
        df["strand"] = "+"
        df.to_csv(
            output[0], sep="\t", header=False, index=False,
            columns=["chrom", "start", "end", "allele", "strand"],
        )


rule gwas_run_ensembl_vep:
    input:
        "results/gwas/filt.ensembl_vep.input.tsv.gz",
        "results/ensembl_vep_cache",
    output:
        "results/gwas/filt.ensembl_vep.output.tsv.gz",  # TODO: make temp
    singularity:
        "docker://ensemblorg/ensembl-vep:release_109.1"
    threads: workflow.cores
    shell:
        """
        vep -i {input[0]} -o {output} --fork {threads} --cache \
        --dir_cache {input[1]} --format ensembl \
        --most_severe --compress_output gzip --tab
        """


rule gwas_process_ensembl_vep:
    input:
        "results/gwas/filt.parquet",
        "results/gwas/filt.ensembl_vep.output.tsv.gz",
    output:
        "results/gwas/filt.annot.parquet",
    run:
        V = pd.read_parquet(input[0])
        V2 = pd.read_csv(
            input[1], sep="\t", header=None, comment="#",
            usecols=[0, 6]
        ).rename(columns={0: "variant", 6: "consequence"})
        V2["chrom"] = V2.variant.str.split("_").str[0]
        V2["pos"] = V2.variant.str.split("_").str[1].astype(int)
        V2["ref"] = V2.variant.str.split("_").str[2].str.split("/").str[0]
        V2["alt"] = V2.variant.str.split("_").str[2].str.split("/").str[1]
        V2.drop(columns=["variant"], inplace=True)
        V = V.merge(V2, on=COORDINATES, how="inner")
        print(V)
        V.to_parquet(output[0], index=False)


rule gwas_match:
    input:
        "results/gwas/filt.annot.parquet",
        "results/tss.parquet",
        "results/exon.parquet",
    output:
        "results/gwas/matched/test.parquet",
    run:
        V = pd.read_parquet(input[0])

        V["start"] = V.pos
        V["end"] = V.start + 1

        tss = pd.read_parquet(input[1], columns=["chrom", "start", "end"])
        exon = pd.read_parquet(input[2], columns=["chrom", "start", "end"])

        V = bf.closest(V, tss).rename(columns={
            "distance": "tss_dist"
        }).drop(columns=["chrom_", "start_", "end_"])
        V = bf.closest(V, exon).rename(columns={
            "distance": "exon_dist"
        }).drop(columns=[
            "start", "end", "chrom_", "start_", "end_"
        ])

        base_match_features = ["maf"]

        consequences = V[V.label].consequence.unique()
        V_cs = []
        for c in consequences:
            print(c)
            V_c = V[V.consequence == c].copy()
            if c == "intron_variant":
                match_features = base_match_features + ["tss_dist", "exon_dist"]
            elif c in ["intergenic_variant", "downstream_gene_variant", "upstream_gene_variant"]:
                match_features = base_match_features + ["tss_dist"]
            else:
                match_features = base_match_features
            for f in match_features:
                V_c[f"{f}_scaled"] = RobustScaler().fit_transform(V_c[f].values.reshape(-1, 1))
            print(V_c.label.value_counts())
            V_c = match_columns(V_c, "label", [f"{f}_scaled" for f in match_features])
            V_c["match_group"] = c + V_c.match_group.astype(str)
            print(V_c.label.value_counts())
            print(V_c.groupby("label")[match_features].median())
            V_c.drop(columns=[f"{f}_scaled" for f in match_features], inplace=True)
            V_cs.append(V_c)
        V = pd.concat(V_cs, ignore_index=True)
        V = sort_chrom_pos(V)
        print(V)
        V.to_parquet(output[0], index=False)
