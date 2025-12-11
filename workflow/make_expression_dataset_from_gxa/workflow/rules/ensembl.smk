rule ensembl_download_genome:
    output:
        "results/ensembl/genome.fa.gz",
    shell:
        "wget -O {output} {config[ensembl][genome]}"


rule ensembl_download_annotation:
    output:
        "results/ensembl/annotation.gff3.gz",
    shell:
        "wget -O {output} {config[ensembl][annotation]}"


rule ensembl_extract_tss:
    input:
        "results/ensembl/annotation.gff3.gz",
    output:
        "results/ensembl/tss.parquet",
    run:
        df = pr.read_gff3(input[0]).df
        df = df.rename(
            columns={
                "Chromosome": "chrom",
                "Start": "start",
                "End": "end",
                "Strand": "strand",
                "Feature": "feature",
            }
        )
        df.chrom = df.chrom.astype(str)
        df.strand = df.strand.astype(str)
        df = df[
            df.chrom.isin(CHROMS) & df.strand.isin(["+", "-"]) & (df.feature == "mRNA")
        ]
        df["pos"] = df.start.where(df.strand == "+", df.end)
        # bin TSS positions (e.g. 100bp bins), so transcript isoforms with slightly
        # different TSS are grouped together
        df.pos = tss_pos_bin(df.pos)
        df = df[["transcript_id", "chrom", "pos", "strand"]].sort_values(
            ["chrom", "pos", "strand"]
        )
        df.to_parquet(output[0], index=False)
